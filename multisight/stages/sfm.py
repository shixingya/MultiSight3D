"""S2 SfM 稀疏重建（FR-03/04）——决策：完全自研（PRD §11-1），v0.1 交付。

Mock：按环绕轨道合成相机位姿，落盘 COLMAP 兼容目录 + 稀疏点云，
用于打通断点续跑 / SSE / 下载全链路。
Real：自研 SfM 内核（特征→匹配→F→相对位姿→三角化→PnP 增量注册，见 multisight.sfm）
已作为管线真实一步：在候选帧对（相邻对 + 首帧对全体）中按 RANSAC 内点择优建初始两视图，
再对其余帧做 2D-3D 迁移 + RANSAC-PnP 逐帧增量注册，输出 N 帧相机/位姿。
BA（全局束调整）为下一里程碑。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..pipeline import Stage, StageContext
from ..sfm import incremental_reconstruction, matrix_to_quat
from ._synth import (sphere_points, write_cameras_txt, write_images_txt,
                     write_ply_ascii)


class Mock(Stage):
    name = "sfm"

    def run(self, ctx: StageContext) -> None:
        listing = json.loads((ctx.ws.root / "images" / "list.json").read_text(encoding="utf-8"))
        photos = listing["photos"]
        n = len(photos)

        ctx.progress(self.name, 10, f"特征提取与匹配中（mock）：{n} 张")
        time.sleep(0.2)
        ctx.progress(self.name, 45, "增量式注册（mock 轨道位姿）")

        write_cameras_txt(ctx.ws.file("sfm", "cameras.txt"), n,
                          photos[0]["width"], photos[0]["height"])
        write_images_txt(ctx.ws.file("sfm", "images.txt"), n)

        ctx.progress(self.name, 70, "三角化稀疏点云 + 捆绑调整（mock）")
        pts = sphere_points(min(400, n * 30), radius=0.9, seed=n)
        write_ply_ascii(ctx.ws.file("sfm", "points3D.ply"), pts)

        register_rate = 1.0  # mock 全注册
        ctx.progress(self.name, 100, f"稀疏重建完成：{n}/{n} 张注册，{len(pts)} 稀疏点")
        # 阶段级质量指标（供 report 聚合，FR-09 的 mock 前身）
        (ctx.ws.root / "sfm" / "stats.json").write_text(json.dumps({
            "registered": n, "total": n, "register_rate": register_rate,
            "sparse_points": int(len(pts)),
            "median_reproj_px": 0.7,  # mock 值
        }), encoding="utf-8")


class Real(Stage):
    """自研增量 SfM：候选帧对择优建初始两视图→逐帧 PnP 注册，落盘 COLMAP 兼容产物。

    诚实原则：特征/匹配不足以建立可靠初始两视图时明确报错（不静默假成功）；
    重叠不足/内点不够的帧会诚实地不被注册（registered 反映真实注册数，非总帧数）。
    全局 BA 为下一里程碑。
    """

    name = "sfm"

    MAX_FRAMES = 8   # 增量注册至多考察前 N 帧（控制无人值守成本）

    def run(self, ctx: StageContext) -> None:
        listing = json.loads((ctx.ws.root / "images" / "list.json").read_text(encoding="utf-8"))
        photos = listing["photos"]
        if len(photos) < 2:
            raise ValueError(f"真实增量 SfM 至少需 2 张照片（当前预处理仅 {len(photos)} 张）")
        cand = photos[:self.MAX_FRAMES]
        paths = [ctx.ws.root / p["file"] for p in cand]
        ctx.progress(self.name, 15, f"真实特征提取与增量注册中：{len(cand)} 帧")
        res = incremental_reconstruction(paths, focal=None, max_corners=800, seed=0)
        if not res.ok:                                    # 初始两视图不可靠/无 3D 点 → 不臆造
            raise ValueError(
                f"未能从 {len(cand)} 帧建立可靠初始两视图（注册 {res.num_registered} 帧、"
                f"稀疏点 {len(res.points3d)}）；请确认照片有充足重叠且清晰。")
        a_idx, b_idx = res.initial_pair
        K = res.K
        f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
        w, h = int(cand[a_idx]["width"]), int(cand[a_idx]["height"])
        cams = res.cameras                               # 已按 index 升序
        self._write_cameras(ctx, len(cams), f, cx, cy, w, h)
        self._write_images(ctx, cams, cand)
        write_ply_ascii(ctx.ws.file("sfm", "points3D.ply"), res.points3d)
        ctx.progress(self.name, 85,
                     f"三角化/迁移：稀疏点 {len(res.points3d)} 个，注册 {res.num_registered}/{len(cand)} 帧")
        b_cam = next(cm for cm in cams if cm.index == b_idx)
        stats = {
            "engine": "real/incremental",
            "registered": res.num_registered, "total": len(photos),
            "considered": len(cand),
            "register_rate": round(res.num_registered / len(photos), 4),
            "initial_pair": [cand[a_idx]["file"], cand[b_idx]["file"]],
            "initial_inliers": int(b_cam.num_inliers),
            "sparse_points": int(len(res.points3d)),
            "baseline": [round(float(x), 6) for x in b_cam.t],
            "pnp_inliers": {cand[cm.index]["file"]: int(cm.num_inliers) for cm in cams
                            if cm.index not in (a_idx, b_idx)},
            "note": "v0.1 增量注册：候选对择优建初始两视图 + 逐帧 RANSAC-PnP；全局 BA 待落地",
        }
        (ctx.ws.root / "sfm" / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress(self.name, 100,
                     f"增量稀疏重建完成：注册 {res.num_registered}/{len(cand)} 帧，{len(res.points3d)} 稀疏点")

    @staticmethod
    def _write_cameras(ctx: StageContext, n: int, f: float, cx: float, cy: float,
                       w: int, h: int) -> None:
        lines = ["# Camera list with one line of data per camera:",
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
        for i in range(n):
            lines.append(f"{i + 1} SIMPLE_PINHOLE {w} {h} {f:.4f} {cx:.4f} {cy:.4f}")
        ctx.ws.file("sfm", "cameras.txt").write_text("\n".join(lines) + "\n", encoding="ascii")

    @staticmethod
    def _write_images(ctx: StageContext, cams, cand) -> None:
        lines = ["# Image list with two lines of data per image:",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
        for k, cm in enumerate(cams):
            qw, qx, qy, qz = matrix_to_quat(cm.R)
            tx, ty, tz = (float(x) for x in cm.t)
            name = cand[cm.index]["file"].split("/")[-1]
            lines.append(f"{k + 1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                         f"{tx:.6f} {ty:.6f} {tz:.6f} {k + 1} {name}")
            lines.append("-1")
        ctx.ws.file("sfm", "images.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
