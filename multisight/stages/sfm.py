"""S2 SfM 稀疏重建（FR-03/04）——决策：完全自研（PRD §11-1），v0.1 交付。

Mock：按环绕轨道合成相机位姿，落盘 COLMAP 兼容目录 + 稀疏点云，
用于打通断点续跑 / SSE / 下载全链路。
Real：自研两视图内核（特征→匹配→F→位姿→三角化，见 multisight.sfm）已落地为管线
第一步：对首两帧做真实两视图重建；多视图增量注册 + BA 为后续里程碑。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..pipeline import Stage, StageContext
from ..sfm import matrix_to_quat, two_view_reconstruction
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
    """自研两视图 SfM：对首两帧做真实重建并落盘 COLMAP 兼容产物。

    诚实原则：特征/匹配不足以建立可靠两视图时明确报错（不静默假成功）；
    多视图增量注册 + BA 为后续里程碑。
    """

    name = "sfm"

    def run(self, ctx: StageContext) -> None:
        listing = json.loads((ctx.ws.root / "images" / "list.json").read_text(encoding="utf-8"))
        photos = listing["photos"]
        if len(photos) < 2:
            raise ValueError(f"真实两视图 SfM 至少需 2 张照片（当前预处理仅 {len(photos)} 张）")
        a, b = photos[0], photos[1]
        pa = ctx.ws.root / a["file"]
        pb = ctx.ws.root / b["file"]
        ctx.progress(self.name, 15, "真实特征提取与匹配中：首两帧")
        res = two_view_reconstruction(pa, pb, focal=None, max_corners=800, seed=0)
        ctx.progress(self.name, 55, f"匹配 {res.num_matches} 对，RANSAC 内点 {res.num_inliers}")
        if not res.registered or res.F is None or res.num_inliers < 8:
            raise ValueError(
                f"首两帧未能建立可靠两视图（匹配 {res.num_matches}、内点 {res.num_inliers}）；"
                f"请确认照片有充足重叠且清晰。当前自研 SfM 仅支持两视图引导，多视图增量注册待落地。")
        K = res.K
        f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
        w, h = int(a["width"]), int(a["height"])
        self._write_cameras(ctx, f, cx, cy, w, h)
        self._write_images(ctx, a, b, res)
        write_ply_ascii(ctx.ws.file("sfm", "points3D.ply"), res.points3d)
        ctx.progress(self.name, 85, f"三角化稀疏点 {len(res.points3d)} 个")
        stats = {
            "engine": "real/two-view",
            "registered": 2, "total": len(photos),
            "register_rate": round(2 / len(photos), 4),
            "matches": res.num_matches, "inliers": res.num_inliers,
            "sparse_points": int(len(res.points3d)),
            "median_reproj_px": round(float(res.median_reproj_px), 4),
            "camera_pair": [a["file"], b["file"]],
            "baseline": [round(float(x), 6) for x in res.t],
            "note": "v0.1 两视图引导：首两帧真实重建；多视图增量注册+BA 待落地",
        }
        (ctx.ws.root / "sfm" / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress(self.name, 100,
                     f"两视图稀疏重建完成：{len(res.points3d)} 稀疏点，内点 {res.num_inliers}")

    @staticmethod
    def _write_cameras(ctx: StageContext, f: float, cx: float, cy: float,
                       w: int, h: int) -> None:
        lines = ["# Camera list with one line of data per camera:",
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                 f"1 SIMPLE_PINHOLE {w} {h} {f:.4f} {cx:.4f} {cy:.4f}",
                 f"2 SIMPLE_PINHOLE {w} {h} {f:.4f} {cx:.4f} {cy:.4f}"]
        ctx.ws.file("sfm", "cameras.txt").write_text("\n".join(lines) + "\n", encoding="ascii")

    @staticmethod
    def _write_images(ctx: StageContext, a: dict, b: dict, res) -> None:
        qw, qx, qy, qz = matrix_to_quat(res.R)
        tx, ty, tz = (float(x) for x in res.t)
        na = a["file"].split("/")[-1]
        nb = b["file"].split("/")[-1]
        lines = ["# Image list with two lines of data per image:",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                 f"1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 {na}", "-1",
                 f"2 {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.6f} {ty:.6f} {tz:.6f} 2 {nb}", "-1"]
        ctx.ws.file("sfm", "images.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
