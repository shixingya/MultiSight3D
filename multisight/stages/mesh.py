"""S4a 网格重建（FR-06），v0.3。Mock：输出占位立方 mesh.obj。

Real：自研传统网格第一档——**距离图像网格化**（`mvs.meshify.depth_map_to_mesh`，纯 numpy）。
读上游 MVS 落盘的逐视图深度图中间产物（mvs/depthmaps/*.npz），把每张规则深度图按网格拓扑
三角化成世界系面片，跨视图直接拼接（v0.3 起步档，未做统一水面/泊松缝合），写出 mesh.obj +
stats.json。诚实原则：无深度图产物即明确报错（不静默假成功、不回退占位）；单视图网格化产
面为 0 时跳过；总面数为 0 报错。跨视图一致性缝合 / 泊松重建为后续里程碑。
"""

from __future__ import annotations

import json

import numpy as np

from ..pipeline import Stage, StageContext
from ..mvs import depth_map_to_mesh
from ._synth import write_obj_cube, write_obj_trimesh


class Mock(Stage):
    name = "mesh"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 30, "空间裁剪与泊松重建（mock：占位立方）")
        write_obj_cube(ctx.ws.file("mesh", "mesh.obj"))
        ctx.progress(self.name, 70, "减面（mock：跳过）")
        ctx.progress(self.name, 100, "网格重建完成：8 顶点 / 12 三角形（占位）")


class Real(Stage):
    """读 MVS 深度图 → 逐视图距离图像网格化 → 跨视图拼接 → mesh.obj。"""

    name = "mesh"

    MAX_EDGE_GAP_REL = 0.0   # 三角形最长边 > 度量基准×该系数则丢弃；0=关闭（先靠留洞，自适应断口阈待做）

    def run(self, ctx: StageContext) -> None:
        dm_dir = ctx.ws.root / "mvs" / "depthmaps"
        npzs = sorted(dm_dir.glob("*.npz")) if dm_dir.is_dir() else []
        if not npzs:
            raise ValueError(
                "缺少 MVS 深度图中间产物（mvs/depthmaps/*.npz），无法网格化；"
                "请先用 real 引擎跑 mvs（MS_ENGINE=real）；跨视图统一网格/泊松为后续里程碑")

        ctx.progress(self.name, 10, f"真实网格：{len(npzs)} 张深度图距离图像网格化")
        all_v: list[np.ndarray] = []
        all_f: list[np.ndarray] = []
        all_c: list[np.ndarray] = []
        base = 0
        done = 0
        for ni, p in enumerate(npzs):
            with np.load(p) as z:
                depth, us, vs = z["depth"], z["us"], z["vs"]
                K, R, t = z["K"], z["R"], z["t"]
                rgb = z["rgb"] if "rgb" in z.files else None
            gap = self._edge_gap(us, depth, K) if self.MAX_EDGE_GAP_REL > 0 else None
            verts, faces = depth_map_to_mesh(us, vs, depth, K, (R, t), max_edge_gap=gap)
            if len(faces) == 0:
                continue
            all_v.append(verts)
            all_f.append(faces + base)                  # 跨视图拼接：面索引整体偏移
            if rgb is not None:                          # 顶点对齐的逐格真彩色（与 depth_map_to_mesh 同 valid 顺序）
                all_c.append(rgb[np.isfinite(depth) & (depth > 0)])
            base += len(verts)
            done += 1
            ctx.progress(self.name, 10 + int(80 * (ni + 1) / len(npzs)),
                         f"视图网格 {done}：累计 {base} 顶点 / {sum(len(f) for f in all_f)} 面")

        if not all_f:
            raise ValueError("深度图未产生任何有效网格面（覆盖/置信不足）；不臆造几何")
        V = np.vstack(all_v)
        F = np.vstack(all_f)
        colors = np.vstack(all_c) if len(all_c) == len(all_v) and len(V) == sum(len(c) for c in all_c) else None
        write_obj_trimesh(ctx.ws.file("mesh", "mesh.obj"), V, F, colors=colors)
        ctx.progress(self.name, 100, f"网格重建完成：{len(V)} 顶点 / {len(F)} 三角形")
        (ctx.ws.root / "mesh" / "stats.json").write_text(json.dumps({
            "engine": "real/range-image-mesh",
            "depth_maps": len(npzs),
            "views_meshed": done,
            "vertices": int(len(V)),
            "faces": int(len(F)),
            "vertex_colors": bool(colors is not None),
            "note": "v0.3 起步：逐视图距离图像网格化直接拼接（顶点带参考视图真彩色）；跨视图统一缝合/泊松重建为后续里程碑",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _edge_gap(self, us, depth, K):
        """世界系断口阈：以中位深度处的水平采样边长（depth·stride/fx）为典型三角边长，
    乘 MAX_EDGE_GAP_REL。仅当 REL>0 时调用（默认关闭）。"""
        med_d = float(np.nanmedian(depth))
        stride = float(abs(us[0, 1] - us[0, 0])) if us.shape[1] > 1 else 1.0
        typ = med_d * stride / float(K[0, 0])
        return typ * self.MAX_EDGE_GAP_REL
