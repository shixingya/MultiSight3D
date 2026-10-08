"""S2 SfM 稀疏重建（FR-03/04）——决策：完全自研（PRD §11-1），v0.1 交付。

Mock：按环绕轨道合成相机位姿，落盘 COLMAP 兼容目录 + 稀疏点云，
用于打通断点续跑 / SSE / 下载全链路。
Real：自研特征提取匹配 + 增量注册 + BA（v0.1 里程碑内逐步替换）。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..pipeline import Stage, StageContext
from ._base import NotImplementedReal
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


class Real(NotImplementedReal):
    name = "sfm"
    milestone = "v0.1"
