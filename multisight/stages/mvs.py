"""S3 MVS 稠密重建（FR-05/12）——决策：学习型深度网络首发 + PatchMatch 兜底（PRD §11-2），v0.2 交付。

Mock：按稀疏点数十倍合成稠密球面点云 fusion.ply。
"""

from __future__ import annotations

import numpy as np

from ..pipeline import Stage, StageContext
from ._base import NotImplementedReal
from ._synth import sphere_points, write_ply_ascii


class Mock(Stage):
    name = "mvs"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 10, "深度图估计（mock：分块合成）")
        sparse_path = ctx.ws.root / "sfm" / "points3D.ply"
        base_n = 0
        if sparse_path.is_file():
            for line in sparse_path.read_text(encoding="ascii").splitlines():
                if line.startswith("element vertex"):
                    base_n = int(line.split()[-1])
                    break
        dense_n = max(2000, base_n * 10)

        points = sphere_points(dense_n, radius=1.0, seed=7)
        # 按高度着色，让预览/下载的点云有可辨视觉内容
        z01 = (points[:, 2] - points[:, 2].min()) / max(1e-6, np.ptp(points[:, 2]))
        colors = np.stack([z01 * 200 + 30, 250 - z01 * 160, (1 - z01) * 220 + 30], axis=1).astype(np.uint8)
        ctx.progress(self.name, 60, f"多视图融合与去噪（mock）：目标 {dense_n} 点")
        write_ply_ascii(ctx.ws.file("mvs", "fusion.ply"), points, colors)
        ctx.progress(self.name, 100, f"稠密点云完成：{dense_n} 点")


class Real(NotImplementedReal):
    name = "mvs"
    milestone = "v0.2"
