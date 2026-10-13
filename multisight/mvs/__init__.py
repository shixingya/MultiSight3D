"""自研 MVS 稠密重建子包（纯 numpy，CPU 优先）。

PRD §11-2 决策「学习型深度网络首发 + PatchMatch 兜底」的落地起点：先交付一个
可验证的**传统多视图立体（MVS）基元**——参考视图平面扫描深度图。它不依赖任何
预训练权重/重型库，纯 numpy 即可跑，作为 CPU 兜底档与后续学习型档的对照基线。

- depth：plane_sweep_depth —— 对参考视图每像素沿视线在深度区间采样，用邻视图
  归一化互相关（NCC）代价取最优深度，产出深度图 + 置信度。
- fuse：depth_map_to_points —— 把（多视图各自的）深度图反投影、跨视图一致性校验后
  融合为稠密点云（诚实：低置信/遮挡像素一律丢弃，不臆造点）。
"""

from __future__ import annotations

from .depth import plane_sweep_depth
from .fuse import depth_map_to_points

__all__ = ["plane_sweep_depth", "depth_map_to_points"]
