"""自研 SfM 内核（纯 numpy + Pillow，无 OpenCV/scipy）。

PRD §11-1 决策「完全自研」：特征提取/匹配 → 对极几何/三角化 → 增量注册 → BA
逐里程碑落地。本包按子模块拆分，先交付经过验证的最小可用件：

- features：Harris 角点检测 + 归一化 patch 描述子 + 比值测试暴力匹配（v0.1）。
"""

from __future__ import annotations

from .features import (
    Corner, detect_corners, describe, detect_and_describe, match_descriptors, to_gray,
)

__all__ = [
    "Corner", "detect_corners", "describe", "detect_and_describe",
    "match_descriptors", "to_gray",
]
