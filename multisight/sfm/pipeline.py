"""两视图真实 SfM 编排：把特征与几何各件串成端到端重建（纯 numpy）。

这是自研 SfM 主线的「可运行内核」：给定两张有重叠/视差的图像，产出相机相对位姿
与稀疏 3D 点——诚实反映匹配/内点质量，特征不足时返回明确的降级结果而非崩溃。
stages/sfm.Real 以此为第一步（先两视图，后续迭代扩到增量注册 + BA）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import detect_and_describe, match_descriptors, to_gray
from .geometry import estimate_fundamental, ransac_fundamental, recover_pose


@dataclass
class TwoViewResult:
    ok: bool
    num_corners_a: int = 0
    num_corners_b: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    F: np.ndarray | None = None
    K: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    median_reproj_px: float = float("nan")

    @property
    def registered(self) -> bool:
        return self.ok and self.R is not None


def default_K(shape, focal: float | None = None) -> np.ndarray:
    """由图像尺寸给一个默认针孔内参：fx=fy=焦距（缺省取长边），光心居中。"""
    h, w = shape[:2]
    f = float(focal) if focal else float(max(h, w))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def two_view_reconstruction(img_a, img_b, *, focal: float | None = None,
                            max_corners: int = 600, patch: int = 8,
                            ratio: float = 0.8, ransac_iters: int = 300,
                            ransac_threshold: float = 1.0,
                            seed: int = 0) -> TwoViewResult:
    """检测→描述→比值匹配→RANSAC F→位姿恢复→三角化。任何环节点不足都诚实降级。"""
    ga = to_gray(img_a)
    gb = to_gray(img_b)
    ca, da = detect_and_describe(ga, max_corners=max_corners, patch=patch)
    cb, db = detect_and_describe(gb, max_corners=max_corners, patch=patch)
    res = TwoViewResult(ok=False, num_corners_a=len(ca), num_corners_b=len(cb))
    matches = match_descriptors(da, db, ratio=ratio, mutual=True)
    res.num_matches = len(matches)
    if len(matches) < 8:                                # 8 点法最低要求
        return res
    idx_a = np.array([i for i, _ in matches])
    idx_b = np.array([j for _, j in matches])
    p1 = np.array([[ca[i].x, ca[i].y] for i in idx_a], dtype=np.float64)
    p2 = np.array([[cb[j].x, cb[j].y] for j in idx_b], dtype=np.float64)

    F, inliers = ransac_fundamental(p1, p2, iters=ransac_iters,
                                    threshold=ransac_threshold, seed=seed)
    res.F = F
    if F is None or inliers.sum() < 8:                  # 无足够内点 → 不臆造位姿
        return res
    res.num_inliers = int(inliers.sum())
    K = default_K(ga.shape, focal)
    res.K = K
    R, t, X = recover_pose(F, K, p1[inliers], p2[inliers])
    res.R, res.t = R, t
    finite = ~np.isnan(X).any(axis=1)
    # 仅保留两相机前方的点（真实三角化应剔除背对相机的解）
    if finite.any():
        xf = X[finite]
        z2 = (R @ xf.T)[2] + t[2]
        front = (xf[:, 2] > 0) & (z2 > 0)
        res.points3d = xf[front]
    else:
        res.points3d = np.zeros((0, 3))
    # 用全部内点重投影误差（像素）估计中位重投影残差，作为质量指标
    try:
        Fref = estimate_fundamental(p1[inliers], p2[inliers])
        h1 = np.column_stack([p1[inliers], np.ones(res.num_inliers)])
        h2 = np.column_stack([p2[inliers], np.ones(res.num_inliers)])
        err = np.abs(np.einsum("ij,jk,ik->i", h2, Fref, h1))
        d1 = (Fref @ h1.T)
        den = d1[0] ** 2 + d1[1] ** 2
        res.median_reproj_px = float(np.median(err / np.sqrt(den + 1e-12)))
    except (np.linalg.LinAlgError, ValueError):
        pass
    res.ok = len(res.points3d) > 0
    return res
