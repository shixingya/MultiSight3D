"""两视图真实 SfM 编排：把特征与几何各件串成端到端重建（纯 numpy）。

这是自研 SfM 主线的「可运行内核」：给定两张有重叠/视差的图像，产出相机相对位姿
与稀疏 3D 点——诚实反映匹配/内点质量，特征不足时返回明确的降级结果而非崩溃。
`select_best_pair` 在受限候选帧对（相邻对 + 首帧对全体）中按 RANSAC 内点数择优，
避免盲目取前两份导致重叠不足退化或误吃重复纹理假阳性。
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


def two_view_from_features(ga, ca, da, gb, cb, db, *, focal: float | None = None,
                           ratio: float = 0.8, ransac_iters: int = 300,
                           ransac_threshold: float = 1.0,
                           seed: int = 0) -> TwoViewResult:
    """给定两帧已提取的角点+描述子，做比值匹配→RANSAC F→位姿恢复→三角化。

    拆出此内部函数，供 select_best_pair 复用（各帧特征只算一次，不随候选对重复检测）。
    """
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
        lines = Fref @ h1.T                          # 3×N：图像 2 中的极线 [a,b,c]ᵀ
        den = lines[0] ** 2 + lines[1] ** 2
        res.median_reproj_px = float(np.median(err / np.sqrt(den + 1e-12)))
    except (ValueError, np.linalg.LinAlgError):
        pass
    res.ok = len(res.points3d) > 0
    return res


def two_view_reconstruction(img_a, img_b, *, focal: float | None = None,
                            max_corners: int = 600, patch: int = 8,
                            ratio: float = 0.8, ransac_iters: int = 300,
                            ransac_threshold: float = 1.0,
                            seed: int = 0) -> TwoViewResult:
    """两帧图像（路径/ndarray）→ 完整两视图重建。任何环节点不足都诚实降级。"""
    ga = to_gray(img_a)
    gb = to_gray(img_b)
    ca, da = detect_and_describe(ga, max_corners=max_corners, patch=patch)
    cb, db = detect_and_describe(gb, max_corners=max_corners, patch=patch)
    return two_view_from_features(ga, ca, da, gb, cb, db, focal=focal, ratio=ratio,
                                  ransac_iters=ransac_iters, ransac_threshold=ransac_threshold,
                                  seed=seed)


def candidate_pairs(n: int) -> list[tuple[int, int]]:
    """受限候选帧对：相邻滑动对 (i, i+1) + 首帧对其余各帧。线性 O(n)，避免 O(n²)。"""
    pairs: set[tuple[int, int]] = set()
    for i in range(n - 1):
        pairs.add((i, i + 1))
    for j in range(2, n):
        pairs.add((0, j))
    return sorted(pairs)


def select_best_pair(images, *, focal: float | None = None, max_corners: int = 600,
                     patch: int = 8, ratio: float = 0.8, ransac_iters: int = 300,
                     ransac_threshold: float = 1.0,
                     seed: int = 0) -> tuple[TwoViewResult, tuple[int, int]]:
    """在候选帧对中选 RANSAC 内点最多的一对做两视图重建。

    返回 (最优结果, (i, j))。各帧特征只提取一次；若全部候选都点不足，返回内点最多的
    那一对（其 ok/registered 仍为假，由调用方据实判断），而非硬造。
    """
    grays = [to_gray(im) for im in images]
    feats = [detect_and_describe(g, max_corners=max_corners, patch=patch) for g in grays]
    best: tuple[TwoViewResult, tuple[int, int]] | None = None
    for (i, j) in candidate_pairs(len(grays)):
        ca, da = feats[i]
        cb, db = feats[j]
        r = two_view_from_features(grays[i], ca, da, grays[j], cb, db, focal=focal,
                                   ratio=ratio, ransac_iters=ransac_iters,
                                   ransac_threshold=ransac_threshold, seed=seed)
        if best is None or r.num_inliers > best[0].num_inliers:
            best = (r, (i, j))
    if best is None:                                     # <2 帧：无候选对
        raise ValueError("select_best_pair 需要至少 2 张图像")
    return best
