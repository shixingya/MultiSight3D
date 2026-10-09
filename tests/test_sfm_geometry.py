"""自研 SfM 对极几何单测（合成双视图，已知相对位姿，零二进制入库）。"""

from __future__ import annotations

import numpy as np

from multisight.sfm import estimate_fundamental, ransac_fundamental, triangulate


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _synthetic_views(n: int = 40, seed: int = 1):
    """相机1=[I|0]、相机2=R,t 投影同一批 3D 点，返回像素对应 (p1,p2) 与真值 F。"""
    rng = np.random.default_rng(seed)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    X = np.column_stack([
        rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n), np.ones(n)])
    R = _rot("y", np.radians(8)) @ _rot("x", np.radians(5))
    t = np.array([0.35, 0.02, 0.05])
    def project(P):
        x = K @ P                       # P: (3, n) 列向量
        return (x[:2] / x[2:3]).T
    p1 = project(X[:, :3].T)             # 相机1: [I|0]
    X2 = X[:, :3] @ R.T + t
    p2 = project(X2.T)
    # 真值基础矩阵 F = K^{-T} [t]_x R K^{-1}
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    F /= np.linalg.norm(F)
    return p1, p2, F


def test_estimated_f_satisfies_epipolar_on_true_correspondences():
    p1, p2, _Ftrue = _synthetic_views()
    F = estimate_fundamental(p1, p2)
    h1 = np.column_stack([p1, np.ones(len(p1))])
    h2 = np.column_stack([p2, np.ones(len(p2))])
    err = np.abs(np.einsum("ij,jk,ik->i", h2, F, h1)) / np.linalg.norm(F, axis=None)
    # 精确投影（无噪声）→ 真对应应几乎严格满足 x2ᵀFx1=0
    assert np.median(err) < 1e-6


def test_ransac_recovers_inliers_and_rejects_outliers():
    p1, p2, Ftrue = _synthetic_views(n=40)
    rng = np.random.default_rng(7)
    n_out = 12
    # 外点：把 p2 侧随机打乱配对（破坏对应关系）
    perm = rng.permutation(len(p2))[:n_out]
    p1o = np.vstack([p1, p1[perm]])
    p2o = np.vstack([p2, p2[rng.permutation(len(p2))[:n_out]]])
    truth_inlier = np.zeros(len(p1o), dtype=bool)
    truth_inlier[:len(p1)] = True

    F, inliers = ransac_fundamental(p1o, p2o, iters=300, threshold=1.0, seed=0)
    assert F is not None
    # 至少恢复大部分真内点，且外点绝大多数被判为外点
    tp = int((inliers & truth_inlier).sum())
    fp = int((inliers & ~truth_inlier).sum())
    assert tp >= 0.85 * len(p1)
    assert fp <= 0.25 * n_out
    #  recovered F 与真值应等价（差一尺度下逐元素方向一致）
    F = F / np.linalg.norm(F)
    Ft = Ftrue / np.linalg.norm(Ftrue)
    agree = min(np.linalg.norm(F - Ft), np.linalg.norm(F + Ft))
    assert agree < 0.2


def test_ransac_short_input_returns_none_mask():
    F, mask = ransac_fundamental(np.zeros((5, 2)), np.zeros((5, 2)))
    assert F is None and len(mask) == 5 and not mask.any()


def test_estimate_requires_min_points():
    import pytest
    with pytest.raises(ValueError):
        estimate_fundamental(np.zeros((7, 2)), np.zeros((7, 2)))


def test_triangulate_recovers_3d_points():
    rng = np.random.default_rng(3)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    n = 50
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)])
    R = _rot("y", np.radians(8)) @ _rot("x", np.radians(5))
    t = np.array([0.35, 0.02, 0.05])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t[:, None]])
    def proj(P, Xw):
        x = P @ np.hstack([Xw, np.ones((len(Xw), 1))]).T
        return (x[:2] / x[2:3]).T
    pts1, pts2 = proj(P1, X), proj(P2, X)
    Xh = triangulate(P1, P2, pts1, pts2)
    assert not np.isnan(Xh).any()
    assert np.allclose(Xh, X, atol=1e-6)   # 精确投影→三角化应近乎复原
