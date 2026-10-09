"""自研 SfM 对极几何单测（合成双视图，已知相对位姿，零二进制入库）。"""

from __future__ import annotations

import numpy as np

from multisight.sfm import (estimate_fundamental, ransac_fundamental, triangulate,
                            recover_pose, solve_pnp_dlt, ransac_pnp,
                            refine_pnp_gauss_newton, project, matrix_to_quat)


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


def test_recover_pose_matches_ground_truth():
    rng = np.random.default_rng(11)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    n = 60
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)])
    R = _rot("y", np.radians(10)) @ _rot("x", np.radians(6))
    t = np.array([0.35, 0.02, 0.05])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ (tx @ R) @ np.linalg.inv(K)
    F /= np.linalg.norm(F)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t[:, None]])
    def proj(P):
        x = P @ np.hstack([X, np.ones((n, 1))]).T
        return (x[:2] / x[2:3]).T
    pts1, pts2 = proj(P1), proj(P2)

    R_rec, t_rec, X_rec = recover_pose(F, K, pts1, pts2)
    assert np.allclose(R_rec, R, atol=1e-6)          # 旋转精确恢复
    t_hat = t / np.linalg.norm(t)
    assert float(np.dot(t_rec, t_hat)) > 0.99        # 平移方向正确
    # 单对视图尺度不可观：t 归一→重建缩放 1/|t|，应与真值成比例
    scale = 1.0 / np.linalg.norm(t)
    assert np.allclose(X_rec, X * scale, atol=1e-4)


def test_solve_pnp_dlt_recovers_absolute_pose():
    rng = np.random.default_rng(21)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    n = 30
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)])
    R = _rot("y", np.radians(12)) @ _rot("x", np.radians(-7))
    t = np.array([0.2, -0.1, 0.35])
    P = np.hstack([R, t[:, None]])
    x = K @ (P @ np.hstack([X, np.ones((n, 1))]).T)
    pts = (x[:2] / x[2:3]).T
    R_rec, t_rec = solve_pnp_dlt(K, X, pts)
    assert np.allclose(R_rec, R, atol=1e-6)          # 精确投影→旋转复原
    assert np.allclose(t_rec, t, atol=1e-6)          # PnP 尺度已知→平移含模长


def test_solve_pnp_requires_min_points():
    import pytest
    K = np.eye(3)
    with pytest.raises(ValueError):
        solve_pnp_dlt(K, np.zeros((5, 3)), np.zeros((5, 2)))


def test_project_forward_consistent_with_manual_projection():
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    R = _rot("y", np.radians(15))
    t = np.array([0.1, 0.2, 0.3])
    X = np.array([[0.0, 0.0, 4.0], [1.0, -0.5, 5.0]])
    P = np.hstack([R, t[:, None]])
    manual = (K @ (P @ np.hstack([X, np.ones((len(X), 1))]).T))
    manual = (manual[:2] / manual[2:3]).T
    assert np.allclose(project(K, R, t, X), manual, atol=1e-9)


def test_ransac_pnp_rejects_outliers():
    rng = np.random.default_rng(31)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    n = 25
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)])
    R = _rot("y", np.radians(10)) @ _rot("x", np.radians(-5))
    t = np.array([0.2, -0.1, 0.3])
    pts = project(K, R, t, X)
    n_out = 10
    Xo = np.vstack([X, X[rng.permutation(n)[:n_out]]])
    xo = np.vstack([pts, pts[rng.permutation(n)[:n_out]] + rng.uniform(40, 90, (n_out, 2))])
    truth = np.zeros(len(Xo), dtype=bool)
    truth[:n] = True
    Rr, tr, mask = ransac_pnp(K, Xo, xo, iters=300, threshold=3.0, seed=0)
    assert Rr is not None
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    assert tp >= 0.8 * n            # 回收大部分真内点
    assert fp <= 0.2 * n_out        # 外点绝大多数被拒
    assert float(np.dot(tr, t) / (np.linalg.norm(tr) * np.linalg.norm(t))) > 0.98  # 平移方向正确


def test_refine_pnp_gauss_newton_descends_from_perturbed_init():
    """LM 精化：从扰动位姿初值出发应把重投影误差降到 ~0 并复原 R/t（锁定三处 bug 修复）。"""
    rng = np.random.default_rng(41)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    n = 40
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)])
    R = _rot("y", np.radians(10)) @ _rot("x", np.radians(-6))
    t = np.array([0.2, -0.1, 0.3])
    x = project(K, R, t, X)
    # 构造有偏初值：旋转偏 ~6°、平移偏 0.1（模拟 DLT 在噪声下的偏差）
    R0 = _rot("y", np.radians(6)) @ R
    t0 = t + np.array([0.06, -0.08, 0.05])
    err_before = float(np.median(np.linalg.norm(project(K, R0, t0, X) - x, axis=1)))
    Rf, tf = refine_pnp_gauss_newton(K, R0, t0, X, x)
    err_after = float(np.median(np.linalg.norm(project(K, Rf, tf, X) - x, axis=1)))
    assert err_before > 5.0                          # 初值确有偏（否则测不到下降）
    assert err_after < 1e-6                          # LM 收敛到近乎零重投影
    assert np.allclose(Rf, R, atol=1e-6)             # 复原真值旋转
    assert np.allclose(tf, t, atol=1e-6)             # 复原真值平移（含模长）


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_matrix_to_quat_roundtrip():
    R = _rot("y", np.radians(37)) @ _rot("z", np.radians(-52))
    q = matrix_to_quat(R)
    assert q[0] >= 0                                  # 约定 w≥0
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert np.allclose(_quat_to_matrix(q), R, atol=1e-9)


def test_matrix_to_quat_identity():
    assert matrix_to_quat(np.eye(3)) == (1.0, 0.0, 0.0, 0.0)
