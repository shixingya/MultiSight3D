"""全局捆绑调整单测（纯合成：已知真值位姿/结构 → 扰动初值 → BA 应回落到真值 basin）。

零二进制入库；BA 固定首相机消除规范自由度，故恢复结果可与真值直接比较。
"""

from __future__ import annotations

import numpy as np

from multisight.sfm import bundle_adjustment, mean_reprojection_error


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(K, R, t, X):
    Xc = X @ R.T + t
    hx = K @ Xc.T
    return (hx[:2] / hx[2:3]).T


def _scene(n_pts=40, seed=3):
    rng = np.random.default_rng(seed)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    X = np.column_stack([rng.uniform(-1, 1, n_pts), rng.uniform(-0.8, 0.8, n_pts),
                         rng.uniform(4, 6, n_pts)])
    cams_true = [(np.eye(3), np.zeros(3)),
                 (_rot("y", np.radians(8)), np.array([0.4, 0.02, 0.05])),
                 (_rot("y", np.radians(16)), np.array([0.8, 0.05, 0.10]))]
    obs = [(c, p, *map(float, _project(K, R, t, X)[p]))
           for c, (R, t) in enumerate(cams_true) for p in range(n_pts)]
    return K, cams_true, X, obs


def test_ba_reduces_reprojection_from_perturbed_init():
    K, cams_true, X_true, obs = _scene()
    rng = np.random.default_rng(11)
    # 扰动初值：首相机保持真值（gauge 锚），其余位姿 + 结构注入噪声
    cams0 = [cams_true[0]]
    for R, t in cams_true[1:]:
        cams0.append((_rot("y", np.radians(2.0)) @ R, t + rng.normal(0, 0.03, 3)))
    X0 = X_true + rng.normal(0, 0.12, X_true.shape)
    err_before = mean_reprojection_error(K, cams0, X0, obs)
    cams, Xs, err_after = bundle_adjustment(K, cams0, X0, obs, iters=40)
    assert err_before > 2.0                            # 初值确有可观误差
    assert err_after < 1e-4                            # BA 收敛到近乎零重投影
    assert np.allclose(cams[1][1], cams_true[1][1], atol=0.05)   # 恢复平移（gauge 已锚定）
    assert np.linalg.norm(Xs - X_true) / np.linalg.norm(X_true) < 0.02  # 结构相对误差 <2%


def test_ba_leaves_identity_when_already_optimal():
    K, cams_true, X_true, obs = _scene(n_pts=30, seed=5)
    err0 = mean_reprojection_error(K, cams_true, X_true, obs)
    cams, Xs, err = bundle_adjustment(K, cams_true, X_true, obs, iters=20)
    assert err <= err0 + 1e-9                          # 单调：不会劣于真值初值
    assert err < 1e-9


def test_ba_noop_with_too_few_cameras():
    K = np.eye(3)
    cams = [(np.eye(3), np.zeros(3))]
    X = np.array([[0.0, 0.0, 2.0]])
    obs = [(0, 0, 0.0, 0.0)]
    out_cams, out_X, err = bundle_adjustment(K, cams, X, obs)
    assert len(out_cams) == 1
    assert np.allclose(out_X, X)
