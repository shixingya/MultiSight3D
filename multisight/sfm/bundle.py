"""全局捆绑调整（Bundle Adjustment）：LM 联合精化相机位姿与 3D 点，最小化总重投影误差。

自研 SfM 主线的收口一环（PRD §11-1「完全自研」，纯 numpy）。给定初始位姿/结构（来自
两视图 + 增量注册）与 2D-3D 观测，用 Levenberg-Marquardt 在「自由参数 = 各相机 6 自由度
（左扰动 exp）+ 各 3D 点 3 自由度」上优化 Σ‖x_obs − π(K,R,t,X)‖²。

规范自由度（gauge）：完全固定首相机 [R₀|t₀]（世界系原点/朝向/尺度随之锁定），其余相机与
全部点自由。带阻尼与单调接受保护，避免病态步跑飞。小规模问题用稠密法方程即可（测试规模）。
"""

from __future__ import annotations

import numpy as np

from .geometry import _exp_so3, _project_to_so3


def _reproj_and_jac(K, R, t, X, xobs):
    """单观测的残差 r=x-π(X) 及雅可比：相机块 (2×6, 左扰动)、点块 (2×3)。退化（z≈0）返回 None。"""
    Xc = R @ X + t
    z = Xc[2]
    if abs(z) < 1e-9:
        return None, None, None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pred = np.array([fx * Xc[0] / z + cx, fy * Xc[1] / z + cy])
    r = xobs - pred
    dpx = np.array([[fx / z, 0.0, -fx * Xc[0] / z ** 2],
                    [0.0, fy / z, -fy * Xc[1] / z ** 2]])
    rx = Xc - t                                    # = R X（左扰动仅作于 R X）
    skew = np.array([[0.0, -rx[2], rx[1]],
                     [rx[2], 0.0, -rx[0]],
                     [-rx[1], rx[0], 0.0]])
    Jc = dpx @ np.hstack([-skew, np.eye(3)])       # 2×6
    Jp = dpx @ R                                   # 2×3
    return r, Jc, Jp


def mean_reprojection_error(K, cameras, points3d, measurements) -> float:
    """观测上的平均重投影误差（像素），衡量重建质量。"""
    if not measurements:
        return float("nan")
    acc = 0.0
    for c, p, u, v in measurements:
        R, t = cameras[c]
        Xc = R @ points3d[p] + t
        if abs(Xc[2]) < 1e-12:
            acc += float("inf")
            continue
        px = K @ Xc
        pred = px[:2] / px[2]
        acc += float(np.hypot(u - pred[0], v - pred[1]))
    return acc / len(measurements)


def bundle_adjustment(K, cameras, points3d, measurements, *, iters: int = 25,
                      tol: float = 1e-9, lam0: float = 1e-3
                      ) -> tuple[list, np.ndarray, float]:
    """LM 全局 BA：联合优化 cameras[1:] 位姿与全部 3D 点，最小化总重投影 SSE。

    cameras: list[(R(3,3), t(3))]；points3d: (Np,3)；measurements: list[(cam_idx, pt_idx, u, v)]。
    首相机固定以消除规范自由度。返回 (精化后 cameras, 精化后 points3d, 平均重投影误差)。
    无可优化参数或全部退化时原样返回。
    """
    K = np.asarray(K, dtype=np.float64)
    cams = [(np.asarray(R, dtype=np.float64).copy(), np.asarray(t, dtype=np.float64).reshape(3).copy())
            for R, t in cameras]
    Xs = np.asarray(points3d, dtype=np.float64).copy()
    ms = [(int(c), int(p), float(u), float(v)) for c, p, u, v in measurements]
    n_cams, n_pts = len(cams), len(Xs)
    if n_cams < 2 or not ms:
        return cams, Xs, mean_reprojection_error(K, cams, Xs, ms)

    # 自由参数布局：cams[1:] 各 6，points 各 3
    cam_free = list(range(1, n_cams))
    cam_base = {c: 6 * (k) for k, c in enumerate(cam_free)}
    n_point_params = 3 * n_pts
    pt_base = len(cam_free) * 6
    D = pt_base + n_point_params

    def _resid_and_J(cams_cur, Xcur):
        r = np.empty(2 * len(ms))
        J = np.zeros((2 * len(ms), D))
        sse = 0.0
        for i, (c, p, u, v) in enumerate(ms):
            R, t = cams_cur[c]
            rr, Jc, Jp = _reproj_and_jac(K, R, t, Xcur[p], np.array([u, v]))
            if rr is None:
                return None, None, np.inf
            r[2 * i:2 * i + 2] = rr
            if c in cam_base:
                J[2 * i:2 * i + 2, cam_base[c]:cam_base[c] + 6] += Jc
            J[2 * i:2 * i + 2, pt_base + 3 * p:pt_base + 3 * p + 3] += Jp
            sse += float(rr @ rr)
        return r, J, sse

    lam = lam0
    r, J, cost = _resid_and_J(cams, Xs)
    if r is None:
        return cams, Xs, mean_reprojection_error(K, cams, Xs, ms)
    for _ in range(iters):
        H = J.T @ J
        g = J.T @ r
        diag = np.diag(np.diag(H)).copy()
        stepped = False
        for _trial in range(8):
            try:
                delta = np.linalg.solve(H + lam * diag, g)
            except np.linalg.LinAlgError:
                lam *= 4.0
                continue
            new_cams = list(cams)
            for c in cam_free:
                d = delta[cam_base[c]:cam_base[c] + 6]
                R0, t0 = cams[c]
                new_cams[c] = (_project_to_so3(_exp_so3(d[:3]) @ R0), t0 + d[3:])
            new_X = Xs + delta[pt_base:pt_base + 3 * n_pts].reshape(n_pts, 3)
            rn, Jn, costn = _resid_and_J(new_cams, new_X)
            if rn is not None and costn < cost:
                cams, Xs, r, J, cost = new_cams, new_X, rn, Jn, costn
                lam = max(lam * 0.5, 1e-9)
                stepped = True
                break
            lam *= 4.0
        if not stepped or abs(cost) < tol:
            break
    return cams, Xs, mean_reprojection_error(K, cams, Xs, ms)
