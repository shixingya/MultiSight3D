"""两视图对极几何（自研，纯 numpy）。

承接 features 的匹配点对，估计基础矩阵 F（满足 x2ᵀ F x1 = 0）：
- estimate_fundamental：Hartley 归一化 + 8 点线性解 + 强制 rank-2（奇异值截断）；
- ransac_fundamental：最小 8 点样本 + 对称极线距离计内点，鲁棒剔除外点。

这是自研 SfM 从「像素匹配」迈向「相机相对位姿/三角化」的关键一步（v0.1）。
输入点集为 (N,2) 像素坐标 (u,v)。
"""

from __future__ import annotations

import numpy as np


def _norm_from_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化：平移到质心、缩放使平均距离=√2。返回 (归一化点, 3×3 变换)。"""
    pts = np.asarray(pts, dtype=np.float64)
    cx, cy = pts.mean(axis=0)
    d = pts - np.array([cx, cy])
    mean = np.sqrt((d ** 2).sum(axis=1).mean()) or 1.0
    s = np.sqrt(2.0) / mean
    T = np.array([[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]])
    hn = np.column_stack([d[:, 0] * s, d[:, 1] * s, np.ones(len(d))])
    return hn, T


def estimate_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """8 点法估计 F（归一化 + rank-2 约束）。需 ≥8 对；退化/共线返回全零附近矩阵。"""
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    if len(p1) < 8 or len(p1) != len(p2):
        raise ValueError("estimate_fundamental 需要 ≥8 对点")
    n1, T1 = _norm_from_pts(p1)
    n2, T2 = _norm_from_pts(p2)
    A = np.column_stack([
        n1[:, 0] * n2[:, 0], n1[:, 1] * n2[:, 0], n2[:, 0],
        n1[:, 0] * n2[:, 1], n1[:, 1] * n2[:, 1], n2[:, 1],
        n1[:, 0], n1[:, 1], np.ones(len(n1)),
    ])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    # 强制 rank-2：把最小奇异值置 0
    U, S, Vt2 = np.linalg.svd(F)
    S2 = np.diag([S[0], S[1], 0.0])
    F = U @ S2 @ Vt2
    F = T2.T @ F @ T1
    n = np.linalg.norm(F)
    return F / n if n > 1e-12 else F


def _sym_epipolar_distance(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """对称极线距离（Hartley&Zisserman 式 11.12）：点到两条对应极线距离之积和。"""
    h1 = np.column_stack([p1[:, 0], p1[:, 1], np.ones(len(p1))])
    h2 = np.column_stack([p2[:, 0], p2[:, 1], np.ones(len(p2))])
    line1 = (F @ h1.T).T          # 对每张 p1，p2 侧的极线 (a,b,c)
    line2 = (F.T @ h2.T).T        # 对每张 p2，p1 侧的极线
    err = np.einsum("ij,ij->i", h2, line1)  # x2ᵀ F x1
    num = err ** 2
    den1 = line1[:, 0] ** 2 + line1[:, 1] ** 2
    den2 = line2[:, 0] ** 2 + line2[:, 1] ** 2
    return num * (1.0 / (den1 + 1e-12) + 1.0 / (den2 + 1e-12))


def ransac_fundamental(pts1: np.ndarray, pts2: np.ndarray, *,
                       iters: int = 200, threshold: float = 1.0,
                       seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC 鲁棒估计 F。返回 (最优 F, 内点掩码)；点数不足或无样本返回 (None, 全 False)。

    threshold 为对称极线距离阈值（像素²量级）。固定 seed 保证测试确定性。
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = len(p1)
    if n < 8 or n != len(p2):
        return None, np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)
    best_F: np.ndarray | None = None
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=8, replace=False)
        try:
            F = estimate_fundamental(p1[idx], p2[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(F).all():
            continue
        d = _sym_epipolar_distance(F, p1, p2)
        mask = d < threshold
        c = int(mask.sum())
        if c > best_count:
            best_count, best_inliers, best_F = c, mask, F
    # 用全部内点再精化一次（≥8 时）
    if best_F is not None and best_count >= 8:
        best_F = estimate_fundamental(p1[best_inliers], p2[best_inliers])
        d = _sym_epipolar_distance(best_F, p1, p2)
        best_inliers = d < threshold
    return best_F, best_inliers


def triangulate(P1: np.ndarray, P2: np.ndarray,
                pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """逐点 DLT 三角化：由两视图投影矩阵 P1,P2 (3×4) 与像素对应 (N,2) 解 (N,3)。

    每点用 4 个方程 (x×P₁X, y×P₁X, x×P₂X, y×P₂X) 的最小二乘解（SVD）；返回齐次归一化后的
    欧氏坐标。退化/无穷远点（最后一维≈0）以 NaN 标记，由上层据实取舍。
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    n = len(pts1)
    out = np.full((n, 3), np.nan)
    for i in range(n):
        u, v = pts1[i]
        u2, v2 = pts2[i]
        A = np.vstack([
            u * P1[2] - P1[0],
            v * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            continue
        out[i] = X[:3] / X[3]
    return out


def recover_pose(F: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从基础矩阵 F 与内参 K 恢复相机2 相对位姿 [R|t]（相机1=[I|0]）。

    E=KᵀFK 的 SVD 分解出 4 组候选 (R,±t)，逐组三角化，取两相机均正深度（手性）
    点数最多的一；返回 (R, t(单位长), X)。t 尺度不可观（单对视图），以单位向量表示。
    """
    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    E = K.T @ F @ K
    U, _S, Vt = np.linalg.svd(E)
    W = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if np.linalg.det(U @ Vt) < 0:
        U = U * np.array([1, 1, -1])            # 强制右手系
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    u3 = U[:, 2]
    cands = [(R1, u3), (R1, -u3), (R2, u3), (R2, -u3)]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
    for R, t in cands:
        if np.linalg.det(R) < 0:
            R = R * -1.0
        P2 = K @ np.hstack([R, t[:, None]])
        X = triangulate(P1, P2, pts1, pts2)
        ok = ~np.isnan(X).any(axis=1)
        if not ok.any():
            continue
        Xf = X[ok]
        z1 = Xf[:, 2]
        z2 = (R @ Xf.T)[2] + t[2]
        score = int(np.count_nonzero((z1 > 0) & (z2 > 0)))
        if best is None or score > best[0]:
            best = (score, R, t / np.linalg.norm(t), X)
    if best is None:
        R, t = cands[0][0], cands[0][1] / np.linalg.norm(cands[0][1])
        return R, t, np.full((len(pts1), 3), np.nan)
    return best[1], best[2], best[3]


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """把任意 3×3 投影到最近旋转（SVD 极分解），强制右手系 det=+1。"""
    U, _S, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def solve_pnp_dlt(K: np.ndarray, points3d: np.ndarray, points2d: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """已知 3D 点 (N,3) + 对应像素 (N,2) + 内参 K，线性 DLT 解相机位姿 [R|t]。

    约定世界系→相机系 x = K [R|t] X（需 ≥6 对、非退化）。归一化像素 xn=K⁻¹x̃ 后按
    xn×(M X̃)=0 组 2N×12 零空间；由 det(M₃) 定尺度符号 λ（det(λR)=λ³），R=M₃/λ 投影到 SO(3)，
    t=M[:,3]/λ。病态/退化抛 ValueError。PnP 中 3D 结构尺度已知，故 t 的模长有意义（区别于两视图）。
    """
    X = np.asarray(points3d, dtype=np.float64)
    x = np.asarray(points2d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n = len(X)
    if n < 6 or len(x) != n:
        raise ValueError("solve_pnp_dlt 需要 ≥6 对 2D-3D 对应")
    Kinv = np.linalg.inv(K)
    hx = np.column_stack([x[:, 0], x[:, 1], np.ones(n)])
    nx = (Kinv @ hx.T).T
    nx = nx[:, :2] / nx[:, 2:3]                       # 归一化像面 (u', v')
    Xh = np.column_stack([X, np.ones(n)])             # (n,4)
    z = np.zeros(4)
    A = np.empty((2 * n, 12))
    for i in range(n):
        u, v = nx[i]
        A[2 * i] = np.concatenate([Xh[i], z, -u * Xh[i]])       # m₁·X̃ - u·m₃·X̃
        A[2 * i + 1] = np.concatenate([z, Xh[i], -v * Xh[i]])   # m₂·X̃ - v·m₃·X̃
    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1]
    M = m.reshape(3, 4)                               # 行 m₁,m₂,m₃ = λ[R|t]
    M3 = M[:, :3]
    det = np.linalg.det(M3)
    if abs(det) < 1e-12:
        raise ValueError("solve_pnp_dlt：退化构型（无法定尺度）")
    lam = np.copysign(np.abs(det) ** (1.0 / 3.0), det)          # λ³=det(λR)
    R = _project_to_so3(M3 / lam)
    t = M[:, 3] / lam
    return R, t


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """旋转矩阵 → 单位四元数 (w, x, y, z)（COLMAP images.txt 惯例，数值稳定分支法）。"""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q                              # 约定 w≥0，写法唯一
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
