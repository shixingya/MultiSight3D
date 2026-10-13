"""深度图 → 稠密 3D 点：反投影 + 跨视图一致性/置信过滤（纯 numpy）。

`plane_sweep_depth` 给出参考视图子采样网格上的深度图；本件把这些 (u,v,depth) 反投影回
世界系点，并按置信度（可见邻视图数）与代价阈值诚实过滤——低置信/高代价/无效深度像素
一律丢弃，绝不把猜测的深度变成点。多视图融合时对各视图分别调用再合并（上层去重）。

相机约定与 depth 一致：世界→相机 x = K(R·W + t)，反投影 W = Rᵀ(depth·K⁻¹·ṽ − t)。
"""

from __future__ import annotations

import numpy as np


def depth_map_to_points(us, vs, depth, K: np.ndarray, RT, *,
                        cost=None, conf=None, min_conf: int = 1,
                        max_cost: float | None = None):
    """把一张深度图（子采样网格 us,vs,depth）反投影为世界点云。

    us/vs/depth：同形 (Hs,Ws) 数组（plane_sweep_depth 的网格与其输出对齐）。
    返回 (points (N,3), mask (N,))：mask 为该采样点是否通过过滤（供上层统计置信）。
    """
    depth = np.asarray(depth, dtype=np.float64)
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    if not (depth.shape == us.shape == vs.shape):
        raise ValueError("us/vs/depth 形状需一致")
    K = np.asarray(K, dtype=np.float64)
    R, t = (np.asarray(RT[0], dtype=np.float64), np.asarray(RT[1], dtype=np.float64).reshape(3))

    d = depth.ravel()
    uu = us.ravel()
    vv = vs.ravel()
    valid = np.isfinite(d) & (d > 0)
    if conf is not None:
        valid &= (np.asarray(conf).ravel() >= min_conf)
    if cost is not None and max_cost is not None:
        valid &= (np.asarray(cost).ravel() <= max_cost)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = np.zeros((d.size, 3))
    X[:, 0] = (uu - cx) / fx * d
    X[:, 1] = (vv - cy) / fy * d
    X[:, 2] = d                                       # 参考相机系坐标 Xc
    W = (X - t) @ R                                   # W = Rᵀ(Xc - t)
    return W[valid], valid


_OFFS3 = np.array([(i, j, k)
                   for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                  dtype=np.int64)


def remove_scatter_noise(points, *, voxel: float, min_neighbors: int = 4, colors=None):
    """统计散噪声剔除：以边长 voxel 的体素格统计每点 3×3×3 邻域内的点数，邻域点数
    < min_neighbors 的孤立点（飘渺噪声/重影）丢弃。纯 numpy（体素字典计数，无需 KDTree）。

    points (N,3)；colors (N,k) 可选，随行过滤。返回 (kept_points, kept_colors 或 None)。
    诚实原则：voxel<=0 或 min_neighbors<=1 时不剔除（全量保留）。"""
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0]
    if n == 0:
        empty = np.zeros((0, 3))
        return empty, (None if colors is None else np.asarray(colors)[0:0])
    if voxel <= 0 or min_neighbors <= 1:
        return P, (None if colors is None else np.asarray(colors))
    key = np.floor(P / voxel).astype(np.int64)
    uk, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    cd = {tuple(int(c) for c in cc): int(k) for cc, k in zip(uk, cnt)}
    occ = np.zeros(len(uk), dtype=np.int64)
    for ci, cc in enumerate(uk):
        s = 0
        for o in _OFFS3:
            s += cd.get((int(cc[0]) + o[0], int(cc[1]) + o[1], int(cc[2]) + o[2]), 0)
        occ[ci] = s
    keep = occ[inv] >= min_neighbors
    out_c = None if colors is None else np.asarray(colors)[keep]
    return P[keep], out_c


def voxel_downsample(points, *, voxel: float, colors=None):
    """体素网格降采样：落入同一边长 voxel 体素的点归并为一个代表点（体素质心），
    颜色（如有）取体素内均值。纯 numpy（np.unique 分组）。用于控制稠密点云/导出体积。

    points (N,3)；colors (N,k) 可选，同步归并。返回 (points_out (M,3), colors_out 或 None)。
    诚实原则：voxel<=0 或空点云时原样返回（不强行归并）；M<=N 恒成立（只降不升）。"""
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0]
    if n == 0:
        return np.zeros((0, 3)), (None if colors is None else np.asarray(colors)[0:0])
    if voxel <= 0:
        return P, (None if colors is None else np.asarray(colors))
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    m = cnt.shape[0]
    out = np.zeros((m, 3), dtype=np.float64)
    np.add.at(out, inv, P)
    out /= cnt[:, None]
    if colors is None:
        return out, None
    C = np.asarray(colors)
    k = C.shape[1] if C.ndim == 2 else 1
    cd = np.zeros((m, k), dtype=np.float64)
    np.add.at(cd, inv, C.astype(np.float64))
    cd /= cnt[:, None]
    if C.dtype == np.uint8:
        cd = np.round(cd).astype(np.uint8)
    return out, cd


def point_cloud_metrics(points):
    """稠密点云尺寸度量（FR-05/FR-08，纯 numpy）。

    centroid 质心；bbox_extent 轴对齐盒展幅；pca_std 沿三个主轴的标准差（协方差矩阵
    特征值平方根，升序）——与坐标轴无关地反映点云真实伸展/扁平程度。返回 dict；
    空点云给诚实零值（count=0）。"""
    P = np.asarray(points, dtype=np.float64)
    n = P.shape[0] if P.size else 0
    if n == 0:
        return {"count": 0, "centroid": [0.0, 0.0, 0.0],
                "bbox_extent": [0.0, 0.0, 0.0], "pca_std": [0.0, 0.0, 0.0]}
    P = P.reshape(-1, 3)
    centroid = P.mean(axis=0)
    bbox_extent = P.max(axis=0) - P.min(axis=0)
    cov = np.cov((P - centroid).T)                      # 3x3 协方差
    eig = np.linalg.eigvalsh(np.atleast_2d(cov))        # 升序特征值
    pca_std = np.sqrt(np.clip(eig, 0.0, None))
    return {"count": int(n),
            "centroid": [round(float(x), 6) for x in centroid],
            "bbox_extent": [round(float(x), 6) for x in bbox_extent],
            "pca_std": [round(float(x), 6) for x in pca_std]}
