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
