"""深度图 → 三角网格（距离图像网格化 / range-image meshing，纯 numpy，v0.3 起步）。

`plane_sweep_depth` 在参考视图的规则子采样网格 (us,vs) 上产出深度图；本件把该深度图按
网格拓扑直接三角化——每个相邻四像素 (i,j)/(i,j+1)/(i+1,j)/(i+1,j+1) 若四角深度均有效，
则生成两个三角形；含任一无效角点的格子跳过，天然在遮挡/无纹理空洞处留洞（诚实：不臆造
几何）。反投影与 `fuse.depth_map_to_points` 共用相机约定：世界→相机 x = K(R·W + t)，
W = Rᵀ(Xc − t)。可选 `max_edge_gap` 丢弃最长边过大的三角形，抑制深度断裂处的假面。

这是 v0.3 网格重建的第一件自研基元（先做可独立验证的逐视图距离图像网格化；跨视图统一
网格 / Poisson 等整体重构为后续里程碑）。
"""

from __future__ import annotations

import numpy as np


def depth_map_to_mesh(us, vs, depth, K: np.ndarray, RT, *,
                      max_edge_gap: float | None = None):
    """把规则采样网格上的深度图三角化为世界系网格。

    us/vs/depth：同形 (Hs,Ws) 数组（plane_sweep_depth 的网格与其输出对齐；无效深度为 nan）。
    返回 (vertices (N,3) 世界系, faces (M,3) int64 索引)。全无效则返回空数组。
    max_edge_gap：若给定（>0），三角形最长边世界长度超过该值则丢弃（抑制假面）。
    """
    depth = np.asarray(depth, dtype=np.float64)
    us = np.asarray(us, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    if not (depth.shape == us.shape == vs.shape):
        raise ValueError("us/vs/depth 形状需一致")
    if depth.ndim != 2:
        raise ValueError("深度图需为二维网格 (Hs,Ws)")
    K = np.asarray(K, dtype=np.float64)
    R, t = (np.asarray(RT[0], dtype=np.float64),
            np.asarray(RT[1], dtype=np.float64).reshape(3))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    hs, ws = depth.shape

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    d = np.where(valid, depth, 0.0)
    Xc = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=-1)   # (Hs,Ws,3) 相机系
    W = (Xc - t) @ R                                                       # 世界系 (Hs,Ws,3)

    vid = -np.ones((hs, ws), dtype=np.int64)
    vid[valid] = np.arange(int(valid.sum()), dtype=np.int64)
    vertices = W[valid]                                                    # (N,3)

    if hs < 2 or ws < 2:
        return vertices, np.zeros((0, 3), dtype=np.int64)

    a = vid[:-1, :-1]      # (i, j)
    b = vid[:-1, 1:]       # (i, j+1)
    c = vid[1:, :-1]       # (i+1, j)
    e = vid[1:, 1:]        # (i+1, j+1)
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (e >= 0)
    if not quad.any():
        return vertices, np.zeros((0, 3), dtype=np.int64)
    A, B, C, E = a[quad], b[quad], c[quad], e[quad]
    # 每格沿 A-E 对角线切成两个三角形，保持环绕方向一致
    faces = np.concatenate([np.stack([A, B, E], axis=1),
                            np.stack([A, E, C], axis=1)], axis=0).astype(np.int64)

    if max_edge_gap is not None and max_edge_gap > 0 and len(faces):
        p = vertices[faces]                              # (M,3,3)
        edges = np.stack([np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
                          np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                          np.linalg.norm(p[:, 2] - p[:, 0], axis=1)], axis=1)  # (M,3)
        keep = edges.max(axis=1) <= max_edge_gap
        faces = faces[keep]

    return vertices, faces
