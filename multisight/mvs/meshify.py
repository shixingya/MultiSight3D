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


def analyze_mesh_topology(vertices, faces):
    """网格拓扑与 FR-09 水密性检查（纯 numpy，无 scipy）。

    将每个三角面的三条无向边排序去重后计数：恰出现 1 次为边界边（开洞），>2 次为非流形边。
    封闭流形网格（水密） ⇔ 无边界边且无非流形边。欧拉示性数 χ = V − E + F（仅计被面引用
    的顶点），可辅助判断连通分量/洞情况。返回 stats 字典（供 mesh/stats.json 与 report 消费）。
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n_verts = int(len(vertices))
    if faces.shape[0] == 0:
        return {"faces": 0, "edges": 0, "boundary_edges": 0, "non_manifold_edges": 0,
                "is_watertight": False, "euler_characteristic": n_verts, "unused_vertices": n_verts}
    he = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)  # (3M,2)
    und = np.sort(he, axis=1)
    key = und[:, 0] * n_verts + und[:, 1]               # 无向边唯一键（索引 < n_verts）
    uniq, counts = np.unique(key, return_counts=True)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())
    used = np.unique(faces.ravel())
    euler = int(len(used) - len(uniq) + len(faces))
    return {
        "faces": int(len(faces)),
        "edges": int(len(uniq)),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "is_watertight": boundary == 0 and non_manifold == 0,
        "euler_characteristic": euler,
        "unused_vertices": int(n_verts - len(used)),
    }


def weld_mesh_vertices(vertices, faces, colors=None, *, tol: float):
    """跨视图缝合首步：重复/重合顶点焊接（纯 numpy，体素量化归并）。

    距离图像网格化逐视图直接拼接时，相邻视图重叠带会产几乎重合的重复顶点。本件按
    `tol` 就近量化顶点坐标（np.round 避免格边界漏焊）后取同格顶点均值合并为一个代表顶点（颜色同样取均值），重映射面索引，
    丢弃因焊接而退化的面（任两顶点重合）。返回 (vertices', faces', colors' 或 None)。
    tol<=0 或空网格时原样返回（不伤数据）。焊接后拓扑更紧凑（边界/重复减少），但仍开洞非水密。
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if tol is None or tol <= 0 or len(vertices) == 0:
        return vertices, faces, (None if colors is None else np.asarray(colors))
    if colors is not None:
        colors = np.asarray(colors)

    keys = np.round(vertices / tol).astype(np.int64)
    # 三维整数坐标 → 单键（限幅避免溢出，实际坐标量级内安全）
    off = keys.min(axis=0)
    k = keys - off
    span = max(int(k.max()) + 1, 1)
    flat = (k[:, 0].astype(np.int64) * span + k[:, 1]) * span + k[:, 2]
    uniq, inv = np.unique(flat, return_inverse=True)
    n_new = len(uniq)
    summed = np.zeros((n_new, 3))
    np.add.at(summed, inv, vertices)
    counts = np.bincount(inv, minlength=n_new)
    new_v = summed / counts[:, None]
    new_c = None
    if colors is not None:
        cd = colors.astype(np.float64)
        summed_c = np.zeros((n_new, cd.shape[1]))
        np.add.at(summed_c, inv, cd)
        mean_c = summed_c / counts[:, None]
        new_c = mean_c.round().astype(colors.dtype)

    new_f = inv[faces]                              # (M,3) 重映射
    degen = (new_f[:, 0] == new_f[:, 1]) | (new_f[:, 1] == new_f[:, 2]) | (new_f[:, 0] == new_f[:, 2])
    new_f = new_f[~degen]
    return new_v, new_f.astype(np.int64), new_c
