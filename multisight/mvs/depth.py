"""平面扫描深度图估计（自研 MVS 第一件，纯 numpy）。

对参考视图的（子采样）像素，沿其视线在给定深度区间采样候选 3D 点，把它投到每个
邻视图、以去均值 L2 归一化互相关（NCC）度量 patch 一致性，跨邻视图平均代价后取
argmin 深度。相机约定与自研 SfM 一致：**世界→相机** x = K(R·W + t)，相机中心 C = -Rᵀt。

诚实原则：无邻视图可见（投影越界/负深度）或最优代价过高的像素标为无效（depth=nan、
cost=+inf），绝不臆造深度。参考视图需有纹理，否则 NCC 退化——由上层据置信度取舍。
"""

from __future__ import annotations

import numpy as np


def _pixel_rays(K: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """像素 (N,) → 归一化相机视线方向 (N,3)（z=1 平面上的方向，未单位化）：K⁻¹·[u,v,1]。"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (np.asarray(us, dtype=np.float64) - cx) / fx
    y = (np.asarray(vs, dtype=np.float64) - cy) / fy
    return np.column_stack([x, y, np.ones_like(x)])


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界点 (N,3) 投到相机 (R,t)：返回 (像素 u,v (N,2), 相机系深度 z (N,))。"""
    Xc = W @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def _bilinear_patches(gray: np.ndarray, us: np.ndarray, vs: np.ndarray, r: int) -> np.ndarray:
    """以 (us,vs) 为中心采集 (2r+1)² patch（双线性采样）→ (N, P²)。越界采样记 NaN。"""
    h, w = gray.shape
    offs = np.arange(-r, r + 1, dtype=np.float64)
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    oy = oy.ravel()
    ox = ox.ravel()
    xs = us[:, None] + ox[None, :]
    ys = vs[:, None] + oy[None, :]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    valid = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    g = gray
    val = (g[y0c, x0c] * (1 - fx) * (1 - fy) + g[y0c, x0c + 1] * fx * (1 - fy)
           + g[y0c + 1, x0c] * (1 - fx) * fy + g[y0c + 1, x0c + 1] * fx * fy)
    return np.where(valid, val, np.nan)


def _normalize_rows(p: np.ndarray) -> np.ndarray:
    """逐行去均值 + L2 归一化；含 NaN 或零范数的行返回全 NaN（视为不可用）。"""
    nan = np.isnan(p).any(axis=1)
    q = np.where(np.isnan(p), 0.0, p)
    q = q - q.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(nrm > 1e-9, q / np.where(nrm > 1e-9, nrm, 1.0), np.nan)
    q[nan] = np.nan
    return q


def plane_sweep_depth(ref_gray, src_grays, K: np.ndarray, ref_RT, src_RTs,
                      depths, *, patch: int = 3, stride: int = 4,
                      max_cost: float = 0.5):
    """参考视图平面扫描深度图。

    参数
    ----
    ref_gray : (H,W) 参考视图灰度（float）。
    src_grays: 邻视图灰度列表（与 src_RTs 一一对应）。
    K        : (3,3) 共享针孔内参。
    ref_RT   : (R, t) 参考视图世界→相机位姿。
    src_RTs  : [(R, t)] 各邻视图世界→相机位姿。
    depths   : (D,) 递增候选深度（相机系 z，单位与位姿平移一致）。
    patch    : patch 半径 r（窗口 (2r+1)²）。stride：像素子采样间隔。
    max_cost : 有效像素的最大平均 NCC 代价（1-NCC），超过判无效。

    返回 depth, cost, conf, grid —— depth/cost/conf 均为参考视图子采样网格 (Hs, Ws) 上的数组
    （depth 最优深度、无效为 nan；cost 最优平均代价；conf 邻视图平均可见数）；
    grid=(us, vs) 为对齐的采样像素坐标，供 fuse.depth_map_to_points 直接反投影。
    """
    ref = np.asarray(ref_gray, dtype=np.float64)
    h, w = ref.shape
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim != 1 or len(depths) == 0:
        raise ValueError("depths 需为非空一维数组")
    if len(src_grays) != len(src_RTs):
        raise ValueError("src_grays 与 src_RTs 数量需一致")
    if not src_grays:
        raise ValueError("plane_sweep_depth 需至少一个邻视图")
    K = np.asarray(K, dtype=np.float64)
    Rr, tr = (np.asarray(ref_RT[0], dtype=np.float64), np.asarray(ref_RT[1], dtype=np.float64).reshape(3))
    srcs = [(np.asarray(g, dtype=np.float64),
             np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3))
            for g, (R, t) in zip(src_grays, src_RTs)]

    # 子采样像素网格（整数坐标，留 patch 边距）
    r = int(patch)
    m = r + 1
    cols = np.arange(m, w - m, stride)
    rows = np.arange(m, h - m, stride)
    gj, _gi = np.meshgrid(rows, cols, indexing="ij")    # (Hs, Ws)
    shape = gj.shape
    vs = gj.astype(np.float64)
    us = np.broadcast_to(cols.astype(np.float64)[None, :], shape).copy()
    flat_u = us.ravel()
    flat_v = vs.ravel()

    rays = _pixel_rays(K, flat_u, flat_v)               # (S,3)，z=1
    ref_patch = _normalize_rows(_bilinear_patches(ref, flat_u, flat_v, r))  # (S,P²)
    S = flat_u.size
    D = depths.size

    best_cost = np.full(S, np.inf)
    best_depth = np.full(S, np.nan)
    best_conf = np.zeros(S)
    for di, d in enumerate(depths):
        # 参考相机视线深度 d 处的世界点：Xc = d·ray → W = Rᵀ(Xc - t)
        Xc = rays * d
        W = (Xc - tr) @ Rr
        costs = np.zeros(S)
        seen = np.zeros(S)
        for g, R, t in srcs:
            proj, z = _project(K, R, t, W)
            inb = (z > 1e-6) & (proj[:, 0] >= m) & (proj[:, 0] < g.shape[1] - m) \
                  & (proj[:, 1] >= m) & (proj[:, 1] < g.shape[0] - m)
            if not inb.any():
                continue
            sp = _normalize_rows(_bilinear_patches(g, proj[inb, 0], proj[inb, 1], r))
            rp = ref_patch[inb]
            ncc = np.nansum(sp * rp, axis=1)            # ∈[-1,1]（有效行）
            ok = ~np.isnan(ncc)
            c = 1.0 - np.where(ok, ncc, -1.0)           # 不可用行代价置大（=2）
            idx = np.where(inb)[0][ok]
            costs[idx] += c[ok]
            seen[idx] += 1.0
        avg = np.where(seen > 0, costs / np.maximum(seen, 1e-9), np.inf)
        better = (seen > best_conf) | ((seen == best_conf) & (avg < best_cost))
        best_cost = np.where(better, avg, best_cost)
        best_depth = np.where(better, d, best_depth)
        best_conf = np.where(better, seen, best_conf)

    valid = (best_cost < max_cost) & (best_conf >= 1)
    depth_out = np.where(valid, best_depth, np.nan).reshape(shape)
    cost_out = np.where(valid, best_cost, np.inf).reshape(shape)
    conf_out = best_conf.reshape(shape)
    return depth_out, cost_out, conf_out, (us, vs)
