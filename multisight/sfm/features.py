"""2D 特征提取与匹配（自研，纯 numpy）。

这是 SfM 的第一块真实算法（FR-03「特征提取与匹配」），刻意不依赖 OpenCV/scipy：
- detect_corners：Harris 角点响应 + 高质量角点筛选 + 非极大值抑制 + 最小间距去重；
- describe：以角点为中心的定尺寸 patch，去均值 + L2 归一化（→ 点积即归一化互相关 NCC）；
- match_descriptors：暴力全对全最近邻 + Lowe 比值测试 + 双向（互）一致性校验。

坐标一律 (x=列, y=行)，与图像/像素惯例一致；输入灰度图为 (H, W) float/uint8。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assets.textures import load_image  # 复用已有的安全图像读取（损坏返回 None）


@dataclass(frozen=True)
class Corner:
    x: int
    y: int
    response: float


def to_gray(path_or_img) -> np.ndarray:
    """读图为 float64 灰度 [0,255]；失败抛 ValueError（上层据实报错，不静默）。"""
    if isinstance(path_or_img, np.ndarray):
        arr = path_or_img.astype(np.float64)
        return arr if arr.ndim == 2 else arr.mean(axis=2)
    im = load_image(path_or_img) if not hasattr(path_or_img, "convert") else path_or_img
    if im is None:
        raise ValueError(f"无法读取图像：{path_or_img}")
    return np.asarray(im.convert("L"), dtype=np.float64)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """可分离均值滤波（沿列、行各一次前缀和差分），O(N)、无 Python 双循环。"""
    if radius <= 0:
        return img.copy()
    h, w = img.shape
    win = 2 * radius + 1
    # 沿列方向：padded 宽 w+2r，前缀和左侧补 0 → 窗和 = cum[i+win]-cum[i]
    padx = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    cumx = np.concatenate([np.zeros((h, 1)), np.cumsum(padx, axis=1)], axis=1)
    out_x = (cumx[:, win:win + w] - cumx[:, 0:w]) / win
    # 沿行方向
    pady = np.pad(out_x, ((radius, radius), (0, 0)), mode="edge")
    cumy = np.vstack([np.zeros((1, w)), np.cumsum(pady, axis=0)])
    return (cumy[win:win + h] - cumy[0:h]) / win


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """中心差分梯度 Ix, Iy；边界用同侧填充保持尺寸一致。"""
    p = np.pad(img, 1, mode="edge")
    gy = p[2:, 1:-1] - p[:-2, 1:-1]        # ∂/∂row
    gx = p[1:-1, 2:] - p[1:-1, :-2]        # ∂/∂col
    return 0.5 * gx, 0.5 * gy


def detect_corners(gray: np.ndarray, *, max_corners: int = 400,
                   quality: float = 0.02, sigma: int = 2,
                   k: float = 0.04, min_distance: int = 8) -> list[Corner]:
    """Harris 角点：梯度结构张量 → R=det−k·trace² → 阈值 + 窗口极大 + 间距抑制。"""
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    if min(h, w) < 2 * (sigma + 2) + 1:
        return []
    gx, gy = _gradients(gray)
    # 结构张量分量，高斯/均值窗口平滑（sigma 作窗半径）
    ixx = _box_blur(gx * gx, sigma)
    iyy = _box_blur(gy * gy, sigma)
    ixy = _box_blur(gx * gy, sigma)
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    resp = det - k * trace * trace
    thr = max(quality * float(resp.max()), 1e-6)
    resp = np.where(resp > thr, resp, 0.0)

    # 非极大值抑制：3x3 局部窗口取最大者保留（向量化，用 sliding_window_view）
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(np.pad(resp, 1, mode="constant"), (3, 3))
    local_max = win.max(axis=(2, 3))
    resp = np.where(local_max == resp, resp, 0.0)

    ys, xs = np.nonzero(resp)
    if xs.size == 0:
        return []
    order = np.argsort(resp[ys, xs])[::-1]          # 响应降序贪心选点
    ys, xs = ys[order], xs[order]
    scores = resp[ys, xs]

    kept: list[Corner] = []
    taken_y: list[int] = []
    taken_x: list[int] = []
    md2 = min_distance * min_distance
    for y, x, s in zip(ys.tolist(), xs.tolist(), scores.tolist()):
        if taken_x:
            dx = np.asarray(taken_x) - x
            dy = np.asarray(taken_y) - y
            if bool(np.any(dx * dx + dy * dy < md2)):
                continue
        kept.append(Corner(x=x, y=y, response=float(s)))
        taken_x.append(x)
        taken_y.append(y)
        if len(kept) >= max_corners:
            break
    return kept


def describe(gray: np.ndarray, corners: list[Corner], *, patch: int = 8) -> np.ndarray:
    """定尺寸 patch 描述子：裁剪 → 缩放到 patch×patch → 去均值 → L2 归一化。

    返回 (N, patch*patch) float 矩阵；越界或空返回 (0,·)。缩放后再归一化，
    使平移/亮度小幅变化下点积≈NCC，便于匹配。
    """
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    r = patch
    if not corners:
        return np.zeros((0, patch * patch))
    desc = np.zeros((len(corners), patch * patch))
    for i, c in enumerate(corners):
        x0, y0 = c.x - r, c.y - r
        x1, y1 = c.x + r, c.y + r
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:      # 只取完整窗口，边界角点跳过
            desc[i] = np.nan
            continue
        win = gray[y0:y1, x0:x1]
        # 最近邻重采样到 patch×patch（无 scipy）
        yy = (np.linspace(0, win.shape[0] - 1, patch)).astype(int)
        xx = (np.linspace(0, win.shape[1] - 1, patch)).astype(int)
        p = win[np.ix_(yy, xx)]
        p = p - p.mean()
        nrm = float(np.linalg.norm(p))
        desc[i] = p.ravel() / nrm if nrm > 1e-9 else 0.0
    valid = ~np.isnan(desc).any(axis=1)
    return desc[valid]


def detect_and_describe(gray: np.ndarray, *, max_corners: int = 400,
                        patch: int = 8, **kw) -> tuple[list[Corner], np.ndarray]:
    """便捷组合：检测角点 → 仅对可完整裁剪的角点产出对齐的描述子。"""
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    corners = detect_corners(gray, max_corners=max_corners, **kw)
    keep = [c for c in corners if patch <= c.x < w - patch and patch <= c.y < h - patch]
    desc = describe(gray, keep, patch=patch)
    return keep, desc


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray, *,
                      ratio: float = 0.8, mutual: bool = True) -> list[tuple[int, int]]:
    """比值测试暴力匹配：返回 A/B 描述子行下标配对。

    描述子已 L2 归一化 → 距离用 SSD=2−2·(a·b)。对每个 a 取最近/次近 b，
    d1/d2<ratio 才接受；mutual 时要求反向也一致。空输入安全返回 []。
    """
    if desc_a.size == 0 or desc_b.size == 0:
        return []
    sim = desc_a @ desc_b.T                       # NCC ∈ [-1,1]
    dist = 2.0 - 2.0 * sim                         # SSD，越小越相似
    # 每个 a 的最近 / 次近 b
    idx1_a = np.argmin(dist, axis=1)
    d1_a = dist[np.arange(len(dist)), idx1_a]
    dist2 = dist.copy()
    dist2[np.arange(len(dist)), idx1_a] = np.inf
    idx2_a = np.argmin(dist2, axis=1)
    d2_a = dist2[np.arange(len(dist)), idx2_a]
    ok_a = (d1_a < ratio * d2_a) & (d1_a < 1.6)     # 比值 + 绝对相似度门槛

    if not mutual:
        return [(int(i), int(idx1_a[i])) for i in range(len(ok_a)) if ok_a[i]]

    # 反向：每个 b 的最近 a
    idx1_b = np.argmin(dist, axis=0)
    pairs: list[tuple[int, int]] = []
    for i in range(len(ok_a)):
        if not ok_a[i]:
            continue
        j = int(idx1_a[i])
        if int(idx1_b[j]) == i:                    # 双向一致
            pairs.append((i, j))
    return pairs
