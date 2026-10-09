"""自研 SfM 特征模块单测（纯合成图像，零二进制入库）。

验证真实算法性质而非 mock：整数平移的双图上，检测+描述+匹配应精确恢复已知位移，
比值测试能拒歧义、边界输入安全返回。
"""

from __future__ import annotations

import numpy as np

from multisight.sfm import (
    Corner, detect_corners, describe, detect_and_describe, match_descriptors, to_gray,
)


def _scene(seed: int = 0) -> np.ndarray:
    """带微弱噪声底 + 若干互异灰度的实心方块（每块产生 4 个强角点）。"""
    rng = np.random.default_rng(seed)
    img = 100 + rng.random((140, 140)) * 8.0
    places = [(20, 20), (70, 24), (24, 78), (86, 84), (60, 104)]  # (x, y)，彼此间距 > 20
    for k, (x, y) in enumerate(places):
        img[y:y + 12, x:x + 12] = 40 + k * 22
    return img


def _shift_views(scene, dx: int, dy: int):
    """同一场景取两个窗口：img1 的角点 (x,y) 在 img2 中出现在 (x-dx, y-dy)。"""
    img1 = scene[0:120, 0:120]
    img2 = scene[dy:dy + 120, dx:dx + 120]
    return img1, img2


def test_detect_finds_corners_near_squares():
    img = _scene()
    corners = detect_corners(img, max_corners=200, min_distance=6)
    assert len(corners) > 0
    # 至少应命中一个方块附近（角点应聚集在方块四角 ±2 内）
    centers = np.array([[x, y] for (x, y) in
                        [(20, 20), (70, 24), (24, 78), (86, 84), (60, 104)]])
    sizes = np.array([[12, 12]] * len(centers))
    hit = 0
    for c in corners:
        for (cx, cy), (sx, sy) in zip(centers, sizes):
            corners_of_block = [(cx, cy), (cx + sx, cy), (cx, cy + sy), (cx + sx, cy + sy)]
            if any(abs(c.x - bx) <= 3 and abs(c.y - by) <= 3 for bx, by in corners_of_block):
                hit += 1
                break
    assert hit >= 5


def test_match_recovers_known_translation():
    scene = _scene()
    dx, dy = 6, 4
    img1, img2 = _shift_views(scene, dx, dy)
    ca, da = detect_and_describe(img1, max_corners=200, min_distance=6, patch=8)
    cb, db = detect_and_describe(img2, max_corners=200, min_distance=6, patch=8)
    assert len(ca) > 0 and len(cb) > 0
    pairs = match_descriptors(da, db, ratio=0.8, mutual=True)
    assert len(pairs) >= 5
    disp = np.array([[cb[j].x - ca[i].x, cb[j].y - ca[i].y] for i, j in pairs])
    # 真实对应应几乎全部落在 (-dx, -dy)（整数平移，误差 ≤1）
    good = np.sum((np.abs(disp[:, 0] + dx) <= 1) & (np.abs(disp[:, 1] + dy) <= 1))
    assert good >= 0.8 * len(pairs)


def test_match_rejects_ambiguous_and_empty():
    # 三行完全相同的描述子 → 次近=最近（皆 0），比值测试应全拒绝（歧义）
    d = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (3, 1))
    assert match_descriptors(d, d, ratio=0.8) == []
    # 空输入安全
    assert match_descriptors(np.zeros((0, 4)), np.zeros((5, 4))) == []
    assert describe(np.zeros((10, 10)), []) .shape[0] == 0


def test_self_match_is_stable():
    img = _scene()[0:120, 0:120]
    c, d = detect_and_describe(img, max_corners=100, min_distance=6)
    pairs = match_descriptors(d, d, ratio=0.8, mutual=True)
    # 自匹配：每个有效描述子应匹配到自身
    assert all(i == j for i, j in pairs)
    assert len(pairs) == len(d)


def test_to_gray_accepts_array_and_rejects_bad():
    a = to_gray(np.arange(12, dtype=float).reshape(3, 4))
    assert a.shape == (3, 4)
    rgb = np.zeros((5, 5, 3))
    assert to_gray(rgb).shape == (5, 5)
    import pytest
    with pytest.raises(ValueError):
        to_gray("no/such/image.png")


def test_corner_dataclass_immutable():
    c = Corner(x=1, y=2, response=3.0)
    assert (c.x, c.y, c.response) == (1, 2, 3.0)
    import pytest
    with pytest.raises(Exception):
        c.x = 9  # frozen
