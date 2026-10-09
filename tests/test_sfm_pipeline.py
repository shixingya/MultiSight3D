"""两视图真实 SfM 编排端到端单测（纯合成渲染图，零二进制入库）。

渲染一个由 3D 点投影到两相机、带真实视差的标志场景，验证检测→匹配→F→位姿→三角化
整条链跑通；并验证不相关图像时诚实降级（点不足即返回，不臆造、不崩溃）。
"""

from __future__ import annotations

import numpy as np

from multisight.sfm import (ReconstructionResult, TwoViewResult, candidate_pairs,
                            incremental_reconstruction, select_best_pair,
                            two_view_reconstruction, default_K)


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(P, X):
    x = P @ np.hstack([X, np.ones((len(X), 1))]).T
    return (x[:2] / x[2:3]).T


def _render(shape, pts, patches):
    h, w = shape
    img = np.full((h, w), 128.0)
    ps = patches[0].shape[0]
    r = ps // 2
    for (u, v), pat in zip(pts, patches):
        ui, vi = int(round(u)), int(round(v))
        if ui - r < 0 or vi - r < 0 or ui + r + 1 > w or vi + r + 1 > h:
            continue
        img[vi - r:vi + r + 1, ui - r:ui + r + 1] = pat
    return img


def _synthetic_pair(n=90, seed=5, patch_px=9):
    rng = np.random.default_rng(seed)
    W, H = 360, 270
    focal = 500.0
    K = default_K((H, W), focal)
    X = np.column_stack([rng.uniform(-1.1, 1.1, n), rng.uniform(-0.8, 0.8, n),
                         rng.uniform(3.5, 5.5, n)])
    R = _rot("y", np.radians(9)) @ _rot("x", np.radians(4))
    t = np.array([0.5, 0.03, 0.06])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t[:, None]])
    p1, p2 = _project(P1, X), _project(P2, X)
    m = patch_px // 2 + 6
    keep = ((p1[:, 0] > m) & (p1[:, 0] < W - m) & (p1[:, 1] > m) & (p1[:, 1] < H - m) &
            (p2[:, 0] > m) & (p2[:, 0] < W - m) & (p2[:, 1] > m) & (p2[:, 1] < H - m))
    p1, p2 = p1[keep], p2[keep]
    k = len(p1)
    # 每个可见点一块唯一随机纹理 patch（两视图同款→强匹配；互异→可区分）
    patches = [20 + 215 * rng.random((patch_px, patch_px)) for _ in range(k)]
    img_a = _render((H, W), p1, patches)
    img_b = _render((H, W), p2, patches)
    return img_a, img_b, R, t / np.linalg.norm(t)


def test_two_view_reconstruction_runs_and_triangulates():
    img_a, img_b, R_true, t_hat = _synthetic_pair()
    res = two_view_reconstruction(img_a, img_b, focal=500.0, max_corners=600,
                                  ransac_threshold=2.0, seed=0)
    assert isinstance(res, TwoViewResult)
    assert res.num_matches >= 8
    assert res.num_inliers >= 8
    assert res.ok and res.registered
    assert len(res.points3d) >= 8
    assert res.median_reproj_px < 2.0                    # 内点极线残差应很小（几何拟合好）
    # 位姿结构合法（精确位姿已在 test_sfm_geometry 用精确投影点验证）
    assert res.R.shape == (3, 3)
    assert np.allclose(res.R @ res.R.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(res.R) - 1.0) < 1e-6
    assert abs(np.linalg.norm(res.t) - 1.0) < 1e-6
    # 三角化点均位于两相机前方（z>0）
    assert np.all(res.points3d[:, 2] > 0)


def test_uncorrelated_images_degrade_honestly():
    rng = np.random.default_rng(0)
    a = rng.random((200, 200)) * 255
    b = rng.random((200, 200)) * 255
    res = two_view_reconstruction(a, b, focal=200.0, seed=0)  # 不应抛异常
    assert isinstance(res, TwoViewResult)
    # 随机噪声要么匹配不足 8，要么无法给出可靠位姿（ok 为假或内点很少）
    if res.num_matches < 8:
        assert not res.ok
    else:
        assert res.num_inliers < 8 or not res.registered


def test_candidate_pairs_is_linear_and_deterministic():
    assert candidate_pairs(2) == [(0, 1)]
    assert candidate_pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]


def test_select_best_pair_picks_the_overlapping_pair():
    # 诱饵：首帧为无关噪声；真正的重叠对在 (1, 2)，应被择优选中
    img_a, img_b, _, _ = _synthetic_pair()
    rng = np.random.default_rng(0)
    decoy = rng.random(img_a.shape) * 255
    res, (i, j) = select_best_pair([decoy, img_a, img_b], focal=500.0,
                                   max_corners=600, ransac_threshold=2.0, seed=0)
    assert (i, j) == (1, 2)
    assert res.registered and res.num_inliers >= 8


def _synthetic_scene(n_frames=4, n_pts=90, seed=7, W=420, H=300, focal=500.0, patch_px=9):
    """同一批 3D 点（唯一纹理 patch）由具充足基线的相机序列观测，返回帧列表。

    基线足够大（避免小基线三角化在尺度放大下产生巨大深度误差，那是 SfM 的经典病态）。
    """
    rng = np.random.default_rng(seed)
    K = default_K((H, W), focal)
    X = np.column_stack([rng.uniform(-1.2, 1.2, n_pts), rng.uniform(-0.9, 0.9, n_pts),
                         rng.uniform(4.5, 6.5, n_pts)])
    frames_pose = []
    for k in range(n_frames):
        Rk = _rot("y", np.radians(4.0 * k))
        tk = np.array([0.35 * k, 0.03 * k, 0.0])
        frames_pose.append((Rk, tk))
    Ps = [K @ np.hstack([Rk, tk[:, None]]) for Rk, tk in frames_pose]
    projs = [_project(P, X) for P in Ps]
    m = patch_px // 2 + 6
    keep = np.ones(n_pts, dtype=bool)
    for pxy in projs:
        keep &= ((pxy[:, 0] > m) & (pxy[:, 0] < W - m) & (pxy[:, 1] > m) & (pxy[:, 1] < H - m))
    idx = np.where(keep)[0]
    Xv = X[idx]
    projs = [pxy[idx] for pxy in projs]
    patches = [20 + 215 * rng.random((patch_px, patch_px)) for _ in range(len(idx))]
    frames = [_render((H, W), pxy, patches) for pxy in projs]
    return frames, Xv, frames_pose


def test_incremental_reconstruction_registers_multiple_views():
    frames, _X, _pose = _synthetic_scene(n_frames=4)
    res = incremental_reconstruction(frames, focal=500.0, max_corners=600,
                                     ransac_threshold=2.0, pnp_reproj_px=8.0,
                                     min_pnp_inliers=8, seed=0)
    assert isinstance(res, ReconstructionResult)
    assert res.ok
    assert res.num_registered >= 3                 # 至少初始对 + 多一帧
    assert len(res.points3d) >= 8
    for cam in res.cameras:
        assert cam.R.shape == (3, 3)
        assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(cam.R) - 1.0) < 1e-6
    # 初始帧位姿为恒等（世界系 = 初始帧 a）
    a = res.initial_pair[0]
    ident = next(c for c in res.cameras if c.index == a)
    assert np.allclose(ident.R, np.eye(3), atol=1e-9)


def test_incremental_reconstruction_needs_two_images():
    import pytest
    with pytest.raises(ValueError):
        incremental_reconstruction([np.zeros((50, 50))])
