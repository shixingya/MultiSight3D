"""自研 MVS 深度图基元 + 融合单测（纯合成渲染，零二进制入库）。

渲染一张位于已知深度的前平行带纹理平面，用两台纯横向平移的邻视图观测，验证
plane_sweep_depth 能把深度恢复到真值附近；再验证 depth_map_to_points 反投影的
点云 z 贴近真值；并验证无纹理场景诚实降级（几乎不产出有效点）、无邻视图报错。
"""

from __future__ import annotations

import numpy as np
import pytest

from multisight.mvs import plane_sweep_depth, depth_map_to_points


def _K(W=80, H=64, focal=120.0):
    return np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])


def _proj(K, R, t, X):
    Xc = X @ np.asarray(R).T + np.asarray(t)
    x = K @ Xc.T
    return (x[:2] / x[2:3]).T


def _render(shape, pts, patches):
    h, w = shape
    img = np.full((h, w), 120.0)
    ps = patches[0].shape[0]
    r = ps // 2
    for (u, v), pat in zip(pts, patches):
        ui, vi = int(round(u)), int(round(v))
        if ui - r < 0 or vi - r < 0 or ui + r + 1 > w or vi + r + 1 > h:
            continue
        img[vi - r:vi + r + 1, ui - r:ui + r + 1] = pat
    return img


def _plane_scene(d0=5.0, seed=3, W=80, H=64, patch_px=5):
    """世界系 = 参考相机系（ref 位姿恒等）。平面点 z=d0，横移邻视图纯平移。

    纹理点间距足够大（投影间距 ≫ patch 尺寸）以避免相邻 patch 重叠涂抹，保证 NCC 判深可辨。
    """
    rng = np.random.default_rng(seed)
    K = _K(W, H)
    xs = np.arange(-1.35, 1.36, 0.45)
    ys = np.arange(-1.05, 1.06, 0.45)
    gx, gy = np.meshgrid(xs, ys)
    X = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, d0)])
    Rr, tr = np.eye(3), np.zeros(3)
    src_pose = [(np.eye(3), np.array([0.3, 0.0, 0.0])),
                (np.eye(3), np.array([-0.3, 0.0, 0.0]))]
    patches = [20 + 215 * rng.random((patch_px, patch_px)) for _ in range(len(X))]
    ref = _render((H, W), _proj(K, Rr, tr, X), patches)
    srcs = [_render((H, W), _proj(K, R, t, X), patches) for R, t in src_pose]
    return K, ref, srcs, src_pose, (Rr, tr), X, d0


def test_plane_sweep_recovers_known_depth():
    K, ref, srcs, src_pose, ref_rt, _X, d0 = _plane_scene()
    depths = np.linspace(d0 - 0.5, d0 + 0.5, 41)        # 0.025 间隔
    dmap, cost, conf, grid = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, depths, patch=2, stride=3, max_cost=0.6)
    assert dmap.shape == cost.shape == conf.shape
    valid = np.isfinite(dmap)
    assert valid.mean() > 0.4                           # 多数带纹理像素应被恢复
    err = np.abs(dmap[valid] - d0)
    assert np.median(err) <= 0.2                        # 恢复到真值附近（合成渲染取整主导残余偏差）
    # 亚像素：相当比例的深度应落在采样网格之间（连续值）而非均为网格点
    step = depths[1] - depths[0]
    off_grid = np.abs((dmap[valid] - d0) / step - np.round((dmap[valid] - d0) / step)) > 1e-4
    assert off_grid.mean() > 0.3


def test_depth_map_to_points_backprojects_to_plane():
    K, ref, srcs, src_pose, ref_rt, _X, d0 = _plane_scene()
    depths = np.linspace(d0 - 0.5, d0 + 0.5, 41)
    dmap, cost, conf, (us, vs) = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, depths, patch=2, stride=3, max_cost=0.6)
    pts, mask = depth_map_to_points(us, vs, dmap, K, ref_rt, conf=conf, min_conf=1, max_cost=0.6)
    assert mask.any()
    assert len(pts) == int(mask.sum())
    # 世界系 = 参考相机系 → 反投影点 z 即深度，应贴近 d0
    z = pts[:, 2]
    assert np.median(np.abs(z - d0)) <= 0.2


def test_textureless_scene_degrades_honestly():
    K, ref, srcs, src_pose, ref_rt, _X, d0 = _plane_scene()
    flat = np.full_like(ref, 128.0)                     # 无纹理：NCC 退化
    depths = np.linspace(d0 - 0.5, d0 + 0.5, 41)
    dmap, cost, conf, _grid = plane_sweep_depth(
        flat, srcs, K, ref_rt, src_pose, depths, patch=2, stride=3, max_cost=0.02)
    assert not np.isfinite(dmap).any()                  # 极严代价阈下不臆造深度


def test_plane_sweep_requires_source_view():
    K, ref, _srcs, _pose, ref_rt, _X, d0 = _plane_scene()
    with pytest.raises(ValueError):
        plane_sweep_depth(ref, [], K, ref_rt, [], np.array([d0]))


def test_out_of_range_depth_rejected_at_boundary():
    K, ref, srcs, src_pose, ref_rt, _X, d0 = _plane_scene()
    far = np.linspace(d0 + 1.0, d0 + 1.5, 12)          # 区间完全不包含真值
    # 真值在区间之外：最优深度绝大多数贴向边界，开启剔除后有效像素应大幅下降
    dmap_rej, _c, _cf, _g = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, far, patch=2, stride=3, max_cost=2.0,
        reject_boundaries=True)
    dmap_keep, _c, _cf, _g = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, far, patch=2, stride=3, max_cost=2.0,
        reject_boundaries=False)
    n_rej = int(np.isfinite(dmap_rej).sum())
    n_keep = int(np.isfinite(dmap_keep).sum())
    assert n_keep > 0                                    # 关闭剔除时仍保留（但深度不可靠）
    assert n_rej < 0.25 * n_keep                         # 边界剔除掉绝大多数越界伪深度
    # 对照：区间包含真值时，边界剔除几乎不影响有效像素与精度
    near = np.linspace(d0 - 0.5, d0 + 0.5, 41)
    dm_rej, _c, _cf, _g = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, near, patch=2, stride=3, max_cost=0.6,
        reject_boundaries=True)
    dm_keep, _c, _cf, _g = plane_sweep_depth(
        ref, srcs, K, ref_rt, src_pose, near, patch=2, stride=3, max_cost=0.6,
        reject_boundaries=False)
    assert abs(int(np.isfinite(dm_rej).sum()) - int(np.isfinite(dm_keep).sum())) <= 5
    assert np.median(np.abs(dm_rej[np.isfinite(dm_rej)] - d0)) <= 0.2
