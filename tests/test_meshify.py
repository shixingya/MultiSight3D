"""深度图→三角网格基元单测（v0.3 起步，纯合成网格，零外部数据）。

验证 depth_map_to_mesh：满格平面网格的顶点/面数与反投影一致性、无效角点留洞、
max_edge_gap 抑制深度断裂假面、全无效诚实返回空、形状不符报错。世界系=参考相机系时
顶点 z 应等于深度真值，且与 fuse.depth_map_to_points 的点集完全一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from multisight.mvs import depth_map_to_mesh, depth_map_to_points

pytest.importorskip("numpy")


def _K(focal=100.0, cx=3.0, cy=2.0):
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])


def _grid(hs=5, ws=6):
    vs, us = np.mgrid[0:hs, 0:ws]
    return us.astype(np.float64), vs.astype(np.float64)


def test_planar_grid_full_mesh():
    us, vs = _grid()
    hs, ws = us.shape
    depth = np.full((hs, ws), 4.0)
    K, RT = _K(), (np.eye(3), np.zeros(3))
    verts, faces = depth_map_to_mesh(us, vs, depth, K, RT)
    assert verts.shape == (hs * ws, 3)
    assert faces.shape[0] == 2 * (hs - 1) * (ws - 1)
    assert faces.dtype == np.int64
    assert faces.min() >= 0 and faces.max() < len(verts)
    assert np.allclose(verts[:, 2], 4.0)                 # 世界系=参考相机系 → z 即深度


def test_mesh_matches_backprojected_points():
    us, vs = _grid()
    depth = np.full(us.shape, 3.5)
    K, RT = _K(), (np.eye(3), np.zeros(3))
    verts, _f = depth_map_to_mesh(us, vs, depth, K, RT)
    pts, _m = depth_map_to_points(us, vs, depth, K, RT, min_conf=0)
    assert np.allclose(verts, pts)                        # 同一反投影，逐点一致


def test_hole_reduces_vertices_and_faces():
    us, vs = _grid()
    hs, ws = us.shape
    depth = np.full((hs, ws), 4.0)
    full_v, full_f = depth_map_to_mesh(us, vs, depth, K := _K(), (np.eye(3), np.zeros(3)))
    d2 = depth.copy()
    d2[2, 3] = np.nan                                     # 抠掉一个内部角点
    v2, f2 = depth_map_to_mesh(us, vs, d2, K, (np.eye(3), np.zeros(3)))
    assert len(v2) == hs * ws - 1
    assert len(f2) < len(full_f)                          # 相邻 4 格退化 → 面减少


def test_max_edge_gap_drops_discontinuity_faces():
    us, vs = _grid()
    depth = np.full(us.shape, 4.0)
    depth[:, 3:] = 60.0                                   # 右半远处 → 跨列边很长
    K, RT = _K(), (np.eye(3), np.zeros(3))
    _, f_full = depth_map_to_mesh(us, vs, depth, K, RT)
    _, f_gap = depth_map_to_mesh(us, vs, depth, K, RT, max_edge_gap=5.0)
    assert 0 < len(f_gap) < len(f_full)                   # 断口面被剔除，同侧仍保留


def test_all_invalid_returns_empty():
    us, vs = _grid()
    verts, faces = depth_map_to_mesh(us, vs, np.full(us.shape, np.nan), _K(), (np.eye(3), np.zeros(3)))
    assert len(verts) == 0 and len(faces) == 0


def test_shape_mismatch_raises():
    us, vs = _grid()
    depth = np.full(us.shape, 4.0)
    with pytest.raises(ValueError):
        depth_map_to_mesh(us, vs[:, :-1], depth, _K(), (np.eye(3), np.zeros(3)))
