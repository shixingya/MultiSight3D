"""深度图→三角网格基元单测（v0.3 起步，纯合成网格，零外部数据）。

验证 depth_map_to_mesh：满格平面网格的顶点/面数与反投影一致性、无效角点留洞、
max_edge_gap 抑制深度断裂假面、全无效诚实返回空、形状不符报错。世界系=参考相机系时
顶点 z 应等于深度真值，且与 fuse.depth_map_to_points 的点集完全一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from multisight.mvs import analyze_mesh_topology, depth_map_to_mesh, depth_map_to_points, mesh_metrics, weld_mesh_vertices

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


def test_topology_tetrahedron_watertight():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], np.int64)
    t = analyze_mesh_topology(v, f)
    assert t["is_watertight"] is True
    assert t["boundary_edges"] == 0 and t["non_manifold_edges"] == 0
    assert t["edges"] == 6 and t["faces"] == 4
    assert t["euler_characteristic"] == 2                 # 封闭可定向亏格 0
    assert t["unused_vertices"] == 0


def test_topology_single_triangle_open():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2]], np.int64)
    t = analyze_mesh_topology(v, f)
    assert t["is_watertight"] is False
    assert t["boundary_edges"] == 3                       # 三条边都只属一个面
    assert t["euler_characteristic"] == 1                 # 3-3+1


def test_topology_non_manifold_edge():
    # 三个三角形共用边 (0,1) → 非流形边
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 1]], float)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]], np.int64)
    t = analyze_mesh_topology(v, f)
    assert t["non_manifold_edges"] >= 1
    assert t["is_watertight"] is False


def test_topology_unused_vertex():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]], float)
    f = np.array([[0, 1, 2]], np.int64)
    t = analyze_mesh_topology(v, f)
    assert t["unused_vertices"] == 1                      # 第 4 个顶点未被引用


def test_topology_empty_faces():
    v = np.zeros((5, 3))
    t = analyze_mesh_topology(v, np.zeros((0, 3), np.int64))
    assert t["faces"] == 0 and t["is_watertight"] is False
    assert t["unused_vertices"] == 5


def test_topology_planar_grid_is_open():
    us, vs = _grid()
    depth = np.full(us.shape, 4.0)
    verts, faces = depth_map_to_mesh(us, vs, depth, _K(), (np.eye(3), np.zeros(3)))
    t = analyze_mesh_topology(verts, faces)
    assert t["boundary_edges"] > 0                        # 平面网格边缘开洞
    assert t["is_watertight"] is False


def test_weld_merges_duplicate_vertices():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0.001, 0, 0], [1.001, 0, 0], [0, 0.999, 0]], float)
    f = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    nv, nf, nc = weld_mesh_vertices(v, f, tol=0.01)
    assert len(nv) == 3                                   # 两组重合顶点各归一
    assert len(nf) == 2 and len(set(nf[0])) == 3           # 两面都非退化
    assert np.array_equal(nf[0], nf[1])                    # 两组同拓扑→焊后重合为同一三角
    assert nc is None


def test_weld_averages_colors():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.002, 0, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]], np.int64)       # 面 1 用重合顶点 3
    c = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)
    nv, nf, nc = weld_mesh_vertices(v, f, c, tol=0.01)
    assert len(nv) == 3                                   # 顶点 0 与 3 焊接
    assert nc.dtype == np.uint8
    # 焊接后代表色为同组顶点色均值（红+红 → 仍为红）
    assert [0.001, 0.0, 0.0] == pytest.approx(list(nv[0]), abs=1e-3)
    assert list(nc[0]) == [255, 0, 0]


def test_weld_drops_degenerate_faces():
    # 同一面内两顶点重合（tol 内）→ 焊接后退化→丢弃
    v = np.array([[0, 0, 0], [0.001, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [2, 3, 0]], np.int64)        # 面0退化，面1保留
    nv, nf, _ = weld_mesh_vertices(v, f, tol=0.01)
    assert len(nv) == 3
    assert len(nf) == 1
    assert (nf[0] == np.array([2, 1, 0])).all() or len(set(nf[0])) == 3


def test_weld_nonpositive_tol_noop():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2]], np.int64)
    nv, nf, _ = weld_mesh_vertices(v, f, tol=0.0)
    assert np.array_equal(nv, v) and np.array_equal(nf, f)


def test_weld_empty_noop():
    nv, nf, nc = weld_mesh_vertices(np.zeros((0, 3)), np.zeros((0, 3), np.int64), tol=0.1)
    assert len(nv) == 0 and len(nf) == 0 and nc is None


# 单位立方体 [0,1]^3，8 顶点 12 三角面，外定向一致（与 analyze_mesh_topology 共用）
_CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_CUBE_F = np.array([[4, 5, 6], [4, 6, 7],          # top +z
                    [0, 3, 2], [0, 2, 1],          # bottom -z
                    [0, 1, 5], [0, 5, 4],          # front -y
                    [3, 7, 6], [3, 6, 2],          # back +y
                    [0, 7, 3], [0, 4, 7],          # left -x
                    [1, 6, 5], [1, 2, 6]], np.int64)   # right +x


def test_mesh_metrics_unit_cube_area_volume():
    m = mesh_metrics(_CUBE_V, _CUBE_F)
    assert m["area"] == pytest.approx(6.0, abs=1e-6)        # 6 个单位正方形
    assert m["volume_signed"] == pytest.approx(1.0, abs=1e-6)  # 外定向→正
    assert m["volume_abs"] == pytest.approx(1.0, abs=1e-6)
    assert m["bbox_extent"] == [1.0, 1.0, 1.0]
    # 与水密拓扑一致（同一闭合立方体）
    assert analyze_mesh_topology(_CUBE_V, _CUBE_F)["is_watertight"]


def test_mesh_metrics_open_face_volume_means_surface():
    # 单三角形（非闭合）：面积正确，有向体积仅为相对于原点的四面体代数量（非真实体积）
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2]], np.int64)
    m = mesh_metrics(v, f)
    assert m["area"] == pytest.approx(0.5, abs=1e-6)
    assert m["volume_signed"] == pytest.approx(0.0, abs=1e-6)  # 过原点面→0


def test_mesh_metrics_empty_and_vertices_only():
    m = mesh_metrics(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    assert m["area"] == 0.0 and m["volume_abs"] == 0.0
    assert m["bbox_extent"] == [0.0, 0.0, 0.0]
    # 有顶点无面：bbox 仍给，面积/体积诚实为 0
    v = np.array([[0, 0, 0], [2, 3, 4]], float)
    m2 = mesh_metrics(v, np.zeros((0, 3), np.int64))
    assert m2["area"] == 0.0 and m2["bbox_extent"] == [2.0, 3.0, 4.0]
