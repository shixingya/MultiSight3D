"""stages/mesh.Real 接线单测：读 MVS 深度图 npz → 距离图像网格化 → mesh.obj（真实一步）。

用手工构造的稠密平面深度图（npz）喂给 mesh.Real，验证产出 mvs→mesh 链路：mesh/mesh.obj
就绪、顶点/面数与网格化预期一致、stats 记录面数 > 0；缺深度图产物时诚实报错（不静默假成功、
不回退占位立方）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.mesh import Real as MeshReal
from multisight.workspace import Workspace

pytest.importorskip("numpy")


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def _write_plane_npz(ws, hs=8, ws_n=10, d0=5.0, hole=False):
    vs, us = np.mgrid[0:hs, 0:ws_n]
    depth = np.full((hs, ws_n), d0, dtype=np.float64)
    if hole:
        depth[hs // 2, ws_n // 2] = np.nan
    K = np.array([[100.0, 0, ws_n / 2.0], [0, 100.0, hs / 2.0], [0, 0, 1.0]])
    np.savez(ws.file("mvs", "depthmaps", "view000.npz"), name="a.jpg", K=K,
             R=np.eye(3), t=np.zeros(3), us=us.astype(np.float64),
             vs=vs.astype(np.float64), depth=depth)
    return hs, ws_n


def test_real_mesh_from_depthmap(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    hs, ws_n = _write_plane_npz(ws)
    MeshReal().run(_ctx(ws))

    assert ws.artifact_ready("mesh")
    stats = json.loads((ws.root / "mesh" / "stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "real/range-image-mesh"
    assert stats["faces"] == 2 * (hs - 1) * (ws_n - 1)
    assert stats["vertices"] == hs * ws_n

    lines = (ws.root / "mesh" / "mesh.obj").read_text(encoding="ascii").splitlines()
    assert sum(ln.startswith("v ") for ln in lines) == hs * ws_n
    assert sum(ln.startswith("f ") for ln in lines) == stats["faces"]


def test_real_mesh_hole_reduces_faces(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    hs, ws_n = _write_plane_npz(ws, hole=True)
    MeshReal().run(_ctx(ws))
    stats = json.loads((ws.root / "mesh" / "stats.json").read_text(encoding="utf-8"))
    assert stats["vertices"] == hs * ws_n - 1            # 洞角点被剔除
    assert stats["faces"] < 2 * (hs - 1) * (ws_n - 1)    # 相邻格退化


def test_real_mesh_requires_depthmaps(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    with pytest.raises(ValueError, match="缺少 MVS 深度图"):
        MeshReal().run(_ctx(ws))
