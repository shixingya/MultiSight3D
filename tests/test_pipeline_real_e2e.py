"""真实管线端到端回归：sfm.Real → mvs.Real → mesh.Real → texture.Real 全链跑通。

用受控合成多视图场景（零二进制入库）驱动自研传统四阶段，断言最终产出合法 GLB
（magic=glTF、v2）、各阶段产物就绪、稠密点/网格面/三角面数均 > 0。锁死 v0.1→v0.3
真实链路不回归（Mesh 起步为逐视图距离图像网格化，稠密点数随合成场景规模而小属正常）。
"""

from __future__ import annotations

import json

import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.mesh import Real as MeshReal
from multisight.stages.mvs import Real as MvsReal
from multisight.stages.sfm import Real as SfmReal
from multisight.stages.texture import Real as TextureReal

pytest.importorskip("numpy")

from tests.test_sfm_pipeline import _synthetic_scene  # noqa: E402
from tests.test_sfm_stage import _make_ws             # noqa: E402


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def test_real_pipeline_end_to_end_glb(tmp_path):
    frames, _X, _pose = _synthetic_scene(n_frames=4)
    ws = _make_ws(tmp_path, frames)

    SfmReal().run(_ctx(ws))
    assert ws.artifact_ready("sfm")
    MvsReal().run(_ctx(ws))
    assert ws.artifact_ready("mvs")
    MeshReal().run(_ctx(ws))
    assert ws.artifact_ready("mesh")
    TextureReal().run(_ctx(ws))
    assert ws.artifact_ready("texture")

    mvs = json.loads((ws.root / "mvs" / "stats.json").read_text(encoding="utf-8"))
    mesh = json.loads((ws.root / "mesh" / "stats.json").read_text(encoding="utf-8"))
    tex = json.loads((ws.root / "texture" / "stats.json").read_text(encoding="utf-8"))
    assert mvs["dense_points"] > 0
    assert mesh["faces"] > 0
    assert tex["triangles"] == mesh["faces"]
    assert tex["vertex_colors"] is True

    # FR-08 PLY 网格产物可回读，顶点/面数与 stats 一致
    from multisight.stages._synth import read_ply_mesh
    pv, pf, pc = read_ply_mesh(ws.root / "mesh" / "mesh.ply")
    assert len(pv) == mesh["vertices"] and len(pf) == mesh["faces"]
    assert pc is not None and pc.shape == (mesh["vertices"], 3)

    glb = (ws.root / "texture" / "model.glb").read_bytes()
    assert glb[:4] == b"glTF" and int.from_bytes(glb[4:8], "little") == 2
