"""stages/texture.Real 接线单测：读 mesh.obj → 逐顶点真彩色 model.glb（真实一步）。

手工构造带/不带顶点色的 mesh.obj 喂给 texture.Real，验证产出合法 GLB（magic=glTF、
含 COLOR_0）、stats 记录三角形数 > 0、artifact_ready("texture")；缺网格产物时诚实报错。
"""

from __future__ import annotations

import json
import struct

import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.texture import Real as TextureReal
from multisight.workspace import Workspace

pytest.importorskip("numpy")


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def _write_obj(ws, color=True):
    lines = ["# test mesh", "o m",
             "v 0 0 0 1 0 0", "v 1 0 0 0 1 0", "v 1 1 0 0 0 1", "v 0 1 0 1 1 0"]
    if not color:
        lines = ["# test mesh", "o m",
                 "v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0"]
    lines += ["f 1 2 3", "f 1 3 4"]
    ws.file("mesh", "mesh.obj").write_text("\n".join(lines) + "\n", encoding="ascii")


def _gltf_json(data: bytes) -> dict:
    json_len = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + json_len].decode("utf-8").rstrip())


def test_real_texture_glb_with_color(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    _write_obj(ws, color=True)
    TextureReal().run(_ctx(ws))

    assert ws.artifact_ready("texture")
    stats = json.loads((ws.root / "texture" / "stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "real/vertex-color-glb"
    assert stats["vertices"] == 4 and stats["triangles"] == 2
    assert stats["vertex_colors"] is True

    data = (ws.root / "texture" / "model.glb").read_bytes()
    assert data[:4] == b"glTF" and int.from_bytes(data[4:8], "little") == 2
    gltf = _gltf_json(data)
    assert "COLOR_0" in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_real_texture_glb_without_color(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    _write_obj(ws, color=False)
    TextureReal().run(_ctx(ws))
    stats = json.loads((ws.root / "texture" / "stats.json").read_text(encoding="utf-8"))
    assert stats["vertex_colors"] is False
    assert (ws.root / "texture" / "model.glb").read_bytes()[:4] == b"glTF"


def test_real_texture_requires_mesh(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    with pytest.raises(ValueError, match="缺少网格产物"):
        TextureReal().run(_ctx(ws))
