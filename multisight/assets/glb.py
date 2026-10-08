"""泛化 GLB（glTF 2.0 单文件）写出器：任意顶点 + 索引 + 可选 UV + 可选内嵌贴图。

延续仓库「手写合法二进制、零第三方依赖」的风格（见 stages/_synth.py），
把 assets.openflight.Mesh 之类的几何序列化成 three.js GLTFLoader 可直接加载的 .glb。
坐标系约定：OpenFlight 为 Z-up，three.js 为 Y-up，导出时做 (x,y,z)->(x,z,-y) 轴变换。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _axis_to_yup(positions: np.ndarray) -> np.ndarray:
    out = np.empty_like(positions)
    out[:, 0] = positions[:, 0]
    out[:, 1] = positions[:, 2]      # three.y = flt.z (up)
    out[:, 2] = -positions[:, 1]     # three.z = -flt.y (forward)
    return out


def write_glb(path: Path | str, positions, indices, *,
              uvs=None, texture_png: Path | str | None = None,
              name: str = "asset", z_up: bool = True) -> Path:
    """写 .glb。positions:(N,3) float；indices:(M,) int；uvs:(N,2) 可选。"""
    pos = np.asarray(positions, dtype="<f4")
    if z_up and len(pos):
        pos = np.ascontiguousarray(_axis_to_yup(np.asarray(positions, dtype=np.float64)).astype("<f4"))
    idx = np.asarray(indices, dtype="<u2" if len(pos) < 65536 else "<u4")

    bin_chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    cursor = 0

    def _add_view(data: bytes, target: int) -> int:
        nonlocal cursor
        pad = (-len(data)) % 4
        data = data + b"\x00" * pad
        buffer_views.append({"buffer": 0, "byteOffset": cursor,
                             "byteLength": len(data) - pad, "target": target})
        bin_chunks.append(data)
        cursor += len(data)
        return len(buffer_views) - 1

    pos_view = _add_view(pos.tobytes(), 34962)
    accessors.append({"bufferView": pos_view, "componentType": 5126, "count": len(pos),
                      "type": "VEC3",
                      "min": pos.min(axis=0).tolist() if len(pos) else [0, 0, 0],
                      "max": pos.max(axis=0).tolist() if len(pos) else [0, 0, 0]})
    idx_view = _add_view(idx.tobytes(), 34963)
    comp = 5123 if idx.dtype == np.uint16 else 5125
    accessors.append({"bufferView": idx_view, "componentType": comp, "count": len(idx),
                      "type": "SCALAR"})

    attributes = {"POSITION": 0}
    if uvs is not None and len(uvs):
        uv = np.asarray(uvs, dtype="<f4")
        uv_view = _add_view(uv.tobytes(), 34962)
        accessors.append({"bufferView": uv_view, "componentType": 5126,
                          "count": len(uv), "type": "VEC2"})
        attributes["TEXCOORD_0"] = len(accessors) - 1

    images: list[dict] = []
    textures: list[dict] = []
    material: dict = {"name": name, "pbrMetallicRoughness":
                      {"baseColorFactor": [0.8, 0.8, 0.85, 1.0],
                       "metallicFactor": 0.0, "roughnessFactor": 0.8}}
    if texture_png and Path(texture_png).is_file():
        blob = Path(texture_png).read_bytes()
        tex_view = _add_view(blob, 0)
        buffer_views[-1].pop("target", None)
        images.append({"bufferView": tex_view, "mimeType": "image/png"})
        textures.append({"source": 0, "sampler": 0})
        material = {"name": name, "pbrMetallicRoughness":
                    {"baseColorTexture": {"index": 0},
                     "metallicFactor": 0.0, "roughnessFactor": 0.9}}

    gltf = {
        "asset": {"version": "2.0", "generator": "multisight3d-asset"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": 1,
                                    "material": 0}]}],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": cursor}],
    }
    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_data = b"".join(bin_chunks)
    bin_data += b"\x00" * ((-len(bin_data)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
    )
    return out
