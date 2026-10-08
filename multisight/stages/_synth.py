"""Mock 管线的程序化合成产物：零二进制资产入库（延续 Butian3D 理念）。

提供 PLY 点云、OBJ 立方网格、GLB 单文件、程序化贴图的极简写出器，
让 mock 引擎产出真实可打开的模型文件，供 WebUI three.js 预览与下载链路验证。
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------- 点云 ----------

def write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """写 ASCII PLY（xyz [+ rgb]）。points: (N,3) float；colors: (N,3) uint8。"""
    n = len(points)
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    for i in range(n):
        xyz = " ".join(f"{v:.5f}" for v in points[i])
        if colors is not None:
            xyz += " " + " ".join(str(int(c)) for c in colors[i])
        lines.append(xyz)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def sphere_points(n: int, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius * (0.9 + 0.1 * rng.random((n, 1)))


# ---------- 网格 ----------

_CUBE_V = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
           (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
_CUBE_F = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
           (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]


def write_obj_cube(path: Path) -> None:
    lines = ["# MultiSight3D mock mesh (placeholder cube)", "o object"]
    for v in _CUBE_V:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for f in _CUBE_F:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------- GLB（glTF 2.0 二进制单文件） ----------

def write_glb_cube(path: Path) -> None:
    """生成含一个立方体的合法 .glb（positions+indices，默认材质）。

    无内嵌法线/UV：查看器加载后由 JS computeVertexNormals 补光。
    真实纹理烘焙产物（v0.3）将替换本函数输出。
    """
    positions = np.array(_CUBE_V, dtype="<f4")
    indices = np.array([i for tri in _CUBE_F for i in tri], dtype="<u2")
    bin_data = positions.tobytes() + indices.tobytes()
    minv = positions.min(axis=0).tolist()
    maxv = positions.max(axis=0).tolist()
    gltf = {
        "asset": {"version": "2.0", "generator": "multisight3d-mock"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mock_cube"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "materials": [{"name": "default"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": minv, "max": maxv},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": indices.nbytes, "target": 34963},
        ],
        "buffers": [{"byteLength": len(bin_data)}],
    }
    json_data = json.dumps(gltf, separators=(",", ":")).encode("ascii")
    json_data += b" " * (-len(json_data) % 4)          # JSON 块以空格补齐 4 字节
    pad = (-len(bin_data) % 4)
    bin_data += b"\x00" * pad                          # BIN 块以零补齐
    total = 12 + 8 + len(json_data) + 8 + len(bin_data)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)      # magic 'glTF', version, length
        + struct.pack("<II", len(json_data), 0x4E4F534A) + json_data   # JSON chunk
        + struct.pack("<II", len(bin_data), 0x004E4942) + bin_data     # BIN chunk
    )


# ---------- 程序化贴图 ----------

def write_checker_texture(path: Path, size: int = 512, cells: int = 8) -> None:
    """双色棋盘 + 中心标记，验证纹理文件下载/预览链路。"""
    img = Image.new("RGB", (size, size), (236, 230, 220))
    d = ImageDraw.Draw(img)
    step = size // cells
    for y in range(cells):
        for x in range(cells):
            if (x + y) % 2:
                d.rectangle([x * step, y * step, (x + 1) * step - 1, (y + 1) * step - 1],
                            fill=(58, 90, 122))
    d.ellipse([size / 2 - 30, size / 2 - 30, size / 2 + 30, size / 2 + 30], outline=(198, 96, 60), width=6)
    img.save(path)


# ---------- COLMAP 兼容目录（SfM 中间交换格式） ----------

def circle_pose(index: int, total: int, radius: float = 2.2) -> tuple[tuple, tuple]:
    """环绕轨道位姿：返回 (四元数 wxyz, 平移 tx ty tz)，相机朝向原点。"""
    yaw = 2 * math.pi * index / max(total, 1)
    # 绕 Y 轴旋转 + 俯仰 -10°（组合成四元数）
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    pitch = math.radians(-10 / 2)
    cp, sp = math.cos(pitch), math.sin(pitch)
    q = (cp * cy, 0.0, sp * cy, cp * sy)               # w, x, y, z
    t = (-radius * math.sin(yaw), -radius * math.sin(pitch) * 2, -radius * math.cos(yaw))
    return q, t


def write_cameras_txt(path: Path, count: int, width: int, height: int) -> None:
    """SIMPLE_PINHOLE 模型：COLMAP cameras.txt 风格（自研 SfM 的最终落盘格式）。"""
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for i in range(count):
        lines.append(f"{i + 1} SIMPLE_PINHOLE {width} {height} 1000 800 600")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_images_txt(path: Path, count: int) -> None:
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for i in range(count):
        (w, x, y, z), (tx, ty, tz) = circle_pose(i, count)
        lines.append(f"{i + 1} {w:.6f} {x:.6f} {y:.6f} {z:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {i + 1} img{i:04d}.jpg")
        lines.append("-1")                             # 2D 观测点占位（mock 无观测）
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
