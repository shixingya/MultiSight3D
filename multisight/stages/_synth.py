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


def write_obj_trimesh(path: Path, verts: np.ndarray, faces: np.ndarray, name: str = "mesh",
                      colors: np.ndarray | None = None) -> None:
    """写 OBJ 三角网格：verts (N,3) float，faces (M,3) int（0-based → 文件内 1-based）。
    colors (N,3) uint8 可选：逐顶点色，写为 `v x y z r g b`（r/g/b 归一 0~1，Blender 可识）。"""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    out = [f"# MultiSight3D mesh ({len(verts)} verts / {len(faces)} tris)", f"o {name}"]
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64) / 255.0
        out += [f"v {x:.5f} {y:.5f} {z:.5f} {r:.4f} {g:.4f} {b:.4f}"
                for (x, y, z), (r, g, b) in zip(verts, c)]
    else:
        out += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
    out += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_ply_mesh(path: Path, verts: np.ndarray, faces: np.ndarray,
                   colors: np.ndarray | None = None) -> None:
    """写 ASCII PLY 三角网格（FR-08：PLY 网格产物）。verts (N,3) float、faces (M,3) int（0-based）。
    colors (N,3) uint8 可选：逐顶点色写为 vertex 的 red/green/blue 属性。"""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n, m = len(verts), len(faces)
    lines = ["ply", "format ascii 1.0",
             f"element vertex {n}",
             "property float x", "property float y", "property float z"]
    if colors is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines += [f"element face {m}", "property list uchar int vertex_indices", "end_header"]
    c = np.asarray(colors) if colors is not None else None
    for i in range(n):
        row = f"{verts[i, 0]:.5f} {verts[i, 1]:.5f} {verts[i, 2]:.5f}"
        if c is not None:
            row += f" {int(c[i, 0])} {int(c[i, 1])} {int(c[i, 2])}"
        lines.append(row)
    for a, b, cc in faces:
        lines.append(f"3 {int(a)} {int(b)} {int(cc)}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """读 ASCII PLY 三角网格（与 write_ply_mesh 对称）：返回 (vertices (N,3), faces (M,3) 0-based,
    colors (N,3) uint8 或 None)。只支持本包写出的 vertex(+可选 red/green/blue) + face(3 索引) 形式。"""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    n_verts = n_faces = 0
    vprops: list[str] = []
    elem = ""
    header_end = None
    for i, ln in enumerate(lines):
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "element":
            elem = tok[1]
            if elem == "vertex":
                n_verts = int(tok[2])
            elif elem == "face":
                n_faces = int(tok[2])
        elif tok[0] == "property" and elem == "vertex" and tok[1] != "list":
            vprops.append(tok[2])
        elif tok[0] == "end_header":
            header_end = i
            break
    if header_end is None:
        raise ValueError("PLY 缺 end_header，非法网格")
    body = lines[header_end + 1:]
    has_color = "red" in vprops and "green" in vprops and "blue" in vprops
    xi, yi, zi = vprops.index("x"), vprops.index("y"), vprops.index("z")
    verts = np.zeros((n_verts, 3), dtype=np.float64)
    colors = np.zeros((n_verts, 3), dtype=np.uint8) if has_color else None
    for r in range(n_verts):
        vals = body[r].split()
        verts[r] = [float(vals[xi]), float(vals[yi]), float(vals[zi])]
        if has_color:
            colors[r] = [int(vals[vprops.index("red")]), int(vals[vprops.index("green")]),
                         int(vals[vprops.index("blue")])]
    faces = np.zeros((n_faces, 3), dtype=np.int64)
    for r in range(n_faces):
        vals = body[n_verts + r].split()
        assert int(vals[0]) == 3, "read_ply_mesh 仅支持三角面"
        faces[r] = [int(vals[1]), int(vals[2]), int(vals[3])]
    return verts, faces, colors


def read_obj_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """读 OBJ 三角网格：返回 (positions (N,3), indices (M,) 0-based, colors (N,3) uint8 或 None)。
    兼容 `v x y z [r g b]`（色归 0~1 → 回乘 255）与多边形面（扇形三角剖分）、`f a/b/c` 组索。"""
    verts: list = []
    cols: list = []
    idx: list = []
    has_color = False
    for ln in Path(path).read_text(encoding="ascii").splitlines():
        if ln.startswith("v "):
            p = ln.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
            if len(p) >= 6:
                has_color = True
                cols.append([float(p[4]), float(p[5]), float(p[6])])
            else:
                cols.append([0.0, 0.0, 0.0])
        elif ln.startswith("f "):
            tri = [int(tok.split("/")[0]) - 1 for tok in ln.split()[1:]]
            for k in range(1, len(tri) - 1):                # 扇形剖分（三角面时仅一次）
                idx.extend([tri[0], tri[k], tri[k + 1]])
    colors = None
    if has_color:
        colors = (np.asarray(cols, dtype=np.float64) * 255.0).clip(0, 255).astype(np.uint8)
    return (np.asarray(verts, dtype=np.float64), np.asarray(idx, dtype=np.int64), colors)


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
