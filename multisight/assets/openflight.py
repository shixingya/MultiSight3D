"""OpenFlight (.flt) 读取：头部元信息可靠解析 + 带校验的几何提取。

OpenFlight 是多米诺（MultiGen-Paradigm）可视化行业标准格式，大端字节序，
文件由线性二进制记录流组成：每条记录 = uint16 opcode + uint16 length（含头），
长度按 4 字节对齐。数据库头记录（opcode 1）固定 308 字节（版本相关），字段：

    off 0   u16 magic (=0x0001)
    off 2   u16 record_length (=308)
    off 4   char[8]  ascii id（常为 "db"）
    off 12  u32 format_revision
    off 16  u32 edit_revision
    off 20  char[32] 最后修改时间（ASCII）
    off 52  ...  子节点 ID 计数器 / 顶点单位(0=米) / 投影类型 等

顶点记录 MgVertexList（opcode 5）：body = u16 n + (对齐) + n×MgVertex，
每个顶点在 v12+ 为 3×float32、更早为 3×fixed32.32。

⚠️ 现实中的导出器（尤其 Maya 插件）常产出非标准变体：面/顶点记录被写成空壳模板、
真实几何混在扩展记录里。因此本模块的 extract_mesh 采取「严格校验、宁缺毋滥」策略：
只有当能组装出数量与包围盒都合理的三角网格时才返回，否则返回 None，由上层回退到
转盘 sprite 展示。describe() 则总是可靠给出格式指纹与记录直方图。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAGIC = 0x0001
_HEADER_LEN = 308
_IVE_MAGIC = bytes([0x04, 0x03, 0x02, 0x01])   # SGI Performer .ive 常见开头字节序标记

# opcode → 名称（覆盖常见节点/几何记录，未知记 '?'）
OPCODES = {
    1: "objectHeader", 2: "groupHeader", 3: "objectData", 4: "vertex",
    5: "vertexList", 6: "vertexListExt", 7: "outerLoop", 8: "innerLoop",
    9: "face", 10: "faceStyle", 11: "faceList", 12: "materialList",
    13: "scMap2D", 14: "material", 15: "lightSource", 16: "texture",
    17: "fog", 18: "patch", 19: "rigidTransform", 20: "lod",
    21: "lodSelector", 22: "classData", 23: "textData", 24: "instance",
    27: "xform", 32: "faceExt", 33: "materialPalette", 34: "materialPaletteList",
}


@dataclass
class OpenFlightInfo:
    """格式指纹（总是可得）。"""
    ok: bool
    version: int = 0
    version_str: str = ""
    ascii_id: str = ""
    format_revision: int = 0
    edit_revision: int = 0
    birth: str = ""
    unix_path: str = ""
    vertex_unit: str = ""
    record_count: int = 0
    opcode_hist: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Mesh:
    positions: list[tuple[float, float, float]]
    indices: list[int]
    @property
    def triangles(self) -> int:
        return len(self.indices) // 3


def _fixed32(buf: bytes, off: int) -> float:
    hi, lo = struct.unpack_from(">ii", buf, off)
    return hi + lo / 4294967296.0


def describe_openflight(path: Path | bytes | str) -> OpenFlightInfo:
    """解析头部 + 走一遍记录流做 opcode 直方图。任何异常都吞成 ok=False。"""
    data = _read_bytes(path)
    if data is None or len(data) < _HEADER_LEN:
        return OpenFlightInfo(ok=False, note="文件过小或不可读")
    magic, version = struct.unpack_from(">HH", data, 0)
    if magic != MAGIC:
        return OpenFlightInfo(ok=False, note=f"magic 非 OpenFlight(0x{magic:04x})")
    ascii_id = data[4:12].split(b"\x00")[0].decode("ascii", "ignore")
    fmt_rev, edit_rev = struct.unpack_from(">II", data, 12)
    birth = data[20:52].split(b"\x00")[0].decode("ascii", "ignore").strip()
    # 顶点坐标单位在偏移 62（1 字节）
    unit_code = data[62] if len(data) > 62 else 255
    unit = {0: "米", 1: "千米", 4: "英尺", 5: "英寸", 8: "海里"}.get(unit_code, "未知")
    major, minor = divmod(version, 100)

    hist: dict[str, int] = {}
    count = 0
    pos = _HEADER_LEN
    n = len(data)
    while pos + 4 <= n:
        op, ln = struct.unpack_from(">HH", data, pos)
        if ln < 4 or pos + ln > n:
            break
        hist[OPCODES.get(op, f"op{op}")] = hist.get(OPCODES.get(op, f"op{op}"), 0) + 1
        count += 1
        pos += ln

    return OpenFlightInfo(
        ok=True, version=version, version_str=f"{major}.{minor}",
        ascii_id=ascii_id, format_revision=fmt_rev, edit_revision=edit_rev,
        birth=birth, unix_path="", vertex_unit=unit,
        record_count=count, opcode_hist=hist,
    )


def describe_ive(path: Path | bytes | str) -> dict[str, Any]:
    """SGI/MultiGen Performer .ive 的诚实格式指纹（仅识别签名，不做几何解析）。

    .ive 与 .flt 常为同一模型的不同封装；完整解析需 OpenSceneGraph/Performer，
    本函数只给出“是不是 .ive + 头部前几个整数 + 文件大小”，绝不猜测几何。
    """
    data = _read_bytes(path)
    if data is None or len(data) < 8:
        return {"ok": False, "note": "文件过小或不可读"}
    sig = data[:4]
    known = sig == _IVE_MAGIC
    h1 = struct.unpack_from(">I", data, 4)[0] if len(data) >= 8 else 0
    return {
        "ok": known, "signature": sig.hex(), "header_int": h1, "size": len(data),
        "note": "Performer .ive 二进制场景（未做几何解析，需 OpenSceneGraph/Performer）"
                if known else "非已知 .ive 签名",
    }


def extract_mesh(path: Path | bytes, *, min_tris: int = 40,
                 max_extent: float = 200.0) -> Mesh | None:
    """带校验的几何提取；不满足合理性返回 None（上层回退转盘）。"""
    data = _read_bytes(path)
    if data is None or len(data) < _HEADER_LEN:
        return None
    positions: list[tuple[float, float, float]] = []
    indices: list[int] = []
    pos, n = _HEADER_LEN, len(data)
    while pos + 4 <= n:
        op, ln = struct.unpack_from(">HH", data, pos)
        if ln < 4 or pos + ln > n:
            break
        if op in (5, 6):                       # vertexList / vertexListExt
            tri = _decode_vertexlist(data, pos + 4, ln - 4)
            if tri:
                base = len(positions)
                positions.extend(tri)
                for k in range(len(tri) - 2):  # 扇形三角化
                    indices.extend([base, base + k + 1, base + k + 2])
        pos += ln

    if len(indices) < min_tris * 3 or not positions:
        return None
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    span = max(max(xs, default=0) - min(xs, default=0),
               max(ys, default=0) - min(ys, default=0),
               max(zs, default=0) - min(zs, default=0))
    if not (0.01 < span <= max_extent):        # 包围盒须是「物件级」尺寸
        return None
    return Mesh(positions=positions, indices=indices)


def _decode_vertexlist(buf: bytes, start: int, body_len: int) -> list[tuple[float, float, float]] | None:
    """把一条 vertexList body 解成顶点三元组；数量/取值不合理返回 None。"""
    if body_len < 2:
        return None
    cnt = struct.unpack_from(">H", buf, start)[0]
    if not (3 <= cnt <= 6):                    # OpenFlight 单条 vertexList 上限 6
        return None
    vstart = start + 4                          # u16 count + 2 pad（4 字节对齐）
    verts: list[tuple[float, float, float]] = []
    for i in range(cnt):
        off = vstart + i * 12
        if off + 12 > start + body_len:
            return None
        x, y, z = struct.unpack_from(">3f", buf, off)
        if not all(math.isfinite(v) and abs(v) < 1e4 for v in (x, y, z)):
            return None
        if all(abs(v) < 1e-6 for v in (x, y, z)):   # 全零=空壳模板，判为无效
            return None
        verts.append((x, y, z))
    return verts


def _read_bytes(path: Path | bytes | str) -> bytes | None:
    if isinstance(path, (bytes, bytearray)):
        return bytes(path)
    p = Path(path)
    try:
        return p.read_bytes() if p.is_file() else None
    except OSError:
        return None
