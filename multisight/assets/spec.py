"""解析随模型附带的《三维模型技术说明.docx》。

docx 本质是 zip，正文在 word/document.xml。这里零依赖地抽段落文本，再按
中文字段标签抽取键值（模型名称 / 模型面数 / 文件格式 / 坐标系 / 比例尺 / 贴图 …）。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any


def _docx_paragraphs(path: Path) -> list[str]:
    """按 <w:p> 段落切分，段内拼接所有 <w:t> 文本。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    paras: list[str] = []
    for chunk in re.split(r"</w:p>", xml):
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", chunk, flags=re.S)
        line = "".join(texts)
        line = (line.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&quot;", '"'))
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            paras.append(line)
    return paras


# 字段标签 → 结果键；值取标签所在段落的下一非空段落（表格里标签/值常分行）。
_FIELD_LABELS: list[tuple[str, str, re.Pattern]] = [
    ("模型名称", "name", re.compile(r"^\S*导弹|^\S*-\S+$|[\w\-\.]+")),
    ("模型面数", "triangles", re.compile(r"(\d+)")),
    ("模型文件格式", "format", re.compile(r".+")),
    ("比例尺", "scale", re.compile(r"[^\s]+")),
]


def _find_value(paras: list[str], label: str) -> str | None:
    for i, p in enumerate(paras):
        if p.replace(" ", "").replace("\u3000", "") == label or p.strip() == label:
            for nxt in paras[i + 1:i + 4]:
                if nxt.strip() and nxt.strip() not in {lab for lab, _, _ in _FIELD_LABELS}:
                    return nxt.strip()
    # 兜底：形如「标签：值」同行
    for p in paras:
        m = re.match(rf"^{re.escape(label)}[:：]?\s*(.+)$", p.strip())
        if m:
            return m.group(1).strip()
    return None


def parse_model_spec(path: Path | str) -> dict[str, Any]:
    """返回结构化元数据；文件缺失/非 docx 时返回 {}（调用方按缺省处理）。"""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".docx":
        return {}
    try:
        paras = _docx_paragraphs(path)
    except (zipfile.BadZipFile, KeyError):
        return {}

    meta: dict[str, Any] = {"raw_paragraphs": paras[:40]}

    name = _find_value(paras, "模型名称")
    if name:
        meta["name"] = name
    tri = _find_value(paras, "模型面数")
    if tri:
        m = re.search(r"(\d+)", tri)
        if m:
            meta["triangles"] = int(m.group(1))
    fmt = _find_value(paras, "模型文件格式")
    if fmt:
        meta["format"] = fmt
    scale = _find_value(paras, "比例尺")
    if scale:
        meta["scale"] = scale

    # 坐标系 / 原点 / 贴图：表格标题与值常分行，优先取真正描述轴/手性的正文行
    # （例：标题「坐标系与原点说明」与值「Y朝向实体正前方…符合右手定则」分属两段）
    joined = "\n".join(paras)
    coord = next((p for p in paras
                  if re.search(r"(右手定则|左手定则|朝向实体)", p)), None)
    if coord is None:
        for p in paras:
            m = re.match(r"^\s*坐标系\s*[:：]\s*(.+)$", p.strip())
            if m and "说明" not in m.group(1):
                coord = m.group(1)
                break
    if coord is None:
        coord = next((p for p in paras
                      if "坐标系" in p and "说明" not in p and re.search(r"[XYZ]", p)), None)
    if coord:
        meta["coordinate_system"] = coord.strip()
    origin = next((p for p in paras if "坐标原点" in p
                   or ("原点" in p and "世界坐标" in p)), None)
    if origin:
        meta["origin"] = origin.strip()
    texnote = next((p for p in paras if "像素" in p or ".tga" in p.lower() or "贴图：" in p
                    or "贴图:" in p), None)
    if texnote:
        meta["texture_note"] = texnote.strip()
    m = re.search(r"([^\s/\\]+\.(?:tga|png|jpe?g|bmp|dds))", joined, flags=re.I)
    if m:
        meta["texture_file"] = m.group(1)
    return meta
