"""资产库扫描/导入：把一个美术资源目录聚合成 AssetBundle。

约定目录形态（与常见武器/道具模型资源包一致）：
    <folder>/
      三维模型文件/*.flt|*.ive|*.tga   （或模型直接放根目录）
      sprites/S.0.png … S.35.png       （转盘帧，可选）
      preview.png / structure.png       （渲染图，可选）
      三维模型技术说明.docx             （元数据，可选）

scan_asset_dir 只读不写；import_asset_dir 把归一化产物（贴图 PNG、可选 GLB、
asset.json 清单）落到 out_dir，供 WebUI / demo 消费。原始二进制不入库。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .openflight import describe_openflight, describe_ive, extract_mesh
from .spec import parse_model_spec
from .textures import normalize_texture, load_image

_MODEL_EXT = {".flt", ".ive"}
_TEX_EXT = {".tga", ".png", ".jpg", ".jpeg", ".bmp"}
_SPRITE_RE = re.compile(r"^S\.(\d+)\.png$", re.I)


def _iter_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


@dataclass
class AssetBundle:
    name: str
    source_dir: str
    meta: dict[str, Any] = field(default_factory=dict)
    openflight: dict[str, Any] = field(default_factory=dict)
    performer: dict[str, Any] = field(default_factory=dict)
    geometry_ok: bool = False
    triangles: int = 0
    texture_png: str | None = None
    sprites: list[str] = field(default_factory=list)
    preview: str | None = None
    structure: str | None = None
    model_files: list[str] = field(default_factory=list)
    glb: str | None = None
    display: str = "turntable"   # 'turntable' | 'glb' | 'texture'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_spec(folder: Path) -> Path | None:
    docs = [p for p in folder.rglob("*.docx") if not p.name.startswith(("~$", "."))]
    for p in docs:
        if "说明" in p.name or "spec" in p.name.lower():
            return p
    return docs[0] if docs else None


def scan_asset_dir(folder: Path | str) -> AssetBundle:
    """只读扫描，聚合元数据/贴图/sprites/模型信息。不做任何写盘。"""
    folder = Path(folder)
    bundle = AssetBundle(name=folder.name, source_dir=str(folder))
    files = _iter_files(folder)

    # 元数据
    spec = _find_spec(folder)
    if spec:
        bundle.meta = parse_model_spec(spec)
        if bundle.meta.get("name"):
            bundle.name = bundle.meta["name"]

    # 模型文件（.flt/.ive）
    models = [p for p in files if p.suffix.lower() in _MODEL_EXT]
    bundle.model_files = sorted(str(p) for p in models)
    flt = next((p for p in models if p.suffix.lower() == ".flt"), None)
    if flt:
        info = describe_openflight(flt)
        bundle.openflight = info.to_dict()
        mesh = extract_mesh(flt)
        if mesh is not None:
            bundle.geometry_ok = True
            bundle.triangles = mesh.triangles
    ive = next((p for p in models if p.suffix.lower() == ".ive"), None)
    if ive:
        bundle.performer = describe_ive(ive)

    # 贴图（优先匹配 spec 里的 texture_file 名）
    texs = [p for p in files if p.suffix.lower() in _TEX_EXT]
    want = (bundle.meta.get("texture_file") or "").lower()
    tex = next((p for p in texs if want and p.name.lower() == want), None)
    if tex is None:
        # 排除 sprites 目录与 preview/structure，挑最大的彩色图当主贴图
        cand = [p for p in texs if "sprite" not in str(p.parent).lower()
                and p.name.lower() not in ("preview.png", "structure.png")]
        tex = max(cand, key=lambda p: p.stat().st_size, default=None)
    bundle.texture_png = str(tex) if tex else None

    # sprites 转盘帧
    sprite_pairs = []
    for p in files:
        m = _SPRITE_RE.match(p.name)
        if m:
            sprite_pairs.append((int(m.group(1)), str(p)))
    bundle.sprites = [s for _, s in sorted(sprite_pairs)]

    # 渲染图
    for p in files:
        low = p.name.lower()
        if low in ("preview.png", "preview.jpg"):
            bundle.preview = str(p)
        elif low in ("structure.png", "structure.jpg"):
            bundle.structure = str(p)

    # 展示模式决策
    if bundle.geometry_ok:
        bundle.display = "glb"
    elif bundle.sprites:
        bundle.display = "turntable"
    elif bundle.preview or bundle.texture_png:
        bundle.display = "texture"
    return bundle


def import_asset_dir(folder: Path | str, out_dir: Path | str) -> AssetBundle:
    """扫描 + 落盘归一化产物（贴图 PNG / 可选 GLB / asset.json）。返回带产物路径的 bundle。"""
    folder, out_dir = Path(folder), Path(out_dir)
    bundle = scan_asset_dir(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 贴图归一化为 PNG
    if bundle.texture_png:
        dst = out_dir / "texture.png"
        if normalize_texture(bundle.texture_png, dst):
            bundle.texture_png = str(dst)

    # sprites 复制为规整命名 frame_000.png …（先清空避免上次导入遗留的多余帧）
    if bundle.sprites:
        sdir = out_dir / "sprites"
        if sdir.exists():
            shutil.rmtree(sdir)
        sdir.mkdir(parents=True, exist_ok=True)
        norm: list[str] = []
        for i, s in enumerate(bundle.sprites):
            im = load_image(s)
            if im is not None:
                d = sdir / f"frame_{i:03d}.png"
                im.convert("RGBA").save(d)
                norm.append(str(d))
        bundle.sprites = norm

    # 渲染图复制
    for attr in ("preview", "structure"):
        src = getattr(bundle, attr)
        if src and load_image(src) is not None:
            d = out_dir / f"{attr}.png"
            load_image(src).convert("RGBA").save(d)
            setattr(bundle, attr, str(d))

    # 真几何 → GLB（仅在校验通过时）
    if bundle.geometry_ok:
        flt = next((m for m in bundle.model_files if m.lower().endswith(".flt")), None)
        mesh = extract_mesh(flt) if flt else None
        if mesh:
            from .glb import write_glb
            glb_path = out_dir / "model.glb"
            write_glb(glb_path, mesh.positions, mesh.indices,
                      texture_png=None, name=bundle.name)
            bundle.glb = str(glb_path)
    elif (out_dir / "model.glb").exists():
        # 本次几何未通过：清除上次导入可能遗留的陈旧 GLB，避免 asset.json 与实际不一致
        (out_dir / "model.glb").unlink()

    (out_dir / "asset.json").write_text(
        json.dumps(_relativize(bundle, out_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def _relativize(bundle: AssetBundle, base: Path) -> dict[str, Any]:
    """asset.json 只存相对 out_dir 的路径，保证资产目录整体可移植。"""
    d = bundle.to_dict()
    d.pop("source_dir", None)

    def rel(p: str | None):
        if not p:
            return p
        try:
            return Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
        except (ValueError, OSError):
            return Path(p).name
    d["texture_png"] = rel(d["texture_png"])
    d["preview"] = rel(d["preview"])
    d["structure"] = rel(d["structure"])
    d["glb"] = rel(d["glb"])
    d["sprites"] = [rel(s) for s in d["sprites"]]
    return d


def load_asset_bundle(folder: Path | str) -> AssetBundle:
    """从 import_asset_dir 产物目录读回 AssetBundle（相对路径解析为绝对）。"""
    folder = Path(folder)
    data = json.loads((folder / "asset.json").read_text(encoding="utf-8"))
    data["source_dir"] = str(folder)
    for k in ("texture_png", "preview", "structure", "glb"):
        if data.get(k):
            data[k] = str(folder / data[k])
    data["sprites"] = [str(folder / s) for s in data.get("sprites", [])]
    known = {f for f in AssetBundle.__dataclass_fields__}
    return AssetBundle(**{k: v for k, v in data.items() if k in known})


def slugify(name: str) -> str:
    """把目录名洗成文件系统/URL 安全的子目录名（保留中文，去非法字符）。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "asset"


def _looks_like_asset_dir(folder: Path) -> bool:
    """子树里只要出现模型/贴图/docx/转盘帧之一，就视为一个资源模型目录。"""
    if not folder.is_dir():
        return False
    for p in _iter_files(folder):
        ext = p.suffix.lower()
        if ext in _MODEL_EXT or ext in _TEX_EXT or ext == ".docx":
            return True
        if _SPRITE_RE.match(p.name):
            return True
    return False


def import_assets_root(parent: Path | str, out_root: Path | str) -> list[tuple[str, Path, AssetBundle]]:
    """批量导入：把「模型库根目录」下每个子目录当作一个资源模型导入。

    逐个产到 <out_root>/<slug>/（asset.json + 贴图 + sprites [+ demo]），
    跳过 .svn 等非资源目录。返回 [(源目录名, 产物目录, bundle)]。
    """
    parent, out_root = Path(parent), Path(out_root)
    results: list[tuple[str, Path, AssetBundle]] = []
    for child in sorted(p for p in parent.iterdir() if p.is_dir()):
        if not _looks_like_asset_dir(child):
            continue
        out = out_root / slugify(child.name)
        results.append((child.name, out, import_asset_dir(child, out)))
    return results


def list_library_asset_dirs(out_root: Path | str) -> list[Path]:
    """列出资产库根下已导入（含 asset.json）的子目录。"""
    root = Path(out_root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "asset.json").is_file())
