"""贴图读取与归一化：TGA / BMP / DDS(受限) / PNG / JPG → PNG。

PIL 原生支持 TGA（本项目的 agm84_c_512.tga 即 512×512 RGB 未压缩），
无需任何第三方转换器。DDS 无法被 PIL 直读，返回 None 由上层降级处理。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

SUPPORTED = {".tga", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: Path | str) -> Image.Image | None:
    """尽力打开图片；不支持/损坏返回 None（不抛，保证资产扫描健壮）。"""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception:  # noqa: BLE001 - 解码失败按「读不了」处理，交由上层回退
        return None


def normalize_texture(src: Path | str, dst: Path | str) -> Path | None:
    """把任意受支持贴图转成 PNG 落到 dst，返回 dst；不支持返回 None。"""
    im = load_image(src)
    if im is None:
        return None
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGBA" if "A" in im.getbands() else "RGB").save(dst)
    return dst
