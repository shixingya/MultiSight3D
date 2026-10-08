"""S1 图像预处理（FR-02）——真实实现：格式校验 / EXIF / 降采样 / 模糊检测。

这是 v0.1 就落地的真实逻辑（不依赖 GPU）：摄影测量的第一痛点是
「拍摄质量差 → 重建失败」，把经验固化成产品能力（沿用 Butian3D reconGuide 思路）。
Mock 与 Real 共用同一实现。
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageOps

from ..pipeline import Stage, StageContext

VALID_FORMATS = {".jpg", ".jpeg", ".png"}
MIN_RECOMMENDED = 15          # 低于该张数给出补拍建议（物体级 close-range）
BLUR_THRESHOLD = 60.0         # Laplacian 方差低于此值判定疑似模糊


def laplacian_variance(gray: np.ndarray) -> float:
    """简易清晰度指标：Laplacian 卷积响应的方差，值越小越糊。"""
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    padded = np.pad(gray.astype(np.float64), 1, mode="edge")
    acc = np.zeros((h, w))
    for i in range(3):
        for j in range(3):
            acc += kernel[i, j] * padded[i:i + h, j:j + w]
    return float(acc.var())


def read_exif_focal(img: Image.Image) -> float | None:
    """读取 35mm 等效焦距（多数手机返回 None，交由 EXIF 缺失兜底）。"""
    try:
        exif = img.getexif()
        focal = exif.get(37385)  # FocalLength
        fpsize = exif.get(41989)  # FocalLengthIn35mmFilm
        if fpsize:
            return float(fpsize)
        if focal:
            return float(focal)
    except Exception:  # noqa: BLE001 - EXIF 损坏不应中断预处理
        pass
    return None


class PreprocessBase(Stage):
    name = "preprocess"

    def run(self, ctx: StageContext) -> None:
        raw_dir = ctx.ws.root / "raw"
        if not raw_dir.is_dir():
            raise FileNotFoundError("工作区缺少 raw/ 输入照片目录")
        src_files = sorted(
            p for p in raw_dir.iterdir()
            if p.suffix.lower() in VALID_FORMATS
        )
        skipped = [p.name for p in raw_dir.iterdir() if p.suffix.lower() not in VALID_FORMATS]
        if not src_files:
            raise ValueError("没有合法的 jpg/png 输入照片")

        max_size = int(ctx.preset.get("max_image_size", 1600))
        out_dir = ctx.ws.dir("images")
        records = []
        for idx, src in enumerate(src_files):
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)          # 按 EXIF 方向摆正
                im = im.convert("RGB")
                w, h = im.size
                scale = min(1.0, max_size / max(w, h))
                if scale < 1.0:
                    im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                gray = np.asarray(im.convert("L"))
                blur = laplacian_variance(gray)
                stem = f"img{idx:04d}"
                dst = out_dir / f"{stem}.jpg"
                im.save(dst, quality=92)
                records.append({
                    "id": stem,
                    "source": src.name,
                    "file": f"images/{dst.name}",
                    "width": im.size[0],
                    "height": im.size[1],
                    "focal_mm_35eq": read_exif_focal(Image.open(src)),
                    "blur_score": round(blur, 1),
                    "suspect_blur": blur < BLUR_THRESHOLD,
                })
            ctx.progress(self.name, int((idx + 1) / len(src_files) * 90),
                         f"已处理 {idx + 1}/{len(src_files)} 张")

        warnings = []
        sharp = [r for r in records if not r["suspect_blur"]]
        if len(records) < MIN_RECOMMENDED:
            warnings.append(f"照片仅 {len(records)} 张，建议 ≥{MIN_RECOMMENDED} 张环绕拍摄提高覆盖")
        if len(sharp) < len(records):
            warnings.append(f"{len(records) - len(sharp)} 张疑似模糊，重建成品可能缺面，建议补拍")
        if skipped:
            warnings.append(f"忽略 {len(skipped)} 个非图片文件：{', '.join(skipped[:5])}")

        summary = {
            "count": len(records),
            "sharp_count": len(sharp),
            "preset": ctx.params.get("preset", "standard"),
            "max_image_size": max_size,
            "warnings": warnings,
            "photos": records,
        }
        ctx.ws.file("images", "list.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress(self.name, 100, f"预处理完成：{len(records)} 张，其中 {len(sharp)} 张清晰")


class Mock(PreprocessBase):
    pass


class Real(PreprocessBase):
    pass
