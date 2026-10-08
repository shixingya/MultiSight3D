"""共享夹具：用 Pillow 生成有纹理的清晰测试照片（零二进制入库）。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def make_photo(path, size=(240, 180), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    # 叠加棋盘结构提高 Laplacian 方差（保证不被误判为模糊）
    gx = np.arange(size[1])[:, None] // 30 + np.arange(size[0])[None, :] // 30
    mask = (gx % 2 == 0)[..., None]
    arr = np.where(mask, np.clip(arr.astype(int) + 80, 0, 255), arr).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=92)


@pytest.fixture()
def photos_dir(tmp_path):
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(6):
        make_photo(d / f"p{i:03d}.jpg", seed=i)
    return d
