"""stages/sfm.Real 管线接线单测：成功落 COLMAP 产物、匹配不足诚实报错、帧数守卫。

用受控合成帧对（含真实视差）构造最小 Workspace，直接跑 Real 阶段，
验证两视图内核已作为管线第一步接入；不相关帧对必须明确失败（不静默假成功）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.sfm import Real
from multisight.workspace import Workspace

pytest.importorskip("numpy")

from tests.test_sfm_pipeline import _synthetic_pair  # noqa: E402  复用受控渲染场景


def _make_ws(tmp_path, img_a, img_b, n_photos=2):
    ws = Workspace.create(tmp_path, task_id="t1")
    idir = ws.dir("images")
    Image.fromarray(np.clip(img_a, 0, 255).astype(np.uint8)).save(idir / "a.jpg", quality=95)
    Image.fromarray(np.clip(img_b, 0, 255).astype(np.uint8)).save(idir / "b.jpg", quality=95)
    h, w = img_a.shape[:2]
    photos = [
        {"file": "images/a.jpg", "width": int(w), "height": int(h), "source": "a.jpg"},
        {"file": "images/b.jpg", "width": int(w), "height": int(h), "source": "b.jpg"},
    ][:n_photos]
    (idir / "list.json").write_text(
        json.dumps({"photos": photos, "warnings": []}), encoding="utf-8")
    return ws


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def test_real_sfm_writes_colmap_products(tmp_path):
    img_a, img_b, _, _ = _synthetic_pair()
    ws = _make_ws(tmp_path, img_a, img_b)
    Real().run(_ctx(ws))

    assert ws.artifact_ready("sfm")
    cams = (ws.root / "sfm" / "cameras.txt").read_text(encoding="ascii").splitlines()
    assert cams[2].startswith("1 SIMPLE_PINHOLE")
    assert cams[3].startswith("2 SIMPLE_PINHOLE")

    imgslines = (ws.root / "sfm" / "images.txt").read_text(encoding="ascii").splitlines()
    # 图 1 恒等位姿；图 2 位姿由 matrix_to_quat 写入（qw 非负）
    assert imgslines[2].startswith("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 a.jpg")
    parts = imgslines[4].split()
    assert parts[0] == "2" and float(parts[1]) >= 0.0

    stats = json.loads((ws.root / "sfm" / "stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "real/two-view"
    assert stats["registered"] == 2 and stats["total"] == 2
    assert stats["inliers"] >= 8 and stats["sparse_points"] > 0
    assert abs(np.linalg.norm(stats["baseline"]) - 1.0) < 1e-3  # t 单位化


def test_real_sfm_raises_on_uncorrelated_pair(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((200, 200)) * 255
    b = rng.random((200, 200)) * 255
    ws = _make_ws(tmp_path, a, b)
    with pytest.raises(ValueError, match="未能建立可靠两视图"):
        Real().run(_ctx(ws))


def test_real_sfm_requires_two_photos(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.random((200, 200)) * 255
    ws = _make_ws(tmp_path, a, a, n_photos=1)
    with pytest.raises(ValueError, match="至少需 2 张"):
        Real().run(_ctx(ws))
