"""stages/sfm.Real 管线接线单测：增量注册落 COLMAP 产物、初始对不可靠诚实报错、帧数守卫。

用受控合成帧（含真实视差/充足基线）构造最小 Workspace，直接跑 Real 阶段，
验证多视图增量注册已作为管线真实一步接入；不相关帧必须明确失败（不静默假成功）。
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

from tests.test_sfm_pipeline import _synthetic_pair, _synthetic_scene  # noqa: E402  复用受控渲染场景


def _make_ws(tmp_path, frames):
    """把若干灰度帧写为 images/fNN.jpg 并生成 list.json（零二进制入库）。"""
    ws = Workspace.create(tmp_path, task_id="t1")
    idir = ws.dir("images")
    photos = []
    for k, fr in enumerate(frames):
        name = f"f{k:02d}.jpg"
        Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).save(idir / name, quality=95)
        h, w = fr.shape[:2]
        photos.append({"file": f"images/{name}", "width": int(w), "height": int(h), "source": name})
    (idir / "list.json").write_text(
        json.dumps({"photos": photos, "warnings": []}), encoding="utf-8")
    return ws


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def _data_lines(path):
    return [ln for ln in path.read_text(encoding="ascii").splitlines()
            if ln and not ln.startswith("#")]


def test_real_sfm_writes_colmap_products(tmp_path):
    img_a, img_b, _, _ = _synthetic_pair()
    ws = _make_ws(tmp_path, [img_a, img_b])
    Real().run(_ctx(ws))

    assert ws.artifact_ready("sfm")
    cam_lines = _data_lines(ws.root / "sfm" / "cameras.txt")
    assert len(cam_lines) == 2                          # 两帧均注册 → 两相机行
    assert all(ln.split()[1] == "SIMPLE_PINHOLE" for ln in cam_lines)

    img_lines = _data_lines(ws.root / "sfm" / "images.txt")
    img_lines = [ln for ln in img_lines if ln != "-1"]
    assert len(img_lines) == 2
    # 初始帧恒等（世界系 = 初始帧 a）：四元数 1,0,0,0，平移 0
    first = img_lines[0].split()
    assert first[0] == "1"
    assert np.allclose([float(x) for x in first[1:8]], [1, 0, 0, 0, 0, 0, 0], atol=1e-9)
    # 第二帧含非平凡位姿
    second = img_lines[1].split()
    assert second[0] == "2" and float(second[1]) >= 0.0

    stats = json.loads((ws.root / "sfm" / "stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "real/incremental"
    assert stats["registered"] == 2 and stats["total"] == 2
    assert stats["initial_inliers"] >= 8 and stats["sparse_points"] > 0
    # 全局 BA 已应用：均重投影有限且小；基线长度由 BA 定（尺度 gauge，不再强制单位）
    assert stats["bundle_applied"] is True
    assert stats["mean_reproj_px"] is not None and stats["mean_reproj_px"] < 2.0
    bl = np.linalg.norm(stats["baseline"])
    assert 0.5 < bl < 1.5


def test_real_sfm_registers_multiple_frames(tmp_path):
    frames, _X, _pose = _synthetic_scene(n_frames=4)
    ws = _make_ws(tmp_path, frames)
    Real().run(_ctx(ws))

    stats = json.loads((ws.root / "sfm" / "stats.json").read_text(encoding="utf-8"))
    assert stats["registered"] >= 3                      # 增量注册多于一对
    assert stats["registered"] == len(_data_lines(ws.root / "sfm" / "cameras.txt"))
    assert stats["sparse_points"] > 0
    # 每张注册照片都应有对应 images.txt 行
    img_entries = [ln for ln in _data_lines(ws.root / "sfm" / "images.txt") if ln != "-1"]
    assert len(img_entries) == stats["registered"]


def test_real_sfm_raises_on_uncorrelated_pair(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.random((200, 200)) * 255
    b = rng.random((200, 200)) * 255
    ws = _make_ws(tmp_path, [a, b])
    with pytest.raises(ValueError, match="可靠初始两视图"):
        Real().run(_ctx(ws))


def test_real_sfm_requires_two_photos(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.random((200, 200)) * 255
    ws = _make_ws(tmp_path, [a])
    with pytest.raises(ValueError, match="至少需 2 张"):
        Real().run(_ctx(ws))
