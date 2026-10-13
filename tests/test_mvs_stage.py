"""stages/mvs.Real 管线接线单测：读 SfM 位姿→平面扫描→融合稠密点云（真实一步）。

用受控合成多视图场景先跑 sfm.Real 产出 COLMAP 位姿/稀疏点，再跑 mvs.Real，验证：
产物 mvs/fusion.ply 就绪、stats 稠密点 > 0；缺 SfM 产物时诚实报错（不静默假成功）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.mvs import Real as MvsReal
from multisight.stages.sfm import Real as SfmReal
from multisight.workspace import Workspace

pytest.importorskip("numpy")

from tests.test_sfm_pipeline import _synthetic_scene  # noqa: E402  复用受控渲染场景
from tests.test_sfm_stage import _make_ws            # noqa: E402  复用工作区构造


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="real")


def _run_sfm(tmp_path):
    frames, _X, _pose = _synthetic_scene(n_frames=4)
    ws = _make_ws(tmp_path, frames)
    SfmReal().run(_ctx(ws))
    return ws


def test_real_mvs_produces_dense_cloud(tmp_path):
    ws = _run_sfm(tmp_path)
    MvsReal().run(_ctx(ws))

    assert ws.artifact_ready("mvs")
    stats = json.loads((ws.root / "mvs" / "stats.json").read_text(encoding="utf-8"))
    assert stats["engine"] == "real/planesweep"
    assert stats["dense_points"] > 0
    assert stats["reference_views"] >= 2

    # fusion.ply 头部与点数据一致、坐标有限
    lines = (ws.root / "mvs" / "fusion.ply").read_text(encoding="ascii").splitlines()
    nvert = int(next(ln.split()[-1] for ln in lines if ln.startswith("element vertex")))
    assert nvert == stats["dense_points"]
    body = [ln.split() for ln in lines[lines.index("end_header") + 1:] if ln.strip()]
    arr = np.array([[float(v) for v in row[:3]] for row in body])
    assert np.isfinite(arr).all() and len(arr) == nvert


def test_real_mvs_requires_sfm_products(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1")
    with pytest.raises(ValueError, match="缺少 SfM 产物"):
        MvsReal().run(_ctx(ws))


def test_real_mvs_requires_two_views(tmp_path):
    # 手工放一份单视图 SfM 产物：应因视图不足诚实报错
    ws = Workspace.create(tmp_path, task_id="t1")
    sfm = ws.dir("sfm")
    (sfm / "cameras.txt").write_text("1 SIMPLE_PINHOLE 40 30 50 20 15\n", encoding="ascii")
    (sfm / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 a.jpg\n-1\n", encoding="ascii")
    pts = "\n".join([f"5 {i} {i} {i}" for i in range(8)])
    (sfm / "points3D.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n" + pts + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="至少需 2 个"):
        MvsReal().run(_ctx(ws))
