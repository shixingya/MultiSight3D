"""管线端到端测试：mock 引擎全链路、失败中断保产物、--resume 续跑、from-stage 校验。"""

import shutil

import pytest

from multisight.events import Bus
from multisight.pipeline import run_pipeline
from multisight.workspace import STAGES, Workspace

pytest.importorskip("numpy")


def _prepare_ws(tmp_path, photos_dir, task_id="task1"):
    ws = Workspace.create(tmp_path, task_id=task_id, params={"preset": "fast"})
    shutil.copytree(photos_dir, ws.root / "raw")
    return ws


def test_full_mock_pipeline(tmp_path, photos_dir):
    ws = _prepare_ws(tmp_path, photos_dir)
    bus = Bus()
    ok = run_pipeline(ws, bus=bus, engine="mock")
    assert ok

    m = ws.manifest
    assert all(m["stages"][s]["status"] == "done" for s in STAGES)
    # 关键产物齐备且为合法文件
    assert (ws.root / "images" / "list.json").is_file()
    assert (ws.root / "sfm" / "cameras.txt").read_text(encoding="ascii").splitlines()[2].startswith("1 SIMPLE_PINHOLE")
    glb = (ws.root / "texture" / "model.glb").read_bytes()
    assert glb[:4] == b"glTF" and int.from_bytes(glb[4:8], "little") == 2
    assert (ws.root / "mvs" / "fusion.ply").read_text(encoding="ascii").startswith("ply")
    assert (ws.root / "report" / "report.json").is_file()

    events = list(bus.history())
    types = {e["type"] for e in events}
    assert {"stage_start", "progress", "stage_done", "end"} <= types
    assert events[-1]["ok"] is True


def test_real_engine_fails_past_sfm_and_keeps_products(tmp_path, photos_dir):
    # 两视图 SfM 已落地；合成噪声夹具的真实链路仍会在某个未落地阶段如实失败，
    # 关键是失败隔离：更早阶段的产物不被清空。sfm 正确性由 test_sfm_stage 精确覆盖。
    ws = _prepare_ws(tmp_path, photos_dir)
    ok = run_pipeline(ws, engine="real")
    assert not ok
    m = ws.manifest
    assert m["stages"]["preprocess"]["status"] == "done"   # 预处理是真实实现
    assert (ws.root / "images" / "list.json").is_file()    # 已完成产物未清空
    failed = [s for s in STAGES if m["stages"][s]["status"] == "failed"]
    assert failed
    assert STAGES.index(failed[0]) > STAGES.index("preprocess")  # 失败在预处理之后


def test_resume_after_failure(tmp_path, photos_dir):
    ws = _prepare_ws(tmp_path, photos_dir)
    assert not run_pipeline(ws, engine="real")            # 在 sfm 失败
    assert run_pipeline(ws, resume=True, engine="mock")   # mock 续跑成功
    assert all(ws.manifest["stages"][s]["status"] == "done" for s in STAGES)


def test_from_stage_requires_prereqs(tmp_path, photos_dir):
    ws = _prepare_ws(tmp_path, photos_dir)
    with pytest.raises(ValueError, match="产物缺失"):
        run_pipeline(ws, from_stage="mesh", engine="mock")   # 前置未跑，拒绝起跑
    assert run_pipeline(ws, engine="mock")                    # 全量跑完
    assert run_pipeline(ws, from_stage="texture", engine="mock")  # 前置齐备后可断点重跑


def test_preprocess_blur_and_warnings(tmp_path, photos_dir):
    from tests.conftest import make_photo  # noqa: F401 - 验证可复用夹具导出
    # 混入一张纯灰图（零 Laplacian 方差）→ 应被标记疑似模糊
    from PIL import Image
    Image.new("RGB", (240, 180), (128, 128, 128)).save(photos_dir / "flat.jpg")
    ws = _prepare_ws(tmp_path, photos_dir)
    assert run_pipeline(ws, engine="mock")
    import json
    listing = json.loads((ws.root / "images" / "list.json").read_text(encoding="utf-8"))
    # 每条照片记录带 focal_px_est 字段（EXIF 缺失时为 None，诚实缺省）
    assert all("focal_px_est" in p for p in listing["photos"])
    flat = [p for p in listing["photos"] if p["source"] == "flat.jpg"]
    assert flat and flat[0]["suspect_blur"] is True
    assert any("模糊" in w for w in listing["warnings"])


def test_focal_px_from_35mm():
    from multisight.stages.preprocess import focal_px_from_35mm
    # 50mm 等效、宽 1000px、全幅 36mm → 50/36*1000 ≈ 1388.889
    assert focal_px_from_35mm(50.0, 1000) == pytest.approx(1388.889, abs=1e-3)
    # 自定义传感器宽
    assert focal_px_from_35mm(35.0, 640, sensor_w_mm=36.0) == pytest.approx(622.222, abs=1e-3)
    # 非法入参诚实返回 None（不臆造焦距）
    assert focal_px_from_35mm(None, 1000) is None
    assert focal_px_from_35mm(50.0, 0) is None
    assert focal_px_from_35mm(50.0, -10) is None
    assert focal_px_from_35mm(0.0, 1000) is None
    assert focal_px_from_35mm(-5.0, 1000) is None
    assert focal_px_from_35mm("abc", 1000) is None
    assert focal_px_from_35mm(50.0, 1000, sensor_w_mm=0) is None
