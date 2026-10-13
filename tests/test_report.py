"""报告阶段单测：聚合各阶段 stats、质量阈值警告、mock 无 stats 不误报。"""

from __future__ import annotations

import json

import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.report import MESH_MIN_FACES, MVS_MIN_DENSE, REPROJ_MAX_PX, Mock as ReportStage
from multisight.workspace import Workspace

pytest.importorskip("numpy")


def _ctx(ws):
    return StageContext(ws=ws, bus=Bus(), engine="mock")


def _run(ws, **stats):
    for stage, payload in stats.items():
        if payload is None:
            continue
        ws.file(stage, "stats.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    ReportStage().run(_ctx(ws))
    return json.loads((ws.root / "report" / "report.json").read_text(encoding="utf-8"))


def _ws(tmp_path):
    return Workspace.create(tmp_path, task_id="t")


def test_mock_no_downstream_warning_without_stats(tmp_path):
    report = _run(_ws(tmp_path))
    # mock 各阶段不落 stats.json → 空 dict → 不应误报稀疏/面数不足
    assert report["mvs"] == {} and report["mesh"] == {}
    assert not any("稠密点云" in w for w in report["warnings"])
    assert not any("网格面数" in w for w in report["warnings"])


def test_sparse_cloud_triggers_warning(tmp_path):
    report = _run(_ws(tmp_path),
                  mvs={"dense_points": MVS_MIN_DENSE - 1},
                  mesh={"faces": MESH_MIN_FACES + 10})
    assert any("稠密点云" in w for w in report["warnings"])
    assert not any("网格面数" in w for w in report["warnings"])
    assert report["mvs"]["dense_points"] == MVS_MIN_DENSE - 1


def test_low_faces_triggers_warning(tmp_path):
    report = _run(_ws(tmp_path),
                  mvs={"dense_points": 10_000},
                  mesh={"faces": MESH_MIN_FACES - 1})
    assert any("网格面数" in w for w in report["warnings"])
    assert not any("稠密点云" in w for w in report["warnings"])


def test_healthy_stats_no_warning(tmp_path):
    report = _run(_ws(tmp_path),
                  mvs={"dense_points": 50_000},
                  mesh={"faces": 12_000},
                  texture={"triangles": 12_000})
    assert not any("稠密点云" in w for w in report["warnings"])
    assert not any("网格面数" in w for w in report["warnings"])
    assert report["texture"]["triangles"] == 12_000


def test_sfm_register_rate_warning(tmp_path):
    report = _run(_ws(tmp_path), sfm={"register_rate": 0.5})
    assert any("注册率" in w for w in report["warnings"])


def test_high_reproj_warning(tmp_path):
    report = _run(_ws(tmp_path), sfm={"register_rate": 1.0, "mean_reproj_px": REPROJ_MAX_PX + 1})
    assert any("重投影误差" in w for w in report["warnings"])


def test_median_reproj_key_also_checked(tmp_path):
    # mock 用 median_reproj_px；低值不应告警
    report = _run(_ws(tmp_path), sfm={"register_rate": 1.0, "median_reproj_px": 0.7})
    assert not any("重投影误差" in w for w in report["warnings"])


def test_null_reproj_no_warning(tmp_path):
    # 真实档重投影不可估计（None）→ 不误报
    report = _run(_ws(tmp_path), sfm={"register_rate": 1.0, "mean_reproj_px": None})
    assert not any("重投影误差" in w for w in report["warnings"])


def test_open_mesh_watertight_warning(tmp_path):
    report = _run(_ws(tmp_path),
                  mesh={"faces": 5000, "watertight": False, "boundary_edges": 312})
    assert any("非水密" in w and "312" in w for w in report["warnings"])


def test_watertight_mesh_no_warning(tmp_path):
    report = _run(_ws(tmp_path),
                  mesh={"faces": 5000, "watertight": True, "boundary_edges": 0})
    assert not any("非水密" in w for w in report["warnings"])


def test_retake_advice_from_signals(tmp_path):
    ws = _ws(tmp_path)
    ws.file("images", "list.json").write_text(json.dumps(
        {"count": 8, "sharp_count": 6, "warnings": []}), encoding="utf-8")
    ws.file("sfm", "stats.json").write_text(json.dumps({"register_rate": 0.7}), encoding="utf-8")
    ws.file("mvs", "stats.json").write_text(json.dumps({"dense_points": MVS_MIN_DENSE - 50}), encoding="utf-8")
    ReportStage().run(_ctx(ws))
    report = json.loads((ws.root / "report" / "report.json").read_text(encoding="utf-8"))
    joined = "\n".join(report["advice"])
    assert "环形再补" in joined and "15°" in joined      # 照片不足→方位/重叠建议
    assert "模糊" in joined                              # 有模糊→清晰度建议
    assert "注册率" in joined and "60~80%" in joined      # 低注册率→重叠度建议
    assert "稠密点稀疏" in joined                          # 稀疏云→MVS 补拍建议


def test_healthy_inputs_no_retake_advice(tmp_path):
    ws = _ws(tmp_path)
    ws.file("images", "list.json").write_text(json.dumps(
        {"count": 40, "sharp_count": 40, "warnings": []}), encoding="utf-8")
    ws.file("sfm", "stats.json").write_text(json.dumps({"register_rate": 0.98}), encoding="utf-8")
    ws.file("mvs", "stats.json").write_text(json.dumps({"dense_points": 60_000}), encoding="utf-8")
    ReportStage().run(_ctx(ws))
    report = json.loads((ws.root / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["advice"] == []


def test_stage_duration_helper():
    from multisight.stages.report import _parse_ts, _stage_duration
    assert _parse_ts("2026-10-13T10:00:00") is not None
    assert _parse_ts(None) is None and _parse_ts("garbage") is None
    assert _stage_duration({"started_at": "2026-10-13T10:00:00",
                            "finished_at": "2026-10-13T10:00:07"}) == 7.0
    assert _stage_duration({"started_at": "2026-10-13T10:00:00"}) is None      # 缺 finished
    assert _stage_duration({"started_at": "2026-10-13T10:00:10",
                            "finished_at": "2026-10-13T10:00:00"}) is None     # 结束早于开始→不编造负值


def test_report_includes_durations(tmp_path):
    ws = _ws(tmp_path)
    manifest = ws.manifest
    manifest["stages"]["sfm"].update({"status": "done", "started_at": "2026-10-13T10:00:00",
                                      "finished_at": "2026-10-13T10:00:12"})
    manifest["stages"]["mesh"].update({"status": "done", "started_at": "2026-10-13T10:00:12",
                                       "finished_at": "2026-10-13T10:00:20"})
    ws._write_manifest(manifest)
    report = _run(ws)
    assert report["stages"]["sfm"]["duration_s"] == 12.0
    assert report["stages"]["mesh"]["duration_s"] == 8.0
    assert report["stages"]["preprocess"]["duration_s"] is None   # 未跑无时间戳
    assert "started_at" not in report["stages"]["sfm"]             # 原始时间戳不外泄
    assert report["pipeline_duration_s"] == 20.0                   # 12 + 8


def test_report_no_durations_all_none(tmp_path):
    report = _run(_ws(tmp_path))   # 全新工作区无任何时间戳
    assert report["pipeline_duration_s"] is None
    assert all(report["stages"][s]["duration_s"] is None for s in report["stages"])
