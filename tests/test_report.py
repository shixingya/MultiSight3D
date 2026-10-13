"""报告阶段单测：聚合各阶段 stats、质量阈值警告、mock 无 stats 不误报。"""

from __future__ import annotations

import json

import pytest

from multisight.events import Bus
from multisight.pipeline import StageContext
from multisight.stages.report import MESH_MIN_FACES, MVS_MIN_DENSE, Mock as ReportStage
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
