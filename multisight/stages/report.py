"""质量自检报告（FR-09 前身）：聚合各阶段指标与警告，输出 report.json。

失败归因 + 补救建议的思路沿用 Butian3D reconGuide：不只甩 error，给可执行建议。
"""

from __future__ import annotations

import json

from ..pipeline import Stage, StageContext
from ..workspace import STAGES


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class Mock(Stage):
    name = "report"

    def run(self, ctx: StageContext) -> None:
        root = ctx.ws.root
        manifest = ctx.ws.manifest
        prep = _read_json(root / "images" / "list.json") or {}
        sfm_stats = _read_json(root / "sfm" / "stats.json") or {}

        warnings = list(prep.get("warnings") or [])
        advice = []
        failed = [s for s in STAGES if manifest["stages"][s]["status"] == "failed"]
        if failed:
            advice.append(f"阶段 {failed[0]} 失败：修复输入后可用 --resume 从断点续跑，已完成产物未清空")
        if sfm_stats and sfm_stats.get("register_rate", 1) < 0.85:
            warnings.append(f"照片注册率 {sfm_stats['register_rate']:.0%} < 85%，检查重叠度与模糊照片")

        report = {
            "task_id": ctx.ws.task_id,
            "engine": ctx.engine,
            "params": ctx.params,
            "photo_count": prep.get("count", 0),
            "stages": {
                s: {k: v for k, v in manifest["stages"][s].items() if k != "started_at"}
                for s in STAGES
            },
            "sfm": sfm_stats,
            "artifacts": ctx.ws.list_artifacts(),
            "warnings": warnings,
            "advice": advice,
        }
        ctx.ws.file("report", "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress(self.name, 100, f"报告完成：{len(ctx.ws.list_artifacts())} 个产物，{len(warnings)} 条警告")


class Real(Mock):
    """报告聚合逻辑与引擎无关，Real 直接复用。"""
