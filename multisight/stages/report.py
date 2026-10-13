"""质量自检报告（FR-09 前身）：聚合各阶段指标与警告，输出 report.json。

失败归因 + 补救建议的思路沿用 Butian3D reconGuide：不只甩 error，给可执行建议。
"""

from __future__ import annotations

import json
from datetime import datetime

from ..pipeline import Stage, StageContext
from ..workspace import STAGES
from .preprocess import MIN_RECOMMENDED

# 下游质量阈：低于此值提示重建可能稀疏/有洞（仅对真实 stats 生效，合成小场景不适用）
MVS_MIN_DENSE = 200
MESH_MIN_FACES = 100
REPROJ_MAX_PX = 1.5          # 稀疏重投影误差超此像素值提示几何拟合欠佳


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


_TS_FMT = "%Y-%m-%dT%H:%M:%S"


def _parse_ts(s):
    """解析 manifest 秒级 ISO 时间戳；缺失/异常返回 None（不因脏时间戳打断报告）。"""
    if not s:
        return None
    try:
        return datetime.strptime(s, _TS_FMT)
    except (ValueError, TypeError):
        return None


def _stage_duration(info):
    """单阶段耗时（秒，保留 1 位小数）；时间戳不全或结束早于开始返回 None（诚实缺省，不编造）。"""
    start = _parse_ts(info.get("started_at"))
    end = _parse_ts(info.get("finished_at"))
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    return round(delta, 1) if delta >= 0 else None


def _retake_advice(prep, sfm_stats, mvs_stats):
    """FR-11 拍摄引导：从信号→具体可执行的补拍建议（方位/重叠度/清晰度）。"""
    advice = []
    count = prep.get("count", 0)
    sharp = prep.get("sharp_count", count)
    if 0 < count < MIN_RECOMMENDED:
        advice.append(
            f"照片 {count} 张不足（建议 ≥{MIN_RECOMMENDED}）：绕物体水平环形再补 {MIN_RECOMMENDED - count}+ 张，"
            f"每隔 ~15° 一张、相邻重叠 ≥60%，并对顶/底俯仰角各补 2~3 张")
    if count and sharp < count:
        advice.append(
            f"{count - sharp} 张模糊：补拍时提高快门速度/增加光照/双手稳定或上三脚架，避免运动模糊与失焦")
    rate = sfm_stats.get("register_rate") if sfm_stats else None
    if rate is not None and rate < 0.85:
        advice.append(
            f"注册率 {rate:.0%} 偏低：对未成功注册的视角补拍，提高帧间重叠至 60~80%，避免纯白/重复纹理与反光面")
    if mvs_stats and mvs_stats.get("dense_points", 0) < MVS_MIN_DENSE:
        advice.append(
            "稠密点稀疏：增拍高纹理区域、缩小拍摄距离、保持相机内参稳定（不要中途变焦）以提高 MVS 匹配质量")
    return advice


class Mock(Stage):
    name = "report"

    def run(self, ctx: StageContext) -> None:
        root = ctx.ws.root
        manifest = ctx.ws.manifest
        prep = _read_json(root / "images" / "list.json") or {}
        sfm_stats = _read_json(root / "sfm" / "stats.json") or {}
        mvs_stats = _read_json(root / "mvs" / "stats.json") or {}
        mesh_stats = _read_json(root / "mesh" / "stats.json") or {}
        tex_stats = _read_json(root / "texture" / "stats.json") or {}

        warnings = list(prep.get("warnings") or [])
        advice = []
        failed = [s for s in STAGES if manifest["stages"][s]["status"] == "failed"]
        if failed:
            advice.append(f"阶段 {failed[0]} 失败：修复输入后可用 --resume 从断点续跑，已完成产物未清空")
        if sfm_stats and sfm_stats.get("register_rate", 1) < 0.85:
            warnings.append(f"照片注册率 {sfm_stats['register_rate']:.0%} < 85%，检查重叠度与模糊照片")
        reproj = sfm_stats.get("mean_reproj_px") if sfm_stats else None
        if reproj is None and sfm_stats:
            reproj = sfm_stats.get("median_reproj_px")
        if reproj is not None and reproj > REPROJ_MAX_PX:
            warnings.append(f"稀疏重投影误差 {reproj:.2f}px > {REPROJ_MAX_PX}px，位姿/结构拟合欠佳；建议补拍清晰、高重叠照片或降低镜头畸变")
        # 下游质量阈值警告（仅当对应阶段有真实 stats 产物时触发；mock 不写 stats 故为空 dict，不误报）
        if mvs_stats and mvs_stats.get("dense_points", 0) < MVS_MIN_DENSE:
            warnings.append(f"稠密点云仅 {mvs_stats.get('dense_points', 0)} 点（<{MVS_MIN_DENSE}），重建可能稀疏/有洞；增加照片或改善纹理/重叠")
        if mesh_stats and mesh_stats.get("faces", 0) < MESH_MIN_FACES:
            warnings.append(f"网格面数仅 {mesh_stats.get('faces', 0)}（<{MESH_MIN_FACES}），覆盖/置信不足；检查深度图质与视图数")
        if mesh_stats and mesh_stats.get("faces", 0) > 0 and mesh_stats.get("watertight") is False:
            warnings.append(f"网格非水密（{mesh_stats.get('boundary_edges', 0)} 条边界边），存在开洞/不封闭；适屏渲染可用，3D 打印前需补洞/缝合")

        advice.extend(_retake_advice(prep, sfm_stats, mvs_stats))

        stages_out = {}
        stage_durations = {}
        for s in STAGES:
            info = manifest["stages"][s]
            dur = _stage_duration(info)
            if dur is not None:
                stage_durations[s] = dur
            stages_out[s] = {k: v for k, v in info.items()
                             if k not in ("started_at", "finished_at")}
            stages_out[s]["duration_s"] = dur
        # 管线总耗时 = 各阶段耗时之和（不含阶段间空闲/排队，反映实际计算量）
        pipeline_duration_s = round(sum(stage_durations.values()), 1) if stage_durations else None

        report = {
            "task_id": ctx.ws.task_id,
            "engine": ctx.engine,
            "params": ctx.params,
            "photo_count": prep.get("count", 0),
            "pipeline_duration_s": pipeline_duration_s,
            "stages": stages_out,
            "sfm": sfm_stats,
            "mvs": mvs_stats,
            "mesh": mesh_stats,
            "texture": tex_stats,
            "artifacts": ctx.ws.list_artifacts(),
            "warnings": warnings,
            "advice": advice,
        }
        ctx.ws.file("report", "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress(self.name, 100, f"报告完成：{len(ctx.ws.list_artifacts())} 个产物，{len(warnings)} 条警告")


class Real(Mock):
    """报告聚合逻辑与引擎无关，Real 直接复用。"""
