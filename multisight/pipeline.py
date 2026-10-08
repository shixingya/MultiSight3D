"""管线执行器：按阶段顺序推进状态机，广播事件，落盘 manifest。

引擎适配器模式（沿用 Butian3D 经验）：MS_ENGINE=mock 使用内置模拟实现跑通
端到端链路；MS_ENGINE=real 加载自研算法实现（按里程碑逐版本落地）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import Bus
from .workspace import STAGES, STATUS_DONE, STATUS_FAILED, STATUS_RUNNING, Workspace

DEFAULT_ENGINE = os.environ.get("MS_ENGINE", "mock")

# 预设档位（FR-10：标准/精细/快速），影响预处理降采样上限等
PRESETS = {
    "fast": {"max_image_size": 1024},
    "standard": {"max_image_size": 1600},
    "fine": {"max_image_size": 2400},
}


@dataclass
class StageContext:
    """阶段运行时上下文：阶段只依赖这些接口，便于替换实现与单测注入。"""

    ws: Workspace
    bus: Bus
    engine: str = DEFAULT_ENGINE
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def preset(self) -> dict[str, Any]:
        return PRESETS.get(self.params.get("preset", "standard"), PRESETS["standard"])

    def progress(self, stage: str, percent: int, message: str = "") -> None:
        """上报阶段进度：写入 manifest 并广播事件。"""
        self.ws.set_stage(stage, progress=percent)
        self.bus.publish("progress", stage=stage, percent=percent, message=message)


class Stage:
    """阶段基类：子类实现 run(ctx)；name 必须取自 STAGES。"""

    name: str = ""

    def run(self, ctx: StageContext) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError


StageFactory = Callable[[], Stage]


def run_pipeline(ws: Workspace, *, from_stage: str | None = None,
                 resume: bool = False, bus: Bus | None = None,
                 engine: str | None = None, params: dict[str, Any] | None = None) -> bool:
    """执行管线。返回是否全部成功。

    - from_stage：跳过该阶段之前的所有阶段（要求此前产物已存在）。
    - resume：从 manifest 中第一个未完成阶段继续（与 from_stage 二选一）。
    """
    from .stages import build_stage

    bus = bus or Bus()
    engine = engine or DEFAULT_ENGINE
    params = params or dict(ws.manifest.get("params") or {})

    if resume:
        from_stage = ws.next_incomplete_stage()
        if from_stage is None:
            bus.publish("log", message="所有阶段已完成，无需续跑")
            return True
    if from_stage and from_stage not in STAGES:
        raise ValueError(f"unknown stage: {from_stage}")

    start = STAGES.index(from_stage) if from_stage else 0
    if from_stage:
        missing = [s for s in STAGES[:start] if not ws.artifact_ready(s)]
        # report 是终点产物，允许在未跑到时不缺件；跳过它做校验没有意义
        missing = [s for s in missing if s != "report"]
        if missing:
            raise ValueError(f"前置阶段产物缺失，无法从 {from_stage} 起跑: {missing}")

    ctx = StageContext(ws=ws, bus=bus, engine=engine, params=params)
    ok = True
    for name in STAGES[start:]:
        stage = build_stage(name, engine)
        ws.set_stage(name, status=STATUS_RUNNING, progress=0)
        bus.publish("stage_start", stage=name)
        try:
            stage.run(ctx)
            ws.set_stage(name, status=STATUS_DONE, progress=100)
            bus.publish("stage_done", stage=name)
        except Exception as exc:  # noqa: BLE001 - 阶段失败需归因而不是炸掉整条管线
            ws.set_stage(name, status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
            bus.publish("stage_failed", stage=name, error=str(exc))
            ok = False
            break  # 保留已完成产物，等待修复后 --resume
    bus.publish("end", ok=ok)
    return ok
