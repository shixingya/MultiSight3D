"""六阶段实现包注册表。

每个子模块暴露 `Mock` 与 `Real` 两个阶段类；build_stage 按引擎选择。
自研算法按 PRD §7 里程碑逐版本填充 Real 实现（v0.1 SfM → v0.3 网格纹理）。
"""

from __future__ import annotations

from ..pipeline import Stage
from ..workspace import STAGES
from . import mesh, mvs, preprocess, report, sfm, texture

# 模块名 → 阶段名，一一对应 STAGES
_MODULES = {
    "preprocess": preprocess,
    "sfm": sfm,
    "mvs": mvs,
    "mesh": mesh,
    "texture": texture,
    "report": report,
}


def build_stage(name: str, engine: str = "mock") -> Stage:
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name}")
    module = _MODULES[name]
    cls = module.Real if engine == "real" else module.Mock
    return cls()


def stage_names() -> list[str]:
    return list(STAGES)
