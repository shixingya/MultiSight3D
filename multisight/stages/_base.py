"""阶段基类工具：Real 引擎的统一「未落地」占位（按里程碑逐步替换）。"""

from __future__ import annotations

from ..pipeline import Stage, StageContext


class NotImplementedReal(Stage):
    """自研算法实装前的占位：明确告知所涉里程碑，避免静默假成功。"""

    milestone = "v0.1"

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError(
            f"自研 {self.name} 算法尚未落地（计划 {self.milestone} 交付，见 doc/PRD.md §7）；"
            f"当前请设 MS_ENGINE=mock 体验端到端管线。"
        )
