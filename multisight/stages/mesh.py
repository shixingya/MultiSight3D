"""S4a 网格重建（FR-06），v0.3 交付。Mock：输出占位立方 mesh.obj。"""

from __future__ import annotations

from ..pipeline import Stage, StageContext
from ._base import NotImplementedReal
from ._synth import write_obj_cube


class Mock(Stage):
    name = "mesh"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 30, "空间裁剪与泊松重建（mock：占位立方）")
        write_obj_cube(ctx.ws.file("mesh", "mesh.obj"))
        ctx.progress(self.name, 70, "减面（mock：跳过）")
        ctx.progress(self.name, 100, "网格重建完成：8 顶点 / 12 三角形（占位）")


class Real(NotImplementedReal):
    name = "mesh"
    milestone = "v0.3"
