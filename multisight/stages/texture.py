"""S4b 纹理映射与 GLB 直出（FR-07/08），v0.3 交付。

Mock：程序化棋盘贴图 + 合法 GLB 立方体——保证 WebUI 预览与下载链路真实可跑。
"""

from __future__ import annotations

from ..pipeline import Stage, StageContext
from ._base import NotImplementedReal
from ._synth import write_checker_texture, write_glb_cube


class Mock(Stage):
    name = "texture"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 30, "UV 展开（mock：盒投影）")
        write_checker_texture(ctx.ws.file("texture", "texture.png"))
        ctx.progress(self.name, 70, "多视图纹理烘焙（mock：程序化棋盘）")
        write_glb_cube(ctx.ws.file("texture", "model.glb"))
        ctx.progress(self.name, 100, "纹理与 GLB 直出完成：texture/model.glb")


class Real(NotImplementedReal):
    name = "texture"
    milestone = "v0.3"
