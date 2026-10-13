"""S4b 纹理映射与 GLB 直出（FR-07/08），v0.3。

Mock：程序化棋盘贴图 + 合法 GLB 立方体——保证 WebUI 预览与下载链路真实可跑。
Real：v0.3 起步档——**逐顶点真彩色 GLB**（无 UV 烘焙）。读上游 mesh.Real 产出的 mesh.obj
（顶点已带参考视图真彩色）→ 用泛化 `assets.glb.write_glb` 直出带 COLOR_0 的 model.glb。
诚实原则：缺网格产物即明确报错（不静默假成功）；UV 展开/多视图纹理烘焙为后续里程碑（故不产出
texture.png，ARTIFACTS[texture] 相应改为仅要求 model.glb）。
"""

from __future__ import annotations

import json

from ..pipeline import Stage, StageContext
from ._synth import write_checker_texture, write_glb_cube, read_obj_trimesh


class Mock(Stage):
    name = "texture"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 30, "UV 展开（mock：盒投影）")
        write_checker_texture(ctx.ws.file("texture", "texture.png"))
        ctx.progress(self.name, 70, "多视图纹理烘焙（mock：程序化棋盘）")
        write_glb_cube(ctx.ws.file("texture", "model.glb"))
        ctx.progress(self.name, 100, "纹理与 GLB 直出完成：texture/model.glb")


class Real(Stage):
    """读 mesh.obj → 逐顶点真彩色 model.glb（v0.3 起步档，无 UV 烘焙）。"""

    name = "texture"

    def run(self, ctx: StageContext) -> None:
        from ..assets.glb import write_glb

        obj = ctx.ws.root / "mesh" / "mesh.obj"
        if not obj.is_file():
            raise ValueError("缺少网格产物（mesh/mesh.obj），无法直出 GLB；请先用 real 引擎跑 mesh")
        ctx.progress(self.name, 30, "读网格与顶点色（real：逐顶点真彩色，无 UV 烘焙）")
        positions, indices, colors = read_obj_trimesh(obj)
        if len(positions) == 0 or len(indices) == 0:
            raise ValueError("网格无顶点/面无法生成 GLB；不臆造模型")
        ctx.progress(self.name, 70, f"写出 GLB：{len(positions)} 顶点 / {len(indices) // 3} 三角面")
        write_glb(ctx.ws.file("texture", "model.glb"), positions, indices,
                  colors=colors, name="reconstruction", z_up=False)
        ctx.progress(self.name, 100, "GLB 直出完成：texture/model.glb")
        (ctx.ws.root / "texture" / "stats.json").write_text(json.dumps({
            "engine": "real/vertex-color-glb",
            "vertices": int(len(positions)),
            "triangles": int(len(indices) // 3),
            "vertex_colors": colors is not None,
            "note": "v0.3 起步：逐顶点真彩色 GLB（无 UV）；UV 展开/多视图纹理烘焙为后续里程碑",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
