# 决策记录（DECISIONS）

自主迭代期间遇到需要产品/架构取向的岔路时，先记录决策与理由，继续推进能做的部分
（延续用户「离开期间全程自主决策、遇绕不过的阻塞先记录待后期处理」的授权）。

## D-001（v0.3）texture 阶段 real 档不产出 texture.png，ARTIFACTS 改为仅要求 model.glb

- **背景**：`stages/texture.Real` 第一档（v0.3 起步）采用**逐顶点真彩色 GLB**（读 `mesh.obj`
  的顶点色 → `assets.glb.write_glb` 写 `COLOR_0`），没有 UV 展开与多视图纹理烘焙，因此不存在
  有意义的光栅贴图 `texture.png`。而原 `ARTIFACTS["texture"]` 同时要求 `texture.png` 与
  `model.glb`，会让 real 档无法通过 `artifact_ready("texture")`。
- **决策**：将 `ARTIFACTS["texture"]` 收敛为 `["texture/model.glb"]`。mock 档仍额外写
  `texture.png`（文件存在即无害，只是不再被 `list_artifacts` 主动列出）。
- **理由**：能力就绪应以「该档真实产出的交付物」为准，不为未落地的 UV 烘焙伪造贴图（诚实原则）。
  预览/下载链路依赖 `model.glb`，去掉 `texture.png` 硬约束不影响 three.js 预览。
- **回退条件**：待 v0.3 后续里程碑实现真实 UV 展开 + 多视图纹理烘焙并产出 `texture.png` 时，
  把它加回 `ARTIFACTS["texture"]`，并让 README「纹理映射」行由 🟡 转 ✅。
- **影响面**：`report`/WebUI 的 texture 阶段产物展示改为单文件 GLB；无测试断言 texture.png
  为 reconstruct 工作区必需产物（asset 库的 `texture.png` 属另一路径，不受影响）。
