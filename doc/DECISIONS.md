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

## D-002（v0.3）GLB 不做 Y-up 轴变换，保留 SfM 世界系 + 输出 bbox 尺寸估计

- **背景**：FR-08 要求输出「坐标系 Y-up 归一化」。但本管线的 SfM 世界系由首帧相机定义，
  **不含重力/朝向上对齐**（无 IMU、未从消失点或 EXIF 铅垂线估 up），因而「哪个轴是 up」
  本身未定义。此时盲目做 `(x,y,z)->(x,z,-y)` 固定轴交换只会把一个任意朝向换成另一个任意
  朝向，不具真实语义、也不诚实。
- **决策**：`stages/texture.Real` 写 GLB 时传 `z_up=False`（保持世界系原样），同时在 `texture/stats.json`
  输出 `bbox_min/bbox_max/bbox_extent` 作为世界单位的**尺寸估计**（服务结果页“模型统计/尺寸估计”）。
- **理由**：宁可不翻转也不伪造对齐；尺度/朝向未知时用包围盒给客观可量尺寸，与“不臆造”的诚实原则一致。
- **重新启用条件**：当 SfM/预处理能估计真实 up（重力对齐、多视图垂直消失点或已知拍摄姿态）时，
  在 `write_glb` 处传 `z_up=True` 做真正的 Y-up 归一化（可选按最大面归一尺度）。
- **影响面**：three.js 预览按包围盒自动居中/调相机，不依赖固定 up；不影响下载产物合法性。
