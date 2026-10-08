# 标准测试数据集（NFR-08）

本目录规划存放 3 组回归基准数据集。**图片不入库**（保持仓库轻量），
每组按以下结构手动放入或脚本拉取：

```
datasets/
├── small-object/     # T1 小物件（钥匙/工具类，硬表面 + 中等纹理）
│   ├── photos/       #   30–60 张环绕拍摄
│   └── truth.json    #   参照物真实尺寸（mm），用于尺寸误差评估
├── textured-doll/    # T2 纹理丰富玩偶（软表面，考验稠密重建细节）
│   ├── photos/
│   └── truth.json
└── ceramic-multipaint/ # T3 多色陶瓷（高反光风险样本，考验纹理烘焙与失败归因）
    ├── photos/
    └── truth.json
```

## 用途
- `pytest` 常规用例不依赖本目录（测试照片由 `tests/conftest.py` 程序化生成）。
- 算法回归脚本（v0.4 交付）将跑满三组数据并输出：注册率 / 重投影误差 /
  表面覆盖率 / 尺寸误差，指标口径见 `doc/PRD.md` §8。
- 拍摄规范见 `doc/SHOOTING_GUIDE.md`（v0.4 随补拍建议 FR-11 一并交付）。

## 许可提示
若引入公开数据集（如 DTU、Tanks and Temples 子集），须逐一核对许可并
在 `LICENSE-NOTICE` 中登记，避免影响 Apache-2.0 商用集成边界（NFR-06）。
