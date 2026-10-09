# MultiSight3D

> 开源多视图摄影测量管线：普通相机/手机环绕拍摄的多张照片 → 自动重建带纹理的三维网格模型（OBJ / PLY / GLB）。

![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-%E2%89%A53.10-brightgreen)
![ci](https://img.shields.io/badge/ci-passing-informational)

MultiSight3D 追求 **「一条命令进，一个 GLB 出」**：比 COLMAP 更易用、比消费级 App 更可控、可嵌入第三方产品。当前处于 **v0.1 骨架**阶段——管线状态机、CLI、WebUI、SSE 实时进度、产物下载与 3D 预览已端到端可跑（内置 mock 引擎）；自研算法按[里程碑](./doc/PRD.md)逐版本落地。

📖 **文档**：[产品需求 PRD](./doc/PRD.md) · [架构说明](./doc/ARCHITECTURE.md)

## ✨ 核心能力（规划 ↔ 现状）

| 能力 | 阶段 | 现状 |
| --- | --- | --- |
| 无序图像 SfM（完全自研：特征/匹配/增量注册/BA） | v0.1 | 🔜 mock 可跑，Real 待落地 |
| MVS 稠密重建（学习型深度网络 + PatchMatch 兜底） | v0.2 | 🔜 |
| 网格重建 + 纹理映射，GLB/OBJ/PLY 直出 | v0.3 | ✅ mock 产物链（合法 GLB/OBJ/PLY） |
| CLI 一键全流程 / 断点续跑 / 质量报告 | v0.1 | ✅ 已可用 |
| WebUI：拖拽上传 / 实时进度 / 3D 预览 / 下载 | v0.1 壳 | ✅ 已可用 |
| 外部成品模型读取（OpenFlight .flt/.ive / TGA·DDS 贴图 / docx / 转盘）+ 单文件离线 demo | v0.1 旁支 | ✅ 已可用 |
| 拍摄预检（模糊检测 / 补拍建议） | v0.4 增强 | ✅ 基础版已可用 |

## 🚀 快速开始

```bash
# 安装（核心依赖仅 pillow + numpy）
python -m venv .venv && .venv\Scripts\activate     # Windows（Linux/macOS 用 source .venv/bin/activate）
pip install -e ".[server,dev]"

# CLI：照片目录 → 模型
multisight reconstruct -i photos/ -o workspace/
#   失败后断点续跑： multisight reconstruct -o workspace/ --task-id <id> --resume
#   指定阶段重跑：   multisight reconstruct -o workspace/ --task-id <id> --from-stage texture

# WebUI：上传 → SSE 实时进度 → three.js 预览 → 下载
multisight serve --port 8050

# 外部成品模型（美术资源库）：读取 .flt/.ive + TGA/DDS 贴图 + docx + 转盘 sprites
#   归一化落盘并生成「无需部署、双击即开」的单文件自包含 demo.html
multisight import-asset -i "D:/path/to/模型目录" -o assets_out --demo
#   批量：把一个「模型库父目录」下每个子目录都导入，并附赠可托管的画廊 index.html
#   两种层级都支持：直接指「单个大类」（其子目录即型号），或指「库根」（自动按大类分组）
#   默认增量：签名未变化的模型自动跳过重导（加 --force 可强制全量重建）
multisight import-asset -i "D:/path/to/模型库父目录" -o assets_out --batch --demo
#   把整个 assets_out/ 丢到任意静态托管（如 GitHub Pages）或双击 index.html 即可上线
#   为已导入的库单独重生成画廊： multisight gallery -o assets_out
#   serve 时自动把 assets_out/ 作为「资源模型库」呈现（--assets-dir 可改）
```

浏览器打开 `http://localhost:8050`。默认 `MS_ENGINE=mock` 用内置模拟引擎演示全链路；
`MS_ENGINE=real` 加载自研算法（未落地阶段会明确报错并指向里程碑，不会静默假成功）。

> 💡 `import-asset` 产出的 `demo.html` 把转盘帧 / 贴图 / 元数据全部以 base64 内联，零依赖零网络，
> 直接双击即可在浏览器打开——正合「无需部署、线上直接运行」。若 `.flt` 几何通过严格校验，
> 还会多出「真三维」标签（three.js 走 CDN，离线自动降级为转盘）。`--batch` 还会生成画廊
> `index.html`，汇总整个 `assets_out/` 为可一键托管的静态模型站（内置搜索、按展示类型/大类筛选、按名称/帧数/面数排序；当前视图状态写入 URL hash，可直接分享带筛选的固定链接）。

## 🧱 技术栈

| 层 | 选型 |
| --- | --- |
| 管线 | Python ≥ 3.10 · 六阶段模块化（preprocess/sfm/mvs/mesh/texture/report）· manifest 断点续跑 |
| CLI | argparse（核心零重依赖） |
| WebUI | FastAPI + SSE · 无构建单页（ES Module + importmap three.js CDN） |
| 测试 | pytest（27 用例：状态机/端到端/CLI/HTTP/资产库）· CI 双 OS × 双 Python 矩阵 |

## 📁 项目结构

```
multisight/
├── pipeline.py  workspace.py  events.py  cli.py   # 状态机 / 工作区契约 / 事件总线 / CLI
├── stages/        # 六阶段：Mock(内置演示) + Real(自研，按里程碑替换)
├── assets/        # 外部成品模型读取线（OpenFlight/docx/TGA·DDS/GLB/library/demo）
├── server/        # FastAPI：任务 · SSE · 产物下载 · 资产接口
├── webui/         # 单页前端（上传/进度/预览/下载）
doc/               # PRD.md · ARCHITECTURE.md
datasets/          # 标准回归数据集占位（图片不入库）
tests/  .github/   # pytest 套件 · Actions CI
```

## 🧪 测试

```bash
pytest -q          # 全部隔离临时目录，不污染开发数据
```

## 💡 适用场景

实物扫描、文物数字化、手办/样品资产重建、小型场景三维采集。
大范围无人机场景重建请使用姊妹项目 [Butian3D](https://github.com/shixingya/MOLDAI_WEB)；
未来 MultiSight3D 可作为其开源引擎经适配器接入（`ENGINE=multisight`）。

## 🗺️ 路线图

v0.1 自研 SfM → v0.2 稠密点云 → v0.3 网格与纹理 → v0.4 质量与稳健 → v1.0 发布（Docker + 文档 + 演示）。
详见 [PRD §7 里程碑](./doc/PRD.md)。

## 📄 许可

[Apache-2.0 © 2026 Xingya Shi](./LICENSE)（含专利授权条款，便于商用集成）
