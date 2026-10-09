# MultiSight3D 架构说明

> 对应 PRD §9 的工程化细化。当前状态：**v0.1 骨架**（CLI + WebUI 双端可跑，mock 引擎端到端）。

## 1. 总览

```
照片输入 → [S1 preprocess] → [S2 sfm] → [S3 mvs] → [S4a mesh] → [S4b texture] → [report]
```

- **双形态同内核**：CLI（`multisight reconstruct`）与 WebUI（`multisight serve`）
  调用同一个 `pipeline.run_pipeline`；算法能力一律先落 CLI，再被 WebUI 复用（PRD §11-3 决策）。
- **引擎适配器**（沿用 Butian3D 经验）：每个阶段模块暴露 `Mock` / `Real` 两个类，
  `MS_ENGINE=mock|real` 或 `--engine` 切换。mock 保证零算法也能端到端演示与测试；
  Real 未落地时抛带里程碑指向的 `NotImplementedError`，绝不静默假成功。

## 2. 目录与模块边界

```
multisight/
├── cli.py            # argparse 子命令：reconstruct / report / stages / import-asset / serve
├── pipeline.py       # 状态机执行器：阶段编排、from-stage/resume、失败中断
├── events.py         # Bus：线程安全发布/订阅 + 历史回放（CLI 控制台 & SSE 共用）
├── workspace.py      # 工作区约定：manifest.json 原子写、产物清单、路径守卫
├── stages/
│   ├── __init__.py   # build_stage(name, engine) 注册表
│   ├── _base.py      # NotImplementedReal 占位基类
│   ├── _synth.py     # mock 合成产物：PLY/OBJ/GLB/贴图/COLMAP 目录写出器
│   └── preprocess.py / sfm.py / mvs.py / mesh.py / texture.py / report.py
├── assets/           # 外部成品模型读取线（与照片重建并行）：
│   │                 #   openflight(.flt 识别+校验几何) / spec(docx) / textures(TGA/DDS→PNG)
│   │                 #   glb(泛化 GLB 写出) / library(扫描·导入聚合) / demo(单文件 HTML)
│   └── __init__.py   # scan/import/load_asset_bundle · render_single_file_html 等出口
├── server/           # FastAPI：任务 CRUD + SSE + 产物下载 + 资产接口 + 静态页
└── webui/            # 无构建单页：index.html + app.js（three.js 走 importmap CDN）
```

**阶段间零代码耦合**：只通过「工作目录 + 标准工件」交接（PRD NFR-05），
后续任一 Real 算法落地只替换对应 `stages/<name>.py` 的 `Real` 类。

## 3. 工作区契约

每个任务一个目录 `workspace/<task_id>/`：

```
manifest.json         # 阶段状态机（pending/running/done/failed + progress + error），原子写
raw/                  # 原始上传/拷入照片
images/  list.json + 降采样后 jpg      # ← preprocess 产物
sfm/     cameras.txt images.txt points3D.ply [+stats.json]  # COLMAP 兼容目录 ← sfm
mvs/     fusion.ply                    # ← mvs
mesh/    mesh.obj                      # ← mesh
texture/ texture.png model.glb         # ← texture（GLB 单文件直出）
report/  report.json                   # ← report（聚合指标 + 警告 + 补救建议）
```

- `--from-stage X`：先校验 X 之前所有阶段 `artifact_ready`，缺件拒跑。
- `--resume`：定位 manifest 中第一个非 done 阶段续跑；失败不清空已完成产物（NFR-04）。
- `task_id` = 时间戳 + 随机 hex，仅由程序生成，天然防路径注入面。

## 4. 事件流与 SSE

```
stage.run ──progress()──▶ Bus.publish ──▶ CLI 控制台渲染线程
                        （history 回放）──▶ SSE /api/tasks/{id}/events
```

- 事件类型：`stage_start / progress / stage_done / stage_failed / end`。
- `Bus.subscribe(replay=True)`：浏览器刷新 / 断线重连不丢进度。
- 任务已结束或服务重启后：按 manifest 一次性回放（含补发 stage_done/stage_failed
  终止态，保证回放与实时链路视觉一致）后关闭流。
- EventSource 无法带 Header → 当前本地匿名可用；FR-16 账号体系落地时沿用
  Butian3D 的 `?token=` 查询参数方案。

## 5. Web API（v0.1）

| 方法 & 路径 | 说明 |
| --- | --- |
| `GET /api/health` | `{status, engine}` |
| `POST /api/tasks` | multipart `photos`（≤500 张、单张 ≤20MB）+ `preset`，后台线程起管线 |
| `GET /api/tasks` | manifest 扫描聚合的任务列表（重启可恢复） |
| `GET /api/tasks/{id}` | 摘要 + `artifacts` + `live` |
| `GET /api/tasks/{id}/events` | **SSE** 实时/回放进度 |
| `GET /api/tasks/{id}/file?path=` | 产物下载（`safe_relpath`：限定工作区内 + 后缀白名单；manifest.json 不对外） |
| `GET /api/assets` | 资源模型库列表（扫 `<assets_dir>/*/asset.json` 摘要：folder/name/display/sprite_count） |
| `GET /api/assets/gallery` | 资源模型画廊页（卡片网格 + 缩略图，`api=True` 链接走 `/demo` 与 `/file`） |
| `GET /api/assets/{folder}/demo` | 现场渲染单文件自包含 HTML（转盘/贴图/元数据，data URI 内联） |
| `GET /api/assets/{folder}/file?path=` | 资产归一化产物下载（`safe_relpath` 限定 assets 根内） |
| `GET /`、`GET /app.js` | WebUI 静态页 |

## 6. 前端选型：无构建 WebUI 壳

v0.1 用「单 HTML + ES Module + importmap(CDN three.js)」零构建方案：
Python 开发者 `pip install -e .[server]` 即可跑起完整 UI，避开 Node 工具链
（PRD 风险表「Windows 安装地狱」对策）。CDN 不可达时优雅降级为下载指引。
若后续 WebUI 复杂度上升（账号/工作区/多页），再评估迁 React/Vite（复用
Butian3D 前端资产：ReconScene 资产加载位、SSE token 方案）。

## 7. 资源模型读取子系统（外部成品模型）

与「照片 → 重建」主管线并行的一条能力线：直接读取美术资源库里的**成品模型**
（OpenFlight `.flt` / Performer `.ive` + TGA/DDS 贴图 + docx 技术说明 + 转盘 `sprites/`），
不必重新摄影测量即可展示。`multisight/assets/` 为纯读取/转换模块，产物只落 `assets_out/`，
**原始美术二进制不入库**（延续 NFR 零二进制理念）。

数据流：

```
资源目录 ──scan_asset_dir──▶ AssetBundle（meta/openflight/贴图/sprites/display 决策）
        ──import_asset_dir─▶ <assets_out>/<folder>/{asset.json, texture.png, sprites/frame_*.png, [model.glb]}
        ──render_single_file_html──▶ 单文件自包含 demo.html（base64 内联，双击即开、离线零网络）
```

关键设计取舍：
- **OpenFlight 是二进制记录流**（大端，每条 = u16 opcode + u16 length，4 字节对齐）；
  `describe_openflight` 解析 308 字节头 + 走记录流做 opcode 直方图，**总是**给出格式指纹。
- **Performer `.ive` 只给诚实指纹**：`describe_ive` 校验已知签名并报告大小，不做几何猜测
  （完整解析需 OpenSceneGraph/Performer）；与 `.flt` 一同列入元数据面板。
- **docx 技术说明零依赖提取**：zip 读 `word/document.xml` 按 `<w:p>` 切段，按中文字段标签取键值。
  表格“标题/值分行”，故坐标系取真正描述轴/手性的正文行（兼容描述行不含“坐标”二字的常见变体），
  并单独抽出“坐标原点…”行作 `origin`。贴图声明名兼容带/不带扩展两种写法（“贴图：feiji_c_01”无后缀也能
  抽出，并用于按 stem 优先匹配真实贴图文件，避免误选更大图）。（真实 52 型号：name/triangles/scale/坐标系/原点
  均 100% 命中，贴图声明 51/52——唯一遗漏是一个损坏的重复 docx。）
- **几何提取「宁缺毋滥」**：现实导出器（尤其 **Maya 的 OpenFlight 插件**）常产出非标准变体
  ——面/顶点记录被写成空壳模板。`extract_mesh` 仅在数量与包围盒都合理（≥`min_tris`、
  span∈(0.01, 200]m、拒绝全零顶点）时才组装网格，否则返回 `None`，上层据 `geometry_ok`
  干净回退到真实渲染转盘，**绝不 ship 错乱网格**。
- **展示模式自动决策**：`display = glb`（几何通过）> `turntable`（有 sprites）> `texture`。
  demo.html 把三种模式做成可切换标签；真三维 GLB 走 CDN three.js，离线时降级提示。
- CLI `import-asset --demo` 与 WebUI「资源模型库」面板（`/api/assets` + iframe 加载
  `/api/assets/{folder}/demo`）共用同一套 `assets` 内核。
- **批量导入 + 可托管画廊**：`import-asset --batch` 能处理两种层级——parent 直接挂型号（按大类批量），
  或 parent 为库根（`find_model_roots` 递归发现型号叶子，以顶层容器名为大类，目录名加 `大类-` 前缀防重名）；
  以包标记（docx/preview/structure）与组件目录黑名单区分“型号”与“三维模型文件/”等子目录；
  归入 `<assets_out>/<slug>/`；默认增量——导入时记录源目录签名（相对路径+字节数），
  未变化的子目录自动跳过重导（`--force` 强制全量）；`--demo` 时额外生成 `assets_out/index.html`（卡片网格画廊，
  相对路径引用同级 `<folder>/demo.html` 与 `<folder>/preview.png`）。整个 `assets_out/` 目录
  可直接丢到任意静态托管（如 GitHub Pages）或双击 `index.html` 打开——真正做到「无需部署、线上直接运行」。
  `multisight gallery -o assets_out` 可为已导入的库单独重生成画廊。画廊内置纯前端交互（搜索 / 按展示类型
  筛选 / 按分类筛选 / 按名称、帧数、面数排序 + 实时计数）；当前视图状态经 `history.replaceState` 写入 URL hash
  （并监听 `hashchange`），因此可分享带筛选的固定链接；全程零依赖零网络，保持目录站自包含。
- **批量导入逐型号容错**：整库无人值守导入时，单个型号包出错（磁盘/解码等意外）被就地隔离——该型号
  记为带“导入失败”警告的占位条目（不落 `asset.json`，故画廊会自动忽略），其余型号照常导入，整批不中断。
- **自研 SfM 内核（`multisight/sfm/`）**：遵循 PRD §11-1「完全自研」，仅依赖 numpy+Pillow（不引 OpenCV/scipy）。
  首件 `features`：Harris 角点（结构张量 + 非极大值抑制 + 间距去重）、去均值 L2 归一化的 patch 描述子、
  带 Lowe 比值测试 + 双向一致校验的暴力匹配；均经合成图单测（整数平移双图上精确恢复位移）。
  `geometry`：归一化 8 点法基础矩阵 + rank-2 约束 + RANSAC 对称极线距离剔外点，以及逐点 DLT 三角化（两视图投影
  矩阵→稀疏 3D 点）；由合成已知位姿双视图验证（极线约束残差、内点回收率、三角化复原精度）。

## 8. 与 PRD 里程碑映射

| 里程碑 | 交付 | 代码钩子（现已就位） |
| --- | --- | --- |
| v0.1 | 骨架 + 自研 SfM | `sfm/`（纯 numpy 特征提取+匹配已落地）；`stages/sfm.Real` 替换点；`_synth` 的 COLMAP 目录即目标落盘格式 |
| v0.2 | 学习型 MVS + PatchMatch 兜底 | `stages/mvs.Real`；设备探测（CUDA/MPS/CPU）选档在 pipeline.params 扩展 |
| v0.3 | 网格 + 纹理烘焙 | `mesh.Real` / `texture.Real`；`_synth.write_glb_*` 替换为真实烘焙产物 |
| v0.4 | 报告/归因/补拍建议/账号隔离 | `report` 已聚合指标与警告；FR-16 加 `server/routes` 鉴权层 |
| v1.0 | 打磨发布 + Docker | `Dockerfile` 待补；CI 双 OS 矩阵已就位 |

## 9. 测试策略

- `tests/conftest.py`：Pillow 程序化生成测试照片（零二进制入库，延续 Butian3D 理念）。
- 分层：workspace 状态机单测 / pipeline 端到端（mock 全链路、real 失败中断、resume、
  from-stage 前置校验、模糊检出警告）/ CLI 行为 / HTTP 接口（上传→轮询→下载→守卫→SSE 回放）。
- CI：`ubuntu × windows × py3.11/3.12` 矩阵 + CLI 冒烟（见 `.github/workflows/ci.yml`）。
