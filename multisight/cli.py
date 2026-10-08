"""MultiSight3D CLI（FR-01）：一条命令进，一个 GLB 出。

零第三方依赖（argparse），保证 pip 装完核心包即可用：
    multisight reconstruct -i photos/ -o workspace/
    multisight reconstruct --task-id <id> --resume
    multisight serve --port 8000
    multisight report --task-id <id>
    multisight stages
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
from pathlib import Path

from .events import Bus
from .pipeline import PRESETS, run_pipeline
from .workspace import STAGES, Workspace


def _console_listener(bus: Bus) -> None:
    """把管线事件渲染到控制台（带阶段前缀，格式对齐 WebUI 时间线）。"""

    def render(event: dict) -> None:
        et = event["type"]
        if et == "progress":
            print(f"  [{event['stage']:<10}] {event['percent']:>3}% {event.get('message', '')}")
        elif et == "stage_start":
            print(f"▶ 阶段开始：{event['stage']}")
        elif et == "stage_done":
            print(f"✔ 阶段完成：{event['stage']}")
        elif et == "stage_failed":
            print(f"✘ 阶段失败：{event['stage']} — {event['error']}", file=sys.stderr)

    q = bus.subscribe(replay=True)  # 回放防线程启动晚于首事件；控制台重复打印可接受
    while True:
        event = q.get()
        render(event)
        if event["type"] == "end":
            bus.unsubscribe(q)
            return bool(event.get("ok"))


def cmd_reconstruct(args: argparse.Namespace) -> int:
    base = Path(args.output)
    if args.resume:
        if not args.task_id:
            print("--resume 需要配合 --task-id 指定已存在的工作区", file=sys.stderr)
            return 2
        ws = Workspace.open(base / args.task_id)
    else:
        if not args.input:
            print("首次运行需要 -i/--input 指定照片目录", file=sys.stderr)
            return 2
        src = Path(args.input)
        if not src.is_dir():
            print(f"输入目录不存在：{src}", file=sys.stderr)
            return 2
        params = {"preset": args.preset, "input": str(src)}
        ws = Workspace.create(base, task_id=args.task_id, params=params)
        raw = ws.dir("raw")
        photos = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not photos:
            print(f"输入目录没有 jpg/png 照片：{src}", file=sys.stderr)
            return 2
        for p in photos:
            shutil.copy2(p, raw / p.name)
        print(f"任务 {ws.task_id}：已收 {len(photos)} 张照片 → {ws.root}")

    bus = Bus()
    listener = threading.Thread(target=_console_listener, args=(bus,), daemon=True)
    listener.start()
    ok = run_pipeline(ws, from_stage=args.from_stage, resume=args.resume,
                      bus=bus, engine=args.engine)
    listener.join(timeout=3)
    if ok:
        print(f"\n完成 ✅  产物目录：{ws.root}")
        for rel in ws.list_artifacts():
            print(f"  - {rel}")
        return 0
    print(f"\n管线中断 ❌  已完成产物保留在 {ws.root}；"
          f"修复输入后运行：multisight reconstruct -o {base} --task-id {ws.task_id} --resume")
    return 1


def cmd_report(args: argparse.Namespace) -> int:
    ws = Workspace.open(Path(args.output) / args.task_id)
    path = ws.root / "report" / "report.json"
    if not path.is_file():
        print("任务尚未产出报告（管线未跑完）", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_import_asset(args: argparse.Namespace) -> int:
    """读取外部资源模型目录（OpenFlight/TGA/docx/sprites），归一化落盘并可生成单文件 demo。

    --output 是「资产库根目录」：每个资源按源目录名归入 <output>/<folder>/，
    与 server /api/assets 的 <folder>/asset.json 约定一致。
    """
    import re
    from .assets import import_asset_dir, build_single_file_html
    src = Path(args.input)
    if not src.is_dir():
        print(f"资源目录不存在：{src}", file=sys.stderr)
        return 2
    slug = re.sub(r'[\\/:*?"<>|]', "_", src.name).strip() or "asset"
    out = Path(args.output) / slug
    bundle = import_asset_dir(src, out)
    print(f"资源模型：{bundle.name}")
    print(f"  展示模式：{bundle.display}  几何校验：{'通过' if bundle.geometry_ok else '未通过（回退真实渲染）'}")
    if bundle.meta.get("triangles"):
        print(f"  技术说明：{bundle.meta['triangles']} 三角面  格式 {bundle.meta.get('format', '-')}")
    if bundle.openflight.get("version_str"):
        print(f"  OpenFlight：v{bundle.openflight['version_str']}  单位 {bundle.openflight.get('vertex_unit', '-')}")
    print(f"  贴图：{'✓' if bundle.texture_png else '✗'}  转盘帧：{len(bundle.sprites)}  GLB：{bundle.glb or '—'}")
    print(f"  产物目录：{out}")
    if args.demo:
        demo_path = out / "demo.html"
        build_single_file_html(bundle, demo_path)
        print(f"\n单文件 demo（双击即开，离线可运行）：{demo_path}")
    return 0


def cmd_stages(args: argparse.Namespace) -> int:
    for i, name in enumerate(STAGES, 1):
        print(f"{i}. {name}")
    print(f"\n预设档位：{', '.join(PRESETS)}")
    print("引擎：MS_ENGINE=mock（内置模拟，端到端可跑）| real（自研算法，按里程碑落地）")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from .server.app import create_app, set_data_dir, set_assets_dir
    except ImportError:
        print("WebUI 需要 server 依赖：pip install -e '.[server]'", file=sys.stderr)
        return 1
    set_data_dir(args.data_dir)
    set_assets_dir(args.assets_dir)
    app = create_app()
    print(f"MultiSight3D WebUI → http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multisight",
                                description="MultiSight3D — 多视图摄影测量管线")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reconstruct", help="一键全流程：照片目录 → 纹理网格")
    r.add_argument("-i", "--input", help="照片目录（首次运行必填）")
    r.add_argument("-o", "--output", default="workspace", help="工作区根目录（默认 workspace/）")
    r.add_argument("--task-id", help="指定/继续任务 ID")
    r.add_argument("--preset", choices=list(PRESETS), default="standard", help="质量档位")
    r.add_argument("--engine", choices=["mock", "real"], default=None, help="引擎（默认取 MS_ENGINE）")
    r.add_argument("--from-stage", choices=STAGES, help="从指定阶段起跑（前置产物需已存在）")
    r.add_argument("--resume", action="store_true", help="从 manifest 断点续跑")
    r.set_defaults(func=cmd_reconstruct)

    rp = sub.add_parser("report", help="打印任务质量报告 JSON")
    rp.add_argument("--output", default="workspace")
    rp.add_argument("--task-id", required=True)
    rp.set_defaults(func=cmd_report)

    st = sub.add_parser("stages", help="列出管线阶段与档位")
    st.set_defaults(func=cmd_stages)

    ia = sub.add_parser("import-asset", help="读取外部资源模型目录（OpenFlight/TGA/docx/sprites）")
    ia.add_argument("-i", "--input", required=True, help="资源模型目录")
    ia.add_argument("-o", "--output", default="assets_out", help="归一化产物输出目录")
    ia.add_argument("--demo", action="store_true", help="额外生成单文件自包含 HTML demo")
    ia.set_defaults(func=cmd_import_asset)

    sv = sub.add_parser("serve", help="启动 WebUI（上传/任务/SSE 进度/预览）")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--data-dir", default="workspace", help="任务工作区根目录")
    sv.add_argument("--assets-dir", default="assets_out", help="资源模型库目录（import-asset 产物）")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
