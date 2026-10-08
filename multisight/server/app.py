"""FastAPI 应用：任务 CRUD + SSE 实时进度 + 产物下载 + WebUI 静态页。

安全基线（沿用 Butian3D 路由层经验）：
- 下载走 safe_relpath 双重守卫（路径解析限定工作区内 + 后缀白名单）；
- 上传限制张数与单文件大小；文件名统一 sanitize；
- task_id 仅由本服务生成（时间戳+hex），不接受用户构造路径。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..pipeline import PRESETS, DEFAULT_ENGINE
from ..workspace import safe_relpath
from .store import TaskManager, summarize

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"

MAX_FILES = 500                 # PRD §5.3：单任务 ≤500 张
MAX_FILE_BYTES = 20 * 1024 * 1024   # 单张 ≤20MB

_data_dir = os.environ.get("MS_DATA_DIR", "workspace")


def set_data_dir(path: str) -> None:
    global _data_dir
    _data_dir = path


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="MultiSight3D", version="0.1.0")
    tm = TaskManager(_data_dir)

    # ---------- 健康与元信息 ----------

    @app.get("/api/health")
    def health():
        return {"status": "ok", "engine": DEFAULT_ENGINE}

    # ---------- 任务 ----------

    @app.post("/api/tasks")
    async def create_task(preset: str = Form("standard"),
                          photos: list[UploadFile] = File(...)):
        if preset not in PRESETS:
            raise HTTPException(400, f"unknown preset: {preset}")
        if not photos or len(photos) > MAX_FILES:
            raise HTTPException(400, f"照片数量需在 1–{MAX_FILES} 张之间")
        payload: list[tuple[str, bytes]] = []
        for f in photos:
            data = await f.read()
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{f.filename} 超过 20MB 限制")
            if data:
                payload.append((f.filename or "photo.jpg", data))
        if not payload:
            raise HTTPException(400, "没有有效的上传内容")
        ws = tm.create_task(payload, preset=preset)
        return {"task_id": ws.task_id}

    @app.get("/api/tasks")
    def list_tasks():
        return {"tasks": tm.list_tasks()}

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        ws = tm.workspace(task_id)
        if ws is None:
            raise HTTPException(404, "task_not_found")
        manifest = ws.manifest
        return {**summarize(manifest),
                "artifacts": ws.list_artifacts(),
                "live": tm.live(task_id) is not None}

    # ---------- SSE 实时进度 ----------

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str, request: Request):
        ws = tm.workspace(task_id)
        if ws is None:
            raise HTTPException(404, "task_not_found")
        entry = tm.live(task_id)

        if entry is None:
            # 已结束/重启后：按 manifest 一次性回放阶段状态后关闭
            manifest = ws.manifest
            async def finished_stream():
                for stage, info in manifest["stages"].items():
                    yield _sse({"type": "progress", "stage": stage,
                                "percent": info.get("progress", 0),
                                "message": info.get("status", "")})
                    # 补发终止态事件，保证回放与实时链路的视觉状态一致（done 绿色/failed 红色）
                    if info.get("status") == "done":
                        yield _sse({"type": "stage_done", "stage": stage})
                    elif info.get("status") == "failed":
                        yield _sse({"type": "stage_failed", "stage": stage,
                                    "error": info.get("error", "")})
                yield _sse({"type": "end",
                            "ok": all(i["status"] == "done"
                                      for i in manifest["stages"].values())})
            return StreamingResponse(finished_stream(), media_type="text/event-stream")

        bus = entry["bus"]

        async def live_stream():
            q = bus.subscribe(replay=True)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.3)
                        yield ": ping\n\n"
                        continue
                    yield _sse(event)
                    if event["type"] == "end":
                        break
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(live_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------- 产物下载 ----------

    @app.get("/api/tasks/{task_id}/file")
    def download(task_id: str, path: str):
        ws = tm.workspace(task_id)
        if ws is None:
            raise HTTPException(404, "task_not_found")
        if path == "manifest.json":
            raise HTTPException(404, "artifact_not_found")   # 内部状态文件不走文件接口
        target = safe_relpath(ws.root, path)
        if target is None or not target.is_file():
            raise HTTPException(404, "artifact_not_found")
        return FileResponse(target, filename=target.name)

    # ---------- WebUI 静态页 ----------

    @app.get("/")
    def index():
        page = WEBUI_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(404, "webui missing")
        return FileResponse(page)

    @app.get("/app.js")
    def app_js():
        js = WEBUI_DIR / "app.js"
        if not js.is_file():
            raise HTTPException(404, "webui missing")
        return FileResponse(js, media_type="application/javascript")

    return app
