"""任务管理器：后台线程跑管线，内存活动注册表 + manifest 落盘（重启可恢复列表）。"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Iterable

from ..events import Bus
from ..pipeline import run_pipeline
from ..workspace import STAGES, Workspace, discover_tasks

_SAFE_NAME = re.compile(r"[^\w.\-]+")


def sanitize_name(name: str) -> str:
    return _SAFE_NAME.sub("_", name) or "photo.jpg"


def summarize(manifest: dict) -> dict:
    """把 manifest 压缩为列表页需要的摘要（含总体百分比）。"""
    stages = manifest["stages"]
    done = sum(1 for s in STAGES if stages[s]["status"] == "done")
    # 总体进度 = 各阶段 progress 均值（done 阶段已由 run_pipeline 置 100，不另加计数）
    overall = int(sum(stages[s].get("progress", 0) for s in STAGES) / len(STAGES))
    statuses = [stages[s]["status"] for s in STAGES]
    if any(v == "running" for v in statuses):
        status = "running"
    elif done == len(STAGES):
        status = "done"
    elif any(v == "failed" for v in statuses):
        status = "failed"
    else:
        status = "queued"  # 全部 pending（后台线程尚未推进）或混合进度
    return {
        "task_id": manifest["task_id"],
        "created_at": manifest.get("created_at"),
        "params": manifest.get("params", {}),
        "status": status,
        "overall": min(99, overall) if status == "running" else overall,
        "stages": {s: {k: stages[s][k] for k in ("status", "progress") if k in stages[s]}
                   for s in STAGES},
    }


class TaskManager:
    def __init__(self, base_dir: Path | str) -> None:
        self.base = Path(base_dir)
        self._live: dict[str, dict] = {}     # task_id -> {bus, thread}
        self._lock = threading.Lock()

    # ---------- 查询 ----------

    def list_tasks(self) -> list[dict]:
        tasks = [summarize(Workspace.open(root).manifest) for root in discover_tasks(self.base)]
        tasks.sort(key=lambda t: t.get("created_at") or "", reverse=True)
        return tasks

    def workspace(self, task_id: str) -> Workspace | None:
        root = self.base / task_id
        if not (root / "manifest.json").is_file():
            return None
        return Workspace.open(root)

    def live(self, task_id: str) -> dict | None:
        with self._lock:
            entry = self._live.get(task_id)
        if entry and entry["thread"].is_alive():
            return entry
        return None

    # ---------- 创建与执行 ----------

    def create_task(self, files: Iterable[tuple[str, bytes]], preset: str = "standard") -> Workspace:
        """files: (原始文件名, 内容字节) 序列；落盘 raw/ 后异步启动管线。"""
        ws = Workspace.create(self.base, params={"preset": preset, "source": "webui"})
        raw = ws.dir("raw")
        for name, data in files:
            (raw / sanitize_name(name)).write_bytes(data)

        bus = Bus()
        entry = {"bus": bus}
        runner = threading.Thread(target=self._run, args=(ws, bus), daemon=True)
        entry["thread"] = runner
        with self._lock:
            self._live[ws.task_id] = entry
        runner.start()
        return ws

    @staticmethod
    def _run(ws: Workspace, bus: Bus) -> None:
        try:
            run_pipeline(ws, bus=bus)
        except Exception:  # noqa: BLE001 - 后台线程兜底，异常已体现在 manifest
            pass
