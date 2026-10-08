"""工作区约定：workspace/<task_id>/ 目录结构 + manifest.json 断点续跑状态。

NFR-04/NFR-05：五阶段以「工作目录 + JSON 清单」交接，任一阶段可替换实现；
任一阶段失败不清空已完成产物，`--resume` 从 manifest 记录的断点继续。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# 阶段顺序即管线顺序（与 PRD §5.1 对应）
STAGES = ["preprocess", "sfm", "mvs", "mesh", "texture", "report"]

# 各阶段的标准产物（相对工作区根目录；report/WebUI 据此判断可下载内容）
ARTIFACTS: dict[str, list[str]] = {
    "preprocess": ["images/list.json", "images/"],
    "sfm": ["sfm/cameras.txt", "sfm/images.txt", "sfm/points3D.ply"],
    "mvs": ["mvs/fusion.ply"],
    "mesh": ["mesh/mesh.obj"],
    "texture": ["texture/texture.png", "texture/model.glb"],
    "report": ["report/report.json"],
}

# 允许通过 HTTP 下载的产物后缀（白名单，防目录穿越外的第二重守卫）
DOWNLOAD_SUFFIXES = {".json", ".txt", ".ply", ".obj", ".mtl", ".png", ".glb", ".jpg", ".jpeg"}

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def new_task_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class Workspace:
    """单个重建任务的工作目录封装。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    # ---------- 创建 / 加载 ----------

    @classmethod
    def create(cls, base: Path | str, task_id: str | None = None,
               params: dict[str, Any] | None = None) -> "Workspace":
        task_id = task_id or new_task_id()
        ws = cls(Path(base) / task_id)
        ws.root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "task_id": task_id,
            "created_at": _now(),
            "params": params or {},
            "stages": {name: {"status": STATUS_PENDING, "progress": 0} for name in STAGES},
        }
        ws._write_manifest(manifest)
        return ws

    @classmethod
    def open(cls, root: Path | str) -> "Workspace":
        ws = cls(root)
        if not ws.manifest_path.exists():
            raise FileNotFoundError(f"not a multisight workspace: {root}")
        return ws

    @property
    def task_id(self) -> str:
        return self.manifest["task_id"]

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    # ---------- 目录与产物 ----------

    def dir(self, *parts: str) -> Path:
        """返回工作区内子目录并确保存在。"""
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def file(self, *parts: str) -> Path:
        """返回工作区内文件路径（自动建父目录；不保证文件已存在）。"""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def artifact_ready(self, stage: str) -> bool:
        """阶段产物是否齐备（目录项只判存在且非空）。"""
        for rel in ARTIFACTS.get(stage, []):
            p = self.root / rel
            if rel.endswith("/"):
                if not p.is_dir() or not any(p.iterdir()):
                    return False
            elif not p.is_file():
                return False
        return True

    def list_artifacts(self) -> list[str]:
        """列出全部实际存在的标准产物（供 API/下载页）。"""
        out = []
        for stage in STAGES:
            for rel in ARTIFACTS.get(stage, []):
                if rel.endswith("/"):
                    continue
                if (self.root / rel).is_file():
                    out.append(rel)
        return out

    # ---------- manifest 状态更新（原子写） ----------

    def set_stage(self, stage: str, *, status: str | None = None,
                  progress: int | None = None, error: str | None = None) -> None:
        manifest = self.manifest
        entry = manifest["stages"][stage]
        if status:
            entry["status"] = status
            if status == STATUS_RUNNING:
                entry["started_at"] = _now()
            elif status in (STATUS_DONE, STATUS_FAILED):
                entry["finished_at"] = _now()
        if progress is not None:
            entry["progress"] = max(0, min(100, int(progress)))
        if error is not None:
            entry["error"] = error
        elif status == STATUS_RUNNING:
            entry.pop("error", None)
        self._write_manifest(manifest)

    def next_incomplete_stage(self) -> str | None:
        """--resume 用：返回第一个未完成的阶段名；全部完成返回 None。"""
        for name in STAGES:
            if self.manifest["stages"][name]["status"] != STATUS_DONE:
                return name
        return None

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """临时文件 + os.replace 原子写，进程中断不会留下半个 JSON。"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.manifest_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.manifest_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def discover_tasks(base: Path | str) -> list[Path]:
    """扫描基准目录下所有合法工作区（server 重启后恢复任务列表）。"""
    base = Path(base)
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file()
    )


def safe_relpath(root: Path, rel: str) -> Path | None:
    """把用户提供的相对路径解析到 root 内；越界/绝对路径/非白名单后缀返回 None。"""
    if not rel or rel.startswith(("/", "\\")) or ":" in rel:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() not in DOWNLOAD_SUFFIXES:
        return None
    return candidate


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
