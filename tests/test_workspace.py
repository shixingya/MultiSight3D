"""工作区约定与断点续跑状态测试（NFR-04/05）。"""

from multisight.workspace import (STATUS_DONE, STATUS_FAILED, STATUS_RUNNING,
                                  Workspace, discover_tasks, safe_relpath,
                                  new_task_id)


def test_create_and_manifest_lifecycle(tmp_path):
    ws = Workspace.create(tmp_path, task_id="t1", params={"preset": "fast"})
    m = ws.manifest
    assert m["task_id"] == "t1"
    assert all(m["stages"][s]["status"] == "pending" for s in m["stages"])

    ws.set_stage("preprocess", status=STATUS_RUNNING)
    ws.set_stage("preprocess", progress=42)
    m = ws.manifest
    assert m["stages"]["preprocess"]["status"] == "running"
    assert m["stages"]["preprocess"]["progress"] == 42

    ws.set_stage("preprocess", status=STATUS_DONE)
    assert ws.next_incomplete_stage() == "sfm"


def test_failure_keeps_error_and_resume_point(tmp_path):
    ws = Workspace.create(tmp_path)
    ws.set_stage("preprocess", status=STATUS_DONE, progress=100)
    ws.set_stage("sfm", status=STATUS_FAILED, error="boom")
    m = ws.manifest
    assert m["stages"]["sfm"]["error"] == "boom"
    # resume 定位到第一个未完成阶段（失败的 sfm）
    assert ws.next_incomplete_stage() == "sfm"
    # 阶段产物未清：preprocess 仍为 done
    assert m["stages"]["preprocess"]["status"] == "done"


def test_artifact_ready(tmp_path):
    ws = Workspace.create(tmp_path)
    assert not ws.artifact_ready("preprocess")
    (ws.dir("images") / "x.jpg").write_bytes(b"x")
    (ws.file("images", "list.json")).write_text("{}", encoding="utf-8")
    assert ws.artifact_ready("preprocess")


def test_discover_tasks(tmp_path):
    Workspace.create(tmp_path, task_id="a")
    Workspace.create(tmp_path, task_id="b")
    (tmp_path / "notatask").mkdir()
    ids = [p.name for p in discover_tasks(tmp_path)]
    assert sorted(ids) == ["a", "b"]


def test_safe_relpath_guards(tmp_path):
    root = tmp_path / "ws"
    (root / "texture").mkdir(parents=True)
    (root / "texture" / "model.glb").write_bytes(b"x")
    (root / "secret.env").write_text("x", encoding="utf-8")
    assert safe_relpath(root, "texture/model.glb") is not None
    assert safe_relpath(root, "../escape.txt") is None          # 目录穿越
    assert safe_relpath(root, "C:/Windows/x.txt") is None       # 绝对路径
    assert safe_relpath(root, "secret.env") is None             # 后缀白名单外
    assert safe_relpath(root, "") is None


def test_new_task_id_unique():
    assert new_task_id() != new_task_id()
