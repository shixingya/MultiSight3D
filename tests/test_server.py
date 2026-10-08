"""WebUI 服务端接口测试：上传建任务 / 列表 / 详情 / 下载守卫 / 参数校验 / SSE 回放。"""

import io
import time

import pytest
from fastapi.testclient import TestClient

from multisight.server import app as appmod

pytest.importorskip("fastapi")


@pytest.fixture()
def client(tmp_path, photos_dir):
    appmod.set_data_dir(str(tmp_path / "ws"))
    app = appmod.create_app()
    with TestClient(app) as c:
        yield c, photos_dir


def _photo_files(photos_dir, n=3):
    files = []
    for p in sorted(photos_dir.iterdir())[:n]:
        files.append(("photos", (p.name, io.BytesIO(p.read_bytes()), "image/jpeg")))
    return files


def wait_done(c, task_id, timeout=60):
    """轮询至管线结束（mock 全链路本地 <2s；CI Windows runner 慢，留足余量）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = c.get(f"/api/tasks/{task_id}").json()
        if t["status"] in ("done", "failed"):
            return t
        time.sleep(0.2)
    raise AssertionError("任务超时未结束")


def test_health(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_and_complete(client):
    c, photos_dir = client
    r = c.post("/api/tasks", data={"preset": "fast"}, files=_photo_files(photos_dir))
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    detail = wait_done(c, task_id)
    assert detail["status"] == "done"
    assert detail["overall"] == 100          # 完成态总进度必须是 100（防双算回归）
    assert "texture/model.glb" in detail["artifacts"]

    # 列表可见
    tasks = c.get("/api/tasks").json()["tasks"]
    assert any(t["task_id"] == task_id for t in tasks)

    # GLB 下载且 magic 正确
    g = c.get(f"/api/tasks/{task_id}/file", params={"path": "texture/model.glb"})
    assert g.status_code == 200 and g.content[:4] == b"glTF"


def test_download_guards(client):
    c, photos_dir = client
    task_id = c.post("/api/tasks", data={"preset": "fast"},
                     files=_photo_files(photos_dir)).json()["task_id"]
    wait_done(c, task_id)
    assert c.get(f"/api/tasks/{task_id}/file", params={"path": "../secret"}).status_code == 404
    assert c.get(f"/api/tasks/{task_id}/file", params={"path": "manifest.json"}).status_code == 404  # 内部状态不对外
    assert c.get(f"/api/tasks/nope/file", params={"path": "sfm/cameras.txt"}).status_code == 404


def test_upload_validation(client):
    c, photos_dir = client
    assert c.post("/api/tasks", data={"preset": "ultra"},
                  files=_photo_files(photos_dir)).status_code == 400     # 非法档位
    assert c.post("/api/tasks", data={"preset": "fast"}).status_code == 422  # 缺 photos 字段


def test_sse_replay_for_finished_task(client):
    c, photos_dir = client
    task_id = c.post("/api/tasks", data={"preset": "fast"},
                     files=_photo_files(photos_dir)).json()["task_id"]
    wait_done(c, task_id)
    r = c.get(f"/api/tasks/{task_id}/events")
    assert r.status_code == 200
    body = r.text
    assert '"type": "progress"' in body and '"type": "end"' in body and '"ok": true' in body
    assert '"type": "stage_done"' in body  # 回放补发终止态，与实时链路视觉一致
