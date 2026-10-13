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
    assert '"type": "stage_done"' in body   # 回放补发终止态，与实时链路视觉一致


def test_task_detail_has_quality(client):
    c, photos_dir = client
    task_id = c.post("/api/tasks", data={"preset": "fast"},
                     files=_photo_files(photos_dir)).json()["task_id"]
    detail = wait_done(c, task_id)
    q = detail["quality"]
    assert q is not None and "warnings" in q and "advice" in q
    # 照片不足（<15）→ report 应给出补拍建议
    assert any("环形" in a or "补" in a for a in q["advice"])
    # FR-10 §5.3 模型统计：至少含照片数与 sfm 指标
    assert q["stats"]["photo_count"] == 3
    assert "register_rate" in q["stats"]


def test_report_endpoint(client):
    c, photos_dir = client
    task_id = c.post("/api/tasks", data={"preset": "fast"},
                     files=_photo_files(photos_dir)).json()["task_id"]
    wait_done(c, task_id)
    r = c.get(f"/api/tasks/{task_id}/report")
    assert r.status_code == 200
    rep = r.json()
    assert set(["warnings", "advice", "stages"]) <= set(rep)
    # 未知任务 → 404
    assert c.get("/api/tasks/nope/report").status_code == 404


def test_webui_quality_wiring(client):
    """前端契约回归：结果页质量卡片（FR-10）挂载点与渲染接线不得被误删。"""
    c, _ = client
    html = c.get("/").text
    assert 'id="quality"' in html and 'id="qbody"' in html and 'id="qstats"' in html
    js = c.get("/app.js")
    assert js.status_code == 200
    body = js.text
    assert "renderQuality" in body and "t.quality" in body   # 消费后端 quality 字段
    assert "q.warnings" in body and "q.advice" in body        # 渲染警告与建议
    assert "renderStats" in body and "q.stats" in body        # FR-10 模型统计渲染


def test_summarize_stats_helper():
    from multisight.server.app import _summarize_stats
    rep = {"photo_count": 30, "pipeline_duration_s": 12.5,
           "sfm": {"register_rate": 0.9, "sparse_points": 500},
           "mvs": {"dense_points": 8000},
           "mesh": {"faces": 4000, "watertight": False, "area": 12.3,
                    "volume_abs": 4.5, "bbox_extent": [1.0, 2.0, 3.0]},
           "texture": {"triangles": 4000}}
    s = _summarize_stats(rep)
    assert s["photo_count"] == 30 and s["pipeline_duration_s"] == 12.5
    assert s["mesh_faces"] == 4000 and s["glb_triangles"] == 4000
    assert s["mesh_area"] == 12.3 and s["mesh_bbox_extent"] == [1.0, 2.0, 3.0]
    # 空/缺字段诚实省略（不塞 None）
    s2 = _summarize_stats({"sfm": {}})
    assert "photo_count" not in s2 and "register_rate" not in s2
