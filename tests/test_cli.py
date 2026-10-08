"""CLI 行为测试：一键全流程 / resume 参数校验 / stages 列表。"""

from multisight.cli import main
from multisight.workspace import Workspace, discover_tasks


def test_reconstruct_end_to_end(tmp_path, photos_dir, capsys):
    out = tmp_path / "ws"
    code = main(["reconstruct", "-i", str(photos_dir), "-o", str(out),
                 "--preset", "fast", "--engine", "mock"])
    assert code == 0
    tasks = discover_tasks(out)
    assert len(tasks) == 1
    ws = Workspace.open(tasks[0])
    assert (ws.root / "texture" / "model.glb").is_file()
    printed = capsys.readouterr().out
    assert "完成" in printed and "model.glb" in printed


def test_reconstruct_with_task_id_and_resume(tmp_path, photos_dir):
    out = tmp_path / "ws"
    assert main(["reconstruct", "-i", str(photos_dir), "-o", str(out),
                 "--task-id", "fixed1", "--engine", "real"]) == 1     # sfm 未落地 → 失败
    assert main(["reconstruct", "-o", str(out), "--task-id", "fixed1",
                 "--resume", "--engine", "mock"]) == 0                # 续跑成功
    ws = Workspace.open(out / "fixed1")
    assert all(ws.manifest["stages"][s]["status"] == "done" for s in ws.manifest["stages"])


def test_reconstruct_input_validation(tmp_path, photos_dir, capsys):
    assert main(["reconstruct", "-o", str(tmp_path / "x")]) == 2          # 缺 -i
    assert main(["reconstruct", "-i", str(tmp_path / "nope"), "-o", str(tmp_path)]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["reconstruct", "-i", str(empty), "-o", str(tmp_path / "y")]) == 2


def test_stages_lists_all(capsys):
    assert main(["stages"]) == 0
    out = capsys.readouterr().out
    for s in ("preprocess", "sfm", "mvs", "mesh", "texture", "report"):
        assert s in out
