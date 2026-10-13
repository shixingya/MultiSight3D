"""COLMAP 兼容目录读取：把自研 SfM 落盘的 cameras.txt/images.txt/points3D.ply 读回。

供 `stages/mvs.Real` 消费上游稀疏重建产物（阶段间零代码耦合、只经标准工件交接，PRD NFR-05）。
仅支持本项目写出的子集：SIMPLE_PINHOLE 相机 + 世界→相机 (q wxyz, t) 位姿 + ASCII xyz PLY。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _data_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="ascii", errors="ignore").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def read_shared_K(cameras_path: Path) -> tuple[np.ndarray, int, int]:
    """从 cameras.txt 取首个 SIMPLE_PINHOLE 内参 → (K 3×3, width, height)。"""
    for ln in _data_lines(cameras_path):
        p = ln.split()
        if p[1] != "SIMPLE_PINHOLE":
            raise ValueError(f"暂不支持相机模型 {p[1]}（仅 SIMPLE_PINHOLE）")
        w, h = int(p[2]), int(p[3])
        f, cx, cy = float(p[4]), float(p[5]), float(p[6])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        return K, w, h
    raise ValueError(f"{cameras_path} 无有效相机行")


def read_images(images_path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """解析 images.txt → [(name, R(3×3), t(3,))]，R 由四元数还原（世界→相机）。"""
    from ..sfm import quat_to_matrix
    lines = _data_lines(images_path)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(lines):
        p = lines[i].split()
        i += 1
        if len(p) < 10:                                  # 仅位姿行有 10 列；"-1" 观测行跳过
            continue
        qw, qx, qy, qz = (float(x) for x in p[1:5])
        t = np.array([float(x) for x in p[5:8]])
        name = p[9]
        out.append((name, quat_to_matrix((qw, qx, qy, qz)), t))
    return out


def read_ply_xyz(ply_path: Path) -> np.ndarray:
    """读 ASCII PLY 的 xyz → (N,3)。头部之外仅取前三个 float 列。"""
    if not ply_path.is_file():
        return np.zeros((0, 3))
    n = 0
    header_done = False
    pts: list[list[float]] = []
    for ln in ply_path.read_text(encoding="ascii", errors="ignore").splitlines():
        if not header_done:
            s = ln.strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s == "end_header":
                header_done = True
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(pts) >= n:
            break
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))
