"""S3 MVS 稠密重建（FR-05/12）——决策：学习型深度网络首发 + PatchMatch 兜底（PRD §11-2），v0.2。

Mock：按稀疏点数十倍合成稠密球面点云 fusion.ply（打通断点续跑/SSE/下载全链路）。
Real：自研传统 MVS 基线（`multisight/mvs`，纯 numpy、CPU 可跑）作为管线真实一步——
读上游 SfM 的 COLMAP 位姿/稀疏点，为每个参考视图做平面扫描深度图、反投影并按置信度/代价
过滤，跨视图融合去重成稠密点云 fusion.ply。诚实原则：视图不足/无稀疏点定标深度区间即明确
报错（不静默假成功）；低置信/高代价/遮挡像素一律丢弃，绝不臆造深度点。学习型/PatchMatch
高档为后续里程碑（本件为 CPU 兜底基线）。
"""

from __future__ import annotations

import json

import numpy as np

from ..pipeline import Stage, StageContext
from ..mvs import plane_sweep_depth, depth_map_to_points, remove_scatter_noise
from ..mvs.colmap_io import read_images, read_ply_xyz, read_shared_K
from ._synth import sphere_points, write_ply_ascii


class Mock(Stage):
    name = "mvs"

    def run(self, ctx: StageContext) -> None:
        ctx.progress(self.name, 10, "深度图估计（mock：分块合成）")
        sparse_path = ctx.ws.root / "sfm" / "points3D.ply"
        base_n = 0
        if sparse_path.is_file():
            for line in sparse_path.read_text(encoding="ascii").splitlines():
                if line.startswith("element vertex"):
                    base_n = int(line.split()[-1])
                    break
        dense_n = max(2000, base_n * 10)

        points = sphere_points(dense_n, radius=1.0, seed=7)
        # 按高度着色，让预览/下载的点云有可辨视觉内容
        z01 = (points[:, 2] - points[:, 2].min()) / max(1e-6, np.ptp(points[:, 2]))
        colors = np.stack([z01 * 200 + 30, 250 - z01 * 160, (1 - z01) * 220 + 30], axis=1).astype(np.uint8)
        ctx.progress(self.name, 60, f"多视图融合与去噪（mock）：目标 {dense_n} 点")
        write_ply_ascii(ctx.ws.file("mvs", "fusion.ply"), points, colors)
        ctx.progress(self.name, 100, f"稠密点云完成：{dense_n} 点")


class Real(Stage):
    """自研传统 MVS：读 SfM 位姿→逐视图平面扫描深度图→反投影融合去重→稠密真彩色 fusion.ply。"""

    name = "mvs"

    MAX_VIEWS = 6      # 至多把前 N 个注册视图作为参考（控制无人值守成本）
    MAX_NB = 3         # 每个参考视图取最近的若干邻视图做立体
    NDEPTH = 24        # 深度区间采样数
    STRIDE = 6         # 像素子采样间隔
    PATCH = 3          # patch 半径
    MAX_COST = 0.5     # 有效像素的最大平均 NCC 代价
    MIN_CONF = 2       # 至少被这么多邻视图一致看到才收点
    VOXEL_REL = 0.6    # 体素去重边长 = 深度采样步长 × 该系数
    MIN_NEIGHBORS = 3  # 散噪声剔除：3×3×3 邻域体素内至少这么多个点才保留
    DENOISE_MIN_PTS = 24  # 融合点过稀疏时跳过统计降噪（避免小场景被误清空）

    def run(self, ctx: StageContext) -> None:
        from ..sfm.features import to_gray
        from ..sfm.features import load_image

        def _load_rgb(path):
            im = load_image(path)
            if im is None:
                raise ValueError(f"无法读取图像：{path}")
            return np.asarray(im.convert("RGB"), dtype=np.uint8)   # (H,W,3)

        sfm = ctx.ws.root / "sfm"
        if not (sfm / "cameras.txt").is_file() or not (sfm / "images.txt").is_file():
            raise ValueError("缺少 SfM 产物（sfm/cameras.txt、sfm/images.txt），无法稠密化；请先跑 sfm")
        K, _w, _h = read_shared_K(sfm / "cameras.txt")
        poses = read_images(sfm / "images.txt")               # [(name, R, t)] 世界→相机
        if len(poses) < 2:
            raise ValueError(f"MVS 至少需 2 个已注册视图（当前 {len(poses)}）；请增加有重叠的照片")
        sparse = read_ply_xyz(sfm / "points3D.ply")
        if len(sparse) < 6:
            raise ValueError("稀疏点过少，无法可靠定标深度区间；请改善 SfM（更多重叠照片）")

        centers = np.array([-R.T @ t for _n, R, t in poses])   # 相机中心（世界系）

        refs = poses[:self.MAX_VIEWS]
        ctx.progress(self.name, 10,
                     f"真实 MVS：{len(refs)} 参考视图 × {self.NDEPTH} 深度平面扫描")
        all_pts: list[np.ndarray] = []
        all_col: list[np.ndarray] = []
        steps: list[float] = []
        bounds_all: list[tuple[float, float]] = []
        dm_dir = ctx.ws.dir("mvs", "depthmaps")         # 逐视图深度图中间产物（供 v0.3 mesh 网格化）
        for vi, (name, Rv, tv) in enumerate(refs):
            # 逐视图鲁棒定标深度区间（稀疏点投影到本视图的正深度）：不同视图到表面的
            # 距离不同，各自取区间才能用有限采样数获得足够深度分辨率。
            z_view = (sparse @ Rv.T + tv)[:, 2]
            dmin, dmax = self._robust_bounds(z_view)
            if dmin is None:
                continue
            depths = np.linspace(dmin, dmax, self.NDEPTH)
            steps.append((dmax - dmin) / max(self.NDEPTH - 1, 1))
            bounds_all.append((dmin, dmax))
            # 取基线最近的若干邻视图（排除自身）
            order = np.argsort([np.linalg.norm(centers[k] - centers[vi])
                                for k in range(len(poses))])
            nb_idx = [k for k in order if k != vi][:self.MAX_NB]
            try:
                ref_rgb = _load_rgb(ctx.ws.root / "images" / name)     # 一次读图同时供扫面(灰)与上色(RGB)
            except (ValueError, OSError):
                continue
            ref_gray = ref_rgb.mean(axis=2)
            src_grays, src_RTs = [], []
            for k in nb_idx:
                nm, Rk, tk = poses[k]
                try:
                    src_grays.append(to_gray(ctx.ws.root / "images" / nm))
                except (ValueError, OSError):
                    continue
                src_RTs.append((Rk, tk))
            if not src_grays:
                continue
            dmap, cost, conf, (us, vs) = plane_sweep_depth(
                ref_gray, src_grays, K, (Rv, tv), src_RTs, depths,
                patch=self.PATCH, stride=self.STRIDE, max_cost=self.MAX_COST)
            pts, mask = depth_map_to_points(us, vs, dmap, K, (Rv, tv),
                                            conf=conf, min_conf=self.MIN_CONF, max_cost=self.MAX_COST)
            if len(pts):
                h, w = ref_gray.shape
                uu = np.clip(us.ravel()[mask].astype(int), 0, w - 1)
                vv = np.clip(vs.ravel()[mask].astype(int), 0, h - 1)
                all_pts.append(pts)
                all_col.append(ref_rgb[vv, uu])                        # 真彩色采样（非灰度）
                # 落盘本视图被采纳的深度图（仅保留达 MIN_CONF/MAX_COST 的像素）供网格化
                accepted = np.isfinite(dmap) & (conf >= self.MIN_CONF) & (cost <= self.MAX_COST)
                gu = np.clip(us.astype(int), 0, w - 1)
                gv = np.clip(vs.astype(int), 0, h - 1)
                np.savez(dm_dir / f"view{vi:03d}.npz", name=name, K=K, R=Rv, t=tv,
                         us=us, vs=vs, depth=np.where(accepted, dmap, np.nan),
                         rgb=ref_rgb[gv, gu])                    # (Hs,Ws,3) 逐格真彩色，与顶点对齐
            ctx.progress(self.name, 10 + int(80 * (vi + 1) / len(refs)),
                         f"视图 {vi + 1}/{len(refs)}：累计稠密点 {sum(len(p) for p in all_pts)}")

        if not all_pts:
            raise ValueError("MVS 未产生任何稠密点（照片纹理/重叠不足或位姿不可靠）；不臆造点云")
        P = np.vstack(all_pts)
        C = np.vstack(all_col).astype(np.uint8)
        voxel = float(np.median(steps)) * self.VOXEL_REL if steps else 0.0
        raw_n = len(P)
        denoised = 0
        if raw_n >= self.DENOISE_MIN_PTS:               # 稀疏云跳过，避免跨视图重复不足时误清真实表面
            Pf, Cf = remove_scatter_noise(P, voxel=voxel, min_neighbors=self.MIN_NEIGHBORS, colors=C)
            if len(Pf) > 0:                             # 若几乎全清（过严）则保留原云，不破坏产物
                P, C = Pf, Cf
                denoised = raw_n - len(P)
        P, C = self._voxel_dedup(P, C, step=voxel)
        write_ply_ascii(ctx.ws.file("mvs", "fusion.ply"), P, C)

        ctx.progress(self.name, 100, f"稠密点云完成：{len(P)} 点")
        (ctx.ws.root / "mvs" / "stats.json").write_text(json.dumps({
            "engine": "real/planesweep",
            "reference_views": len(refs),
            "total_views": len(poses),
            "depth_range": [round(float(min(b[0] for b in bounds_all)), 4),
                            round(float(max(b[1] for b in bounds_all)), 4)] if bounds_all else [0.0, 0.0],
            "num_depths": self.NDEPTH,
            "dense_points": int(len(P)),
            "raw_fused_points": int(raw_n),
            "noise_removed": int(denoised),
            "sparse_points": int(len(sparse)),
            "note": "v0.2 传统 MVS 基线（平面扫描 NCC + 逐视图鲁棒定标 + 跨视图置信融合 + 参考视图真彩色采样）；学习型/PatchMatch 高档待落地",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _robust_bounds(z) -> tuple:
        """从一维正深度鲁棒定标扫描区间：先用 median±k·MAD 剔除极端远点/退化值（SfM
        稀疏点常含射影退化的千万级远点，会把 percentile 直接撑爆），再取存活点 2%~98%
        分位并保证最小宽度。无有效正值时返回 (None, None)。"""
        z = np.asarray(z, dtype=np.float64).ravel()
        z = z[np.isfinite(z) & (z > 1e-6)]
        if z.size == 0:
            return None, None
        med = float(np.median(z))
        if z.size < 6:
            return max(1e-3, med * 0.5), med * 2.0
        sigma = 1.4826 * float(np.median(np.abs(z - med)))
        if sigma <= 1e-6:
            sigma = max(float(np.percentile(z, 95) - np.percentile(z, 5)), 1e-3) / 3.92
        keep = z[(z > med - 8.0 * sigma) & (z < med + 8.0 * sigma)]
        if keep.size < 6:
            keep = z
        lo, hi = np.percentile(keep, 2), np.percentile(keep, 98)
        span = hi - lo
        min_span = max(med * 0.3, 4.0 * sigma)
        if span < min_span:                            # 主导表面过于集中时适当撑开，避免区间贴死
            c = 0.5 * (lo + hi)
            lo, hi = c - min_span / 2, c + min_span / 2
        return float(max(1e-3, lo - 0.05 * span)), float(hi + 0.05 * span)

    @staticmethod
    def _voxel_dedup(P: np.ndarray, C: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
        """按边长 step 的体素去重：每格保留首个点（避免同一表面点被多视图重复堆积）。"""
        if len(P) == 0 or step <= 0:
            return P, C
        key = np.floor(P / step).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        first.sort()
        return P[first], C[first]
