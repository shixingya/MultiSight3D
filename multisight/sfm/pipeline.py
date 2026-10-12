"""两视图真实 SfM 编排：把特征与几何各件串成端到端重建（纯 numpy）。

这是自研 SfM 主线的「可运行内核」：给定两张有重叠/视差的图像，产出相机相对位姿
与稀疏 3D 点——诚实反映匹配/内点质量，特征不足时返回明确的降级结果而非崩溃。
`select_best_pair` 在受限候选帧对（相邻对 + 首帧对全体）中按 RANSAC 内点数择优，
避免盲目取前两份导致重叠不足退化或误吃重复纹理假阳性。
stages/sfm.Real 以此为第一步（先两视图，后续迭代扩到增量注册 + BA）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import detect_and_describe, match_descriptors, to_gray
from .geometry import (estimate_fundamental, ransac_fundamental, recover_pose,
                       ransac_pnp, triangulate, project)
from .bundle import bundle_adjustment, mean_reprojection_error


@dataclass
class TwoViewResult:
    ok: bool
    num_corners_a: int = 0
    num_corners_b: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    F: np.ndarray | None = None
    K: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    median_reproj_px: float = float("nan")

    @property
    def registered(self) -> bool:
        return self.ok and self.R is not None


def default_K(shape, focal: float | None = None) -> np.ndarray:
    """由图像尺寸给一个默认针孔内参：fx=fy=焦距（缺省取长边），光心居中。"""
    h, w = shape[:2]
    f = float(focal) if focal else float(max(h, w))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def two_view_from_features(ga, ca, da, gb, cb, db, *, focal: float | None = None,
                           ratio: float = 0.8, ransac_iters: int = 300,
                           ransac_threshold: float = 1.0,
                           seed: int = 0) -> TwoViewResult:
    """给定两帧已提取的角点+描述子，做比值匹配→RANSAC F→位姿恢复→三角化。

    拆出此内部函数，供 select_best_pair 复用（各帧特征只算一次，不随候选对重复检测）。
    """
    res = TwoViewResult(ok=False, num_corners_a=len(ca), num_corners_b=len(cb))
    matches = match_descriptors(da, db, ratio=ratio, mutual=True)
    res.num_matches = len(matches)
    if len(matches) < 8:                                # 8 点法最低要求
        return res
    idx_a = np.array([i for i, _ in matches])
    idx_b = np.array([j for _, j in matches])
    p1 = np.array([[ca[i].x, ca[i].y] for i in idx_a], dtype=np.float64)
    p2 = np.array([[cb[j].x, cb[j].y] for j in idx_b], dtype=np.float64)

    F, inliers = ransac_fundamental(p1, p2, iters=ransac_iters,
                                    threshold=ransac_threshold, seed=seed)
    res.F = F
    if F is None or inliers.sum() < 8:                  # 无足够内点 → 不臆造位姿
        return res
    res.num_inliers = int(inliers.sum())
    K = default_K(ga.shape, focal)
    res.K = K
    R, t, X = recover_pose(F, K, p1[inliers], p2[inliers])
    res.R, res.t = R, t
    finite = ~np.isnan(X).any(axis=1)
    # 仅保留两相机前方的点（真实三角化应剔除背对相机的解）
    if finite.any():
        xf = X[finite]
        z2 = (R @ xf.T)[2] + t[2]
        front = (xf[:, 2] > 0) & (z2 > 0)
        res.points3d = xf[front]
    else:
        res.points3d = np.zeros((0, 3))
    # 用全部内点重投影误差（像素）估计中位重投影残差，作为质量指标
    try:
        Fref = estimate_fundamental(p1[inliers], p2[inliers])
        h1 = np.column_stack([p1[inliers], np.ones(res.num_inliers)])
        h2 = np.column_stack([p2[inliers], np.ones(res.num_inliers)])
        err = np.abs(np.einsum("ij,jk,ik->i", h2, Fref, h1))
        lines = Fref @ h1.T                          # 3×N：图像 2 中的极线 [a,b,c]ᵀ
        den = lines[0] ** 2 + lines[1] ** 2
        res.median_reproj_px = float(np.median(err / np.sqrt(den + 1e-12)))
    except (ValueError, np.linalg.LinAlgError):
        pass
    res.ok = len(res.points3d) > 0
    return res


def two_view_reconstruction(img_a, img_b, *, focal: float | None = None,
                            max_corners: int = 600, patch: int = 8,
                            ratio: float = 0.8, ransac_iters: int = 300,
                            ransac_threshold: float = 1.0,
                            seed: int = 0) -> TwoViewResult:
    """两帧图像（路径/ndarray）→ 完整两视图重建。任何环节点不足都诚实降级。"""
    ga = to_gray(img_a)
    gb = to_gray(img_b)
    ca, da = detect_and_describe(ga, max_corners=max_corners, patch=patch)
    cb, db = detect_and_describe(gb, max_corners=max_corners, patch=patch)
    return two_view_from_features(ga, ca, da, gb, cb, db, focal=focal, ratio=ratio,
                                  ransac_iters=ransac_iters, ransac_threshold=ransac_threshold,
                                  seed=seed)


def candidate_pairs(n: int) -> list[tuple[int, int]]:
    """受限候选帧对：相邻滑动对 (i, i+1) + 首帧对其余各帧。线性 O(n)，避免 O(n²)。"""
    pairs: set[tuple[int, int]] = set()
    for i in range(n - 1):
        pairs.add((i, i + 1))
    for j in range(2, n):
        pairs.add((0, j))
    return sorted(pairs)


def select_best_pair(images, *, focal: float | None = None, max_corners: int = 600,
                     patch: int = 8, ratio: float = 0.8, ransac_iters: int = 300,
                     ransac_threshold: float = 1.0,
                     seed: int = 0) -> tuple[TwoViewResult, tuple[int, int]]:
    """在候选帧对中选 RANSAC 内点最多的一对做两视图重建。

    返回 (最优结果, (i, j))。各帧特征只提取一次；若全部候选都点不足，返回内点最多的
    那一对（其 ok/registered 仍为假，由调用方据实判断），而非硬造。
    """
    grays, feats = _extract_features(images, max_corners, patch)
    best: tuple[TwoViewResult, tuple[int, int]] | None = None
    for (i, j) in candidate_pairs(len(grays)):
        r = _score_pair(grays, feats, i, j, focal=focal, ratio=ratio,
                        ransac_iters=ransac_iters, ransac_threshold=ransac_threshold, seed=seed)
        if best is None or r.num_inliers > best[0].num_inliers:
            best = (r, (i, j))
    if best is None:                                     # <2 帧：无候选对
        raise ValueError("select_best_pair 需要至少 2 张图像")
    return best


def _extract_features(images, max_corners: int, patch: int):
    """逐帧灰度化 + 角点/描述子（只算一次，供选对/增量注册复用）。"""
    grays = [to_gray(im) for im in images]
    feats = [detect_and_describe(g, max_corners=max_corners, patch=patch) for g in grays]
    return grays, feats


def _score_pair(grays, feats, i, j, *, focal, ratio, ransac_iters, ransac_threshold, seed):
    ca, da = feats[i]
    cb, db = feats[j]
    return two_view_from_features(grays[i], ca, da, grays[j], cb, db, focal=focal,
                                  ratio=ratio, ransac_iters=ransac_iters,
                                  ransac_threshold=ransac_threshold, seed=seed)


@dataclass
class RegisteredCamera:
    index: int
    R: np.ndarray
    t: np.ndarray
    num_inliers: int = 0


@dataclass
class ReconstructionResult:
    """多视图增量注册结果（世界系 = 初始帧 a 的相机系，尺度由首对单位基线定）。"""
    cameras: list[RegisteredCamera] = field(default_factory=list)
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tracks: list[dict] = field(default_factory=list)          # tid -> {frame_idx: corner_idx}
    K: np.ndarray | None = None                                # 共享针孔内参（3×3）
    num_frames: int = 0
    num_registered: int = 0
    initial_pair: tuple[int, int] = (-1, -1)
    pnp_inliers: dict[int, int] = field(default_factory=dict)  # frame_idx -> pnp 内点数
    bundle_applied: bool = False
    mean_reproj_px: float = float("nan")               # BA 后均重投影误差（未 BA 则为注册后直接计算）
    grown_points: int = 0                              # 增量注册中三角化的新 3D 点数（结构增长）
    extended_obs: int = 0                              # 已知点获得的新观测补记数（强化地图/BA）

    @property
    def ok(self) -> bool:
        return self.num_registered >= 2 and len(self.points3d) > 0


def incremental_reconstruction(images, *, focal: float | None = None, max_corners: int = 600,
                               patch: int = 8, ratio: float = 0.8, ransac_iters: int = 300,
                               ransac_threshold: float = 1.0, pnp_reproj_px: float = 4.0,
                               min_pnp_inliers: int = 10, refine: bool = True,
                               grow: bool = True, seed: int = 0) -> ReconstructionResult:
    """增量注册：选最优初始对做两视图→建 3D 轨→逐帧 PnP（2D-3D 迁移 + RANSAC）注册→（可选）结构增长→（可选）全局 BA。

    v0.1 范围：以初始对三角化的 3D 点为地图，向后续帧做特征迁移并用 RANSAC-PnP 求位姿；
    grow=True 时每帧注册后就地做结构增长（新点三角化 + 已知点补观测）；
    refine=True 时再跑一轮全局捆绑调整联合精化位姿与结构。
    重叠不足/内点不够的帧会诚实地不被注册。
    """
    grays, feats = _extract_features(images, max_corners, patch)
    n = len(grays)
    if n < 2:
        raise ValueError("incremental_reconstruction 需要至少 2 张图像")
    res = ReconstructionResult(num_frames=n)

    init = None
    for (i, j) in candidate_pairs(n):
        r = _score_pair(grays, feats, i, j, focal=focal, ratio=ratio,
                        ransac_iters=ransac_iters, ransac_threshold=ransac_threshold, seed=seed)
        if init is None or r.num_inliers > init[0].num_inliers:
            init = (r, (i, j))
    two, (a, b) = init
    res.initial_pair = (a, b)
    K = default_K(grays[a].shape, focal)
    res.K = K
    if not two.registered or two.F is None or two.num_inliers < 8:
        return res                                       # 初始对不可靠→诚实降级

    ca, da = feats[a]
    cb, db = feats[b]
    matches = match_descriptors(da, db, ratio=ratio, mutual=True)
    ia = np.array([x for x, _ in matches])
    ib = np.array([y for _, y in matches])
    p1 = np.array([[ca[x].x, ca[x].y] for x in ia], dtype=np.float64)
    p2 = np.array([[cb[y].x, cb[y].y] for y in ib], dtype=np.float64)
    F, inl = ransac_fundamental(p1, p2, iters=ransac_iters, threshold=ransac_threshold, seed=seed)
    R, t, X = recover_pose(F, K, p1[inl], p2[inl])
    iinl = np.where(inl)[0]                              # 匹配集中的下标→内点
    z1 = X[:, 2]
    z2 = (R @ X.T)[2] + t[2]
    front = (~np.isnan(X).any(axis=1)) & (z1 > 0) & (z2 > 0)
    pts: list[np.ndarray] = []
    tracks: list[dict] = []
    for k in np.where(front)[0]:
        m = iinl[k]
        pts.append(X[k])
        tracks.append({a: int(ia[m]), b: int(ib[m])})
    # 角点反查索引 frame -> {corner_idx: tid}：结构增长时判重/补观测的关键
    corner_tid: dict[int, dict[int, int]] = {a: {}, b: {}}
    for tid, obs in enumerate(tracks):
        for frame, ci in obs.items():
            corner_tid.setdefault(frame, {})[ci] = tid
    res.points3d = np.array(pts) if pts else np.zeros((0, 3))
    res.tracks = tracks
    res.cameras = [RegisteredCamera(a, np.eye(3), np.zeros(3), two.num_inliers),
                   RegisteredCamera(b, R, t, two.num_inliers)]
    registered = {a, b}

    for c in range(n):
        if c in registered:
            continue
        cc, dc = feats[c]
        Xobs, xobs, tid_of, ic_of = [], [], [], []
        claimed: set[int] = set()
        for r in list(registered):
            dr = feats[r][1]
            mcr = match_descriptors(dc, dr, ratio=ratio, mutual=True)  # (ic, ir)
            ir2ic: dict[int, int] = {}
            for ic, ir in mcr:
                ir2ic.setdefault(ir, ic)
            for tid, obs in enumerate(tracks):
                if tid in claimed or r not in obs:
                    continue
                ic = ir2ic.get(obs[r])
                if ic is not None:
                    Xobs.append(pts[tid])
                    xobs.append([cc[ic].x, cc[ic].y])
                    tid_of.append(tid)
                    ic_of.append(ic)
                    claimed.add(tid)
        if len(Xobs) < 6:
            continue
        Rc, tc, mask = ransac_pnp(K, np.array(Xobs), np.array(xobs, dtype=np.float64),
                                  threshold=pnp_reproj_px, seed=seed)
        ninl = int(mask.sum())
        if Rc is None or ninl < min_pnp_inliers:
            continue
        res.cameras.append(RegisteredCamera(c, Rc, tc, ninl))
        res.pnp_inliers[c] = ninl
        registered.add(c)
        ct = corner_tid.setdefault(c, {})
        for i in np.where(mask)[0]:                     # 用内点扩展轨迹（供后续帧迁移）
            ic = ic_of[i]
            if ic in ct:                                # 该角点已归属别的轨迹 → 跳过，保一像素一三角量（防 BA 双计）
                continue
            tracks[tid_of[i]][c] = ic
            ct[ic] = tid_of[i]
        # 结构增长：新注册帧与所有已注册帧角点两两匹配，已知点补观测、未知点新增
        if grow:
            _grow_structure(res, feats, K, c, registered, pts, tracks, corner_tid,
                            ratio=ratio, max_reproj_px=pnp_reproj_px)

    res.points3d = np.array(pts) if pts else np.zeros((0, 3))
    res.tracks = tracks
    res.cameras.sort(key=lambda cm: cm.index)
    res.num_registered = len(res.cameras)
    _bundle_refine(res, feats, a, refine)
    return res


def _grow_structure(res: ReconstructionResult, feats, K: np.ndarray, new_frame: int,
                    registered: set[int], pts: list, tracks: list,
                    corner_tid: dict, *, ratio: float = 0.8,
                    min_parallax_deg: float = 1.0, max_reproj_px: float = 4.0) -> None:
    """增量结构增长（v0.1 下一里程碑）：把新注册帧与已注册帧做角点两两匹配。

    - 新帧角点已归属某轨迹 → 跳过（PnP 阶段已处理）；
    - 对端角点已知（属轨迹）而新帧角点未知 → 给已知点补一条观测（强化地图/BA）；
    - 两端角点皆未知 → 用两帧投影矩阵三角化一个新 3D 点（需正深度、足够视差、重投影达标）。
    就地扩展 pts/tracks/corner_tid 并累计 res.grown_points/extended_obs。诚实原则：
    退化/小基线/残差过大的候选一律不新增，绝不臆造结构。
    """
    cam_by = {cm.index: cm for cm in res.cameras}
    P_new = K @ np.hstack([cam_by[new_frame].R, cam_by[new_frame].t.reshape(3, 1)])
    cc = feats[new_frame][0]
    ct_new = corner_tid.setdefault(new_frame, {})
    min_par = np.sin(np.radians(min_parallax_deg))
    for r in registered:
        if r == new_frame:
            continue
        cr = feats[r][0]
        matches = match_descriptors(feats[new_frame][1], feats[r][1], ratio=ratio, mutual=True)
        cm_r = cam_by[r]
        P_r = K @ np.hstack([cm_r.R, cm_r.t.reshape(3, 1)])
        ct_r = corner_tid.setdefault(r, {})
        # 相机中心（世界系）= -Rᵀ t，用于视差角判定
        C_new = -cam_by[new_frame].R.T @ cam_by[new_frame].t
        C_r = -cm_r.R.T @ cm_r.t
        for ic, ir in matches:
            if ic in ct_new:                            # 新帧该角点已有 3D 点
                continue
            tid_r = ct_r.get(ir)
            if tid_r is not None:                       # 已知点：补一条新观测
                if new_frame not in tracks[tid_r]:
                    tracks[tid_r][new_frame] = ic
                    ct_new[ic] = tid_r
                    res.extended_obs += 1
                continue
            pc = np.array([[cc[ic].x, cc[ic].y]], dtype=np.float64)
            pr = np.array([[cr[ir].x, cr[ir].y]], dtype=np.float64)
            X = triangulate(P_new, P_r, pc, pr)
            if np.isnan(X).any():
                continue
            x3 = X[0]
            # 正深度（两相机前方）+ 视差角足够（拒小基线无穷远歧义）
            d_new = x3 - C_new
            d_r = x3 - C_r
            nn, nr = np.linalg.norm(d_new), np.linalg.norm(d_r)
            if nn <= 1e-9 or nr <= 1e-9:
                continue
            cosang = float(np.clip(np.dot(d_new, d_r) / (nn * nr), -1.0, 1.0))
            if np.sin(np.arccos(cosang)) < min_par:
                continue
            if (x3[2] <= 0) or (((cm_r.R @ x3) + cm_r.t)[2] <= 0):
                continue
            # 重投影残差守卫（两视图均达标才收）
            if (np.linalg.norm(project(K, cam_by[new_frame].R, cam_by[new_frame].t, x3[None]) - pc)
                    > max_reproj_px or
                    np.linalg.norm(project(K, cm_r.R, cm_r.t, x3[None]) - pr) > max_reproj_px):
                continue
            tid = len(pts)
            pts.append(x3)
            tracks.append({new_frame: ic, r: ir})
            ct_new[ic] = tid
            ct_r[ir] = tid
            res.grown_points += 1


def _bundle_refine(res: ReconstructionResult, feats, anchor: int, refine: bool) -> None:
    """把注册结果转为 BA 观测并（可选）联合精化位姿与结构；就地写回 res。

    BA 固定首参数为锚点相机（世界系 = 初始帧 anchor，保 gauge）；观测不足或 BA 失败则仅算质量指标（诚实降级）。
    """
    if len(res.cameras) < 2 or len(res.points3d) == 0 or res.K is None:
        return
    by_index = {cm.index: cm for cm in res.cameras}
    order = [anchor] + [cm.index for cm in res.cameras if cm.index != anchor]
    pos = {f: k for k, f in enumerate(order)}
    ba_cams = [(np.asarray(by_index[f].R, dtype=np.float64).copy(),
                np.asarray(by_index[f].t, dtype=np.float64).copy()) for f in order]
    meas = []
    for tid, obs in enumerate(res.tracks):
        for frame, corner_idx in obs.items():
            k = pos.get(frame)
            if k is None:
                continue
            corner = feats[frame][0][corner_idx]
            meas.append((k, tid, float(corner.x), float(corner.y)))
    if len(meas) < 6:
        return
    if refine:
        try:
            ba_cams, X, _ = bundle_adjustment(res.K, ba_cams, res.points3d, meas)
            res.points3d = X
            res.bundle_applied = True
        except (ValueError, np.linalg.LinAlgError):
            pass
    res.mean_reproj_px = float(mean_reprojection_error(res.K, ba_cams, res.points3d, meas))
    for k, f in enumerate(order):
        by_index[f].R, by_index[f].t = ba_cams[k]
