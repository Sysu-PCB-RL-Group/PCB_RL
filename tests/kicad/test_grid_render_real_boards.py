# -*- coding: utf-8 -*-
"""`env/grid_render`（L/G 解析渲染器）的**真实板回归**：两块板、两个固定局面。

场景（Owner 2026-09-17 指定，均为"拿走若干件后的局面 + 指定待放件"）
⚠️ **前提**：除待放件外，其余件**一律保持导入时的源位姿** ⇒ `pad.aabb/center` 恰好等于真实几何，
   本文件的金表**只在该前提下有效**。一旦场景改为策略落子（位姿 ≠ 源位姿），必须先用
   `pcb_model.place_at` 位姿化 + `pcb_model.state_keepouts(model, placed)`（口径 A 唯一入口）再算，并**重标金表**
   （见台账 §0.6 o 与 `tests/kicad/test_pose_override.py`）：
  ① `lr1388-esp32-camera-photometer…--2感光板`：拿走 **U7 / USBC1**，待放 **U7**
  ② `bliysfyt-night-light…--Board2`：拿走 **U1 / C1**，待放 **U1**

三层断言（越往下越严）：
  A. `test_legal_counts_match_golden`  —— **合法格数金表**（4 旋转）+ cell 尺寸：任何数学/口径改动都会红（快）
  B. `test_sampling_matches_judge`     —— 抽样对拍 `Judge`（均匀 + **边界格**）：保证与判定器同源（快）
  C. `test_exhaustive_matches_judge`   —— **全量穷举 160,000 次双向对拍**：最严，需 `PCB_RL_SLOW=1` 打开（~50s/板）
  D. `test_source_pose_legality`       —— 待放件**源位姿**的合法性金表（局面自洽性）

金表来源（2026-09-17 实测）：
  probe112/113（lr1388·U7）    ：合法格 2586/1836/2732/1604；穷举 160000 次 0 不一致；源位姿合法
  probe114（bliysfyt·U1）      ：合法格   54/ 189/ 455/  84；穷举 160000 次 0 不一致；源位姿合法
  probe122（新增 4 场景，Owner 指定）：jackt_IMU_U1 864/948/810/1008、wzh_main_Q2 486/504/585/721、
      verysunshine_U4 720/1557/770/1623（**源位姿本身违规 `soft.pad_via×1`** ⇒ 台账"专家在违规点不可复现"）、
      coloreasypico_D2 10820/9063/10644/9218

运行（需要 KiCad 环境）：
    $env:KICAD_PATH="E:\\Kicad"; python -m pytest tests/kicad/test_grid_render_real_boards.py -v
再加全量穷举：
    $env:PCB_RL_SLOW="1"; python -m pytest tests/kicad/test_grid_render_real_boards.py -v
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr  # noqa: E402

# ⚠️ 必须**模块级**导入 `pcb_model`：`tests/conftest.py` 的 autouse fixture 会把
# `sys.modules["utils.kicad_utils"]` 换成 MagicMock（且为**全局粘性**注入）⇒ 若在
# fixture 内部才 import，就会拿到 Mock 模型（footprints 为空），导致本文件"静默跳过"。
# 模块级导入发生在**收集期**，早于任何测试执行，故不受污染。
from utils.kicad_utils import pcb_model as _PCB_MODEL  # noqa: E402

pytestmark = pytest.mark.require_kicad

# ---- 六个固定场景（含金表；goldens 由 probe122 生成） ----
SCENARIOS = [
    dict(
        tag="lr1388_U7",
        board_sub="lr1388",
        drop=("U7", "USBC1"),
        comp="U7",
        # ⚠️ **2026-09-18 重标**（口径 (A)：开槽随件；见 §0.6 p / probe169）：
        #    本场景"拿走 USBC1" ⇒ 它自带的 2 个开槽随之离板 ⇒ rot0/rot2 各多出 44/63 个合法格。
        #    旧值（开槽留在源位姿）= (2586, 1836, 2732, 1604)。
        counts=(2630, 1836, 2795, 1604),          # 每个旋转的合法格数
        cell_mm=(0.2546, 0.1650),                 # (cell_x, cell_y)
        source_legal=True,
    ),
    dict(
        tag="bliysfyt_U1",
        board_sub="bliysfyt",
        drop=("U1", "C1"),
        comp="U1",
        counts=(54, 189, 455, 84),
        cell_mm=(0.0817, 0.1215),
        source_legal=True,
    ),
    dict(                                         # 新增（Owner 2026-09-17）
        tag="jackt_IMU_U1",
        board_sub="jackt",
        drop=("U1",),
        comp="U1",
        counts=(864, 948, 810, 1008),
        cell_mm=(0.0762, 0.0762),
        source_legal=True,
    ),
    dict(
        tag="wzh_main_Q2",
        board_sub="wzh2814685",
        drop=("Q2", "Q1", "C1"),
        comp="Q2",
        counts=(486, 494, 585, 707),              # 含 _EPS 保守项（板边平局：24 格）
        cell_mm=(0.2250, 0.0500),                 # 45×10mm 板 ⇒ 极端各向异性 cell
        source_legal=True,
    ),
    dict(
        tag="verysunshine_U4",
        board_sub="verysunshine",
        drop=("U9", "U4"),
        comp="U4",
        counts=(720, 1557, 770, 1623),
        cell_mm=(0.3007, 0.1673),
        source_legal=False,                       # ⚠️ 专家源位姿本身违规（soft.pad_via×1）
    ),
    dict(
        tag="coloreasypico_D2",
        board_sub="coloreasypico2",
        drop=("R2", "D2", "D3", "R8"),
        comp="D2",
        counts=(10820, 9063, 10639, 9218),        # 含 _EPS 保守项（障碍阈值平局：5 格）
        cell_mm=(0.2550, 0.1050),
        source_legal=True,
    ),
]
IDS = [s["tag"] for s in SCENARIOS]


def _real_pcb_model():
    """取真实的 `pcb_model` 模块；若被 Mock 污染则**显式失败**（不静默跳过）。"""
    if _PCB_MODEL is None or not getattr(_PCB_MODEL, "__file__", None):
        pytest.fail(
            "检测到 Mock 污染：`utils.kicad_utils` 已被 tests/conftest.py 的 autouse fixture "
            "替换为 MagicMock（该注入是全局粘性的）⇒ 本文件必须**单独运行**，"
            "或在收集顺序上不与 tests/ 根目录的 mock 测试同会话。"
        )
    return _PCB_MODEL


@pytest.fixture(scope="session")
def scenes(all_kicad_pcb_paths):
    """按场景加载模型并构造局面：`tag -> (model, comp, placed, spec)`（session 级，只解析一次）。"""
    pcb_model = _real_pcb_model()

    out = {}
    for sc in SCENARIOS:
        hits = [p for p in all_kicad_pcb_paths
                if sc["board_sub"] in os.path.basename(os.path.dirname(p))]
        if not hits:
            pytest.skip(f"未找到板：{sc['board_sub']}")
        path = hits[0]
        model = pcb_model.build_model(pcb_model.pcbnew.LoadBoard(path), path)
        byref = {f.ref: f for f in model.footprints}
        miss = [r for r in sc["drop"] if r not in byref]
        if sc["comp"] not in byref or miss:
            pytest.skip(f"{sc['tag']}：板内缺少 ref（待放={sc['comp']}，缺失={miss}）")
        placed = [f for f in model.footprints if f.ref not in set(sc["drop"])]
        out[sc["tag"]] = (model, byref[sc["comp"]], placed, gr.grid_spec(model, m=gr.GRID_M))
    return out


def _scen(tag):
    return next(s for s in SCENARIOS if s["tag"] == tag)


# ===================== A. 合法格数金表 =====================


@pytest.mark.parametrize("tag", IDS)
def test_legal_counts_match_golden(tag, scenes):
    """4 个旋转的合法格数必须与金表一致（+ cell 尺寸一致）。"""
    sc = _scen(tag)
    model, comp, placed, spec = scenes[tag]

    assert spec.mm_of(0, 0)  # 走一遍映射，确保 spec 可用
    assert spec.cell_x == pytest.approx(sc["cell_mm"][0], abs=1e-3), \
        f"{tag} cell_x 变了：{spec.cell_x:.4f}（金表 {sc['cell_mm'][0]}）"
    assert spec.cell_y == pytest.approx(sc["cell_mm"][1], abs=1e-3), \
        f"{tag} cell_y 变了：{spec.cell_y:.4f}（金表 {sc['cell_mm'][1]}）"

    L = gr.legal_masks(model, placed, comp, spec)
    assert L.shape == (4, gr.GRID_M, gr.GRID_M)
    got = tuple(int(L[k].sum()) for k in range(4))
    assert got == sc["counts"], f"{tag} 合法格数变了：got={got} 金表={sc['counts']}"


# ===================== D. 源位姿合法性 =====================


@pytest.mark.parametrize("tag", IDS)
def test_source_pose_legality(tag, scenes):
    """待放件的**源位姿**在本局面下的合法性（局面自洽性金表）。"""
    from env.violations import Judge, pose_of

    sc = _scen(tag)
    model, comp, placed, _spec = scenes[tag]
    judge = Judge(model)
    others = {f.ref: pose_of(f) for f in placed}
    res = judge.judge(comp.ref, pose_of(comp), others)
    assert res.is_violation == (not sc["source_legal"]), \
        f"{tag} 源位姿合法性变了：{res.describe()}"


# ===================== B. 抽样对拍 =====================


@pytest.mark.parametrize("tag", IDS)
def test_sampling_matches_judge(tag, scenes):
    """抽样（均匀 + 边界格）对拍：**危险方向必须 0**；过度封锁只能出现在"阈值平局"。"""
    model, comp, placed, spec = scenes[tag]
    res = gr.verify_legal(model, placed, comp, spec, n=250, seed=17)
    assert res.checked >= 4 * 250
    assert res.danger == 0, \
        f"{tag} 出现**危险误判** {res.danger} 例：{res.danger_examples[:3]}"
    not_tie = [c for c in res.over_cells
               if not _is_threshold_tie(model, comp, placed, spec, *c)]
    assert not not_tie, f"{tag} 抽样中出现**非阈值平局**的过度封锁：{not_tie[:5]}"


# ===================== C. 全量穷举（慢） =====================


def _obstacles_of(model, comp, placed):
    """当前件的障碍集合（已完成过滤）：[(kind, aabb, layers, netcode), ...]"""
    out = []
    for f in placed:
        for p in f.pads:
            out.append(("pad", p.aabb, p.layers, p.netcode))
    for v in model.vias:
        out.append(("via", v.aabb, v.layers, v.netcode))
    for k in model.keepouts:
        if k.owner_ref != comp.ref:
            out.append(("keepout", k.aabb, k.layers, None))
    return out


def _is_threshold_tie(model, comp, placed, spec, k, i, j, tol=1e-6):
    """该格是否只是"恰好卡在阈值上"——即浮点平局导致的保守封锁。

    两种平局都算（**必须同时检查**，否则会误判为"非平局"）：
      * **障碍阈值平局**：`gap` 恰好 = 0.2（软间距边界）
      * **板框边缘平局**：pad 恰好与板边齐平（`edge_margin` = 0；`Judge` 判"贴边合法"）
    """
    from utils.kicad_utils.pcb_model import rotate_aabb

    x, y = spec.mm_of(i, j)
    va = model.board_outline.valid_area(0.0)
    obs = _obstacles_of(model, comp, placed)
    gap_best, edge_best = None, None
    for pad in comp.pads:
        if not pad.local_aabb:
            continue
        a = rotate_aabb(pad.local_aabb, k * 90.0, (x, y))
        m = min(a[0] - va[0], va[2] - a[2], a[1] - va[1], va[3] - a[3])
        if edge_best is None or abs(m) < abs(edge_best):
            edge_best = m
        for kind, abb, layers, code in obs:
            if not (set(pad.layers) & set(layers)):
                continue
            if kind != "pad":
                if pad.netcode == 0 or code == 0 or code == pad.netcode:
                    continue
            g = max(a[0] - abb[2], abb[0] - a[2], a[1] - abb[3], abb[1] - a[3])
            if gap_best is None or g < gap_best:
                gap_best = g
    tie_gap = gap_best is not None and abs(gap_best - gr.SOFT_GAP_MM) <= tol
    tie_edge = edge_best is not None and abs(edge_best) <= tol
    return tie_gap or tie_edge


def _exhaustive(model, comp, placed, spec):
    """全格 × 4 旋转双向对拍。

    Returns: (n_ok, n_bad, n_forb, over_cells)
      over_cells = [(k, i, j)] —— L 说禁放但 Judge 判合法的格（应当全部是"阈值平局"）
    """
    from env.violations import Judge, Pose, pose_of

    L = gr.legal_masks(model, placed, comp, spec)
    judge = Judge(model)
    others = {f.ref: pose_of(f) for f in placed}
    n_ok = n_bad = n_forb = 0
    over_cells = []
    for k in range(4):
        for i, j in np.argwhere(L[k]):
            x, y = spec.mm_of(int(i), int(j))
            if judge.judge(comp.ref, Pose(x, y, k), others).is_violation:
                n_bad += 1
            else:
                n_ok += 1
        for i, j in np.argwhere(~L[k]):
            x, y = spec.mm_of(int(i), int(j))
            if judge.judge(comp.ref, Pose(x, y, k), others).is_violation:
                n_forb += 1
            else:
                over_cells.append((k, int(i), int(j)))
    return n_ok, n_bad, n_forb, over_cells


@pytest.mark.parametrize("tag", IDS)
@pytest.mark.skipif(not os.environ.get("PCB_RL_SLOW"),
                    reason="全量穷举较慢（~50s/板）；设 PCB_RL_SLOW=1 打开")
def test_exhaustive_matches_judge(tag, scenes):
    """全量穷举（双向）：

    * **危险方向必须严格为 0**：`L=True` 的每一格，`Judge` 必须判**不违规**
      （这是"违规零容忍"的硬要求）。
    * **过度封锁（`L=False` 但 `Judge` 判合法）只能是"阈值平局"**：
      `gap` 恰好 = 0.2 的格在浮点上无唯一答案，`grid_render` 用 1nm 的保守外扩
      把它们一律判为禁放（安全方向）；本测试要求这些格 `|gap−0.2| ≤ 1e-6` 且有数量上限。
    """
    sc = _scen(tag)
    model, comp, placed, spec = scenes[tag]
    n_ok, n_bad, n_forb, over_cells = _exhaustive(model, comp, placed, spec)

    total_legal = sum(sc["counts"])
    assert n_ok + n_bad == total_legal, f"{tag} 可行点数变了：{n_ok + n_bad} ≠ 金表 {total_legal}"
    assert n_bad == 0, f"{tag} 出现**危险误判** {n_bad} 例（L 说可放但 Judge 判违规）"
    assert len(over_cells) <= 200, f"{tag} 过度封锁异常偏多：{len(over_cells)} 例"
    not_tie = [c for c in over_cells if not _is_threshold_tie(model, comp, placed, spec, *c)]
    assert not not_tie, f"{tag} 存在**非阈值平局**的过度封锁 {len(not_tie)} 例：{not_tie[:5]}"
    assert n_forb + len(over_cells) == 4 * spec.m * spec.m - total_legal


# ===================== E. G 层（组合 / 共用标尺 / 退化守卫） =====================
#
# G 的口径（Owner 2026-09-17 定稿）：
#     s_hpwl = (−ΔHPWL)/D      s_disp = Δσ/D            （D = 板对角线）
#     z_*    = clip((s_* − p5)/(p95 − p5), 0, 1)        ← 4 张表算完**共用一把尺**（表从不合并）
#     G      = 0.7·z_hpwl + 0.3·z_disp（合法格）；非法格 = −inf
#     某分量池内恒定 ⇒ 该分量给 0.5；两分量同时恒定 ⇒ G ≡ 0.5 且 degenerate=True
#     （`degenerate` 只作元数据：不进 G 张量、不进网络输入）

EXPERT_G_GOLDEN = {          # probe130/142 实测（0.7:0.3；专家格 = 源格，非法时取同旋转最近合法格）
    # ⚠️ 2026-09-17 `is_power_net` 修补后重标：6 个局面里**只有 jackt 变了**
    # （它的 U1 挂在新增判定的 `3V3`(11 pad) 上 ⇒ 该 net 不再计入 HPWL/离散度）
    "lr1388_U7": 1.0000,
    "bliysfyt_U1": 0.8839,
    "jackt_IMU_U1": 0.2005,         # 修补前 0.1863
    "wzh_main_Q2": 0.9366,
    "verysunshine_U4": 0.8230,      # ⚠️ 源位姿非法 ⇒ 代理格
    "coloreasypico_D2": 0.9821,
}


def _expert_cell(comp, spec, L):
    """专家格 `(k,i,j,dist)`：源格合法则用它，否则取**同一旋转内**最近的合法格。"""
    ci, cj = spec.cell_of(*comp.anchor)
    k = comp.rotation_index
    if L[k][ci, cj]:
        return k, ci, cj, 0.0
    ck = np.argwhere(L[k])
    assert len(ck), "该旋转没有合法格，无法构造代理专家格"
    xs, ys = spec.mm_of(ck[:, 0], ck[:, 1])
    d = np.hypot(xs - comp.anchor[0], ys - comp.anchor[1])
    b = int(np.argmin(d))
    return k, int(ck[b, 0]), int(ck[b, 1]), float(d[b])


def _z(s, lo, hi):
    """稳健 min-max（池内恒定 ⇒ 常数 0.5，防 0/0）。"""
    return np.full_like(s, 0.5) if hi - lo < 1e-12 else np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def _sigma_of(ctrs):
    n = len(ctrs)
    if n == 0:
        return 0.0
    cx = sum(c[0] for c in ctrs) / n
    cy = sum(c[1] for c in ctrs) / n
    return math.sqrt(max(sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in ctrs) / n, 0.0))


def _sigma_before(model, refs):
    """未放件时的 σ（net bbox 中心的标准差；只算 ≥2 个已放 pad 的非电源 net）。"""
    ctrs = []
    for net in model.nets.values():
        if net.is_power:
            continue
        pts = [(p.center[0], p.center[1]) for p in net.pads if p.ref in refs]
        if len(pts) >= 2:
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            ctrs.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    return _sigma_of(ctrs)


def _sigma_after(model, refs, comp, k, x, y):
    """把 comp 放到 (x,y,k) 后的 σ（独立构造：逐 net 取 pad 中心极值）。"""
    ctrs = []
    for net in model.nets.values():
        if net.is_power:
            continue
        pts = [(p.center[0], p.center[1]) for p in net.pads if p.ref in refs]
        for p in net.pads:
            if p.ref == comp.ref and p.local_aabb:
                a = _PCB_MODEL.rotate_aabb(p.local_aabb, k * 90.0, (x, y))
                pts.append(((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0))
        if len(pts) >= 2:
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            ctrs.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    return _sigma_of(ctrs)


def _hpwl_with(model, refs, comp, x, y, k):
    """把 comp 放到 (x,y,k) 后的 HPWL（独立构造，与 `hpwl_gain_maps` 不同实现）。"""
    total = 0.0
    for net in model.nets.values():
        if net.is_power:
            continue
        xs, ys = [], []
        for p in net.pads:
            if p.ref in refs:
                xs.append(p.center[0]); ys.append(p.center[1])
            elif p.ref == comp.ref and p.local_aabb:
                a = _PCB_MODEL.rotate_aabb(p.local_aabb, k * 90.0, (x, y))
                xs.append((a[0] + a[2]) / 2.0); ys.append((a[1] + a[3]) / 2.0)
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


@pytest.mark.parametrize("tag", IDS)
def test_G_domain_range_and_flag(tag, scenes):
    """E1：G 的有限域严格 = L、合法格 ∈ [0,1]、非法格 −inf；6 个局面都不退化。"""
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    G, degen = gr.gain_maps(model, placed, comp, spec, legal=L)

    assert G.shape == (4, gr.GRID_M, gr.GRID_M) and G.dtype == np.float32
    fin = np.isfinite(G)
    assert np.array_equal(fin, L), f"{tag}：G 的有限域必须严格等于 L"
    assert G[fin].min() >= 0.0 and G[fin].max() <= 1.0, f"{tag}：合法格 G 超出 [0,1]"
    assert bool(np.isneginf(G[~fin]).all()), f"{tag}：非法格必须为 −inf"
    assert degen is False, f"{tag}：该件的 net 已有已放 pad，不应退化"


@pytest.mark.parametrize("tag", IDS)
def test_G_uses_shared_ruler_across_rotations(tag, scenes):
    """E2：**共用一把尺**——4 张表算完后在同一个样本池上取一次 p5/p95（表从不合并；与逐旋转归一化不同）。"""
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    G, _ = gr.gain_maps(model, placed, comp, spec, legal=L)
    H = gr.hpwl_gain_maps(model, placed, comp, spec)
    Dd = gr.dispersion_delta(model, placed, comp, spec, L)
    ob = model.board_outline.bbox
    D = math.hypot(ob[2] - ob[0], ob[3] - ob[1])

    cells = [np.argwhere(L[k]) for k in range(4)]
    sh = [H[k][c[:, 0], c[:, 1]].astype(np.float64) / D for k, c in enumerate(cells) if len(c)]
    sd = [Dd[k][c[:, 0], c[:, 1]].astype(np.float64) / D for k, c in enumerate(cells) if len(c)]
    ph, pd = np.concatenate(sh), np.concatenate(sd)
    lo_h, hi_h = np.percentile(ph, 5), np.percentile(ph, 95)
    lo_d, hi_d = np.percentile(pd, 5), np.percentile(pd, 95)

    ref = np.full(G.shape, -np.inf, dtype=np.float64)        # 照定义独立写一遍
    per_rot = np.full(G.shape, -np.inf, dtype=np.float64)    # 违规写法（逐旋转归一化）
    for k, c in enumerate(cells):
        if not len(c):
            continue
        hk = H[k][c[:, 0], c[:, 1]].astype(np.float64) / D
        dk = Dd[k][c[:, 0], c[:, 1]].astype(np.float64) / D
        ref[k][c[:, 0], c[:, 1]] = 0.7 * _z(hk, lo_h, hi_h) + 0.3 * _z(dk, lo_d, hi_d)
        l1, h1 = np.percentile(hk, [5, 95])
        l2, h2 = np.percentile(dk, [5, 95])
        per_rot[k][c[:, 0], c[:, 1]] = 0.7 * _z(hk, l1, h1) + 0.3 * _z(dk, l2, h2)

    fin = np.isfinite(ref)
    assert np.array_equal(fin, L), "参考实现的有限域也应等于 L"
    assert np.allclose(G[fin], ref[fin], atol=2e-6), \
        f"{tag}：G 与『合并池 + 一次分位』的定义不一致（可能被改成逐旋转归一化）"
    assert np.abs(per_rot[fin] - G[fin]).max() > 1e-3, \
        f"{tag}：共用标尺与逐旋转归一化结果居然相同，测试失去区分度"


@pytest.mark.parametrize("tag", IDS)
def test_G_expert_cell_golden(tag, scenes):
    """E3：专家格（或同旋转最近合法格）的 G 金表——任何口径改动都会红。"""
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    G, _ = gr.gain_maps(model, placed, comp, spec, legal=L)
    k, i, j, _dist = _expert_cell(comp, spec, L)
    assert L[k][i, j]
    assert float(G[k][i, j]) == pytest.approx(EXPERT_G_GOLDEN[tag], abs=2e-3), \
        f"{tag}：专家格 G 变了（got {float(G[k][i, j]):.4f}）"


@pytest.mark.parametrize("tag", IDS)
def test_H_matches_independent_hpwl(tag, scenes):
    """E4：HPWL 分量 `H = −ΔHPWL` 与独立重算一致（各旋转首末格 + 全池最优格）。"""
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    H = gr.hpwl_gain_maps(model, placed, comp, spec)
    refs = {f.ref for f in placed}
    base = model.hpwl(refs)

    picks = []
    for k in range(4):
        ck = np.argwhere(L[k])
        if len(ck):
            picks += [(k, ck[0]), (k, ck[-1])]
    kb, ib, jb = np.unravel_index(int(np.argmax(np.where(L, H, -np.inf))), H.shape)
    picks.append((int(kb), np.array([ib, jb])))

    for k, (i, j) in picks:
        x, y = spec.mm_of(int(i), int(j))
        pred = float(H[k][int(i), int(j)])
        act = base - _hpwl_with(model, refs, comp, x, y, int(k))
        assert pred == pytest.approx(act, abs=1e-4), \
            f"{tag}：H[{k}][{i},{j}] 预测 {pred:.6f} ≠ 独立重算 {act:.6f}"


@pytest.mark.parametrize("tag", IDS)
def test_dispersion_matches_direct_construction(tag, scenes):
    """E5：离散度分量 `Δσ` 与"直接构造全部 net 中心"一致（格中心 mm 为锚点）。"""
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    Dd = gr.dispersion_delta(model, placed, comp, spec, L)
    refs = {f.ref for f in placed}
    sb = _sigma_before(model, refs)

    picks = [(_expert_cell(comp, spec, L)[:3])]
    for k in range(4):
        ck = np.argwhere(L[k])
        if len(ck):
            picks.append((k, int(ck[len(ck) // 2, 0]), int(ck[len(ck) // 2, 1])))
    for k, i, j in picks:
        assert L[k][i, j]
        x, y = spec.mm_of(i, j)
        exp = _sigma_after(model, refs, comp, k, x, y) - sb
        assert float(Dd[k][i, j]) == pytest.approx(exp, abs=1e-9), \
            f"{tag}：Δσ[{k}][{i},{j}] = {float(Dd[k][i, j]):.6e} ≠ 直接构造 {exp:.6e}"


def test_degenerate_guard_synthetic(scenes):
    """E6：退化守卫——单合法格 / 空合法集（防 0/0 与 NaN）。"""
    model, comp, placed, spec = scenes["lr1388_U7"]
    L = gr.legal_masks(model, placed, comp, spec)

    one = np.zeros_like(L)
    one[0, 50, 60] = True
    G1, d1 = gr.gain_maps(model, placed, comp, spec, legal=one)
    assert d1 is True, "单合法格 ⇒ 池内两分量都恒定 ⇒ 必须判退化"
    assert float(G1[0, 50, 60]) == pytest.approx(0.5), "退化分量应给常数 0.5"
    assert bool(np.isneginf(G1[0, 0, 0])), "非法格仍为 −inf"
    assert int(np.isfinite(G1).sum()) == 1

    zero = np.zeros_like(L)
    G0, d0 = gr.gain_maps(model, placed, comp, spec, legal=zero)
    assert d0 is True and bool(np.isneginf(G0).all()), "空合法集（死锁）⇒ 全 −inf 且退化"


def test_degenerate_guard_empty_board(scenes):
    """E7：真实退化——空板 ⇒ 该件的 net 上都没有已放 pad ⇒ 两分量同时退化、G ≡ 0.5。"""
    model, comp, _placed, spec = scenes["lr1388_U7"]
    G, degen = gr.gain_maps(model, [], comp, spec)
    assert degen is True, "空板上该件的 net 全都无已放 pad ⇒ 应判退化"
    fin = np.isfinite(G)
    assert fin.any() and np.allclose(G[fin], 0.5), "退化时合法格 G 必须恒为 0.5"
    assert bool(np.isneginf(G[~fin]).all())
