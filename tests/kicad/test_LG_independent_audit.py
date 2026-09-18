# -*- coding: utf-8 -*-
"""`L` / `G` 的**第三方独立验收**（把 probe151/154/155 固化成回归测试）。

目的：**不信任既有测试**（既有测试大量与实现同源）。本文件的几何、索引换算、判定、
逐点重算**全部独立重写**，只把 `env/grid_render.py` 当**被验收对象**调用；基准优先级：
**KiCad 本体（pcbnew 实测）> 独立实现**。

验收纪律（台账 §3.9，两次踩坑换来的）：
  1. **不自证**：不得复用被测代码；优先外部权威。
  2. **样本必须非空**：每个验收都先断言"比较点数 / 采样格数 > 0"（防"空验收假绿灯"）。
  3. **身份配对必须唯一**：footprint 用构造器 ref 规则配对，pad 用**枚举序 index 对齐**
     （禁按焊盘号配对；禁用截断目录名当键 —— 两块同前缀板会被合并）。
  4. **偏差必须归类给量**：`danger`（放过违规）恒 0；`over`（过度封锁）必须逐例落在
     "阈值平局 / 板框边缘平局"上，并打印 `|余量|`。
  5. **误差给量级解释**：H 是 float32、disp/G 是 float64 ⇒ 预期误差分别 ~1e-6 / ~1e-12 mm。

运行（重档：旋转专项需 `PCB_RL_SLOW=1`）：
    $env:KICAD_PATH="E:\\Kicad"; python -m pytest tests/kicad/test_LG_independent_audit.py -q
    $env:PCB_RL_SLOW="1"; python -m pytest tests/kicad/test_LG_independent_audit.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr                      # noqa: E402  ← 被验收对象
from utils.kicad_utils import pcb_model as pm          # noqa: E402

pytestmark = pytest.mark.require_kicad
SLOW = pytest.mark.skipif(not os.environ.get("PCB_RL_SLOW"),
                          reason="重档验收（旋转专项）：置 PCB_RL_SLOW=1 打开")

SOFT = gr.SOFT_GAP_MM
TOL_MM = 1e-9          # 几何/平局容差（远严于板精度下限 1nm，但宽于实测 1e-13）

SCENARIOS = [
    dict(tag="lr1388_U7", board_sub="lr1388", drop=("U7", "USBC1"), comp="U7", over_golden=0),
    dict(tag="bliysfyt_U1", board_sub="bliysfyt", drop=("U1", "C1"), comp="U1", over_golden=0),
    dict(tag="jackt_IMU_U1", board_sub="jackt", drop=("U1",), comp="U1", over_golden=0),
    dict(tag="wzh_main_Q2", board_sub="wzh2814685", drop=("Q2", "Q1", "C1"), comp="Q2", over_golden=24),
    dict(tag="verysunshine_U4", board_sub="verysunshine", drop=("U9", "U4"), comp="U4", over_golden=0),
    dict(tag="coloreasypico_D2", board_sub="coloreasypico2",
         drop=("R2", "D2", "D3", "R8"), comp="D2", over_golden=40),
]
IDS = [s["tag"] for s in SCENARIOS]

# 20 板里的**全部 16 个非正交件**（金表；由 probe154 实测锁定）：(board_sub, ref, 原始角, 转换角 idx)
NONORTH = [
    ("MiniMos", "U15", -45.0, 3),
    ("Mini控制板", "C13", -135.0, 2),
    ("Mini控制板", "U14", -135.0, 2),
    ("Mini控制板", "U16", 45.0, 0),
    ("Mini控制板", "R36", -45.0, 3),
    ("Mini控制板", "SW1", 45.0, 0),
    ("esp32s3r8n8", "C19", 135.0, 1),
    ("esp32s3r8n8", "C50", -135.0, 2),
    ("esp32s3r8n8", "L6", 45.0, 0),
    ("esp32s3r8n8", "U3", -45.0, 3),
    ("esp32s3r8n8", "C47", 135.0, 1),
    ("esp32s3r8n8", "L4", -45.0, 3),
    ("gd32vw553", "C55", -45.0, 3),
    ("gd32vw553", "U1", -135.0, 2),
    ("lyg0927", "X1", -0.6172, 0),      # 全数据集唯一"非等距"角（-0.6172°，不是 -0.62）
    ("lyg0927", "U1", -45.0, 3),
]

# 旋转专项的"待放件本身是非正交件"场面（drop = 该件 ∪ pad 数最多的 3 件；不够再退到 8 件）
ROT_SCENES = [("Mini控制板", "U14"), ("Mini控制板", "SW1"),
              ("gd32vw553", "U1"), ("lyg0927", "X1"), ("lyg0927", "U1")]


# ========================= 独立几何（= 定义，不复用被测代码） =========================

def rel_rot(local_aabb, k):
    """局部 AABB（相对锚点、θ=0）→ 相对锚点的旋转后 AABB：板坐标 = 锚点 + R(−θ)·局部。"""
    th = math.radians(k * 90.0)
    c, s = math.cos(th), math.sin(th)
    x1, y1, x2, y2 = local_aabb
    xs, ys = [], []
    for lx, ly in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        xs.append(lx * c + ly * s)
        ys.append(-lx * s + ly * c)
    return (min(xs), min(ys), max(xs), max(ys))


def rel_pt(rel_offset, k):
    """局部偏移（相对锚点、θ=0）→ 相对锚点的旋转后中心。"""
    th = math.radians(k * 90.0)
    c, s = math.cos(th), math.sin(th)
    lx, ly = rel_offset
    return (lx * c + ly * s, -lx * s + ly * c)


def _layers_mask(layers):
    m = 0
    for L in layers:
        m |= 1 << int(L)
    return m


def audit_geometry(path, model, n_comp=20):
    """① KiCad 本体对拍：把 footprint 摆到 4 个转换角，读 pcbnew 自己的 AABB/中心。"""
    brd = pm.pcbnew.LoadBoard(path)
    builder = pm.PcbModelBuilder()
    canon = {}
    for idx, fp in enumerate(brd.GetFootprints()):
        canon[builder._footprint_ref(fp, idx)] = fp
    picks = sorted([f for f in model.footprints if f.pads],
                   key=lambda f: (-len(f.pads), f.ref))[:n_comp]
    worst_a = worst_c = 0.0
    nfp = npt = nskip = 0
    for f in picks:
        fp = canon.get(f.ref)
        if fp is None:
            nskip += 1
            continue
        live = list(fp.Pads())
        if len(live) != len(f.pads):
            nskip += 1
            continue
        nfp += 1
        saved = fp.GetOrientationDegrees()
        try:
            for k in range(4):
                fp.SetOrientationDegrees(k * 90.0)
                for p, lp in zip(f.pads, live):        # ← index 对齐（纪律 3）
                    if not p.local_aabb:
                        continue
                    got = pm.PcbModelBuilder._aabb(lp.GetBoundingBox())
                    ra = rel_rot(p.local_aabb, k)
                    exp = (ra[0] + f.anchor[0], ra[1] + f.anchor[1],
                           ra[2] + f.anchor[0], ra[3] + f.anchor[1])
                    worst_a = max(worst_a, max(abs(got[i] - exp[i]) for i in range(4)))
                    gc = (pm._mm(lp.GetPosition().x), pm._mm(lp.GetPosition().y))
                    rc = rel_pt(p.rel_offset, k)
                    ec = (rc[0] + f.anchor[0], rc[1] + f.anchor[1])
                    worst_c = max(worst_c, math.hypot(gc[0] - ec[0], gc[1] - ec[1]))
                    npt += 1
        finally:
            fp.SetOrientationDegrees(saved)
    return nfp, npt, nskip, worst_a, worst_c


# ========================= 独立实现 L（暴力栅格化，无差分数组） =========================

def independent_legal(model, placed, comp, spec):
    m = spec.m
    X = spec.x0 + (np.arange(m) + 0.5) * spec.cell_x
    Y = spec.y0 + (np.arange(m) + 0.5) * spec.cell_y
    bx = model.board_outline.bbox

    P = [(np.array(p.aabb), _layers_mask(p.layers)) for f in placed for p in f.pads if p.aabb]
    V = [(np.array(v.aabb), _layers_mask(v.layers), int(v.netcode or 0)) for v in model.vias]
    # 口径 (A)：开槽随件 —— 只用【当前局面】的禁布区（未放置件的槽不在板上，§0.6 p）
    kps = pm.state_keepouts(model, placed)
    K = [(np.array(k.aabb), _layers_mask(k.layers)) for k in kps]
    P_a = (np.array([q[0] for q in P]), np.array([q[1] for q in P])) if P else None
    V_a = (np.array([q[0] for q in V]), np.array([q[1] for q in V]),
           np.array([q[2] for q in V])) if V else None
    K_a = (np.array([q[0] for q in K]), np.array([q[1] for q in K])) if K else None
    own_slots = np.array([k.owner_ref == comp.ref for k in kps]) if K else np.zeros(0, bool)

    out = np.zeros((4, m, m), dtype=bool)
    for k in range(4):
        inb = np.ones((m, m), dtype=bool)
        forbid = np.zeros((m, m), dtype=bool)
        for pad in comp.pads:
            if not pad.local_aabb:
                continue
            rel = rel_rot(pad.local_aabb, k)
            lo_x, hi_x = bx[0] - rel[0], bx[2] - rel[2]
            lo_y, hi_y = bx[1] - rel[1], bx[3] - rel[3]
            ok = np.zeros((m, m), dtype=bool)
            ok[int(np.searchsorted(X, lo_x, "left")):int(np.searchsorted(X, hi_x, "right")),
               int(np.searchsorted(Y, lo_y, "left")):int(np.searchsorted(Y, hi_y, "right"))] = True
            inb &= ok
            pm_ = _layers_mask(pad.layers)
            netc = int(pad.netcode or 0)
            batches = []
            if P_a is not None:
                sel = (P_a[1] & pm_) != 0
                if sel.any():
                    batches.append(P_a[0][sel])
            if V_a is not None:
                sel = (V_a[1] & pm_) != 0
                if netc != 0:
                    sel &= (V_a[2] != 0) & (V_a[2] != netc)
                else:
                    sel = np.zeros_like(sel)
                if sel.any():
                    batches.append(V_a[0][sel])
            if K_a is not None:
                sel = ((K_a[1] & pm_) != 0) & ~own_slots
                if sel.any():
                    batches.append(K_a[0][sel])
            for boxes in batches:
                lx = boxes[:, 0] - rel[2] - SOFT
                hx = boxes[:, 2] - rel[0] + SOFT
                ly = boxes[:, 1] - rel[3] - SOFT
                hy = boxes[:, 3] - rel[1] + SOFT
                a0 = np.searchsorted(X, lx, "right")
                a1 = np.searchsorted(X, hx, "left")
                b0 = np.searchsorted(Y, ly, "right")
                b1 = np.searchsorted(Y, hy, "left")
                keep = (a1 > a0) & (b1 > b0)
                for u, v, w, z in zip(a0[keep], a1[keep], b0[keep], b1[keep]):
                    forbid[int(u):int(v), int(w):int(z)] = True
        out[k] = inb & ~forbid
    return out


def cell_min_slack(model, placed, comp, spec, k, i, j):
    """该格的"最小余量"（mm）：出界 ≥0 且各 gap−SOFT ≥0 ⇒ 独立实现判合法；≈0 ⇒ 平局。"""
    x, y = spec.mm_of(i, j)
    best, why = float("inf"), ""
    bx = model.board_outline.bbox
    for pad in comp.pads:
        if not pad.local_aabb:
            continue
        rel = rel_rot(pad.local_aabb, k)
        box = (x + rel[0], y + rel[1], x + rel[2], y + rel[3])
        for s, w in ((box[0] - bx[0], "board.x1"), (bx[2] - box[2], "board.x2"),
                     (box[1] - bx[1], "board.y1"), (bx[3] - box[3], "board.y2")):
            if s < best:
                best, why = s, w
        pm_ = _layers_mask(pad.layers)
        netc = int(pad.netcode or 0)
        for f in placed:
            for q in f.pads:
                if not q.aabb or not (_layers_mask(q.layers) & pm_):
                    continue
                g = max(box[0] - q.aabb[2], q.aabb[0] - box[2],
                        box[1] - q.aabb[3], q.aabb[1] - box[3]) - SOFT
                if g < best:
                    best, why = g, f"pad({f.ref}.{q.number})"
        for v in model.vias:
            if not (_layers_mask(v.layers) & pm_):
                continue
            if netc == 0 or not v.netcode or v.netcode == netc:
                continue
            g = max(box[0] - v.aabb[2], v.aabb[0] - box[2],
                    box[1] - v.aabb[3], v.aabb[1] - box[3]) - SOFT
            if g < best:
                best, why = g, "via"
        for kp in pm.state_keepouts(model, placed):
            if not (_layers_mask(kp.layers) & pm_) or kp.owner_ref == comp.ref:
                continue
            g = max(box[0] - kp.aabb[2], kp.aabb[0] - box[2],
                    box[1] - kp.aabb[3], kp.aabb[1] - box[3]) - SOFT
            if g < best:
                best, why = g, f"keepout({kp.kind})"
    return best, why


# ========================= 独立算 G（按定义直接算） =========================

def _hpwl(centers):
    tot = 0.0
    for pts in centers.values():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        tot += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return tot


def _sigma(centers):
    cs = []
    for pts in centers.values():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cs.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    n = len(cs)
    if n == 0:
        return 0.0
    cx = sum(c[0] for c in cs) / n
    cy = sum(c[1] for c in cs) / n
    return math.sqrt(max(sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in cs) / n, 0.0))


def base_centers(model, placed_refs):
    out = {}
    for net in model.nets.values():
        if net.is_power:
            continue
        out[net.net_id] = [(p.center[0], p.center[1]) for p in net.pads if p.ref in placed_refs]
    return out


def centers_with(model, comp, k, x, y, base):
    out = {nid: list(v) for nid, v in base.items()}
    for pad in comp.pads:
        if not pad.local_aabb or pad.net_id is None:
            continue
        net = model.nets.get(pad.net_id)
        if net is None or net.is_power:
            continue
        d = rel_pt(pad.rel_offset, k)
        out.setdefault(pad.net_id, []).append((x + d[0], y + d[1]))
    return out


def _z(s, lo, hi):
    return np.full_like(s, 0.5) if hi - lo < 1e-12 else np.clip((s - lo) / (hi - lo), 0.0, 1.0)


# ========================= 场面加载 =========================

def _scen(tag):
    return next(s for s in SCENARIOS if s["tag"] == tag)


def _board(paths, sub):
    hits = [p for p in paths if sub in os.path.basename(os.path.dirname(p))]
    assert hits, f"未找到板：{sub}"
    return hits[0]


@pytest.fixture(scope="session")
def scenes(all_kicad_pcb_paths):
    """`tag -> (model, comp, placed, spec)`（与 test_grid_render_real_boards 同一套场面）。"""
    out = {}
    for sc in SCENARIOS:
        path = _board(all_kicad_pcb_paths, sc["board_sub"])
        model = pm.build_model(pm.pcbnew.LoadBoard(path), path)
        byref = {f.ref: f for f in model.footprints}
        assert sc["comp"] in byref, f"{sc['tag']}：板内无 {sc['comp']}"
        placed = [f for f in model.footprints if f.ref not in set(sc["drop"])]
        out[sc["tag"]] = (model, byref[sc["comp"]], placed, gr.grid_spec(model, m=gr.GRID_M))
    return out


# ========================= A. 几何 vs KiCad 本体（外部权威） =========================

@pytest.mark.parametrize("tag", IDS)
def test_geometry_matches_kicad_authority(tag, all_kicad_pcb_paths):
    path = _board(all_kicad_pcb_paths, _scen(tag)["board_sub"])
    model = pm.build_model(pm.pcbnew.LoadBoard(path), path)
    nfp, npt, nskip, wa, wc = audit_geometry(path, model)
    assert npt > 0 and nfp > 0, "样本为空（纪律 2）"
    assert nskip == 0, f"{tag}: 有 {nskip} 个 footprint 未能配对"
    assert max(wa, wc) < TOL_MM, f"{tag}: 反解 vs KiCad 本体差 {wa:.3e}/{wc:.3e} mm"


# ========================= B. L 独立全量互证 =========================

@pytest.mark.parametrize("tag", IDS)
def test_L_matches_independent_rasterization(tag, scenes):
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    Li = independent_legal(model, placed, comp, spec)
    assert Li.any() and L.any(), "样本为空（纪律 2）"

    danger = int((L & ~Li).sum())
    assert danger == 0, f"{tag}: L 放过了 {danger} 个违规格（危险方向必须恒 0）"

    over = ~L & Li
    assert int(over.sum()) == _scen(tag)["over_golden"], (
        f"{tag}: 过度封锁数变了（got {int(over.sum())}，金表 {_scen(tag)['over_golden']}）")
    ko, io, jo = np.where(over)
    worst = 0.0
    for a, b, c in zip(ko, io, jo):
        s, _why = cell_min_slack(model, placed, comp, spec, int(a), int(b), int(c))
        worst = max(worst, abs(s))
    assert worst < TOL_MM, f"{tag}: 过度封锁不是平局（最大 |余量| = {worst:.3e} mm）"


# ========================= C/D. G 分量 + 组合层独立重算 =========================

@pytest.mark.parametrize("tag", IDS)
def test_G_matches_independent_recompute(tag, scenes):
    model, comp, placed, spec = scenes[tag]
    L = gr.legal_masks(model, placed, comp, spec)
    H = gr.hpwl_gain_maps(model, placed, comp, spec)
    Dd = gr.dispersion_delta(model, placed, comp, spec, L)
    G, _degen = gr.gain_maps(model, placed, comp, spec, legal=L)

    base = base_centers(model, {f.ref for f in placed})
    h0, s0 = _hpwl(base), _sigma(base)

    rng = np.random.default_rng(20260917)
    eH = eD = 0.0
    n_pt = 0
    for k in range(4):
        c = np.argwhere(L[k])
        if not len(c):
            continue
        hk = H[k][c[:, 0], c[:, 1]]
        dk = Dd[k][c[:, 0], c[:, 1]]
        idx = [int(np.argmax(hk)), int(np.argmin(hk)), int(np.argmax(dk)), int(np.argmin(dk))]
        idx += list(rng.choice(len(c), size=min(400, len(c)), replace=False))
        for t in idx:
            i, j = int(c[t, 0]), int(c[t, 1])
            x, y = spec.mm_of(i, j)
            ctr = centers_with(model, comp, k, x, y, base)
            eH = max(eH, abs(float(H[k][i, j]) - (h0 - _hpwl(ctr))))
            eD = max(eD, abs(float(Dd[k][i, j]) - (_sigma(ctr) - s0)))
            n_pt += 1
    assert n_pt > 100, "样本为空（纪律 2）"
    assert eH < 1e-4, f"{tag}: ΔHPWL 与独立重算差 {eH:.3e} mm（预期 ≤ float32 精度 ~1e-5）"
    assert eD < 1e-10, f"{tag}: Δσ 与独立重算差 {eD:.3e} mm（预期 ~1e-12）"

    # 组合层：有限域 = L、∈[0,1]、非法格 −inf、共用一把尺（4 旋转 max G 不应全为 1.0）
    fin = np.isfinite(G)
    assert np.array_equal(fin, L), f"{tag}: G 的有限域必须严格等于 L"
    assert G[fin].min() >= 0.0 and G[fin].max() <= 1.0
    assert bool(np.isneginf(G[~fin]).all())
    per_rot = [float(G[k][np.isfinite(G[k])].max()) for k in range(4)]
    if tag != "lr1388_U7":       # lr1388 顶部饱和（4 个旋转都触顶 1.0）属已知
        assert not all(abs(v - 1.0) < 1e-6 for v in per_rot), \
            f"{tag}: 4 个旋转 max G 全为 1.0 ⇒ 疑似退化成逐旋转归一化"


# ========================= E. 旋转专项：16 个非正交件（KiCad 本体） =========================

@SLOW
def test_nonorthogonal_geometry_matches_kicad(all_kicad_pcb_paths):
    checked = 0
    worst = 0.0
    min_shift = float("inf")
    for sub, ref, raw, idx in NONORTH:
        path = _board(all_kicad_pcb_paths, sub)
        model = pm.build_model(pm.pcbnew.LoadBoard(path), path)
        f = next((x for x in model.footprints if x.ref == ref), None)
        assert f is not None, f"{sub}: 板内无 {ref}"
        assert abs(f.rotation_deg_raw - raw) < 1e-6, \
            f"{ref}: 原始角变了 {f.rotation_deg_raw}（金表 {raw}）"
        assert f.rotation_index == idx, f"{ref}: 转换角变了 {f.rotation_index}（金表 {idx}）"

        brd = pm.pcbnew.LoadBoard(path)
        builder = pm.PcbModelBuilder()
        canon = {}
        for i, fp in enumerate(brd.GetFootprints()):
            canon[builder._footprint_ref(fp, i)] = fp
        fp = canon[ref]
        saved = fp.GetOrientationDegrees()
        try:
            fp.SetOrientationDegrees(raw)
            raw_c = [(pm._mm(q.GetPosition().x), pm._mm(q.GetPosition().y)) for q in fp.Pads()]
            fp.SetOrientationDegrees(idx * 90.0)
            conv_c = [(pm._mm(q.GetPosition().x), pm._mm(q.GetPosition().y)) for q in fp.Pads()]
            min_shift = min(min_shift, max(math.hypot(a[0] - b[0], a[1] - b[1])
                                           for a, b in zip(raw_c, conv_c)))
            for k in range(4):
                fp.SetOrientationDegrees(k * 90.0)
                for pad, lp in zip(f.pads, list(fp.Pads())):
                    if not pad.local_aabb:
                        continue
                    rc = rel_pt(pad.rel_offset, k)
                    ec = (f.anchor[0] + rc[0], f.anchor[1] + rc[1])
                    gc = (pm._mm(lp.GetPosition().x), pm._mm(lp.GetPosition().y))
                    worst = max(worst, math.hypot(gc[0] - ec[0], gc[1] - ec[1]))
                    ra = rel_rot(pad.local_aabb, k)
                    ea = (ra[0] + f.anchor[0], ra[1] + f.anchor[1],
                          ra[2] + f.anchor[0], ra[3] + f.anchor[1])
                    ga = pm.PcbModelBuilder._aabb(lp.GetBoundingBox())
                    worst = max(worst, max(abs(ga[i] - ea[i]) for i in range(4)))
                    checked += 1
        finally:
            fp.SetOrientationDegrees(saved)
    assert checked > 0, "样本为空（纪律 2）"
    assert worst < TOL_MM, f"非正交件反解 vs KiCad 差 {worst:.3e} mm"
    # 这条断言保证"确实走到了归一化路径"：归一化必须真的让焊盘移动
    assert min_shift > 1e-3, f"归一化未产生位移（min {min_shift:.3e} mm）⇒ 本测试失去意义"


# ========================= F. 旋转专项：待放件本身是非正交件时的 L/G =========================

@SLOW
@pytest.mark.parametrize("sub,ref", ROT_SCENES)
def test_nonorthogonal_scene_LG(sub, ref, all_kicad_pcb_paths):
    path = _board(all_kicad_pcb_paths, sub)
    model = pm.build_model(pm.pcbnew.LoadBoard(path), path)
    byref = {f.ref: f for f in model.footprints}
    comp = byref[ref]
    assert abs(comp.rotation_deg_raw % 90.0) > 1e-9, f"{ref} 不是非正交件"
    top3 = sorted(model.footprints, key=lambda f: -len(f.pads))[:3]
    placed = [f for f in model.footprints if f.ref not in ({ref} | {f.ref for f in top3})]
    spec = gr.grid_spec(model, m=gr.GRID_M)
    L = gr.legal_masks(model, placed, comp, spec)
    if not L.any():                       # 空间不足 ⇒ 再拿走 pad 最多的 8 件（保证场面有效）
        top8 = sorted(model.footprints, key=lambda f: -len(f.pads))[:8]
        placed = [f for f in model.footprints if f.ref not in ({ref} | {f.ref for f in top8})]
        L = gr.legal_masks(model, placed, comp, spec)

    Li = independent_legal(model, placed, comp, spec)
    assert L.any(), f"{sub}/{ref}: 无合法格（场面无效，纪律 2）"
    danger = int((L & ~Li).sum())
    assert danger == 0, f"{sub}/{ref}: L 放过 {danger} 个违规格"
    over = ~L & Li
    ko, io, jo = np.where(over)
    worst = 0.0
    for a, b, c in zip(ko, io, jo):
        s, _why = cell_min_slack(model, placed, comp, spec, int(a), int(b), int(c))
        worst = max(worst, abs(s))
    assert worst < TOL_MM, f"{sub}/{ref}: over 不是平局（最大 |余量| = {worst:.3e} mm）"

    H = gr.hpwl_gain_maps(model, placed, comp, spec)
    Dd = gr.dispersion_delta(model, placed, comp, spec, L)
    base = base_centers(model, {f.ref for f in placed})
    h0, s0 = _hpwl(base), _sigma(base)
    rng = np.random.default_rng(7)
    eH = eD = 0.0
    n_pt = 0
    for k in range(4):
        c = np.argwhere(L[k])
        if not len(c):
            continue
        take = c[rng.choice(len(c), size=min(300, len(c)), replace=False)]
        for a, b in take:
            x, y = spec.mm_of(int(a), int(b))
            ctr = centers_with(model, comp, k, x, y, base)
            eH = max(eH, abs(float(H[k][int(a), int(b)]) - (h0 - _hpwl(ctr))))
            eD = max(eD, abs(float(Dd[k][int(a), int(b)]) - (_sigma(ctr) - s0)))
            n_pt += 1
    assert n_pt > 0, "样本为空（纪律 2）"     # 本场面合法格少（如 lyg0927/U1 仅 98 格）⇒ 只要求非空
    assert eH < 1e-4 and eD < 1e-10, f"{sub}/{ref}: G 分量差 {eH:.3e}/{eD:.3e} mm"
