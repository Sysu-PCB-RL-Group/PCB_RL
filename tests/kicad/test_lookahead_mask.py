# -*- coding: utf-8 -*-
"""`env/lookahead.py`（单步预测掩膜）的回归测试 —— 台账 §0.6 未决 n。

验收纪律（台账 §3.9）：
  1. **不自证**：参照实现**独立重写**（不复用 `lookahead` 的任何函数），且**禁布区槽随件平移**
     （早期探针漏了这一步 ⇒ 会假"活"）。
  2. **样本非空**：每个对拍都断言抽到的候选数 > 0。
  3. **偏差归类给量**：只允许 `mask ⊆ L`；对拍要求**零不一致**。
  4. 边界：`L_next0` 全空 ⇒ **全部 pass**（mask == L，不清空动作空间）。
"""
import glob
import math
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr                    # noqa: E402
from env import lookahead as la                      # noqa: E402
from env.order_plan import plan_order                # noqa: E402
from utils.kicad_utils import pcb_model as pm        # noqa: E402

pytestmark = pytest.mark.require_kicad
SLOW = pytest.mark.skipif(not os.environ.get("PCB_RL_SLOW"),
                          reason="重档：空板（稀疏极端）压力测试，置 PCB_RL_SLOW=1 打开")
SOFT = gr.SOFT_GAP_MM
SUBS = ["bliysfyt", "jackt", "verysunshine"]


# ---------------- 独立参照（不复用 lookahead 的实现） ----------------

def _boxes(comp, k):
    pb, pmk = [], []
    for pad in comp.pads:
        if pad.local_aabb:
            pb.append(pm.rotate_aabb(pad.local_aabb, k * 90.0, (0.0, 0.0)))
            pmk.append(sum(1 << int(L) for L in pad.layers))
    return np.asarray(pb, dtype=float).reshape(-1, 4), np.asarray(pmk, dtype=np.int64)


def _slots(comp, model, k):
    sb, smk = [], []
    th = math.radians(comp.rotation_index * 90.0)
    c, s = math.cos(th), math.sin(th)
    for kp in model.keepouts:
        if kp.owner_ref != comp.ref:
            continue
        xs, ys = [], []
        for bx, by in ((kp.aabb[0], kp.aabb[1]), (kp.aabb[2], kp.aabb[1]),
                       (kp.aabb[2], kp.aabb[3]), (kp.aabb[0], kp.aabb[3])):
            dx, dy = bx - comp.anchor[0], by - comp.anchor[1]
            xs.append(dx * c - dy * s)
            ys.append(dx * s + dy * c)
        sb.append(pm.rotate_aabb((min(xs), min(ys), max(xs), max(ys)), k * 90.0, (0.0, 0.0)))
        smk.append(sum(1 << int(L) for L in kp.layers))
    return np.asarray(sb, dtype=float).reshape(-1, 4), np.asarray(smk, dtype=np.int64)


def _hit(a, am, b, bm):
    if len(a) == 0 or len(b) == 0:
        return False
    lay = (am[:, None] & bm[None, :]) != 0
    if not lay.any():
        return False
    dx = np.maximum(a[:, 0, None] - b[None, :, 2], b[None, :, 0] - a[:, 2, None])
    dy = np.maximum(a[:, 1, None] - b[None, :, 3], b[None, :, 1] - a[:, 3, None])
    return bool(((np.maximum(dx, dy) < SOFT) & lay).any())


def alive_ref(model, now, k, px, py, nxt, Ln, spec):
    """独立参照：放 `comp_now@(k,px,py)` 后，`nxt` 是否仍有合法格（4 旋转任一）。"""
    an, am = _boxes(now, k)
    asl, asm = _slots(now, model, k)
    a_pads = an + np.array([px, py, px, py])
    a_slots = asl + np.array([px, py, px, py]) if len(asl) else asl
    for kf in range(4):
        bn, bm = _boxes(nxt, kf)
        bsl, bsm = _slots(nxt, model, kf)
        for ci, cj in np.argwhere(Ln[kf]):
            x, y = spec.mm_of(int(ci), int(cj))
            sh = np.array([x, y, x, y])
            b_pads = bn + sh
            b_slots = bsl + sh if len(bsl) else bsl      # ← 槽必须跟着 cell 平移！
            if _hit(a_pads, am, b_pads, bm):
                continue
            if _hit(a_slots, asm, b_pads, bm):
                continue
            # ⚠️ **不检查** `now.pad ↔ next.slot`：口径 (A) 下 `nxt` 自家槽对自己**豁免**
            #    （`legal_masks`/`Judge` 都 `owner_ref == comp.ref ⇒ 跳过`）⇒ 该项多算会假报警
            #    （2026-09-18 校准；`next` 自带槽的局面见 test_mask_matches_recompute_sparse）。
            return True
    return False


# ---------------- 场面 ----------------

@pytest.fixture(scope="session")
def scenes(all_kicad_pcb_paths):
    out = {}
    for sub in SUBS:
        hits = [p for p in all_kicad_pcb_paths
                if sub in os.path.basename(os.path.dirname(p))]
        assert hits, f"未找到板 {sub}"
        model = pm.build_model(pm.pcbnew.LoadBoard(hits[0]), hits[0])
        byref = {f.ref: f for f in model.footprints}
        order = [r for r in plan_order(model) if r in byref]
        now, nxt = byref[order[0]], byref[order[1]]
        placed = [f for f in model.footprints if f.ref not in {now.ref, nxt.ref}]
        spec = gr.grid_spec(model, m=gr.GRID_M)
        out[sub] = (model, now, nxt, placed, spec)
    return out


@pytest.mark.parametrize("sub", SUBS)
def test_mask_is_subset_of_L_and_keeps_actions(sub, scenes):
    model, now, nxt, placed, spec = scenes[sub]
    L = gr.legal_masks(model, placed, now, spec)
    Ln = gr.legal_masks(model, placed, nxt, spec)
    mask, info = la.lookahead_mask(model, placed, now, nxt, spec, legal=L, legal_next=Ln)

    assert mask.shape == L.shape and mask.dtype == bool
    assert not (mask & ~L).any(), "mask 必须 ⊆ L（更严格）"
    assert info["dead_next"] is False
    assert int(Ln.sum()) > 0, "本场景下一个件本来就有位置"
    if int(L.sum()) > 0:
        assert int(mask.sum()) > 0, "不允许把动作空间清空"


@pytest.mark.parametrize("sub", SUBS)
def test_mask_matches_independent_reference(sub, scenes):
    """与独立参照逐候选对拍（抽样，含每旋转的候选）。"""
    model, now, nxt, placed, spec = scenes[sub]
    L = gr.legal_masks(model, placed, now, spec)
    Ln = gr.legal_masks(model, placed, nxt, spec)
    mask, _info = la.lookahead_mask(model, placed, now, nxt, spec, legal=L, legal_next=Ln)

    rng = np.random.default_rng(20260918)
    picks = []
    for k in range(4):
        c = np.argwhere(L[k])
        if len(c):
            take = c[rng.choice(len(c), size=min(40, len(c)), replace=False)]
            picks += [(k, int(a), int(b)) for a, b in take]
    assert len(picks) > 0, "样本为空（纪律 2）"

    bad = []
    for k, i, j in picks:
        px, py = spec.mm_of(i, j)
        ref = alive_ref(model, now, k, px, py, nxt, Ln, spec)
        if ref != bool(mask[k][i, j]):
            bad.append((k, i, j, bool(mask[k][i, j]), ref))
    assert not bad, f"{sub}: 与独立参照不一致 {len(bad)} 个：{bad[:4]}"


def test_dead_next_passes_through(scenes):
    """边界（Owner 定）：`L_next0` 全空 ⇒ 全部 pass（保留 L 不变），不许清空动作空间。"""
    model, now, nxt, placed, spec = scenes["bliysfyt"]
    L = gr.legal_masks(model, placed, now, spec)
    zeros = np.zeros_like(L)
    mask, info = la.lookahead_mask(model, placed, now, nxt, spec, legal=L, legal_next=zeros)
    assert info["dead_next"] is True
    assert np.array_equal(mask, L), "已死未来件 ⇒ 必须原样 pass"
    assert int(mask.sum()) == int(L.sum())


@pytest.mark.parametrize("sub", SUBS)
def test_lookahead_cost_is_acceptable(sub, scenes):
    """成本守门（宽松阈值，防性能回退）：单步掩膜 ≤ 3 s；并打印实际值。"""
    model, now, nxt, placed, spec = scenes[sub]
    L = gr.legal_masks(model, placed, now, spec)
    Ln = gr.legal_masks(model, placed, nxt, spec)
    t0 = time.time()
    mask, info = la.lookahead_mask(model, placed, now, nxt, spec, legal=L, legal_next=Ln)
    dt = time.time() - t0
    cut = int(L.sum()) - int(mask.sum())
    print(f"\n[{sub}] L={int(L.sum())} L_next0={int(Ln.sum())} ⇒ 保留 {int(mask.sum())}"
          f"（砍 {cut}）｜{dt * 1000:.0f} ms/步｜kf 解决数 {info['resolved_at']}")
    assert dt < 3.0, f"{sub}: 掩膜耗时 {dt:.2f}s 超阈值"


@pytest.mark.parametrize("sub", SUBS)
def test_compress_legal_is_consistent_with_mask(sub, scenes):
    """`compress_legal`（物化版增量压缩）的判定必须与掩膜逐格一致，且 ⊆ `L_next0`。

    这是 Owner 口径的自洽检查：`L_next(after) = L_next0 \\ 新非法区`，非空 ⟺ 掩膜保留。
    """
    model, now, nxt, placed, spec = scenes[sub]
    L = gr.legal_masks(model, placed, now, spec)
    Ln = gr.legal_masks(model, placed, nxt, spec)
    mask, _info = la.lookahead_mask(model, placed, now, nxt, spec, legal=L, legal_next=Ln)

    rng = np.random.default_rng(7)
    picks = []
    for k in range(4):
        c = np.argwhere(L[k])
        if len(c):
            take = c[rng.choice(len(c), size=min(20, len(c)), replace=False)]
            picks += [(k, int(a), int(b)) for a, b in take]
    assert len(picks) > 0, "样本为空（纪律 2）"

    buf = np.zeros_like(Ln)
    for k, i, j in picks:
        after = la.compress_legal(model, now, k, i, j, nxt, Ln, spec, out=buf)
        assert not (after & ~Ln).any(), "压缩只会减少格子（单调性）"
        assert bool(after.any()) == bool(mask[k][i, j]), \
            f"({k},{i},{j}): compress_legal 非空={bool(after.any())} 与掩膜={bool(mask[k][i, j])} 不一致"


@SLOW
def test_sparse_extreme_empty_board(all_kicad_pcb_paths):
    """稀疏极端（空板）：`L_now`/`L_next0` 都是几万格 ⇒ 不许内存爆炸或超时。"""
    for sub in ("lr1388", "bliysfyt"):
        hits = [p for p in all_kicad_pcb_paths
                if sub in os.path.basename(os.path.dirname(p))]
        model = pm.build_model(pm.pcbnew.LoadBoard(hits[0]), hits[0])
        byref = {f.ref: f for f in model.footprints}
        order = [r for r in plan_order(model) if r in byref]
        now, nxt = byref[order[0]], byref[order[1]]
        spec = gr.grid_spec(model, m=gr.GRID_M)
        L = gr.legal_masks(model, [], now, spec)
        Ln = gr.legal_masks(model, [], nxt, spec)
        t0 = time.time()
        mask, info = la.lookahead_mask(model, [], now, nxt, spec, legal=L, legal_next=Ln)
        dt = time.time() - t0
        print(f"\n[空板 {sub}] L={int(L.sum())} L_next0={int(Ln.sum())}"
              f" ⇒ 保留 {int(mask.sum())}｜{dt * 1000:.0f} ms")
        assert mask.any() and dt < 10.0


# ---------------- 稀疏局面的正确性（2026-09-18：唯一会大量走【精确路径】且"砍"很多的场面） ----------------

def _sparse_scene(all_kicad_pcb_paths, sub, now_ref=None, next_ref=None):
    hits = [p for p in all_kicad_pcb_paths if sub in os.path.basename(os.path.dirname(p))]
    assert len(hits) == 1, f"{sub}: 需唯一匹配（实际 {len(hits)}）"
    model = pm.build_model(pm.pcbnew.LoadBoard(hits[0]), hits[0])
    byref = {f.ref: f for f in model.footprints}
    order = [r for r in plan_order(model) if r in byref]
    seq = order[:2] if now_ref is None else [now_ref, next_ref]
    return model, byref[seq[0]], byref[seq[1]], gr.grid_spec(model, m=gr.GRID_M)


SPARSE_SCENES = [("bliysfyt", None, None),          # 空板：砍 ~850（占 10%），精确路径主力
                 ("lr1388", "SW8", "USBC1")]         # next 自带 2 槽（覆盖"排除 next.slot"的口径）


@pytest.mark.parametrize("sub,now_ref,next_ref", SPARSE_SCENES)
def test_mask_matches_recompute_sparse(sub, now_ref, next_ref, all_kicad_pcb_paths):
    """稀疏局面：**被砍候选**（全查或抽 120）+ **保留候选**（抽 60）与"从零重算"逐候选对拍。

    真相 = `legal_masks(model, [now@(k,i,j)], nxt, spec, keepouts=state_keepouts(...))`（真放下去后重算）。
    两个方向都必须是 0：**误留**（危险，掩膜说活、实际堵死）/ **误杀**（丢动作）。
    """
    model, now, nxt, spec = _sparse_scene(all_kicad_pcb_paths, sub, now_ref, next_ref)
    L = gr.legal_masks(model, [], now, spec)
    Ln = gr.legal_masks(model, [], nxt, spec)
    mask, info = la.lookahead_mask(model, [], now, nxt, spec, legal=L, legal_next=Ln)
    cut = [tuple(int(v) for v in t) for t in np.argwhere(L & ~mask)]
    kept = [tuple(int(v) for v in t) for t in np.argwhere(mask)]
    rng = np.random.default_rng(20260918)
    cut_pick = cut if len(cut) <= 120 else [cut[t] for t in sorted(rng.choice(len(cut), 120, False))]
    kept_pick = [kept[t] for t in sorted(rng.choice(len(kept), min(60, len(kept)), False))]

    bad = []
    for k, i, j in cut_pick + kept_pick:
        x, y = spec.mm_of(i, j)
        at = pm.place_at(now, (x, y), k)
        after = gr.legal_masks(model, [at], nxt, spec,
                               keepouts=pm.state_keepouts(model, [at]))
        if bool(after.any()) != bool(mask[k][i, j]):
            bad.append((k, i, j, bool(mask[k][i, j]), bool(after.any())))
    print(f"\n[{sub}/{now.ref}→{nxt.ref}] L={int(L.sum())} 砍={len(cut)}（查 {len(cut_pick)}）"
          f" 保留={len(kept)}（查 {len(kept_pick)}）⇒ 不一致 {len(bad)}")
    assert len(cut_pick) + len(kept_pick) > 0, "样本为空（纪律 2）"
    assert not bad, f"{sub}: 与'从零重算'不一致 {len(bad)}：{bad[:4]}"


@SLOW
@pytest.mark.parametrize("sub,now_ref,next_ref", SPARSE_SCENES)
def test_mask_matches_recompute_sparse_all_cuts(sub, now_ref, next_ref, all_kicad_pcb_paths):
    """重档：稀疏局面的**被砍候选全部** + 保留候选抽 300 与"从零重算"对拍。

    （不"全枚举保留"是因为稀疏局面可达 8.8 万候选 × ~15ms = 22 分钟；而"砍"是唯一会漏的方向。）
    """
    model, now, nxt, spec = _sparse_scene(all_kicad_pcb_paths, sub, now_ref, next_ref)
    L = gr.legal_masks(model, [], now, spec)
    Ln = gr.legal_masks(model, [], nxt, spec)
    mask, _info = la.lookahead_mask(model, [], now, nxt, spec, legal=L, legal_next=Ln)
    cut = [tuple(int(v) for v in t) for t in np.argwhere(L & ~mask)]
    kept = [tuple(int(v) for v in t) for t in np.argwhere(mask)]
    rng = np.random.default_rng(11)
    kept_pick = [kept[t] for t in sorted(rng.choice(len(kept), min(300, len(kept)), False))]
    bad = []
    for k, i, j in cut + kept_pick:
        x, y = spec.mm_of(i, j)
        at = pm.place_at(now, (x, y), k)
        after = gr.legal_masks(model, [at], nxt, spec, keepouts=pm.state_keepouts(model, [at]))
        if bool(after.any()) != bool(mask[k][i, j]):
            bad.append((k, i, j, bool(mask[k][i, j]), bool(after.any())))
    print(f"\n[{sub}] 砍 {len(cut)}(全查) + 保留 {len(kept_pick)}(抽) ⇒ 不一致 {len(bad)}")
    assert not bad, f"{sub}: 不一致 {bad[:4]}"


@pytest.mark.parametrize("sub,now_ref,next_ref", SPARSE_SCENES)
def test_conflict_table_matches_geometry(sub, now_ref, next_ref, all_kicad_pcb_paths):
    """位移表的**索引约定**必须与几何一致：`T[next−now]` ⟺ 两件在该格差下冲突（全枚举）。

    ⚠️ 2026-09-18 实锤 bug：查表用 `now − next`（符号写反）⇒ 只在"精确路径命中 + 模板不对称"时
    **漏杀**（误留危险动作）；稠密场面（砍≈0）完全看不出来。本用例直接钉住符号，且**全枚举**。
    """
    model, now, nxt, spec = _sparse_scene(all_kicad_pcb_paths, sub, now_ref, next_ref)
    n_cmp = 0
    for k in range(4):
        for kf in range(4):
            tpl = la.template_rects(model, now, k, nxt, kf)
            if not len(tpl):
                continue
            T, Ki, Kj = la.conflict_table(model, now, k, nxt, kf, spec)
            dI = np.arange(-Ki, Ki + 1)                 # next_i − now_i
            dJ = np.arange(-Kj, Kj + 1)
            DX = dI[:, None] * spec.cell_x
            DY = dJ[None, :] * spec.cell_y
            geo = np.zeros((len(dI), len(dJ)), dtype=bool)
            for x1, x2, y1, y2 in tpl:                  # 开区间，与模板同义
                geo |= (x1 < DX) & (DX < x2) & (y1 < DY) & (DY < y2)
            assert T.shape == geo.shape, f"{sub} (k={k},kf={kf}): 表形状 {T.shape} ≠ {geo.shape}"
            n = int((T != geo).sum())
            assert n == 0, (f"{sub} (k={k},kf={kf}): 位移表与几何不一致 {n} 格"
                            f"（索引约定必须是 next − now）")
            n_cmp += geo.size
    print(f"\n[{sub}] 位移表 vs 几何：全枚举 {n_cmp} 个 (k,kf,dI,dJ) ⇒ 0 不一致")
    assert n_cmp > 0, "样本为空（纪律 2）"
