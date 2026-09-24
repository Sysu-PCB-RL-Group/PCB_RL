# -*- coding: utf-8 -*-
"""reward 口径与 β 的回归测试（`env/reward.py` + `scripts/beta_calibration.py`）。

覆盖：
  A. `β_crit` 的**手算对拍**（纯函数 `beta_crit_from_curves`，独立于任何板/专家数据）；
  B. `β = margin · β_crit(该板)` 的定义（含 `β_crit = 0` 的合法板、记忆化）；
  C. 单步 reward 的**手算对拍** + `H_ref` / `D` 的取值口径（全部缓存板）；
  D. **β 的下界守门**：逐板（沿专家曲线、**独立重算**）`β ≥ β_crit` ⇒ `margin ≥ 1`；
  E. **σ 两条路径互证**：`dispersion_of(新) − dispersion_of(旧) == dispersion_delta[格]`。

⚠️ 本文件**不需要 KiCad**（板子从 P1 缓存读）；缓存缺失 ⇒ skip（与 `test_model_cache.py` 同风格）。
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr                                 # noqa: E402
from env import order_plan as op                                  # noqa: E402
from env import reward as rw                                      # noqa: E402
from utils.kicad_utils import model_cache as mc                   # noqa: E402
from utils.kicad_utils import pcb_model as pm                     # noqa: E402


@pytest.fixture(scope="module")
def cached_boards():
    try:
        boards = [e["board"] for e in mc.read_index()["boards"]]
    except mc.CacheError:
        pytest.skip("没有 data/model_cache/index.json（先跑 scripts/dump_models.py）")
    if not boards:
        pytest.skip("缓存索引里没有板")
    return boards


# ===================== A. β_crit 的手算对拍 =====================

def test_beta_crit_from_curves_matches_hand_computed():
    """`H = [0,2,5,6], H_ref = 6, α = 0` ⇒ 逐 m 的项 = 1/3, 1/3, 1/6（m=0 与 m=1 并列最大）。"""
    H, S = [0.0, 2.0, 5.0, 6.0], [0.0, 0.0, 0.0, 0.0]
    crit, m_star = rw.beta_crit_from_curves(H, S, 6.0, 1.0, alpha=0.0)
    assert crit == pytest.approx(1.0 / 3.0)
    assert m_star == 0, "并列最大时应取最先出现的 m"
    # m_min = 1 ⇒ 排除"开局即空"这一停点
    crit1, m1 = rw.beta_crit_from_curves(H, S, 6.0, 1.0, alpha=0.0, m_min=1)
    assert crit1 == pytest.approx(1.0 / 3.0) and m1 == 1


def test_beta_crit_sigma_term_can_drive_it_to_zero():
    """σ 收益足够大 ⇒ 每个 m 的项都 ≤ 0 ⇒ `β_crit = 0`（合法结果，`m_star = None`）。"""
    H, S = [0.0, 2.0, 5.0, 6.0], [0.0, 0.0, 0.0, 10.0]
    crit, m_star = rw.beta_crit_from_curves(H, S, 6.0, 1.0, alpha=1.0)
    assert crit == 0.0 and m_star is None
    # σ 项恒定 ⇒ 结论与"关掉 σ 项"一致
    S0 = [3.0, 3.0, 3.0, 3.0]
    assert rw.beta_crit_from_curves(H, S0, 6.0, 1.0, alpha=5.6)[0] == \
        pytest.approx(rw.beta_crit_from_curves(H, S0, 6.0, 1.0, alpha=0.0)[0])


# ===================== B. β = margin · β_crit(该板) =====================

def test_beta_is_margin_times_board_requirement(cached_boards):
    rw.clear_beta_cache()
    seen_zero = False
    for board in cached_boards:
        model = mc.load_model(board, allow_build=False)
        order = list(op.plan_order(model))
        crit, _m = rw.beta_crit_of(model, order)
        beta = rw.beta_of(model, order)
        assert beta == pytest.approx(rw.BETA_MARGIN * crit), f"{board}: β ≠ margin·β_crit"
        assert beta >= 0.0
        assert beta == rw.beta_of(model, order), "记忆化后取值必须一致"
        seen_zero |= (crit == 0.0)
    assert rw.BETA_MARGIN >= 1.0, "margin < 1 ⇒ 该板可能出现「停更划算」"
    assert seen_zero, "样本里应至少有一块 β_crit = 0 的板（否则该边界没被覆盖）"


def test_beta_cache_can_be_cleared(cached_boards):
    model = mc.load_model(cached_boards[0], allow_build=False)
    a = rw.beta_of(model)
    rw.clear_beta_cache()
    assert rw.beta_of(model) == pytest.approx(a)


# ===================== C. 单步 reward 与两把尺 =====================

def test_step_reward_matches_hand_computed():
    """手算：`d_hpwl=-2, H_ref=100, d_sigma=0.5, D=50, β=0.25` ⇒ `0.02 + 0.056 + 0.25`。"""
    got = rw.step_reward(d_hpwl=-2.0, d_sigma=0.5, h_ref=100.0, d_board=50.0, beta=0.25)
    assert got == pytest.approx(0.02 + rw.ALPHA * 0.5 / 50.0 + 0.25)
    assert rw.step_reward(d_hpwl=-1, d_sigma=0, h_ref=100, d_board=50, beta=0) > 0     # HPWL 变好
    assert rw.step_reward(d_hpwl=0, d_sigma=-1, h_ref=100, d_board=50, beta=0) < 0     # σ 变差
    assert rw.GAMMA == 1.0


def test_h_ref_and_d_board_on_all_boards(cached_boards):
    for board in cached_boards:
        model = mc.load_model(board, allow_build=False)
        h, d = rw.h_ref_of(model), rw.d_board_of(model)
        assert h > 0.0 and h == pytest.approx(float(model.hpwl())), f"{board}: H_ref 口径不符"
        ob = model.board_outline.bbox
        assert d == pytest.approx(math.hypot(ob[2] - ob[0], ob[3] - ob[1])) and d > 0.0


# ===================== D. β 的下界守门（独立重算） =====================

def _independent_beta_crit(model, order, H_ref, D, alpha):
    """测试自带实现（不调用被测代码）：专家曲线上逐 m 取 max。"""
    byref = {f.ref: f for f in model.footprints}
    clones = [pm.place_at(byref[r], byref[r].anchor, byref[r].rotation_index) for r in order]
    H, S = [0.0], [0.0]
    for m in range(1, len(order) + 1):
        sub = clones[:m]
        H.append(float(model.hpwl(refs={f.ref for f in sub},
                                  poses={f.ref: (f.anchor[0], f.anchor[1], f.rotation_index)
                                        for f in sub})))
        S.append(float(gr.dispersion_of(model, sub)))
    n = len(order)
    best = -math.inf
    for m in range(0, n):
        best = max(best, ((H[n] - H[m]) / H_ref - alpha * (S[n] - S[m]) / D) / (n - m))
    return max(0.0, best)


def test_beta_covers_independent_requirement(cached_boards):
    """**β 的静态守门**：逐板用独立实现重算 `β_crit` ⇒ `β ≥ β_crit`（即 margin ≥ 1 且 β 跟得上）。"""
    rows, worst = [], (0.0, None)
    for board in cached_boards:
        model = mc.load_model(board, allow_build=False)
        order = list(op.plan_order(model))
        if not order:
            continue
        H_ref, D = rw.h_ref_of(model), rw.d_board_of(model)
        crit = _independent_beta_crit(model, order, H_ref, D, rw.ALPHA)
        beta = rw.beta_of(model, order)
        rows.append((board, len(order), crit, beta, beta * len(order)))
        assert beta >= crit - 1e-9, f"{board}: β = {beta:.4f} < 要求 {crit:.4f}"
        if crit > worst[0]:
            worst = (crit, board)

    assert rows, "样本为空（纪律 2）"
    bns = np.array([r[4] for r in rows])
    print(f"\n[β 体检] max β_crit = {worst[0]:.4f}（{worst[1]}）｜β·n：min {bns.min():.3f} "
          f"p50 {np.median(bns):.3f} max {bns.max():.3f}（旧全局口径恒 2.5）")
    assert rw.BETA_MARGIN >= 1.0
    assert bns.max() <= 3.0, f"β·n 最大 {bns.max():.2f} ⇒ 完成项又变回主导，检查 margin/口径"


# ===================== E. σ 两条路径互证 =====================

def test_dispersion_of_equals_delta_along_a_placement(cached_boards):
    board, model = None, None
    for b in cached_boards[:6]:
        m = mc.load_model(b, allow_build=False)
        if len(m.footprints) >= 2:
            board, model = b, m
            break
    if model is None:
        pytest.skip("前 6 块板没有足够多的件")

    order = list(op.plan_order(model))
    byref = {f.ref: f for f in model.footprints}
    comp = byref[order[0]]
    spec = gr.grid_spec(model, m=gr.GRID_M)
    L = gr.legal_masks(model, (), comp, spec)
    cand = np.argwhere(L)
    assert len(cand), f"{board}: 第一件开局无合法格（样本无效）"

    for k, i, j in cand[:5]:
        one = np.zeros_like(L)
        one[k, i, j] = True
        delta = float(gr.dispersion_delta(model, (), comp, spec, one)[k, i, j])
        after = pm.place_at(comp, spec.mm_of(int(i), int(j)), int(k))
        got = gr.dispersion_of(model, (after,)) - gr.dispersion_of(model, ())
        assert got == pytest.approx(delta, abs=1e-9), f"{board}: {k}@({i},{j}) σ 两路不一致"
