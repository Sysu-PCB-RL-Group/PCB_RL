# -*- coding: utf-8 -*-
"""放置完成度（台账 §0.6 m；Owner 2026-09-22 定稿）的回归测试。

覆盖：
  A. 三个量的端点与单调性（空局面 = 0；放完 = 1；逐步非减）
  B. 主指标 == `BoardState.progress`；`as_info()` 键值自洽（主键 = `completion`）
  C. 件面积的**单点定义**与位姿无关（`local_hull_area`）；并证明"用当前位姿 AABB"会变
     ⇒ 选 θ=0 是有判别力的设计，不是空转
  D. ③ 的机械正确性：非负、端点 0/1、前缀和 == 手算、与 `hpwl` 独立重算一致；零贡献件不改变 ③
  E. 边界：`n = 0`（分母 0 ⇒ 1.0）、`with_contrib=False`（③ = `None`）、`index` 越界抛错、
     表与状态的顺序不一致抛错
  F. 全 20 板：表长 == `plan_order` 长度、`at(0)/at(n)` 端点、面积总和自洽；且 ② 与 ① 确有差异
"""
import os

import numpy as np
import pytest

from env import board_state as bs
from env import completion as cp
from env import grid_render as gr
from env import order_plan as op
from utils.kicad_utils import pcb_model as pm

pytestmark = pytest.mark.require_kicad


def _pick(boards, sub):
    """按板目录名（唯一子串）取 `(path, board, model)`。"""
    hits = [(p, b, m) for p, b, m, _r in boards if sub in os.path.basename(os.path.dirname(p))]
    assert len(hits) == 1, f"板名 {sub!r} 匹配到 {len(hits)} 块（须唯一）"
    return hits[0]


def _first_legal(model, placed, comp, spec, k=0):
    """空/当前局面下取第一个合法格 `(k, i, j)`（确定、便宜）。"""
    L = gr.legal_masks(model, list(placed), comp, spec)
    cells = np.argwhere(L[k])
    assert len(cells), "没有合法格（样本为空 —— 纪律 2）"
    return k, int(cells[0, 0]), int(cells[0, 1])


# ---------------- A. 端点与单调性 ----------------

def test_endpoints_and_monotonic(boards_with_models):
    """空局面 ⇒ 三量全 0；放完 ⇒ 三量全 1；沿前缀逐步非减。"""
    _p, _b, model = _pick(boards_with_models, "bliysfyt")
    t = cp.CompletionTable.build(model)                       # with_contrib=True
    r0, rn = t.at(0), t.at(t.n)

    assert r0.count_ratio == 0.0 and r0.area_ratio == 0.0 and r0.contrib_ratio == 0.0
    assert not r0.done
    assert rn.count_ratio == 1.0 and rn.area_ratio == 1.0 and rn.contrib_ratio == 1.0
    assert rn.done and rn.primary == 1.0

    seq = [t.at(k) for k in range(t.n + 1)]
    for k in range(t.n):
        assert seq[k].area_ratio <= seq[k + 1].area_ratio + 1e-12
        assert seq[k].contrib_ratio <= seq[k + 1].contrib_ratio + 1e-12
        assert seq[k].index == k


# ---------------- B. 与 BoardState / info 的一致性 ----------------

def test_primary_matches_board_state_and_info_keys(boards_with_models):
    """主指标 == `BoardState.progress`（连放 3 步逐个对拍）；`as_info()` 键值自洽。"""
    _p, _b, model = _pick(boards_with_models, "bliysfyt")
    t = cp.CompletionTable.build(model, with_contrib=False)
    st = bs.BoardState.start(model, t.refs)
    spec = gr.grid_spec(model, m=gr.GRID_M)

    assert t.of(st).primary == 0.0 == st.progress
    for _ in range(3):
        k, i, j = _first_legal(model, st.placed, st.comp, spec)
        st = st.place(k, i, j)
        rep = t.of(st)
        assert rep.primary == pytest.approx(st.progress)
        assert rep.index == st.index and rep.n == st.n_total

    info = t.of(st).as_info()
    for key in ("completion", "completion_count", "completion_area", "completion_contrib",
                "n_placed", "n_total", "area_placed_mm2", "area_total_mm2",
                "contrib_placed_mm", "contrib_total_mm"):
        assert key in info, f"info 缺键 {key}"
    assert info["completion"] == t.of(st).primary
    assert info["completion_count"] == info["completion"]
    assert info["n_placed"] == st.index and info["n_total"] == st.n_total
    assert info["completion_contrib"] is None           # with_contrib=False

    # 前缀语义：已完成的那几件 == order[:index]
    assert tuple(st.placed_refs) == t.refs[: st.index]


# ---------------- C. 面积口径：位姿无关 + 4 旋转下恒等 ----------------

def test_piece_area_is_pose_invariant(boards_with_models):
    """② 的定标依据：`local_hull_area`（θ=0 固有尺寸）**与位姿无关**。

    同时钉住一条**实测事实**：4 个旋转只有 90° 倍数 ⇒ hull AABB 只是"宽高互换"
    ⇒ **AABB 面积在 4 个旋转下恒等**（所以"用 θ=0 局部面积"与"用当前位姿 AABB 面积"
    数值相同；选 θ=0 是为了**单点定义**、与 `order_plan` 的面积权重同一把尺，
    不是因为它更"抗抖动"）。样本里至少有一件是各向异性的（否则"宽高互换"无判别力）。
    """
    _p, _b, model = _pick(boards_with_models, "verysunshine")   # ⚠️ 别用 "li-chuang"（4 块同前缀）
    anisotropic = 0
    for f in model.footprints:
        a_loc = f.local_hull_area
        seen = set()
        for k in (0, 1, 2, 3):
            g = pm.place_at(f, f.anchor, k)
            assert g.local_hull_area == pytest.approx(a_loc), \
                f"{f.ref}: local_hull_area 随位姿变了（面积口径被破坏）"
            a = g.hull_aabb
            seen.add(round((a[2] - a[0]) * (a[3] - a[1]), 6))
        assert len(seen) == 1, f"{f.ref}: 4 个旋转下 AABB 面积不一致：{seen}"
        assert abs(next(iter(seen)) - a_loc) < 1e-6
        x0, y0, x1, y1 = f.local_hull_aabb
        if abs((x1 - x0) - (y1 - y0)) > 1e-6:
            anisotropic += 1
    assert anisotropic > 0, "全部件的 hull 都是正方形？（宽高互换无判别力）"


# ---------------- D. ③ 的机械正确性 ----------------

def test_contrib_mechanics_against_hpwl(boards_with_models):
    """③ 的原料 `c_t` 与 `hpwl` 独立重算一致、非负；前缀和 == 手算；零贡献件不改变 ③。"""
    _p, _b, model = _pick(boards_with_models, "verysunshine")
    refs = tuple(op.plan_order(model))
    all_refs = set(refs)
    full = float(model.hpwl(refs=all_refs))

    c = cp.marginal_contributions(model, refs)
    assert set(c) == all_refs
    for r in refs:
        direct = full - float(model.hpwl(refs=all_refs - {r}))
        assert c[r] == pytest.approx(direct, abs=1e-9)       # 定义自洽（独立重算）
        assert c[r] >= -1e-9, f"{r}: c_t 出现负值（应恒 ≥ 0）"

    t = cp.CompletionTable.build(model, refs, with_contrib=True)
    acc = 0.0
    for k, r in enumerate(refs, start=1):
        acc += c[r]
        assert t.at(k).contrib_placed == pytest.approx(acc)
    assert t.at(t.n).contrib_total == pytest.approx(sum(c.values()))
    assert t.at(0).contrib_ratio == 0.0 and t.at(t.n).contrib_ratio == 1.0

    # 零贡献件（数据集中真实存在）插入前缀不改变分子
    zeros = [r for r in refs if c[r] == 0.0]
    if zeros:
        z = zeros[0]
        k = refs.index(z)
        assert t.at(k + 1).contrib_placed == pytest.approx(t.at(k).contrib_placed), \
            f"零贡献件 {z} 改变了完成度分子（应不变）"


def test_contrib_ratio_differs_from_count_ratio(boards_with_models):
    """贡献加权与件数占比**不是同一个数**（否则"记三个量"无意义）。"""
    _p, _b, model = _pick(boards_with_models, "verysunshine")
    t = cp.CompletionTable.build(model, with_contrib=True)
    diffs = [abs(t.at(k).contrib_ratio - t.at(k).count_ratio) for k in range(t.n + 1)]
    assert max(diffs) > 0.05, f"贡献占比与件数占比几乎相同（max Δ = {max(diffs):.4f}）"


# ---------------- E. 边界 ----------------

def test_zero_n_table():
    """`n = 0` ⇒ 比值 1.0（没有可放的东西 ⇒ 视为已完成）；越界抛错。"""
    t = cp.CompletionTable(refs=(), area_prefix=(0.0,))
    r = t.at(0)
    assert r.primary == 1.0 and r.area_ratio == 1.0 and r.contrib_ratio is None and r.done
    for bad in (-1, 1):
        with pytest.raises(ValueError):
            t.at(bad)


def test_with_contrib_false_and_order_mismatch(boards_with_models):
    """③ 可整块关闭；表与状态顺序不一致必须抛错（防"表建错板"的静默错）。"""
    _p, _b, model = _pick(boards_with_models, "bliysfyt")
    t = cp.CompletionTable.build(model, with_contrib=False)
    assert t.contrib_prefix is None
    assert t.at(0).contrib_ratio is None and t.at(t.n).contrib_ratio is None

    st = bs.BoardState.start(model, tuple(reversed(t.refs)))         # 顺序不同的状态
    with pytest.raises(ValueError):
        t.of(st)


# ---------------- F. 全 20 板 ----------------

def test_all_boards_endpoints_and_area_totals(boards_with_models):
    """全 20 板：表长 == `plan_order` 长度、端点 0/1、面积总和 == `piece_areas` 之和。"""
    n_boards = 0
    for _p, _b, model, _r in boards_with_models:
        order = tuple(op.plan_order(model))
        t = cp.CompletionTable.build(model, order, with_contrib=False)
        assert t.n == len(order) and t.n > 0
        assert t.at(0).primary == 0.0 and t.at(t.n).primary == 1.0
        areas = cp.piece_areas(model)
        assert set(areas) == set(order)                    # order 覆盖全部 footprint
        assert t.at(t.n).area_total == pytest.approx(sum(areas.values()))
        assert all(v >= 0.0 for v in areas.values())
        n_boards += 1
    assert n_boards == len(boards_with_models) and n_boards > 1
