# -*- coding: utf-8 -*-
"""`BoardState` + `placed` 契约 + 导入期自检的回归测试（台账 §0.6 q；依据 probe171）。

覆盖：
  A. `audit_placed` / `require_clean`：抓到四类问题（自含 / 漏件 / 外来 / 重复）+ 计数 + "只告警一次"
  B. `BoardState.placed` 是**新鲜克隆**（几何真的跟着走，不是源位姿快照）——**平移 Δ 的独立校验**
  C. 状态给出的帧 == 手工构造的帧（`placed` / `keepouts` / `others` 三形状同源）；
     并证明"传源位姿"会静默改变 L（这就是 §0.6 q-③ 的代价）
  D. `index` 只增不减、无"移动已放件"入口、`poses` 只读、旧状态不被修改（§0.6 q-⑦ 的单调性前提）
  E. `validate_model`：全数据集 0 项；人为破坏（缺 `local_aabb` / 缺 hull / ref 重复）必须被抓到
  F. `build(validate=True)` 接线：注入假问题 ⇒ 抛 `ModelValidationError`；`validate=False` 不抛
"""
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest

from env import board_state as bs
from env import grid_render as gr
from utils.kicad_utils import pcb_model as pm

pytestmark = pytest.mark.require_kicad


def _pick(boards, sub):
    """按板目录名（唯一子串）取 `(path, board, model)`。"""
    hits = [(p, b, m) for p, b, m, _r in boards if sub in os.path.basename(os.path.dirname(p))]
    assert len(hits) == 1, f"板名 {sub!r} 匹配到 {len(hits)} 块（须唯一）"
    return hits[0]


def _legal_far_cell(model, comp, spec, placed=()):
    """在空局面下取一个"离源位姿最远"的合法格 `(k, i, j)`（旋转固定 0，便于用纯平移校验几何）。"""
    L = gr.legal_masks(model, list(placed), comp, spec)
    cells = np.argwhere(L[0])
    assert len(cells), "空局面下居然没有合法格（样本为空，纪律 2）"
    ax, ay = comp.anchor
    cx = spec.x0 + (cells[:, 0] + 0.5) * spec.cell_x
    cy = spec.y0 + (cells[:, 1] + 0.5) * spec.cell_y
    d = (cx - ax) ** 2 + (cy - ay) ** 2
    t = int(np.argmax(d))
    return 0, int(cells[t, 0]), int(cells[t, 1])


# ---------------- A. 契约体检 ----------------

def test_audit_detects_four_kinds_and_reports(boards_with_models):
    """四类问题（自含 / 漏件 / 外来 / 重复）逐个可判；`strict=True` 抛错并点名 ref。"""
    _path, _board, model = _pick(boards_with_models, "bliysfyt")
    byref = {f.ref: f for f in model.footprints}
    st0 = bs.BoardState.start(model, tuple(bs.plan_order(model)))
    spec = gr.grid_spec(model, m=gr.GRID_M)
    k, i, j = _legal_far_cell(model, st0.comp, spec)
    st = st0.place(k, i, j)
    first = st.placed_refs[0]

    clean = bs.audit_placed(model, st.placed, st.comp, st.order)
    assert clean.clean and clean.n_placed == 1, clean.describe()

    self_inc = bs.audit_placed(model, list(st.placed) + [st.comp], st.comp, st.order)
    assert self_inc.self_included and not self_inc.clean

    missing = bs.audit_placed(model, [], st.comp, st.order)
    assert missing.missing == (first,) and not missing.clean

    foreign = replace(byref[st.comp.ref], ref="__zzz_foreign__")
    extra = bs.audit_placed(model, list(st.placed) + [foreign], st.comp, st.order)
    assert extra.extra == ("__zzz_foreign__",) and not extra.clean

    dup = bs.audit_placed(model, list(st.placed) * 2, st.comp, st.order)
    assert dup.duplicated == (first,) and not dup.clean

    with pytest.raises(bs.PlacedContractError) as ei:
        bs.require_clean(model, list(st.placed) + [st.comp], st.comp, st.order, strict=True,
                         where="A")
    assert st.comp.ref in str(ei.value) and "静默" in str(ei.value)

    bs.reset_audit_stats()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        bs.require_clean(model, [], st.comp, st.order, strict=False)       # 第一次：告警
        bs.require_clean(model, [], st.comp, st.order, strict=False)       # 第二次：不再告警
    assert bs.audit_failures() == 2, "计数必须累加（供训练日志监控）"
    assert len(rec) == 1, f"同类问题应只提示一次（实际 {len(rec)}）"
    assert "PCB_RL_AUDIT=1" in str(rec[0].message)


# ---------------- B. 派生的是"新鲜克隆" ----------------

def test_state_derives_fresh_clones(boards_with_models):
    """`place()` 后 `placed` 的几何必须**真的跟着走**（用纯平移量独立校验，不靠 `rotate_aabb` 自证）。"""
    _path, _board, model = _pick(boards_with_models, "bliysfyt")
    src = {f.ref: f for f in model.footprints}
    st0 = bs.BoardState.start(model)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    ref0 = st0.order[0]
    k, i, j = _legal_far_cell(model, st0.comp, spec)
    x, y = spec.mm_of(i, j)
    assert k == 0

    st = st0.place(k, i, j)
    assert st.index == 1 and st.placed_refs == (ref0,)
    assert st.progress == pytest.approx(1.0 / st.n_total)
    assert st.audit().clean, st.audit().describe()
    assert st.poses[ref0] == (x, y, 0)

    clone, s = st.placed[0], src[ref0]
    assert clone is not s, "派生必须是克隆，不能把模型对象直接塞进来"
    # ① 锚点位移精确等于格中心 − 源锚点（纯平移，独立于旋转公式）
    assert abs(clone.anchor[0] - x) < 1e-12 and abs(clone.anchor[1] - y) < 1e-12
    # ② 每个 pad 的中心位移 = 同一个平移向量（k=0 ⇒ 只平移）
    d = (x - s.anchor[0], y - s.anchor[1])
    dev = max(max(abs((c2.center[t] - p.center[t]) - d[t]) for t in (0, 1))
              for p, c2 in zip(s.pads, clone.pads))
    assert dev < 1e-9, f"pad 没跟着走（最大偏差 {dev:.3e} mm）"
    # ③ 与源位姿几何确实不同（否则本用例失去判别力）
    assert any(c2.aabb != p.aabb for p, c2 in zip(s.pads, clone.pads)), \
        "挪到最远的合法格后几何竟与源位姿相同 ⇒ 样本失去判别力"
    assert st.placed is st.placed, "派生量应带缓存（同帧不重复算）"


# ---------------- C. 帧的一致性与"源位姿"的代价 ----------------

def test_state_frame_matches_manual_and_exposes_staleness(boards_with_models):
    """状态的 `placed`/`keepouts`/`others` 与手工构造同源；传源位姿会静默改变 L。"""
    _path, _board, model = _pick(boards_with_models, "lr1388")
    byref = {f.ref: f for f in model.footprints}
    st0 = bs.BoardState.start(model)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    ref0 = st0.order[0]
    k, i, j = _legal_far_cell(model, st0.comp, spec)
    x, y = spec.mm_of(i, j)
    st = st0.place(k, i, j)
    comp = st.comp

    # ① 手工用 `place_at` 直接构造（**独立路径**，不经 `derive_placed`）
    manual = (pm.place_at(byref[ref0], (x, y), k),)
    L_state = gr.legal_masks(model, st.placed, comp, spec, keepouts=st.keepouts)
    L_manual = gr.legal_masks(model, manual, comp, spec,
                              keepouts=pm.state_keepouts(model, manual))
    assert np.array_equal(L_state, L_manual), "状态派生出的帧与手工构造不一致"
    assert st.keepouts == pm.state_keepouts(model, st.placed)
    assert st.others == {ref0: (x, y, k)}
    assert st.others == bs.others_of(st.placed)

    # ② 反面：传**源位姿**（陈旧对象）⇒ L 静默变化（这就是"忘记用 place_at 克隆"的代价）
    L_stale = gr.legal_masks(model, (byref[ref0],), comp, spec)
    diff = int((L_state ^ L_stale).sum())
    print(f"\n[lr1388] 状态帧 L={int(L_state.sum())}｜源位姿帧 L={int(L_stale.sum())}"
          f"｜差异 {diff} 格")
    assert diff > 0, "该局面下'源位姿 vs 位姿化'居然无差别 ⇒ 用例失去判别力"
    assert int((L_state & ~L_stale).sum()) > 0, "应存在'正确帧合法、陈旧帧非法'的格（丢动作）"


# ---------------- D. 只进不退 + 无"移动已放件"入口 ----------------

def test_index_only_moves_forward(boards_with_models):
    """索引只增、`poses` 只读、旧状态不被修改；放完即 `done` 且再放报错。"""
    _path, _board, model = _pick(boards_with_models, "jackt")
    order = tuple(bs.plan_order(model))
    assert len(order) >= 3, f"该板只有 {len(order)} 件，样本不足（纪律 2）"
    st0 = bs.BoardState.start(model, order[:3])          # 只走 3 件，省时间
    spec = gr.grid_spec(model, m=gr.GRID_M)

    assert st0.index == 0 and st0.placed == () and st0.progress == 0.0
    assert not st0.done and st0.next_comp is not None
    assert not hasattr(st0, "move") and not hasattr(st0, "unplace"), \
        "不允许存在'移动已放件'的写入口（否则 lookahead 的单调性前提失效）"

    st = st0
    for step in range(3):
        k, i, j = _legal_far_cell(model, st.comp, spec, placed=st.placed)
        prev = st
        st = st.place(k, i, j)
        assert prev.index == step and len(prev.placed) == step, "旧状态被就地修改了"
        assert st.index == step + 1 and st.check().clean, st.audit().describe()

    assert st.done and st.progress == 1.0 and st.next_comp is None
    with pytest.raises(IndexError):
        st.place(0, 0, 0)
    with pytest.raises(TypeError):
        st.poses[order[0]] = (0.0, 0.0, 0)               # 只读视图
    with pytest.raises(ValueError):
        bs.BoardState(model=model, order=order, index=2, _poses={order[0]: (0.0, 0.0, 0)})


# ---------------- E. 导入期自检 ----------------

def test_validate_model_clean_on_all_boards_and_catches_breakage(boards_with_models):
    """全数据集 0 项；人为破坏（缺 `local_aabb` / 缺 hull / ref 重复）必须被抓到。"""
    problems = []
    for path, _board, model, _r in boards_with_models:
        for s in pm.validate_model(model):
            problems.append((os.path.basename(os.path.dirname(path)), s))
    assert not problems, f"真实数据出现导入期问题（{len(problems)} 项）：{problems[:5]}"

    _path, _board, model = _pick(boards_with_models, "bliysfyt")
    f = model.footprints[0]
    pad = f.pads[0]

    broke_pad = replace(f, pads=(replace(pad, local_aabb=()),) + f.pads[1:])
    probs = pm.validate_model(replace(model, footprints=(broke_pad,) + model.footprints[1:]))
    assert any(f"{f.ref}/{pad.number}" in s and "local_aabb" in s for s in probs), probs

    broke_hull = replace(f, local_hull_aabb=())
    probs = pm.validate_model(replace(model, footprints=(broke_hull,) + model.footprints[1:]))
    assert any("local_hull_aabb" in s for s in probs), probs

    dup = replace(model, footprints=(f, f) + model.footprints[1:])
    probs = pm.validate_model(dup)
    assert any("ref 重复" in s for s in probs), probs


# ---------------- F. 接线 ----------------

def test_build_wires_validation(boards_with_models, monkeypatch):
    """`build(validate=True)` 遇到问题必须抛 `ModelValidationError`；`validate=False` 放行。"""
    path, board, _model, _r = boards_with_models[0]
    monkeypatch.setattr(pm, "validate_model", lambda m: ["fake problem"])
    pm.PcbModelBuilder().build(board, path, validate=False)            # 不抛
    with pytest.raises(pm.ModelValidationError) as ei:
        pm.PcbModelBuilder().build(board, path, validate=True)
    assert "fake problem" in str(ei.value) and "导入期自检失败" in str(ei.value)
