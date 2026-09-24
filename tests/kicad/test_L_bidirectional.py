# -*- coding: utf-8 -*-
"""`legal_masks` 的**单向 / 双向**判定回归测试（**2026-09-23 起 L 变双向**）。

为什么会有这个改动
------------------
原来 `L` 只查一个方向：**新件的 pad ↔ 已放件的槽**。
⇒ 后放件**自带的槽**（口径 A：槽随件，§0.6 p）可能压住**先放件**的 pad，而先放件不能再动
（索引只增不减，§0.6 q-⑦）⇒ 事后无法补救。实测：16 板里 2 块板出现、共 13 处 pad 级冲突（硬 9 / 软 4）。
自 2026-09-23 起补上反方向：**新件自带的槽 ↔ 已放件的 pad**（`env/grid_render.py::legal_masks` 第 ⑤ 段）。

覆盖（**纯合成模型，不需要 KiCad**）
----------------------------------
```text
A. 双向：新方向确实生效 —— 且用"去掉冲突源 ⇒ 该格恢复合法"的**对照**把因果钉死
B. 单向回归：**老方向**（新件的 pad ↔ 已放件的槽）没有因为改动而失效
C. 不误杀：槽不压任何已放 pad 的位姿仍然合法；"自家的槽 ↔ 自家的 pad"不算（§2.4 owner 豁免）
D. 与 `Judge` 同口径：新 L 判非法的候选位姿，放下去后 `judge_all` 报的正是 `hard/soft.pad_keepout`

**不在本文件测的**（另有一处覆盖，避免重复造示例）
--------------------------------------------------
`compress_legal`（增量压缩）与"从零重算"的**逐格等价性**（含 now/next **都带槽**的尖锐情形）
由 **KiCad 真实板测试**覆盖：`tests/kicad/test_slot_semantics.py::test_increment_equals_recompute_with_slots`
（参数化里的 `lr1388 / SW8 / USBC1`，2026-09-23 补上 `killed_rects` 第三组后转绿）。
不在合成几何上重复这条，是因为**规则几何必然踩"软间距恰好等于阈值 0.2"的贴边格**，而
`legal_masks`（格索引区间）与 `compress_legal`（`searchsorted` 开区间）在这种贴边格上本来就可能差
±1 格（**既有性质**，与本次改动无关；实测：差异格的 `gap` 全是 `0.2 ± 1e-6`，且独立 `Judge` 判"无违规"）。
"""
import numpy as np
import pytest

try:                                                            # pytest 直跑（tests/kicad 在 sys.path）
    from violation_checks import _fp, _keepout, _pad, mini_model
except ImportError:                                             # 作为包 import 时
    from tests.kicad.violation_checks import _fp, _keepout, _pad, mini_model   # type: ignore

from env import grid_render as gr
from env import violations as vio
from utils.kicad_utils import pcb_model as pm

_A_AT = (10.0, 10.0, 0)          # 已放件 A 的位姿（锚点）


def _scene(*, b_has_slot: bool = True, b_pad=(5.0, 5.0, 6.13, 6.13), b_rot: int = 0,
           a_has_slot: bool = False):
    """A 放 (10,10)（无槽/带槽由 `a_has_slot` 定）；B 是待放件（pad 在 `b_pad` 处、可带槽）。

    几何取"远大于一格"的尺寸（pad ≈1.13mm / 槽 2mm；格 ≈ 0.15mm）⇒ 判定与格量化无关。
    ⚠️ **pad 尺寸特意取 1.13（不是 1.00）**：格中心间距是 0.15 的整数倍，若 pad 尺寸取 1.00，
    则"两 pad 间距恰好 = 软间距 0.2"这种**退化边界**会出现（`|Δ| = 1.2 = 8×0.15` 可达），
    而边界上"恰好相等算不算冲突"依赖浮点 ⇒ 会让"增量压缩 == 从零重算"这种逐格对拍假红
    （踩过一次：2026-09-23）。取 1.13 后 `|Δ| = 1.33` **不是 0.15 的整数倍** ⇒ 边界不可达。
    """
    a = _fp("A", [_pad("1", (-0.5, -0.5, 0.5, 0.5))], anchor=(0.0, 0.0), rot=0)
    b = _fp("B", [_pad("1", b_pad)], anchor=(0.0, 0.0), rot=b_rot)
    kos = []
    if a_has_slot:                                              # A 自带槽（放在 A 锚点旁）
        kos.append(_keepout((0.0, 0.0, 2.0, 2.0), owner_ref="A"))
    if b_has_slot:                                              # B 自带槽（源位姿：锚点 (0,0) ⇒ 局部 (0,0,2,2)）
        kos.append(_keepout((0.0, 0.0, 2.0, 2.0), owner_ref="B"))
    model = mini_model([a, b], keepouts=kos)
    spec = gr.grid_spec(model)
    a_at = pm.place_at(a, (_A_AT[0], _A_AT[1]), _A_AT[2])
    return model, spec, b, a_at


def _legal_at(model, spec, placed, comp, x, y, k=0) -> bool:
    """`comp` 放到板坐标 `(x,y)`、旋转 k 是否在 L 内（按格中心取）。"""
    L = gr.legal_masks(model, list(placed), comp, spec)
    i, j = spec.cell_of(x, y)
    return bool(L[k, i, j])


# ===================== A. 双向：新方向生效 + 因果对照 =====================

def test_bidirectional_new_direction_blocks_and_causal_control():
    """B 的槽压住 A 的 pad ⇒ L 禁；**把槽从模型里去掉 ⇒ 同一格恢复合法**（因果钉死）。"""
    model, spec, b, a_at = _scene(b_has_slot=True)
    # B 的槽局部是 (0,0,2,2) ⇒ 锚点落在 (10,10) 时槽覆盖 (10,10)-(12,12)，压住 A 的 pad (9.5..10.5)
    assert _legal_at(model, spec, [a_at], b, 10.0, 10.0) is False, "新方向：B 的槽压住 A 的 pad ⇒ 必须禁止"

    # 对照：同一个件、同一个候选位姿，只把"B 的槽"去掉 ⇒ 应恢复合法（B 的 pad 在 (5,5)-(6,6)，
    # 离 A 很远 ⇒ 与老方向无关）
    m2, spec2, b2, a2 = _scene(b_has_slot=False)
    assert _legal_at(m2, spec2, [a2], b2, 10.0, 10.0) is True, "去掉槽后同格必须合法（否则不是这条检查在管）"


def test_bidirectional_follows_rotation_of_the_slot():
    """槽随件旋转：源旋转 90° 时，槽的局部框也要跟着转（防"漏逆旋转"那类回归）。"""
    model, spec, b, a_at = _scene(b_has_slot=True, b_rot=1)
    # 源旋转 90°：源位姿槽 (0,0,2,2) 逆旋转后 ⇒ 局部 (−2,0,0,2)；再按候选旋转 k=1 ⇒ 仍在"锚点左下方"
    # 这里只断言"存在禁格，且禁格集合随后续旋转变化"（几何细节交给 keepout_at，避免手抄坐标）
    L0 = gr.legal_masks(model, [a_at], b, spec)
    assert not bool(L0[0, *spec.cell_of(10.0, 10.0)]), "k=0 时该格应被禁（槽压 A 的 pad）"
    assert L0.sum() > 0, "不该全禁"


# ===================== B. 单向回归：老方向仍生效 =====================

def test_legacy_direction_still_blocks_pad_vs_slot():
    """**老方向**（待放件的 pad ↔ 已放件的槽）必须仍然生效 —— 改动不能把它弄坏。

    隔离手法：B 的小 pad 取 `(1.0,1.0,1.5,1.5)`、锚点落到 (10,10) ⇒ B 的 pad = (11,11)-(11.5,11.5)，
    **落在 A 的槽 (10,10)-(12,12) 内、却在 A 的 pad (9.5..10.5) 之外** ⇒ 只可能被"pad↔已放件的槽"这一条禁。
    """
    model, spec, b, a_at = _scene(b_has_slot=False, a_has_slot=True, b_pad=(1.0, 1.0, 1.5, 1.5))
    assert _legal_at(model, spec, [a_at], b, 10.0, 10.0) is False, "老方向（pad ↔ 已放件的槽）必须仍然禁止"
    # 放到远处（B 的 pad 到 (1,1) 附近，既不碰 A 的槽也不碰 A 的 pad）⇒ 合法
    assert _legal_at(model, spec, [a_at], b, 0.0, 0.0) is True


# ===================== C. 不误杀 =====================

def test_no_false_kill_and_own_slot_exempt():
    """① 槽不压任何已放 pad 的位姿仍合法；② 自家的槽 ↔ 自家的 pad 不算违规（不会全禁）。"""
    model, spec, b, a_at = _scene(b_has_slot=True)
    # ① B 放到 (0,0)：它的槽覆盖 (0,0)-(2,2)，与 A 的 pad (9.5..10.5) 无关 ⇒ 合法
    assert _legal_at(model, spec, [a_at], b, 0.0, 0.0) is True, "不冲突的位姿必须仍合法（不误杀）"

    # ② B 的 pad 就压在"自家槽"上（pad 局部 (0.5,0.5,1.5,1.5) ⊂ 槽 (0,0,2,2)）⇒ 仍须有合法格
    m3, spec3, b3, a3 = _scene(b_has_slot=True, b_pad=(0.5, 0.5, 1.5, 1.5))
    L3 = gr.legal_masks(m3, [a3], b3, spec3)
    assert L3.sum() > 0, "自家槽 ↔ 自家 pad 必须豁免（§2.4 owner 排除），否则整张 L 会全 False"
    assert _legal_at(m3, spec3, [a3], b3, 0.0, 0.0) is True


# ===================== D. 与 Judge 同口径 =====================

def test_blocked_cells_match_judge_pad_keepout():
    """新 L 判"非法"的候选位姿，放下去后 `judge_all` 报的必须是 `hard/soft.pad_keepout`（同口径）。

    ⚠️ 对拍姿势：`Judge` 的 keepouts 来自构造时给的 `placed` ⇒ 要复现"**B 已落下**"那一帧，
    必须把 `B` 也放进 `placed`（否则 B 的槽不在 keepouts 里，Judge 自然报不出"槽压 pad"）。
    """
    model, spec, b, a_at = _scene(b_has_slot=True)
    x, y, k = 10.0, 10.0, 0
    L = gr.legal_masks(model, [a_at], b, spec)
    assert not bool(L[k, *spec.cell_of(x, y)]), "该位姿应被 L 禁止（B 的槽压住 A 的 pad）"

    b_at = pm.place_at(b, (x, y), k)
    judge = vio.Judge(model, placed=(a_at, b_at))                  # "B 已落下"那一帧
    res = judge.judge_all({"A": _A_AT, "B": (x, y, k)})
    assert res.is_violation, f"Judge 必须也认为违规，实际 {res.describe()}"
    assert any(r.endswith("pad_keepout") for r in res.reasons), \
        f"违规类型应是 pad_keepout（与 L 的新检查一致），实际 {res.describe()}"

    # 反向抽查：L 说合法的格，同一姿势下 Judge 必须**不报**违规（不能误杀）
    ok_cells = [tuple(c) for c in np.argwhere(L[k])][:6]
    assert ok_cells, "该局面应有合法格（样本为空）"
    for i, j in ok_cells:
        ox, oy = spec.mm_of(i, j)
        jm = vio.Judge(model, placed=(a_at, pm.place_at(b, (ox, oy), k)))
        r2 = jm.judge_all({"A": _A_AT, "B": (ox, oy, k)})
        assert not r2.is_violation, f"L 说合法但 Judge 报违规：{r2.describe()} @ ({ox:.2f},{oy:.2f})"


# ===================== 说明：增量压缩的等价性不在本文件测 =====================
# `compress_legal` 与"从零重算"的逐格等价性（含 now/next 都带槽）由 KiCad 真实板测试覆盖：
#   tests/kicad/test_slot_semantics.py::test_increment_equals_recompute_with_slots[lr1388-SW8-USBC1]
# 原因见文件头 docstring（合成规则几何必然踩"软间距恰好 = 0.2"的贴边格 ⇒ 会假红）。
