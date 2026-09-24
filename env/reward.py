# -*- coding: utf-8 -*-
"""reward 的口径与常数（台账 §7.3 定稿）—— env 组装时的**唯一入口**。

```text
r_t = (−ΔHPWL_t)/H_ref + α·(Δσ_t)/D + β            γ = 1
β   = BETA_MARGIN · β_crit(该板)                   ← ✅ 2026-09-22（Owner："β 只能反映 HPWL 早停的惩罚"）

β_crit(该板) = max over m = 0…n−1 [ (H_n − H_m)/H_ref − α·(σ_n − σ_m)/D ] / (n − m)
   · H_m / σ_m = 把【前 m 件（固定顺序）】放在各自【源/专家位姿】时的 HPWL / 跨 net 离散度
   · 判据来源：**任意能停的前缀**（`L = ∅`）都必须劣于"把剩下的放完" ⇒ β 的下界就是这个 max
   · 实现：**随板现算**（`beta_crit_of`，~30 ms/板，纯 numpy，**不需要 KiCad、不需要离线表**）⇒ 新板零配置
   · H_ref = 专家完整布局 HPWL（即 HPWL 项的分母）；D = 板框对角线（σ 项的分母）
```

为什么是"现算的 β_crit × 小余量"，而不是一个拍出来的大常数
------------------------------------------------------
1. **β 的语义就是"HPWL 早停惩罚"** ⇒ 它的大小应由**该板**"继续放完要付的代价"决定，
   而不是一个全局经验式（`k/n` 只是这条曲线的**粗略拟合**：`c = n·β_crit` 实测 0.000 ~ 1.714，p50 0.80）。
2. **β 不能更小**（这是硬下界）：若 `β < β_crit`，会出现"把尾部堵死 ⇒ 提前终止 ⇒ 已放件少 ⇒ HPWL 天然小 ⇒
   回报反而更高"的反目标行为。实测 `max β_crit = 0.1256`（`jackt`，n=10）。
3. **β 也不必更大**：`β·n`（完成项整局分量）实测从"全局 `k/n` 口径"的 **2.5** 降到 **0.000 ~ 1.51（p50 0.80）**
   ⇒ 三项（HPWL 项 / σ 项 / 完成项）在整局上从 `1 : 0.97 : 2.5` 变成 **`1 : 0.97 : 0.80`**
   ⇒ **完成项不再"占大头"**，与另两项对等（这正是 Owner 要的）。
4. **`BETA_MARGIN` 只留"口径近似"这一层余量**（专家曲线给的是**下界**：专家布局最省 ⇒ 继续放最便宜）：
   `"早训期策略更差"那一段由【训练期死锁率/完成度闭环校准】兜`（β 是唯一可在线调的旋钮），
   不再靠把静态 β 顶大。

与"β 与动作无关"的关系（常被误读）
--------------------------------
固定顺序 + 强制放件 ⇒ **件数 = 步序号的确定函数** ⇒ 同一步内所有动作的 β **逐位相同** ⇒
**β 不参与任何一步的动作排序**（`argmax_a Q` 不受影响，实测即使 β=10 也一样，probe213）。
它只决定两件事：① "放完 vs 停在 m"的取舍（这就是它的全部使命）② value 的尺度/形状。
⚠️ 但β 在"迷茫步"（跨动作差异为 0 的步，实测占 33%~50%）上**就是唯一的非零信号**（probe198/213）。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from env import grid_render as gr
from env.order_plan import plan_order
from utils.kicad_utils import pcb_model as pm

__all__ = [
    "ALPHA",
    "BETA_MARGIN",
    "GAMMA",
    "beta_crit_from_curves",
    "expert_curves",
    "beta_crit_of",
    "beta_of",
    "h_ref_of",
    "d_board_of",
    "step_reward",
    "clear_beta_cache",
]

ALPHA = 5.6          # σ 项系数（probe202："每步判别力 7:3"，与 G 的 0.7:0.3 同义）
BETA_MARGIN = 1.2    # β = BETA_MARGIN × β_crit(专家曲线)；余量只为覆盖"口径近似"（早训期由闭环安全阀兜）
GAMMA = 1.0          # episode 有限且会终止；γ<1 会让"为未来预留空间"的延迟代价被二次打折


# ===================== β 的临界值（判据本体） =====================

def beta_crit_from_curves(H: Sequence[float], S: Sequence[float], H_ref: float, D: float,
                          alpha: float = ALPHA, m_min: int = 0) -> Tuple[float, Optional[int]]:
    """**纯函数**：`(β_crit ≥ 0, m*)`，`β_crit = max over m ∈ [m_min, n) [ ... ] / (n − m)`。

    `raw_max < 0` ⇒ 该板对 β 无要求（σ 收益已超过继续下去的 HPWL 代价）⇒ 返 `(0.0, None)`。
    """
    n = len(H) - 1
    raw, arg = -math.inf, None
    for m in range(int(m_min), n):
        term = (H[n] - H[m]) / H_ref
        if alpha:
            term -= alpha * (S[n] - S[m]) / D
        v = term / (n - m)
        if v > raw:
            raw, arg = v, m
    if raw == -math.inf:                    # 空板（n ≤ 0）
        return 0.0, None
    return max(0.0, raw), (arg if raw > 0 else None)


def expert_curves(model: pm.PcbModel, order=None) -> Tuple[list, list]:
    """专家前缀曲线 `(H[0..n], S[0..n])`：前 m 件放各自**源位姿**时的 HPWL / σ（mm）。"""
    seq = tuple(order) if order is not None else tuple(plan_order(model))
    byref = {f.ref: f for f in model.footprints}
    clones = [pm.place_at(byref[r], byref[r].anchor, byref[r].rotation_index) for r in seq]
    H, S = [0.0], [0.0]
    for m in range(1, len(seq) + 1):
        sub = clones[:m]
        H.append(float(model.hpwl(refs={f.ref for f in sub},
                                  poses={f.ref: (f.anchor[0], f.anchor[1], f.rotation_index)
                                        for f in sub})))
        S.append(float(gr.dispersion_of(model, sub)))
    return H, S


def beta_crit_of(model: pm.PcbModel, order=None,
                 alpha: float = ALPHA) -> Tuple[float, Optional[int]]:
    """该板的 `β_crit`（**随板现算**，~30 ms；判据 = 上述 max）。"""
    seq = tuple(order) if order is not None else tuple(plan_order(model))
    if not seq:
        return 0.0, None
    H_ref = h_ref_of(model)
    if not (H_ref > 0.0):
        return 0.0, None
    H, S = expert_curves(model, seq)
    return beta_crit_from_curves(H, S, H_ref, d_board_of(model), alpha)


# ===================== β（每板一个常数） =====================

_BETA_CACHE: Dict[tuple, float] = {}


def beta_of(model: pm.PcbModel, order=None, *, margin: float = BETA_MARGIN,
            use_cache: bool = True) -> float:
    """`β = margin · β_crit(该板)`（**每 episode 恒定**；与状态/动作无关 ⇒ 不改动作排序）。

    成本 ~30 ms/板 ⇒ 内部按 `(model.path, n, margin, alpha)` 记忆化（= 台账 §5.5.1 的"A 层常驻"），
    env 在 `reset` 时取一次即可（每步调用也不会重复计算）。
    """
    seq = tuple(order) if order is not None else tuple(plan_order(model))
    key = (str(getattr(model, "path", "")), len(seq), float(margin), float(ALPHA))
    if use_cache and key in _BETA_CACHE:
        return _BETA_CACHE[key]
    crit, _m = beta_crit_of(model, seq)
    val = float(margin) * float(crit)
    if use_cache:
        _BETA_CACHE[key] = val
    return val


def clear_beta_cache() -> None:
    """清缓存（换了缓存版本 / 改了几何口径后的自检用）。"""
    _BETA_CACHE.clear()


# ===================== 归一化的两把尺 =====================

def h_ref_of(model: pm.PcbModel) -> float:
    """`H_ref` = 该板**专家（源位姿）完整布局**的 HPWL（mm）—— reward 里 HPWL 项的分母。

    ⚠️ 口径 = `PcbModel.hpwl()` 本身（排除电源网、连接点 = pad 中心、不分层；§7.1）。
    训练用本函数（真值）；**推理陌生板**的兜底 `U = N_net2·D` 见 §7.1.1（尚未实现）。
    """
    return float(model.hpwl())


def d_board_of(model: pm.PcbModel) -> float:
    """`D` = 板框外接矩形对角线（mm）—— σ 项的分母。"""
    ob = model.board_outline.bbox
    return math.hypot(ob[2] - ob[0], ob[3] - ob[1])


def step_reward(*, d_hpwl: float, d_sigma: float, h_ref: float, d_board: float,
                beta: float) -> float:
    """单步 reward（§7.3 定稿）：`(−ΔHPWL)/H_ref + α·(Δσ)/D + β`。

    `d_hpwl = HPWL_after − HPWL_before`（mm，可负）；`d_sigma = σ_after − σ_before`（mm）。
    前两项是**势差** ⇒ 一局合计 = `[−HPWL_final + α·σ_final]/H_ref`（保总量、不可 game）；
    `β` 是**故意**破坏守恒的进度项（防"少放件天然占便宜"）。
    """
    return (-d_hpwl) / h_ref + ALPHA * d_sigma / d_board + beta
