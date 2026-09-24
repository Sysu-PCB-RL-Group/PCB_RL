# -*- coding: utf-8 -*-
"""放置完成度（台账 §0.6 m，**Owner 2026-09-22 定稿**）：三个原始量 + 一个可直接进 `info` 的报告。

口径（定稿）
------------

| # | 量 | 定义 | 角色 |
|---|---|---|---|
| ① | **件数占比** | `index / n` | **主指标**（日志 / 测评 / `best_model` 选择都用它） |
| ② | **面积占比** | `Σ_{已放} A_t / Σ_{全部} A_t`，`A_t = Footprint.local_hull_area`（θ=0） | 诊断 |
| ③ | **贡献占比** | `Σ_{已放} c_t / Σ_{全部} c_t`，`c_t` = 该件对**专家布局** HPWL 的边际贡献 | 诊断（**唯一引用专家布局的量**） |

三条依据
--------

1. **为什么主指标用 ①**：`c_t` 极度偏斜（`c/D` 中位 **0.026** / max **20.12**，probe131）⇒
   贡献加权会被少数巨件主导（"放完 90% 的件"可能只值 10% 的贡献）；它更适合当
   **reward 的进度奖**（与 HPWL 同量纲），不适合当"完成度"标量。
2. **为什么 ② 用 θ=0 的局部面积**：它是**件面积的单点定义**
   （`pcb_model.Footprint.local_hull_area`，`order_plan` 的面积权重/ tie-break 也用它）
   —— 与位姿无关、与"当前局面"无关，因此"同一件放哪儿/怎么转"都不改变分母或分子。
   （实测补充：4 个旋转只有 90° 倍数 ⇒ hull AABB 只是**宽高互换** ⇒ **AABB 面积在 4 个旋转下恒等**，
   所以"当前位姿 AABB 面积"与本口径**数值相同**；选 θ=0 纯粹是为了**单点定义**。）
3. **为什么三个都能 O(1)**：固定顺序 ⇒ "已放集合 = `order[:index]`" **恒为前缀** ⇒
   预计算前缀和（`CompletionTable`），任意 `index` 都是两次查表 ⇒ **每步记日志零成本**。

三条边界（都按"完成度"的语义定）
--------------------------------

* `n == 0` 或分母为 0 ⇒ 比值记 **1.0**（没有可放的东西 ⇒ 视为已完成）；
* `c_t` 全为 0（理论上不会：每板都有 net）⇒ ③ 记 `None`（**不猜**）；
* ③ 可在建表时整块关掉（`with_contrib=False`）⇒ 省掉 n 次 `hpwl`。

与 reward 的关系（勿混）
------------------------

reward 的完成项是**常数 β**（§7.3 定稿），**不用**这里的 ②③；②③ 只进 `info`（诊断/日志）。
本模块**不产生** reward。

另：死锁终止时除本模块的三个量外，`info` 还必须带
`deadlock_kind ∈ {structural, order_locked, policy}` / `at ∈ {L, mask}` 与
`board_state.audit_failures()` 快照（§0.6 m / §5.4③）—— 那些由 env 侧组装，不在本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from env.order_plan import plan_order
from utils.kicad_utils.pcb_model import PcbModel

__all__ = [
    "CompletionReport",
    "CompletionTable",
    "marginal_contributions",
    "piece_areas",
]


# ===================== 单件量（②③ 的原料） =====================

def piece_areas(model: PcbModel) -> Dict[str, float]:
    """每件的面积（mm²）= `Footprint.local_hull_area`（**θ=0 ⇒ 与位姿/旋转无关**）。"""
    return {f.ref: f.local_hull_area for f in model.footprints}


def marginal_contributions(model: PcbModel, order: Optional[Sequence[str]] = None,
                           exclude_power: bool = True) -> Dict[str, float]:
    """`c_t` = 该件对**专家布局**（源位姿）HPWL 的边际贡献（mm，恒 ≥ 0）。

    ```text
    c_t = HPWL(全部件) − HPWL(全部件 − {t})       ← 全部在【源位姿】下算（导入快照）
    ```

    离线表（每板 n 次 `hpwl`）；**只用于 ③（诊断），不进 reward**。
    """
    refs = tuple(order) if order is not None else tuple(plan_order(model))
    all_refs = set(refs)
    if not all_refs:
        return {}
    full = float(model.hpwl(refs=all_refs, exclude_power=exclude_power))
    return {r: full - float(model.hpwl(refs=all_refs - {r}, exclude_power=exclude_power))
            for r in refs}


def _ratio(num: float, den: float) -> float:
    """`num / den`；`den <= 0` ⇒ **1.0**（没有可放的东西 ⇒ 视为已完成；§0.6 m 边界）。"""
    if den <= 0.0:
        return 1.0
    return float(num) / float(den)


# ===================== 快照 =====================

@dataclass(frozen=True)
class CompletionReport:
    """某一刻的完成度快照（三个量 + 主指标）；`index` = 已放件数。"""

    index: int
    n: int
    area_placed: float
    area_total: float
    contrib_placed: Optional[float] = None
    contrib_total: Optional[float] = None

    # ---- 三个量 ----
    @property
    def count_ratio(self) -> float:
        """① **件数占比（主指标）**。"""
        return _ratio(self.index, self.n)

    @property
    def area_ratio(self) -> float:
        """② 面积占比（θ=0 局部 hull 面积）。"""
        return _ratio(self.area_placed, self.area_total)

    @property
    def contrib_ratio(self) -> Optional[float]:
        """③ 贡献占比（专家布局的边际 HPWL 贡献份额）；表未建 ③ 时为 `None`。"""
        if self.contrib_placed is None or self.contrib_total is None:
            return None
        return _ratio(self.contrib_placed, self.contrib_total)

    # ---- 便捷 ----
    @property
    def primary(self) -> float:
        """对外只认这一个数（= ①）。"""
        return self.count_ratio

    @property
    def done(self) -> bool:
        return self.index >= self.n

    def as_info(self, prefix: str = "") -> Dict[str, Any]:
        """进 `info` 的扁平字典（**主指标键名 = `completion`**）。"""
        return {
            f"{prefix}completion": self.count_ratio,            # ★ 主指标 = 件数占比
            f"{prefix}completion_count": self.count_ratio,
            f"{prefix}completion_area": self.area_ratio,
            f"{prefix}completion_contrib": self.contrib_ratio,  # 可能 None（未建 ③）
            f"{prefix}n_placed": self.index,
            f"{prefix}n_total": self.n,
            f"{prefix}area_placed_mm2": self.area_placed,
            f"{prefix}area_total_mm2": self.area_total,
            f"{prefix}contrib_placed_mm": self.contrib_placed,
            f"{prefix}contrib_total_mm": self.contrib_total,
        }

    def describe(self) -> str:
        cr = self.contrib_ratio
        return (f"完成度 {self.primary:.1%}（{self.index}/{self.n} 件）"
                f"｜面积 {self.area_ratio:.1%}"
                f"｜贡献 {'—' if cr is None else format(cr, '.1%')}")


# ===================== 每板一张查表（O(1)） =====================

@dataclass(frozen=True)
class CompletionTable:
    """**每板一个**的完成度前缀和表 ⇒ 任意 `index` 都是 O(1)。

    固定顺序下"已放集合 = `order[:index]`"恒是前缀 ⇒ 整局只需建一次。
    """

    refs: Tuple[str, ...]
    area_prefix: Tuple[float, ...]                      # 长度 n+1：Σ_{order[:k]} A
    contrib_prefix: Optional[Tuple[float, ...]] = None  # 长度 n+1：Σ_{order[:k]} c（可选）

    @property
    def n(self) -> int:
        return len(self.refs)

    @classmethod
    def build(cls, model: PcbModel, order: Optional[Sequence[str]] = None,
              with_contrib: bool = True) -> "CompletionTable":
        """建表（离线；`with_contrib=True` 会跑 n 次 `hpwl` —— 每板一次即可）。"""
        refs = tuple(order) if order is not None else tuple(plan_order(model))
        areas = piece_areas(model)
        missing = [r for r in refs if r not in areas]
        if missing:
            raise KeyError(f"order 里有不在该板上的 ref：{missing[:5]}")

        ap = [0.0]
        for r in refs:
            ap.append(ap[-1] + areas[r])

        cp: Optional[Tuple[float, ...]] = None
        if with_contrib and refs:
            contribs = marginal_contributions(model, refs)
            if any(v != 0.0 for v in contribs.values()):
                acc = [0.0]
                for r in refs:
                    acc.append(acc[-1] + contribs[r])
                cp = tuple(acc)
        return cls(refs=refs, area_prefix=tuple(ap), contrib_prefix=cp)

    def at(self, index: int) -> CompletionReport:
        """按"已放件数"取快照（`index ∈ [0, n]`；越界抛错）。"""
        if not (0 <= int(index) <= self.n):
            raise ValueError(f"index 越界：{index} ∉ [0, {self.n}]")
        i = int(index)
        return CompletionReport(
            index=i, n=self.n,
            area_placed=self.area_prefix[i], area_total=self.area_prefix[-1],
            contrib_placed=None if self.contrib_prefix is None else self.contrib_prefix[i],
            contrib_total=None if self.contrib_prefix is None else self.contrib_prefix[-1],
        )

    def of(self, state) -> CompletionReport:
        """从 `BoardState` 取（**校验顺序一致** ⇒ 防"表/状态来自不同板"的静默错）。"""
        if tuple(state.order) != self.refs:
            raise ValueError("状态与完成度表的顺序不一致（表建错板了？）")
        return self.at(int(state.index))
