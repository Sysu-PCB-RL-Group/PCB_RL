"""局面的**唯一状态载体**：只存位姿，`placed` / `keepouts` / `others` 全部**派生**。

为什么要它（台账 §0.6 q，probe171 实测）—— "已放置"此前**没有唯一载体**，由调用方每次以
三种形状临时传（`Sequence[Footprint]` / `Mapping[ref, Pose]` / 隐式烘焙进 `L_next0`），
且"谁是已放置"由调用方自己算，**零校验** ⇒ 三类**静默**错：

```text
① placed 误含 comp 自己 ⇒ L 静默缩小甚至全 0（jackt 152→0、bliysfyt 274→0）
                          ⚠️ 而"L 全 0"正是 §5.4③ 的【死锁终止条件】⇒ 调用方 bug 伪装成"真实死锁"
② placed 漏一件        ⇒ 静默多出合法格（lr1388 漏 D1 = 1151 格），抽样 Judge 全判违规（危险方向）
③ 传源位姿（过期几何） ⇒ G 侧 ΔHPWL 最大 15.06mm（jackt）、L 侧 4 格放过违规（probe165）
```

本模块把状态收敛成**一份 `{ref: (x, y, rot)}`**（只记已放件），所有下游形状都由它现算派生：

```text
poses  ──派生──▶ placed  = 每个已放件的 place_at 克隆（**每次都新建 ⇒ 不存在"陈旧对象"**）
       ──派生──▶ keepouts = state_keepouts(model, placed)          （口径 A：开槽随件）
       ──派生──▶ others   = {ref: (x, y, rot)}                      （Judge 的形状）
```

**为什么校验放在这里、而**不是**放进 `legal_masks`**：`legal_masks(model, placed, comp)` 不知道 `order`，
无法区分两种"看起来缺件"的情形 ——
  * **合法**：`order` 里排在 `comp` 之后（还没轮到）的件，本来就不该在 `placed` 里；
  * **危险**：本该已放的件被漏掉（probe171：lr1388 漏 `D1` ⇒ 静默多出 1151 格且 `Judge` 全判违规）。
⇒ **校验必须发生在"知道顺序"的地方**（本类 / env）。在 `legal_masks` 里查会满屏误报
（测试与探针惯用的"拿走几件、其余保持源位姿"场面就属于合法的第一种）。

三条不变量（测试钉住）：
  1. **索引只增不减**：`place()` 只把 `order[index]` 落子并把 `index+1`，**没有**"移动已放件"的 API
     ⇒ 从结构上保证 §0.6 q-⑦ 的单调性前提（`lookahead` 的"放件只增障碍"成立）。
  2. **派生而非持有**：`placed`/`keepouts` 每次由 `poses` 现算（带缓存，落子后自动失效）
     ⇒ 消灭"忘记替换成位姿化克隆"这一类错误（§0.6 o）。
  3. **可校验**：`audit()` / `check()` 用 `audit_placed` 做 O(件数) 的契约检查
     （自含 / 漏件 / 外来 ref / 重复），不干净时按 `PCB_RL_AUDIT=1` 抛错，平时计数 + 首次告警。
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from env.order_plan import plan_order
from utils.kicad_utils.pcb_model import (
    Footprint,
    KeepoutRegion,
    PcbModel,
    place_at,
    state_keepouts,
)

Pose = Tuple[float, float, int]           # (x_mm, y_mm, 转换角 {0,1,2,3})
AUDIT_ENV = "PCB_RL_AUDIT"                # 与项目其它审计开关一致：置 1 ⇒ 违规直接抛错


# ===================== 契约校验（②） =====================

@dataclass(frozen=True)
class PlacedAudit:
    """`placed` 契约的体检报告（**只看集合，不看几何**）。"""

    n_placed: int
    missing: Tuple[str, ...]        # order 里排在 comp 之前、却没进 placed（漏件 ⇒ 静默多出合法格）
    extra: Tuple[str, ...]          # 不在该板上的 ref
    duplicated: Tuple[str, ...]     # 同一 ref 出现多次（会让"按 ref 建键"丢件）
    self_included: bool             # placed 里含 comp 自己（L 静默缩小甚至全 0）
    comp_ref: Optional[str] = None  # 体检时的当前件（只为报告可定位）

    @property
    def clean(self) -> bool:
        return not (self.missing or self.extra or self.duplicated or self.self_included)

    def describe(self) -> str:
        if self.clean:
            return f"clean（{self.n_placed} 件已放）"
        bits: List[str] = []
        if self.comp_ref is not None:
            bits.append(f"comp={self.comp_ref}")
        if self.self_included:
            bits.append("含 comp 自己 ⇒ L 会静默缩小甚至全 0（会被误判为死锁）")
        if self.missing:
            bits.append(f"漏 {len(self.missing)} 件：{list(self.missing[:6])} ⇒ 静默多出合法格（危险方向）")
        if self.extra:
            bits.append(f"含 {len(self.extra)} 个不在该板上的 ref：{list(self.extra[:6])}")
        if self.duplicated:
            bits.append(f"重复 ref：{list(self.duplicated[:6])}")
        return "；".join(bits)


class PlacedContractError(AssertionError):
    """`placed` 契约被违反（§0.6 q-①②）：自含 / 漏件 / 外来 / 重复。"""


_FAILURES = 0
_WARNED: set = set()


def audit_placed(model: PcbModel, placed: Sequence[Footprint],
                 comp: Optional[Footprint] = None,
                 order: Optional[Sequence[str]] = None) -> PlacedAudit:
    """O(件数) 的契约体检：**期望 = `order` 中排在 `comp` 之前的那些件**。

    Args:
        placed: 参与比较的"已放置"集合（**不含** comp）。
        comp: 当前待判定/待放置的件（None ⇒ 期望 = order 全集）。
        order: 固定放置顺序（`plan_order` 的产物）；缺省退化为"模型全集 − {comp}"。
    """
    refs = [f.ref for f in placed]
    seen, dup = set(), set()
    for r in refs:
        if r in seen:
            dup.add(r)
        seen.add(r)
    known = {f.ref for f in model.footprints}
    extra = tuple(sorted(r for r in seen if r not in known))
    self_inc = comp is not None and comp.ref in seen
    if order is not None:
        seq = list(order)
        cut = seq.index(comp.ref) if (comp is not None and comp.ref in seq) else len(seq)
        want = set(seq[:cut])
    else:
        want = known - ({comp.ref} if comp is not None else set())
    return PlacedAudit(len(refs), tuple(sorted(want - seen)), extra, tuple(sorted(dup)), self_inc,
                       comp.ref if comp is not None else None)


def require_clean(model: PcbModel, placed: Sequence[Footprint],
                  comp: Optional[Footprint] = None,
                  order: Optional[Sequence[str]] = None,
                  strict: Optional[bool] = None, where: str = "") -> PlacedAudit:
    """同 `audit_placed`；不干净时按 `strict` **抛错**（默认读 `PCB_RL_AUDIT=1`）或计数 + 首次告警。

    Returns:
        体检报告（`strict=True` 且不干净时**抛 `PlacedContractError`**，不返回）。
    """
    global _FAILURES
    rep = audit_placed(model, placed, comp, order)
    if rep.clean:
        return rep
    if strict is None:
        strict = os.environ.get(AUDIT_ENV) == "1"
    msg = f"[placed 契约]{(' ' + where) if where else ''} {rep.describe()}"
    if strict:
        raise PlacedContractError(msg)
    _FAILURES += 1
    key = rep.describe().split("：")[0].split(" ⇒")[0]
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg + f"（同类只提示一次；置 {AUDIT_ENV}=1 可让它直接失败）", stacklevel=2)
    return rep


def audit_failures() -> int:
    """累计"发现问题但没抛错"的次数（供训练日志/监控；置 `PCB_RL_AUDIT=1` 时恒为 0）。"""
    return _FAILURES


def reset_audit_stats() -> None:
    """清零计数与"已告警"标记（测试用）。"""
    global _FAILURES
    _FAILURES = 0
    _WARNED.clear()


# ===================== 形状桥接（唯一实现，避免各处手写） =====================

def derive_placed(model: PcbModel, poses: Mapping[str, Pose]) -> Tuple[Footprint, ...]:
    """`{ref: (x, y, rot)}` → 位姿化 `placed` 元组（**每次现算克隆** ⇒ 不会过期）。"""
    byref = {f.ref: f for f in model.footprints}
    out = []
    for ref, (x, y, rot) in poses.items():
        src = byref.get(ref)
        if src is None:
            raise KeyError(f"poses 里有不在该板上的 ref：{ref}")
        out.append(place_at(src, (float(x), float(y)), int(rot)))
    return tuple(out)


def others_of(placed: Sequence[Footprint]) -> Dict[str, Pose]:
    """`placed` → `Judge` 的 `others` 形状 `{ref: (x, y, rot)}`（**唯一实现**）。"""
    return {f.ref: (float(f.anchor[0]), float(f.anchor[1]), int(f.rotation_index)) for f in placed}


# ===================== 状态本体（①） =====================

@dataclass
class BoardState:
    """一块板的局面：`order`（固定顺序）+ `index`（下一个待放）+ `poses`（**只记已放件**）。

    外部请用 `BoardState.start(model)` 起手、`st.place(k, i, j)` 前进（返回**新状态**，不改自己）。
    `placed` / `keepouts` / `others` / `audit` 都是派生量。
    """

    model: PcbModel
    order: Tuple[str, ...]
    index: int = 0
    _poses: Dict[str, Pose] = field(default_factory=dict, repr=False)

    # ---- 构造与内部不变量 ----
    def __post_init__(self) -> None:
        self._byref = {f.ref: f for f in self.model.footprints}
        self._placed_cache: Optional[Tuple[Footprint, ...]] = None
        self._ko_cache: Optional[Tuple[KeepoutRegion, ...]] = None
        if not (0 <= self.index <= len(self.order)):
            raise ValueError(f"index 越界：{self.index} ∉ [0, {len(self.order)}]")
        want = set(self.order[: self.index])
        if set(self._poses) != want:
            got = sorted(set(self._poses))
            raise ValueError(
                "poses 与 order[:index] 不一致（外部构造请用 BoardState.start + place）："
                f"多 {sorted(set(got) - want)[:5]} / 少 {sorted(want - set(got))[:5]}")

    @classmethod
    def start(cls, model: PcbModel, order: Optional[Iterable[str]] = None) -> "BoardState":
        """起手状态（index=0、poses={}）；`order` 缺省 = `plan_order(model)`（固定顺序）。"""
        seq = tuple(order) if order is not None else tuple(plan_order(model))
        known = {f.ref for f in model.footprints}
        unknown = [r for r in seq if r not in known]
        if unknown:
            raise ValueError(f"order 里有不在该板上的 ref：{unknown[:5]}")
        if len(set(seq)) != len(seq):
            raise ValueError("order 有重复 ref")
        return cls(model=model, order=seq)

    # ---- 基本查询 ----
    @property
    def n_total(self) -> int:
        return len(self.order)

    @property
    def placed_refs(self) -> Tuple[str, ...]:
        return self.order[: self.index]

    @property
    def remaining_refs(self) -> Tuple[str, ...]:
        return self.order[self.index:]

    @property
    def done(self) -> bool:
        return self.index >= len(self.order)

    @property
    def progress(self) -> float:
        """已放件数占比 `index / n_total`（**"放置完成度"的精确定义仍是 §0.6 m 未决项**，这里只给原始比值）。"""
        return self.index / len(self.order) if self.order else 1.0

    @property
    def poses(self) -> Mapping[str, Pose]:
        """**只读**视图：外部改不了内部状态（避免绕过 `place()` 的"移动已放件"）。"""
        from types import MappingProxyType
        return MappingProxyType(self._poses)

    @property
    def comp(self) -> Footprint:
        """当前待放置的件（`order[index]` 的模型对象）。"""
        if self.done:
            raise IndexError("已放完（index == n_total）")
        return self._byref[self.order[self.index]]

    @property
    def next_comp(self) -> Optional[Footprint]:
        """前瞻对象（`order[index+1]`；不存在则 None）。"""
        if self.index + 1 >= len(self.order):
            return None
        return self._byref[self.order[self.index + 1]]

    # ---- 派生量 ----
    @property
    def placed(self) -> Tuple[Footprint, ...]:
        """已放件的**位姿化克隆**（派生自 `poses`；同名 `place_at` 产物 = `legal_masks` 要的形状）。"""
        if self._placed_cache is None:
            self._placed_cache = derive_placed(self.model, {r: self._poses[r] for r in self.placed_refs})
        return self._placed_cache

    @property
    def keepouts(self) -> Tuple[KeepoutRegion, ...]:
        """当前局面的禁布区（口径 A：**只有已放件**的槽在场且随件位姿）。"""
        if self._ko_cache is None:
            self._ko_cache = state_keepouts(self.model, self.placed)
        return self._ko_cache

    @property
    def others(self) -> Dict[str, Pose]:
        """`Judge.judge(..., others=)` 的形状（与 `placed` 同源）。"""
        return others_of(self.placed)

    # ---- 前进（唯一的写入口） ----
    def place(self, k: int, i: int, j: int, spec=None, legal=None) -> "BoardState":
        """把当前件放到"格中心 `(i, j)` + 旋转 `k`"，返回**新状态**（不改自己 ⇒ 可回滚/对比）。

        Args:
            k: 旋转 `0~3`。
            i, j: 格索引（须在 `[0, m)` 内）。
            spec: 栅格映射；缺省 `grid_spec(model)`（200×200 `fill`）。
            legal: 可选，`legal_masks` 的产物（给定时要求 `legal[k, i, j] == True`）。
        """
        if self.done:
            raise IndexError("已放完，不能再 place")
        k = int(k)
        if k not in (0, 1, 2, 3):
            raise ValueError(f"旋转必须是 0~3，收到 {k}")
        if spec is None:
            from env.grid_render import grid_spec
            spec = grid_spec(self.model)
        m = spec.m
        if not (0 <= int(i) < m and 0 <= int(j) < m):
            raise ValueError(f"格索引越界：({i}, {j}) ∉ [0,{m})²")
        if legal is not None and not bool(legal[k, int(i), int(j)]):
            raise ValueError(f"动作 {k}@({i},{j}) 按给定 L 非法（legal[{k},{i},{j}] = False）")
        x, y = spec.mm_of(int(i), int(j))
        poses = dict(self._poses)
        poses[self.order[self.index]] = (float(x), float(y), k)
        return BoardState(model=self.model, order=self.order, index=self.index + 1, _poses=poses)

    # ---- 体检 ----
    def audit(self) -> PlacedAudit:
        """自检（自含 / 漏件 / 外来 / 重复）。由本类构造的状态恒为 `clean` —— 这是"独立复核"。"""
        comp = None if self.done else self.comp
        return audit_placed(self.model, self.placed, comp, self.order)

    def check(self, strict: Optional[bool] = None) -> PlacedAudit:
        """同 `audit()`，但可按 `strict` 抛错（缺省读 `PCB_RL_AUDIT`）。"""
        comp = None if self.done else self.comp
        return require_clean(self.model, self.placed, comp, self.order,
                             strict=strict, where=f"order[{self.index}]/{comp.ref if comp else '-'}")
