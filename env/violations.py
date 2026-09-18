"""违规判定（**单点判定器**）—— 硬重叠 + 软间距 + 出界 三层筛选合并为一个总违规项。

标准依据：`docs/讨论结果/导入信息重构.md` §3.1~§3.5（判定式）+ §1.4（层）+ §1.5（板框）。

**Owner 拍案（2026-09-16）**
1. **放置必须禁止违规** ⇒ 本判定器服务于**动作屏蔽（mask）**；后续 reward 里**不再有违规罚分**
   （违规从根本上被禁止，不再作为"可接受但要罚"的状态）。
2. 违规 = **硬重叠 ∨ 软间距 ∨ 出界** 合并成**一个总违规项**；返回时**附带违规方向**（分类），
   **不含细节**（不返回具体 pad 对、不返回 gap 量值）。
3. **元件不跨层移动**（动作空间**不含层动作**）⇒ footprint 与 pad 都只在**自己原本的层**里活动；
   pad 的层就是**导入时的记录事实**，**不需要任何换算**。因此 `Pose` 里**没有层**：
   "翻面"在本 API 里**不可表示**（类型层面堵死，而不是靠运行时断言）。

判定式（逐字对应 §3.1~§3.5）
```text
硬重叠 A  pad↔pad    ：同层 ∧ 不同 footprint ∧ AABB 相交(gap<0) ∧ 不看 net
硬重叠 B  pad↔via    ：同层 ∧ 相交 ∧ 双方 netcode≠0 ∧ netcode 不相等
硬重叠 C  pad↔禁布区 ：同层（开槽两层通用）∧ 相交 ∧ 排除禁布区项所属 footprint 自己的 pad
软间距 S1 pad↔pad    ：跨封装 ∧ 同层 ∧ 不看 net ∧ 0≤gap<0.2
软间距 S2 pad↔via    ：同层 ∧ 双方 netcode≠0 ∧ netcode 不相等 ∧ 0≤gap<0.2
软间距 S3 pad↔禁布区 ：同层 ∧ 排除所属 footprint ∧ 0≤gap<0.2
出界                 ：footprint 的 pad 的 AABB 越出板框外接矩形（不管 net、不豁免）
```
`gap = max(a.x1−b.x2, b.x1−a.x2, a.y1−b.y2, b.y1−a.y2)`：**`gap<0` = 相交；`gap=0` = 相切（不算相交）**。

几何口径：全部用 **pad AABB**，在**候选位姿**（锚点 `(x, y)` + 转换角）下由
`板坐标 = 锚点 + R(−θ)·局部` 现算（`rotate_aabb(pad.local_aabb, ...)`，θ 为 90° 倍数 ⇒ 精确）；
板框用**外接矩形、不内缩**。
层口径：**"同层"= pad 的记录层集合有交集**（`Pad.layers` ∈ {F} / {B} / {F,B}，§1.4 的源数据事实，
因元件不跨层移动而**恒定**）；开槽视作 F/B 两层通用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from utils.kicad_utils.pcb_model import (
    AABB,
    Footprint,
    KeepoutRegion,
    state_keepouts,
    Pad,
    PcbModel,
    Via,
    rotate_aabb,
)

# ===================== 违规方向（分类；不含细节） =====================
HARD_PAD_PAD = "hard.pad_pad"            # 硬重叠 A
HARD_PAD_VIA = "hard.pad_via"            # 硬重叠 B
HARD_PAD_KEEPOUT = "hard.pad_keepout"    # 硬重叠 C
SOFT_PAD_PAD = "soft.pad_pad"            # 软间距 S1
SOFT_PAD_VIA = "soft.pad_via"            # 软间距 S2
SOFT_PAD_KEEPOUT = "soft.pad_keepout"    # 软间距 S3
OUT_OF_BOUNDS = "out_of_bounds"          # 出界

ALL_REASONS: Tuple[str, ...] = (
    HARD_PAD_PAD, HARD_PAD_VIA, HARD_PAD_KEEPOUT,
    SOFT_PAD_PAD, SOFT_PAD_VIA, SOFT_PAD_KEEPOUT,
    OUT_OF_BOUNDS,
)

_HARD_REASONS = frozenset({HARD_PAD_PAD, HARD_PAD_VIA, HARD_PAD_KEEPOUT})
_SOFT_REASONS = frozenset({SOFT_PAD_PAD, SOFT_PAD_VIA, SOFT_PAD_KEEPOUT})

# 软间距阈值 = `.kicad_pro` 的 netclass `Default.clearance`（§3.4：严格 0.2，不做豁免）
SOFT_GAP_MM = 0.2

# pad AABB 缓存上限（位姿空间是连续的 ⇒ 必须封顶，否则训练中会无界增长）
_PAD_CACHE_MAX = 8192

class Pose(NamedTuple):
    """位姿 = **锚点位置（mm，连续坐标）+ 转换角 `{0,1,2,3}`**。

    **不含层**：元件只在原本的层里活动（拍案 3）⇒ 跨层移动不可表示。
    坐标单位 mm + KiCad 板坐标（**Y 向下**）；是否量化到网格属 env 层（台账 §5.7 精度分工：判定连续 mm、动作离散到 cell）。
    """

    x: float
    y: float
    rot: int


def pose_of(fp: Footprint, rotation_index: Optional[int] = None) -> Pose:
    """footprint 的**源位姿**（导入位置 + 转换角；`rotation_index` 可覆盖角度）。"""
    return Pose(fp.anchor[0], fp.anchor[1],
                fp.rotation_index if rotation_index is None else rotation_index)


# ===================== 结果（总违规项 + 方向） =====================


@dataclass(frozen=True)
class ViolationResult:
    """判定结果：**一个总违规项** + **违规方向**（分类）。

    Attributes:
        reasons: 命中的违规方向（`ALL_REASONS` 的子集，按固定顺序、去重）；空元组 = 无违规。
        counts: 每个方向的**命中次数**（仅分类计数，不含具体对象/量值）。
    """

    reasons: Tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_violation(self) -> bool:
        """总违规项：硬重叠 ∨ 软间距 ∨ 出界（三者任一命中即为 True）。"""
        return bool(self.reasons)

    @property
    def hard(self) -> bool:
        """是否含硬重叠（A/B/C）。"""
        return any(r in _HARD_REASONS for r in self.reasons)

    @property
    def soft(self) -> bool:
        """是否含软间距（S1/S2/S3）。"""
        return any(r in _SOFT_REASONS for r in self.reasons)

    @property
    def out_of_bounds(self) -> bool:
        """是否含出界。"""
        return OUT_OF_BOUNDS in self.reasons

    def __bool__(self) -> bool:
        return self.is_violation

    def describe(self) -> str:
        """一行式描述（给日志/可视化用），例：`hard.pad_via + out_of_bounds`。"""
        if not self.reasons:
            return "ok"
        return " + ".join(f"{r}×{self.counts.get(r, 1)}" for r in self.reasons)

    def __str__(self) -> str:  # pragma: no cover - 调试友好
        return f"ViolationResult({self.describe()})"


_NO_VIOLATION = ViolationResult()


# ===================== 几何工具 =====================


def gap(a: AABB, b: AABB) -> float:
    """两 AABB 的间距：`<0` 相交、`==0` 相切、`>0` 分离。"""
    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])


def _overlap(a: AABB, b: AABB) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _expand(aabb: AABB, d: float) -> AABB:
    """AABB 各边外扩 `d`（粗筛用：**必须**外扩软间距，否则 `0≤gap<软间距` 的对会被漏掉）。"""
    return (aabb[0] - d, aabb[1] - d, aabb[2] + d, aabb[3] + d)


class _ViaGrid:
    """via 的均匀网格索引（via 是静态对象 ⇒ 建一次；避免每次候选都扫 3422 个 via）。"""

    def __init__(self, vias: Sequence[Via], cell: float = 2.0):
        self.cell = cell
        self.grid: Dict[Tuple[int, int], List[Via]] = {}
        for v in vias:
            for key in self._keys(v.aabb):
                self.grid.setdefault(key, []).append(v)

    def _keys(self, box: AABB) -> Iterable[Tuple[int, int]]:
        c = self.cell
        for gx in range(int(math.floor(box[0] / c)), int(math.floor(box[2] / c)) + 1):
            for gy in range(int(math.floor(box[1] / c)), int(math.floor(box[3] / c)) + 1):
                yield (gx, gy)

    def query(self, box: AABB, pad_mm: float = 0.0) -> List[Via]:
        """取与 `box`（外扩 `pad_mm`）可能相交的 via 候选（可能含重复，调用方按对象去重）。"""
        expanded = (box[0] - pad_mm, box[1] - pad_mm, box[2] + pad_mm, box[3] + pad_mm)
        seen: Dict[int, Via] = {}
        for key in self._keys(expanded):
            for v in self.grid.get(key, ()):
                seen[id(v)] = v
        return list(seen.values())


# ===================== 判定器 =====================


class Judge:
    """单点判定器：静态模型（`PcbModel`）+ 动态摆放 → `ViolationResult`。

    设计要点：
      * **三处消费点共用同一实现**（`env.step`、掩码预演、专家轨迹/诊断），杜绝"口径靠注释同步"；
      * `others` 由调用方给定 ⇒ 既可"只比已放置"（step/掩码），也可"比全局快照"（专家轨迹）；
      * 只依赖转换数据（`PcbModel`），不读 pcbnew ⇒ 可在训练进程里快速调用。
    """

    def __init__(self, model: PcbModel, soft_gap: float = SOFT_GAP_MM, margin: float = 0.0,
                 keepouts: Optional[Sequence[KeepoutRegion]] = None,
                 placed: Optional[Sequence[Footprint]] = None):
        self.model = model
        self.soft_gap = soft_gap
        # 板框外接矩形；**不内缩**（margin=0）；如需内缩由调用方显式给
        self.valid_area: AABB = model.board_outline.valid_area(margin)
        self._fp: Dict[str, Footprint] = {f.ref: f for f in model.footprints}
        # 禁布区（开槽）：**口径 (A) —— 槽属于封装、随件移动**（probe169 用 KiCad 本体裁决；§0.6 p）。
        #   * `keepouts=` 直接给集合（最高优先，测试/特殊口径用）
        #   * `placed=`  给**当前局面的已放件** ⇒ 据此推导（**env 必须传**；未放置件的槽不上板）
        #   * 缺省        = `model.keepouts`（= "视全部件都在源位姿"的**诊断/参照口径**，
        #                  与旧行为一致；对参照布局的违规统计不受影响）
        # ⚠️ 注意别用 `judge(..., others=...)` 去推 keepout —— `others` 是"比较对象"，
        #    未必等于"已放件"（诊断用例会传空集/子集），混用会让 keepout 消失（2026-09-18 踩过）。
        if keepouts is not None:
            self._keepouts: Tuple[KeepoutRegion, ...] = tuple(keepouts)
        elif placed is not None:
            self._keepouts = state_keepouts(model, tuple(placed))
        else:
            self._keepouts = model.keepouts
        self._via_grid = _ViaGrid(model.vias)
        # pad AABB 现算（随位姿变化）⇒ 有界缓存；层是记录事实、不参与缓存键
        self._pad_cache: Dict[Tuple, Tuple[Tuple[Pad, AABB, Tuple[int, ...]], ...]] = {}

    # ---------- 几何口径 ----------

    def _pads_at(self, ref: str, x: float, y: float, rot: int):
        """`ref` 在候选位姿 `(x, y, rot)` 下的 pad 列表：`(pad, AABB, pad 的记录层)`。"""
        key = (ref, x, y, rot)
        cached = self._pad_cache.get(key)
        if cached is not None:
            return cached
        fp = self._fp[ref]
        out = tuple(
            (pad, rotate_aabb(pad.local_aabb, rot * 90.0, (x, y)), pad.layers) for pad in fp.pads
        )
        if len(self._pad_cache) >= _PAD_CACHE_MAX:
            self._pad_cache.clear()          # 有界：训练里候选位姿极多，避免无界增长
        self._pad_cache[key] = out
        return out

    def _hull_at(self, ref: str, x: float, y: float, rot: int) -> AABB:
        """`ref` 在候选位姿 `(x, y, rot)` 下的 hull AABB（仅用于粗筛）。"""
        return rotate_aabb(self._fp[ref].local_hull_aabb, rot * 90.0, (x, y))

    # ---------- 判定：单件（mask / step 用） ----------

    def blocks(self, ref: str, pose: Pose, others: Mapping[str, Pose]) -> bool:
        """快速布尔判定（命中即返回 True，用于动作屏蔽的热路径）。"""
        return self.judge(ref, pose, others, _short_circuit=True).is_violation

    def judge(self, ref: str, pose: Pose,
              others: Mapping[str, Pose], _short_circuit: bool = False) -> ViolationResult:
        """判定"把 `ref` 放到候选位姿 `pose`"是否违规（只与前 `others` 比）。

        Args:
            ref: 目标 footprint 的 ref（须在 `model.footprints` 里）。
            pose: 候选位姿 `(x, y, 转换角)`（**无层**：元件只在原层活动）。
                  `Pose` 或等价的 `(x, y, rot)` 三元组都可以。
            others: 其它**参与比较**的对象：`ref → pose`（同上）。
                    传入"已放置集合"= step/掩码口径；传入"全局快照"= 专家轨迹口径。
            _short_circuit: True 时只关心"有没有"，命中首个方向即返回（略过计数）。
        """
        x, y, rot = pose
        cand = self._pads_at(ref, x, y, rot)
        cand_hull = self._hull_at(ref, x, y, rot)
        counts: Dict[str, int] = {}

        def hit(reason: str) -> bool:
            counts[reason] = counts.get(reason, 0) + 1
            return _short_circuit

        # ① 出界：本件 pad 的 AABB 越出板框外接矩形（不管 net、不豁免）
        for _pad, aabb, _layers in cand:
            if self._is_out(aabb) and hit(OUT_OF_BOUNDS):
                return self._result(counts)

        # ② 本件 pad ↔ 其它 footprint 的 pad（硬重叠 A / 软间距 S1）
        for oref, o_pose in others.items():
            if oref == ref:
                continue
            if oref not in self._fp:
                continue                     # 零 pad 对象（如 TT马达）不在 footprints 里（已作禁布区）
            ox, oy, orot = o_pose
            if not _overlap(_expand(cand_hull, self.soft_gap), self._hull_at(oref, ox, oy, orot)):
                continue                     # 粗筛（外扩软间距）：再远则任何 pad 都不可能 < 软间距
            for _pad, aabb, layers in cand:
                for _opad, oaabb, olayers in self._pads_at(oref, ox, oy, orot):
                    if not (set(layers) & set(olayers)):
                        continue             # 异层不比
                    g = gap(aabb, oaabb)
                    if g < 0.0:
                        if hit(HARD_PAD_PAD):
                            return self._result(counts)
                    elif g < self.soft_gap:
                        if hit(SOFT_PAD_PAD):
                            return self._result(counts)

        # ③ 本件 pad ↔ via（硬重叠 B / 软间距 S2：双方 netcode≠0 ∧ 不相等）
        for _pad, aabb, layers in cand:
            if not _pad.has_net:
                continue
            for v in self._via_grid.query(aabb, self.soft_gap):
                if not v.has_net or v.netcode == _pad.netcode:
                    continue
                if not (set(layers) & set(v.layers)):
                    continue
                g = gap(aabb, v.aabb)
                if g < 0.0:
                    if hit(HARD_PAD_VIA):
                        return self._result(counts)
                elif g < self.soft_gap:
                    if hit(SOFT_PAD_VIA):
                        return self._result(counts)

        # ④ 本件 pad ↔ 禁布区（硬重叠 C / 软间距 S3：排除禁布区项所属 footprint 自己的 pad）
        for k in self._keepouts:
            if k.owner_ref == ref:
                continue
            if not _overlap(_expand(cand_hull, self.soft_gap), k.aabb):
                continue                     # 粗筛（外扩软间距）
            for _pad, aabb, layers in cand:
                if not (set(layers) & set(k.layers)):
                    continue
                g = gap(aabb, k.aabb)
                if g < 0.0:
                    if hit(HARD_PAD_KEEPOUT):
                        return self._result(counts)
                elif g < self.soft_gap:
                    if hit(SOFT_PAD_KEEPOUT):
                        return self._result(counts)

        return self._result(counts)

    # ---------- 判定：全板（诊断 / 验收 / 终局） ----------

    def judge_all(self, placements: Mapping[str, Pose]) -> ViolationResult:
        """整板判定：对 `placements` 里的每个 footprint 跑一次 `judge`。

        用于诊断 / 参照复现 / episode 终局校验（O(件数²) 但件数 ≤ 97，且每层都有粗筛）。

        **配对去重**：pad↔pad 是"对称对"，若每件都与"其余全部"比会把同一对计两次
        ⇒ 这里只让每件与 **ref 序在后**的件比（`r > ref`），保证**每一对只计一次**（§3.1）。
        仅影响 `others` 相关的 pad↔pad 分支；出界 / via / 禁布区都是"pad↔静态对象"，
        与 `others` 无关，不受影响。
        """
        counts: Dict[str, int] = {}
        for ref, pose in placements.items():
            others = {r: p for r, p in placements.items() if r > ref}
            res = self.judge(ref, pose, others)
            for r, c in res.counts.items():
                counts[r] = counts.get(r, 0) + c
        return self._result(counts)

    def source_placements(self) -> Dict[str, Pose]:
        """源布局（导入位置 + 转换角）—— 用于复现 §0.4 参照诊断数字。"""
        return {f.ref: pose_of(f) for f in self.model.footprints}

    # ---------- 内部 ----------

    def _is_out(self, aabb: AABB) -> bool:
        ax1, ay1, ax2, ay2 = self.valid_area
        return aabb[0] < ax1 or aabb[1] < ay1 or aabb[2] > ax2 or aabb[3] > ay2

    @staticmethod
    def _result(counts: Dict[str, int]) -> ViolationResult:
        if not counts:
            return _NO_VIOLATION
        reasons = tuple(r for r in ALL_REASONS if r in counts)
        return ViolationResult(reasons=reasons, counts=dict(counts))


# ===================== 便捷入口 =====================


def judge_model(model: PcbModel, soft_gap: float = SOFT_GAP_MM, margin: float = 0.0) -> ViolationResult:
    """对"源布局"整板判定（诊断用最短路径）。"""
    judge = Judge(model, soft_gap=soft_gap, margin=margin)
    return judge.judge_all(judge.source_placements())
