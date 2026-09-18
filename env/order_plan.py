"""固定放置顺序的规划（Owner 2026-09-17 拍案：**动态加权**）。

背景：固定顺序（动作空间不含"选件"）需要一个确定性规则。旧规则"面积降序"会让
**卫星件（去耦电容、小阻容）来得很晚**，导致"每步有没有参照物"很差：实测
迷茫步（该件的非电源 net 全冷）12.7~13.0%、一条 net 的"首→次 pad"步距中位 7.6~7.8 步。

本模块的规则：

```text
score_t(r) = w·a_norm(r) + (1 − w)·c_norm_t(r)          r ∈ 未放件
  a_norm    : log1p(hull 面积) 在【全部元件】上 min-max 到 [0,1]（静态，整局只算一次）
  c_t(r)    : |非电源 nets(r) ∩ warm_t|
              warm_t = 已放件涉及的非电源 net 集合（**每步增长**）
  c_norm_t  : 把 {c_t(r)} 在【当步剩余候选】上 min-max 到 [0,1]
  电源专属件（非电源 net 为空，占 23.5%）：c 改用 |全部 nets(r) ∩ warm_all_t|（兜底，
              避免被推到序列末尾；这类件在 HPWL/离散度里本就零贡献）
按 score 取最大；平手用 **面积↓ → pad 数↓ → ref↑**（全序）
```

权重语义：连通性拉满的候选最多能抹平 `(1−w)/w` 的面积差（w=0.7 ⇒ 约 43%）。
实测（20 板，w=0.7）：迷茫步从 13.0% 降到约 8~9%、"首→次"步距从 7.8 降到约 5~6 步，
而"大件先放"（前 25% 步已放面积、前 10% 大件位置）基本不动。

契约（三条，测试钉住）：
  1. **纯函数**：只依赖 netlist + 常量 ⇒ 与元件位置、源布局、策略**全部无关**
     ⇒ 可离线预计算、可冻结（训练时零成本）。
  2. **确定性**：tie-break 必须是**全序且以 `ref` 收尾** —— Python 的 str 哈希随机化
     （PYTHONHASHSEED）会让 set/dict 迭代顺序跨进程不同，若 key 不到 ref 这一级，
     跨进程结果可能漂移。
  3. **与策略无关** ⇒ "动态"只是派生方式，产物是一张**固定列表**（不破坏"固定顺序"拍案，
     专家库重放/检索键始终一致）。

复杂度：O(N²·net 数)，N ≤ 97 ⇒ 整板 0.14~5.2 ms（20 板合计 ≈ 37 ms），且可离线。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.kicad_utils.pcb_model import Footprint, PcbModel

DEFAULT_W_AREA = 0.7          # 面积权重（Owner 2026-09-17 拍案；连通性权重 = 1 − 该值）


def _nets_of(comps: Sequence[Footprint], model: PcbModel):
    """→ (非电源 net 集合, 全部 net 集合, 面积, pad 数) 各自按 ref 索引。"""
    np_nets: Dict[str, Set[str]] = {}
    all_nets: Dict[str, Set[str]] = {}
    area: Dict[str, float] = {}
    npads: Dict[str, int] = {}
    for f in comps:
        a = f.local_hull_aabb
        area[f.ref] = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
        npads[f.ref] = len(f.pads)
        nps, als = set(), set()
        for p in f.pads:
            net = model.nets.get(p.net_id) if p.has_net else None
            if net is None:
                continue
            als.add(net.net_id)
            if not net.is_power:
                nps.add(net.net_id)
        np_nets[f.ref], all_nets[f.ref] = nps, als
    return np_nets, all_nets, area, npads


def area_scores(area: Dict[str, float]) -> Dict[str, float]:
    """`log1p(面积)` 在**全部元件**上 min-max 到 [0,1]（静态分）。

    必须取 log：面积分布是重尾的（中位 5.1mm²、最大 3184mm²，≈600×），
    线性 min-max 会把 99% 的件压成 ≈0。
    """
    ys = {r: math.log1p(v) for r, v in area.items()}
    lo, hi = min(ys.values()), max(ys.values())
    span = hi - lo
    return {r: ((y - lo) / span if span > 0 else 0.0) for r, y in ys.items()}


def plan_order(model: PcbModel, comps: Optional[Iterable[Footprint]] = None,
               w_area: float = DEFAULT_W_AREA,
               power_fallback: bool = True) -> List[str]:
    """规划该板的固定放置顺序（确定性；可离线预计算）。

    Args:
        comps: 参与规划的元件（缺省 = 全部 footprint）。
        w_area: 面积权重 `w`（连通性权重 `1−w`）；=1.0 即退化为"纯面积降序"。
        power_fallback: 对"非电源 net 为空"的件是否用"共享电源网数"兜底。
    """
    comps = list(model.footprints) if comps is None else list(comps)
    np_nets, all_nets, area, npads = _nets_of(comps, model)
    a_norm = area_scores(area)

    order: List[str] = []
    warm: Set[str] = set()          # 已亮过的非电源 net
    warm_all: Set[str] = set()      # 已亮过的全部 net
    left = {f.ref for f in comps}
    while left:
        raw: Dict[str, int] = {}
        for r in left:
            if power_fallback and not np_nets[r]:
                raw[r] = len(all_nets[r] & warm_all)
            else:
                raw[r] = len(np_nets[r] & warm)
        lo_c, hi_c = min(raw.values()), max(raw.values())
        span = hi_c - lo_c
        # ⚠️ tie-break 必须是全序且以 ref 收尾（防 str 哈希随机化导致跨进程漂移）
        nxt = min(left, key=lambda r: (
            -(w_area * a_norm[r] + (1.0 - w_area) * ((raw[r] - lo_c) / span if span else 0.0)),
            -area[r], -npads[r], r))
        order.append(nxt)
        left.discard(nxt)
        warm |= np_nets[nxt]
        warm_all |= all_nets[nxt]
    return order


def order_metrics(model: PcbModel, order: Sequence[str]) -> Dict[str, float]:
    """顺序诊断指标（供测试与调参）：

    * `fog_a`      ：迷茫步占比 —— 该件有非电源 net、但**全部还冷**（G 无参照物）
    * `delay_med`  ：一条 net 的"第 1 个 pad → 第 2 个 pad"相隔步数（credit assignment 延迟）中位
    * `a25`        ：前 25% 步数内已放掉的总面积占比（"大件先放"强度）
    * `big_pos`    ：前 10% 大件的平均归一化步位置（越小越早放）
    * `pw_pos`     ：电源专属件的平均归一化步位置（nan 表示该板没有这类件）
    """
    byref = {f.ref: f for f in model.footprints}
    _, all_nets, _, _ = _nets_of([byref[r] for r in order], model)
    np_nets = {r: {n for n in ns if not model.nets[n].is_power} for r, ns in all_nets.items()}
    area = {r: max(0.0, (byref[r].local_hull_aabb[2] - byref[r].local_hull_aabb[0])
                   * (byref[r].local_hull_aabb[3] - byref[r].local_hull_aabb[1])) for r in order}

    n = len(order)
    pos = {r: t for t, r in enumerate(order)}
    warm, first, second, fog = set(), {}, {}, 0
    for t, r in enumerate(order):
        nets = np_nets[r]
        if nets and not (nets & warm):
            fog += 1
        for x in nets:
            first.setdefault(x, t)
        for x in nets:
            if x in first and first[x] < t and x not in second:
                second[x] = t
        warm |= nets
    delay = [second[x] - first[x] for x in second]
    cum = 0.0
    total_area = sum(area.values())
    a25 = 0.0
    for t, r in enumerate(order):
        cum += area[r]
        if t == max(0, int(0.25 * n) - 1):
            a25 = cum / total_area if total_area else 0.0
    big = sorted(order, key=lambda r: (-area[r], r))[:max(1, n // 10)]
    pw = [pos[r] / (n - 1) for r in order if not np_nets[r]]
    return dict(fog_a=fog / n,
                delay_med=float(sorted(delay)[len(delay) // 2]) if delay else 0.0,
                a25=a25,
                big_pos=float(sum(pos[r] for r in big) / len(big) / (n - 1)) if n > 1 else 0.0,
                pw_pos=float(sum(pw) / len(pw)) if pw else float("nan"))
