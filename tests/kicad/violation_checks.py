# -*- coding: utf-8 -*-
"""规则检查集合（**与实现解耦**）：既能验收 `env.violations`，也能用来"杀"变异体。

用法（两种消费方共用同一份检查）：
    from violation_checks import SYNTHETIC_CHECKS, REAL_CHECKS_FAST, REAL_CHECKS_SLOW, ...
    for name, fn in SYNTHETIC_CHECKS:      fn(mod)
    for name, fn in REAL_CHECKS_FAST:      fn(mod, real_fast)
    for name, fn in REAL_CHECKS_SLOW:      fn(mod, real_all)

签名约定
  * `mod`：`env.violations` 模块对象（真实现，或由变异体源码加载出的同名模块）；
  * `real`：`RealData`（真实板 `PcbModel` 集合，只读）。
 ⇒ 变异测试只需把 `mod` 换成变异体，就能判断"这套检查能不能发现这个 bug"。

设计原则：**能失败**。断言都写明"期望 vs 实际"；微型模型（`mini_model`）覆盖全部 7 类 +
全部边界（相切 / 阈值端点 / 层隔离 / net 过滤 / 归属排除 / 自身不出现在 `others`）；
真实数据检查覆盖"§0.4 数字"与 probe83/85 已取证的实例。
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.kicad_utils.pcb_model import (  # noqa: E402
    LAYER_B,
    LAYER_F,
    BoardOutline,
    Footprint,
    KeepoutRegion,
    Pad,
    PcbModel,
    Via,
    rotate_aabb,
)

# ===================== 微型模型（不依赖 pcbnew，几何全部显式给出） =====================

# 板框取负起点：便于构造"A 的 AABB 恰好结束于 x=0、B 恰好起始于 x=0.2"的**精确**边界用例
MINI_BBOX = (-10.0, -10.0, 20.0, 20.0)
MINI_BOARD = BoardOutline(bbox=MINI_BBOX, outline_count=1, slots=())


def _union(boxes: Sequence[Tuple[float, float, float, float]]):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _pad(number: str, local_aabb, layers=(LAYER_F,), netcode: int = 0) -> Pad:
    x1, y1, x2, y2 = local_aabb
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return Pad(
        ref="", number=number, netcode=netcode,
        net_id=(f"N{netcode}" if netcode else None),
        layers=tuple(layers), aabb=local_aabb, center=(cx, cy), rel_offset=(cx, cy),
        size=(x2 - x1, y2 - y1), shape=0, angle_deg=0.0, drill=(0.0, 0.0),
        attribute=0, prop=0, local_aabb=local_aabb,
    )


def _fp(ref: str, pads, anchor=(0.0, 0.0), rot: int = 0, layer: int = LAYER_F) -> Footprint:
    pads = tuple(replace(p, ref=ref) for p in pads)
    hull = _union([p.local_aabb for p in pads])       # hull ⊇ pads ⇒ 粗筛必然安全
    return Footprint(
        ref=ref, fpid="mini", anchor=anchor, rotation_deg_raw=rot * 90.0,
        rotation_index=rot, layer=layer, flipped=False,
        hull_aabb=rotate_aabb(hull, rot * 90.0, anchor), pads=pads, local_hull_aabb=hull,
    )


def _via(center, netcode: int, layers=(LAYER_F, LAYER_B), diameter: float = 0.2) -> Via:
    cx, cy = center
    h = diameter / 2.0
    return Via(
        netcode=netcode, net_id=(f"N{netcode}" if netcode else None), layers=tuple(layers),
        center=center, diameter=diameter, drill=0.1, aabb=(cx - h, cy - h, cx + h, cy + h),
    )


def _keepout(aabb, owner_ref: Optional[str] = None, layers=(LAYER_F, LAYER_B)) -> KeepoutRegion:
    return KeepoutRegion(kind="mini_keepout", aabb=aabb, owner_ref=owner_ref, layers=tuple(layers))


def mini_model(footprints=(), vias=(), keepouts=()) -> PcbModel:
    return PcbModel(
        path="<mini>", footprints=tuple(footprints), vias=tuple(vias), nets={},
        board_outline=MINI_BOARD, keepouts=tuple(keepouts),
    )


# 隔离场景：把**真实几何**放进"无 via / 无禁布区 / 超大板框"的模型里，
# 从而把判定隔离到某一条规则（构造用例时排除环境噪声）
ISOLATED_BBOX = (-500.0, -500.0, 500.0, 500.0)


def isolated_model(footprints) -> PcbModel:
    return PcbModel(
        path="<isolated>", footprints=tuple(footprints), vias=(), nets={},
        board_outline=BoardOutline(bbox=ISOLATED_BBOX, outline_count=1, slots=()), keepouts=(),
    )


# ===================== 断言工具 =====================


def _expect(mod: Any, res: Any, reasons: Sequence[str], where: str,
            counts: Optional[Dict[str, int]] = None) -> None:
    """断言"违规方向集合"（有序元组）+ 可选的分类计数 + 总违规项标志。"""
    want = tuple(reasons)
    got = tuple(res.reasons)
    assert got == want, f"{where}: reasons 期望 {want}，实际 {got}"
    assert res.is_violation == bool(want), f"{where}: is_violation 期望 {bool(want)}，实际 {res.is_violation}"
    # 分层标志用**模块常量集合**推导（不用字符串前缀——前缀是展示命名约定，不是语义分类）
    hard_codes = {mod.HARD_PAD_PAD, mod.HARD_PAD_VIA, mod.HARD_PAD_KEEPOUT}
    soft_codes = {mod.SOFT_PAD_PAD, mod.SOFT_PAD_VIA, mod.SOFT_PAD_KEEPOUT}
    assert res.hard == bool(hard_codes & set(want)), f"{where}: hard 标志不符（{res.hard}）"
    assert res.soft == bool(soft_codes & set(want)), f"{where}: soft 标志不符（{res.soft}）"
    assert res.out_of_bounds == (mod.OUT_OF_BOUNDS in want), f"{where}: out_of_bounds 标志不符（{res.out_of_bounds}）"
    if counts is not None:
        assert dict(res.counts) == dict(counts), f"{where}: counts 期望 {dict(counts)}，实际 {dict(res.counts)}"


def _judge(mod: Any, model: PcbModel, ref: str, pose, others=None):
    j = mod.Judge(model)
    return j.judge(ref, pose, others or {}), j


# ===================== 合成检查（不依赖真实板，毫秒级） =====================


def check_clean_placement(mod: Any) -> None:
    """正确放置：两个相距 3mm 的同层件 ⇒ 无违规。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"B": (10.0, 5.0, 0)})
    _expect(mod, res, (), "clean placement（间距 3mm）", {})


def check_others_may_contain_self(mod: Any) -> None:
    """`others` 里出现自身时，不得自判违规（自身 pad 天然与自己重叠）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"A": (5.0, 5.0, 0)})
    _expect(mod, res, (), "others 含自身", {})


def check_same_footprint_pads_never_collide(mod: Any) -> None:
    """同封装内部 pad 重叠不算违规（硬条件"不同 footprint"）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0)), _pad("2", (1.0, 1.0, 3.0, 3.0))])
    model = mini_model([a])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {})
    _expect(mod, res, (), "同封装内部重叠", {})


def check_hard_pad_pad_on_overlap(mod: Any) -> None:
    """硬重叠 A：跨封装 ∧ 同层 ∧ 相叠 0.5mm ⇒ `hard.pad_pad`。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"B": (6.5, 5.0, 0)})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "硬重叠 A（0.5mm）", {mod.HARD_PAD_PAD: 1})


def check_hard_pad_pad_on_tiny_overlap(mod: Any) -> None:
    """硬重叠 A：**仅相叠 0.05mm** 也必须判硬重叠（不能漏小量重叠）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"B": (6.95, 5.0, 0)})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "硬重叠 A（0.05mm）", {mod.HARD_PAD_PAD: 1})


def check_hard_pad_pad_ignores_net(mod: Any) -> None:
    """硬重叠 A **不看 net**：同 net 焊盘相叠仍违规（堵"同 net 堆叠"漏洞）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=5)])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=5)])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"B": (6.5, 5.0, 0)})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "硬重叠 A（同 net）", {mod.HARD_PAD_PAD: 1})


def check_pad_pad_layer_isolation(mod: Any) -> None:
    """异层不比：F 面 pad 与 B 面 pad 重叠 ⇒ 无违规。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_F,))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_B,))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (5.0, 5.0, 0), {"B": (6.5, 5.0, 0)})
    _expect(mod, res, (), "异层 pad 重叠", {})


def check_pth_pad_meets_both_layers(mod: Any) -> None:
    """PTH pad（层 (F,B)）在**两面都**参与：与 F 面 pad、B 面 pad 相叠都算硬重叠。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_F, LAYER_B))])
    b_f = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_F,))])
    b_b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_B,))])
    for other, tag in ((b_f, "对 F 面"), (b_b, "对 B 面")):
        res, _ = _judge(mod, mini_model([a, other]), "A", (5.0, 5.0, 0), {"B": (6.5, 5.0, 0)})
        _expect(mod, res, (mod.HARD_PAD_PAD,), f"PTH {tag}", {mod.HARD_PAD_PAD: 1})


def check_tangent_is_soft_not_hard(mod: Any) -> None:
    """相切（gap == 0）**不算相交** ⇒ 只算软间距 S1。"""
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0))])          # AABB (-2,0,0,2)
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (0.0, 0.0, 0), {"B": (0.0, 0.0, 0)})  # B AABB (0,0,2,2)
    _expect(mod, res, (mod.SOFT_PAD_PAD,), "相切（gap=0）", {mod.SOFT_PAD_PAD: 1})


def check_soft_gap_boundary_is_exclusive(mod: Any) -> None:
    """软间距端点：`gap` **恰好 == 0.2** ⇒ 不违规（`0 ≤ gap < 0.2` 是半开区间）。

    注意：本用例的"被测 pad 对"间距正好落在粗筛边界上，若 hull 与 pad 等大，粗筛会先把这对
    滤掉 ⇒ 断言变成空转（`<` 与 `<=` 将无法区分）。故给 A 加一个**远处的 pad**把 hull 撑大，
    保证这对能通过粗筛、真正走到阈值比较那一步。
    """
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0)),          # x2 = 0.0（精确）
                  _pad("9", (-1.0, 10.0, 1.5, 11.0))])       # 撑大 hull（板内、远离被测 pad 对）
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (0.0, 0.0, 0), {"B": (mod.SOFT_GAP_MM, 0.0, 0)})
    _expect(mod, res, (), f"gap == {mod.SOFT_GAP_MM}（端点不违规）", {})


def check_soft_gap_just_inside(mod: Any) -> None:
    """软间距 S1：`gap == 0.125` ⇒ `soft.pad_pad`。"""
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (0.0, 0.0, 0), {"B": (0.125, 0.0, 0)})
    _expect(mod, res, (mod.SOFT_PAD_PAD,), "软间距 0.125", {mod.SOFT_PAD_PAD: 1})


def check_soft_gap_requires_same_layer(mod: Any) -> None:
    """异层的近距 pad 不算软间距。"""
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0), layers=(LAYER_F,))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_B,))])
    model = mini_model([a, b])
    res, _ = _judge(mod, model, "A", (0.0, 0.0, 0), {"B": (0.125, 0.0, 0)})
    _expect(mod, res, (), "异层软间距", {})


def check_via_hard_and_filters(mod: Any) -> None:
    """硬重叠 B 的完整条件式：同层 ∧ 相交 ∧ 双方 net≠0 ∧ net 不相等。"""
    via = _via((10.0, 10.0), netcode=7)                        # AABB (9.9,9.9,10.1,10.1)
    cases = [
        ((10.0, 10.0), 5, (LAYER_F,), (mod.HARD_PAD_VIA,), "异网相叠 ⇒ 违规"),
        ((10.0, 10.0), 7, (LAYER_F,), (), "同 net ⇒ 不违规（in-pad via 是常规做法）"),
        ((10.0, 10.0), 0, (LAYER_F,), (), "pad 无 net ⇒ 不违规（散热过孔阵列）"),
        ((10.0, 10.0), 5, (LAYER_B,), (mod.HARD_PAD_VIA,), "pad 在 B 面（via 两层）⇒ 违规"),
    ]
    for anchor, netcode, layers, want, why in cases:
        a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=layers, netcode=netcode)])
        res, _ = _judge(mod, mini_model([a], vias=[via]), "A", (anchor[0] - 1.0, anchor[1] - 1.0, 0), {})
        _expect(mod, res, want, f"pad↔via：{why}", {want[0]: 1} if want else {})
    # via 无 net ⇒ 不参与（即便 pad 有 net）
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=5)])
    res, _ = _judge(mod, mini_model([a], vias=[_via((10.0, 10.0), netcode=0)]), "A", (9.0, 9.0, 0), {})
    _expect(mod, res, (), "pad↔via：via 无 net ⇒ 不违规", {})
    # via 只在 F 面 ⇒ B 面 pad 不参与
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_B,), netcode=5)])
    res, _ = _judge(mod, mini_model([a], vias=[_via((10.0, 10.0), netcode=7, layers=(LAYER_F,))]),
                    "A", (9.0, 9.0, 0), {})
    _expect(mod, res, (), "pad↔via：via 不在 B 面 ⇒ 不违规", {})


def check_via_soft_gap(mod: Any) -> None:
    """软间距 S2：pad 与异网 via 相距 0.125mm ⇒ `soft.pad_via`；恰好 0.2mm ⇒ 不违规。"""
    via = _via((10.0, 10.0), netcode=7)                        # 左边界 x = 9.9
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=5)])  # 右边界在 anchor.x + 2
    model = mini_model([a], vias=[via])
    res, _ = _judge(mod, model, "A", (9.9 - 2.0 - 0.125, 9.0, 0), {})
    _expect(mod, res, (mod.SOFT_PAD_VIA,), "软间距 S2（0.125）", {mod.SOFT_PAD_VIA: 1})
    # 端点：via 右边界精确 = 0.0（直径 0.25 ⇒ 中心 −0.125，均为二进制精确值）、pad 左边界 = 0.2
    via0 = _via((-0.125, 0.0), netcode=7, diameter=0.25)
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=5)])
    res, _ = _judge(mod, mini_model([b], vias=[via0]), "B", (mod.SOFT_GAP_MM, 0.0, 0), {})
    _expect(mod, res, (), f"pad↔via gap == {mod.SOFT_GAP_MM}（端点不违规）", {})


def check_keepout_rules(mod: Any) -> None:
    """硬重叠 C / 软间距 S3：同层 ∧ 相交 / 0≤gap<0.2 ∧ **排除所属 footprint**。"""
    ko = _keepout((5.0, 5.0, 8.0, 8.0), owner_ref="OWNER")
    # 相交 ⇒ 硬重叠 C
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    res, _ = _judge(mod, mini_model([a], keepouts=[ko]), "A", (6.0, 6.0, 0), {})
    _expect(mod, res, (mod.HARD_PAD_KEEPOUT,), "禁布区相交", {mod.HARD_PAD_KEEPOUT: 1})
    # 所属 footprint 自己的 pad ⇒ 排除
    o = _fp("OWNER", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    res, _ = _judge(mod, mini_model([o], keepouts=[ko]), "OWNER", (6.0, 6.0, 0), {})
    _expect(mod, res, (), "禁布区所属 footprint 自己的 pad ⇒ 排除", {})
    # 相切（gap == 0）⇒ 只算软间距 S3
    res, _ = _judge(mod, mini_model([a], keepouts=[ko]), "A", (8.0, 6.0, 0), {})
    _expect(mod, res, (mod.SOFT_PAD_KEEPOUT,), "禁布区相切（gap=0）", {mod.SOFT_PAD_KEEPOUT: 1})
    # 0.1mm ⇒ 软间距 S3
    res, _ = _judge(mod, mini_model([a], keepouts=[ko]), "A", (8.1, 6.0, 0), {})
    _expect(mod, res, (mod.SOFT_PAD_KEEPOUT,), "禁布区软间距 0.1", {mod.SOFT_PAD_KEEPOUT: 1})
    # 恰好 == 0.2 ⇒ 不违规（"禁布区右边界 = 0.0、pad 左边界 = 0.2"避免浮点误差；
    # 另加一个远 pad 撑大 hull，使这对能通过粗筛 —— 否则断言会因粗筛而空转）
    a2 = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0)), _pad("9", (-1.0, 10.0, 1.5, 11.0))])
    ko0 = _keepout((-3.0, 5.0, 0.0, 8.0), owner_ref="OWNER")
    res, _ = _judge(mod, mini_model([a2], keepouts=[ko0]), "A", (mod.SOFT_GAP_MM, 6.0, 0), {})
    _expect(mod, res, (), f"禁布区 gap == {mod.SOFT_GAP_MM}", {})
    # 禁布区只在 F 面 ⇒ B 面 pad 不参与
    ko_f = _keepout((5.0, 5.0, 8.0, 8.0), owner_ref="OWNER", layers=(LAYER_F,))
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0), layers=(LAYER_B,))])
    res, _ = _judge(mod, mini_model([b], keepouts=[ko_f]), "B", (6.0, 6.0, 0), {})
    _expect(mod, res, (), "禁布区异层", {})


def check_out_of_bounds_rules(mod: Any) -> None:
    """出界：AABB 越出板框外接矩形即违规（不管 net）；**贴边（gap=0）不算**。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a])
    cases = [
        ((9.0, 9.0), (), "完全在板内"),
        ((18.0, 9.0), (), "右边缘齐平（x2 == 板右边）"),
        ((18.0 + 1e-3, 9.0), (mod.OUT_OF_BOUNDS,), "越界 1µm"),
        ((40.0, 9.0), (mod.OUT_OF_BOUNDS,), "完全在板外"),
        ((-12.0, 9.0), (mod.OUT_OF_BOUNDS,), "左上越界"),
    ]
    for anchor, want, why in cases:
        res, _ = _judge(mod, model, "A", (anchor[0], anchor[1], 0), {})
        _expect(mod, res, want, f"出界：{why}", {want[0]: 1} if want else {})
    # 无 net 的 pad 同样判出界
    a0 = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0), netcode=0)])
    res, _ = _judge(mod, mini_model([a0]), "A", (40.0, 9.0, 0), {})
    _expect(mod, res, (mod.OUT_OF_BOUNDS,), "出界：无 net pad", {mod.OUT_OF_BOUNDS: 1})


def check_three_layers_aggregate(mod: Any) -> None:
    """三层合并：一个位姿同时触发 4 个方向 ⇒ 方向按固定顺序去重、计数正确、标志正确。

    场景整体平移到板框外（T 的 AABB 越出左边界）⇒ 一次判定同时得到
    硬重叠 A（与 P）+ 软间距 S2（与 via）+ 软间距 S3（与禁布区）+ 出界。
    """
    t = _fp("T", [_pad("1", (-0.5, -0.5, 0.5, 0.5), netcode=5)])
    p = _fp("P", [_pad("1", (-0.5, -0.5, 0.5, 0.5), netcode=5)])
    via = _via((-12.7, 0.0), netcode=9, diameter=0.2)             # AABB (-12.8,-0.1,-12.6,0.1)
    ko = _keepout((-12.5, 0.6, -11.5, 1.0), owner_ref="OWNER")
    model = mini_model([t, p], vias=[via], keepouts=[ko])
    pose = (-12.0, 0.0, 0)                                        # T 的 AABB = (-12.5,-0.5,-11.5,0.5) ⇒ 越界
    res, _ = _judge(mod, model, "T", pose, {"P": (-12.8, 0.0, 0)})  # P 的 AABB = (-13.3,…,-12.3,…) ⇒ 相叠
    want = (mod.HARD_PAD_PAD, mod.SOFT_PAD_VIA, mod.SOFT_PAD_KEEPOUT, mod.OUT_OF_BOUNDS)
    _expect(mod, res, want, "三层合并（4 方向）", {r: 1 for r in want})
    assert res.describe() != "ok" and res.describe().count("×1") == 4, f"describe 异常：{res.describe()}"
    assert bool(res) is True, "总违规项应为 True"
    assert bool(mod.ViolationResult()) is False, "空结果应为 False"


def check_cache_is_pose_keyed(mod: Any) -> None:
    """同一 `ref` 在不同位姿上判定，结果必须各自独立（AABB 缓存键必须含位置与角度）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a])
    j = mod.Judge(model)
    assert j.blocks("A", (100.0, 5.0, 0), {}) is True, "板外应被判出界"
    assert j.blocks("A", (5.0, 5.0, 0), {}) is False, "板内不应被误判（缓存不得串味）"
    # 源姿态越界、转 90° 后回到板内 ⇒ 缓存键若漏掉位置或角度，后两次断言必然失败
    assert j.blocks("A", (18.0, 20.0, 0), {}) is True, "0° 时越出下边界"
    assert j.blocks("A", (18.0, 20.0, 1), {}) is False, "同位置转 90° 后回到板内"
    assert j.blocks("A", (18.0, 20.0, 0), {}) is True, "再次回到 0° 仍应越界"


def check_blocks_hot_path(mod: Any) -> None:
    """`blocks()` 必须与 `judge().is_violation` 一致（含**只有软间距**的情形）。"""
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    j = mod.Judge(model)
    assert j.blocks("A", (0.0, 0.0, 0), {"B": (0.125, 0.0, 0)}) is True, "只有软间距也应被拦"
    assert j.blocks("A", (0.0, 0.0, 0), {"B": (9.0, 0.0, 0)}) is False, "合法位姿不应被拦"


def check_judge_all_counts_pair_once(mod: Any) -> None:
    """整板判定：pad↔pad 是**对称对**，每一对只能计一次（双侧各算一次即错）。"""
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    j = mod.Judge(model)
    res = j.judge_all({"A": (5.0, 5.0, 0), "B": (6.5, 5.0, 0)})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "judge_all 配对去重", {mod.HARD_PAD_PAD: 1})


def check_margin_shrinks_valid_area(mod: Any) -> None:
    """`Judge(margin=...)` 必须真的内缩有效区（env 的 `BOARD_MARGIN` 依赖它，§4.1 板框）。

    probe87-B 取证：忽略 `margin` 的实现在原检查集下**存活** ⇒ 补此检查（+ 对应变异体）。
    """
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a])
    pose = (18.0, 9.0, 0)                        # pad AABB = (18,9,20,11)：右边界齐平板边
    assert mod.Judge(model, margin=0.0).blocks("A", pose, {}) is False, "margin=0 时贴边不应判出界"
    assert mod.Judge(model, margin=0.5).blocks("A", pose, {}) is True, "margin=0.5 时应判出界（内缩 0.5mm）"


def check_custom_soft_gap(mod: Any) -> None:
    """`Judge(soft_gap=...)` 必须真的改变软间距阈值（阈值口径见台账 §3.4 + §5.3④）。

    probe89-I 取证：忽略该参数的实现在原检查集下**存活** ⇒ 补此检查（+ 对应变异体）。
    """
    a = _fp("A", [_pad("1", (-2.0, 0.0, 0.0, 2.0)), _pad("9", (-1.0, 10.0, 1.5, 11.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])
    pa = (0.0, 0.0, 0)
    assert mod.Judge(model).blocks("A", pa, {"B": (0.15, 0.0, 0)}) is True, \
        "默认阈值 0.2：gap 0.15 应违规"
    assert mod.Judge(model, soft_gap=0.1).blocks("A", pa, {"B": (0.15, 0.0, 0)}) is False, \
        "阈值 0.1：gap 0.15 不应违规"
    assert mod.Judge(model, soft_gap=0.3).blocks("A", pa, {"B": (0.25, 0.0, 0)}) is True, \
        "阈值 0.3：gap 0.25 应违规"


def check_convenience_entry(mod: Any) -> None:
    """便捷入口 `judge_model()` 必须与 `Judge(...).judge_all(source_placements())` 等价。

    probe90-M 取证：把它掏空的实现在原检查集下**存活** ⇒ 补此检查（+ 对应变异体）。
    （该入口目前无生产消费方；要么像这里一样覆盖，要么删除，不能留"无人验证的公开 API"。）
    """
    a = _fp("A", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    b = _fp("B", [_pad("1", (0.0, 0.0, 2.0, 2.0))])
    model = mini_model([a, b])                   # 两件同位 ⇒ 必然硬重叠
    judge = mod.Judge(model)
    direct = judge.judge_all(judge.source_placements())
    entry = mod.judge_model(model)
    assert tuple(entry.reasons) == (mod.HARD_PAD_PAD,), \
        f"judge_model 应报硬重叠，实际 {entry.describe()}"
    assert dict(entry.counts) == dict(direct.counts), \
        f"judge_model 与 Judge.judge_all 计数不一致：{dict(entry.counts)} vs {dict(direct.counts)}"
    assert tuple(entry.reasons) == tuple(direct.reasons), \
        f"judge_model 与 Judge.judge_all 方向不一致：{entry.describe()} vs {direct.describe()}"


SYNTHETIC_CHECKS: List[Tuple[str, Callable[[Any], None]]] = [
    ("clean_placement", check_clean_placement),
    ("margin_shrinks_valid_area", check_margin_shrinks_valid_area),
    ("custom_soft_gap", check_custom_soft_gap),
    ("convenience_entry", check_convenience_entry),
    ("others_may_contain_self", check_others_may_contain_self),
    ("same_footprint_pads_never_collide", check_same_footprint_pads_never_collide),
    ("hard_pad_pad_on_overlap", check_hard_pad_pad_on_overlap),
    ("hard_pad_pad_on_tiny_overlap", check_hard_pad_pad_on_tiny_overlap),
    ("hard_pad_pad_ignores_net", check_hard_pad_pad_ignores_net),
    ("pad_pad_layer_isolation", check_pad_pad_layer_isolation),
    ("pth_pad_meets_both_layers", check_pth_pad_meets_both_layers),
    ("tangent_is_soft_not_hard", check_tangent_is_soft_not_hard),
    ("soft_gap_boundary_is_exclusive", check_soft_gap_boundary_is_exclusive),
    ("soft_gap_just_inside", check_soft_gap_just_inside),
    ("soft_gap_requires_same_layer", check_soft_gap_requires_same_layer),
    ("via_hard_and_filters", check_via_hard_and_filters),
    ("via_soft_gap", check_via_soft_gap),
    ("keepout_rules", check_keepout_rules),
    ("out_of_bounds_rules", check_out_of_bounds_rules),
    ("three_layers_aggregate", check_three_layers_aggregate),
    ("cache_is_pose_keyed", check_cache_is_pose_keyed),
    ("blocks_hot_path", check_blocks_hot_path),
    ("judge_all_counts_pair_once", check_judge_all_counts_pair_once),
]


# ===================== 真实数据 =====================

_BATCH_DIR = os.path.join(ROOT, "data", "kicad-native-batch")

# 快速子集：覆盖 7 类中的 A/B/C/S1/S2/出界 + "整板零违规" + 干净件
FAST_BOARD_KEYS = ("Mini控制板切分", "esp32s3r8n8", "lyg0927", "pid-jian-yi", "elisaday")


class RealData:
    """真实板 `PcbModel` 集合（只读）。"""

    def __init__(self, entries: Sequence[Tuple[str, PcbModel]]):
        self.entries = list(entries)

    def get(self, keyword: str) -> PcbModel:
        for tag, model in self.entries:
            if keyword in tag:
                return model
        raise AssertionError(f"未找到板（关键字 {keyword!r}）；现有：{[t[:40] for t, _ in self.entries]}")

    def models(self) -> List[PcbModel]:
        return [m for _, m in self.entries]


def _load_real(keywords: Optional[Sequence[str]] = None) -> RealData:
    from utils.kicad_utils import pcb_model  # 触发 bootstrap（DLL 路径 + wx 抑制）

    paths = sorted(glob.glob(os.path.join(_BATCH_DIR, "*", "*.kicad_pcb")))
    paths = [p for p in paths if not p.endswith("best_layout.kicad_pcb")]
    assert paths, f"未找到真实板：{_BATCH_DIR}"
    entries = []
    for path in paths:
        tag = os.path.basename(os.path.dirname(path))
        if keywords is not None and not any(k in tag for k in keywords):
            continue
        board = pcb_model.pcbnew.LoadBoard(path)
        entries.append((tag, pcb_model.build_model(board, path)))
    return RealData(entries)


@lru_cache(maxsize=None)
def real_data_fast() -> RealData:
    """快速子集（5 块板）。"""
    return _load_real(FAST_BOARD_KEYS)


@lru_cache(maxsize=None)
def real_data_all() -> RealData:
    """全部真实板（20 块，用于 §0.4 全量回归）。"""
    return _load_real(None)


def _judge_model(mod: Any, model: PcbModel):
    j = mod.Judge(model)
    return j, j.source_placements()


# ---- 快速真实检查 ----

REAL_FAST_EXPECT: Dict[str, Dict[str, int]] = {
    "Mini控制板切分": {"hard.pad_pad": 2, "hard.pad_via": 8, "soft.pad_via": 21},
    "esp32s3r8n8": {"hard.pad_via": 20, "soft.pad_pad": 3, "soft.pad_via": 28, "out_of_bounds": 60},
    "lyg0927": {"hard.pad_keepout": 4, "soft.pad_via": 3},
    "elisaday": {"soft.pad_via": 2},
}


def check_real_board_numbers(mod: Any, real: RealData) -> None:
    """真实板整板 `judge_all(源布局)` 的逐类计数（probe85 实测，锁定回归）。"""
    for key, want in REAL_FAST_EXPECT.items():
        judge, placements = _judge_model(mod, real.get(key))
        res = judge.judge_all(placements)
        assert dict(res.counts) == want, f"{key}: judge_all 期望 {want}，实际 {dict(res.counts)}"
        assert res.is_violation is True


def check_real_clean_board(mod: Any, real: RealData) -> None:
    """PID 板源布局**零违规**（含 TT马达 hull 禁布区与各封装自带开槽）。"""
    judge, placements = _judge_model(mod, real.get("pid-jian-yi"))
    res = judge.judge_all(placements)
    assert not res.is_violation, f"PID 板源布局应为零违规，实际 {res.describe()}"


def check_real_single_instances(mod: Any, real: RealData) -> None:
    """单件判定的**精确计数**（probe83/85 已取证的实例）。"""
    # 硬重叠 A：Mini控制板 C10 ↔ U14（gap −0.2225）
    model = real.get("Mini控制板切分")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("C10", pl["C10"], {"U14": pl["U14"]})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "真实 A: C10×U14", {mod.HARD_PAD_PAD: 1})
    # 硬重叠 A：Mini控制板 D3 ↔ D2（gap −0.2970）
    res = judge.judge("D3", pl["D3"], {"D2": pl["D2"]})
    _expect(mod, res, (mod.HARD_PAD_PAD,), "真实 A: D3×D2", {mod.HARD_PAD_PAD: 1})
    # 硬重叠 B + 软间距 S2：esp32s3r8n8 U3（不与其它件比，隔离静态对象）
    model = real.get("esp32s3r8n8")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("U3", pl["U3"], {})
    _expect(mod, res, (mod.HARD_PAD_VIA, mod.SOFT_PAD_VIA), "真实 B/S2: esp32 U3",
            {mod.HARD_PAD_VIA: 19, mod.SOFT_PAD_VIA: 15})
    # 硬重叠 B：Mini控制板 SW1（自家开槽被正确排除，只剩过孔）
    model = real.get("Mini控制板切分")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("SW1", pl["SW1"], {})
    _expect(mod, res, (mod.HARD_PAD_VIA, mod.SOFT_PAD_VIA), "真实 B/S2: SW1",
            {mod.HARD_PAD_VIA: 7, mod.SOFT_PAD_VIA: 5})
    # 硬重叠 C：lyg0927 U1 压 RST0/BOOT0 开槽
    model = real.get("lyg0927")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("U1", pl["U1"], {})
    _expect(mod, res, (mod.HARD_PAD_KEEPOUT, mod.SOFT_PAD_VIA), "真实 C: lyg0927 U1",
            {mod.HARD_PAD_KEEPOUT: 4, mod.SOFT_PAD_VIA: 2})
    # 出界：esp32 J1 全部 20 个 pad 越界（该件同时受静态过孔影响 ⇒ 只断言出界部分）
    model = real.get("esp32s3r8n8")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("J1", pl["J1"], {})
    assert mod.OUT_OF_BOUNDS in res.reasons, f"esp32 J1 应出界，实际 {res.describe()}"
    assert res.counts[mod.OUT_OF_BOUNDS] == 20, \
        f"esp32 J1 越界 pad 数应为 20，实际 {res.counts.get(mod.OUT_OF_BOUNDS)}"


def check_real_soft_s1_instance(mod: Any, real: RealData) -> None:
    """软间距 S1：esp32 C54 ↔ L6（真实既存 0.030mm 窄缝）。"""
    model = real.get("esp32s3r8n8")
    judge, pl = _judge_model(mod, model)
    res = judge.judge("C54", pl["C54"], {"L6": pl["L6"]})
    _expect(mod, res, (mod.SOFT_PAD_PAD,), "真实 S1: C54×L6", {mod.SOFT_PAD_PAD: 1})


def check_real_clean_footprints(mod: Any, real: RealData) -> None:
    """elisa 板三个干净件（probe85 实测 U15/H3/R23）在**全局快照**下无违规。"""
    model = real.get("elisaday")
    judge, pl = _judge_model(mod, model)
    for ref in ("U15", "H3", "R23"):
        others = {r: p for r, p in pl.items() if r != ref}
        res = judge.judge(ref, pl[ref], others)
        assert not res.is_violation, f"elisaday {ref} 应为干净，实际 {res.describe()}"


def check_real_constructed_oob(mod: Any, real: RealData) -> None:
    """构造（真实几何）：把某个件整体挪到板外 100mm ⇒ 只有出界。"""
    model = real.get("elisaday")
    judge, pl = _judge_model(mod, model)
    ref = next(iter(pl))
    p = pl[ref]
    res = judge.judge(ref, (p.x + 100.0, p.y + 100.0, p.rot), {})
    _expect(mod, res, (mod.OUT_OF_BOUNDS,), "构造: 挪到板外", None)


def _find_pair_fixture(real: RealData):
    """找一对"同封装名 ∧ 同层"的真实件（用于构造重叠/间距）。**优先 pad 最少的封装**。

    为什么优先 pad 少：本文件用"两件错开一个 pad 高度 + 0.1mm"构造软间距，
    pad 越少 ⇒ 需要排除的"交叉 pad 对"越少 ⇒ 断言越唯一（1-pad 封装时只有一对）。
    """
    best = None
    for _tag, model in real.entries:
        groups: Dict[Tuple[str, int], List[Footprint]] = {}
        for fp in model.footprints:
            groups.setdefault((fp.fpid, fp.layer), []).append(fp)
        for _key, fps in sorted(groups.items()):
            if len(fps) < 2:
                continue
            if len(fps[0].pads) == 1:
                return model, fps[0], fps[1]
            if best is None or len(fps[0].pads) < len(best[1].pads):
                best = (model, fps[0], fps[1])
    assert best is not None, "未找到同封装同层的真实件对"
    return best


def check_real_constructed_hard_a(mod: Any, real: RealData) -> None:
    """构造（真实几何，隔离场景）：同封装同层两件放在**同一位置** ⇒ 必然硬重叠。"""
    _model, a, b = _find_pair_fixture(real)
    judge = mod.Judge(isolated_model([a, b]))
    pose_a = (a.anchor[0], a.anchor[1], a.rotation_index)
    res = judge.judge(b.ref, pose_a, {a.ref: pose_a})
    assert tuple(res.reasons) == (mod.HARD_PAD_PAD,), \
        f"同封装同层两件同位应只有硬重叠，实际 {res.describe()}（{a.ref}/{b.ref} @ {a.fpid}）"


def check_real_constructed_soft_pair(mod: Any, real: RealData) -> None:
    """构造（真实几何，隔离场景）：同封装同层两件沿 y 错开"pad 高 + 0.1mm" ⇒ 只有软间距 S1。"""
    _model, a, b = _find_pair_fixture(real)
    height = max(p.size[1] for p in a.pads)
    pose_a = (a.anchor[0], a.anchor[1], a.rotation_index)
    pose_b = (pose_a[0], pose_a[1] + height + 0.1, pose_a[2])
    judge = mod.Judge(isolated_model([a, b]))
    res = judge.judge(b.ref, pose_b, {a.ref: pose_a})
    assert tuple(res.reasons) == (mod.SOFT_PAD_PAD,), \
        (f"错开 0.1mm 应只有软间距，实际 {res.describe()}"
         f"（{a.ref}/{b.ref} @ {a.fpid}，pad 高 {height}mm）")


def check_real_constructed_soft_keepout(mod: Any, real: RealData) -> None:
    """构造（真实几何）：把某个件贴到**真实禁布区**旁 0.1mm ⇒ 软间距 S3（源数据里 S3 = 0 例）。"""
    model = real.get("pid-jian-yi")
    judge, pl = _judge_model(mod, model)
    ko = max(model.keepouts, key=lambda k: (k.aabb[2] - k.aabb[0]) * (k.aabb[3] - k.aabb[1]))
    assert ko.owner_ref not in pl, "禁布区归属件不应在 footprint 集合里（零 pad 对象）"
    ref = next(r for r in sorted(pl) if r != ko.owner_ref)
    fp = next(f for f in model.footprints if f.ref == ref)
    pad = min(fp.pads, key=lambda p: p.local_aabb[0])       # 取最左 pad：其余 pad 落在更右（远离禁布区）
    lx1, ly1, lx2, ly2 = pad.local_aabb
    cx, cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
    hx, hy = (fp.local_hull_aabb[0] + fp.local_hull_aabb[2]) / 2.0, \
             (fp.local_hull_aabb[1] + fp.local_hull_aabb[3]) / 2.0
    kx, ky = (ko.aabb[0] + ko.aabb[2]) / 2.0, (ko.aabb[1] + ko.aabb[3]) / 2.0
    # 四个方向各构造"该 pad 与禁布区在单轴上相距 0.1mm、另一轴对齐"
    sides = [
        ("pad 在禁布区右侧", (ko.aabb[2] + 0.1 - lx1, ky - cy)),
        ("pad 在禁布区左侧", (ko.aabb[0] - 0.1 - lx2, ky - cy)),
        ("pad 在禁布区下方", (kx - cx, ko.aabb[3] + 0.1 - ly1)),
        ("pad 在禁布区上方", (kx - cx, ko.aabb[1] - 0.1 - ly2)),
    ]
    hits = []
    for side, (ax, ay) in sides:
        for rot in (0, 1, 2, 3):
            res = judge.judge(ref, (ax, ay, rot), {})
            if tuple(res.reasons) == (mod.SOFT_PAD_KEEPOUT,):
                hits.append((side, rot, dict(res.counts)))
    assert hits, f"未能在真实禁布区旁构造出「只有软间距 S3」的位姿（{ref} @ {ko.kind}）"
    for side, rot, counts in hits:
        assert counts == {mod.SOFT_PAD_KEEPOUT: 1}, f"{side}/rot{rot}: 期望 1 例 S3，实际 {counts}"


REAL_CHECKS_FAST: List[Tuple[str, Callable[[Any, RealData], None]]] = [
    ("real_board_numbers", check_real_board_numbers),
    ("real_clean_board", check_real_clean_board),
    ("real_single_instances", check_real_single_instances),
    ("real_soft_s1_instance", check_real_soft_s1_instance),
    ("real_clean_footprints", check_real_clean_footprints),
    ("real_constructed_oob", check_real_constructed_oob),
    ("real_constructed_hard_a", check_real_constructed_hard_a),
    ("real_constructed_soft_pair", check_real_constructed_soft_pair),
    ("real_constructed_soft_keepout", check_real_constructed_soft_keepout),
]


# ---- 全量真实检查（慢：20 板） ----

DOC_EXPECT_ALL: Dict[str, int] = {
    "hard.pad_pad": 4,
    "hard.pad_via": 36,
    "hard.pad_keepout": 4,
    "soft.pad_pad": 5,
    "soft.pad_via": 133,
    "soft.pad_keepout": 0,
    "out_of_bounds": 160,
}


def check_real_all_boards_match_doc(mod: Any, real: RealData) -> None:
    """§0.4 全量回归：20 板源布局的 7 类计数必须与台账一致。"""
    counts: Dict[str, int] = {}
    clean = dirty = 0
    for model in real.models():
        judge = mod.Judge(model)
        pl = judge.source_placements()
        res = judge.judge_all(pl)
        for r, c in res.counts.items():
            counts[r] = counts.get(r, 0) + c
        for ref, pose in pl.items():
            others = {r: p for r, p in pl.items() if r != ref}
            if judge.judge(ref, pose, others).is_violation:
                dirty += 1
            else:
                clean += 1
    got = {r: counts.get(r, 0) for r in DOC_EXPECT_ALL}
    assert got == DOC_EXPECT_ALL, f"§0.4 不符：期望 {DOC_EXPECT_ALL}，实际 {got}"
    assert (clean, dirty) == (777, 80), f"单件 clean/dirty 期望 (777, 80)，实际 {(clean, dirty)}"


REAL_CHECKS_SLOW: List[Tuple[str, Callable[[Any, RealData], None]]] = [
    ("real_all_boards_match_doc", check_real_all_boards_match_doc),
]


# ===================== 运行器（供变异测试复用） =====================


class RunReport(NamedTuple):
    """一次检查运行的报告。

    Attributes:
        killed: **断言失败**清单（该变异体被这组检查"发现"）。
        errors: **非断言异常**清单（接口被破坏 / 检查自身出错）—— **不得计入"被杀"**，
                否则"把实现打崩"的变异体会被误判为"被检查集发现"（probe87-C 取证）。
    """

    killed: List[str]
    errors: List[str]

    def __bool__(self) -> bool:
        return bool(self.killed)

    @property
    def ok(self) -> bool:
        return not self.killed and not self.errors


def run_checks(mod: Any, groups: Sequence[str] = ("synthetic", "real_fast", "real_slow")
               ) -> RunReport:
    """跑指定检查组，返回 `RunReport`（`killed` = 断言失败，`errors` = 其它异常）。

    每条记录为 `"<组>::<检查名>: <信息>"`，便于变异测试报告"被哪个检查杀掉"。
    """
    killed: List[str] = []
    errors: List[str] = []

    def _run(group: str, name: str, fn, *args) -> None:
        try:
            fn(*args)
        except AssertionError as e:
            killed.append(f"{group}::{name}: {e}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{group}::{name}: {type(e).__name__}: {e}")

    if "synthetic" in groups:
        for name, fn in SYNTHETIC_CHECKS:
            _run("synthetic", name, fn, mod)
    for group, checks, data in (("real_fast", REAL_CHECKS_FAST, real_data_fast),
                                ("real_slow", REAL_CHECKS_SLOW, real_data_all)):
        if group not in groups:
            continue
        real = data()
        for name, fn in checks:
            _run(group, name, fn, mod, real)
    return RunReport(killed=killed, errors=errors)
