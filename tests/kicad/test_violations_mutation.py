# -*- coding: utf-8 -*-
"""违规判定**变异测试**：往 `env/violations.py` 注入**语义变异**，验证测试集一定能发现它。

思路（"专门挑刺"）
  * 变异体 = 把真实现源码里的**一处语义**改坏（改 `<` 为 `<=`、删掉层过滤、去掉 net 豁免、
    把方向码写错……），写到临时文件后以独立模块名加载（**不改动仓库里的源文件**）；
  * 用与正向测试**完全相同**的检查集（`violation_checks.run_checks`）去跑变异体；
  * **只要有任一检查失败 ⇒ 变异体被杀（killed）**；全部通过 ⇒ **存活（survived）= 测试有洞**。
  ⇒ 这套测试保证："规则被改坏"不会静默通过。

为什么不用 mutmut/cosmic-ray：本项目需要"变异体目录"精确对应台账的判定式条款，
且检查里含真实板数据（外部依赖），自建 harness 更可控、无额外依赖。

注意：**等价变异**（只动优化、不改语义，如删掉 hull 粗筛）不在目录内 —— 它理应存活，
放进目录会制造假警报。

运行：
    $env:KICAD_PATH="E:\\Kicad"; python -m pytest tests/kicad/test_violations_mutation.py -v
"""
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, NamedTuple

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import env.violations as violations  # noqa: E402
from violation_checks import run_checks  # noqa: E402

SRC_PATH = Path(ROOT) / "env" / "violations.py"

# 变异体使用的检查组：合成（毫秒级）+ 真实板快速子集（1s 级）⇒ 足以杀掉目录内全部变异
CHECK_GROUPS = ("synthetic", "real_fast")


class Mutant(NamedTuple):
    id: str          # 变异体标识（pytest 用例 id）
    desc: str        # 变异内容（人话）
    old: str         # 锚点片段（必须**唯一**命中真实现源码）
    new: str         # 替换后的（错误）实现
    kills_by: str    # 预期由哪个检查杀掉（自证这套测试覆盖到该条款）


# 锚点片段全部取自台账判定式所在行：改这些行 = 改规则
MUTANTS: List[Mutant] = [
    Mutant(
        "hard_a_tangent", "硬重叠 A：把「相切」也算相交（`< 0.0` → `<= 0.0`）",
        "                    if g < 0.0:\n                        if hit(HARD_PAD_PAD):",
        "                    if g <= 0.0:\n                        if hit(HARD_PAD_PAD):",
        "tangent_is_soft_not_hard",
    ),
    Mutant(
        "hard_a_miss_tiny_overlap", "硬重叠 A：漏掉 <0.1mm 的相叠（`< 0.0` → `< -0.1`）",
        "                    if g < 0.0:\n                        if hit(HARD_PAD_PAD):",
        "                    if g < -0.1:\n                        if hit(HARD_PAD_PAD):",
        "hard_pad_pad_on_tiny_overlap",
    ),
    Mutant(
        "others_self_not_skipped", "未排除 `others` 里的自身 ⇒ 自身 pad 自判违规",
        "        for oref, o_pose in others.items():\n            if oref == ref:\n                continue",
        "        for oref, o_pose in others.items():\n            if False:\n                continue",
        "others_may_contain_self",
    ),
    Mutant(
        "pad_pad_layer_ignored", "pad↔pad 不做同层匹配（异层也判）",
        "                    if not (set(layers) & set(olayers)):\n                        continue             # 异层不比",
        "                    if False:\n                        continue             # 异层不比",
        "pad_pad_layer_isolation / pth_pad_meets_both_layers",
    ),
    Mutant(
        "soft_pad_pad_closed_interval", "软间距 S1：端点 0.2 也算违规（`<` → `<=`）",
        "                    elif g < self.soft_gap:\n                        if hit(SOFT_PAD_PAD):",
        "                    elif g <= self.soft_gap:\n                        if hit(SOFT_PAD_PAD):",
        "soft_gap_boundary_is_exclusive",
    ),
    Mutant(
        "soft_pad_via_closed_interval", "软间距 S2：端点 0.2 也算违规（`<` → `<=`）",
        "                elif g < self.soft_gap:\n                    if hit(SOFT_PAD_VIA):",
        "                elif g <= self.soft_gap:\n                    if hit(SOFT_PAD_VIA):",
        "via_soft_gap（gap == 0.2 端点用例）",
    ),
    Mutant(
        "soft_keepout_closed_interval", "软间距 S3：端点 0.2 也算违规（`<` → `<=`）",
        "                elif g < self.soft_gap:\n                    if hit(SOFT_PAD_KEEPOUT):",
        "                elif g <= self.soft_gap:\n                    if hit(SOFT_PAD_KEEPOUT):",
        "keepout_rules（gap == 0.2 端点用例）",
    ),
    Mutant(
        "soft_gap_halved", "默认软间距减半（0.2 → 0.1）",
        ("    def __init__(self, model: PcbModel, soft_gap: float = SOFT_GAP_MM, margin: float = 0.0,\n"
         "                 keepouts: Optional[Sequence[KeepoutRegion]] = None,\n"
         "                 placed: Optional[Sequence[Footprint]] = None):"),
        ("    def __init__(self, model: PcbModel, soft_gap: float = SOFT_GAP_MM / 2.0, margin: float = 0.0,\n"
         "                 keepouts: Optional[Sequence[KeepoutRegion]] = None,\n"
         "                 placed: Optional[Sequence[Footprint]] = None):"),
        "soft_gap_just_inside（0.125mm）",
    ),
    Mutant(
        "via_same_net_not_exempt", "pad↔via 未豁免同 net（in-pad via 全变违规）",
        "                if not v.has_net or v.netcode == _pad.netcode:",
        "                if not v.has_net:",
        "via_hard_and_filters（同 net 用例）",
    ),
    Mutant(
        "via_net0_counted", "pad↔via 未要求 via 有 net",
        "                if not v.has_net or v.netcode == _pad.netcode:",
        "                if v.netcode == _pad.netcode:",
        "via_hard_and_filters（via 无 net 用例）",
    ),
    Mutant(
        "pad_net0_counted", "pad↔via 未要求 pad 有 net（散热过孔阵列会变违规）",
        "        for _pad, aabb, layers in cand:\n            if not _pad.has_net:\n                continue",
        "        for _pad, aabb, layers in cand:\n            if False:\n                continue",
        "via_hard_and_filters（pad 无 net 用例）",
    ),
    Mutant(
        "via_layer_ignored", "pad↔via 不做同层匹配",
        "                if not (set(layers) & set(v.layers)):\n                    continue",
        "                if False:\n                    continue",
        "via_hard_and_filters（via 只在 F 面用例）",
    ),
    Mutant(
        "keepout_owner_not_excluded", "禁布区未排除所属 footprint 自己的 pad",
        "        for k in self._keepouts:\n            if k.owner_ref == ref:\n                continue",
        "        for k in self._keepouts:\n            if False:\n                continue",
        "keepout_rules（归属排除）/ 真实 SW1",
    ),
    Mutant(
        "keepout_tangent_hard", "禁布区相切算硬重叠 C（`<` → `<=`）",
        "                if g < 0.0:\n                    if hit(HARD_PAD_KEEPOUT):",
        "                if g <= 0.0:\n                    if hit(HARD_PAD_KEEPOUT):",
        "keepout_rules（相切 ⇒ 只算 S3）",
    ),
    Mutant(
        "keepout_layer_ignored", "禁布区不做同层匹配（F-only 禁布区也拦 B 面 pad）",
        "                if not (set(layers) & set(k.layers)):\n                    continue",
        "                if False:\n                    continue",
        "keepout_rules（异层）",
    ),
    Mutant(
        "oob_closed_boundary", "出界端点取闭区间（贴边也算越界）",
        "        return aabb[0] < ax1 or aabb[1] < ay1 or aabb[2] > ax2 or aabb[3] > ay2",
        "        return aabb[0] < ax1 or aabb[1] < ay1 or aabb[2] >= ax2 or aabb[3] > ay2",
        "out_of_bounds_rules（贴边）",
    ),
    Mutant(
        "oob_not_checked", "完全不判出界",
        "            if self._is_out(aabb) and hit(OUT_OF_BOUNDS):",
        "            if False and hit(OUT_OF_BOUNDS):",
        "out_of_bounds_rules / 真实 J1",
    ),
    Mutant(
        "judge_all_double_count", "整板判定双侧重复计数（同一对算两次）",
        "            others = {r: p for r, p in placements.items() if r > ref}",
        "            others = {r: p for r, p in placements.items() if r != ref}",
        "judge_all_counts_pair_once / 真实板逐类计数",
    ),
    Mutant(
        "reason_label_swap", "方向码写错（`hard.pad_pad` → `soft.pad_pad`）",
        'HARD_PAD_PAD = "hard.pad_pad"            # 硬重叠 A',
        'HARD_PAD_PAD = "soft.pad_pad"            # 硬重叠 A',
        "全部精确方向断言",
    ),
    Mutant(
        "reason_order_shuffled", "`ALL_REASONS` 顺序错乱（汇总顺序失去意义）",
        "ALL_REASONS: Tuple[str, ...] = (\n    HARD_PAD_PAD, HARD_PAD_VIA, HARD_PAD_KEEPOUT,\n"
        "    SOFT_PAD_PAD, SOFT_PAD_VIA, SOFT_PAD_KEEPOUT,\n    OUT_OF_BOUNDS,\n)",
        "ALL_REASONS: Tuple[str, ...] = (\n    OUT_OF_BOUNDS,\n"
        "    SOFT_PAD_KEEPOUT, SOFT_PAD_VIA, SOFT_PAD_PAD,\n"
        "    HARD_PAD_KEEPOUT, HARD_PAD_VIA, HARD_PAD_PAD,\n)",
        "全部精确方向断言",
    ),
    Mutant(
        "hard_flag_swallows_soft", "`hard` 标志把软间距也算进来",
        '    def hard(self) -> bool:\n        """是否含硬重叠（A/B/C）。"""\n'
        "        return any(r in _HARD_REASONS for r in self.reasons)",
        '    def hard(self) -> bool:\n        """是否含硬重叠（A/B/C）。"""\n'
        "        return bool(self.reasons)",
        "soft-only 用例的 hard 标志",
    ),
    Mutant(
        "is_violation_hard_only", "总违规项只认硬重叠（软间距/出界不算）",
        "        return bool(self.reasons)",
        "        return any(r in _HARD_REASONS for r in self.reasons)",
        "软间距 / 出界用例",
    ),
    Mutant(
        "blocks_ignores_soft", "`blocks()` 只拦硬重叠",
        "        return self.judge(ref, pose, others, _short_circuit=True).is_violation",
        "        return self.judge(ref, pose, others, _short_circuit=True).hard",
        "blocks_hot_path",
    ),
    Mutant(
        "gap_sign_flip", "`gap()` 符号翻转（相交/分离判反）",
        "    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])",
        "    return max(b[2] - a[0], a[2] - b[0], b[3] - a[1], a[3] - b[1])",
        "全部几何用例",
    ),
    Mutant(
        "pad_layer_from_footprint", "pad 层改用元件面（丢弃 pad 独立记录的层）",
        "        out = tuple(\n            (pad, rotate_aabb(pad.local_aabb, rot * 90.0, (x, y)), pad.layers) "
        "for pad in fp.pads\n        )",
        "        out = tuple(\n            (pad, rotate_aabb(pad.local_aabb, rot * 90.0, (x, y)), (fp.layer,)) "
        "for pad in fp.pads\n        )",
        "pth_pad_meets_both_layers / 异层用例",
    ),
    Mutant(
        "cache_key_drops_position", "AABB 缓存键漏掉位置 ⇒ 位姿之间串味",
        "        key = (ref, x, y, rot)",
        "        key = (ref, rot)",
        "cache_is_pose_keyed",
    ),
    Mutant(
        "source_placements_rot_zero", "源布局忽略导入角度（全部按 0°）",
        "        return {f.ref: pose_of(f) for f in self.model.footprints}",
        "        return {f.ref: Pose(f.anchor[0], f.anchor[1], 0) for f in self.model.footprints}",
        "真实板逐类计数（旋转敏感）",
    ),
    # ---- 参数 / 入口类（probe87-B、probe89-I、probe90-M 取证：原检查集对这类**全盲**）----
    Mutant(
        "margin_ignored", "`Judge` 忽略 `margin`（板框不内缩）",
        "        self.valid_area: AABB = model.board_outline.valid_area(margin)",
        "        self.valid_area: AABB = model.board_outline.valid_area(0.0)",
        "margin_shrinks_valid_area",
    ),
    Mutant(
        "soft_gap_param_ignored", "`Judge` 忽略 `soft_gap` 构造参数（永远用默认 0.2）",
        "        self.soft_gap = soft_gap",
        "        self.soft_gap = SOFT_GAP_MM",
        "custom_soft_gap",
    ),
    Mutant(
        "convenience_entry_neutered", "便捷入口 `judge_model()` 掏空（永远返回「无违规」）",
        "    judge = Judge(model, soft_gap=soft_gap, margin=margin)\n"
        "    return judge.judge_all(judge.source_placements())",
        "    return ViolationResult()",
        "convenience_entry",
    ),
]


def _require_kicad() -> None:
    if not os.environ.get("KICAD_PATH") or not os.path.isdir(os.environ["KICAD_PATH"]):
        pytest.skip("KICAD_PATH not set or does not exist")
    pytest.importorskip("pcbnew", reason="需要 KiCad 环境（KICAD_PATH）")


def _load_mutant(mutant: Mutant, tmp_path):
    """把变异后的源码写到临时文件，并以独立模块名加载（**不动仓库源文件**）。"""
    src = SRC_PATH.read_text(encoding="utf-8")
    hits = src.count(mutant.old)
    assert hits == 1, (f"变异体 {mutant.id} 的锚点片段命中 {hits} 次（应为 1）"
                       f"——实现改动后需同步维护变异体目录")
    name = f"violations_mutant_{mutant.id}"
    path = Path(tmp_path) / f"{name}.py"
    path.write_text(src.replace(mutant.old, mutant.new, 1), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"变异体 {mutant.id} 无法加载（{type(e).__name__}: {e}）") from e
    return mod


def test_mutant_catalog_is_applicable():
    """变异体目录必须**逐条**命中当前源码（否则目录腐化，变异测试名存实亡）。"""
    src = SRC_PATH.read_text(encoding="utf-8")
    bad = [(m.id, src.count(m.old)) for m in MUTANTS if src.count(m.old) != 1]
    assert not bad, f"这些变异体的锚点片段不唯一/不存在（id, 命中次数）：{bad}"


@pytest.mark.require_kicad
def test_baseline_passes_fast_checks():
    """基线自检：**未变异**的实现必须通过全部检查（否则"被杀"的结论不可信）。"""
    _require_kicad()
    report = run_checks(violations, CHECK_GROUPS)
    assert not report.errors, "基线运行出现非断言异常（检查自身有问题）：\n" + "\n".join(report.errors)
    assert not report.killed, "基线未通过检查：\n" + "\n".join(report.killed)


@pytest.mark.require_kicad
@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.id for m in MUTANTS])
def test_mutant_is_killed(mutant, tmp_path):
    """每个变异体都必须被检查集**以断言失败**发现（异常不算"被杀"）。"""
    _require_kicad()
    mod = _load_mutant(mutant, tmp_path)
    report = run_checks(mod, CHECK_GROUPS)
    assert not report.errors, (
        f"变异体 {mutant.id} 让检查集直接抛异常 ⇒ 无法判定是否被杀（不得计入击杀）：\n"
        + "\n".join(report.errors[:5])
    )
    assert report.killed, (
        f"⚠️ 变异体存活：{mutant.id}\n"
        f"    变异内容：{mutant.desc}\n"
        f"    预期被杀检查：{mutant.kills_by}\n"
        f"    ⇒ 检查集全部通过，说明该条款**未被任何测试覆盖**，请补检查"
    )
