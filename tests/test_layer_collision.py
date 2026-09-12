"""层感知碰撞（Config.LAYER_AWARE_COLLISION）的四组用例。

背景
----
旧行为：所有元件的 XY 投影互斥，不看层 —— 底面被当成"用不了"，
可放置面积白白浪费一半。但**不能**改成"层不同就 continue"：

    if component.layer != other.layer:
        if not component.has_pth and not other.has_pth:
            continue

同一条 XY 上叠两个元件，只在**两面都是纯 SMD** 时才真的不冲突。
通孔封装（排针/连接器/插件件，本数据集大量存在）的孔贯穿所有层，
叠上去就是物理干涉；安装孔(NPTH)同理。keepout 禁布区与板框出界
也与层无关，不随层豁免。

判定收敛在 PCBUtils.layers_allow_xy_overlap 一处，
env._detect_collisions 与 ExpertTrajectoryGenerator._is_legal_placement 共用，
两处标准必须一致（否则专家会产出 env 判违规的布局）。

四组用例
--------
1. 异面纯 SMD：允许 XY 重叠（开关开）
2. 任一方含通孔：仍保守（开关开也不能豁免）
3. 同面：行为与开关无关（不回归）
4. 开关关：完全等价旧行为（默认值回归保护）
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.expert_trajectory import ExpertTrajectoryGenerator
from env.mock_pcb import MockComponent, create_patched_env
from env.pcb_layout_env import Config
from env.pcb_utils import PCBUtils


# 完全重叠的一对 AABB（硬重叠面积 4.0 mm²）
BBOX = (0.0, 0.0, 2.0, 2.0)
OVERLAP_AREA = 4.0


def _env_with(subject: MockComponent, other: MockComponent, layer_aware: bool):
    """最小环境：subject 待放置，other 已在 placed_list 里。"""
    env = create_patched_env()
    env.kicad.components = {subject.id: subject, other.id: other}
    env.fixed_list = []
    env.placed_list = [other.id]
    env.keepout_zones = []
    env.valid_area = (-10.0, -10.0, 10.0, 10.0)
    Config.LAYER_AWARE_COLLISION = layer_aware
    return env


def _comp(cid, layer, has_pth):
    return MockComponent(
        id=cid, footprint="X", package_type=0,
        bbox=BBOX, footprint_w=2.0, footprint_h=2.0,
        rotation=0, layer=layer, has_pth=has_pth,
        is_fixed=False, is_placed=False, source_bbox=BBOX,
    )


@pytest.fixture(autouse=True)
def _restore_switch():
    """开关是类属性，用例会改它 —— 用完必须还原，否则污染其它测试。"""
    orig = Config.LAYER_AWARE_COLLISION
    yield
    Config.LAYER_AWARE_COLLISION = orig


def _hard(subject, other, layer_aware):
    return _env_with(subject, other, layer_aware)._detect_collisions(subject)[0]


def _expert_legal(subject, other, layer_aware):
    """专家侧的同一判定（_is_legal_placement 是 staticmethod）。"""
    Config.LAYER_AWARE_COLLISION = layer_aware
    snap = {
        subject.id: {"bbox": BBOX, "package_type": 0,
                     "layer": subject.layer, "has_pth": subject.has_pth},
        other.id: {"bbox": BBOX, "package_type": 0,
                   "layer": other.layer, "has_pth": other.has_pth},
    }
    return ExpertTrajectoryGenerator._is_legal_placement(
        BBOX, subject.id, snap, (-10.0, -10.0, 10.0, 10.0), env=None)


# ==================== 第 1 组：异面纯 SMD ====================

def test_group1_opposite_layer_smd_may_overlap():
    """开关打开时，异面纯 SMD 的重叠必须被豁免（净收益就来自这里）。"""
    a, b = _comp("A", layer=1, has_pth=False), _comp("B", layer=0, has_pth=False)
    assert _hard(a, b, layer_aware=True) == 0.0
    assert _hard(b, a, layer_aware=True) == 0.0, "判定必须与顺序无关"
    assert PCBUtils.layers_allow_xy_overlap(0, False, 1, False, True) is True


def test_group1_expert_side_agrees():
    """专家轨迹侧必须给出同一结论（否则专家产出 env 判违规的布局）。"""
    a, b = _comp("A", layer=1, has_pth=False), _comp("B", layer=0, has_pth=False)
    assert _expert_legal(a, b, layer_aware=True) is True


def test_group1_inner_layer_is_not_exempt():
    """层编码 2（内层/未知）语义不明，一律保守，不得当异面放行。"""
    a = _comp("A", layer=1, has_pth=False)
    for weird in (2, 3):
        b = _comp("B", layer=weird, has_pth=False)
        assert _hard(a, b, layer_aware=True) == OVERLAP_AREA, f"layer={weird} 被误放行"


def test_group1_keepout_and_bounds_not_exempt():
    """层感知只管元件-元件：keepout 与出界不受层影响。"""
    a, b = _comp("A", layer=1, has_pth=False), _comp("B", layer=0, has_pth=False)
    env = _env_with(a, b, layer_aware=True)
    env.keepout_zones = [BBOX]
    env.valid_area = (5.0, 5.0, 6.0, 6.0)     # a.bbox 完全在板外
    hard, _soft, oob, in_keepout = env._detect_collisions(a)
    assert hard == 0.0, "异面 SMD 重叠本身被豁免"
    assert in_keepout is True, "keepout 不应随层豁免"
    assert oob > 0.0, "出界不应随层豁免"


# ==================== 第 2 组：含通孔 ====================

def test_group2_pth_never_exempt():
    """任一方含通孔（PTH/NPTH，孔贯穿所有层）就必须保守。"""
    for pth_a, pth_b in [(True, False), (False, True), (True, True)]:
        a = _comp("A", layer=1, has_pth=pth_a)
        b = _comp("B", layer=0, has_pth=pth_b)
        assert _hard(a, b, layer_aware=True) == OVERLAP_AREA, \
            f"pth=({pth_a},{pth_b}) 被误放行 —— 通孔叠上去是物理干涉"
        assert _expert_legal(a, b, layer_aware=True) is False
    assert PCBUtils.layers_allow_xy_overlap(0, True, 1, False, True) is False
    assert PCBUtils.layers_allow_xy_overlap(0, False, 1, True, True) is False


def test_group2_missing_has_pth_defaults_conservative():
    """没有 has_pth 属性的元件（旧数据/第三方实现）必须按"有孔"处理。"""
    assert PCBUtils.layers_allow_xy_overlap(0, False, 1, False, True) is True
    # 显式 None/缺失 -> 调用方 getattr 默认 False，但层不可辨时要保守
    assert PCBUtils.layers_allow_xy_overlap(0, False, 0, False, True) is False


# ==================== 第 3 组：同面（不回归） ====================

def test_group3_same_layer_unaffected_by_switch():
    """同面元件的判定与开关无关（开关只放行异面）。"""
    for lay in (0, 1):
        for aware in (False, True):
            a, b = _comp("A", layer=lay, has_pth=False), _comp("B", layer=lay, has_pth=False)
            assert _hard(a, b, layer_aware=aware) == OVERLAP_AREA, \
                f"layer={lay} aware={aware}: 同面重叠被误豁免"
            assert _expert_legal(a, b, layer_aware=aware) is False


def test_group3_identical_component_excluded():
    """自身不与自己比。"""
    a = _comp("A", layer=1, has_pth=True)
    env = _env_with(a, _comp("B", layer=0, has_pth=True), layer_aware=True)
    env.placed_list = [a.id]          # 自己也进了 placed_list
    assert env._detect_collisions(a)[0] == 0.0


# ==================== 第 4 组：开关关闭 = 旧行为 ====================

def test_default_switch_is_off_for_ab_testing():
    """默认值必须是关：层感知碰撞要先用开关做 A/B，验证过收益与风险再放行。

    单独成一条：如果谁在别处（或跑 A/B 时）全局强制打开了开关，
    失败的应当只有这一条"你覆盖了默认"的提示，而不是让第 4 组一起变红。
    """
    assert Config.LAYER_AWARE_COLLISION is False, \
        "LAYER_AWARE_COLLISION 默认被改成 True 了 —— 先做 A/B 再改默认"


def test_group4_switch_off_is_old_behaviour():
    """开关关闭时，异面纯 SMD 照旧算硬重叠 —— 与改动前逐字等价。"""
    for lay_a, lay_b in [(0, 1), (1, 0)]:
        a = _comp("A", layer=lay_a, has_pth=False)
        b = _comp("B", layer=lay_b, has_pth=False)
        assert _hard(a, b, layer_aware=False) == OVERLAP_AREA
        assert _expert_legal(a, b, layer_aware=False) is False
    assert PCBUtils.layers_allow_xy_overlap(0, False, 1, False, False) is False


def test_group4_switch_off_soft_spacing_unchanged():
    """软间距项在开关关闭时也必须与旧行为一致（非重叠的近邻）。"""
    a = MockComponent(id="A", footprint="X", package_type=0, bbox=(0, 0, 2, 2),
                      footprint_w=2.0, footprint_h=2.0, layer=1)
    b = MockComponent(id="B", footprint="X", package_type=0, bbox=(2.05, 0, 4.05, 2),
                      footprint_w=2.0, footprint_h=2.0, layer=0)
    off = _env_with(a, b, layer_aware=False)._detect_collisions(a)
    expected = PCBUtils.bbox_overlap_penalty(
        a.bbox, b.bbox, Config.spacing_threshold_for(0, 0))
    assert off[0] == pytest.approx(expected[0])
    assert off[1] == pytest.approx(expected[1])
    assert off[0] == 0.0 and off[1] > 0.0, "前提：这是一对只有软间距的异面近邻"
