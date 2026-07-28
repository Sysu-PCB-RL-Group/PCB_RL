"""pcb_utils 纯函数单元测试（无外部依赖）。

覆盖 SPEC §6 核心约束检测：C1 越界、C2 碰撞/间距、C5 net 聚合、
C6 拥塞、AABB clamp、面积排名、合成环境工厂。
所有测试函数均为静态纯函数，适合参数化测试。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from env.pcb_utils import PCBUtils

# ==================== C1：越界面积 ====================

# valid_area 格式：(x_min, y_min, x_max, y_max) — 4-tuple


def test_out_of_bounds_inside():
    """元件完全在板内 → 越界面积为 0。"""
    area = PCBUtils.out_of_bounds_area((10, 10, 20, 20), (0, 0, 100, 80))
    assert area == 0.0


def test_out_of_bounds_partial_left():
    """元件左侧越界 → 越界面积 > 0。"""
    area = PCBUtils.out_of_bounds_area((-5, 10, 10, 20), (0, 0, 100, 80))
    assert area > 0.0


def test_out_of_bounds_partial_bottom():
    """元件底部越界 → 越界面积 > 0。"""
    area = PCBUtils.out_of_bounds_area((10, -5, 20, 10), (0, 0, 100, 80))
    assert area > 0.0


def test_out_of_bounds_completely_outside():
    """元件完全在板外 → 越界面积 = 元件面积。"""
    bbox = (200, 200, 210, 210)  # 10×10mm
    area = PCBUtils.out_of_bounds_area(bbox, (0, 0, 100, 80))
    assert area == 100.0  # 10 * 10


def test_out_of_bounds_zero_bbox():
    """空 bbox → 越界面积为 0。"""
    area = PCBUtils.out_of_bounds_area((0, 0, 0, 0), (0, 0, 100, 80))
    assert area == 0.0


# ==================== C2：AABB 碰撞 + 间距 ====================

# bbox_overlap_penalty(a, b, spacing_threshold) → (hard_overlap, soft_spacing)


def test_bbox_overlap_no_overlap():
    """两个分离的 bbox 应无重叠。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 10, 10), (20, 20, 30, 30), 0.2
    )
    assert hard == 0.0
    assert soft == 0.0


def test_bbox_overlap_with_overlap():
    """重叠的 bbox → hard_overlap > 0。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 10, 10), (5, 5, 15, 15), 0.2
    )
    assert hard > 0.0


def test_bbox_overlap_adjacent():
    """相邻（共边）的 bbox 应无重叠（硬重叠=0），但可能有软间距惩罚。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 10, 10), (10, 0, 20, 10), 5.0
    )
    assert hard == 0.0
    # 共边 gap=0，若 threshold > 0 则 soft > 0
    assert soft > 0.0


def test_bbox_overlap_one_contains_other():
    """一个元件完全包含另一个 → hard_overlap = 小元件面积。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 50, 50), (10, 10, 20, 20), 0.2
    )
    assert abs(hard - 100.0) < 0.01  # 10×10


def test_soft_spacing_far():
    """间距 > 阈值 → soft_spacing = 0（无惩罚）。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 10, 10), (25, 25, 35, 35), 0.2  # gap > 0.2
    )
    assert hard == 0.0
    assert soft == 0.0


def test_soft_spacing_close():
    """间距 < 阈值 → soft_spacing > 0 且 hard_overlap = 0。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 0, 10, 10), (11, 0, 21, 10), 5.0  # gap=1 < 5，无重叠
    )
    assert hard == 0.0
    assert soft > 0.0


# ==================== C5：网络聚合 ====================


def test_net_aggregation_no_placed():
    """空列表 → 返回 0.0。"""
    result = PCBUtils.net_aggregation([])
    assert result == 0.0


def test_net_aggregation_one_element():
    """单元件 → 返回 0.0（< 2 个时无聚合参考）。"""
    result = PCBUtils.net_aggregation([(10, 10, 20, 20)])
    assert result == 0.0


def test_net_aggregation_two_elements():
    """两元件 → 返回平均距离 > 0。"""
    result = PCBUtils.net_aggregation([(0, 0, 10, 10), (50, 50, 60, 60)])
    assert result > 0.0


# ==================== C5-1：去耦电容识别 ====================


def test_decoupling_empty_nets():
    """空 nets → 空 map。"""
    result = PCBUtils.identify_decoupling_pairs({}, {})
    assert result == {}


def test_decoupling_identifies_ic_cap_pairs():
    """电源网络上 IC + MLCC 电容 → 应识别为去耦对。"""
    from env.mock_pcb import MockComponent, MockNet

    nets = {
        "VCC": MockNet(net_id="VCC", connections=[("U1", "1"), ("C1", "1")], is_power=True),
        "GND": MockNet(net_id="GND", connections=[("U1", "2"), ("C2", "1")], is_power=True),
        "SIG1": MockNet(net_id="SIG1", connections=[("U1", "3"), ("R1", "1")], is_power=False),
    }
    components = {
        "U1": MockComponent(id="U1", footprint="QFP-44", package_type=2, nets=["VCC", "GND", "SIG1"]),
        "C1": MockComponent(id="C1", footprint="C_0603", package_type=0, nets=["VCC"]),
        "C2": MockComponent(id="C2", footprint="C_0805", package_type=0, nets=["GND"]),
        "R1": MockComponent(id="R1", footprint="R_0805", package_type=0, nets=["SIG1"]),
    }

    pairs = PCBUtils.identify_decoupling_pairs(nets, components)
    assert "U1" in pairs, "U1 should have decoupling caps"
    assert "C1" in pairs["U1"], "C1 should be paired with U1 on VCC"
    assert "C2" in pairs["U1"], "C2 should be paired with U1 on GND"


def test_decoupling_no_power_net():
    """非电源网络不应产生去耦对。"""
    from env.mock_pcb import MockComponent, MockNet

    nets = {"SIG1": MockNet(net_id="SIG1", connections=[("U1", "1"), ("C1", "1")], is_power=False)}
    components = {
        "U1": MockComponent(id="U1", footprint="QFP-44", package_type=2, nets=["SIG1"]),
        "C1": MockComponent(id="C1", footprint="C_0603", package_type=0, nets=["SIG1"]),
    }
    pairs = PCBUtils.identify_decoupling_pairs(nets, components)
    assert pairs == {}, "Non-power net should not produce decoupling pairs"


def test_min_pin_to_cap_distance_no_pins():
    """无引脚信息 → 退化为 center 距离。"""
    from env.mock_pcb import MockComponent

    ic = MockComponent(id="U1", bbox=(0, 0, 10, 10), pins=[])
    cap = MockComponent(id="C1", bbox=(50, 50, 52, 51))
    dist = PCBUtils.min_pin_to_cap_distance(ic, cap)
    # ic.center=(5,5), cap.center=(51,50.5), dist ≈ 64mm
    assert dist > 40.0


def test_min_pin_to_cap_distance_with_pins():
    """有引脚 → 取最近引脚的欧氏距离。"""
    from env.mock_pcb import MockComponent

    ic = MockComponent(id="U1", bbox=(0, 0, 10, 10),
                       pins=[("1", 0, 5), ("44", 10, 5)])
    cap = MockComponent(id="C1", bbox=(12, 2, 14, 3))
    # pin 44 at abs=(10,5), cap.center=(13,2.5), dist≈3.9mm
    dist = PCBUtils.min_pin_to_cap_distance(ic, cap)
    assert dist < 10.0


# ---- 拥塞标准差 ----


def test_congestion_with_components():
    """有元件时拥塞 std > 0。"""
    from env.mock_pcb import MockComponent, MockNet

    comps = [
        MockComponent(id="U1", bbox=(10, 10, 20, 20), nets=["N1", "N2"]),
        MockComponent(id="U2", bbox=(60, 50, 70, 60), nets=["N1"]),
    ]
    nets = {
        "N1": MockNet(net_id="N1"),
        "N2": MockNet(net_id="N2"),
    }
    std = PCBUtils.congestion_std(comps, nets, (0, 0, 100, 80), cell_size=5.0)
    assert std > 0.0, "Congestion std should be > 0 with placed components"


def test_bbox_overlap_diagonal_corner():
    """对角分离的 bbox → hard=0, soft=0（角对角，无共享边）。"""
    hard, soft = PCBUtils.bbox_overlap_penalty(
        (0, 10, 10, 20), (10, 0, 20, 10), 5.0
    )
    assert hard == 0.0
    assert soft == 0.0


# ==================== C6：拥塞均衡度 ====================


def test_congestion_empty():
    """无已放置元件 → std=0。"""
    std = PCBUtils.congestion_std([], {}, (0, 0, 100, 80), 5.0)
    assert std == 0.0


# ==================== AABB clamp ====================


def test_clamp_inside():
    """元件已在板内 → 不变。"""
    clamped = PCBUtils.clamp_to_board(10, 10, 20, 20, (0, 0, 100, 80))
    assert clamped == (10, 10, 20, 20)


def test_clamp_left_out():
    """左越界 → 平移入板。"""
    clamped = PCBUtils.clamp_to_board(-5, 10, 5, 20, (0, 0, 100, 80))
    assert clamped[0] >= 0  # x1 修正


def test_clamp_right_out():
    """右越界 → 平移入板。"""
    clamped = PCBUtils.clamp_to_board(90, 10, 110, 20, (0, 0, 100, 80))
    assert clamped[2] <= 100  # x2 修正


def test_clamp_preserves_size():
    """clamp 应保持元件宽高不变。"""
    w, h = 10.0, 10.0
    clamped = PCBUtils.clamp_to_board(-5, -5, 5, 5, (0, 0, 100, 80))
    assert abs((clamped[2] - clamped[0]) - w) < 0.01
    assert abs((clamped[3] - clamped[1]) - h) < 0.01


# ==================== 禁止区域 ====================


def test_check_keepout_empty():
    """无禁止区域 → 不相交。"""
    result = PCBUtils.check_keepout((10, 10, 20, 20), [])
    assert result is False


def test_check_keepout_no_overlap():
    """元件与 keepout 无交集 → False。"""
    zones = [(50, 50, 60, 60)]
    result = PCBUtils.check_keepout((10, 10, 20, 20), zones)
    assert result is False


def test_check_keepout_overlap():
    """元件与 keepout 有交集 → True。"""
    zones = [(10, 10, 20, 20)]
    result = PCBUtils.check_keepout((15, 15, 25, 25), zones)
    assert result is True


# ==================== 面积排名 ====================


def test_build_area_ranks_empty():
    """空列表 → 空 dict。"""
    ranks = PCBUtils.build_area_ranks([])
    assert ranks == {}


def test_build_area_ranks_order():
    """面积越大 rank 越小。"""
    class MockComp:
        def __init__(self, cid, area):
            self.id = cid
            self.footprint_area = area

    comps = [
        MockComp("big", 100.0),
        MockComp("mid", 50.0),
        MockComp("sml", 25.0),
    ]
    ranks = PCBUtils.build_area_ranks(comps)
    assert ranks["big"] == 1
    assert ranks["mid"] == 2
    assert ranks["sml"] == 3


# ==================== 合成数据工厂 ====================

def test_synthetic_pcb_has_correct_counts():
    """create_patched_env 应产生正确的元件数量。"""
    from unittest import mock

    _m = mock.MagicMock()
    sys.modules["utils.kicad_utils"] = _m

    from env.mock_pcb import create_patched_env, MockKiCadUtils, MockComponent, MockNet
    import utils

    utils.kicad_utils = sys.modules["utils.kicad_utils"]
    utils.kicad_utils.KiCadUtils = MockKiCadUtils
    utils.kicad_utils.Component = MockComponent
    utils.kicad_utils.Net = MockNet

    env = create_patched_env()
    assert env.N_movable == 5
    assert len(env.fixed_list) == 2
    assert env.board_bounds == ((0.0, 0.0), (100.0, 80.0))
