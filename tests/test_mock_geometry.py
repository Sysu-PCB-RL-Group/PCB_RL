"""几何一致性测试（mock 侧 + 纯 Component 类，无需 KiCad）。

验收 4 项几何修复（对应几何缺陷分析文档）：
1. dims_at_rotation：宽高随旋转交换（奇数 delta 交换，偶数不交换）
2. pin_abs_positions：引脚绝对坐标随旋转同步
3. env.step：bbox 用精确尺寸（dims_at_rotation），不用 int() 网格缩小碰撞框
4. _assemble_action / _clamp_action_to_board：网格占用 ceil 向上取整
"""

import importlib.util
import math
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- Mock 注入（与 test_env.py 一致：env 模块 from utils.kicad_utils import ...）----
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

from env.mock_pcb import (
    create_patched_env, MockKiCadUtils, MockComponent, MockNet,
)
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_layout_env import Config

# ---- 真实 Component 类：绕开 mock，直接加载源文件 ----
# 仅测纯计算（dims_at_rotation / pin_abs_positions），不触发 KiCad 导入
_real_src = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "utils", "kicad_utils", "kicad_utils.py"))
_spec = importlib.util.spec_from_file_location("_real_kicad_utils", _real_src)
_real_kicad_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real_kicad_utils)
RealComponent = _real_kicad_utils.Component


# ==================== 1. dims_at_rotation ====================


def _make_mock_r1(**kw):
    """R1 0805：源姿态 (3.0, 1.5)，源旋转 0°，引脚在左右边缘。"""
    defaults = dict(
        id="R1", footprint="R_0805", package_type=0,
        bbox=(0, 0, 3, 1.5), footprint_w=3.0, footprint_h=1.5,
        source_rotation=0, source_bbox=(0, 0, 3, 1.5),
        pins=[("1", 0, 0.75), ("2", 3, 0.75)],
    )
    defaults.update(kw)
    return MockComponent(**defaults)


def test_dims_at_rotation_even_odd_delta():
    """偶数 delta（0°/180°）不交换，奇数 delta（90°/270°）交换。"""
    r1 = _make_mock_r1()
    assert r1.dims_at_rotation(0) == (3.0, 1.5)
    assert r1.dims_at_rotation(2) == (3.0, 1.5)
    assert r1.dims_at_rotation(1) == (1.5, 3.0)
    assert r1.dims_at_rotation(3) == (1.5, 3.0)


def test_dims_at_rotation_source_rotation_nonzero():
    """source_rotation=270°：以源姿态为基准交换（LED 真实数据 2.311×4.4705）。"""
    r1 = _make_mock_r1(source_rotation=3, footprint_w=2.311, footprint_h=4.4705)
    assert r1.dims_at_rotation(3) == (2.311, 4.4705)   # 源姿态
    assert r1.dims_at_rotation(1) == (2.311, 4.4705)   # delta=2 偶数
    assert r1.dims_at_rotation(0) == (4.4705, 2.311)   # delta=1 奇数
    assert r1.dims_at_rotation(2) == (4.4705, 2.311)   # delta=3 奇数


def test_dims_at_rotation_real_component():
    """真实 kicad_utils.Component 的 dims_at_rotation 与 mock 一致。"""
    r1 = RealComponent(
        id="R1", footprint="R_0805", package_type=0,
        bbox=(0, 0, 3, 1.5), rotation=0, layer=0,
        is_fixed=False, is_placed=False,
        footprint_w=2.311, footprint_h=4.4705,
        source_rotation=3, source_bbox=(0, 0, 2.311, 4.4705),
    )
    assert r1.dims_at_rotation(3) == (2.311, 4.4705)
    assert r1.dims_at_rotation(1) == (2.311, 4.4705)
    assert r1.dims_at_rotation(0) == (4.4705, 2.311)
    assert r1.dims_at_rotation(2) == (4.4705, 2.311)


# ==================== 2. pin_abs_positions ====================


def test_pin_abs_positions_rotation0():
    """rotation=0：引脚绝对坐标 = 中心 + 源姿态相对偏移（左右边缘）。"""
    r1 = _make_mock_r1(bbox=(10.0, 20.0, 13.0, 21.5), rotation=0)
    pins = r1.pin_abs_positions()
    assert len(pins) == 2
    assert pins[0] == ("1", 10.0, 20.75)
    assert pins[1] == ("2", 13.0, 20.75)


def test_pin_abs_positions_rotation90():
    """rotation=1：+90° 后宽高交换，引脚转到上下边缘（不再失真）。"""
    r1 = _make_mock_r1(bbox=(3.0, 5.0, 4.5, 8.0), rotation=1)
    pins = r1.pin_abs_positions()
    # bbox (3,5,4.5,8) 中心 (3.75,6.5)；pin1(源左)→上边缘, pin2(源右)→下边缘
    assert pins[0] == ("1", 3.75, 8.0)
    assert pins[1] == ("2", 3.75, 5.0)


def test_pin_abs_positions_rotation180_270():
    """rotation=2/3：180° 翻转，270° 转到下/上边缘。"""
    r1 = _make_mock_r1(bbox=(0.0, 0.0, 3.0, 1.5), rotation=2)
    pins = r1.pin_abs_positions()
    # 中心 (1.5,0.75)；180°：pin1→右边缘, pin2→左边缘
    assert pins[0] == ("1", 3.0, 0.75)
    assert pins[1] == ("2", 0.0, 0.75)

    r1 = _make_mock_r1(bbox=(0.0, 0.0, 1.5, 3.0), rotation=3)
    pins = r1.pin_abs_positions()
    # 中心 (0.75,1.5)；270°：pin1(源左)→下边缘, pin2→上边缘
    assert pins[0] == ("1", 0.75, 0.0)
    assert pins[1] == ("2", 0.75, 3.0)


def test_pin_abs_positions_real_component():
    """真实 Component 用预计算偏移：绝对坐标随【层 + 旋转】同步。"""
    comp = RealComponent(
        id="U1", footprint="QFP-44", package_type=2,
        bbox=(10, 20, 20, 30), rotation=1, layer=0,
        is_fixed=False, is_placed=False,
        pins=[("1", 0, 5), ("44", 10, 5)],
        source_rotation=0, source_bbox=(0, 0, 10, 10),
        pin_offsets_by_pose={
            (0, 0): [(-5.0, 0.0), (5.0, 0.0)],
            (0, 1): [(0.0, 5.0), (0.0, -5.0)],
            (0, 2): [(5.0, 0.0), (-5.0, 0.0)],
            (0, 3): [(0.0, -5.0), (0.0, 5.0)],
            # 背面：局部镜像后同角度下与正面不同（这里给一组可区分的值）
            (1, 0): [(-5.0, 0.0), (5.0, 0.0)],
            (1, 1): [(0.0, -5.0), (0.0, 5.0)],
            (1, 2): [(5.0, 0.0), (-5.0, 0.0)],
            (1, 3): [(0.0, 5.0), (0.0, -5.0)],
        },
    )
    # bbox (10,20,20,30) 中心 (15,25)，当前 rotation=1
    assert comp.pin_abs_positions() == [("1", 15.0, 30.0), ("44", 15.0, 20.0)]
    # 指定 rotation=3 → 垂直翻转
    assert comp.pin_abs_positions(3) == [("1", 15.0, 20.0), ("44", 15.0, 30.0)]
    # 同一 rotation 下换层必须换一组偏移（旧实现只看 rotation，这里会返回正面值）
    assert comp.pin_abs_positions(1, layer=1) == [("1", 15.0, 20.0), ("44", 15.0, 30.0)]
    assert comp.pin_abs_positions(1, layer=1) != comp.pin_abs_positions(1, layer=0)
    # layer 省略时取 self.layer
    comp.layer = 1
    assert comp.pin_abs_positions(1) == [("1", 15.0, 20.0), ("44", 15.0, 30.0)]


# ==================== 3. env.step 精确 bbox ====================


def _env_with_r1():
    env = create_patched_env()
    env.reset()
    idx = [c.id for c in env.movable_list].index("R1")
    return env, idx


def test_env_step_exact_bbox_with_rotation_swap():
    """step 用 dims_at_rotation 精确重建 bbox，90° 时宽高交换。"""
    env, idx = _env_with_r1()
    gx1, gy1 = 10, 20
    x1 = env.valid_area[0] + gx1 * Config.CELL_SIZE   # 1 + 2.0 = 3.0
    y1 = env.valid_area[1] + gy1 * Config.CELL_SIZE   # 1 + 4.0 = 5.0
    grid_w = math.ceil(1.5 / Config.CELL_SIZE)        # 8
    grid_h = math.ceil(3.0 / Config.CELL_SIZE)        # 15
    _, reward, _, _, info = env.step((idx, gx1, gy1, gx1 + grid_w, gy1 + grid_h, 1, 0))
    assert info["placed_count"] > 0
    comp = env.movable_list[idx]
    assert comp.bbox == (x1, y1, x1 + 1.5, y1 + 3.0)   # 精确尺寸，非网格
    assert comp.rotation == 1


def test_env_step_bbox_not_shrunk_by_int_grid():
    """即使 action 传 int() 缩小的网格(19)，bbox 仍用精确 3.95mm 宽度。"""
    env, idx = _env_with_r1()
    r1 = env.movable_list[idx]
    r1.footprint_w, r1.footprint_h = 3.95, 2.9
    r1.source_bbox = (0, 0, 3.95, 2.9)
    gx1, gy1 = 10, 20
    x1 = env.valid_area[0] + gx1 * Config.CELL_SIZE
    y1 = env.valid_area[1] + gy1 * Config.CELL_SIZE
    # int(3.95/0.2)=19 → placed_w=3.8，与目标 3.95 差 0.15 ≤ 容差，校验通过
    _, _, _, _, info = env.step((idx, gx1, gy1, gx1 + 19, gy1 + 14, 0, 0))
    assert info["placed_count"] > 0
    comp = env.movable_list[idx]
    assert comp.bbox == (x1, y1, x1 + 3.95, y1 + 2.9)


def test_env_step_rejects_grid_size_mismatch():
    """网格尺寸与目标旋转真实尺寸差超过 CELL_SIZE → invalid。"""
    env, idx = _env_with_r1()
    r1 = env.movable_list[idx]
    r1.footprint_w, r1.footprint_h = 3.95, 2.9
    r1.source_bbox = (0, 0, 3.95, 2.9)
    # rot=1 → dims (2.9, 3.95)；错传 rot=0 的网格 (20,15) → placed_w=4.0 vs 2.9 差 1.1
    _, reward, _, _, info = env.step((idx, 10, 20, 10 + 20, 20 + 15, 1, 0))
    assert reward == 0.0
    assert "invalid_action" in info   # _invalid_action 只带 reason，无 placed_count


# ==================== 4. ceil 网格占用 ====================


def _set_pending(env, idx, rot):
    for step_id, a in [(0, idx), (1, 10), (2, 20), (3, rot), (4, 0)]:
        env._update_pending(a, step_id=step_id)


def test_assemble_action_ceil_grid():
    """_assemble_action 网格占用 ceil 取整（3.95→20，非 int 19）。"""
    env, idx = _env_with_r1()
    r1 = env.movable_list[idx]
    r1.footprint_w, r1.footprint_h = 3.95, 2.9
    r1.source_bbox = (0, 0, 3.95, 2.9)
    _set_pending(env, idx, rot=0)
    action = env._assemble_action()
    assert action[3] - action[1] == math.ceil(3.95 / Config.CELL_SIZE)   # 20
    assert action[4] - action[2] == math.ceil(2.9 / Config.CELL_SIZE)    # 15


def test_assemble_action_rotation_swap_grid():
    """_assemble_action 按目标旋转交换宽高后再 ceil。"""
    env, idx = _env_with_r1()
    r1 = env.movable_list[idx]
    r1.footprint_w, r1.footprint_h = 3.95, 2.9
    r1.source_bbox = (0, 0, 3.95, 2.9)
    _set_pending(env, idx, rot=1)
    action = env._assemble_action()   # rot=1 → dims (2.9, 3.95)
    assert action[3] - action[1] == math.ceil(2.9 / Config.CELL_SIZE)    # 15
    assert action[4] - action[2] == math.ceil(3.95 / Config.CELL_SIZE)   # 20


def test_clamp_action_to_board_uses_ceil_and_rotation():
    """run_inference._clamp_action_to_board 用 dims_at_rotation + ceil 重建网格。"""
    from run_inference import _clamp_action_to_board
    env, idx = _env_with_r1()
    r1 = env.movable_list[idx]
    r1.footprint_w, r1.footprint_h = 3.95, 2.9
    r1.source_bbox = (0, 0, 3.95, 2.9)
    out = _clamp_action_to_board(env, idx, 10, 20, 10 + 19, 20 + 14, 1, 0)
    assert out[0] == idx and out[5] == 1 and out[6] == 0
    assert out[3] - out[1] == math.ceil(2.9 / Config.CELL_SIZE)          # 15
    assert out[4] - out[2] == math.ceil(3.95 / Config.CELL_SIZE)         # 20
