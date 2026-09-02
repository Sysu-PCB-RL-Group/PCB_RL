"""MockPCB 单元测试。

验证 MockKiCadUtils 的 save_initial_state / load_board reset 机制、
create_patched_env 的正确性、JSON 加载器的字段校验。
"""

import numpy as np
import pytest
from unittest import mock

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- Mock 注入 ----
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

from env.mock_pcb import (
    create_patched_env, create_env_from_json,
    load_pcb_from_json, build_synthetic_pcb,
    MockKiCadUtils, MockComponent, MockNet,
)
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet


# ==================== MockComponent / MockNet ====================


def test_mock_component_defaults():
    """MockComponent 默认值与 kicad_utils.Component 一致。"""
    c = MockComponent(id="U1")
    assert c.footprint == ""
    assert c.package_type == 4
    assert c.bbox == (0.0, 0.0, 0.0, 0.0)
    assert c.rotation == 0
    assert c.layer == 0
    assert c.is_fixed is False
    assert c.is_placed is False
    assert c.nets == []
    assert c.pins == []
    # 修正后默认值为 0.0（对齐 kicad_utils.Component）
    assert c.footprint_w == 0.0
    assert c.footprint_h == 0.0


def test_mock_component_center():
    """center 属性正确计算。"""
    c = MockComponent(id="U1", bbox=(10, 20, 30, 40))
    assert c.center == (20.0, 30.0)


def test_mock_net_defaults():
    """MockNet 默认值正确。"""
    n = MockNet(net_id="N1")
    assert n.net_id == "N1"
    assert n.connections == []
    assert n.is_power is False


# ==================== MockKiCadUtils ====================


def test_save_and_load_reset():
    """save_initial_state → 修改元件 → load_board 恢复。"""
    kicad = MockKiCadUtils()
    c1 = MockComponent(id="U1", bbox=(10, 10, 20, 20), is_fixed=False, is_placed=False)
    c2 = MockComponent(id="J1", bbox=(0, 0, 5, 5), is_fixed=True, is_placed=True)
    kicad.components["U1"] = c1
    kicad.components["J1"] = c2
    kicad.save_initial_state()

    # 修改元件状态
    c1.bbox = (50, 50, 60, 60)
    c1.rotation = 2
    c1.is_placed = True

    # load_board 应恢复
    kicad.load_board("dummy")
    assert c1.bbox == (10, 10, 20, 20)
    assert c1.rotation == 0
    assert c1.is_placed is False
    # 固定元件也恢复
    assert c2.bbox == (0, 0, 5, 5)


def test_load_board_without_save():
    """未调用 save_initial_state → load_board 不报错（向后兼容）。"""
    kicad = MockKiCadUtils()
    kicad.components["U1"] = MockComponent(id="U1", bbox=(1, 2, 3, 4))
    kicad.load_board("dummy")  # 不应该报错
    # bbox 不变（因为没有保存初始状态）
    assert kicad.components["U1"].bbox == (1, 2, 3, 4)


# ==================== build_synthetic_pcb ====================


def test_synthetic_pcb_counts():
    """build_synthetic_pcb 产生正确的元件/网络数量。"""
    kicad = build_synthetic_pcb()
    assert len(kicad.components) == 7  # 5 movable + 2 fixed
    assert len(kicad.nets) == 3
    assert kicad.board_bounds == ((0, 0), (100, 80))


def test_synthetic_pcb_fixed_vs_movable():
    """固定元件 J1/J2 vs 可移动元件 U1/U2/C1/C2/R1。"""
    kicad = build_synthetic_pcb()
    movable = [c for c in kicad.components.values() if not c.is_fixed]
    fixed = [c for c in kicad.components.values() if c.is_fixed]
    assert len(movable) == 5
    assert len(fixed) == 2
    assert all(c.id.startswith("J") for c in fixed)


# ==================== create_patched_env ====================


def test_patched_env_basics():
    """create_patched_env 产生可用的 env。"""
    env = create_patched_env()
    assert env.N_movable == 5
    assert len(env.fixed_list) == 2
    assert env.board_bounds == ((0, 0), (100, 80))
    assert env.grid_width > 0
    assert env.grid_height > 0


def test_patched_env_reset():
    """reset 后 env 回到初始状态。"""
    from env.pcb_layout_env import Config
    from dqn import DQNAgent
    import torch

    env = create_patched_env()
    agent = DQNAgent(
        n_movable=env.N_movable,
        grid_w=Config.GRID_W, grid_h=Config.GRID_H,
        device=torch.device("cpu"),
    )

    # 运行一个 episode（修改 env 状态）
    from run_inference import run_inference_episode
    run_inference_episode(env, agent)

    # reset 后应清除 placed_list
    env.reset()
    assert env.placed_list == []
    assert env.current_step == 0


# ==================== load_pcb_from_json ====================


def test_load_pcb_from_json_counts():
    """加载 JSON → 元件/网络计数非零即可（不同 PCB 数据不同）。"""
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "模拟", "pcb_001.json"
    )
    kicad = load_pcb_from_json(json_path)
    assert len(kicad.components) >= 1
    assert len(kicad.nets) >= 1


def test_load_pcb_from_json_correct_structure():
    """加载的 JSON 包含 movable 和 fixed 元件。"""
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "模拟", "pcb_001.json"
    )
    kicad = load_pcb_from_json(json_path)
    movable = [c for c in kicad.components.values() if not c.is_fixed]
    fixed = [c for c in kicad.components.values() if c.is_fixed]
    assert len(movable) >= 1, "Should have at least 1 movable component"
    assert len(fixed) >= 1, "Should have at least 1 fixed component"


def test_load_pcb_from_json_missing_field():
    """缺少 board_bounds 字段 → ValueError。"""
    import json, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"components": [], "nets": []}, f)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="board_bounds"):
            load_pcb_from_json(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_create_env_from_json():
    """create_env_from_json 产生可用的 env。"""
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "模拟", "pcb_001.json"
    )
    env = create_env_from_json(json_path)
    assert env.N_movable == 5
    assert len(env.fixed_list) == 2
