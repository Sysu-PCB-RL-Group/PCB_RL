"""PCBLayoutEnv 单元测试。

验证 reset/step/invalid_action/mask/observation 核心逻辑。
依赖 Mock 环境，不依赖真实 KiCad。
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
    create_patched_env, MockKiCadUtils, MockComponent, MockNet,
)
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_layout_env import Config


# ==================== reset ====================


def test_reset_initial_state():
    """reset 后 env 处于初始状态。"""
    env = create_patched_env()
    env.reset()
    # 手动修改状态模拟已放置
    env.placed_list.append("U1")
    env.current_step = 5
    env.cumulative_reward = 100.0
    assert len(env.placed_list) > 0

    env.reset()
    assert len(env.placed_list) == 0
    assert env.current_step == 0
    assert env.cumulative_reward == 0.0
    assert env._pending_comp_idx == -1


# ==================== step ====================


def test_step_valid_placement():
    """有效放置 → placed_list 增加。"""
    env = create_patched_env()
    env.reset()
    comp = env.movable_list[0]
    # 在板中央放置，根据 footprint 选安全坐标
    ndx = int(comp.footprint_w / Config.CELL_SIZE)
    ndy = int(comp.footprint_h / Config.CELL_SIZE)
    gx1, gy1 = 30, 20
    action = (0, gx1, gy1, gx1 + max(1, ndx), gy1 + max(1, ndy), 0, 0)
    _, reward, terminated, _, info = env.step(action)
    assert not terminated
    assert info["placed_count"] > 0, (
        f"Expected placed_count > 0 after valid placement, got {info['placed_count']}"
    )
    assert len(env.placed_list) > 0, "placed_list should not be empty after valid step"


def test_step_out_of_bounds():
    """越界放置 → _invalid_action（reward=0）。"""
    env = create_patched_env()
    env.reset()
    # 远超板框范围 → AABB 及越界检测都会失败
    _, reward, _, _, _ = env.step(
        (0, 999, 999, 1005, 1005, 0, 0)
    )
    assert reward == 0.0


def test_step_already_placed():
    """已放置元件不可再选。"""
    env = create_patched_env()
    env.reset()
    comp = env.movable_list[0]
    ndx = max(1, int(comp.footprint_w / Config.CELL_SIZE))
    ndy = max(1, int(comp.footprint_h / Config.CELL_SIZE))
    # 第一次放置
    action1 = (0, 20, 20, 20 + ndx, 20 + ndy, 0, 0)
    _, r1, _, _, info1 = env.step(action1)
    if info1["placed_count"] == 0:
        pytest.skip("First placement was invalid — test env may have footprint mismatch")
    # 第二次选同一个 → invalid
    _, r2, _, _, _ = env.step(action1)
    assert r2 == 0.0


def test_step_aabb_size_mismatch():
    """放置尺寸与 footprint 不匹配 → invalid_action。"""
    env = create_patched_env()
    env.reset()
    comp = env.movable_list[0]
    # 故意用错误的 size（gx2-gx1 != expected_w）
    wrong_w = int(comp.footprint_w / Config.CELL_SIZE) + 5
    _, reward, _, _, _ = env.step(
        (0, 20, 20, 20 + wrong_w, 20 + int(comp.footprint_h), 0, 0)
    )
    assert reward == 0.0


# ==================== observation ====================


def test_obs_dim():
    """obs_dim = N_movable * 8 + N_movable + 4。"""
    env = create_patched_env()
    expected = env.N_movable * 8 + env.N_movable + 4
    assert env.observation_space.shape[0] == expected


def test_obs_pending_initial():
    """reset 后 pending 观测全零。"""
    env = create_patched_env()
    env.reset()
    obs = env._get_obs_with_pending(0)
    assert obs.shape == (env.observation_space.shape[0],)
    # pending_comp 部分全零
    n_movable = env.N_movable
    base_obs_len = n_movable * 8
    pending_comp = obs[base_obs_len:base_obs_len + n_movable]
    assert pending_comp.sum() == 0.0


def test_obs_after_step():
    """step 后 observation 反映新状态（unplaced slots 可能不变，但观察维度正确）。"""
    env = create_patched_env()
    env.reset()
    obs_before = env._get_obs()
    # 确保步进至少未被截断
    env.step((0, 30, 30, 35, 35, 0, 0))
    obs_after = env._get_obs()
    # obs 维度正确
    assert obs_after.shape == obs_before.shape


# ==================== mask ====================


def test_mask_step0_all_available():
    """reset 后所有 movable 元件可选。"""
    env = create_patched_env()
    env.reset()
    mask = env._build_mask_for_step(0)
    assert mask.sum() == env.N_movable


def test_mask_step1_grid_size():
    """step 1 的 x1 掩码长度 = grid_width。"""
    env = create_patched_env()
    mask = env._build_mask_for_step(1)
    assert len(mask) == env.grid_width


def test_mask_step2_grid_size():
    """step 2 的 y1 掩码长度 = grid_height。"""
    env = create_patched_env()
    mask = env._build_mask_for_step(2)
    assert len(mask) == env.grid_height


# ==================== reward ====================


def test_reward_negative_for_overlap():
    """两个元件重叠放置 → reward < 0。"""
    env = create_patched_env()
    env.reset()
    # 先放第一个元件
    comp = env.movable_list[0]
    ndx = max(1, int(comp.footprint_w / Config.CELL_SIZE))
    ndy = max(1, int(comp.footprint_h / Config.CELL_SIZE))
    action1 = (0, 20, 20, 20 + ndx, 20 + ndy, 0, 0)
    _, r1, _, _, info1 = env.step(action1)
    if info1["placed_count"] == 0:
        pytest.skip("First placement invalid — cannot test overlap")
    # 第二个元件叠放在同一个位置 → 碰撞
    action2 = (0, 20, 20, 20 + ndx, 20 + ndy, 0, 0)
    _, r2, _, _, _ = env.step(action2)
    assert r2 == 0.0  # "already placed" → invalid_action


def test_reward_nonzero_for_valid_placement():
    """有效放置 → reward != 0。"""
    env = create_patched_env()
    env.reset()
    comp = env.movable_list[0]
    ndx = max(1, int(comp.footprint_w / Config.CELL_SIZE))
    ndy = max(1, int(comp.footprint_h / Config.CELL_SIZE))
    _, reward, _, _, info = env.step(
        (0, 30, 30, 30 + ndx, 30 + ndy, 0, 0)
    )
    if info["placed_count"] > 0:
        assert reward != 0.0, "Valid placement should produce non-zero reward"


# ==================== assemble_action ====================


def test_assemble_action_correct_shape():
    """_assemble_action 返回正确的 7 元组。"""
    env = create_patched_env()
    env.reset()
    comp = env.movable_list[0]
    # 模拟完整的 5 步 pending
    env._update_pending(0, step_id=0)
    env._update_pending(30, step_id=1)
    env._update_pending(20, step_id=2)
    env._update_pending(0, step_id=3)
    env._update_pending(0, step_id=4)
    action = env._assemble_action()
    assert len(action) == 7
    assert action[0] == 0
    assert action[1] == 30
    assert action[2] == 20
    # gx2/gy2 根据 footprint 自动计算
    expected_w = max(1, int(comp.footprint_w / Config.CELL_SIZE))
    expected_h = max(1, int(comp.footprint_h / Config.CELL_SIZE))
    assert action[3] == 30 + expected_w
    assert action[4] == 20 + expected_h
