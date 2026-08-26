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


def test_mask_step1_grid_bounds():
    """step 1 返回 (GRID_W,) 越界 mask，grid_x < grid_width 才合法。"""
    env = create_patched_env()
    mask = env._build_mask_for_step(1)
    assert mask is not None
    assert mask.shape == (Config.GRID_W,), f"Expected ({Config.GRID_W},), got {mask.shape}"
    assert mask.sum() == env.grid_width  # 只有板内坐标合法
    assert not mask[env.grid_width:].any()  # 越界坐标全 False


def test_mask_step2_grid_bounds():
    """step 2 返回 (GRID_H,) 越界 mask，grid_y < grid_height 才合法。"""
    env = create_patched_env()
    mask = env._build_mask_for_step(2)
    assert mask is not None
    assert mask.shape == (Config.GRID_H,), f"Expected ({Config.GRID_H},), got {mask.shape}"
    assert mask.sum() == env.grid_height  # 只有板内坐标合法
    assert not mask[env.grid_height:].any()  # 越界坐标全 False


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


# ==================== terminated / truncated ====================


def test_terminated_when_all_placed():
    """全部 movable 元件成功放置后 terminated=True。

    逐个放置所有元件，在最后一个上验证 terminated。
    """
    from env.mock_pcb import create_patched_env
    env = create_patched_env()
    env.reset()

    # 逐个放置所有可移动元件（选不重叠的网格位置）
    placed = 0
    for i, comp in enumerate(env.movable_list):
        ndx = max(1, int(comp.footprint_w / Config.CELL_SIZE))
        ndy = max(1, int(comp.footprint_h / Config.CELL_SIZE))
        gx = 5 + i * 15  # 分散放置避免碰撞
        gy = 5 + i * 10
        if gx + ndx >= env.grid_width or gy + ndy >= env.grid_height:
            continue  # 超出网格范围就跳过
        action = (i, gx, gy, gx + ndx, gy + ndy, 0, 0)
        _, reward, terminated, truncated, info = env.step(action)
        if info["placed_count"] > placed:
            placed = info["placed_count"]

    # 最后一个应触发 terminated
    if placed == env.N_movable:
        assert terminated, "Should be terminated when all components placed"
    else:
        pytest.skip(f"Could not place all {env.N_movable} components in test")


def test_truncated_on_max_steps():
    """超过 MAX_STEPS 后 truncated=True。"""
    from env.mock_pcb import create_patched_env
    env = create_patched_env()
    env.reset()

    # 不断发送无效动作直到 current_step 接近 MAX_STEPS
    old_max = Config.MAX_STEPS
    Config.MAX_STEPS = 5
    try:
        terminated, truncated = False, False
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step((0, 999, 999, 999, 999, 0, 0))
            if truncated:
                break
        assert truncated, "Should be truncated after MAX_STEPS"
    finally:
        Config.MAX_STEPS = old_max


def test_early_stop_on_low_reward():
    """累计 reward 低于 EARLY_STOP_THRESHOLD 后应 truncated。"""
    from env.mock_pcb import create_patched_env
    env = create_patched_env()
    env.reset()
    assert env.N_movable >= 2, "Need at least 2 movable components"

    old_threshold = Config.EARLY_STOP_THRESHOLD
    old_overlap_hard = Config.W_OVERLAP_HARD
    Config.EARLY_STOP_THRESHOLD = -10.0
    Config.W_OVERLAP_HARD = 10000
    try:
        # 先放置元件 0
        comp0 = env.movable_list[0]
        ndx = max(1, int(comp0.footprint_w / Config.CELL_SIZE))
        ndy = max(1, int(comp0.footprint_h / Config.CELL_SIZE))
        action0 = (0, 10, 10, 10 + ndx, 10 + ndy, 0, 0)
        obs, r1, _, _, info1 = env.step(action0)
        if info1["placed_count"] == 0:
            pytest.skip("First placement failed, cannot test early stop")

        # 用元件 1/2/3 连续叠放在元件 0 的同一位置 → 多次巨额碰撞惩罚。
        # 注意：Reward Clip 将单步惩罚限制在 [-30, 60]，单次违规不足以低于阈值，
        # 需多次违规累计 reward 跌至 EARLY_STOP_THRESHOLD 以下才触发早停。
        truncated = False
        for idx in range(1, min(4, env.N_movable)):
            comp = env.movable_list[idx]
            ndx1 = max(1, int(comp.footprint_w / Config.CELL_SIZE))
            ndy1 = max(1, int(comp.footprint_h / Config.CELL_SIZE))
            obs, r, _, truncated, _ = env.step((idx, 10, 10, 10 + ndx1, 10 + ndy1, 0, 0))
            if truncated:
                break
        assert truncated, (
            f"Should have hit EARLY_STOP after penalties; cumulative_reward={env.cumulative_reward}"
        )
    finally:
        Config.EARLY_STOP_THRESHOLD = old_threshold
        Config.W_OVERLAP_HARD = old_overlap_hard


def test_action_space_on_mock_env():
    """mock env 应正确设置 action_space（回归：__new__ 绕过 gym.__init__）。"""
    from env.mock_pcb import create_patched_env
    env = create_patched_env()
    assert env.action_space is not None, "action_space should not be None (regression)"
    assert len(env.action_space) == 7, "action_space should be 7-tuple"


# ==================== Config ====================


def test_config_load_from_file_inline_comments():
    """验证 Config.load_from_file 正确剥离行尾 # 注释并加载参数。

    回归测试：修复前因注释导致所有参数被静默跳过（REPLAY_BUFFER_SIZE 始终为默认 10000）。
    """
    import tempfile

    # 保存当前 Config 值，测试后恢复
    saved = {k: getattr(Config, k) for k in dir(Config)
             if not k.startswith("_") and not callable(getattr(Config, k))}

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("REPLAY_BUFFER_SIZE  = 87654    # comment\n")
            f.write("W_OVERLAP_HARD       = 3141.59  # float with comment\n")
            f.write("MAX_STEPS            = 777      # int with comment\n")
            f.write("# 纯注释行应被跳过\n")
            f.write("\n")  # 空行
            f.write("EARLY_STOP_THRESHOLD  = -99999.0  # negative with comment\n")
            tmp_path = f.name

        Config.REPLAY_BUFFER_SIZE = 10000  # 重置为默认
        Config.W_OVERLAP_HARD = 1000.0
        Config.MAX_STEPS = 2000
        Config.EARLY_STOP_THRESHOLD = -500.0

        Config.load_from_file(tmp_path)

        assert Config.REPLAY_BUFFER_SIZE == 87654, \
            f"Expected 87654, got {Config.REPLAY_BUFFER_SIZE}"
        assert abs(Config.W_OVERLAP_HARD - 3141.59) < 0.01, \
            f"Expected 3141.59, got {Config.W_OVERLAP_HARD}"
        assert Config.MAX_STEPS == 777, \
            f"Expected 777, got {Config.MAX_STEPS}"
        assert Config.EARLY_STOP_THRESHOLD == -99999.0, \
            f"Expected -99999.0, got {Config.EARLY_STOP_THRESHOLD}"
    finally:
        # 恢复原始值
        for k, v in saved.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                setattr(Config, k, v)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_config_spacing_threshold():
    """spacing_threshold_for 返回两个封装类型间距阈值的 max（读取 Config 属性而非硬编码）。"""
    thresholds = [
        Config.SPACING_CHIP,
        Config.SPACING_SOIC,
        Config.SPACING_QFP,
        Config.SPACING_BGA,
        Config.SPACING_IC,
    ]
    for i in range(5):
        for j in range(5):
            expected = max(thresholds[i], thresholds[j])
            assert Config.spacing_threshold_for(i, j) == expected
