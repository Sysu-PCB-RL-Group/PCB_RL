"""真实 KiCad PCBLayoutEnv 集成测试。

验证环境初始化、reset、step 在真实数据下的行为。
"""

import os
import sys
import pytest


@pytest.mark.require_kicad
class TestRealEnv:
    """PCBLayoutEnv  + 真实 KiCad 数据。"""

    @pytest.fixture
    def real_env(self, kicad_pcb_path):
        """创建真实 PCB 环境并 reset。"""
        # 确保 KiCad 环境已初始化（conftest 的 pytest_runtest_setup 已做）
        from env.pcb_layout_env import PCBLayoutEnv
        env = PCBLayoutEnv(kicad_pcb_path)
        env.reset()
        return env

    def test_env_initialization(self, real_env):
        """环境初始化后应有正确的元件数、网络数、板框。"""
        assert real_env.N_movable > 0, "No movable components"
        assert real_env.grid_width > 0, "grid_width should be > 0"
        assert real_env.grid_height > 0, "grid_height should be > 0"
        assert len(real_env.movable_list) == real_env.N_movable
        # 板框应为有效值
        (min_x, min_y), (max_x, max_y) = real_env.board_bounds
        assert max_x > min_x
        assert max_y > min_y

    def test_reset_clears_state(self, real_env):
        """reset 后 placed_list 应为空，current_step 为 0。"""
        assert len(real_env.placed_list) == 0, "placed_list should be empty after reset"
        assert real_env.current_step == 0, "current_step should be 0 after reset"

    def test_step_0_action_space(self, real_env):
        """step 0 的动作空间：N_movable 个候选，mask 固定 MAX_MOVABLE 长度（Dict obs）。"""
        from env.pcb_layout_env import Config

        obs = real_env._get_obs_with_pending(0)
        mask = real_env._build_mask_for_step(0)
        assert mask.sum() == real_env.N_movable, (
            f"step 0 mask should have {real_env.N_movable} valid actions, got {mask.sum()}"
        )
        assert mask.shape[0] == Config.MAX_MOVABLE
        assert obs["component_slots"].shape == (Config.MAX_MOVABLE, 8)
        assert obs["global_features"].shape == (8,)

    def test_step_valid_action_does_not_crash(self, real_env):
        """执行一个有效 step 0 动作不应崩溃。

        只验证不崩溃，不验证 reward 值（随机权重下 reward 无意义）。
        """
        from env.pcb_layout_env import Config

        mask = real_env._build_mask_for_step(0)
        # 选择第一个可用元件
        valid_idx = int(mask.nonzero()[0][0])
        # step 0 action: (comp_idx, x1=0, y1=0, x2=0, y2=0, rot=0, lay=0)
        action = (valid_idx, 0, 0, 0, 0, 0, 0)
        obs_next, reward, terminated, truncated, info = real_env.step(action)
        assert obs_next["component_slots"].shape == (Config.MAX_MOVABLE, 8)
        assert obs_next["global_features"].shape == (8,)
        assert isinstance(reward, (int, float))

    def test_reset_resets_after_step(self, real_env):
        """step 后 reset，状态应恢复初始。"""
        mask = real_env._build_mask_for_step(0)
        valid_idx = int(mask.nonzero()[0][0])
        real_env.step((valid_idx, 0, 0, 0, 0, 0, 0))
        real_env.reset()
        assert len(real_env.placed_list) == 0
        assert real_env.current_step == 0

    def test_board_bounds_reasonable(self, real_env):
        """板框边界应为合理的正尺寸。"""
        (min_x, min_y), (max_x, max_y) = real_env.board_bounds
        board_w = max_x - min_x
        board_h = max_y - min_y
        assert board_w > 1.0, f"Board width too small: {board_w}mm"
        assert board_h > 1.0, f"Board height too small: {board_h}mm"
        assert real_env.grid_width >= 1, "grid_width must be >= 1"
        assert real_env.grid_height >= 1, "grid_height must be >= 1"
