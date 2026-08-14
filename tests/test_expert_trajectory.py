"""专家轨迹生成模块测试（无需 KiCad）。

验证：
1. 反向拆解：完整布局 → 逐件拆解 → 回溯链正确性
2. 微量变异：位置/旋转变异概率统计
3. Transition 构造：5 步子步骤 + 正确 pending 状态
4. 序列化：save/load 往返一致性
5. ExpertReplayBuffer：专家常驻 + PER 采样比例
6. BC 预训练：learn_bc() loss 下降

Mock 数据来源：env.mock_pcb。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- mock utils.kicad_utils 必须在 import env 之前 ----
from unittest import mock
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

import logging
import tempfile
import json

import numpy as np
import torch

from env.mock_pcb import create_patched_env, MockKiCadUtils, MockComponent, MockNet
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_layout_env import Config
from env.expert_trajectory import (
    ExpertTrajectoryGenerator,
    generate_and_save_expert_transitions,
)
from dqn import DQNAgent
from dqn.replay_buffer import ExpertReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===================== 测试 1: 反向拆解 =====================

def test_decompose_backward():
    """验证反向拆解回溯链的正确性。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=1,
        epsilon_decompose=1.0,  # 强制随机选（覆盖所有路径）
    )

    chain = generator.decompose_backward(env)

    # 拆解链长度 = 可动元件数
    assert len(chain) == env.N_movable, \
        f"Chain length {len(chain)} != N_movable {env.N_movable}"

    # 验证每条 snapshot 的完整性
    for comp_id, snapshot in chain:
        assert comp_id in snapshot, f"Component {comp_id} not in snapshot"
        comp_data = snapshot[comp_id]
        assert "bbox" in comp_data
        assert "rotation" in comp_data
        assert "layer" in comp_data
        assert len(comp_data["bbox"]) == 4

    logger.info(f"test_decompose_backward: PASSED (chain length={len(chain)})")


# ===================== 测试 2: 拆解顺序策略 =====================

def test_decompose_order_strategy():
    """验证 ε 策略：ε=0 时始终选最小面积，ε=1 时随机。"""
    env = create_patched_env()

    # ε=0: 始终选最小面积
    gen_min = ExpertTrajectoryGenerator(trajectories_per_board=1, epsilon_decompose=0.0)
    chain_min = gen_min.decompose_backward(env)

    # 验证第一个拆掉的元件面积是最小的
    first_comp_id = chain_min[0][0]
    first_comp = env.kicad.components[first_comp_id]
    for other_id in [c.id for c in env.movable_list]:
        if other_id == first_comp_id:
            continue
        other = env.kicad.components[other_id]
        assert first_comp.footprint_area <= other.footprint_area, \
            f"Expected smallest area first, got {first_comp_id}({first_comp.footprint_area}) vs {other_id}({other.footprint_area})"

    logger.info("test_decompose_order_strategy: PASSED")


# ===================== 测试 3: 微量变异 =====================

def test_mutate_component_state():
    """验证变异概率统计正确（20% 位置，10% 旋转，70% 无变异）。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        pos_noise_prob=0.2,
        rot_noise_prob=0.1,
    )

    # 创建一个包含所有 movable 元件的快照
    env.reset()
    for comp in env.movable_list:
        comp.is_placed = True
    env.placed_list = [c.id for c in env.movable_list]

    snapshot = ExpertTrajectoryGenerator._snapshot_all_components(env)

    # 多次变异统计
    n_trials = 1000
    pos_mutations = 0
    rot_mutations = 0

    for comp_id in [c.id for c in env.movable_list]:
        for _ in range(n_trials // len(env.movable_list)):
            mutated = generator._mutate_component_state(comp_id, snapshot, env)
            if mutated["bbox"] != snapshot[comp_id]["bbox"]:
                pos_mutations += 1
            if mutated["rotation"] != snapshot[comp_id]["rotation"]:
                rot_mutations += 1

    pos_rate = pos_mutations / n_trials
    rot_rate = rot_mutations / n_trials

    # 允许 ±5% 偏差
    assert 0.15 <= pos_rate <= 0.25, \
        f"Position mutation rate {pos_rate:.3f} not in [0.15, 0.25]"
    assert 0.05 <= rot_rate <= 0.15, \
        f"Rotation mutation rate {rot_rate:.3f} not in [0.05, 0.15]"

    logger.info(
        f"test_mutate_component_state: PASSED "
        f"(pos_rate={pos_rate:.3f}, rot_rate={rot_rate:.3f})"
    )


# ===================== 测试 4: Transition 构造 =====================

def test_trajectory_to_transitions():
    """验证轨迹 → transition 转换的正确性。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=1,
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,  # 无变异，便于验证
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)

    # 每条轨迹 = N_movable × 5 子步骤
    expected_count = env.N_movable * 5
    assert len(transitions) == expected_count, \
        f"Expected {expected_count} transitions, got {len(transitions)}"

    # 验证每条 transition 结构
    for t in transitions:
        obs, action, reward, next_obs, mask, done, step_id, next_step_id = t

        # 观测结构
        assert "component_slots" in obs
        assert "global_features" in obs
        assert obs["component_slots"].shape == (Config.MAX_MOVABLE, 8)
        assert obs["global_features"].shape == (8,)

        # step_id 范围
        assert 0 <= step_id <= 4

        # next_step_id: -1（终端）或 0~4
        assert next_step_id in (-1, 0, 1, 2, 3, 4)

        # step 0-3 的 done 必须为 False
        if step_id < 4:
            assert done is False

        # action 必须是整数
        assert isinstance(action, (int, np.integer))

    logger.info(
        f"test_trajectory_to_transitions: PASSED ({len(transitions)} transitions)"
    )


# ===================== 测试 5: 序列化往返 =====================

def test_save_load_transitions():
    """验证 transition 序列化 → 反序列化往返一致性。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=1,
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)

    with tempfile.TemporaryDirectory() as tmpdir:
        generator.save_transitions(transitions, tmpdir, "test.kicad_pcb")
        filepath = os.path.join(tmpdir, "test_expert.json")

        assert os.path.isfile(filepath), f"File not created: {filepath}"

        loaded = ExpertTrajectoryGenerator.load_transitions(filepath)
        assert len(loaded) == len(transitions), \
            f"Loaded {len(loaded)} != original {len(transitions)}"

        # 验证第一条和最后一条的关键字段
        for i in [0, -1]:
            orig = transitions[i]
            load = loaded[i]
            np.testing.assert_array_almost_equal(
                orig[0]["component_slots"], load[0]["component_slots"],
                decimal=4
            )
            assert orig[1] == load[1], f"Action mismatch at index {i}"
            assert abs(orig[2] - load[2]) < 1e-6, f"Reward mismatch at index {i}"

    logger.info("test_save_load_transitions: PASSED")


# ===================== 测试 6: ExpertReplayBuffer 基本功能 =====================

def test_expert_replay_buffer():
    """验证 ExpertReplayBuffer 的专家常驻 + 采样比例。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=1,
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)
    n_expert = len(transitions)

    # 创建 buffer
    buffer = ExpertReplayBuffer(
        expert_capacity=n_expert,
        normal_capacity=n_expert * 4,  # total = 5 × n_expert
        expert_ratio=0.2,
        per_alpha=0.6,
        per_beta=0.4,
    )

    # 加载专家经验
    buffer.load_expert_transitions(transitions)
    assert buffer.get_expert_count() == n_expert
    assert buffer.get_normal_count() == 0  # 还没加普通经验

    # 添加一些普通经验
    for i in range(10):
        # 使用第一条 transition 的 obs 作为模板
        obs = {k: v.copy() for k, v in transitions[0][0].items()}
        next_obs = {k: v.copy() for k, v in transitions[0][3].items()}
        buffer.add_normal(
            obs, i % 5, 0.0, next_obs, None, False,
            step_id=i % 5, next_step_id=(i % 5) + 1 if (i % 5) < 4 else -1,
        )

    assert buffer.get_normal_count() == 10

    # 验证 is_ready
    assert buffer.is_ready(10)

    # 采样
    batch = buffer.sample(10)
    # 返回 8 元组（与 ReplayBuffer.sample() 兼容）
    assert len(batch) == 8

    logger.info("test_expert_replay_buffer: PASSED")


# ===================== 测试 7: BC 预训练 loss 下降 =====================

def test_bc_pretrain_loss_decrease():
    """验证 learn_bc() 训练后 loss 下降。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=1,
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)

    # 创建 Agent
    device = torch.device("cpu")
    agent = DQNAgent(
        n_movable=Config.MAX_MOVABLE,
        grid_w=Config.GRID_W,
        grid_h=Config.GRID_H,
        device=device,
        lr=0.001,
    )

    # BC 预训练（少量 epoch）
    epoch_losses = agent.learn_bc(
        expert_transitions=transitions,
        batch_size=8,
        epochs=5,
        verbose=False,
    )

    # 验证 loss 下降
    assert len(epoch_losses) == 5
    assert epoch_losses[-1] < epoch_losses[0] * 0.95, \
        f"Loss did not decrease: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}"

    logger.info(
        f"test_bc_pretrain_loss_decrease: PASSED "
        f"(loss: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f})"
    )


# ===================== 测试 8: 便捷函数 generate_and_save =====================

def test_generate_and_save():
    """验证便捷函数 generate_and_save_expert_transitions()。"""
    env = create_patched_env()

    with tempfile.TemporaryDirectory() as tmpdir:
        transitions, filepath = generate_and_save_expert_transitions(
            env=env,
            pcb_name="test_mock.kicad_pcb",
            output_dir=tmpdir,
            trajectories_per_board=2,
            epsilon_decompose=1.0,
            pos_noise_prob=0.0,
            rot_noise_prob=0.0,
        )

        assert os.path.isfile(filepath), f"File not created: {filepath}"
        assert len(transitions) > 0, "No transitions generated"

        # 验证可以从文件加载
        loaded = ExpertTrajectoryGenerator.load_transitions(filepath)
        assert len(loaded) == len(transitions)

    logger.info(
        f"test_generate_and_save: PASSED ({len(transitions)} transitions)"
    )


# ===================== 运行全部测试 =====================

if __name__ == "__main__":
    test_decompose_backward()
    test_decompose_order_strategy()
    test_mutate_component_state()
    test_trajectory_to_transitions()
    test_save_load_transitions()
    test_expert_replay_buffer()
    test_bc_pretrain_loss_decrease()
    test_generate_and_save()
    logger.info("=" * 50)
    logger.info("All expert trajectory tests PASSED!")
