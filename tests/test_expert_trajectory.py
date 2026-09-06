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
    gen_min = ExpertTrajectoryGenerator(epsilon_decompose=0.0)
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
    """验证变异概率统计正确（20% 位置，10% 旋转，70% 无变异）。

    使用手动构造的"分散布局"快照（元件互不重叠、不越界、间距 > 4mm），
    避免 mock 默认元件都堆在原点导致变异全部触发重叠回退，从而让
    位置变异率统计失真。
    """
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        pos_noise_prob=0.2,
        rot_noise_prob=0.1,
    )

    va_xmin, va_ymin, va_xmax, va_ymax = env.valid_area

    # 分散布局：元件间距 > 4mm，保证 ±2 格变异不会撞到相邻元件
    dispersed = {
        "U1": {"bbox": (va_xmin + 10, va_ymin + 10, va_xmin + 20, va_ymin + 20),
                "rotation": 0, "layer": 0, "is_placed": True},
        "U2": {"bbox": (va_xmin + 40, va_ymin + 10, va_xmin + 46, va_ymin + 14),
                "rotation": 0, "layer": 0, "is_placed": True},
        "C1": {"bbox": (va_xmin + 10, va_ymin + 40, va_xmin + 12, va_ymin + 41),
                "rotation": 0, "layer": 0, "is_placed": True},
        "C2": {"bbox": (va_xmin + 20, va_ymin + 40, va_xmin + 22, va_ymin + 41),
                "rotation": 0, "layer": 0, "is_placed": True},
        "R1": {"bbox": (va_xmin + 40, va_ymin + 40, va_xmin + 43, va_ymin + 41.5),
                "rotation": 0, "layer": 0, "is_placed": True},
    }

    # 多次变异统计
    n_trials = 1000
    pos_mutations = 0
    rot_mutations = 0

    for comp_id in dispersed:
        for _ in range(n_trials // len(dispersed)):
            mutated = generator._mutate_component_state(comp_id, dispersed, env)
            if mutated["bbox"] != dispersed[comp_id]["bbox"]:
                pos_mutations += 1
            if mutated["rotation"] != dispersed[comp_id]["rotation"]:
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


def test_is_legal_placement():
    """验证合法性检查（越界 / 硬重叠）的确定性判定。

    注意：软间距（gap < 阈值但无物理交叉）不在此拦截——它只产生轻微
    reward 惩罚、不构成违规（has_violations），且量化后完美布局在 0.2mm
    边界可能出现软间距，若一并拦截会导致 base 回退失败。
    """
    valid_area = (0.0, 0.0, 100.0, 80.0)
    snapshot = {
        "A": {"bbox": (10, 10, 20, 20), "is_placed": True, "package_type": 0, "layer": 0},
        "B": {"bbox": (30, 10, 40, 20), "is_placed": True, "package_type": 0, "layer": 0},
    }

    # 越界（右边界超出）
    assert not ExpertTrajectoryGenerator._is_legal_placement(
        (95, 10, 105, 20), "A", snapshot, valid_area
    ), "Should reject out-of-bounds"
    # 与 B 硬重叠
    assert not ExpertTrajectoryGenerator._is_legal_placement(
        (35, 10, 45, 20), "A", snapshot, valid_area
    ), "Should reject overlap with B"
    # 软间距（gap=0.1，无物理交叉）→ 允许（仅产生轻微 reward 惩罚）
    assert ExpertTrajectoryGenerator._is_legal_placement(
        (19.9, 10, 29.9, 20), "A", snapshot, valid_area
    ), "Soft spacing should be allowed (not a violation)"
    # 合法（无重叠、无软间距、无越界）
    assert ExpertTrajectoryGenerator._is_legal_placement(
        (50, 10, 60, 20), "A", snapshot, valid_area
    ), "Should accept legal placement"
    # 跳过自身（A 的原始位置与 A 自身重叠不算）
    assert ExpertTrajectoryGenerator._is_legal_placement(
        (10, 10, 20, 20), "A", snapshot, valid_area
    ), "Should skip self"

    logger.info("test_is_legal_placement: PASSED")


def test_is_legal_placement_global_check():
    """全局检查：即使碰撞对象是尚未放置的元件，也应拦截。

    修复 with_expert 崩坏的关键：此前 _is_legal_placement 只检查
    is_placed=True 的元件，先放置的元件变异后侵入"稍后才放置"元件空间
    会漏检，导致后续元件 reward 为负。改为遍历所有元件（含未放置者）。
    """
    valid_area = (0.0, 0.0, 100.0, 80.0)
    # B 尚未放置（is_placed=False），但全局检查不应跳过它
    snapshot = {
        "A": {"bbox": (10, 10, 20, 20), "is_placed": True, "package_type": 0, "layer": 0},
        "B": {"bbox": (35, 10, 45, 20), "is_placed": False, "package_type": 0, "layer": 0},
    }
    # 候选与尚未放置的 B 硬重叠 → 必须拦截
    assert not ExpertTrajectoryGenerator._is_legal_placement(
        (40, 10, 50, 20), "A", snapshot, valid_area
    ), "Should reject overlap with not-yet-placed B (global check)"
    # 不与任何元件重叠 → 接受
    assert ExpertTrajectoryGenerator._is_legal_placement(
        (60, 10, 70, 20), "A", snapshot, valid_area
    ), "Should accept non-overlapping placement"

    logger.info("test_is_legal_placement_global_check: PASSED")


def _setup_collision_free_env():
    """构造碰撞自由的 mock 环境（默认 mock 把所有 movable 堆在原点，
    不是合法完美布局，无法验证"无负 reward"）。

    将 5 个 movable 元件排布为非重叠的"完美布局"，相邻间隙 0.3mm
    （> 软间距阈值 0.2，故基准 reward 无惩罚；变异 +0.4mm 时才会
    翻转为硬重叠，用于压力测试全局检查的回退）。固定元件 j1/j2 在
    板两侧，movable 排布在中间避开。
    """
    env = create_patched_env()
    kicad = env.kicad
    # (x1, y1, x2, y2)，均落在 valid_area (≈1,1,99,79) 内
    layout = {
        "U1": (20.0, 10.0, 30.0, 20.0),    # 10×10
        "U2": (30.3, 10.0, 36.3, 14.0),    # 6×4，与 U1 间隙 0.3
        "R1": (36.6, 10.0, 39.6, 11.5),    # 3×1.5，与 U2 间隙 0.3
        "C1": (20.0, 22.0, 22.0, 23.0),    # 2×1，第二排
        "C2": (30.3, 22.0, 32.3, 23.0),    # 2×1，第二排
    }
    for cid, bb in layout.items():
        kicad.components[cid].bbox = bb
        # 同步初始状态，保证 env.reset() 恢复到碰撞自由布局
        kicad._initial_state[cid]["bbox"] = bb
    return env


def test_generated_transitions_no_negative_reward():
    """回归测试：生成的专家 transition 不得含负 reward（硬重叠违规）。

    此前因 _is_legal_placement 只检查已放置元件，先放置的元件变异后侵入
    尚未放置元件空间导致漏检，产生负 reward 坏样本拖垮 with_expert 训练。
    改为全局硬重叠检查后（且只拦截硬重叠、允许软间距），所有 transition
    的 reward 必须 > 0。

    使用碰撞自由的基准布局（_setup_collision_free_env），高变异率下
    充分覆盖"变异翻转为硬重叠 → 全局检查拦截并回退 base"路径。
    """
    env = _setup_collision_free_env()
    generator = ExpertTrajectoryGenerator(
        epsilon_decompose=1.0,
        pos_noise_prob=0.5,   # 高变异率，增加碰撞概率，压力测试
        rot_noise_prob=0.3,
    )
    # 生成多轨迹（高变异下充分覆盖漏检路径）
    all_transitions = []
    for _ in range(20):
        chain = generator.decompose_backward(env)
        all_transitions.extend(generator.trajectory_to_transitions(env, chain)[0])

    n_negative = sum(1 for t in all_transitions if t[2] < 0.0)
    assert n_negative == 0, \
        f"Found {n_negative} negative-reward transitions out of {len(all_transitions)}"
    logger.info(
        f"test_generated_transitions_no_negative_reward: PASSED "
        f"({len(all_transitions)} transitions, 0 negative)"
    )


# ===================== 测试 3.5: 坐标量化一致性 =====================

def test_quantize_bbox_to_grid():
    """量化 bbox 与 reward 重建公式一致（全程统一量化坐标）。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator()

    va = env.valid_area
    cell = Config.CELL_SIZE
    # 连续坐标 bbox（含非网格对齐小数位）
    bbox = (va[0] + 10.35 * cell, va[1] + 5.62 * cell,
            va[0] + 10.35 * cell + 1.0, va[1] + 5.62 * cell + 1.0)

    gx1, gy1, qbbox = generator._quantize_bbox_to_grid(env, bbox)

    # 量化公式：gx1 = round((x1 - va[0]) / cell)，clamp 到 [0, grid-1]
    expected_gx1 = int(round((bbox[0] - va[0]) / cell))
    expected_gy1 = int(round((bbox[1] - va[1]) / cell))
    assert gx1 == expected_gx1
    assert gy1 == expected_gy1
    # 量化 bbox 重建公式与 _compute_expert_reward 一致
    assert abs(qbbox[0] - (va[0] + gx1 * cell)) < 1e-9
    assert abs(qbbox[1] - (va[1] + gy1 * cell)) < 1e-9
    # 宽高保持（= footprint 尺寸）
    assert abs((qbbox[2] - qbbox[0]) - (bbox[2] - bbox[0])) < 1e-9
    assert abs((qbbox[3] - qbbox[1]) - (bbox[3] - bbox[1])) < 1e-9

    logger.info("test_quantize_bbox_to_grid: PASSED")


def test_quantized_bbox_is_grid_aligned():
    """量化 bbox 的坐标必须落在网格上（CELL_SIZE 整数倍偏移）。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator()

    va = env.valid_area
    cell = Config.CELL_SIZE
    # 故意给非网格对齐的连续坐标
    bbox = (va[0] + 3.7 * cell, va[1] + 7.3 * cell,
            va[0] + 3.7 * cell + 2.0, va[1] + 7.3 * cell + 2.0)

    _, _, qbbox = generator._quantize_bbox_to_grid(env, bbox)
    # 量化后 x1/y1 必须网格对齐（相对 valid_area 是 CELL_SIZE 整数倍）
    assert abs((qbbox[0] - va[0]) / cell - round((qbbox[0] - va[0]) / cell)) < 1e-6
    assert abs((qbbox[1] - va[1]) / cell - round((qbbox[1] - va[1]) / cell)) < 1e-6

    logger.info("test_quantized_bbox_is_grid_aligned: PASSED")


def test_quantize_clamps_to_grid_bounds():
    """量化坐标越界时 clamp 到网格范围 [0, grid-1]。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator()

    va = env.valid_area
    cell = Config.CELL_SIZE
    # 远超右/下边界的连续坐标
    bbox = (va[2] + 50, va[3] + 50, va[2] + 50 + 2.0, va[3] + 50 + 2.0)

    gx1, gy1, qbbox = generator._quantize_bbox_to_grid(env, bbox)
    assert gx1 == env.grid_width - 1
    assert gy1 == env.grid_height - 1
    assert qbbox[0] <= va[2]
    assert qbbox[1] <= va[3]

    logger.info("test_quantize_clamps_to_grid_bounds: PASSED")


def test_trajectory_uses_quantized_bbox():
    """trajectory_to_transitions 产出的 transition bbox 必须网格对齐。

    回归测试：此前变异产生连续坐标 bbox 直接写入 env 状态，而 reward/next_obs
    用量化坐标，两套坐标不一致导致软间距边界翻转 → 坏专家样本（reward≈1~4）。
    修复后全程统一量化坐标，transition 中元件 bbox 必须网格对齐。
    """
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        epsilon_decompose=1.0,
        pos_noise_prob=0.2,   # 启用位置变异，覆盖量化路径
        rot_noise_prob=0.1,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)[0]

    va = env.valid_area
    cell = Config.CELL_SIZE

    # 检查所有 step4 的 next_obs（放置后状态）中已放置元件 bbox 是否网格对齐
    for t in transitions:
        if t[6] != 4:   # step_id == 4
            continue
        next_obs = t[3]
        slots = next_obs["component_slots"]
        for i in range(len(slots)):
            if slots[i][0] > 0.5:   # placed_flag
                x1, y1, x2, y2 = slots[i][1], slots[i][2], slots[i][3], slots[i][4]
                # bbox 坐标相对 valid_area 应为 CELL_SIZE 整数倍
                for v, base in [(x1, va[0]), (y1, va[1])]:
                    offset = (v - base) / cell
                    assert abs(offset - round(offset)) < 1e-3, \
                        f"bbox not grid-aligned: ({x1},{y1},{x2},{y2}) offset={offset}"
                # 宽高为正（footprint 尺寸）
                assert x2 > x1 and y2 > y1

    logger.info("test_trajectory_uses_quantized_bbox: PASSED")


# ===================== 测试 4: Transition 构造 =====================

def test_trajectory_to_transitions():
    """验证轨迹 → transition 转换的正确性。"""
    env = create_patched_env()
    generator = ExpertTrajectoryGenerator(
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,  # 无变异，便于验证
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)[0]

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
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)[0]

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
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)[0]
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
        epsilon_decompose=1.0,
        pos_noise_prob=0.0,
        rot_noise_prob=0.0,
    )

    chain = generator.decompose_backward(env)
    transitions = generator.trajectory_to_transitions(env, chain)[0]

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
            total_transitions=50,
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


# ===================== 测试 9: raw 轨迹缓存结构 =====================

def test_raw_trajectory_cache():
    """验证便捷函数同时产出 raw 轨迹缓存，且结构符合 schema。

    raw 轨迹每步含 comp_id / target / base / mut_type / reward_full；
    轨迹级含 shift 与 total_reward（== sum(reward_full)）。
    """
    env = create_patched_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_and_save_expert_transitions(
            env=env,
            pcb_name="test_mock.kicad_pcb",
            output_dir=tmpdir,
            total_transitions=60,
            epsilon_decompose=1.0,
            pos_noise_prob=0.3,
            rot_noise_prob=0.2,
        )
        raw_path = os.path.join(tmpdir, "test_mock_expert_raw.json")
        assert os.path.isfile(raw_path), f"Raw cache not created: {raw_path}"
        raw = ExpertTrajectoryGenerator.load_raw_trajectories(raw_path)
        assert len(raw) > 0

        for traj in raw:
            assert "shift" in traj and len(traj["shift"]) == 2
            assert "steps" in traj and len(traj["steps"]) == env.N_movable
            assert "total_reward" in traj
            # total_reward 必须等于各步 reward_full 之和（拆分前完整 reward）
            assert abs(traj["total_reward"]
                       - sum(s["reward_full"] for s in traj["steps"])) < 1e-6
            for s in traj["steps"]:
                assert s["comp_id"]
                assert set(s["target"].keys()) == {"bbox", "rotation", "layer"}
                assert set(s["base"].keys()) == {"bbox", "rotation", "layer"}
                assert s["mut_type"] in ("none", "pos", "rot", "both")
                assert isinstance(s["reward_full"], float)

    logger.info("test_raw_trajectory_cache: PASSED")


# ===================== 测试 10: 专家 has_violations 含软间距（对齐锁） =====================

def test_expert_has_violations_includes_soft():
    """回归锁：专家 has_violations 必须把软间距(soft>0)计入违规，与训练侧
    env.step (L381-386) 同口径。

    否则未来某板专家布局含紧贴时，专家/训练侧 reward 标准会分叉（2026-09-05
    决策：推翻方案Y"不把 soft 计入 has_violations"）。用 mock 直接控制
    _detect_collisions 返回值，隔离软间距变量。
    """
    env = _setup_collision_free_env()
    generator = ExpertTrajectoryGenerator()
    # 第一个 movable 元件（U1，已在 _setup_collision_free_env 设好 bbox）
    comp_idx = 0
    comp = env.movable_list[comp_idx]
    cell = Config.CELL_SIZE
    va = env.valid_area
    x1, y1, x2, y2 = comp.bbox
    gx1 = int(round((x1 - va[0]) / cell))
    gy1 = int(round((y1 - va[1]) / cell))
    expert_actions = [comp_idx, gx1, gy1, comp.rotation, comp.layer]

    # 基准：无违规（soft=0）
    with mock.patch.object(env, "_detect_collisions",
                           return_value=(0.0, 0.0, 0.0, False)):
        reward_clean = generator._compute_expert_reward(env, comp_idx, expert_actions)
    # 紧贴：软间距>0，但无硬重叠/出界/keepout
    with mock.patch.object(env, "_detect_collisions",
                           return_value=(0.0, 1.5, 0.0, False)):
        reward_soft = generator._compute_expert_reward(env, comp_idx, expert_actions)

    # 含软间距时必须视为违规（更差），且差额至少含固定失败成本
    assert reward_soft < reward_clean, \
        f"Soft spacing must count as violation: clean={reward_clean}, soft={reward_soft}"
    assert reward_clean - reward_soft >= Config.W_VIOLATION_FLAT - 1e-6, \
        f"Soft violation should cost >= W_VIOLATION_FLAT({Config.W_VIOLATION_FLAT}): " \
        f"diff={reward_clean - reward_soft}"
    logger.info(
        f"test_expert_has_violations_includes_soft: PASSED "
        f"(clean={reward_clean:.3f}, soft={reward_soft:.3f})"
    )


# ===================== 运行全部测试 =====================

if __name__ == "__main__":
    test_decompose_backward()
    test_decompose_order_strategy()
    test_mutate_component_state()
    test_is_legal_placement()
    test_is_legal_placement_global_check()
    test_generated_transitions_no_negative_reward()
    test_trajectory_to_transitions()
    test_save_load_transitions()
    test_expert_replay_buffer()
    test_bc_pretrain_loss_decrease()
    test_generate_and_save()
    test_raw_trajectory_cache()
    test_expert_has_violations_includes_soft()
    logger.info("=" * 50)
    logger.info("All expert trajectory tests PASSED!")
