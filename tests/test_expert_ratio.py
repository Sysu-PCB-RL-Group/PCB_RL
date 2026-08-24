"""EXPERT_BUFFER_RATIO=0.7 冒烟测试（非 KiCad，mock 拦截 kicad_utils DLL）。

验证：
1. Config 从 argument.txt 加载后 EXPERT_BUFFER_RATIO == 0.7；
2. ExpertReplayBuffer 容量公式不变（expert=20010, normal=80040）；
3. sample(64) 按 70/30 拆分：n_expert=44, n_normal=20。
"""
import os
import sys
from unittest.mock import MagicMock

# 必须在 import env 前 mock，避免加载 KiCad DLL
sys.modules["utils.kicad_utils"] = MagicMock()

import numpy as np  # noqa: E402
from env.pcb_layout_env import Config  # noqa: E402
from dqn.replay_buffer import ExpertReplayBuffer  # noqa: E402

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "argument.txt")


def _dummy_transition():
    obs = {
        "component_slots": np.zeros((Config.MAX_MOVABLE, 8), dtype=np.float32),
        "global_features": np.zeros(8, dtype=np.float32),
    }
    return (obs, 0, 0.0, obs, None, False, 0, 1)


def test_config_ratio_is_0_7():
    Config.load_from_file(_CONFIG_PATH)
    assert Config.EXPERT_BUFFER_RATIO == 0.7, (
        f"EXPERT_BUFFER_RATIO 应为 0.7，实际 {Config.EXPERT_BUFFER_RATIO}"
    )


def test_buffer_capacity_unchanged():
    buf = ExpertReplayBuffer(
        expert_capacity=20010, normal_capacity=80040,
        expert_ratio=Config.EXPERT_BUFFER_RATIO,
    )
    assert buf.expert_capacity == 20010
    assert buf.normal_capacity == 80040


def test_sample_split_70_30():
    buf = ExpertReplayBuffer(
        expert_capacity=20010, normal_capacity=80040,
        expert_ratio=Config.EXPERT_BUFFER_RATIO, per_alpha=0.6, per_beta=0.4,
    )
    # 注入少量 dummy（远小于容量，但覆盖采样所需最小值）
    buf.load_expert_transitions([_dummy_transition() for _ in range(100)])
    for _ in range(100):
        buf.add_normal(*_dummy_transition())

    batch_size = 64
    n_expert_expected = int(batch_size * Config.EXPERT_BUFFER_RATIO)  # 44
    n_normal_expected = batch_size - n_expert_expected             # 20

    buf.sample(batch_size)
    indices = buf._last_sampled_indices
    n_expert = sum(1 for i in indices if i < len(buf.expert_buffer))
    n_normal = len(indices) - n_expert

    assert len(indices) == batch_size
    assert n_expert == n_expert_expected, f"n_expert={n_expert}, 期望 {n_expert_expected}"
    assert n_normal == n_normal_expected, f"n_normal={n_normal}, 期望 {n_normal_expected}"
