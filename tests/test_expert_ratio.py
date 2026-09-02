"""专家组 buffer 配置冒烟测试（非 KiCad，mock 拦截 kicad_utils DLL）。

验证：
1. Config 从 argument.txt 加载后 EXPERT_BUFFER_RATIO == 0.7，且 EXPERT_TOTAL_TRANSITIONS
   由 EXPERT_TOTAL_BUFFER × EXPERT_BUFFER_RATIO 派生 == 70000；
2. ExpertReplayBuffer 容量由派生口径决定（expert=70000, normal=30000，合计=总buffer）；
3. sample() 对整池均匀随机：当池中专家占比≈0.7 时，每 batch 专家占比也≈0.7（统计容差，
   而非硬拆分）。
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


def test_config_ratio_and_transitions_derived():
    Config.load_from_file(_CONFIG_PATH)
    assert Config.EXPERT_BUFFER_RATIO == 0.7, (
        f"EXPERT_BUFFER_RATIO 应为 0.7，实际 {Config.EXPERT_BUFFER_RATIO}"
    )
    # 派生：生成目标 = 总 buffer × 专家占比 = 70000
    assert Config.EXPERT_TOTAL_TRANSITIONS == int(
        Config.EXPERT_TOTAL_BUFFER * Config.EXPERT_BUFFER_RATIO
    ), "EXPERT_TOTAL_TRANSITIONS 应派生为 总buffer×占比"
    assert Config.EXPERT_TOTAL_TRANSITIONS == 70000, (
        f"EXPERT_TOTAL_TRANSITIONS 应为 70000，实际 {Config.EXPERT_TOTAL_TRANSITIONS}"
    )


def test_buffer_capacity_derived():
    Config.load_from_file(_CONFIG_PATH)
    expert_capacity = Config.EXPERT_TOTAL_TRANSITIONS
    normal_capacity = Config.EXPERT_TOTAL_BUFFER - Config.EXPERT_TOTAL_TRANSITIONS
    buf = ExpertReplayBuffer(
        expert_capacity=expert_capacity, normal_capacity=normal_capacity,
        expert_ratio=Config.EXPERT_BUFFER_RATIO,
    )
    assert buf.expert_capacity == 70000
    assert buf.normal_capacity == 30000
    assert buf.expert_capacity + buf.normal_capacity == Config.EXPERT_TOTAL_BUFFER


def test_sample_expert_fraction_matches_pool():
    Config.load_from_file(_CONFIG_PATH)
    expert_capacity = Config.EXPERT_TOTAL_TRANSITIONS
    normal_capacity = Config.EXPERT_TOTAL_BUFFER - Config.EXPERT_TOTAL_TRANSITIONS
    buf = ExpertReplayBuffer(
        expert_capacity=expert_capacity, normal_capacity=normal_capacity,
        expert_ratio=Config.EXPERT_BUFFER_RATIO, per_alpha=0.6, per_beta=0.4,
    )
    # 池中专家占比≈0.7：注入 7000 专家 + 3000 探索
    buf.load_expert_transitions([_dummy_transition() for _ in range(7000)])
    for _ in range(3000):
        buf.add_normal(*_dummy_transition())

    batch_size = 64
    n_batches = 300
    fracs = []
    for _ in range(n_batches):
        buf.sample(batch_size)
        indices = buf._last_sampled_indices
        n_expert = sum(1 for i in indices if i < len(buf.expert_buffer))
        fracs.append(n_expert / batch_size)

    mean_frac = float(np.mean(fracs))
    # 均匀随机：样本专家占比应≈池中专家占比(0.7)，允许统计波动
    assert 0.65 <= mean_frac <= 0.75, f"平均专家占比 {mean_frac:.3f} 应≈0.7"
    # 且非每批硬 44：批次间应有波动（标准差合理）
    assert np.std(fracs) > 0.02, "均匀采样应有批次间波动，不应恒为硬 44/20"
