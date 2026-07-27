"""ReplayBuffer 单元测试。

验证环形缓冲区读写、next_step_id 路由、采样完整性。
纯逻辑测试，不依赖 env 或模型。
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dqn.replay_buffer import ReplayBuffer


def _make_obs(dim: int = 10) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def test_add_and_len():
    """添加数据后 len 正确递增。"""
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    for i in range(10):
        buf.add(obs, i % 5, 1.0 if i % 5 == 4 else 0.0,
                obs, np.ones(5, dtype=bool), False, i % 5, (i % 5) + 1)
    assert len(buf) == 10


def test_capacity_overflow():
    """超过 capacity 后旧数据被覆盖。"""
    buf = ReplayBuffer(capacity=5)
    obs = _make_obs()
    for i in range(10):
        buf.add(obs, i, float(i), obs, np.ones(5, dtype=bool),
                False, i % 5, (i % 5) + 1)
    assert len(buf) == 5  # max is capacity


def test_is_ready():
    """buffer 满 BATCH_SIZE 后 is_ready=True。"""
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    batch_size = 8
    for i in range(batch_size - 1):
        buf.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, 0, 1)
    assert not buf.is_ready(batch_size)
    buf.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, 0, 1)
    assert buf.is_ready(batch_size)


def test_sample_shapes():
    """采样返回的 8 元组形状正确。"""
    buf = ReplayBuffer(capacity=100)
    obs_dim = 10
    obs = _make_obs(obs_dim)
    for i in range(20):
        sid = i % 5
        buf.add(obs, i % 3, 1.0 if sid == 4 else 0.0,
                obs, np.ones(5, dtype=bool), False, sid, sid + 1 if sid < 4 else 0)
    batch = buf.sample(8)
    assert batch[0].shape == (8, obs_dim)     # obs
    assert batch[1].shape == (8,)              # actions
    assert batch[2].shape == (8,)              # rewards
    assert batch[3].shape == (8, obs_dim)      # next_obs
    assert len(batch[4]) == 8                  # masks (list)


def test_next_step_id_routing():
    """step 0-3 的 next_step_id = sid+1，step 4 = 0（非终端）。
    采样随机，逐条验证 step_id → next_step_id 映射。
    """
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    mask = np.ones(5, dtype=bool)

    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else 0
        buf.add(obs, sid, 0.0, obs, mask, False, sid, next_sid)

    batch = buf.sample(5)
    step_ids = batch[6]
    next_sids = batch[7]

    expected = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
    for sid, nsid in zip(step_ids, next_sids):
        assert nsid == expected[int(sid)], f"step {int(sid)} → next {int(nsid)}, expected {expected[int(sid)]}"


def test_terminal_next_step_id():
    """终端 episode → step 4 的 next_step_id = -1。"""
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    mask = np.ones(5, dtype=bool)

    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else -1
        buf.add(obs, sid, 0.0, obs, mask, sid == 4, sid, next_sid)

    batch = buf.sample(5)
    step_ids = batch[6]
    next_sids = batch[7]

    expected = {0: 1, 1: 2, 2: 3, 3: 4, 4: -1}
    for sid, nsid in zip(step_ids, next_sids):
        assert nsid == expected[int(sid)], f"step {int(sid)} → next {int(nsid)}, expected {expected[int(sid)]}"


def test_reward_only_step4():
    """仅 step_id=4 的 transition 有非零 reward。"""
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    mask = np.ones(5, dtype=bool)
    for sid in range(5):
        step_reward = 10.0 if sid == 4 else 0.0
        next_sid = sid + 1 if sid < 4 else 0
        buf.add(obs, sid, step_reward, obs, mask, False, sid, next_sid)

    batch = buf.sample(5)
    for i, (sid, r) in enumerate(zip(batch[6], batch[2])):
        if sid == 4:
            assert r == 10.0, f"step 4 should have reward 10, got {r}"
        else:
            assert r == 0.0, f"step {sid} should have reward 0, got {r}"


def test_sample_randomness():
    """连续两次采样结果应不同（概率性检查）。"""
    buf = ReplayBuffer(capacity=100)
    obs = _make_obs()
    for i in range(50):
        sid = i % 5
        buf.add(obs, i, float(i), obs, np.ones(5, dtype=bool),
                False, sid, sid + 1 if sid < 4 else 0)
    b1 = buf.sample(10)
    b2 = buf.sample(10)
    # 两次采样的 action 不应该完全相同
    assert not np.array_equal(b1[1], b2[1])
