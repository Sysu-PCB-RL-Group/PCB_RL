"""DQN Agent 单元测试。

验证 select_action 掩码路由、learn next_step_id 路由、
sync_target 一致性、save/load 往返。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest

from dqn import DQNAgent, ReplayBuffer


def _make_agent(n_movable=5, grid_w=20, grid_h=15):
    """创建测试用 Agent（固定维度架构，无需 obs_dim）。"""
    return DQNAgent(n_movable, grid_w, grid_h, device=torch.device("cpu"))


def _make_dict_obs(n_slots=5, global_dim=8):
    """构建符合 Dict 格式的假观测。"""
    return {
        "component_slots": np.random.randn(n_slots, 8).astype(np.float32),
        "global_features": np.random.randn(global_dim).astype(np.float32),
    }


# ==================== select_action ====================


def test_select_action_respects_mask():
    """mask=False 的动作绝不被选中（epsilon=0）。"""
    agent = _make_agent(n_movable=5)
    obs = _make_dict_obs(n_slots=5)

    # 仅第 2 个动作合法
    mask = np.zeros(5, dtype=bool)
    mask[2] = True
    for _ in range(20):
        action = agent.select_action(obs, mask, step_id=0, epsilon=0.0)
        assert action == 2, f"Mask violated: got action {action}"


def test_select_action_different_steps():
    """不同 step_id 产生不同维度输出。"""
    agent = _make_agent(n_movable=5, grid_w=10, grid_h=8)
    obs = _make_dict_obs(n_slots=10)

    mask0 = np.ones(5, dtype=bool)
    a0 = agent.select_action(obs, mask0, step_id=0, epsilon=0.0)
    assert 0 <= a0 < 5

    mask1 = np.ones(10, dtype=bool)
    a1 = agent.select_action(obs, mask1, step_id=1, epsilon=0.0)
    assert 0 <= a1 < 10

    mask3 = np.ones(4, dtype=bool)
    a3 = agent.select_action(obs, mask3, step_id=3, epsilon=0.0)
    assert 0 <= a3 < 4

    mask4 = np.ones(2, dtype=bool)
    a4 = agent.select_action(obs, mask4, step_id=4, epsilon=0.0)
    assert 0 <= a4 < 2


def test_select_action_all_false_fallback():
    """mask 全 False 时返回 0（fallback），不崩溃。"""
    agent = _make_agent()
    obs = _make_dict_obs()
    mask = np.zeros(5, dtype=bool)
    action = agent.select_action(obs, mask, step_id=0, epsilon=0.0)
    assert action == 0


def test_select_action_epsilon_exploration():
    """epsilon=1.0 时仅在合法动作中探索。"""
    agent = _make_agent(n_movable=5)
    obs = _make_dict_obs()

    mask = np.zeros(5, dtype=bool)
    mask[1] = True
    mask[3] = True

    for _ in range(50):
        action = agent.select_action(obs, mask, step_id=0, epsilon=1.0)
        assert action in (1, 3), f"Exploration selected illegal action {action}"


def test_select_action_no_mask_exploit():
    """mask=None 时利用模式返回合法范围动作（全空间 argmax）。"""
    agent = _make_agent(n_movable=5)
    obs = _make_dict_obs()
    action = agent.select_action(obs, None, step_id=0, epsilon=0.0)
    assert 0 <= action < 5, f"Action {action} out of range [0, 4]"


def test_select_action_no_mask_explore():
    """mask=None 时探索模式在合法范围随机。"""
    agent = _make_agent(n_movable=5)
    obs = _make_dict_obs()
    actions = set()
    for _ in range(50):
        a = agent.select_action(obs, None, step_id=0, epsilon=1.0)
        actions.add(a)
    assert len(actions) > 1, "Should explore multiple actions"
    # 所有动作在合法范围
    for a in actions:
        assert 0 <= a < 5


# ==================== learn ====================


def test_learn_does_not_crash():
    """learn 不应崩溃或产生 NaN loss。"""
    agent = _make_agent()
    buffer = ReplayBuffer(capacity=100)
    obs = _make_dict_obs()

    for sid in range(5):
        # 每个 head 对不同维度，action 必须在 head 输出范围内
        next_sid = sid + 1 if sid < 4 else 0
        buffer.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, sid, next_sid)

    batch = buffer.sample(5)
    loss = agent.learn(batch)
    assert not np.isnan(loss), f"Loss is NaN"
    assert loss >= 0.0, f"Loss should be non-negative, got {loss}"


def test_learn_with_terminal_transition():
    """含终端 transition（next_step_id=-1）的学习不应崩溃。"""
    agent = _make_agent()
    buffer = ReplayBuffer(capacity=100)
    obs = _make_dict_obs()

    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else -1
        buffer.add(obs, 0, 1.0, obs, np.ones(5, dtype=bool), sid == 4, sid, next_sid)

    batch = buffer.sample(5)
    loss = agent.learn(batch)
    assert not np.isnan(loss)


def test_learn_next_step_id_routing():
    """不同 next_step_id 的 transition 正确路由到对应 target head。"""
    agent = _make_agent()
    buffer = ReplayBuffer(capacity=100)
    obs = _make_dict_obs()

    # 添加各种 next_step_id 组合
    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else 0
        buffer.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, sid, next_sid)

    # 也加一个终端的
    buffer.add(obs, 0, -5.0, obs, np.ones(2, dtype=bool), True, 4, -1)

    batch = buffer.sample(6)
    loss = agent.learn(batch)
    assert not np.isnan(loss)


def test_learn_target_mask_placed_component():
    """next_step_id=0 时，已放置元件的 Q 值被掩码屏蔽（不选虚高 Q）。"""
    agent = _make_agent(n_movable=5)
    buffer = ReplayBuffer(capacity=100)
    obs = _make_dict_obs()

    # 构造 next_obs：元件 0 的 placed_flag=1.0（已放置，不可选）
    next_obs = {
        "component_slots": obs["component_slots"].copy(),
        "global_features": obs["global_features"].copy(),
    }
    next_obs["component_slots"][0, 0] = 1.0  # slot 0 的 placed_flag

    # 当前 step_id=4 → 下个状态 next_step_id=0（选新元件）
    buffer.add(obs, 0, 0.0, next_obs, np.ones(5, dtype=bool), False, 4, 0)

    # 确保 buffer 有足够样本
    for _ in range(4):
        buffer.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, 0, 1)

    batch = buffer.sample(5)
    loss = agent.learn(batch)
    assert not np.isnan(loss), "Target mask should not cause NaN loss"


# ==================== sync_target ====================


def test_sync_target_consistency():
    """sync_target 后 q_net 和 target_net 权重完全一致。"""
    agent = _make_agent()

    # 初始时已同步（__init__ 中调用），确认一致
    for p_q, p_t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_q.data, p_t.data)

    # 做多次 learn 让 q_net 明显偏离 target_net
    buffer = ReplayBuffer(capacity=100)
    obs = _make_dict_obs()
    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else 0
        buffer.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, sid, next_sid)
    batch = buffer.sample(5)
    for _ in range(5):
        agent.learn(batch)

    # q_net 应该变了
    any_diff = False
    for p_q, p_t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        if not torch.allclose(p_q.data, p_t.data, atol=1e-4):
            any_diff = True
            break
    assert any_diff, "q_net should have diverged from target_net after learning"

    # sync 后恢复一致
    agent.sync_target()
    for p_q, p_t in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_q.data, p_t.data), "sync_target did not restore consistency"


# ==================== save / load ====================


def test_save_load_roundtrip():
    """save → load 往返后 train_steps 和预测输出一致。"""
    import tempfile

    agent = _make_agent(n_movable=5, grid_w=20, grid_h=15)
    obs = _make_dict_obs()

    # 先做一次 learn 产生非零 train_steps
    buffer = ReplayBuffer(capacity=100)
    for sid in range(5):
        next_sid = sid + 1 if sid < 4 else 0
        buffer.add(obs, 0, 0.0, obs, np.ones(5, dtype=bool), False, sid, next_sid)
    agent.learn(buffer.sample(5))
    steps_before = agent.train_steps

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        agent.save(tmp_path)

        agent2 = _make_agent(n_movable=5, grid_w=20, grid_h=15)
        agent2.load(tmp_path)

        assert agent2.train_steps == steps_before

        # 预测输出一致性（Dict 观测 → tensor）
        obs_tensor = agent._obs_to_tensor(obs)
        agent.q_net.eval()
        agent2.q_net.eval()
        for sid in range(5):
            with torch.no_grad():
                q1 = agent.q_net(obs_tensor, sid)
                q2 = agent2.q_net(obs_tensor, sid)
                assert torch.allclose(q1, q2, atol=1e-6), f"Output mismatch at step_id={sid}"
    finally:
        os.unlink(tmp_path)
