"""DQN 网络单元测试。

验证 5-Head 共享 MLP 的 forward shape、head 路由、梯度流。
纯模型测试，不依赖 env 或 agent。
"""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dqn.network import DQN


def _make_net(n_movable=5, grid_w=10, grid_h=8):
    return DQN(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h)


def _make_obs(batch=1, n_movable=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "component_slots": torch.tensor(rng.standard_normal((batch, n_movable, 8)).astype(np.float32)),
        "global_features": torch.tensor(rng.standard_normal((batch, 8)).astype(np.float32)),
    }


def test_forward_shapes():
    """每个 head 输出形状正确。"""
    n_movable, grid_w, grid_h = 5, 10, 8
    net = _make_net(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h)
    obs = _make_obs(batch=3, n_movable=n_movable)
    assert net(obs, 0).shape == (3, n_movable)   # comp head
    assert net(obs, 1).shape == (3, grid_w)       # x1 head
    assert net(obs, 2).shape == (3, grid_h)       # y1 head
    assert net(obs, 3).shape == (3, 4)            # rot head
    assert net(obs, 4).shape == (3, 2)            # layer head


def test_head_independence():
    """不同 head 产生不同输出（非共享最后一层）。"""
    grid_w, grid_h = 10, 8
    net = _make_net(n_movable=grid_w, grid_w=grid_w, grid_h=grid_h)
    obs = _make_obs(batch=4, n_movable=grid_w)
    # head 0 (comp, dim=n_movable) 与 head 1 (x1, dim=grid_w)
    # 当 n_movable == grid_w 时维度相同，可直接比较
    out0 = net(obs, 0)
    out1 = net(obs, 1)
    assert not torch.allclose(out0, out1, atol=1e-4)


def test_batch_independence():
    """同一 head、不同 batch 条目产生不同输出。"""
    net = _make_net()
    obs = _make_obs(batch=4, n_movable=5)
    out = net(obs, 0)
    for i in range(3):
        assert not torch.allclose(out[i], out[i + 1], atol=1e-4)


def test_deterministic_eval():
    """eval 模式下相同输入 → 相同输出。"""
    net = _make_net()
    net.eval()
    obs = _make_obs(batch=1, n_movable=5)
    out1 = net(obs, 0)
    out2 = net(obs, 0)
    assert torch.allclose(out1, out2, atol=1e-8)


def test_gradient_flow():
    """训练模式下梯度正常传播。"""
    net = _make_net()
    net.train()
    obs = _make_obs(batch=3, n_movable=5)
    for sid in range(5):
        net.zero_grad()
        out = net(obs, sid)
        loss = out.sum()
        loss.backward()
        # step 0 走 head_comp（不经 shared）；step 1-4 走 shared + 对应 head。
        # 断言：当前 head 前向确实产生了梯度（针对 step 0 不检查 shared）。
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters()
        )
        assert has_grad, f"No gradient propagated for head {sid}"


def test_invalid_step_id():
    """无效 step_id 抛出 IndexError。"""
    net = _make_net()
    obs = _make_obs(batch=1, n_movable=5)
    with pytest.raises(IndexError):
        net(obs, 5)


def test_obs_dim_auto_compute():
    """head 输出维度严格等于构造时给定的 n_movable/grid_w/grid_h。"""
    n_movable, grid_w, grid_h = 7, 12, 9
    net = _make_net(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h)
    obs = _make_obs(batch=2, n_movable=n_movable)
    assert net(obs, 0).shape[-1] == n_movable
    assert net(obs, 1).shape[-1] == grid_w
    assert net(obs, 2).shape[-1] == grid_h
    assert net(obs, 3).shape[-1] == 4
    assert net(obs, 4).shape[-1] == 2
