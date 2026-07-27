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


def _make_net(obs_dim=10, n_movable=5, grid_w=10, grid_h=8):
    return DQN(obs_dim, n_movable, grid_w, grid_h)


def test_forward_shapes():
    """每个 head 输出形状正确。"""
    n_movable, grid_w, grid_h = 5, 10, 8
    net = _make_net(obs_dim=50, n_movable=n_movable, grid_w=grid_w, grid_h=grid_h)
    x = torch.randn(3, 50)

    assert net(x, 0).shape == (3, n_movable)   # comp head
    assert net(x, 1).shape == (3, grid_w)       # x1 head
    assert net(x, 2).shape == (3, grid_h)       # y1 head
    assert net(x, 3).shape == (3, 4)            # rot head
    assert net(x, 4).shape == (3, 2)            # layer head


def test_head_independence():
    """不同 head 产生不同输出（非共享最后一层）。"""
    grid_w, grid_h = 10, 8
    net = _make_net(obs_dim=50, n_movable=grid_w, grid_w=grid_w, grid_h=grid_h)
    x = torch.randn(4, 50)
    # head 0 (comp, dim=n_movable) 与 head 1 (x1, dim=grid_w)
    # 当 n_movable == grid_w 时维度相同，可直接比较
    out0 = net(x, 0)
    out1 = net(x, 1)
    assert not torch.allclose(out0, out1, atol=1e-4)


def test_batch_independence():
    """同一 head、不同 batch 条目产生不同输出。"""
    net = _make_net(obs_dim=20)
    x = torch.randn(4, 20)
    out = net(x, 0)
    for i in range(3):
        assert not torch.allclose(out[i], out[i + 1], atol=1e-4)


def test_deterministic_eval():
    """eval 模式下相同输入 → 相同输出。"""
    net = _make_net(obs_dim=20)
    net.eval()
    x = torch.randn(1, 20)
    out1 = net(x, 0)
    out2 = net(x, 0)
    assert torch.allclose(out1, out2, atol=1e-8)


def test_gradient_flow():
    """训练模式下梯度正常传播。"""
    net = _make_net(obs_dim=20)
    net.train()
    x = torch.randn(3, 20)
    for sid in range(5):
        net.zero_grad()
        out = net(x, sid)
        loss = out.sum()
        loss.backward()
        # 检查 shared 层有梯度
        has_grad = False
        for name, param in net.named_parameters():
            if "shared" in name and param.grad is not None:
                has_grad = has_grad or param.grad.abs().sum() > 0
        assert has_grad, f"No gradient in shared layers for head {sid}"


def test_invalid_step_id():
    """无效 step_id 抛出 IndexError。"""
    net = _make_net()
    x = torch.randn(1, 10)
    with pytest.raises(IndexError):
        net(x, 5)


def test_obs_dim_auto_compute():
    """obs_dim 可通过 N_movable*8 + N_movable + 4 计算。"""
    n_movable = 10
    obs_dim = n_movable * 8 + n_movable + 4
    net = DQN(obs_dim, n_movable, 20, 15)
    x = torch.randn(2, obs_dim)
    assert net(x, 0).shape == (2, n_movable)
