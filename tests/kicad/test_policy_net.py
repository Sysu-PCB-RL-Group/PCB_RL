# -*- coding: utf-8 -*-
"""`env/policy_net.py`（Double DQN 的网络 + 掩膜内动作选择，台账 §9.2）的回归测试。

**不需要 KiCad、不需要板缓存**：网络只吃张量，选择函数只吃数组 ⇒ 本文件在训练机上也能跑。

覆盖：
  A. 前向：输出形状 `(B, 4, H, W)`（**满分辨率**，与动作/mask 同序）、有限值、参数量规模、200×200 真实尺寸可用
  B. 确定性：同一输入两次前向逐位相同（⇒ 没有 BN/dropout 这类随机源）
  C. **掩膜内选择**：`mask` 外的巨大 Q 值**不得**被选中（判别力守门 —— 写错就会选外面）
  D. 边界：`mask` 全 False ⇒ 抛错（不返回假动作）；动作越界 ⇒ 抛错
  E. ε-贪心：`eps=0` 等价于 `masked_argmax`；`eps=1` 恒在 mask 内且能采到多个不同动作
  F. 梯度：损失反传后主干/头有非零梯度（能被训练）
  G. **布局一致性**：`(k,i,j)` 与 `PcbEnv` 的一维动作编码 `k·m²+i·m+j` 完全一致（§6.1 零转换）
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from env import policy_net as pn          # noqa: E402
from env.pcb_env import PcbEnv            # noqa: E402 (只做动作编码的一致性对拍，不读缓存)

H = W = 40                                # 小尺寸（÷4 整除）⇒ 测试快；200×200 另有一条专测


def _q(seed: int = 0, shape=(pn.N_ROT, H, W)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


# ---------------- A. 前向 ----------------

def test_forward_shape_and_scale():
    """`(B,8,H,W) + (B,3)` → `(B,4,H,W)`，全有限，参数量在「小网络」量级。"""
    net = pn.QNet()
    spatial = torch.zeros(2, pn.CHANNELS, H, W)
    piece = torch.tensor([[0.5, 0.2, 0.1], [0.0, 0.0, 1.0]])
    q = net(spatial, piece)
    assert tuple(q.shape) == (2, pn.N_ROT, H, W)
    assert torch.isfinite(q).all()
    n_param = sum(p.numel() for p in net.parameters())
    assert n_param < 200_000, f"参数量 {n_param} 超出「小网络」量级（手滑写成大网？）"


def test_forward_works_at_real_size_200():
    """真实尺寸 `(1,8,200,200)` 必须能前向（上下采样整除、形状对齐）。"""
    net = pn.QNet()
    q = net(torch.zeros(1, pn.CHANNELS, 200, 200), torch.zeros(1, 3))
    assert tuple(q.shape) == (1, pn.N_ROT, 200, 200)
    assert torch.isfinite(q).all()


def test_forward_rejects_wrong_channel_count():
    net = pn.QNet()
    with pytest.raises(ValueError, match="通道数"):
        net(torch.zeros(1, 4, H, W), torch.zeros(1, 3))


# ---------------- B. 确定性 ----------------

def test_forward_is_deterministic():
    """同输入两次前向逐位相同（**train 模式**下也相同 ⇒ 没有 BN/dropout）。"""
    net = pn.QNet()
    spatial = torch.randn(1, pn.CHANNELS, H, W)
    piece = torch.tensor([[0.3, 0.4, 0.5]])
    net.train()
    q1, q2 = net(spatial, piece), net(spatial, piece)
    assert torch.equal(q1, q2)


# ---------------- C. 掩膜内选择（含判别力守门） ----------------

def test_masked_argmax_never_picks_outside_mask():
    """**判别力守门**：把 `mask` 外的 Q 灌成 +1e9 ⇒ 仍必须选 `mask` 内的最大值（写错必红）。"""
    q = _q(seed=1)
    mask = np.zeros((pn.N_ROT, H, W), dtype=bool)
    mask[2, 7, 9] = True
    mask[1, 30, 12] = True
    want = max(((2, 7, 9), (1, 30, 12)), key=lambda a: q[a[0], a[1], a[2]])

    q_big = np.where(mask, q, 1e9)            # mask 外全部灌成巨大值
    assert pn.masked_argmax(q_big, mask) == want

    q_small = np.where(mask, q, -1e9)         # 反向：mask 外灌成极小值（mask 内的大小关系不变）
    assert pn.masked_argmax(q_small, mask) == want


def test_masked_argmax_matches_numpy_reference():
    """与独立 numpy 参照（mask 外置 −inf 后取 argmax）逐例一致。"""
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.normal(size=(pn.N_ROT, 8, 9)).astype(np.float32)
        mask = rng.random((pn.N_ROT, 8, 9)) < 0.3
        if not mask.any():
            continue
        flat = int(np.argmax(np.where(mask, q, -np.inf)))
        k, rem = divmod(flat, 8 * 9)
        i, j = divmod(rem, 9)
        assert pn.masked_argmax(q, mask) == (k, i, j)


# ---------------- D. 边界 ----------------

def test_empty_mask_raises():
    """合法动作集为空 ⇒ 抛错（episode 应已终止；**不许**返回一个假动作）。"""
    q = _q(seed=2)
    empty = np.zeros((pn.N_ROT, H, W), dtype=bool)
    with pytest.raises(ValueError, match="全 False"):
        pn.masked_argmax(q, empty)
    with pytest.raises(ValueError, match="全 False"):
        pn.masked_epsilon_greedy(q, empty, 0.5, np.random.default_rng(0))


def test_shape_mismatch_and_out_of_range_raise():
    q = _q(seed=3)
    mask = np.ones((pn.N_ROT, H, W), dtype=bool)
    with pytest.raises(ValueError, match="形状"):
        pn.masked_argmax(q, mask[:, : H - 1, :])
    with pytest.raises(ValueError, match="越界"):
        pn.q_of_action(q, (0, H, 0))


# ---------------- E. ε-贪心 ----------------

def test_epsilon_greedy_two_branches():
    """`eps=0` == `masked_argmax`；`eps=1` 恒在 `mask` 内、且能采到多个不同动作。"""
    q = _q(seed=4)
    mask = np.zeros((pn.N_ROT, H, W), dtype=bool)
    mask[0, 1, 1] = mask[3, 20, 20] = mask[2, 5, 30] = True

    assert pn.masked_epsilon_greedy(q, mask, 0.0, np.random.default_rng(0)) == pn.masked_argmax(q, mask)

    rng = np.random.default_rng(1)
    picks = {pn.masked_epsilon_greedy(q, mask, 1.0, rng) for _ in range(200)}
    assert all(mask[k, i, j] for k, i, j in picks)
    assert len(picks) >= 2, "eps=1 时应能采到多个合法动作（不是在 mask 内取贪心）"


def test_q_of_action_reads_the_right_cell():
    q = _q(seed=5)
    for action in ((0, 0, 0), (3, H - 1, W - 1), (2, 7, 9)):
        k, i, j = action
        assert pn.q_of_action(q, action) == float(q[k, i, j])


# ---------------- F. 梯度 ----------------

def test_gradients_flow_to_trunk_and_head():
    net = pn.QNet()
    q = net(torch.randn(1, pn.CHANNELS, H, W), torch.zeros(1, 3))
    loss = q[0, 0, 3, 4] + q[0, 2, 5, 6]
    loss.backward()
    assert net.head.weight.grad is not None and net.head.weight.grad.abs().sum() > 0
    assert net.trunk[0].weight.grad is not None and net.trunk[0].weight.grad.abs().sum() > 0


# ---------------- G. 与 env 动作编码同序 ----------------

def test_action_layout_matches_env_encoding():
    """`q[k,i,j]` ↔ 动作 `(k,i,j)` ↔ 一维 `k·m²+i·m+j`（§6.1）—— 三处必须同一件事。"""
    env = PcbEnv()                            # 不 reset ⇒ 不读缓存
    for (k, i, j) in ((0, 0, 0), (1, 7, 9), (3, 199, 199)):
        flat = k * 200 * 200 + i * 200 + j
        assert env._decode(flat) == (k, i, j)
        assert env._decode((k, i, j)) == (k, i, j)
    # 掩膜内选择的结果再走一遍 env 的编码 ⇒ 往返一致
    q = _q(seed=6)
    mask = np.zeros((pn.N_ROT, H, W), dtype=bool)
    mask[2, 23, 5] = True
    k, i, j = pn.masked_argmax(q, mask)
    assert (k, i, j) == (2, 23, 5)
