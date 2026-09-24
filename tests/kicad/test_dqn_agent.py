# -*- coding: utf-8 -*-
"""`env/dqn_agent.py`（Double DQN 本体，台账 §9.2）的回归测试。

**不需要 KiCad、不需要板缓存**（全用合成的小尺寸 obs，`16×16` ⇒ 秒级）。

覆盖：
  A. 打包/解包：`L` 逐位无损；`G` 只有 1/255 量化误差；形状与通道序（`c = rot·2 + (0=L,1=G)`）不变
  B. **内存核算**：一条经验 380,025 B（@200×200）、容量 10,000 = 3.80 GB —— 与 Owner 2026-09-23 的讨论一致
  C. 回放：写入/采样形状与取值自洽；环形覆盖（`len` 封顶、只剩最后 capacity 条）
  D. **Double DQN 的 TD 目标**：动作由**在线网**选、价值由**目标网**评 ⇒ 与 vanilla（目标网自选自评）**可区分**
  E. 无合法动作（终局那一帧 `mask` 全 False）⇒ 那一行 `y = r`（**不许**出现 `−inf` 或 NaN）
  F. 一次更新：损失有限、参数真的变了；目标网络**硬同步**只在 `steps % target_sync == 0` 时发生
  G. ε 计划：`steps=0` → `eps_start`、退火到 `eps_end`；`explore=False` ⇒ 纯贪心
  H. 断点：`state_dict`/`load_state_dict` 往返后同输入同输出
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from env import dqn_agent as da          # noqa: E402
from env import policy_net as pn         # noqa: E402

H = W = 16                               # 小尺寸（÷4 整除、H·W 可被 8 整除）
CH = pn.CHANNELS


def _spatial(rng, h=H, w=W):
    """造一个合法的 spatial：L 通道 ∈ {0,1}、G 通道 ∈ [0,1]（非法格按 §5.5 净化为 0 的情形也覆盖）。"""
    sp = np.zeros((CH, h, w), dtype=np.float32)
    sp[0::2] = (rng.random((CH // 2, h, w)) < 0.4).astype(np.float32)
    sp[1::2] = rng.random((CH // 2, h, w)).astype(np.float32)
    sp[1::2][sp[0::2] < 0.5] = 0.0        # G 的非法格 = 0（与 sanitize_gain 一致）
    return sp


def _obs(rng, h=H, w=W):
    return {"spatial": _spatial(rng, h, w), "piece": rng.random(3).astype(np.float32)}


def _mask(rng, h=H, w=W):
    m = rng.random((pn.N_ROT, h, w)) < 0.3
    if not m.any():
        m[0, 0, 0] = True
    return m


class _Stub(nn.Module):
    """返回固定 Q 图的替身网（`(B,4,H,W)`），用于把 Double DQN 的目标算清。"""

    def __init__(self, q):
        super().__init__()
        self.q = torch.as_tensor(q, dtype=torch.float32)
        self.dummy = nn.Parameter(torch.zeros(1))       # 占位：让 Adam 有参数可用（替身网本身无参数）

    def forward(self, spatial, piece):
        return self.q.unsqueeze(0).expand(spatial.shape[0], *self.q.shape)


# ---------------- A. 打包/解包 ----------------

def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    sp = _spatial(rng)
    Lb, Gu = da.pack_spatial(sp)
    assert Lb.shape == (pn.N_ROT, H * W // 8) and Lb.dtype == np.uint8
    assert Gu.shape == (pn.N_ROT, H, W) and Gu.dtype == np.uint8
    back = da.unpack_spatial(Lb, Gu)
    assert np.array_equal(back[0::2], sp[0::2]), "L 通道必须逐位无损"
    assert np.abs(back[1::2] - sp[1::2]).max() <= 1.0 / 255.0 + 1e-7, "G 通道只允许 1/255 量化误差"


def test_pack_rejects_bad_input():
    with pytest.raises(ValueError, match="偶数"):
        da.pack_spatial(np.zeros((3, H, W), np.float32))
    with pytest.raises(ValueError, match="整除"):
        da.pack_spatial(np.zeros((8, 3, 5), np.float32))
    bad = np.zeros((CH, H, W), np.float32)
    bad[1, 0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        da.pack_spatial(bad)


# ---------------- B. 内存核算 ----------------

def test_per_transition_bytes_matches_the_discussed_numbers():
    """@200×200：一条 **380,025 B**、容量 10,000 = **3.80 GB**（Owner 2026-09-23 定容量的依据）。"""
    r = da.PackedReplay(capacity=1, shape=(8, 200, 200))
    assert r.per_transition_bytes == 380_025
    assert da.DEFAULT_CAPACITY == 10_000
    total = r.per_transition_bytes * da.DEFAULT_CAPACITY
    assert abs(total / 1e9 - 3.80) < 0.01, f"容量 10,000 应为 3.80 GB，实得 {total / 1e9:.2f} GB"
    assert abs(r.nbytes - r.per_transition_bytes) < 100, "预分配数组的总字节应与逐条核算一致"


# ---------------- C. 回放 ----------------

def test_replay_add_sample_roundtrip():
    rng = np.random.default_rng(1)
    buf = da.PackedReplay(capacity=8, shape=(CH, H, W))
    keep = []
    for t in range(5):
        obs, nxt = _obs(rng), _obs(rng)
        act, r, mask, done = (t % 4, t % H, (t * 3) % W), 0.5 - t, _mask(rng), (t == 4)
        buf.add(obs, act, r, nxt, mask, done)
        keep.append((obs, act, r, nxt, mask, done))
    assert len(buf) == 5

    b = buf.sample(5, rng)
    assert b["spatial"].shape == (5, CH, H, W) and b["next_spatial"].shape == (5, CH, H, W)
    assert b["next_mask"].shape == (5, pn.N_ROT, H, W) and b["next_mask"].dtype == bool
    # 采样是**有放回**的 ⇒ 用 reward 当唯一钥匙把每条采样对回原始经验（本例 reward 各不相同）
    by_r = {round(r, 6): (obs, act, r, nxt, mask, done) for obs, act, r, nxt, mask, done in keep}
    for i in range(5):
        obs, act, r, nxt, mask, done = by_r[round(float(b["reward"][i]), 6)]
        assert np.array_equal(b["spatial"][i][0::2], obs["spatial"][0::2])          # L 无损
        assert np.abs(b["spatial"][i][1::2] - obs["spatial"][1::2]).max() <= 1 / 255 + 1e-7
        assert np.array_equal(b["next_spatial"][i][0::2], nxt["spatial"][0::2])
        assert np.array_equal(b["next_mask"][i], mask)
        assert bool(b["done"][i]) == done
        k, ii, jj = act
        assert b["action"][i] == k * H * W + ii * W + jj


def test_replay_wraps_around():
    rng = np.random.default_rng(2)
    buf = da.PackedReplay(capacity=3, shape=(CH, H, W))
    for t in range(7):
        buf.add(_obs(rng), (0, 0, 0), float(t), _obs(rng), _mask(rng), False)
    assert len(buf) == 3
    seen = set()                                                       # 采样有放回 ⇒ 多抽几轮看全集
    for _ in range(30):
        seen |= set(buf.sample(3, rng)["reward"].tolist())
    assert seen == {4.0, 5.0, 6.0}, f"环形覆盖后只应剩最后 3 条，实得 {sorted(seen)}"


def test_sampling_is_uniform_no_priority():
    """Owner 2026-09-23：**不做 PER** ⇒ 采样必须均匀、回放里不该有优先级/权重字段。

    固定种子 ⇒ 本测试是确定性的（不是概率性通过/失败）。
    """
    rng = np.random.default_rng(11)
    buf = da.PackedReplay(capacity=4, shape=(CH, H, W))
    for t in range(4):
        buf.add(_obs(rng), (0, 0, 0), float(t), _obs(rng), _mask(rng), False)

    counts = [0, 0, 0, 0]
    for _ in range(4_000):
        counts[int(round(float(buf.sample(1, rng)["reward"][0])))] += 1
    for i, c in enumerate(counts):
        assert abs(c - 1_000) <= 150, f"槽 {i} 被采 {c} 次（期望 ~1000）⇒ 采样不像是均匀的（上 PER 了？）"
    assert not any("priorit" in a.lower() or "weight" in a.lower() for a in dir(buf)), \
        "回放里不该出现优先级/重要性权重字段（Owner：暂不做 PER）"


def test_sample_before_enough_data_raises():
    buf = da.PackedReplay(capacity=4, shape=(CH, H, W))
    with pytest.raises(ValueError, match="采不出"):
        buf.sample(2, np.random.default_rng(0))


# ---------------- D/E. Double DQN 的 TD 目标 ----------------

def test_td_target_is_double_dqn_not_vanilla():
    """动作由**在线网**选、价值由**目标网**评 ⇒ 与"目标网自选自评"结果不同（判别力守门）。"""
    rng = np.random.default_rng(3)
    q_on = rng.normal(size=(pn.N_ROT, H, W)).astype(np.float32)
    q_tg = rng.normal(size=(pn.N_ROT, H, W)).astype(np.float32)
    mask = np.zeros((pn.N_ROT, H, W), dtype=bool)
    mask[1, 2, 3] = mask[0, 9, 9] = mask[3, 5, 5] = True

    agent = da.DoubleDQNAgent(net=_Stub(q_on), target=_Stub(q_tg), capacity=4,
                              shape=(CH, H, W), gamma=0.5)
    next_spatial = torch.zeros(1, CH, H, W)
    next_piece = torch.zeros(1, 3)
    y = agent.td_target(next_spatial, next_piece,
                        torch.as_tensor(mask)[None], torch.tensor([2.0]), torch.tensor([False]))

    cells = [(1, 2, 3), (0, 9, 9), (3, 5, 5)]
    a_on = max(cells, key=lambda c: q_on[c])       # 在线网选动作
    a_tg = max(cells, key=lambda c: q_tg[c])       # vanilla 会选的目标网 argmax
    assert a_on != a_tg, "样本没造好：两个 argmax 应不同，否则这条测试没有判别力"
    assert float(y[0]) == pytest.approx(2.0 + 0.5 * float(q_tg[a_on]))
    assert float(y[0]) != pytest.approx(2.0 + 0.5 * float(q_tg[a_tg]))


def test_td_target_handles_empty_next_mask():
    """终局那一帧 `mask` 全 False ⇒ `y = r`（不许 −inf / NaN）。"""
    agent = da.DoubleDQNAgent(capacity=4, shape=(CH, H, W), gamma=1.0)
    y = agent.td_target(torch.zeros(2, CH, H, W), torch.zeros(2, 3),
                        torch.zeros(2, pn.N_ROT, H, W, dtype=torch.bool),
                        torch.tensor([1.5, -0.25]), torch.tensor([True, True]))
    assert torch.isfinite(y).all()
    assert y.tolist() == pytest.approx([1.5, -0.25])


def test_done_zeroes_the_bootstrap():
    agent = da.DoubleDQNAgent(capacity=4, shape=(CH, H, W), gamma=1.0)
    mask = torch.ones(2, pn.N_ROT, H, W, dtype=torch.bool)
    y = agent.td_target(torch.zeros(2, CH, H, W), torch.zeros(2, 3), mask,
                        torch.tensor([1.0, 1.0]), torch.tensor([False, True]))
    assert y[1].item() == pytest.approx(1.0), "done=True 时不该加 bootstrap 项"


# ---------------- F. 更新与硬同步 ----------------

def _tiny_agent(**kw):
    kw.setdefault("capacity", 32)
    kw.setdefault("shape", (CH, H, W))
    kw.setdefault("batch_size", 8)
    kw.setdefault("target_every_episodes", 2)
    return da.DoubleDQNAgent(seed=0, **kw)


def _fill(agent, n, rng):
    for _ in range(n):
        agent.observe(_obs(rng), (0, 1, 2), 0.1, _obs(rng), _mask(rng), False)


def test_update_returns_finite_loss_and_changes_params():
    rng = np.random.default_rng(4)
    agent = _tiny_agent()
    _fill(agent, 12, rng)
    before = [p.detach().clone() for p in agent.net.parameters()]
    loss = agent.update()
    assert loss is not None and np.isfinite(loss)
    changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, agent.net.parameters()))
    assert changed, "一次更新后参数应当变化"
    assert agent.updates == 1


def test_update_returns_none_before_enough_data():
    agent = _tiny_agent()
    _fill(agent, 3, np.random.default_rng(5))
    assert agent.update() is None


def test_target_network_hard_sync_per_episodes():
    """Owner 2026-09-23：**每 K 个 episode 硬同步一次**；同一 episode 内多次 `update()` 只同步一次。"""
    rng = np.random.default_rng(6)
    agent = _tiny_agent(target_every_episodes=2)
    _fill(agent, 12, rng)

    def perturb():
        with torch.no_grad():
            for p in agent.net.parameters():
                p.add_(0.1)

    def same():
        return all(torch.equal(a, b) for a, b in zip(agent.net.parameters(), agent.target.parameters()))

    perturb()
    agent.episodes = 1                                     # 只过了 1 个 episode ⇒ 不同步
    agent.update()
    assert not same(), "未满 K 个 episode 不该同步目标网"

    agent.episodes = 2                                     # 满 2 个 ⇒ 同步
    agent.update()
    assert same(), "满 K 个 episode 应当硬同步"

    perturb()                                              # 同一 episode 内反复 update ⇒ 不再跟随
    agent.update()
    agent.update()
    assert not same(), "同一 episode 内只应同步一次"


def test_episode_counter_advances_on_done():
    rng = np.random.default_rng(10)
    agent = _tiny_agent()
    agent.observe(_obs(rng), (0, 0, 0), 0.0, _obs(rng), _mask(rng), False)
    assert agent.episodes == 0
    agent.observe(_obs(rng), (0, 0, 0), 0.0, _obs(rng), _mask(rng), True)
    assert agent.episodes == 1 and agent.steps == 2
    agent.end_episode()                                    # 也可由训练循环显式调
    assert agent.episodes == 2


# ---------------- G. ε 计划与动作合法性 ----------------

def test_epsilon_curve_is_smooth_exponential_between_bounds():
    """Owner 2026-09-23：`ε(ep) = 0.1 + 0.8·exp(−ep/τ)` —— 起点 0.9、单调下降、只逼近 0.1。"""
    agent = _tiny_agent(eps_top=0.9, eps_bottom=0.1, eps_tau_episodes=100.0)
    assert agent.epsilon() == pytest.approx(0.9), "第 0 个 episode 应恰为 e_top"
    vals = []
    for ep in (0, 10, 50, 100, 500, 2_000):
        agent.episodes = ep
        vals.append(agent.epsilon())
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), "必须单调下降"
    assert all(0.1 <= v <= 0.9 for v in vals), "必须落在 [e_bottom, e_top]"

    agent.episodes = 100
    assert agent.epsilon() == pytest.approx(0.1 + 0.8 * math.exp(-1.0))     # 与公式逐点一致
    agent.episodes = 100_000
    # 极限值 = e_bottom（x 很大时 exp(−x) 在浮点下下溢到 0 ⇒ 恰好取到 e_bottom，不会越过）
    assert 0.1 <= agent.epsilon() <= 0.1 + 1e-6, "只逼近 e_bottom，不越过"


def test_greedy_mode_ignores_epsilon():
    """`explore=False` ⇒ 纯贪心（与 `masked_argmax` 逐位一致），不受当前 ε 影响。"""
    agent = _tiny_agent()
    rng = np.random.default_rng(7)
    obs, mask = _obs(rng), _mask(rng)
    agent.episodes = 0                     # ε = 0.9（最大探索）
    got = agent.act(obs, mask, explore=False)
    assert got == pn.masked_argmax(agent.q_values(obs)[0], mask)


def test_act_stays_inside_mask_even_with_wild_q():
    """替身网把 mask 外全设成 +1e6 ⇒ `act` 仍必须选 mask 内（**探索时也一样**）。"""
    q = np.zeros((pn.N_ROT, H, W), dtype=np.float32)
    rng = np.random.default_rng(8)
    mask = np.zeros((pn.N_ROT, H, W), dtype=bool)
    mask[2, 4, 5] = mask[0, 1, 1] = True
    q[~mask] = 1e6
    agent = da.DoubleDQNAgent(net=_Stub(q), capacity=8, shape=(CH, H, W),
                              eps_top=1.0, eps_bottom=1.0, seed=0)
    for _ in range(30):
        k, i, j = agent.act(_obs(rng), mask)
        assert mask[k, i, j]


# ---------------- H. 断点 ----------------

def test_state_dict_roundtrip():
    rng = np.random.default_rng(9)
    a1 = _tiny_agent()
    _fill(a1, 12, rng)
    a1.update()
    sd = a1.state_dict()

    a2 = _tiny_agent()
    a2.load_state_dict(sd)
    obs = _obs(rng)
    assert torch.equal(a1.q_values(obs), a2.q_values(obs))
    assert (a2.steps, a2.updates) == (a1.steps, a1.updates)
