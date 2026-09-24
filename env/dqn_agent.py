# -*- coding: utf-8 -*-
"""Double DQN 本体：**打包回放** + TD 目标 + ε 计划 —— 台账 §9.2（Owner 2026-09-23："算法暂定为 double dqn"）。

为什么必须打包（实测，2026-09-23）
--------------------------------
新一代 obs 是整张 200×200 图：`spatial (8,200,200) float32` = **1.28 MB/步**，一条经验要 obs + next_obs
⇒ 原样 **2.56 MB/条**（容量 10,000 = **25.6 GB**，放不下）。打包后 **380 KB/条**（10,000 = **3.8 GB** ✅）：
```text
L 通道（本来就只有 0/1）→ np.packbits   : 4×200×200/8 = 20,000 B
G 通道（∈ [0,1]）      → uint8（×255）  : 4×200×200   = 160,000 B
mask(s')（同上按位）                     : 20,000 B
piece (3,) float32 + 动作/奖励/done      : ~30 B
```
⇒ **打包是无损/近无损的**（`L` 精确；`G` 量化误差 ≤ 1/255 ≈ 0.004，而 G 本就是分位归一后的 [0,1] 量）。
⚠️ 上一代能开 `REPLAY_BUFFER_SIZE = 100000` 是因为它的 obs 只有 **608 B**（`component_slots (18,8)` + 8 个全局特征），
不是"当年的 buffer 更聪明"。

Double DQN 的两个要点（与 vanilla DQN 的差别只在这里）
----------------------------------------------------
```text
a*  = argmax_{a ∈ mask(s')} Q_θ(s', a)        ← 用【在线网】选动作
y   = r + γ · Q_θ⁻(s', a*)                    ← 用【目标网】评估该动作（vanilla 是"目标网自己选+评估"）
```
`γ` 直接用 `env.reward.GAMMA = 1.0`（单点定义，不另设常数）；`done=True`（放完 ∨ mask 空）时 `y = r`。
**候选集 = 存入的 `mask(s')`**（不是 obs 里的 `L`）：lookahead 不进 obs，只有把 `mask` 一起存下来，
目标才与行为策略**完全同口径**（代价 +20 KB/条，占 380 KB 的 5%）。

容量与三个超参（Owner 2026-09-23）
-----------------------------------
```text
① 容量 **10,000**（"暂定 buffer = 10000 先吧"）＝ 3.80 GB（打包后）；单进程 19 步/s ⇒ 约 9 分钟填满
② 目标网络**每 `target_every_episodes` 个 episode 硬同步一次**（按 episode 记数，不按 env 步）
   —— Owner 2026-09-23 先拍 100，后因"只跑 100 局"改为 **20**（默认 20；CLI `--target-every-episodes` 可调）
③ ε 是**非线性平滑曲线**（Owner："改用非线性的平滑曲线，如 e^(−x)，e_top = 0.9、e_bottom = 0.1"）：
   `ε(ep) = e_bottom + (e_top − e_bottom)·exp(−ep / τ)`，τ = `eps_tau_episodes`（**我方自定 = 2,000**，可调）
   ⇒ `ε(0) = 0.9`、单调下降、`ep → ∞` 时趋于 `0.1`（永远 ≥ e_bottom）；探索始终只在 `mask` 内
④ 优化器 **Adam**（Owner："学习率使用 adam"）—— `lr = 1e-4`（与上一代 `config/argument.txt` 的 `LEARNING_RATE = 0.0001` 一致）
```
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from env import policy_net as pn
from env.reward import GAMMA

__all__ = ["PackedReplay", "DoubleDQNAgent", "pack_spatial", "unpack_spatial", "DEFAULT_CAPACITY"]

DEFAULT_CAPACITY = 10_000       # Owner 2026-09-23 暂定（3.8 GB @ 200×200）
DEFAULT_SHAPE = (8, 200, 200)   # spatial 的形状（§5.5）


# ===================== 打包 / 解包 =====================

def pack_spatial(spatial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`spatial (8,H,W)` → `(L_bits uint8 (4, H·W/8), G_u8 uint8 (4, H, W))`（见模块说明的字节账）。"""
    sp = np.asarray(spatial, dtype=np.float32)
    if sp.ndim != 3 or sp.shape[0] % 2:
        raise ValueError(f"spatial 必须是 (偶数, H, W)，收到 {sp.shape}")
    n = sp.shape[0] // 2
    L = sp[0::2]                                  # c = rot·2 → L
    G = sp[1::2]                                  # c = rot·2 + 1 → G
    h, w = sp.shape[1], sp.shape[2]
    if (h * w) % 8:
        raise ValueError(f"H·W 必须能被 8 整除（按位打包），收到 {h}×{w}")
    if L.min() < 0.0 or L.max() > 1.0 or G.min() < 0.0 or G.max() > 1.0:
        raise ValueError("spatial 的取值必须在 [0,1]（§5.5：L ∈ {0,1}、G ∈ [0,1] 且非法格已被净化为 0）")
    # ⚠️ 先展平成 (n, H·W) 再按位打包（**不能按 axis=2 逐行打包**：W 不是 8 的倍数时行尾填充位会破坏行序）
    L_bits = np.packbits((L.reshape(n, -1) > 0.5), axis=1)         # (4, H·W/8)
    G_u8 = np.rint(np.clip(G, 0.0, 1.0) * 255.0).astype(np.uint8)  # (4, H, W)
    return L_bits, G_u8


def unpack_spatial(L_bits: np.ndarray, G_u8: np.ndarray) -> np.ndarray:
    """`(L_bits, G_u8)` → `spatial (8,H,W) float32`（`pack_spatial` 的逆，`G` 保留 1/255 量化误差）。"""
    n, nbytes = L_bits.shape
    bits = np.unpackbits(np.asarray(L_bits, dtype=np.uint8), axis=1)
    hw = nbytes * 8
    L = bits[:, :hw].reshape(n, G_u8.shape[1], G_u8.shape[2]).astype(np.float32)
    G = np.asarray(G_u8, dtype=np.float32) / 255.0
    out = np.empty((2 * n, G_u8.shape[1], G_u8.shape[2]), dtype=np.float32)
    out[0::2] = L
    out[1::2] = G
    return out


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    """`mask (4,H,W) bool` → 按位 `(4, H·W/8) uint8`。"""
    m = np.asarray(mask, dtype=bool)
    return np.packbits(m.reshape(m.shape[0], -1), axis=1)


def _unpack_mask(bits: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), axis=1)[:, : h * w].reshape(-1, h, w).astype(bool)


# ===================== 打包回放 =====================

class PackedReplay:
    """环形回放（**预分配**打包数组，不存 float32 原图）。

    每条经验 = `(obs, action, reward, next_obs, next_mask, done)`；`obs/next_obs` 是 env 的 obs 字典
    （`{spatial, piece}`），`next_mask` 是**落子后**那一帧的 `action_masks()`（用于 Double DQN 的候选集）。
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, shape: Sequence[int] = DEFAULT_SHAPE) -> None:
        c = int(capacity)
        if c <= 0:
            raise ValueError(f"capacity 必须 > 0，收到 {capacity}")
        ch, h, w = (int(v) for v in shape)
        if ch % 2 or (h * w) % 8:
            raise ValueError(f"shape 必须满足 通道数为偶数 且 H·W 可被 8 整除，收到 {tuple(shape)}")
        self.capacity, self.shape = c, (ch, h, w)
        self.h, self.w, self.n = h, w, ch // 2
        self.bits_per_rot = h * w // 8            # 一个旋转的 L（或 mask）打包后字节数（200×200 ⇒ 5000）
        # —— 预分配（~380 KB/条 @200×200；容量 10,000 ⇒ 3.8 GB）——
        self._Lc = np.zeros((c, self.n, self.bits_per_rot), np.uint8)
        self._Gc = np.zeros((c, self.n, h, w), np.uint8)
        self._Pc = np.zeros((c, 3), np.float32)
        self._Ln = np.zeros((c, self.n, self.bits_per_rot), np.uint8)
        self._Gn = np.zeros((c, self.n, h, w), np.uint8)
        self._Pn = np.zeros((c, 3), np.float32)
        self._Mn = np.zeros((c, self.n, self.bits_per_rot), np.uint8)
        self._a = np.zeros(c, np.int64)
        self._r = np.zeros(c, np.float32)
        self._done = np.zeros(c, bool)
        self._pos = 0
        self._size = 0

    # ---- 基本信息 ----
    def __len__(self) -> int:
        return self._size

    @property
    def per_transition_bytes(self) -> int:
        """一条经验的字节数（含 next 侧与 mask；**用于内存核算/测试**）。"""
        L = self.n * self.bits_per_rot               # L 按位
        G = self.n * self.h * self.w                 # G uint8
        return int(2 * (L + G) + L + 3 * 4 + 8 + 4 + 1)

    @property
    def nbytes(self) -> int:
        """整个回放占用的字节数（预分配）。"""
        return int(2 * (self._Lc.nbytes + self._Gc.nbytes) + self._Mn.nbytes
                   + self._Pc.nbytes + self._Pn.nbytes + self._a.nbytes + self._r.nbytes + self._done.nbytes)

    # ---- 写入 / 采样 ----
    def add(self, obs: Mapping[str, np.ndarray], action: Sequence[int], reward: float,
            next_obs: Mapping[str, np.ndarray], next_mask: np.ndarray, done: bool) -> None:
        i = self._pos
        Lb, Gu = pack_spatial(np.asarray(obs["spatial"]))
        self._Lc[i], self._Gc[i] = Lb, Gu
        self._Pc[i] = np.asarray(obs["piece"], dtype=np.float32).reshape(3)
        Lb2, Gu2 = pack_spatial(np.asarray(next_obs["spatial"]))
        self._Ln[i], self._Gn[i] = Lb2, Gu2
        self._Pn[i] = np.asarray(next_obs["piece"], dtype=np.float32).reshape(3)
        self._Mn[i] = _pack_mask(next_mask)
        k, ii, jj = (int(v) for v in action)
        self._a[i] = k * self.h * self.w + ii * self.w + jj
        self._r[i] = float(reward)
        self._done[i] = bool(done)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """**均匀随机**（有放回）采一批 → 解包成 numpy（`obs/next_obs` 是 `(B,8,H,W)`，`mask` 是 `(B,4,H,W)` bool）。

        ⚠️ **没有实现 PER**（优先级经验回放）：不存优先级、不做重要性采样权重、不做分段和树
        —— Owner 2026-09-23："我们暂时不做这个 PER"。要加 PER 就是另一件事（会动采样分布、需要 β/权重校正）。
        """
        if self._size < batch_size:
            raise ValueError(f"回放里只有 {self._size} 条，采不出 {batch_size} 条（先攒够再学）")
        idx = rng.integers(0, self._size, size=int(batch_size))
        obs = np.stack([unpack_spatial(self._Lc[i], self._Gc[i]) for i in idx])
        nxt = np.stack([unpack_spatial(self._Ln[i], self._Gn[i]) for i in idx])
        mask = np.stack([_unpack_mask(self._Mn[i], self.h, self.w) for i in idx])
        return {
            "spatial": obs,
            "piece": self._Pc[idx].copy(),
            "action": self._a[idx].copy(),
            "reward": self._r[idx].copy(),
            "next_spatial": nxt,
            "next_piece": self._Pn[idx].copy(),
            "next_mask": mask,
            "done": self._done[idx].copy(),
        }


# ===================== Double DQN =====================

class DoubleDQNAgent:
    """打包回放 + Double DQN 更新 + 掩膜内 ε-贪心。

    Args:
        net: 在线网（缺省 `policy_net.QNet()`）；`target` 缺省 = `net` 的深拷（硬同步）。
        capacity / shape: 回放容量与 obs 形状（容量 10,000 ⇒ 3.80 GB @ 200×200）。
        target_every_episodes: **每多少个 episode 硬同步一次目标网**。Owner 2026-09-23 先拍 **100**（长跑口径），
            随后因"只跑 100 局"改为 **20**（同步 5 次：ep20/40/60/80/100）⇒ 默认值 = 20。
        eps_top / eps_bottom / eps_tau_episodes: ε 曲线 `ε(ep) = e_bottom + (e_top−e_bottom)·exp(−ep/τ)`
            （Owner 2026-09-23 拍：e_top = 0.9、e_bottom = 0.1；τ 是**我方自定**的时间常数，单位 episode）。
        lr / batch_size / gamma / device: 见模块说明（Adam + `lr = 1e-4`）。
    """

    def __init__(self, net: Optional[nn.Module] = None, *,
                 capacity: int = DEFAULT_CAPACITY, shape: Sequence[int] = DEFAULT_SHAPE,
                 lr: float = 1e-4, batch_size: int = 64, target_every_episodes: int = 20,
                 eps_top: float = 0.9, eps_bottom: float = 0.1, eps_tau_episodes: float = 2_000.0,
                 gamma: float = GAMMA, device: str = "cpu",
                 target: Optional[nn.Module] = None, seed: Optional[int] = None) -> None:
        self.device = torch.device(device)
        self.net = (net if net is not None else pn.QNet()).to(self.device)
        # 目标网缺省 = **在线网的深拷**（不假设它一定是 `QNet` ⇒ 任何 `nn.Module` 都能用）
        self.target = copy.deepcopy(self.net) if target is None else target.to(self.device)
        self.target.eval()
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_every_episodes = int(target_every_episodes)
        self.eps_top, self.eps_bottom = float(eps_top), float(eps_bottom)
        if not (0.0 <= self.eps_bottom <= self.eps_top <= 1.0):
            raise ValueError(f"ε 范围必须满足 0 ≤ e_bottom ≤ e_top ≤ 1，收到 ({eps_bottom}, {eps_top})")
        self.eps_tau = float(eps_tau_episodes)
        if self.eps_tau <= 0.0:
            raise ValueError(f"eps_tau_episodes 必须 > 0，收到 {eps_tau_episodes}")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))     # Owner 2026-09-23：Adam
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = PackedReplay(capacity=capacity, shape=shape)
        self.rng = np.random.default_rng(seed)
        self.steps = 0            # env 步（只作统计/日志）
        self.episodes = 0         # episode 数（决定 ε 与目标同步）
        self.updates = 0          # 梯度步数
        self._synced_at_episode = 0

    # ---- 行为策略 ----
    def epsilon(self) -> float:
        """`ε(ep) = e_bottom + (e_top − e_bottom)·exp(−ep / τ)` —— Owner 2026-09-23 拍的非线性平滑曲线。

        `ep = 0` ⇒ `e_top`（0.9）；单调下降；`ep → ∞` 时趋于 `e_bottom`（0.1，**只逼近不越过**）。
        """
        x = self.episodes / self.eps_tau
        return self.eps_bottom + (self.eps_top - self.eps_bottom) * math.exp(-x)

    @torch.no_grad()
    def q_values(self, obs: Mapping[str, np.ndarray], batch: bool = False) -> torch.Tensor:
        """`obs` → Q（`(1|B,4,H,W)`）。"""
        spatial = torch.as_tensor(np.asarray(obs["spatial"], dtype=np.float32), device=self.device)
        piece = torch.as_tensor(np.asarray(obs["piece"], dtype=np.float32), device=self.device)
        if not batch:
            spatial, piece = spatial.unsqueeze(0), piece.reshape(1, -1)
        self.net.eval()
        q = self.net(spatial, piece)
        self.net.train()
        return q

    def act(self, obs: Mapping[str, np.ndarray], mask: np.ndarray, *,
            explore: bool = True) -> Tuple[int, int, int]:
        """掩膜内 ε-贪心（`explore=False` ⇒ 纯贪心，评测/推理用）。"""
        q = self.q_values(obs)[0]
        eps = self.epsilon() if explore else 0.0
        return pn.masked_epsilon_greedy(q, mask, eps, self.rng)

    # ---- 经验写入（一次 `env.step` 调一次） ----
    def observe(self, obs, action, reward, next_obs, next_mask, done) -> None:
        self.replay.add(obs, action, reward, next_obs, next_mask, done)
        self.steps += 1
        if done:
            self.end_episode()

    def end_episode(self) -> None:
        """一个 episode 结束（`observe(..., done=True)` 会自动调；训练循环也可显式调）。

        它驱动两件事：ε 曲线的横坐标、以及"每 `target_every_episodes` 个 episode 硬同步目标网"。
        """
        self.episodes += 1

    # ---- Double DQN 的 TD 目标 ----
    @torch.no_grad()
    def td_target(self, next_spatial: torch.Tensor, next_piece: torch.Tensor,
                  next_mask: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """`y = r + γ·(1−done)·Q_target(s', argmax_{a∈mask(s')} Q_online(s', a))`。

        **候选集为空**（终局那一帧 `mask` 全 False）⇒ 该行直接用 `r`（不取 −inf 相加）。
        """
        q_on = self.net(next_spatial, next_piece)
        b = q_on.shape[0]
        flat = q_on.reshape(b, -1)
        mflat = next_mask.reshape(b, -1).to(torch.bool)
        has = mflat.any(dim=1)
        idx = torch.where(mflat, flat, torch.full_like(flat, -float("inf"))).argmax(dim=1)
        q_tgt = self.target(next_spatial, next_piece).reshape(b, -1).gather(1, idx[:, None]).squeeze(1)
        q_tgt = torch.where(has, q_tgt, torch.zeros_like(q_tgt))
        return reward + self.gamma * (~done) * q_tgt

    # ---- 一次梯度步 ----
    def update(self) -> Optional[float]:
        """采一批做一次 Double DQN 更新（回放不够 `batch_size` ⇒ 返回 `None`，不学）。"""
        if len(self.replay) < self.batch_size:
            return None
        b = self.replay.sample(self.batch_size, self.rng)
        t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=self.device)  # noqa: E731
        spatial, piece = t(b["spatial"]), t(b["piece"])
        action = torch.as_tensor(b["action"], dtype=torch.long, device=self.device)
        y = self.td_target(t(b["next_spatial"]), t(b["next_piece"]),
                           torch.as_tensor(b["next_mask"], dtype=torch.bool, device=self.device),
                           t(b["reward"]), torch.as_tensor(b["done"], dtype=torch.bool, device=self.device))
        q = self.net(spatial, piece).reshape(len(b["action"]), -1).gather(1, action[:, None]).squeeze(1)
        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)   # 自定：防偶发大 target
        self.opt.step()
        self.updates += 1
        self._maybe_sync_target()
        return float(loss.detach())

    def _maybe_sync_target(self) -> bool:
        """**每 `target_every_episodes` 个 episode 硬同步一次**（Owner 2026-09-23；按 episode 记数）。

        判据是"自上次同步以来已过满 K 个 episode" ⇒ 同一个 episode 内调多少次 `update()` 都只同步一次。
        """
        k = self.target_every_episodes
        if k > 0 and (self.episodes - self._synced_at_episode) >= k:
            self.target.load_state_dict(self.net.state_dict())
            self._synced_at_episode = self.episodes
            return True
        return False

    # ---- 断点 ----
    def state_dict(self) -> Dict[str, Any]:
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "steps": self.steps, "updates": self.updates,
                "episodes": self.episodes, "synced_at_episode": self._synced_at_episode}

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.net.load_state_dict(sd["net"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])
        self.steps, self.updates = int(sd["steps"]), int(sd["updates"])
        self.episodes = int(sd.get("episodes", 0))
        self._synced_at_episode = int(sd.get("synced_at_episode", 0))
