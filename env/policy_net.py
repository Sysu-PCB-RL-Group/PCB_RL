# -*- coding: utf-8 -*-
"""Double DQN 的**策略/价值网络 + 掩膜内动作选择** —— 台账 §9.2 的第一块（Owner 2026-09-23 拍：算法暂定 Double DQN）。

网络形态（与 §6.1 的动作编码**同序**，零转换）
------------------------------------------------
```text
输入  spatial (B, 8, 200, 200) float32   ← 通道序 c = rot·2 + (0=L, 1=G)，见 §5.5
      piece   (B, 3)         float32   ← [pad/PAD_MAX, log1p(净)/log1p(NET_MAX), t/n]
输出  q       (B, 4, 200, 200) float32 ← **每个旋转一张 Q 图**；q[b, k, i, j] 对应动作 a = (k, i, j)
                                           即一维化 a = k·m² + i·m + j（§6.1）
```
为什么是"空间 Q 图"而不是"16 万维 FC"：动作空间天然是 `4×200×200`，且 `L`/`G`/`mask` 与它逐轴对应
⇒ 卷积输出直接就是 Q 表（平移等价、参数少、一次前向拿到全部动作分值）；FC 会把这个结构拆掉。

**算法（Double DQN）要的三件东西都由本模块提供**：
`q = QNet(...)` ｜ `masked_argmax(q, mask)`（**行为动作**：ε-greedy 的贪心支；也是评测用的贪心策略）
｜ `masked_epsilon_greedy(q, mask, eps, rng)`（**行为动作**：ε 支 —— **只在 mask 内均匀采**，
因为 Owner 2026-09-22 拍了"mask 外的动作被 mask 阻挡"，见环境层 `env/pcb_env.py`）。
⚠️ 注意分工：**动作合法性由 `mask` 决定，不由 Q 值决定** —— Q 在非法格上的取值不参与任何选择。

本模块**只依赖 torch**（不碰缓存/KiCad），所以它的测试不需要设备环境（`tests/kicad/test_policy_net.py`）。

⚠️ 本文件里"我方定"的部分（Owner 未逐条拍，已如实标注在 §9.2）
    * 主干 = 两次 stride-2 卷积（200→100→50）后**双线性上采样两次**回到 200×200 ⇒ 算力降 ~16×，
      但 Q 图仍是**满分辨率**（动作粒度不变）；代价 = Q 图是"平滑"出来的（相邻格共享特征）
    * `piece` 的注入方式 = 小 MLP 映射成 `width` 维向量后**逐像素相加**（不做 FiLM/注意力；先把东西跑起来）
    * 不加 BatchNorm（保持前向确定性 ⇒ 测试可断言逐位相同；BN 的统计量也会随 batch 漂）
    * 不用 Dueling/PopArt/PER（§7.4 A8 的稳定化选项留作后续对照；当前 reward 已按 `÷H_ref`、`÷D` 归一）
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["QNet", "masked_argmax", "masked_epsilon_greedy", "q_of_action",
           "obs_to_tensors", "CHANNELS", "N_ROT"]

CHANNELS = 8      # spatial 通道数（= 4 旋转 × {L, G}，§5.5）
N_ROT = 4         # 旋转数 = 动作第一维（§6.1）


# ===================== 网络 =====================

class QNet(nn.Module):
    """`spatial + piece → q (B, 4, H, W)`。

    Args:
        width: 主干通道数（32 ⇒ 参数量 ~8.5 万；够小，CPU 也能跑）。
        piece_dim: `piece` 的维度（3，§5.5）。
        n_rot: 旋转数 = 输出通道数（4）。
    """

    def __init__(self, width: int = 32, piece_dim: int = 3, n_rot: int = N_ROT,
                 in_ch: int = CHANNELS) -> None:
        super().__init__()
        self.width = int(width)
        self.n_rot = int(n_rot)
        w = int(width)
        # 主干：两次 stride-2（分辨率 ÷4，算力 ÷16）
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1), nn.ReLU(inplace=True),        # /2
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),         # /4
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(inplace=True),
        )
        # `piece` → 与主干同宽度的向量，逐像素相加（最简的注入方式）
        self.piece_mlp = nn.Sequential(
            nn.Linear(piece_dim, w), nn.ReLU(inplace=True),
            nn.Linear(w, 2 * w),
        )
        # 解码：两次 ×2 上采样回到原分辨率（满分辨率 Q 图）
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(2 * w, w, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(w, w // 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(w // 2, self.n_rot, 1)

    def forward(self, spatial: torch.Tensor, piece: torch.Tensor) -> torch.Tensor:
        """→ `q (B, n_rot, H, W)`；非法格用什么值都行（选择一律走 `mask`）。"""
        if spatial.dim() != 4:
            raise ValueError(f"spatial 必须是 (B, C, H, W)，收到 {tuple(spatial.shape)}")
        if spatial.shape[1] != CHANNELS:
            raise ValueError(f"spatial 通道数必须是 {CHANNELS}（§5.5 通道序），收到 {spatial.shape[1]}")
        x = self.trunk(spatial)
        g = self.piece_mlp(piece)
        x = x + g[:, :, None, None]
        return self.head(self.up2(self.up1(x)))


def obs_to_tensors(obs: Mapping[str, Any], device: Optional[torch.device] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 env 的 obs 字典（单样本）转成 batch=1 的张量：`(spatial (1,8,H,W), piece (1,3))`。"""
    spatial = torch.as_tensor(np.asarray(obs["spatial"], dtype=np.float32)).unsqueeze(0)
    piece = torch.as_tensor(np.asarray(obs["piece"], dtype=np.float32)).reshape(1, -1)
    if device is not None:
        spatial, piece = spatial.to(device), piece.to(device)
    return spatial, piece


# ===================== 掩膜内动作选择（行为策略的两种支） =====================

def _as_numpy(x: Any, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if a.ndim != 3:
        raise ValueError(f"{name} 必须是 (4, H, W)，收到 {a.shape}")
    return a


def _check_shapes(q: np.ndarray, mask: np.ndarray) -> None:
    if q.shape != mask.shape:
        raise ValueError(f"q {q.shape} 与 mask {mask.shape} 形状必须一致（同序：q[k,i,j] ↔ 动作 (k,i,j)）")
    if q.shape[0] != N_ROT:
        raise ValueError(f"第一维必须是 {N_ROT} 个旋转，收到 {q.shape[0]}")


def masked_argmax(q: Any, mask: Any) -> Tuple[int, int, int]:
    """`mask` 内的最大 Q 对应的动作 `(k, i, j)` —— **`mask` 外的取值一律不参与**。

    `mask` 全 False（"合法动作集为空"）⇒ 抛 `ValueError`：按 §5.4 这时 episode 应已终止，
    调用方不该再问"选哪个动作"（而不是返回一个假动作）。
    """
    qn, mn = _as_numpy(q, "q"), np.asarray(mask, dtype=bool)
    _check_shapes(qn, mn)
    if not mn.any():
        raise ValueError("mask 全 False（合法动作集为空）⇒ episode 应已终止，调用方不该再选动作")
    idx = int(np.argmax(np.where(mn, qn, -np.inf)))
    k, rem = divmod(idx, mn.shape[1] * mn.shape[2])
    i, j = divmod(rem, mn.shape[2])
    return int(k), int(i), int(j)


def masked_epsilon_greedy(q: Any, mask: Any, eps: float, rng: np.random.Generator
                          ) -> Tuple[int, int, int]:
    """ε-贪心（**两支都只在 `mask` 内**）：`1−eps` 取 `masked_argmax`，`eps` 在 `mask` 内均匀采。

    为什么 ε 支不能全空间采：动作空间 160,000 格里 `mask` 常只有几百~几万格（实测 t=0 时 `mask/L`
    占全空间 0.2%~68%），全空间均匀采有 ~95% 概率落在外面 ⇒ 被环境层直接挡住（Owner 2026-09-22 ⑦）。
    """
    qn, mn = _as_numpy(q, "q"), np.asarray(mask, dtype=bool)
    _check_shapes(qn, mn)
    cells = np.argwhere(mn.reshape(-1))
    if not len(cells):
        raise ValueError("mask 全 False（合法动作集为空）⇒ episode 应已终止，调用方不该再选动作")
    if float(eps) > 0.0 and float(rng.random()) < float(eps):
        flat = int(cells[int(rng.integers(len(cells))), 0])
        h, w = mn.shape[1], mn.shape[2]
        k, rem = divmod(flat, h * w)
        i, j = divmod(rem, w)
        return int(k), int(i), int(j)
    return masked_argmax(qn, mn)


def q_of_action(q: Any, action: Tuple[int, int, int]) -> float:
    """取某个动作的 Q 值（训练时构造 TD 目标/损失用；越界抛错）。"""
    qn = _as_numpy(q, "q")
    k, i, j = (int(v) for v in action)
    if not (0 <= k < qn.shape[0] and 0 <= i < qn.shape[1] and 0 <= j < qn.shape[2]):
        raise ValueError(f"动作越界：{action} ∉ [0,{qn.shape[0]})×[0,{qn.shape[1]})×[0,{qn.shape[2]})")
    return float(qn[k, i, j])
