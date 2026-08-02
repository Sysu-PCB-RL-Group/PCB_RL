"""DQN Agent：epsilon-greedy 选择 + 按 step_id 路由学习 + target 同步。

基于 SPEC v2.2 §4.5 和 §10.1。
"""

import random
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DQN
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """DQN Agent，管理 Q 网络、target 网络、epsilon-greedy 探索和学习更新。"""

    def __init__(
        self,
        obs_dim: int,
        n_movable: int,
        grid_w: int,
        grid_h: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        """初始化 Agent。

        Args:
            obs_dim: 观测维度
            n_movable: 可移动元件数
            grid_w: 网格宽度
            grid_h: 网格高度
            device: 计算设备
            lr: Adam 学习率
            gamma: 折扣因子
        """
        self.device = device
        self.gamma = gamma
        self.n_movable = n_movable
        self.grid_w = grid_w
        self.grid_h = grid_h

        # Q 网络 & target 网络
        self.q_net = DQN(obs_dim, n_movable, grid_w, grid_h).to(device)
        self.target_net = DQN(obs_dim, n_movable, grid_w, grid_h).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target 网络始终 eval 模式

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 训练步数计数器（用于 target 同步判定）
        self.train_steps: int = 0

    def select_action(
        self,
        obs: np.ndarray,
        mask: Optional[np.ndarray],
        step_id: int,
        epsilon: float = 0.0,
    ) -> int:
        """epsilon-greedy 动作选择。

        Args:
            obs: 观测向量 shape=(obs_dim,)
            mask: 动作掩码 shape=(K,)，True=合法；None=全合法
            step_id: 子步骤编号 0-4
            epsilon: 探索概率

        Returns:
            选中的动作索引
        """
        # 确定动作空间大小
        head_dims = [self.n_movable, self.grid_w, self.grid_h, 4, 2]
        n_actions = head_dims[step_id]

        # 无掩码：所有动作合法
        if mask is None:
            if random.random() < epsilon:
                return random.randint(0, n_actions - 1)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(obs_tensor, step_id).squeeze(0)
                return int(q_values.argmax().item())

        # 有掩码
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            logger.error(
                f"select_action: mask all-False at step_id={step_id}! "
                f"Returning 0 as fallback — check _build_mask_for_step."
            )
            return 0  # fallback

        # 探索
        if random.random() < epsilon:
            return int(valid_indices[random.randint(0, len(valid_indices) - 1)])

        # 利用
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor, step_id).squeeze(0)  # shape=(K,)
            # 掩码：非法动作 → -inf
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
            q_values[~mask_tensor] = -float('inf')
            return int(q_values.argmax().item())

    def learn(self, batch) -> float:
        """从 mini-batch 学习，按 next_step_id 路由目标 Q 网络。

        Q-Learning 目标：
            target = r + γ * max_{a'} Q_target(s', a', next_step_id) * (1 - done)
            其中 next_step_id=-1 表示终端状态（无 next Q）

        Target 端掩码：仅 next_step_id=0 时从 next_obs 提取 placed_flag
        构建掩码，屏蔽已放置元件的虚高 Q 值。

        Args:
            batch: ReplayBuffer.sample() 返回的 8 元组

        Returns:
            本次更新的总 loss 值
        """
        obs_batch, actions, rewards, next_obs_batch, masks, dones, step_ids, next_step_ids = batch

        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        batch_size = obs_batch.shape[0]

        # 当前 Q 值：按 step_id 分组走对应 Head
        q_vals = torch.zeros(batch_size, 1, device=self.device)
        for sid in range(5):
            step_idx = (step_ids == sid)
            if not step_idx.any():
                continue
            idx = torch.BoolTensor(step_idx).to(self.device)
            q_all = self.q_net(obs_tensor[idx], sid)
            q_vals[idx] = q_all.gather(1, actions_tensor[idx])

        # 目标 Q 值：按 next_step_id 分组走对应 Head
        # next_step_id=0 时掩码（屏蔽已放置元件），其余不掩码
        target_q = torch.zeros(batch_size, 1, device=self.device)
        with torch.no_grad():
            for nsid in range(5):
                ns_idx = (next_step_ids == nsid)
                if not ns_idx.any():
                    continue
                nidx = torch.BoolTensor(ns_idx).to(self.device)
                next_q_all = self.target_net(next_obs_tensor[nidx], nsid)

                if nsid == 0:
                    # 从 next_obs 提取 placed_flag，构建元件选择掩码
                    # obs 结构：每元件 8 维，第一个维度 = placed_flag (1.0=已放置)
                    placed_flags = next_obs_tensor[nidx, :self.n_movable * 8]
                    placed_batch = placed_flags[:, ::8]  # shape=(n_subset, n_movable)
                    comp_mask = (placed_batch < 0.5).to(self.device)  # True=未放置=可选
                    # 保护：若所有元件已放置，不设 inf（避免 max=-inf）
                    if comp_mask.any():
                        next_q_all[~comp_mask] = -float('inf')

                target_q[ns_idx] = next_q_all.max(1)[0].unsqueeze(1)

        # 标准 DQN 目标：r + γ * max Q(s', a') * (1 - done)
        target = rewards_tensor + self.gamma * target_q * (1.0 - dones_tensor)

        loss = self.loss_fn(q_vals, target)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        return loss.item()

    def sync_target(self):
        """将 Q 网络参数同步到 target 网络。"""
        self.target_net.load_state_dict(self.q_net.state_dict())
        logger.debug(f"Target network synced at train_step={self.train_steps}")

    def save(self, path: str):
        """保存模型检查点。

        Args:
            path: 保存路径
        """
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """加载模型检查点。

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = checkpoint['train_steps']
        logger.info(f"Checkpoint loaded from {path}, train_steps={self.train_steps}")
