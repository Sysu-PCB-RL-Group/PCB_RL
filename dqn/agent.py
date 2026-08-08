"""DQN Agent：epsilon-greedy 选择 + 按 step_id 路由学习 + target 同步。

基于 SPEC v2.3：Dict 观测 + 共享 slot encoder 网络。
Target 端掩码与 Actor 侧对称：
  nsid=0: placed_flag 屏蔽已放置元件
  nsid=1: grid_width 屏蔽越界 x1 坐标
  nsid=2: grid_height 屏蔽越界 y1 坐标
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
from env.pcb_layout_env import Config

logger = logging.getLogger(__name__)


class DQNAgent:
    """DQN Agent，管理 Q 网络、target 网络、epsilon-greedy 探索和学习更新。"""

    def __init__(
        self,
        n_movable: int,
        grid_w: int,
        grid_h: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        """初始化 Agent。

        Args:
            n_movable: Head 0 输出维度 (固定 = MAX_MOVABLE)
            grid_w: Head 1 输出维度 (固定 = GRID_W)
            grid_h: Head 2 输出维度 (固定 = GRID_H)
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
        self.q_net = DQN(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h).to(device)
        self.target_net = DQN(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 训练步数计数器（用于 target 同步判定）
        self.train_steps: int = 0

    def _obs_to_tensor(self, obs: dict) -> dict:
        """将 numpy dict 观测转为 torch tensors（单样本，加 batch 维）。"""
        return {
            "component_slots": torch.FloatTensor(obs["component_slots"]).unsqueeze(0).to(self.device),
            "global_features": torch.FloatTensor(obs["global_features"]).unsqueeze(0).to(self.device),
        }

    def _batch_obs_to_tensor(self, obs_batch: dict) -> dict:
        """将 batched dict 观测转为 torch tensors。"""
        return {
            "component_slots": torch.FloatTensor(obs_batch["component_slots"]).to(self.device),
            "global_features": torch.FloatTensor(obs_batch["global_features"]).to(self.device),
        }

    def select_action(
        self,
        obs: dict,
        mask: Optional[np.ndarray],
        step_id: int,
        epsilon: float = 0.0,
    ) -> int:
        """epsilon-greedy 动作选择。

        Args:
            obs: Dict 观测 {"component_slots": (M,8), "global_features": (8,)}
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
            obs_tensor = self._obs_to_tensor(obs)
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
            return 0

        if random.random() < epsilon:
            return int(valid_indices[random.randint(0, len(valid_indices) - 1)])

        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor, step_id).squeeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
            q_values[~mask_tensor] = -float('inf')
            # 打破平局：多个动作 Q 值相同时随机选择，避免确定性死锁
            # （例如所有未放置元件 slot 全为零 → Q 值完全相等 → 每次都选同一个）
            max_q = q_values.max()
            tied = (q_values == max_q)
            tied_indices = tied.nonzero(as_tuple=True)[0]
            if tied_indices.numel() > 1:
                return int(tied_indices[torch.randint(0, len(tied_indices), (1,))].item())
            return int(tied_indices[0].item())

    def learn(self, batch) -> float:
        """从 mini-batch 学习，按 next_step_id 路由目标 Q 网络。

        Q-Learning 目标：
            target = r + γ * max_{a'} Q_target(s', a', next_step_id) * (1 - done)

        Target 端掩码（与 Actor 侧对称）：
            nsid=0: component_slots[:,:,0] → placed_flag 屏蔽已放置元件
            nsid=1: global_features[0] → grid_width 屏蔽越界 x1 坐标
            nsid=2: global_features[1] → grid_height 屏蔽越界 y1 坐标
            nsid=3/4: 无需掩码（全合法）

        Args:
            batch: ReplayBuffer.sample() 返回的 8 元组

        Returns:
            本次更新的总 loss 值
        """
        obs_batch, actions, rewards, next_obs_batch, masks, dones, step_ids, next_step_ids = batch

        obs_tensor = self._batch_obs_to_tensor(obs_batch)
        next_obs_tensor = self._batch_obs_to_tensor(next_obs_batch)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        batch_size = actions_tensor.shape[0]

        # 当前 Q 值：按 step_id 分组走对应 Head
        q_vals = torch.zeros(batch_size, 1, device=self.device)
        for sid in range(5):
            step_idx = (step_ids == sid)
            if not step_idx.any():
                continue
            idx = torch.BoolTensor(step_idx).to(self.device)
            sub_obs = {
                "component_slots": obs_tensor["component_slots"][idx],
                "global_features": obs_tensor["global_features"][idx],
            }
            q_all = self.q_net(sub_obs, sid)
            q_vals[idx] = q_all.gather(1, actions_tensor[idx])

        # 目标 Q 值：按 next_step_id 分组走对应 Head，同步 Actor 侧 mask
        target_q = torch.zeros(batch_size, 1, device=self.device)
        with torch.no_grad():
            for nsid in range(5):
                ns_idx = (next_step_ids == nsid)
                if not ns_idx.any():
                    continue
                nidx = torch.BoolTensor(ns_idx).to(self.device)
                sub_next = {
                    "component_slots": next_obs_tensor["component_slots"][nidx],
                    "global_features": next_obs_tensor["global_features"][nidx],
                }
                next_q_all = self.target_net(sub_next, nsid)

                if nsid == 0:
                    # 从 Dict 观测提取 placed_flag（每个 slot 的第 0 维）
                    placed_flags = sub_next["component_slots"][:, :, 0]  # (n_subset, M)
                    comp_mask = (placed_flags < 0.5).to(self.device)  # True=未放置=可选
                    if comp_mask.any():
                        next_q_all[~comp_mask] = -float('inf')

                elif nsid == 1:
                    # x1 坐标掩码：屏蔽超出真实网格宽度的坐标
                    # global_features[0] = grid_width / GRID_W
                    grid_w = (sub_next["global_features"][:, 0] * Config.GRID_W).round().long()
                    x_mask = torch.arange(Config.GRID_W, device=self.device).unsqueeze(0) < grid_w.unsqueeze(1)
                    if x_mask.any():
                        next_q_all[~x_mask] = -float('inf')

                elif nsid == 2:
                    # y1 坐标掩码：屏蔽超出真实网格高度的坐标
                    # global_features[1] = grid_height / GRID_H
                    grid_h = (sub_next["global_features"][:, 1] * Config.GRID_H).round().long()
                    y_mask = torch.arange(Config.GRID_H, device=self.device).unsqueeze(0) < grid_h.unsqueeze(1)
                    if y_mask.any():
                        next_q_all[~y_mask] = -float('inf')

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
        """保存模型检查点。"""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """加载模型检查点。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_steps = checkpoint['train_steps']
        logger.info(f"Checkpoint loaded from {path}, train_steps={self.train_steps}")
