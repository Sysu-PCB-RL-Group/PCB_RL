"""DQN Agent：epsilon-greedy 选择 + 按 step_id 路由学习 + target 同步 + BC 预训练。

基于 SPEC v2.3：Dict 观测 + 共享 slot encoder 网络。
Target 端掩码与 Actor 侧对称：
  nsid=0: placed_flag 屏蔽已放置元件
  nsid=1: grid_width 屏蔽越界 x1 坐标
  nsid=2: grid_height 屏蔽越界 y1 坐标
"""

import random
import logging
from typing import Optional, List, Tuple

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

        # 冻结的 BC 策略锚点（用于 BC 正则化，阻止 RL 把 BC 好策略学坏）
        self.bc_net = DQN(n_movable=n_movable, grid_w=grid_w, grid_h=grid_h).to(device)
        self.bc_net.eval()
        self.bc_net_active: bool = False
        self.bc_reg_lambda: float = float(getattr(Config, "BC_REG_LAMBDA", 0.0))
        self.bc_reg_temp: float = float(getattr(Config, "BC_REG_TEMP", 1.0))
        self.bc_reg_value_lambda: float = float(getattr(Config, "BC_REG_VALUE_LAMBDA", 0.0))
        self.last_bc_reg: float = 0.0
        self.last_bc_val: float = 0.0

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 训练步数计数器（用于 target 同步判定）
        self.train_steps: int = 0

    def _obs_to_tensor(self, obs: dict) -> dict:
        """将 numpy dict 观测转为 torch tensors（单样本，加 batch 维）。"""
        return {
            "component_slots": torch.from_numpy(obs["component_slots"]).float().unsqueeze(0).to(self.device),
            "global_features": torch.from_numpy(obs["global_features"]).float().unsqueeze(0).to(self.device),
        }

    def _batch_obs_to_tensor(self, obs_batch: dict) -> dict:
        """将 batched dict 观测转为 torch tensors。"""
        return {
            "component_slots": torch.from_numpy(obs_batch["component_slots"]).float().to(self.device),
            "global_features": torch.from_numpy(obs_batch["global_features"]).float().to(self.device),
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
            # 打破平局，防止确定性死锁（与有掩码路径一致）
            max_q = q_values.max()
            tied = (q_values == max_q)
            tied_indices = tied.nonzero(as_tuple=True)[0]
            if tied_indices.numel() > 1:
                return int(tied_indices[torch.randint(0, len(tied_indices), (1,))].item())
            return int(tied_indices[0].item())

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

    def learn(self, batch, is_weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
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
            is_weights: PER 重要性采样权重 shape=(batch_size,)，None=均匀权重

        Returns:
            (loss_value, td_errors) 元组，td_errors 供 PER 优先级更新
        """
        obs_batch, actions, rewards, next_obs_batch, masks, dones, step_ids, next_step_ids = batch

        obs_tensor = self._batch_obs_to_tensor(obs_batch)
        next_obs_tensor = self._batch_obs_to_tensor(next_obs_batch)
        actions_tensor = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones_tensor = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        batch_size = actions_tensor.shape[0]

        # 当前 Q 值：按 step_id 分组走对应 Head
        q_vals = torch.zeros(batch_size, 1, device=self.device)
        for sid in range(5):
            step_idx = (step_ids == sid)
            if not step_idx.any():
                continue
            idx = torch.tensor(step_idx, dtype=torch.bool, device=self.device)
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
                nidx = torch.tensor(ns_idx, dtype=torch.bool, device=self.device)
                sub_next = {
                    "component_slots": next_obs_tensor["component_slots"][nidx],
                    "global_features": next_obs_tensor["global_features"][nidx],
                }
                next_q_all = self.target_net(sub_next, nsid)

                if nsid == 0:
                    # 从 Dict 观测提取 placed_flag（每个 slot 的第 0 维）
                    placed_flags = sub_next["component_slots"][:, :, 0]  # (n_subset, M)
                    comp_mask = (placed_flags < 0.5).to(self.device)  # True=未放置=可选
                    next_q_all[~comp_mask] = -float('inf')

                elif nsid == 1:
                    # x1 坐标掩码：屏蔽超出真实网格宽度的坐标
                    # global_features[0] = grid_width / GRID_W
                    grid_w = (sub_next["global_features"][:, 0] * Config.GRID_W).round().long()
                    x_mask = torch.arange(self.grid_w, device=self.device).unsqueeze(0) < grid_w.unsqueeze(1)
                    next_q_all[~x_mask] = -float('inf')

                elif nsid == 2:
                    # y1 坐标掩码：屏蔽超出真实网格高度的坐标
                    # global_features[1] = grid_height / GRID_H
                    grid_h = (sub_next["global_features"][:, 1] * Config.GRID_H).round().long()
                    y_mask = torch.arange(self.grid_h, device=self.device).unsqueeze(0) < grid_h.unsqueeze(1)
                    next_q_all[~y_mask] = -float('inf')

                target_q[ns_idx] = next_q_all.max(1)[0].unsqueeze(1)

        # 标准 DQN 目标：r + γ * max Q(s', a') * (1 - done)
        target = rewards_tensor + self.gamma * target_q * (1.0 - dones_tensor)

        # PER 重要性采样加权：逐样本平方误差 × IS 权重 → 加权平均
        element_loss = torch.nn.functional.mse_loss(q_vals, target, reduction="none")
        if is_weights is not None:
            weights_tensor = torch.from_numpy(is_weights).float().unsqueeze(1).to(self.device)
            loss = (element_loss * weights_tensor).mean()
        else:
            loss = element_loss.mean()

        # BC 正则化：同时锚定策略分布(KL)与 Q 值幅度(L2)到冻结 BC，阻止 Q 值发散
        if (self.bc_reg_lambda > 0 or self.bc_reg_value_lambda > 0) and self.bc_net_active:
            kl_reg, val_reg = self._bc_regularization(obs_tensor, step_ids)
            self.last_bc_reg = kl_reg.item()
            self.last_bc_val = val_reg.item()
            loss = loss + self.bc_reg_lambda * kl_reg + self.bc_reg_value_lambda * val_reg
            if self.train_steps % 200 == 0 or self.train_steps < 5:
                logger.info(
                    f"[BC-REG] step={self.train_steps} kl={self.last_bc_reg:.4f} "
                    f"val={self.last_bc_val:.3f} λkl={self.bc_reg_lambda} λval={self.bc_reg_value_lambda} "
                    f"eff_kl={self.bc_reg_lambda * self.last_bc_reg:.3f} eff_val={self.bc_reg_value_lambda * self.last_bc_val:.3f}"
                )
        else:
            self.last_bc_reg = 0.0
            self.last_bc_val = 0.0

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        # 逐样本 TD-error（供 PER 优先级更新）
        td_errors = (target - q_vals).detach().cpu().numpy().flatten()
        return loss.item(), td_errors

    def init_bc_anchor(self):
        """用当前 Q 网络（BC 预训练后权重）初始化冻结 BC 策略锚点，供 RL 阶段正则化。"""
        self.bc_net.load_state_dict(self.q_net.state_dict())
        self.bc_net.eval()
        self.bc_net_active = True
        logger.info(
            f"BC anchor initialized for regularization "
            f"(λ={self.bc_reg_lambda}, τ={self.bc_reg_temp})"
        )

    def _bc_regularization(self, obs_tensor: dict, step_ids: np.ndarray):
        """BC 正则化：同时返回 (策略 KL, 值 L2) 两项，均相对冻结 BC 策略。

        - 策略 KL：当前策略分布 ‖ BC 策略分布（仅合法动作空间），锚定动作选择。
        - 值 L2：MSE(Q_当前, Q_BC)（掩码后，仅合法动作），锚定 Q 值幅度，防 Q 值发散。
        两者均在 batch 与该 batch 出现的 head 上平均。
        """
        total_kl = torch.tensor(0.0, device=self.device)
        total_vm = torch.tensor(0.0, device=self.device)
        total_n = 0
        tau = self.bc_reg_temp
        for sid in range(5):
            step_idx = (step_ids == sid)
            if not step_idx.any():
                continue
            idx = torch.tensor(step_idx, dtype=torch.bool, device=self.device)
            sub_obs = {
                "component_slots": obs_tensor["component_slots"][idx],
                "global_features": obs_tensor["global_features"][idx],
            }
            cur_q = self.q_net(sub_obs, sid)
            with torch.no_grad():
                bc_q = self.bc_net(sub_obs, sid)  # (n, K_sid)

            # 合法动作掩码（与 TD 目标端对称）
            if sid == 0:
                placed_flags = sub_obs["component_slots"][:, :, 0]
                legal = (placed_flags < 0.5).to(self.device)
            elif sid == 1:
                grid_w = (sub_obs["global_features"][:, 0] * Config.GRID_W).round().long()
                legal = torch.arange(self.grid_w, device=self.device).unsqueeze(0) < grid_w.unsqueeze(1)
            elif sid == 2:
                grid_h = (sub_obs["global_features"][:, 1] * Config.GRID_H).round().long()
                legal = torch.arange(self.grid_h, device=self.device).unsqueeze(0) < grid_h.unsqueeze(1)
            else:
                legal = torch.ones_like(cur_q, dtype=torch.bool)

            cur_qm = cur_q.clone(); bc_qm = bc_q.clone()
            # 值 L2 用：非法动作对齐为 0（差值=0，不产生 NaN）
            cur_qv = cur_q.clone(); bc_qv = bc_q.clone()
            cur_qv[~legal] = 0.0; bc_qv[~legal] = 0.0
            # KL 用：非法动作置 -inf 后再 softmax
            cur_qm[~legal] = float('-inf'); bc_qm[~legal] = float('-inf')
            # 防御：整行全非法（理论上不会发生）→ 置 0，避免 softmax NaN
            all_masked = (~legal).all(dim=-1)
            if all_masked.any():
                cur_qm[all_masked] = 0.0; bc_qm[all_masked] = 0.0
                cur_qv[all_masked] = 0.0; bc_qv[all_masked] = 0.0

            # 策略 KL
            cur_p = torch.nn.functional.softmax(cur_qm / tau, dim=-1)
            bc_p = torch.nn.functional.softmax(bc_qm / tau, dim=-1)
            kl = (cur_p * (cur_p.clamp_min(1e-8).log() - bc_p.clamp_min(1e-8).log())).sum(dim=-1)
            total_kl += kl.sum()

            # 值 L2（非法已对齐为 0，差值=0，无 NaN）
            vm = ((cur_qv - bc_qv) ** 2).mean()
            total_vm += vm

            total_n += int(kl.numel())
        scale = max(total_n, 1)
        return total_kl / scale, total_vm / scale

    def learn_bc(self, expert_transitions: List[Tuple], batch_size: int = 64,
                 epochs: int = 50, verbose: bool = True) -> List[float]:
        """BC（Behavior Cloning）预训练：用专家 transition 做监督学习。

        使用交叉熵损失直接拟合专家动作分布，而非 Q-learning TD 目标。
        将专家动作视为 one-hot 标签，Q 网络输出视为 logits。

        Args:
            expert_transitions: 专家 transition 列表
            batch_size: 每次采样的批量大小
            epochs: 训练 epoch 数
            verbose: 是否打印每个 epoch 的 loss

        Returns:
            每个 epoch 的平均 loss 列表
        """
        epoch_losses = []

        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(len(expert_transitions))
            epoch_loss_sum = 0.0
            n_batches = 0

            for start in range(0, len(expert_transitions), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = [expert_transitions[i] for i in batch_indices]

                obs_list, actions_list, _, _, _, _, step_ids, _ = zip(*batch)

                obs_tensor = self._batch_obs_to_tensor(
                    self._batch_dict_obs_from_list(obs_list)
                )
                actions_tensor = torch.tensor(actions_list, dtype=torch.long, device=self.device)
                step_ids_arr = np.array(step_ids, dtype=np.int64)

                # 按 step_id 分组计算交叉熵
                total_loss = torch.tensor(0.0, device=self.device)
                count = 0

                for sid in range(5):
                    step_idx = (step_ids_arr == sid)
                    if not step_idx.any():
                        continue
                    idx = torch.tensor(step_idx, dtype=torch.bool, device=self.device)
                    sub_obs = {
                        "component_slots": obs_tensor["component_slots"][idx],
                        "global_features": obs_tensor["global_features"][idx],
                    }
                    q_all = self.q_net(sub_obs, sid)  # (n, head_dim)
                    target = actions_tensor[idx]       # (n,)

                    # 交叉熵：将 Q 值视为 logits
                    ce_loss = nn.CrossEntropyLoss()(q_all, target)
                    total_loss += ce_loss
                    count += 1

                if count > 0:
                    avg_batch_loss = total_loss / count
                    self.optimizer.zero_grad()
                    avg_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss_sum += avg_batch_loss.item()
                    n_batches += 1

            avg_epoch_loss = epoch_loss_sum / max(n_batches, 1)
            epoch_losses.append(avg_epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"BC pretrain epoch {epoch + 1}/{epochs}, loss={avg_epoch_loss:.6f}")

        # BC 预训练完成后同步 target 网络
        self.sync_target()
        logger.info(f"BC pretrain complete: {epochs} epochs, final loss={epoch_losses[-1]:.6f}")

        return epoch_losses

    @staticmethod
    def _batch_dict_obs_from_list(obs_list: list) -> dict:
        """将 dict 观测列表拼接为 batched dict。"""
        return {
            "component_slots": np.stack([o["component_slots"] for o in obs_list]),
            "global_features": np.stack([o["global_features"] for o in obs_list]),
        }

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
