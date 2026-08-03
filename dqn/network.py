"""DQN 网络：5-Head 自回归结构 + Dict 观测 + 共享 Slot Encoder。

基于 SPEC v2.3：Dict 观测 → slot encoder (共享权重) → 分支。
Step 0 直接走 per-slot head，Step 1-4 走 mean pooling + shared body。
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """5-Head 自回归 DQN，接收 Dict 观测。

    观测结构:
      component_slots: (batch, MAX_MOVABLE, 8)
      global_features: (batch, 8)

    架构:
      Slot Encoder (共享 Linear(8, 16))
        ├── Step 0: head_comp(Linear(16, 1)) → per-slot Q
        └── Step 1-4: mean_pool + concat(global) → shared body → 4 heads
    """

    def __init__(self, slot_dim: int = 8, slot_enc_dim: int = 16,
                 n_movable: int = 384, grid_w: int = 512, grid_h: int = 512):
        """初始化网络。

        Args:
            slot_dim: 每个元件 slot 的特征维度 (8)
            slot_enc_dim: slot encoder 输出维度 (16)
            n_movable: Head 0 输出维度 (固定 = MAX_MOVABLE)
            grid_w: Head 1 输出维度 (固定 = GRID_W)
            grid_h: Head 2 输出维度 (固定 = GRID_H)
        """
        super().__init__()

        # Slot encoder: 对所有 slot 共享权重
        self.slot_encoder = nn.Linear(slot_dim, slot_enc_dim)

        # Head 0: 对每个 slot 独立打分 → Q 值
        self.head_comp = nn.Linear(slot_enc_dim, 1)

        # Shared body (step 1-4): pooled_slot(16) + global(8) = 24 → 256 → 128
        shared_in = slot_enc_dim + 8
        self.shared = nn.Sequential(
            nn.Linear(shared_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # Head 1-4: 离散动作
        self.head_x1 = nn.Linear(128, grid_w)
        self.head_y1 = nn.Linear(128, grid_h)
        self.head_rot = nn.Linear(128, 4)
        self.head_lay = nn.Linear(128, 2)

    def forward(self, obs: dict, step_id: int) -> torch.Tensor:
        """前向传播。

        Args:
            obs: dict with "component_slots" (B, N_slot, 8) and "global_features" (B, 8)
            step_id: 子步骤 0-4

        Returns:
            Q 值向量
              step_id=0 → (B, N_slot)
              step_id=1 → (B, GRID_W)
              step_id=2 → (B, GRID_H)
              step_id=3 → (B, 4)
              step_id=4 → (B, 2)
        """
        slots = obs["component_slots"]        # (B, N_slot, 8)
        global_f = obs["global_features"]     # (B, 8)
        B = slots.shape[0]

        # Slot encoder (共享，逐 slot)
        slot_feat = self.slot_encoder(slots)  # (B, N_slot, 16)

        if step_id == 0:
            # Per-slot Q → (B, N_slot)
            return self.head_comp(slot_feat).squeeze(-1)

        # Step 1-4: mean pool → concat global → shared body → head
        pooled = slot_feat.mean(dim=1)        # (B, 16)
        x = torch.cat([pooled, global_f], dim=-1)  # (B, 24)
        x = self.shared(x)                    # (B, 128)

        heads = [None, self.head_x1, self.head_y1, self.head_rot, self.head_lay]
        return heads[step_id](x)
