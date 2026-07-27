"""DQN 网络：5-Head 自回归结构。

基于 SPEC v2.2 §4.4。
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """5-Head 自回归 DQN 网络。

    Shared 3 层 MLP Body → 5 个独立 Linear Head，
    每次 forward 只走一个 Head，其余 Head 不在计算图上。
    """

    def __init__(self, obs_dim: int, n_movable: int, grid_w: int, grid_h: int):
        """初始化网络。

        Args:
            obs_dim: 观测维度 (N_movable*8 + N_movable + 4)
            n_movable: 可移动元件数
            grid_w: 网格宽度
            grid_h: 网格高度
        """
        super().__init__()

        # 共享 Body：3 层 MLP
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # 5 个独立 Head（每步一个）
        self.head_comp = nn.Linear(128, n_movable)   # Step 0: 选元件
        self.head_x1   = nn.Linear(128, grid_w)       # Step 1: 选 x1
        self.head_y1   = nn.Linear(128, grid_h)       # Step 2: 选 y1
        self.head_rot  = nn.Linear(128, 4)            # Step 3: 选旋转
        self.head_lay  = nn.Linear(128, 2)            # Step 4: 选层

        self._heads = [self.head_comp, self.head_x1,
                       self.head_y1, self.head_rot, self.head_lay]

    def forward(self, obs: torch.Tensor, step_id: int) -> torch.Tensor:
        """前向传播：共享 Body → 指定 Head。

        Args:
            obs: 观测向量 shape=(batch, obs_dim)
            step_id: 子步骤编号 0-4

        Returns:
            Q 值向量 shape=(batch, head_output_dim)
        """
        x = self.shared(obs)
        return self._heads[step_id](x)
