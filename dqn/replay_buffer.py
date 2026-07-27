"""经验回放缓冲区：环形队列存储。

基于 SPEC v2.2 §4.5 和 §9.4。
"""

from typing import List, Tuple, Optional
import numpy as np


class ReplayBuffer:
    """环形缓冲区，存储自回归 DQN transition。

    每条 transition: (obs, action, reward, next_obs, mask, done, step_id, next_step_id)
    - next_step_id: 下个状态的子步骤编号，-1 表示终端状态（done=True 时不需要）
    - step 0-3 的 reward=0（中间步不分配 reward），仅 step 4 获得环境 reward
    """

    def __init__(self, capacity: int):
        """初始化缓冲区。

        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mask: np.ndarray,
        done: bool,
        step_id: int,
        next_step_id: int,
    ):
        """添加一条 transition。

        Args:
            obs: 当前观测 (obs_dim,)
            action: 选中的动作索引
            reward: 延迟 reward（仅 step_id=4 时非零）
            next_obs: 下一步观测 (obs_dim,)
            mask: 当前子步骤的动作掩码 (K,)
            done: 是否终止（仅 step_id=4 时可能为 True）
            step_id: 当前子步骤编号 0-4
            next_step_id: 下个子步骤编号，-1 表示终端
        """
        data = (obs, action, reward, next_obs, mask, done, step_id, next_step_id)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """随机采样一个 mini-batch。

        Args:
            batch_size: 采样数量

        Returns:
            8 元组 (obs, actions, rewards, next_obs, masks, dones, step_ids, next_step_ids)
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        obs_list, actions, rewards, next_obs_list, masks, dones, step_ids, next_step_ids = zip(*batch)

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs_list, dtype=np.float32),
            list(masks),              # 保持 list 以支持不同长度的 mask
            np.array(dones, dtype=np.float32),
            np.array(step_ids, dtype=np.int64),
            np.array(next_step_ids, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够样本进行训练。

        Args:
            batch_size: 所需批量大小

        Returns:
            True 如果缓冲区样本数 >= batch_size
        """
        return len(self.buffer) >= batch_size
