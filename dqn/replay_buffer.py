"""经验回放缓冲区：环形队列存储 + 专家经验常驻 + 优先经验回放（PER）。

基于 SPEC v2.2 §4.5 和 §9.4，扩展支持专家轨迹注入和 PER。
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
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        mask: Optional[np.ndarray],
        done: bool,
        step_id: int,
        next_step_id: int,
    ):
        """添加一条 transition。

        Args:
            obs: 当前 Dict 观测 {"component_slots": (M,8), "global_features": (8,)}
            action: 选中的动作索引
            reward: 延迟 reward（仅 step_id=4 时非零）
            next_obs: 下一步 Dict 观测，结构同 obs
            mask: 当前子步骤的动作掩码 (K,) 或 None（无掩码）
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

    @staticmethod
    def _batch_dict_obs(obs_list: list) -> dict:
        """将 dict 观测列表拼接为 batched dict。"""
        return {
            "component_slots": np.stack([o["component_slots"] for o in obs_list]),
            "global_features": np.stack([o["global_features"] for o in obs_list]),
        }

    def sample(self, batch_size: int):
        """随机采样一个 mini-batch。

        Args:
            batch_size: 采样数量

        Returns:
            8 元组 (obs_dict, actions, rewards, next_obs_dict, masks, dones, step_ids, next_step_ids)
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        obs_list, actions, rewards, next_obs_list, masks, dones, step_ids, next_step_ids = zip(*batch)

        return (
            self._batch_dict_obs(obs_list),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            self._batch_dict_obs(next_obs_list),
            list(masks),
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


# ===================== ExpertReplayBuffer（扩展 ReplayBuffer） =====================

class ExpertReplayBuffer(ReplayBuffer):
    """支持专家经验常驻 + 优先经验回放（PER）的扩展缓冲区。

    设计要点：
    - 总容量 = expert_multiplier × 专家 transition 总数
    - 20% 容量为专家经验（常驻，不被覆盖）
    - 80% 容量为普通探索经验（环形覆盖）
    - PER: 基于 TD-error 的优先级采样（proportional prioritization）

    Attributes:
        expert_buffer: 固定大小的专家 transition 列表（常驻）
        expert_capacity: 专家经验容量
        normal_capacity: 普通经验容量
        expert_ratio: 每次采样中专家经验的占比
        per_alpha: PER α 参数（0 = 均匀采样，1 = 完全按优先级）
        per_beta: 重要性采样 β 参数（0 = 无修正，1 = 完全修正）
        priorities: 每条经验的 TD-error 优先级数组
        _normal_position: 普通 buffer 写入位置
    """

    def __init__(
        self,
        expert_capacity: int,
        normal_capacity: int,
        expert_ratio: float = 0.2,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
    ):
        """初始化专家回放缓冲区。

        Args:
            expert_capacity: 专家经验容量（常驻，不可覆盖）
            normal_capacity: 普通经验容量（环形覆盖）
            expert_ratio: 每次采样中专家经验占比（0.0 ~ 1.0）
            per_alpha: PER α 参数，控制优先级影响程度
            per_beta: 重要性采样 β 参数初始值
        """
        total_capacity = expert_capacity + normal_capacity
        super().__init__(capacity=total_capacity)

        self.expert_capacity = expert_capacity
        self.normal_capacity = normal_capacity
        self.expert_ratio = expert_ratio
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = 0.001  # 每次采样 β 递增量（向 1.0 收敛）

        # 专家经验独立存储（常驻，不可覆盖）
        self.expert_buffer: List[Tuple] = []

        # 普通经验独立存储（预分配 + 环形覆盖），父类 self.buffer 不再使用
        self.normal_buffer: List[Optional[Tuple]] = [None] * normal_capacity
        self._normal_count: int = 0  # 有效条目计数（O(1)，避免每次 O(n) 扫描）

        # 优先级数组：前 expert_capacity 为专家经验，后 normal_capacity 为普通经验
        self.priorities = np.zeros(total_capacity, dtype=np.float32)

        # 普通 buffer 写入位置（相对于 normal_capacity）
        self._normal_position = 0

        # 最近一次采样索引（供 update_priorities 回写 TD-error）
        self._last_sampled_indices: List[int] = []
        self._last_is_weights: np.ndarray = np.array([], dtype=np.float32)

        # 最小优先级（防止 0 优先级导致永不被采样）
        self._epsilon = 1e-6

    # ==================== 专家经验管理 ====================

    def load_expert_transitions(self, expert_transitions: List[Tuple]):
        """加载专家 transition 到常驻区。

        专家经验一旦加载就不可被覆盖。如果数量超过 expert_capacity，
        随机采样 expert_capacity 条保留。

        Args:
            expert_transitions: 专家 transition 列表
        """
        if len(expert_transitions) > self.expert_capacity:
            indices = np.random.choice(
                len(expert_transitions), size=self.expert_capacity, replace=False
            )
            self.expert_buffer = [expert_transitions[i] for i in indices]
        else:
            self.expert_buffer = list(expert_transitions)

        # 专家区为均匀采样，无需初始化优先级（priorities 仅普通区使用）

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"ExpertReplayBuffer: loaded {len(self.expert_buffer)} expert transitions "
            f"(capacity={self.expert_capacity})"
        )

    # ==================== 普通经验管理 ====================

    def add_normal(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        mask: Optional[np.ndarray],
        done: bool,
        step_id: int,
        next_step_id: int,
    ):
        """添加一条普通（探索）transition。

        写入到 normal buffer 的预分配数组中，环形覆盖。

        Args:
            obs: 当前 Dict 观测
            action: 选中的动作索引
            reward: 延迟 reward
            next_obs: 下一步 Dict 观测
            mask: 当前子步骤的动作掩码
            done: 是否终止
            step_id: 当前子步骤编号
            next_step_id: 下个子步骤编号
        """
        data = (obs, action, reward, next_obs, mask, done, step_id, next_step_id)
        was_empty = self.normal_buffer[self._normal_position] is None
        self.normal_buffer[self._normal_position] = data
        if was_empty and self._normal_count < self.normal_capacity:
            self._normal_count += 1

        # 新经验的初始优先级设为当前最大值（鼓励新经验被学习）
        max_prio = self.priorities.max() if self.priorities.max() > 0 else 1.0
        priority_idx = self.expert_capacity + self._normal_position
        self.priorities[priority_idx] = max_prio

        self._normal_position = (self._normal_position + 1) % self.normal_capacity

    def add(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        mask: Optional[np.ndarray],
        done: bool,
        step_id: int,
        next_step_id: int,
    ):
        """重写 add()，将探索经验写入普通区域。

        Args:
            obs: 当前 Dict 观测
            action: 选中的动作索引
            reward: 延迟 reward
            next_obs: 下一步 Dict 观测
            mask: 当前子步骤的动作掩码
            done: 是否终止
            step_id: 当前子步骤编号
            next_step_id: 下个子步骤编号
        """
        self.add_normal(obs, action, reward, next_obs, mask, done, step_id, next_step_id)

    # ==================== PER 采样 ====================

    def sample(self, batch_size: int):
        """整池均匀随机采样（专家常驻区全量 + 探索区有效槽）。

        设计：专家常驻区容量 = n_expert（目标 70000，永远全量有效），探索区容量 =
        EXPERT_EXPLORE_CAPACITY（30000，环形覆盖、只挤掉本区旧值）。对整池做均匀
        随机采样后，因专家区占 7/10，自然得到约 7:3 组成——无需在采样时强制比例
        （与「抽的时候不严格按比例」一致）。IS 权重统一 = 1.0（无 PER 修正）。

        采样后可通过 get_last_sampled_indices() / get_last_is_weights()
        获取本次采样的索引和 IS 权重，用于 TD-error 回写（此处 IS 恒为 1.0）。

        Args:
            batch_size: 采样数量

        Returns:
            8 元组 (obs_dict, actions, rewards, next_obs_dict, masks, dones, step_ids, next_step_ids)
        """
        # 候选全局索引池：专家区全量（常驻）+ 探索区已填充的有效槽
        expert_idx = list(range(len(self.expert_buffer)))
        explore_local = [i for i, x in enumerate(self.normal_buffer) if x is not None]
        explore_idx = [self.expert_capacity + i for i in explore_local]

        pool = expert_idx + explore_idx
        if len(pool) == 0:
            raise RuntimeError(
                f"ExpertReplayBuffer.sample({batch_size}): 缓冲区为空，无法采样 "
                f"(expert={len(self.expert_buffer)}, normal={self._normal_count})"
            )

        # 池小于批次时允许放回采样，避免取空报错（小批次对 DQN 训练无害）
        replace = len(pool) < batch_size
        chosen = np.random.choice(pool, size=batch_size, replace=replace).tolist()

        # 记录本次采样索引（供 update_priorities 回写 TD-error，本采样未用 PER 权重）
        self._last_sampled_indices = chosen
        self._last_is_weights = np.ones(batch_size, dtype=np.float32)  # 均匀采样无 IS 修正

        # 从 buffer 取数据
        batch = []
        for idx in chosen:
            if idx < len(self.expert_buffer):
                batch.append(self.expert_buffer[idx])
            else:
                batch.append(self.normal_buffer[idx - self.expert_capacity])

        obs_list, actions, rewards, next_obs_list, masks, dones, step_ids, next_step_ids = zip(*batch)

        result = (
            self._batch_dict_obs(obs_list),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            self._batch_dict_obs(next_obs_list),
            list(masks),
            np.array(dones, dtype=np.float32),
            np.array(step_ids, dtype=np.int64),
            np.array(next_step_ids, dtype=np.int64),
        )

        return result

    def _compute_is_weights(self, probs: np.ndarray, total_n: int) -> np.ndarray:
        """计算 PER 重要性采样权重。

        w_i = (1 / (N * P(i))) ^ β

        Args:
            probs: 采样概率数组
            total_n: 总经验数

        Returns:
            重要性采样权重数组（已归一化到 max=1.0）
        """
        weights = (1.0 / (total_n * probs + self._epsilon)) ** self.per_beta
        # 归一化：max=1.0 以保证梯度稳定
        weights /= max(weights.max(), self._epsilon)
        return weights.astype(np.float32)

    # ==================== 优先级更新 ====================

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """根据 TD-error 更新经验优先级（仅普通区，专家区为均匀采样）。

        Args:
            indices: 经验在 buffer 中的索引列表
            td_errors: 对应的 TD-error 绝对值
        """
        for idx, td_err in zip(indices, td_errors):
            # 专家区索引 (< expert_capacity) 为均匀采样，跳过优先级更新
            if self.expert_capacity <= idx < len(self.priorities):
                self.priorities[idx] = abs(float(td_err)) + self._epsilon

    # ==================== 兼容性方法 ====================

    def get_last_sampled_indices(self) -> List[int]:
        """返回最近一次 sample() 的采样索引列表（供 update_priorities 回写 TD-error）。"""
        return self._last_sampled_indices

    def get_last_is_weights(self) -> np.ndarray:
        """返回最近一次 sample() 的重要性采样权重（供 learn() 中 IS 修正）。"""
        return self._last_is_weights

    def __len__(self) -> int:
        return len(self.expert_buffer) + self._normal_count

    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够样本进行训练。"""
        return len(self) >= batch_size

    def get_expert_count(self) -> int:
        """返回专家经验数量。"""
        return len(self.expert_buffer)

    def get_normal_count(self) -> int:
        """返回普通经验数量。"""
        return self._normal_count
