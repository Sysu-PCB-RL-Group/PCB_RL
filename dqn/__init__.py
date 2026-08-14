"""DQN 自回归模块。

基于 SPEC v2.2 §4.4-4.5 实现 5-Head DQN 网络及其训练组件。
"""

from .network import DQN
from .replay_buffer import ReplayBuffer, ExpertReplayBuffer
from .agent import DQNAgent

__all__ = ['DQN', 'ReplayBuffer', 'ExpertReplayBuffer', 'DQNAgent']
