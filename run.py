"""训练循环 Runner：自回归 5 步 DQN + epsilon-greedy + curriculum learning。

基于 SPEC v2.2 §4.5、§10。
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

from env.pcb_layout_env import PCBLayoutEnv, Config
from dqn import DQNAgent, ReplayBuffer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class Runner:
    """训练循环控制器：自回归 5 步 DQN 训练"""

    # 5 步子步骤名称
    STEP_NAMES = ["comp", "x1", "y1", "rot", "lay"]

    def __init__(self, args):
        """初始化训练控制器。

        加载配置文件、设置随机种子。Agent 和 Buffer 在 _init_agent 中按 PCB 文件维度懒加载。

        Args:
            args: CLI 参数对象（argparse.Namespace）
        """
        self.args = args

        # 从 argument.txt 加载超参数
        config_path = getattr(args, "config_path", None)
        if config_path and os.path.isfile(config_path):
            Config.load_from_file(config_path)

        # CLI 参数覆盖 argument.txt
        if hasattr(args, "max_episode_step") and args.max_episode_step is not None:
            Config.MAX_STEPS = args.max_episode_step
        if hasattr(args, "total_episodes") and args.total_episodes is not None:
            Config.TOTAL_EPISODES = args.total_episodes

        # 随机种子
        seed = getattr(args, "seed", 114514)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )

        # 数据 & 保存目录
        self.data_dir = args.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.save_dir = getattr(args, "save_dir", "./result")
        os.makedirs(self.save_dir, exist_ok=True)

        # 全局 episode 计数器（跨 PCB 文件）
        self.episode_num: int = 0

        # 最佳布局追踪
        self.best_layout_reward: float = float("-inf")
        self.best_layout_state: dict = {}
        self.best_layout_episode: int = -1

    def run(self):
        """主训练循环。

        遍历所有 PCB 文件 → 按 env 维度创建 Agent/Buffer → 训练 total_episodes 轮。
        每个 episode：5 步自回归决策 → 共享 reward 写入 buffer → 每个 env step 后
        buffer 够一个 batch 就采样做一次 BP（DQN off-policy 在线学习）。
        """
        pcb_files = self._get_pcb_files()
        if not pcb_files:
            logger.error(f"No PCB files found in {self.data_dir}")
            return

        logger.info(f"Found {len(pcb_files)} PCB files: {pcb_files}")
        total_episodes = getattr(self.args, "total_episodes", Config.TOTAL_EPISODES)

        for file_path in pcb_files:
            pcb_name = os.path.basename(file_path)
            logger.info(f"===== Training on: {pcb_name} =====")

            env = self._create_env(file_path)
            try:
                # 按当前 env 维度初始化 Agent 和 Buffer
                agent, buffer = self._init_agent_and_buffer(env)

                # 重置本 PCB 的最佳布局追踪
                pcb_best_reward = float("-inf")
                pcb_best_state = None
                pcb_best_episode = -1

                for ep in range(total_episodes):
                    self.episode_num += 1
                    ep_reward, ep_steps, ep_placed, avg_loss = self._run_episode(
                        env, agent, buffer, pcb_name
                    )

                    # 追踪当前 PCB 最佳布局
                    if ep_reward > pcb_best_reward:
                        pcb_best_reward = ep_reward
                        pcb_best_state = self._snapshot_state(env)
                        pcb_best_episode = self.episode_num

                    # 记录 & 保存
                    self._log_episode(
                        pcb_name, ep_reward, ep_steps, ep_placed, avg_loss,
                    )

                    if self.episode_num % Config.SAVE_INTERVAL == 0:
                        self._save_checkpoint(agent, pcb_name)

                # 每块 PCB 训练完后导出其最佳布局
                if pcb_best_state is not None:
                    self.best_layout_reward = pcb_best_reward
                    self.best_layout_state = pcb_best_state
                    self.best_layout_episode = pcb_best_episode
                    self._export_best_layout(pcb_name)
            finally:
                env.close()

    # ==================== 单个 episode ====================

    def _run_episode(
        self, env: PCBLayoutEnv, agent: DQNAgent,
        buffer: ReplayBuffer, pcb_name: str,
    ):
        """执行单个 episode：5 步自回归 + 每步 learn + 写入 buffer。

        DQN 是 off-policy 算法，在当前策略收集数据的同时从 replay buffer
        采样历史 transition 进行 Q-learning。训练粒度 = 每个 env step 后。

        Args:
            env: PCB 布局环境
            agent: DQN Agent
            buffer: 经验回放缓冲区
            pcb_name: PCB 文件名（用于 checkpoint 命名）

        Returns:
            (total_reward, step_count, placed_count, avg_loss)
        """
        epsilon = self._compute_epsilon()
        env.W_OUT_OF_BOUNDS_CURRENT = self._compute_curriculum_weight()

        env.reset()
        env._reset_pending()
        done = False
        episode_reward = 0.0
        step_count = 0
        placed_count = 0
        losses = []  # 每步 learn 的 loss

        while not done:
            # ---- 5 步自回归决策 ----
            # 注意：obs 直接从 env._get_obs_with_pending() 读取 env 内部状态，
            # 不需要从 env.reset() 或 env.step() 的返回值传递
            transitions = []  # [(obs, action, mask, step_id), ...]

            for step_id in range(5):
                mask = env._build_mask_for_step(step_id)
                obs_with_pending = env._get_obs_with_pending(step_id)
                action = agent.select_action(obs_with_pending, mask, step_id, epsilon)
                env._update_pending(action, step_id)
                transitions.append((obs_with_pending, action, mask, step_id))

            # ---- 组装 7 元动作 → env.step() ----
            full_action = env._assemble_action()
            obs_next, reward, terminated, truncated, _info = env.step(full_action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            placed_count = _info.get("placed_count", 0)

            # ---- 写入 buffer（共享 reward，中间步 reward=0） ----
            for i, (s, a, mask, sid) in enumerate(transitions):
                # 确定 next_obs 和 next_step_id
                if sid < 4:
                    # step 0-3: 下个状态是下一个子步骤的观测
                    next_s = transitions[i + 1][0]  # 下条 transition 的 obs
                    next_sid = sid + 1
                    step_done = False
                else:
                    # step 4: 下个状态是 env 返回的布局观测（pending 已重置）
                    next_s = obs_next
                    next_sid = 0 if not done else -1
                    step_done = done

                # 仅 step 4 获得环境 reward，中间步 reward=0
                step_reward = reward if sid == 4 else 0.0

                buffer.add(s, a, step_reward, next_s, mask, step_done, sid, next_sid)

            # ---- 每步 learn：buffer 攒够就采样训练 ----
            if buffer.is_ready(Config.BATCH_SIZE):
                batch = buffer.sample(Config.BATCH_SIZE)
                loss = agent.learn(batch)
                losses.append(loss)

                # 定期同步 target 网络
                if agent.train_steps % Config.TARGET_UPDATE_FREQ == 0:
                    agent.sync_target()

            # 重置 pending，准备下一颗元件放置
            env._reset_pending()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return episode_reward, step_count, placed_count, avg_loss

    # ==================== 训练机制 ====================

    def _compute_epsilon(self) -> float:
        """跨 episode 线性衰减 epsilon。

        Returns:
            当前 epsilon 值，不低于 EPSILON_END
        """
        eps_start = Config.EPSILON_START
        eps_end = Config.EPSILON_END
        decay_eps = Config.EPSILON_DECAY_EPISODES
        decay = (eps_start - eps_end) / max(decay_eps, 1)
        return max(eps_end, eps_start - self.episode_num * decay)

    def _compute_curriculum_weight(self) -> float:
        """跨 episode 线性递增出界惩罚权重。

        Returns:
            当前出界惩罚权重，从 W_OUT_OF_BOUNDS_INITIAL → W_OUT_OF_BOUNDS_FINAL
        """
        start_ep = Config.CURRICULUM_START_EP
        end_ep = Config.CURRICULUM_END_EP

        progress = max(0.0, min(1.0,
            (self.episode_num - start_ep) / max(end_ep - start_ep, 1)
        ))
        return (
            Config.W_OUT_OF_BOUNDS_INITIAL
            + progress * (Config.W_OUT_OF_BOUNDS_FINAL - Config.W_OUT_OF_BOUNDS_INITIAL)
        )

    # ==================== 工具方法 ====================

    def _init_agent_and_buffer(self, env: PCBLayoutEnv):
        """根据 env 尺寸创建 DQN Agent 和 ReplayBuffer。

        obs_dim = N_movable * 8 (raw) + N_movable (comp onehot) + 4 (gx1, gy1, rot, lay)

        Args:
            env: 已加载 PCB 的环境实例

        Returns:
            (DQNAgent, ReplayBuffer) 元组
        """
        obs_dim = env.N_movable * 8 + env.N_movable + 4
        agent = DQNAgent(
            obs_dim=obs_dim,
            n_movable=env.N_movable,
            grid_w=env.grid_width,
            grid_h=env.grid_height,
            device=self.device,
            lr=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
        )
        buffer = ReplayBuffer(capacity=Config.REPLAY_BUFFER_SIZE)
        logger.info(
            f"Agent created: obs_dim={obs_dim}, N_movable={env.N_movable}, "
            f"grid=({env.grid_width},{env.grid_height}), device={self.device}"
        )
        return agent, buffer

    def _create_env(self, pcb_path: str) -> PCBLayoutEnv:
        """创建环境实例。

        Args:
            pcb_path: PCB 文件路径

        Returns:
            已加载 PCB 的 PCBLayoutEnv 实例
        """
        config_path = getattr(self.args, "config_path", None)
        return PCBLayoutEnv(pcb_path, config_path=config_path)

    def _get_pcb_files(self) -> list:
        """递归扫描数据目录获取所有 PCB 文件。

        每个 KiCad 项目在独立子目录中，递归搜索。
        过滤掉名称含 测试/test/_step/_export 的衍生文件。

        Returns:
            排序后的 PCB 文件绝对路径列表
        """
        files = []
        if not os.path.isdir(self.data_dir):
            return files
        for root, dirs, fnames in os.walk(self.data_dir):
            for f in fnames:
                if not f.lower().endswith((".kicad_pcb", ".pcb")):
                    continue
                # 过滤衍生文件
                if any(kw in f for kw in ("测试", "test", "_step", "_export", "_manual", "_roundtrip")):
                    continue
                files.append(os.path.join(root, f))
        return sorted(files)

    def _log_episode(
        self, pcb_name: str, total_reward: float,
        steps: int, placed: int, loss: float,
    ):
        """记录单个 episode 的统计信息。

        Args:
            pcb_name: PCB 文件名
            total_reward: episode 累计 reward
            steps: 环境 step 次数（元件放置数）
            placed: 已放置元件数
            loss: 学习 loss（0.0 表示未训练）
        """
        log_interval = Config.LOG_INTERVAL
        if self.episode_num % log_interval != 0:
            return

        epsilon = self._compute_epsilon()
        logger.info(
            f"[Ep {self.episode_num:6d}] {pcb_name:30s} "
            f"R={total_reward:8.2f}  steps={steps:3d}  placed={placed:3d}  "
            f"ε={epsilon:.3f}  loss={loss:.4f}"
        )

    def _save_checkpoint(self, agent: DQNAgent, pcb_name: str):
        """保存模型检查点。

        Args:
            agent: DQN Agent
            pcb_name: PCB 文件名（用于命名）
        """
        ckpt_path = os.path.join(
            self.save_dir,
            f"{pcb_name.replace('.kicad_pcb', '')}_ep{self.episode_num}.pt",
        )
        agent.save(ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

    # ==================== 最佳布局追踪 ====================

    def _snapshot_state(self, env: PCBLayoutEnv) -> dict:
        """提取 env 中所有元件的完整状态快照。（复用 scripts/run_inference.py）"""
        # TODO: 未来统一提取到 utils/snapshot.py 共享模块
        from scripts.run_inference import snapshot_state
        return snapshot_state(env)

    def _export_best_layout(self, pcb_name: str):
        """导出训练中的最佳布局到 output/ 目录。

        将 self.best_layout_state 写入 JSON，并可选调用可视化。

        Args:
            pcb_name: PCB 文件名（用于命名）
        """
        if not self.best_layout_state:
            logger.warning("No best layout to export")
            return

        output_dir = os.path.join(
            getattr(self.args, "save_dir", "./result"),
            "best_layout",
        )
        os.makedirs(output_dir, exist_ok=True)

        import json
        json_path = os.path.join(
            output_dir,
            f"{pcb_name.replace('.kicad_pcb', '')}_best.json",
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "episode": self.best_layout_episode,
                "reward": self.best_layout_reward,
                **self.best_layout_state,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Best layout exported: {json_path} "
                     f"(ep={self.best_layout_episode}, R={self.best_layout_reward:.2f})")
