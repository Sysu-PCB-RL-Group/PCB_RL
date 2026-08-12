"""训练循环 Runner：自回归 5 步 DQN + epsilon-greedy + curriculum learning。
跨板子随机选板训练，Agent/Buffer 全局共享。

基于 SPEC v2.2 §4.5、§10。
"""

import glob
import logging
import json
import math
import os
import random
import shutil
from datetime import datetime
from typing import Optional, Tuple

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

        加载配置文件、设置随机种子。Agent/Buffer 跨板子共享。

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
        self.save_dir = getattr(args, "save_dir", "./result")
        os.makedirs(self.save_dir, exist_ok=True)

        # 全局 episode 计数器
        self.episode_num: int = 0

        # ---- 板子池（run() 中填充） ----
        self.pcb_pool: list[tuple[str, str]] = []  # [(pcb_name, filepath), ...]
        self.envs: dict[str, PCBLayoutEnv] = {}     # {pcb_name: env}

        # ---- Per-board 独立统计 ----
        self.per_board_episode: dict[str, int] = {}       # {pcb_name: ep_count}
        self.per_board_best_reward: dict[str, float] = {} # {pcb_name: best_r}
        self.per_board_best_state: dict[str, dict] = {}   # {pcb_name: snapshot}
        self.per_board_best_episode: dict[str, int] = {}  # {pcb_name: best_ep}
        self.per_board_rewards: dict[str, list] = {}      # {pcb_name: [r1, r2, ...]}
        self.per_board_placed: dict[str, list] = {}       # {pcb_name: [p1, p2, ...]}
        self.per_board_loss_history: dict[str, list] = {} # {pcb_name: [(ep, loss), ...]}

        # ---- 全局最佳（用于保存 best_model.pt） ----
        self.global_best_reward: float = float("-inf")
        self.global_best_episode: int = -1
        self.global_best_pcb: str = ""

        # 各个 PCB 的训练统计（用于最终汇总）
        self._pcb_history: list[dict] = []

    def run(self):
        """主训练循环：跨板子随机选板，Agent/Buffer 共享。"""
        # 文件日志
        log_path = os.path.join(self.save_dir, "training_log.txt")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_path}")

        # ---- 1. 扫描全部 PCB 文件 ----
        pcb_files = self._get_pcb_files()
        if not pcb_files:
            logger.error(f"No PCB files found in {self.data_dir}")
            return

        logger.info(f"Found {len(pcb_files)} PCB files:")
        for f in pcb_files:
            logger.info(f"  {f}")

        logger.info(
            f"Config: MAX_MOVABLE={Config.MAX_MOVABLE}, "
            f"GRID_W={Config.GRID_W}, GRID_H={Config.GRID_H}"
        )

        self.total_episodes = getattr(self.args, "total_episodes", Config.TOTAL_EPISODES)

        # ---- 2. 预加载全部 env ----
        for file_path in pcb_files:
            pcb_name = os.path.basename(file_path)
            try:
                env = self._create_env(file_path)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"跳过无法加载的板: {pcb_name}: {e}")
                continue

            if env.N_movable < 1 or len(env.kicad.nets) < 1:
                logger.warning(
                    f"跳过非电路板: {pcb_name} "
                    f"(N_movable={env.N_movable}, nets={len(env.kicad.nets)})"
                )
                continue

            self.envs[pcb_name] = env
            self.pcb_pool.append((pcb_name, file_path))

        if not self.pcb_pool:
            logger.error("No valid PCB boards loaded")
            return

        logger.info(f"Loaded {len(self.pcb_pool)} valid boards")

        # ---- 清理旧 per-board 记录（避免多次训练结果混叠） ----
        for pcb_name, _ in self.pcb_pool:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            board_dir = os.path.join(self.save_dir, pcb_short)
            if os.path.isdir(board_dir):
                shutil.rmtree(board_dir)
                logger.info(f"Cleaned old board dir: {board_dir}")
        # 清理旧的汇总文件
        for fname in ("training_summary.json",):
            fpath = os.path.join(self.save_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

        # ---- 3. 初始化 per-board 状态 ----
        for pcb_name, _ in self.pcb_pool:
            self.per_board_episode[pcb_name] = 0
            self.per_board_best_reward[pcb_name] = float("-inf")
            self.per_board_best_state[pcb_name] = {}
            self.per_board_best_episode[pcb_name] = -1
            self.per_board_rewards[pcb_name] = []
            self.per_board_placed[pcb_name] = []
            self.per_board_loss_history[pcb_name] = []

        # ---- 4. 创建共享 Agent / Buffer ----
        agent, buffer = self._init_agent_and_buffer()

        # ---- 5. 备份旧 best_model.pt ----
        self._backup_previous_results()

        pcb_names = [n for n, _ in self.pcb_pool]

        # ---- 6. 预热期：总共 500 轮，随机选板，纯探索不学习 ----
        WARMUP_TOTAL = 500
        logger.info(
            f"Warming up: {WARMUP_TOTAL} ep (random board, ε=1.0, no learning)"
        )
        for _ in range(WARMUP_TOTAL):
            pcb_name = random.choice(pcb_names)
            env = self.envs[pcb_name]
            self._run_episode(env, agent, buffer, pcb_name, ep_num=0, learn=False)
        logger.info(f"Warmup complete ({len(buffer)} transitions in buffer)")

        # ---- 7. 正式训练：每 episode 随机选板 ----
        logger.info(f"Training: {self.total_episodes} episodes (random board selection)")
        logger.info(f"Global best will be saved to: {os.path.join(self.save_dir, 'best_model.pt')}")

        for ep in range(self.total_episodes):
            self.episode_num += 1

            # 随机选板
            pcb_name = random.choice(pcb_names)
            env = self.envs[pcb_name]
            self.per_board_episode[pcb_name] += 1
            board_ep = self.per_board_episode[pcb_name]

            # 运行一个 episode
            ep_reward, ep_steps, ep_placed, avg_loss = self._run_episode(
                env, agent, buffer, pcb_name,
                ep_num=board_ep, learn=True,
            )
            self.per_board_rewards[pcb_name].append(ep_reward)
            self.per_board_placed[pcb_name].append(ep_placed)
            self.per_board_loss_history[pcb_name].append((board_ep, avg_loss))

            # 检查 per-board 最佳
            if ep_reward > self.per_board_best_reward[pcb_name]:
                self.per_board_best_reward[pcb_name] = ep_reward
                self.per_board_best_state[pcb_name] = self._snapshot_state(env)
                self.per_board_best_episode[pcb_name] = self.episode_num

            # 检查全局最佳 → 覆盖保存 best_model.pt
            if ep_reward > self.global_best_reward:
                self.global_best_reward = ep_reward
                self.global_best_episode = self.episode_num
                self.global_best_pcb = pcb_name
                ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                agent.save(ckpt_path)
                logger.info(
                    f"[NEW GLOBAL BEST] R={self.global_best_reward:.2f} "
                    f"board={pcb_name} ep={self.global_best_episode} → {ckpt_path}"
                )

            # 日志
            self._log_episode(pcb_name, ep_reward, ep_steps, ep_placed, avg_loss)

            # 定期画 per-board loss 曲线
            if board_ep % 100 == 0:
                self._plot_loss_curve(pcb_name)

        # ---- 8. 训练结束：per-board 分析 & 全局汇总 ----
        for pcb_name in pcb_names:
            self._plot_loss_curve(pcb_name)
            best_state = self.per_board_best_state.get(pcb_name)
            if best_state:
                self._export_best_layout(pcb_name, best_state,
                                         self.per_board_best_reward[pcb_name],
                                         self.per_board_best_episode[pcb_name])

            self._write_pcb_summary(pcb_name)

        # 保存最终 checkpoint
        if getattr(self.args, "explicit_total_episodes", False):
            ckpt_path = os.path.join(self.save_dir, f"final_ep{self.episode_num}.pt")
            agent.save(ckpt_path)
            logger.info(f"Final checkpoint saved: {ckpt_path}")

        self._write_final_summary()

        # 清理
        for env in self.envs.values():
            env.close()

        logger.removeHandler(file_handler)
        file_handler.close()

    # ==================== 单个 episode ====================

    def _run_episode(
        self, env: PCBLayoutEnv, agent: DQNAgent,
        buffer: ReplayBuffer, pcb_name: str,
        ep_num: int = 0,
        learn: bool = True,
    ):
        """执行单个 episode：5 步自回归 + 写入 buffer（+ 可选 learn）。

        Args:
            env: PCB 布局环境
            agent: DQN Agent
            buffer: 经验回放缓冲区
            pcb_name: PCB 文件名
            ep_num: 该板子当前的 episode 计数（用于 epsilon/curriculum）
            learn: 是否执行学习（预热期 = False）

        Returns:
            (total_reward, step_count, placed_count, avg_loss)
        """
        epsilon = self._compute_epsilon(pcb_name)
        env.W_OUT_OF_BOUNDS_CURRENT = self._compute_curriculum_weight(pcb_name)
        env.current_episode = ep_num  # quality_scale 升温依据

        env.reset()
        env._reset_pending()
        done = False
        episode_reward = 0.0
        step_count = 0
        placed_count = 0
        losses = []

        while not done:
            # ---- 5 步自回归决策 ----
            transitions = []  # [(obs, action, mask, step_id), ...]

            for step_id in range(5):
                mask = env._build_mask_for_step(step_id)
                obs_with_pending = env._get_obs_with_pending(step_id)
                action = agent.select_action(obs_with_pending, mask, step_id, epsilon)
                env._update_pending(action, step_id)
                transitions.append((obs_with_pending, action, mask, step_id))

            # ---- 组装 7 元动作 → env.step() ----
            # 训练不使用 clamp：Agent 通过越界惩罚自然学习合法坐标。
            # env.step() 内部已有 bounds 校验和 W_OUT_OF_BOUNDS 惩罚机制。
            full_action = env._assemble_action()
            obs_next, reward, terminated, truncated, _info = env.step(full_action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            placed_count = _info.get("placed_count", 0)

            # ---- 写入 buffer（5 步按比例共享 reward） ----
            share_weights = [0.2, 0.3, 0.3, 0.1, 0.1]
            for i, (s, a, mask, sid) in enumerate(transitions):
                if sid < 4:
                    next_s = transitions[i + 1][0]
                    next_sid = sid + 1
                    step_done = False
                else:
                    next_s = obs_next
                    next_sid = 0 if not done else -1
                    step_done = done

                step_reward = reward * share_weights[sid]
                buffer.add(s, a, step_reward, next_s, mask, step_done, sid, next_sid)

            # ---- 每步 learn（预热期跳过） ----
            if learn:
                if buffer.is_ready(Config.BATCH_SIZE):
                    batch = buffer.sample(Config.BATCH_SIZE)
                    loss = agent.learn(batch)
                    losses.append(loss)

                    if agent.train_steps % Config.TARGET_UPDATE_FREQ == 0:
                        agent.sync_target()

            env._reset_pending()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return episode_reward, step_count, placed_count, avg_loss

    # ==================== 训练机制 ====================

    def _compute_epsilon(self, pcb_name: str) -> float:
        """跨 episode 指数衰减 epsilon（per-board 独立计数）。

        Args:
            pcb_name: 板子名

        Returns:
            当前 epsilon 值，不低于 EPSILON_MIN
        """
        eps_min = Config.EPSILON_MIN
        eps_start = Config.EPSILON_START
        t = self.per_board_episode.get(pcb_name, 0)
        if t <= 0:
            return eps_start
        return eps_min + (eps_start - eps_min) * math.exp(-4.0 * t / Config.EPSILON_DECAY_EPISODES)

    def _compute_curriculum_weight(self, pcb_name: str) -> float:
        """跨 episode 线性递增出界惩罚权重（per-board 独立）。

        Args:
            pcb_name: 板子名

        Returns:
            当前出界惩罚权重
        """
        start_ep = Config.CURRICULUM_START_EP
        end_ep = Config.CURRICULUM_END_EP
        t = self.per_board_episode.get(pcb_name, 0)

        progress = max(0.0, min(1.0,
            (t - start_ep) / max(end_ep - start_ep, 1)
        ))
        return (
            Config.W_OUT_OF_BOUNDS_INITIAL
            + progress * (Config.W_OUT_OF_BOUNDS_FINAL - Config.W_OUT_OF_BOUNDS_INITIAL)
        )

    # ==================== 工具方法 ====================

    def _init_agent_and_buffer(self):
        """创建全局共享的 DQN Agent 和 ReplayBuffer。

        网络使用 Dict 观测（slot encoder 共享权重），Head 维度 = MAX 常量，
        变长逻辑通过 mask 实现。

        Returns:
            (DQNAgent, ReplayBuffer) 元组
        """
        agent = DQNAgent(
            n_movable=Config.MAX_MOVABLE,
            grid_w=Config.GRID_W,
            grid_h=Config.GRID_H,
            device=self.device,
            lr=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
        )
        buffer = ReplayBuffer(capacity=Config.REPLAY_BUFFER_SIZE)
        logger.info(
            f"Agent created (shared): Dict obs (slots={Config.MAX_MOVABLE}×8 + global=8), "
            f"head_dims=({Config.MAX_MOVABLE},{Config.GRID_W},{Config.GRID_H}), "
            f"device={self.device}"
        )
        return agent, buffer

    def _create_env(self, pcb_path: str):
        """创建环境实例。

        Args:
            pcb_path: PCB 文件路径（.kicad_pcb）

        Returns:
            已加载 PCB 的 PCBLayoutEnv 实例
        """
        config_path = getattr(self.args, "config_path", None)
        return PCBLayoutEnv(pcb_path, config_path=config_path)

    def _get_pcb_files(self) -> list:
        """递归扫描数据目录获取所有 PCB 文件（委托到 utils.pcb_scan）。"""
        from utils.pcb_scan import find_pcb_files
        return find_pcb_files(self.data_dir)

    def _log_episode(
        self, pcb_name: str, total_reward: float,
        steps: int, placed: int, loss: float,
    ):
        """记录单个 episode 的统计信息。

        Args:
            pcb_name: PCB 文件名
            total_reward: episode 累计 reward
            steps: 环境 step 次数
            placed: 已放置元件数
            loss: 学习 loss（0.0 表示未训练）
        """
        log_interval = Config.LOG_INTERVAL
        if self.episode_num % log_interval != 0:
            return

        epsilon = self._compute_epsilon(pcb_name)
        board_short = pcb_name.replace(".kicad_pcb", "")
        logger.info(
            f"[Ep {self.episode_num:6d}] {board_short:30s} "
            f"({self.per_board_episode.get(pcb_name, 0):4d}@board) "
            f"R={total_reward:8.2f}  steps={steps:3d}  placed={placed:3d}  "
            f"ε={epsilon:.3f}  loss={loss:.4f}"
        )

    def _backup_previous_results(self):
        """训练开始前备份旧 best_model.pt，最多保留 5 份。"""
        best_path = os.path.join(self.save_dir, "best_model.pt")
        if not os.path.isfile(best_path):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.save_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        backup_name = f"best_model_{timestamp}.pt"
        dst = os.path.join(backup_dir, backup_name)
        shutil.move(best_path, dst)
        logger.info(f"Backed up: best_model.pt → backup/{backup_name}")

        # 清理旧备份：超过 5 份删最旧的
        all_backups = sorted(glob.glob(os.path.join(backup_dir, "best_model_*.pt")))
        while len(all_backups) > 5:
            oldest = all_backups[0]
            try:
                os.remove(oldest)
                logger.info(f"Removed old backup: backup/{os.path.basename(oldest)}")
            except OSError:
                pass
            all_backups.pop(0)

    def _plot_loss_curve(self, pcb_name: str):
        """生成 Loss-Episode 曲线图，保存到 result/<board>/ 子目录。

        Args:
            pcb_name: PCB 文件名
        """
        from utils.visualization import plot_loss_curve

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        board_dir = os.path.join(self.save_dir, pcb_short)
        os.makedirs(board_dir, exist_ok=True)

        loss_history = self.per_board_loss_history.get(pcb_name, [])
        json_path = os.path.join(board_dir, "loss_history.json")
        png_path = os.path.join(board_dir, "loss_curve.png")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, ensure_ascii=False)

        plot_loss_curve(loss_history, png_path, pcb_name=pcb_short, window=100)

    # ==================== 最佳布局追踪 ====================

    def _snapshot_state(self, env: PCBLayoutEnv) -> dict:
        """提取 env 中所有元件的完整状态快照。"""
        from scripts.run_inference import snapshot_state
        return snapshot_state(env)

    def _export_best_layout(self, pcb_name: str, best_state: dict,
                            best_reward: float, best_episode: int):
        """导出训练中的最佳布局到 result/<board>/best_layout/ 目录。

        Args:
            pcb_name: PCB 文件名
            best_state: snapshot_state 返回值
            best_reward: 对应的 reward
            best_episode: 对应的 episode 编号
        """
        if not best_state:
            logger.warning(f"No best layout to export for {pcb_name}")
            return

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        output_dir = os.path.join(self.save_dir, pcb_short, "best_layout")
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "best.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "episode": best_episode,
                "reward": best_reward,
                **best_state,
            }, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Best layout exported: {json_path} "
            f"(ep={best_episode}, R={best_reward:.2f})"
        )

    def _write_pcb_summary(self, pcb_name: str):
        """单个 PCB 训练完成后写汇总日志。"""
        rewards = self.per_board_rewards.get(pcb_name, [])
        placed_list = self.per_board_placed.get(pcb_name, [])
        best_r = self.per_board_best_reward.get(pcb_name, float("-inf"))
        best_ep = self.per_board_best_episode.get(pcb_name, -1)
        env = self.envs.get(pcb_name)
        n_movable = env.N_movable if env else 0

        if not rewards:
            return

        first_n = max(1, len(rewards) // 5)
        last_n = max(1, len(rewards) // 5)

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        logger.info("=" * 50)
        logger.info(f"PCB 训练完成: {pcb_short}")
        logger.info(f"  训练轮数: {len(rewards)}")
        logger.info(f"  元件数: {n_movable}")
        logger.info(f"  前 20% 平均 R: {sum(rewards[:first_n]) / first_n:.2f}")
        logger.info(f"  后 20% 平均 R: {sum(rewards[-last_n:]) / last_n:.2f}")
        logger.info(f"  最终 5 轮 R: {[f'{r:.1f}' for r in rewards[-5:]]}")
        logger.info(f"  前 20% 平均 placed: {sum(placed_list[:first_n]) / first_n:.1f}")
        logger.info(f"  后 20% 平均 placed: {sum(placed_list[-last_n:]) / last_n:.1f}")
        logger.info(f"  最佳 R: {best_r:.2f} (ep {best_ep})")

        self._pcb_history.append({
            "pcb": pcb_name,
            "n_movable": n_movable,
            "episodes": len(rewards),
            "best_reward": best_r,
            "best_episode": best_ep,
            "first_avg_r": sum(rewards[:first_n]) / first_n,
            "last_avg_r": sum(rewards[-last_n:]) / last_n,
            "first_avg_placed": sum(placed_list[:first_n]) / first_n,
            "last_avg_placed": sum(placed_list[-last_n:]) / last_n,
        })

        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._pcb_history, f, indent=2, ensure_ascii=False)
        logger.info(f"  汇总已保存: {summary_path}")

    def _write_final_summary(self):
        """全部 PCB 训练完成后写最终汇总。"""
        logger.info("=" * 50)
        logger.info("全部训练完成 — 最终汇总")
        for h in self._pcb_history:
            improved = h["last_avg_r"] > h["first_avg_r"]
            arrow = "↑" if improved else "↓"
            logger.info(
                f"  {h['pcb'][:40]:40s}  "
                f"eps={h['episodes']:5d}  "
                f"R: {h['first_avg_r']:8.1f} → {h['last_avg_r']:8.1f} {arrow}  "
                f"placed: {h['first_avg_placed']:5.1f} → {h['last_avg_placed']:5.1f}  "
                f"best_R={h['best_reward']:.1f}"
            )
        logger.info(
            f"\n全局最佳: R={self.global_best_reward:.2f} "
            f"board={self.global_best_pcb} ep={self.global_best_episode}"
        )
        logger.info(f"全局最佳权重: {os.path.join(self.save_dir, 'best_model.pt')}")
        logger.info(f"日志文件: {os.path.join(self.save_dir, 'training_log.txt')}")
        logger.info(f"汇总 JSON: {os.path.join(self.save_dir, 'training_summary.json')}")
