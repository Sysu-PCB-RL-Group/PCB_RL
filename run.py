"""训练循环 Runner：自回归 5 步 DQN + epsilon-greedy + curriculum learning。

基于 SPEC v2.2 §4.5、§10。
"""

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
        self.mock_mode = getattr(args, "mock", False)
        os.makedirs(self.data_dir, exist_ok=True)
        self.save_dir = getattr(args, "save_dir", "./result")
        os.makedirs(self.save_dir, exist_ok=True)

        # 全局 episode 计数器（跨 PCB 文件）
        self.episode_num: int = 0

        # 最佳布局追踪
        self.best_layout_reward: float = float("-inf")
        self.best_layout_state: dict = {}
        self.best_layout_episode: int = -1

        # Loss 历史（用于绘制 loss-episode 曲线）
        self.loss_history: list = []  # [(episode_num, avg_loss), ...]

        # 各个 PCB 的训练统计（用于最终汇总）
        self._pcb_history: list[dict] = []

    def run(self):
        """主训练循环。"""
        # 文件日志（离线查看用）
        log_path = os.path.join(self.save_dir, "training_log.txt")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_path}")

        pcb_files = self._get_pcb_files()
        if not pcb_files:
            logger.error(f"No PCB files found in {self.data_dir}")
            return

        logger.info(f"Found {len(pcb_files)} PCB files: {pcb_files}")
        logger.info(
            f"Config: MAX_MOVABLE={Config.MAX_MOVABLE}, GRID_W={Config.GRID_W}, GRID_H={Config.GRID_H}"
        )
        self.total_episodes = getattr(self.args, "total_episodes", Config.TOTAL_EPISODES)

        for file_path in pcb_files:
            pcb_name = os.path.basename(file_path)

            try:
                env = self._create_env(file_path)
            except (ValueError, RuntimeError) as e:
                # 空板/缺陷板/加载失败：跳过不参与训练
                logger.warning(f"跳过无法加载的板: {pcb_name}: {e}")
                continue

            try:
                # C1: 内容校验——过滤空板/纯机械板（N_movable≥1 且网络数≥1）
                # 否则 DQN 空动作空间（mask 全 False）会导致训练崩
                if env.N_movable < 1 or len(env.kicad.nets) < 1:
                    logger.warning(
                        f"跳过非电路板: {pcb_name} "
                        f"(N_movable={env.N_movable}, nets={len(env.kicad.nets)})"
                    )
                    continue

                logger.info(f"===== Training on: {pcb_name} =====")
                agent, buffer = self._init_agent_and_buffer(env)

                self.pcb_episode_num = 0
                self._backup_previous_results(pcb_name)
                pcb_best_reward = float("-inf")
                pcb_best_state = None
                pcb_best_episode = -1
                ep_rewards = []
                ep_placed_list = []

                for ep in range(self.total_episodes):
                    self.episode_num += 1
                    self.pcb_episode_num += 1
                    ep_reward, ep_steps, ep_placed, avg_loss = self._run_episode(
                        env, agent, buffer, pcb_name
                    )
                    ep_rewards.append(ep_reward)
                    ep_placed_list.append(ep_placed)

                    if ep_reward > pcb_best_reward:
                        pcb_best_reward = ep_reward
                        pcb_best_state = self._snapshot_state(env)
                        pcb_best_episode = self.episode_num
                        self._save_best_checkpoint(agent, pcb_name, pcb_best_reward, pcb_best_episode)

                    self.loss_history.append((self.pcb_episode_num, avg_loss))
                    self._log_episode(
                        pcb_name, ep_reward, ep_steps, ep_placed, avg_loss,
                    )

                    if self.pcb_episode_num % 100 == 0:
                        self._plot_loss_curve(pcb_name)

                self._plot_loss_curve(pcb_name)

                if pcb_best_state is not None:
                    self.best_layout_reward = pcb_best_reward
                    self.best_layout_state = pcb_best_state
                    self.best_layout_episode = pcb_best_episode
                    self._export_best_layout(pcb_name)

                if getattr(self.args, "explicit_total_episodes", False):
                    ckpt_path = os.path.join(
                        self.save_dir,
                        f"{pcb_name.replace('.kicad_pcb', '')}_ep{self.episode_num}.pt",
                    )
                    agent.save(ckpt_path)
                    logger.info(f"Final checkpoint saved: {ckpt_path}")

                # PCB 训练汇总
                self._write_pcb_summary(
                    pcb_name, env.N_movable, ep_rewards, ep_placed_list,
                    pcb_best_reward, pcb_best_episode,
                )
            finally:
                env.close()

        self._write_final_summary()
        logger.removeHandler(file_handler)
        file_handler.close()

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
        env.current_episode = self.pcb_episode_num  # quality_scale 升温依据

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
                # 仅 step 0 需要掩码（筛选已放置元件）
                mask = env._build_mask_for_step(step_id) if step_id == 0 else None
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
        """跨 episode 指数衰减 epsilon（per-PCB 独立计数）。

        曲线：前期快速下降（高探索→利用过渡），后期平缓（精细调优）。
        公式：epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * exp(-4 * t / Emax)
        其中 k=4 保证在 total_episodes 结束时 epsilon 接近 EPSILON_MIN。

        Returns:
            当前 epsilon 值，不低于 EPSILON_MIN
        """
        eps_min = Config.EPSILON_MIN
        eps_start = Config.EPSILON_START
        t = self.pcb_episode_num
        if t <= 0:
            return eps_start
        return eps_min + (eps_start - eps_min) * math.exp(-4.0 * t / self.total_episodes)

    def _compute_curriculum_weight(self) -> float:
        """跨 episode 线性递增出界惩罚权重。

        Returns:
            当前出界惩罚权重，从 W_OUT_OF_BOUNDS_INITIAL → W_OUT_OF_BOUNDS_FINAL
        """
        start_ep = Config.CURRICULUM_START_EP
        end_ep = Config.CURRICULUM_END_EP

        progress = max(0.0, min(1.0,
            (self.pcb_episode_num - start_ep) / max(end_ep - start_ep, 1)
        ))
        return (
            Config.W_OUT_OF_BOUNDS_INITIAL
            + progress * (Config.W_OUT_OF_BOUNDS_FINAL - Config.W_OUT_OF_BOUNDS_INITIAL)
        )

    # ==================== 工具方法 ====================

    def _init_agent_and_buffer(self, env: PCBLayoutEnv):
        """根据固定 MAX 常量创建 DQN Agent 和 ReplayBuffer。

        网络使用 Dict 观测（slot encoder 共享权重），Head 维度 = MAX 常量，
        变长逻辑通过 mask 实现。不同 PCB 共享同一网络结构。

        Args:
            env: 已加载 PCB 的环境实例

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
            f"Agent created: Dict obs (slots={Config.MAX_MOVABLE}×8 + global=8), "
            f"head_dims=({Config.MAX_MOVABLE},{Config.GRID_W},{Config.GRID_H}), "
            f"env(N_movable={env.N_movable}, grid={env.grid_width}x{env.grid_height}), device={self.device}"
        )
        return agent, buffer

    def _create_env(self, pcb_path: str):
        """创建环境实例。

        Args:
            pcb_path: PCB 文件路径（.kicad_pcb 或 .json）

        Returns:
            已加载 PCB 的 PCBLayoutEnv 实例
        """
        if self.mock_mode:
            from env.mock_pcb import create_env_from_json
            return create_env_from_json(pcb_path)
        config_path = getattr(self.args, "config_path", None)
        return PCBLayoutEnv(pcb_path, config_path=config_path)

    def _get_pcb_files(self) -> list:
        """递归扫描数据目录获取所有 PCB 文件。

        KiCad 模式：扫描 .kicad_pcb / .pcb 文件
        Mock 模式：扫描 .json 文件

        Returns:
            排序后的 PCB 文件绝对路径列表
        """
        extensions = (".json",) if self.mock_mode else (".kicad_pcb", ".pcb")
        files = []
        if not os.path.isdir(self.data_dir):
            return files
        for root, dirs, fnames in os.walk(self.data_dir):
            for f in fnames:
                if not f.lower().endswith(extensions):
                    continue
                # 过滤衍生文件
                if any(kw in f for kw in ("测试", "test", "_step", "_export", "_manual", "_roundtrip", "movecall")):
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

    def _save_best_checkpoint(self, agent: DQNAgent, pcb_name: str,
                              best_r: float, best_ep: int):
        """保存当前最佳模型权重（覆盖旧文件，只保留最优）。"""
        pcb_short = pcb_name.replace(".kicad_pcb", "")
        ckpt_path = os.path.join(self.save_dir, f"{pcb_short}_best.pt")
        agent.save(ckpt_path)
        logger.info(f"Best checkpoint saved: {ckpt_path} (R={best_r:.2f}, ep={best_ep})")

    def _backup_previous_results(self, pcb_name: str):
        """训练开始前备份上次的结果文件，避免覆盖丢失。

        Args:
            pcb_name: PCB 文件名
        """
        pcb_short = pcb_name.replace(".kicad_pcb", "")
        files_to_backup = [
            f"{pcb_short}_best.pt",
            f"{pcb_short}_loss_history.json",
            f"{pcb_short}_loss_curve.png",
        ]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.save_dir, "backup", timestamp)
        for fname in files_to_backup:
            src = os.path.join(self.save_dir, fname)
            if os.path.isfile(src):
                os.makedirs(backup_dir, exist_ok=True)
                shutil.move(src, os.path.join(backup_dir, fname))
                logger.info(f"Backed up: {fname} → {backup_dir}/")

    def _plot_loss_curve(self, pcb_name: str):
        """生成 Loss-Episode 曲线图并保存到 save_dir。

        Args:
            pcb_name: PCB 文件名（用于标题和命名）
        """
        from utils.visualization import plot_loss_curve

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        json_path = os.path.join(self.save_dir, f"{pcb_short}_loss_history.json")
        png_path = os.path.join(self.save_dir, f"{pcb_short}_loss_curve.png")

        # 1. 保存原始数据到 JSON（用于后续离线重画）
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.loss_history, f, ensure_ascii=False)

        # 2. 画图
        plot_loss_curve(self.loss_history, png_path, pcb_name=pcb_short, window=100)

    def _regenerate_loss_plot(self, pcb_name: str):
        """从已保存的 JSON 重新生成 loss 曲线图（无需重新训练）。

        Args:
            pcb_name: PCB 文件名
        """
        from utils.visualization import plot_loss_curve

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        json_path = os.path.join(self.save_dir, f"{pcb_short}_loss_history.json")
        png_path = os.path.join(self.save_dir, f"{pcb_short}_loss_curve.png")

        if not os.path.isfile(json_path):
            logger.warning(f"Loss history JSON not found: {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            loss_history = json.load(f)
        plot_loss_curve(loss_history, png_path, pcb_name=pcb_short, window=100)
        logger.info(f"Loss curve regenerated from {json_path}")

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

    def _write_pcb_summary(self, pcb_name: str, n_movable: int,
                           rewards: list, placed_list: list,
                           best_r: float, best_ep: int):
        """单个 PCB 训练完成后写汇总日志。"""
        import json
        first_n = max(1, len(rewards) // 5)
        last_n = max(1, len(rewards) // 5)

        logger.info("=" * 50)
        logger.info(f"PCB 训练完成: {pcb_name}")
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
                f"R: {h['first_avg_r']:8.1f} → {h['last_avg_r']:8.1f} {arrow}  "
                f"placed: {h['first_avg_placed']:5.1f} → {h['last_avg_placed']:5.1f}  "
                f"best_R={h['best_reward']:.1f}"
            )
        logger.info(f"日志文件: {os.path.join(self.save_dir, 'training_log.txt')}")
        logger.info(f"汇总 JSON: {os.path.join(self.save_dir, 'training_summary.json')}")
        logger.info(f"最佳布局: {os.path.join(self.save_dir, 'best_layout')}/")
