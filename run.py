"""训练循环 Runner：两阶段 DQN 训练（BC 预训练 + RL + 专家注入）。

Phase 1: BC 预训练 — 仅用专家 transition 做监督学习（50 epoch）
Phase 2: RL 训练 — 正常 DQN + 20% 专家常驻 buffer + PER 优先经验回放

基于 SPEC v2.2 §4.5、§10，扩展专家轨迹注入方案。
"""

import glob
import time
import logging
import json
import math
import os
import random
import shutil
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import torch

from env.pcb_layout_env import PCBLayoutEnv, Config
from env.expert_trajectory import generate_and_save_expert_transitions
from dqn import DQNAgent
from dqn.replay_buffer import ReplayBuffer, ExpertReplayBuffer

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

        # 路线3：MRV 顺序约束（训练侧改动，需与推理同步开启并重训做对照）
        if getattr(args, "mrv_order", False):
            Config.ENABLE_MRV_ORDER = True
            logger.info(
                "路线3 MRV 顺序约束: ON (step0 只放行剩余合法落子最少的元件)"
            )

        # 随机种子
        seed = getattr(args, "seed", 114514)
        random.seed(seed)
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
        self.per_board_reward_history: dict[str, list] = {} # {pcb_name: [(ep, reward), ...]}

        # ---- 全局最佳（用于保存 best_model.pt） ----
        self.global_best_reward: float = float("-inf")
        self.global_best_episode: int = -1
        self.global_best_pcb: str = ""

        # 各个 PCB 的训练统计（用于最终汇总）
        self._pcb_history: list[dict] = []

        # ---- 专家轨迹注入相关 ----
        self.expert_transitions_dir: str = ""
        self.expert_transitions_count: int = 0  # 专家 transition 总数
        self.use_expert: bool = getattr(args, "use_expert", True)  # 是否启用专家注入
        # 可选：由外层（对照实验调度器）注入的专家目录覆盖值，
        # 优先于 Config.EXPERT_TRANSITIONS_DIR（字符串路径无法经 argument.txt 注入）
        self._expert_transitions_dir_override: str = getattr(
            args, "expert_transitions_dir", "") or ""

        # ---- 失败 step 监控（诊断坏循环：失败类型分布，零行为改动） ----
        # 窗口累计，每次 _log_episode 实际打印后清零（窗口 = LOG_INTERVAL ep，
        # 首窗含 warmup）。by_type[n]=触发该违规的失败 step 数（一次失败可计多种），
        # penalty=累计惩罚绝对值；oob_only=纯出界失败（无重叠/keepout，1a 修复的靶点）。
        self._fail_window: dict = {
            "episodes": 0,          # 窗口内 episode 数
            "fail_steps": 0,        # 失败 step 总数（reward < 0）
            "max_consecutive_fail": 0,  # 同一元件连续失败最长串（坏循环直接证据）
            "oob_only": 0,          # 纯出界失败次数（无重叠/软间距/keepout）
            "fuse_triggered": 0,    # 保险丝触发次数（连败 ≥ 阈值强制终止 episode）
            "by_type": {t: {"n": 0, "penalty": 0.0} for t in
                        ("overlap_hard", "overlap_soft", "out_of_bounds", "in_keepout")},
        }
        self._last_fail_comp: int = -1   # 上次失败的元件索引（连败串跟踪）
        self._consecutive_fail: int = 0

    def run(self):
        """两阶段训练循环：Phase 1 BC 预训练 + Phase 2 RL + 专家注入。"""
        # ---- 备份旧产物（必须在打开新日志文件之前，否则 Windows 锁文件导致 move 失败） ----
        self._backup_previous_results()

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
            self.per_board_reward_history[pcb_name] = []

        pcb_names = [n for n, _ in self.pcb_pool]

        # ---- 5. 生成/加载专家 transition ----
        all_expert_transitions = self._generate_or_load_expert_transitions()
        self.expert_transitions_count = len(all_expert_transitions)
        logger.info(f"Total expert transitions: {self.expert_transitions_count}")

        # ---- 6. Phase 1: BC 预训练 ----
        if self.use_expert and self.expert_transitions_count > 0:
            self._phase1_bc_pretrain(all_expert_transitions, pcb_names)
        else:
            logger.warning("专家注入已禁用或无专家 transition，跳过 BC 预训练")

        # ---- 7. Phase 2: RL 训练 + 专家注入 ----
        if self.use_expert and self.expert_transitions_count > 0:
            agent, buffer = self._phase2_setup(all_expert_transitions, pcb_names)
            agent.init_bc_anchor()  # 冻结 BC 策略锚点，供 RL 阶段 BC 正则化
        else:
            # 回退到标准 ReplayBuffer（不使用专家注入）
            agent, buffer = self._init_agent_and_buffer()
            bc_ckpt_path = os.path.join(self.save_dir, "bc_pretrained.pt")
            if os.path.isfile(bc_ckpt_path):
                agent.load(bc_ckpt_path)
                logger.info(f"Phase 2: loaded BC pretrained weights from {bc_ckpt_path}")

        # ---- 8. Phase 2 预热：混合探索填充 buffer（不学习） ----
        WARMUP_TOTAL = 500
        logger.info(f"Phase 2 warmup: {WARMUP_TOTAL} ep (epsilon=0.5, no learning)")
        # epsilon=0.5：50% BC 模型引导 + 50% 随机探索，预热时不清空 per_board_episode
        for _ in range(WARMUP_TOTAL):
            pcb_name = random.choice(pcb_names)
            env = self.envs[pcb_name]
            ep_reward, ep_steps, ep_placed, _ = self._run_episode(
                env, agent, buffer, pcb_name,
                ep_num=0, learn=False, epsilon=0.5,
            )
            self.per_board_rewards[pcb_name].append(ep_reward)
            self.per_board_placed[pcb_name].append(ep_placed)
        logger.info(f"Warmup complete ({len(buffer)} transitions in buffer)")

        # ---- 9. 正式训练：每 episode 随机选板 ----
        logger.info(f"Phase 2 RL training: {self.total_episodes} episodes (random board selection)")
        logger.info(f"Global best will be saved to: {os.path.join(self.save_dir, 'best_model.pt')}")

        # early stopping：全局 best 停滞超过 patience 个 episode 则提前终止（0=不早停）
        es_patience = int(getattr(self.args, "early_stop_patience", 0) or 0)
        es_last_best_ep = 0  # 记录全局 best 被更新时的 episode_num
        if es_patience > 0:
            logger.info(
                f"Early stopping ENABLED: stop after global best is stagnant for "
                f"{es_patience} episodes (best_ep={self.global_best_episode})"
            )

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
            self.per_board_reward_history[pcb_name].append((board_ep, ep_reward))

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
                es_last_best_ep = self.episode_num
                logger.info(
                    f"[NEW GLOBAL BEST] R={self.global_best_reward:.2f} "
                    f"board={pcb_name} ep={self.global_best_episode} → {ckpt_path}"
                )

            # target 网络同步（按 episode 计数）
            if self.episode_num % Config.TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

            # 日志
            self._log_episode(pcb_name, ep_reward, ep_steps, ep_placed, avg_loss)

            # 定期画 per-board loss / reward 曲线
            if board_ep % 100 == 0:
                self._plot_loss_curve(pcb_name)
                self._plot_reward_curve(pcb_name)

            # early stopping：全局 best 停滞超过 patience 则提前终止
            if es_patience > 0 and (self.episode_num - es_last_best_ep) >= es_patience:
                logger.info(
                    f"EARLY STOP: global best R={self.global_best_reward:.2f} "
                    f"(ep={self.global_best_episode}) stagnant for {es_patience} episodes, "
                    f"stopping at ep={self.episode_num} (planned {self.total_episodes})"
                )
                break

        # ---- 10. 训练结束：per-board 分析 & 全局汇总 ----
        for pcb_name in pcb_names:
            self._plot_loss_curve(pcb_name)
            self._plot_reward_curve(pcb_name)
            best_state = self.per_board_best_state.get(pcb_name)
            if best_state:
                self._export_best_layout(pcb_name, best_state,
                                         self.per_board_best_reward[pcb_name],
                                         self.per_board_best_episode[pcb_name])
            self._write_pcb_summary(pcb_name)

        # 保存最终 checkpoint（最多保留 5 份 final_ep*.pt）
        if getattr(self.args, "explicit_total_episodes", False):
            ckpt_path = os.path.join(self.save_dir, f"final_ep{self.episode_num}.pt")
            agent.save(ckpt_path)
            logger.info(f"Final checkpoint saved: {ckpt_path}")

            # 清理旧 final_ep*.pt，最多保留 5 份
            all_finals = sorted(glob.glob(os.path.join(self.save_dir, "final_ep*.pt")))
            while len(all_finals) > 5:
                oldest = all_finals[0]
                try:
                    os.remove(oldest)
                    logger.info(f"Removed old final checkpoint: {os.path.basename(oldest)}")
                except OSError:
                    pass
                del all_finals[0]

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
        epsilon: Optional[float] = None,
    ):
        """执行单个 episode：5 步自回归 + 写入 buffer（+ 可选 learn）。

        Args:
            env: PCB 布局环境
            agent: DQN Agent
            buffer: 经验回放缓冲区
            pcb_name: PCB 文件名
            ep_num: 该板子当前的 episode 计数（用于 epsilon/curriculum）
            learn: 是否执行学习（预热期 = False）
            epsilon: 覆盖探索率，None 则按 curriculum 自动衰减

        Returns:
            (total_reward, step_count, placed_count, avg_loss)
        """
        if epsilon is None:
            epsilon = self._compute_epsilon(pcb_name)
        env.W_OUT_OF_BOUNDS_CURRENT = self._compute_curriculum_weight(pcb_name)
        env.current_episode = ep_num  # quality_scale 升温依据
        self._fail_window["episodes"] += 1  # 失败监控窗口 episode 计数（含 warmup）

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

            # ---- 失败 step 监控（仅统计，不改任何行为） ----
            # reward<0 的 step 即失败放置（has_violations=True，元件没放上），
            # 从 info.reward_breakdown 提取违规类型分布，量化坏循环入口。
            if reward < 0.0:
                self._fail_window["fail_steps"] += 1
                rbd = _info.get("reward_breakdown", {}) or {}
                hit = {}
                for t in ("overlap_hard", "overlap_soft", "out_of_bounds"):
                    v = rbd.get(t, 0.0)
                    if v < 0.0:
                        hit[t] = True
                        self._fail_window["by_type"][t]["n"] += 1
                        self._fail_window["by_type"][t]["penalty"] += abs(v)
                if _info.get("in_keepout"):
                    hit["in_keepout"] = True
                    self._fail_window["by_type"]["in_keepout"]["n"] += 1
                if hit.get("out_of_bounds") and not any(
                    hit.get(k) for k in ("overlap_hard", "overlap_soft", "in_keepout")
                ):
                    self._fail_window["oob_only"] += 1
                # 同一元件连续失败串（坏循环 = 反复重试同一元件）
                comp_idx = transitions[0][1]  # step0 动作 = 本回合选中的元件
                if comp_idx == self._last_fail_comp:
                    self._consecutive_fail += 1
                else:
                    self._consecutive_fail = 1
                self._last_fail_comp = comp_idx
                self._fail_window["max_consecutive_fail"] = max(
                    self._fail_window["max_consecutive_fail"], self._consecutive_fail
                )
                # 保险丝：同一元件连续失败 ≥ 阈值 → 强制终止 episode。
                # 打断"微小惩罚反复重试"的坏循环（原靠 -150 累计惩罚掐断，
                # 平均要 40~90 次重试，浪费训练步数并污染 buffer）。
                if (
                    getattr(Config, "FUSE_MAX_CONSECUTIVE_FAIL", 0) > 0
                    and self._consecutive_fail >= Config.FUSE_MAX_CONSECUTIVE_FAIL
                ):
                    self._fail_window["fuse_triggered"] += 1
                    self._consecutive_fail = 0  # 重置连败串，下一 episode 重新计数
                    done = True

            # ---- 写入 buffer（5 步按比例共享 reward，做 per-board 静态归一化） ----
            # 静态归一化：除以完成奖励上限 W_COMPLETION，使不同板 reward 同量纲，
            # 避免元件数差异导致跨板 reward 量纲悬殊、Q 网络目标被极端值主导。
            # 放满全部元件 → reward=200 → 归一化后 = 1.0（所有板一致）。
            share_weights = [0.2, 0.3, 0.3, 0.1, 0.1]
            norm_denom = float(Config.W_COMPLETION)
            reward_norm = reward / norm_denom
            for i, (s, a, mask, sid) in enumerate(transitions):
                if sid < 4:
                    next_s = transitions[i + 1][0]
                    next_sid = sid + 1
                    step_done = False
                else:
                    next_s = obs_next
                    next_sid = 0 if not done else -1
                    step_done = done

                step_reward = reward_norm * share_weights[sid]
                buffer.add(s, a, step_reward, next_s, mask, step_done, sid, next_sid)

            # ---- 每步 learn（预热期跳过） ----
            if learn:
                if buffer.is_ready(Config.BATCH_SIZE):
                    batch = buffer.sample(Config.BATCH_SIZE)
                    is_weights = buffer.get_last_is_weights() if isinstance(buffer, ExpertReplayBuffer) else None
                    loss, td_errors = agent.learn(batch, is_weights=is_weights)
                    losses.append(loss)

                    # PER 优先级更新（ExpertReplayBuffer）
                    if isinstance(buffer, ExpertReplayBuffer):
                        indices = buffer.get_last_sampled_indices()
                        if indices:
                            buffer.update_priorities(indices, td_errors)

            env._reset_pending()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return episode_reward, step_count, placed_count, avg_loss

    # ==================== 训练机制 ====================

    def _compute_epsilon(self, pcb_name: str) -> float:
        """跨 episode 指数衰减 epsilon（per-board 独立计数）。

        衰减基准自适应：按单板预算（total_episodes / 板数）收敛，且不低于
        EPSILON_DECAY_EPISODES。单板时即按 total_episodes 收敛；多板时折算到
        每板实际训练次数，避免 ε 被全局 total_episodes 稀释而收敛不到下限。

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
        total_episodes = getattr(self, "total_episodes", Config.TOTAL_EPISODES)
        # ε 用 per-board 独立计数 t，衰减基准也需折算到单板预算：
        # 单板（n_boards=1）时 budget == total_episodes，与原需求一致；
        # 多板时每板平均只训练 total_episodes/n_boards 次，若不折算，
        # t 永远达不到 total_episodes，ε 收敛不到 EPSILON_MIN。
        n_boards = max(1, len(getattr(self, "pcb_pool", [])))
        per_board_budget = total_episodes / n_boards
        # ε 衰减修复：衰减区间直接跟随实际单板训练预算 per_board_budget，
        # 而非被 EPSILON_DECAY_EPISODES（默认 10000）钳制。原 max() 在短训练
        # （--total_episodes 2000）时 decay 区间恒为 10000，ε 全程 ~0.5 几乎不降，
        # 模型始终半随机半利用，训练不收敛（各板后 20% R 均退化）。
        # 改后 ε 在单板预算末收敛到 EPSILON_MIN：t→per_board_budget 时
        # exp(-4)≈0.018，ε≈eps_min，训练后期充分"利用"学到的策略。
        # 治本实验：若显式配置 EPSILON_DECAY_EPISODES(>0 且 < budget)，则用作衰减区间，
        # 使 ε 提前降到低值（诊断发现 ep198 时 ε 仍 0.39，模型被自身随机探索破坏）。
        decay_episodes = per_board_budget
        cfg_decay = int(getattr(Config, "EPSILON_DECAY_EPISODES", 0) or 0)
        if cfg_decay > 0:
            decay_episodes = min(per_board_budget, float(cfg_decay))
        return eps_min + (eps_start - eps_min) * math.exp(-4.0 * t / decay_episodes)

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

    # ==================== 专家轨迹生成/加载 ====================

    def _generate_or_load_expert_transitions(self) -> List[Tuple]:
        """生成或从磁盘加载所有板子的专家 transition。

        优先从磁盘加载已缓存的专家数据；如果不存在则实时生成。

        Returns:
            全部专家 transition 的列表
        """
        # 基线组（use_expert=False）不接触专家数据，直接短路返回
        if not self.use_expert:
            logger.info("专家注入已禁用，跳过专家轨迹生成/加载")
            return []

        # 读取专家注入超参数
        total_transitions = getattr(Config, "EXPERT_TOTAL_TRANSITIONS", 20000)
        max_trajectories = getattr(Config, "EXPERT_MAX_TRAJECTORIES", 5000)
        epsilon_decompose = getattr(Config, "EXPERT_EPSILON_DECOMPOSE", 0.5)
        pos_noise_prob = getattr(Config, "EXPERT_POS_NOISE_PROB", 0.2)
        rot_noise_prob = getattr(Config, "EXPERT_ROT_NOISE_PROB", 0.1)
        # 雷2修复：专家 reward 携带惩罚项，与 RL 真实环境一致（默认 True）。
        # 修复前默认 False 且 argument.txt 未定义 → 重新生成专家数据时
        # 会产出"仅品质奖励、无惩罚"的轨迹，与训练 reward 口径不一致（实验 D 残留）。
        expert_apply_penalties = bool(getattr(Config, "EXPERT_APPLY_PENALTIES", True))

        # 输出目录（相对路径 → 基于项目根目录的绝对路径）
        # 优先使用外层调度器注入的覆盖目录（对照实验隔离缓存），否则用 Config 默认值
        expert_dir = self._expert_transitions_dir_override
        if not expert_dir:
            expert_dir = getattr(Config, "EXPERT_TRANSITIONS_DIR", "data/expert_transitions")
        if not os.path.isabs(expert_dir):
            expert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), expert_dir)
        self.expert_transitions_dir = expert_dir
        os.makedirs(self.expert_transitions_dir, exist_ok=True)

        all_transitions = []

        for pcb_name, _ in self.pcb_pool:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            filepath = os.path.join(self.expert_transitions_dir, f"{pcb_short}_expert.json")

            if os.path.isfile(filepath):
                # 从磁盘加载已缓存的专家数据
                from env.expert_trajectory import ExpertTrajectoryGenerator
                transitions = ExpertTrajectoryGenerator.load_transitions(filepath)
                logger.info(
                    f"[{pcb_short}] Loaded {len(transitions)} expert transitions from cache"
                )
            else:
                # 实时生成
                env = self.envs[pcb_name]
                transitions, _ = generate_and_save_expert_transitions(
                    env=env,
                    pcb_name=pcb_name,
                    output_dir=self.expert_transitions_dir,
                    total_transitions=total_transitions,
                    max_trajectories=max_trajectories,
                    epsilon_decompose=epsilon_decompose,
                    pos_noise_prob=pos_noise_prob,
                    rot_noise_prob=rot_noise_prob,
                    apply_penalties=expert_apply_penalties,
                )

            all_transitions.extend(transitions)

        return all_transitions

    # ==================== Phase 1: BC 预训练 ====================

    def _phase1_bc_pretrain(self, all_expert_transitions: List[Tuple], pcb_names: List[str]):
        """Phase 1: BC 预训练 — 仅用专家 transition 做监督学习。

        Args:
            all_expert_transitions: 全部专家 transition 列表
            pcb_names: PCB 名称列表
        """
        bc_epochs = getattr(Config, "EXPERT_BC_EPOCHS", 50)
        logger.info("=" * 60)
        logger.info(f"Phase 1: BC Pretraining — {bc_epochs} epochs on expert transitions")
        logger.info(f"Expert transitions: {len(all_expert_transitions)}")
        logger.info("=" * 60)

        # 创建 Agent（无 buffer，纯监督学习）
        agent = DQNAgent(
            n_movable=Config.MAX_MOVABLE,
            grid_w=Config.GRID_W,
            grid_h=Config.GRID_H,
            device=self.device,
            lr=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
        )

        # BC 预训练
        epoch_losses = agent.learn_bc(
            expert_transitions=all_expert_transitions,
            batch_size=Config.BATCH_SIZE,
            epochs=bc_epochs,
            verbose=True,
        )

        # 保存 BC 预训练后的 checkpoint
        bc_ckpt_path = os.path.join(self.save_dir, "bc_pretrained.pt")
        agent.save(bc_ckpt_path)
        logger.info(f"BC pretrained checkpoint saved: {bc_ckpt_path}")

        # 保存 BC loss 曲线数据
        bc_loss_path = os.path.join(self.save_dir, "bc_loss_history.json")
        with open(bc_loss_path, "w", encoding="utf-8") as f:
            json.dump(epoch_losses, f, ensure_ascii=False)
        logger.info(f"BC loss history saved: {bc_loss_path}")

        # 设置全局最佳初始值（避免 Phase 2 首次最佳覆盖 BC checkpoint）
        self.global_best_reward = float("-inf")

    # ==================== Phase 2: RL + 专家注入 设置 ====================

    def _phase2_setup(
        self,
        all_expert_transitions: List[Tuple],
        pcb_names: List[str],
    ) -> Tuple[DQNAgent, ExpertReplayBuffer]:
        """Phase 2: 创建 Agent 和 ExpertReplayBuffer，加载 BC 权重。

        Buffer 容量 = EXPERT_BUFFER_MULTIPLIER × 专家数
        其中 20% 为专家经验常驻区。

        Args:
            all_expert_transitions: 全部专家 transition 列表
            pcb_names: PCB 名称列表

        Returns:
            (DQNAgent, ExpertReplayBuffer) 元组
        """
        n_expert = len(all_expert_transitions)
        buffer_multiplier = getattr(Config, "EXPERT_BUFFER_MULTIPLIER", 5)
        expert_ratio = getattr(Config, "EXPERT_BUFFER_RATIO", 0.2)
        per_alpha = getattr(Config, "PER_ALPHA", 0.6)
        per_beta = getattr(Config, "PER_BETA_START", 0.4)

        # Expert buffer 容量 = 专家数（常驻，不可覆盖）
        # Normal buffer 容量 = (multiplier - 1) × 专家数（环形覆盖）
        expert_capacity = n_expert
        normal_capacity = (buffer_multiplier - 1) * n_expert

        # 创建 Agent
        agent = DQNAgent(
            n_movable=Config.MAX_MOVABLE,
            grid_w=Config.GRID_W,
            grid_h=Config.GRID_H,
            device=self.device,
            lr=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
        )

        # 加载 BC 预训练权重（如果存在）
        bc_ckpt_path = os.path.join(self.save_dir, "bc_pretrained.pt")
        if os.path.isfile(bc_ckpt_path):
            agent.load(bc_ckpt_path)
            logger.info(f"Phase 2: loaded BC pretrained weights from {bc_ckpt_path}")
        else:
            logger.warning("Phase 2: BC checkpoint not found, starting from scratch")

        # 创建 ExpertReplayBuffer
        buffer = ExpertReplayBuffer(
            expert_capacity=expert_capacity,
            normal_capacity=normal_capacity,
            expert_ratio=expert_ratio,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

        # 加载专家 transition 到常驻区
        buffer.load_expert_transitions(all_expert_transitions)

        logger.info(
            f"Phase 2 setup: ExpertReplayBuffer "
            f"(expert={expert_capacity}, normal={normal_capacity}, "
            f"total={expert_capacity + normal_capacity}, "
            f"ratio={expert_ratio}, PER α={per_alpha}, β={per_beta})"
        )

        return agent, buffer

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
        """递归扫描数据目录获取所有 PCB 文件（委托到 utils.pcb_scan），可选过滤板子名。"""
        from utils.pcb_scan import find_pcb_files
        all_files = find_pcb_files(self.data_dir)

        # 按 --exclude 参数排除（子串匹配，可多个用英文逗号分隔）
        exclude_str = getattr(self.args, "exclude", None)
        if exclude_str:
            exclusions = [e.strip() for e in exclude_str.split(",") if e.strip()]
            excluded = [f for f in all_files
                        if any(e in os.path.basename(f) for e in exclusions)]
            if excluded:
                all_files = [f for f in all_files if f not in excluded]
                logger.info(
                    f"Excluded {len(excluded)}/{len(excluded) + len(all_files)} files "
                    f"matching {exclusions}: {[os.path.basename(f) for f in excluded]}"
                )

        # 按 --board 参数过滤
        board_filter = getattr(self.args, "board", None)
        if board_filter:
            filtered = [f for f in all_files if board_filter in os.path.basename(f)]
            if not filtered:
                logger.warning(
                    f"Board filter '{board_filter}' matched no files, "
                    f"using all {len(all_files)} files"
                )
                return all_files
            logger.info(
                f"Board filter '{board_filter}' matched {len(filtered)}/{len(all_files)} files"
            )
            return filtered
        return all_files

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

        # ---- 失败监控窗口输出（每 LOG_INTERVAL ep 一次，输出后清零） ----
        fw = self._fail_window
        if fw["fail_steps"] > 0:
            fs = fw["fail_steps"]
            parts = []
            for t in ("out_of_bounds", "overlap_hard", "overlap_soft", "in_keepout"):
                d = fw["by_type"][t]
                parts.append(f"{t}={100.0 * d['n'] / fs:.1f}%({d['n']})")
            logger.info(
                f"[FAIL MONITOR] 窗口{fw['episodes']}ep 失败step={fs} "
                f"保险丝={fw['fuse_triggered']} 纯出界="
                f"{100.0 * fw['oob_only'] / fs:.1f}%({fw['oob_only']}) "
                f"同元件连败最长={fw['max_consecutive_fail']} | " + " ".join(parts)
            )
        # 清零窗口（保留结构复用）
        fw["episodes"] = 0
        fw["fail_steps"] = 0
        fw["max_consecutive_fail"] = 0
        fw["oob_only"] = 0
        fw["fuse_triggered"] = 0
        for t in fw["by_type"]:
            fw["by_type"][t]["n"] = 0
            fw["by_type"][t]["penalty"] = 0.0
        self._last_fail_comp = -1
        self._consecutive_fail = 0

    def _safe_move(self, src_path: str, dst: str, max_retries: int = 10):
        """健壮地把 src 移动到 dst，对 Windows 瞬时文件锁做重试。

        背景：Windows 下 `shutil.move` 跨目录时内部先 `os.rename`，若目标已存在则
        先删目标、再 `os.unlink(src)`。当 src 被其它进程（日志查看器/杀毒扫描/
        上次训练进程退出延迟释放句柄）瞬时占用时会抛 `PermissionError: WinError 32`，
        导致训练启动直接中断。

        本方法改用"复制 + 删除"两段式，并对 PermissionError 退避重试；重试耗尽则
        返回（由调用方降级跳过备份，绝不中断训练）。

        Args:
            src_path: 源文件路径
            dst: 目标文件路径
            max_retries: 最大重试次数（默认 10，每次间隔 0.5s，共约 5s）
        """
        if not os.path.isfile(src_path):
            return
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # 首次直接尝试 move（最快路径，正常场景一步到位）
                    shutil.move(src_path, dst)
                else:
                    # 之后改为复制 + 删除，对锁更宽容
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.copy2(src_path, dst)
                    os.remove(src_path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))  # 退避重试
            except OSError as e:
                # 非锁类错误直接忽略（文件不存在等），由调用方降级
                last_err = e
                break
        logger.warning(f"safe_move failed for {os.path.basename(src_path)}: {last_err}")

    def _backup_previous_results(self):
        """训练开始前备份旧产物，每种最多保留 5 份。

        保护的产物：best_model.pt / training_log.txt / bc_loss_history.json。
        bc_pretrained.pt 不备份（Phase 2 加载后即被 best_model.pt 继承，无需存档）。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.save_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        # 备份的产物及对应的备份命名模板
        artifacts = [
            ("best_model.pt", f"best_model_{{ts}}.pt"),
            ("training_log.txt", f"training_log_{{ts}}.txt"),
            ("bc_loss_history.json", f"bc_loss_history_{{ts}}.json"),
        ]

        for src_name, backup_tpl in artifacts:
            src_path = os.path.join(self.save_dir, src_name)
            if not os.path.isfile(src_path):
                continue
            backup_name = backup_tpl.format(ts=timestamp)
            dst = os.path.join(backup_dir, backup_name)
            self._safe_move(src_path, dst)
            if os.path.isfile(dst):
                logger.info(f"Backed up: {src_name} → backup/{backup_name}")
            else:
                # 备份失败（文件被占用且重试耗尽）：降级跳过，不中断训练
                logger.warning(
                    f"Backup skipped for {src_name} (file locked, retries exhausted) — 继续训练，不中断"
                )

            # 按类型清理旧备份：最多保留 5 份
            prefix = backup_name.rsplit("_2", 1)[0]  # "best_model" / "training_log" / "bc_loss_history"
            ext = backup_name.rsplit(".", 1)[-1]
            pattern = os.path.join(backup_dir, f"{prefix}_*.{ext}")
            all_backups = sorted(glob.glob(pattern))
            while len(all_backups) > 5:
                oldest = all_backups[0]
                try:
                    os.remove(oldest)
                    logger.info(f"Removed old backup: backup/{os.path.basename(oldest)}")
                except OSError:
                    pass
                del all_backups[0]

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

    def _plot_reward_curve(self, pcb_name: str):
        """生成 Reward-Episode 曲线图，保存到 result/<board>/ 子目录。

        与 _plot_loss_curve 对称：写 reward_history.json + 输出 reward_curve.png。
        仅统计正式训练阶段的 reward（warmup 阶段不记录）。

        Args:
            pcb_name: PCB 文件名
        """
        from utils.visualization import plot_reward_curve

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        board_dir = os.path.join(self.save_dir, pcb_short)
        os.makedirs(board_dir, exist_ok=True)

        reward_history = self.per_board_reward_history.get(pcb_name, [])
        json_path = os.path.join(board_dir, "reward_history.json")
        png_path = os.path.join(board_dir, "reward_curve.png")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reward_history, f, ensure_ascii=False)

        plot_reward_curve(reward_history, png_path, pcb_name=pcb_short, window=100)

    # ==================== 最佳布局追踪 ====================

    def _snapshot_state(self, env: PCBLayoutEnv) -> dict:
        """提取 env 中所有元件的完整状态快照。"""
        from run_inference import snapshot_state
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
