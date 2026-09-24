"""100 次独立推理评测（对比实验专用评测标准）。

对单模型连续跑 n 次（默认 100）独立单 episode 贪心推理，仅收集每条的
最终结果三项指标：完成度(placed/N_movable)、尝试次数(step_count)、reward(终局独立质量分)。

复用 run_inference.py::run_inference_episode（默认 inference_epsilon=0.01 打破 argmax
死锁），不复制推理逻辑。供 run_compare.py 在对比实验结尾对 A/B 两组 best_model 各评 100 次。

用法（被 run_compare.py 直接调用，亦可独立调试）：
    from inference_eval import run_inference_eval
    stats = run_inference_eval(pcb_path, agent, n=100)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from env.pcb_layout_env import Config
from run_inference import run_inference_episode

logger = logging.getLogger(__name__)


def _summarize(values: List[float]) -> Dict[str, float]:
    """对一组数值求 mean/std/min/max（空列表返回全 0）。"""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_inference_eval(
    pcb_path: str,
    agent: "torch.nn.Module",
    n: int = 100,
    inference_epsilon: float = 0.01,
    seed: Optional[int] = None,
) -> Dict:
    """对单模型连续跑 n 次独立单 episode 推理，返回 100 条最终结果 + 统计。

    每次推理 = env.reset() 后跑一个完整 episode（run_inference_episode），
    无论是否放满都记一条最终结果。复现性：可选 seed 固定 torch/numpy。

    Args:
        pcb_path: .kicad_pcb 文件路径
        agent: 已加载权重的 DQNAgent（run_inference_episode 仅用其 select_action）
        n: 推理次数（默认 100）
        inference_epsilon: 贪心推理探索率（默认 0.01，避免 argmax 死锁）
        seed: 可选随机种子（固定复现）

    Returns:
        dict: {
            "n": int,
            "completions_count":  List[int],    # 每轮 placed_count
            "completions_ratio":  List[float],  # 每轮 placed_count / N_movable
            "attempts":           List[int],    # 每轮 step_count（含失败重试）
            "rewards":            List[float],   # 每轮 final_layout_reward（终局独立质量分）
            "stats": {
                "completion_ratio": {mean,std,min,max},
                "attempts":        {mean,std,min,max},
                "rewards":         {mean,std,min,max},
                "full_place_rate": float,        # 完整放置(placed==N)占比
            },
        }
    """
    from env.pcb_layout_env import PCBLayoutEnv

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ---- 推理专用配置（与 run_inference._run_single_board_inference 一致）----
    saved_early_stop = Config.EARLY_STOP_THRESHOLD
    Config.EARLY_STOP_THRESHOLD = float("-inf")
    # 推理临时 env 实例属性，在创建 env 后设置
    env = None
    saved_oob = None
    try:
        env = PCBLayoutEnv(os.path.abspath(pcb_path))
        saved_oob = env.W_OUT_OF_BOUNDS_CURRENT
        env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

        n_movable = env.N_movable
        completions_count: List[int] = []
        completions_ratio: List[float] = []
        attempts: List[int] = []
        rewards: List[float] = []

        for i in range(1, n + 1):
            # run_inference_episode 内部已 reset，返回单 episode 最终结果
            ep_reward, step_count, placed_count, _actions, final_layout_reward = (
                run_inference_episode(
                    env, agent, inference_epsilon=inference_epsilon
                )
            )
            completions_count.append(int(placed_count))
            completions_ratio.append(
                float(placed_count) / n_movable if n_movable > 0 else 0.0
            )
            attempts.append(int(step_count))
            rewards.append(float(final_layout_reward))

            if i % 20 == 0 or i == 1:
                logger.info(
                    f"[inference_eval] {i:3d}/{n}  "
                    f"placed={placed_count}/{n_movable}  "
                    f"attempts={step_count:3d}  reward={final_layout_reward:8.2f}"
                )

        full_place_rate = (
            sum(1 for c in completions_count if c == n_movable) / n if n > 0 else 0.0
        )
        stats = {
            "completion_ratio": _summarize(completions_ratio),
            "attempts": _summarize(attempts),
            "rewards": _summarize(rewards),
            "full_place_rate": float(full_place_rate),
        }

        logger.info(
            f"[inference_eval] done n={n}  "
            f"full_place_rate={full_place_rate:.2%}  "
            f"mean_completion={stats['completion_ratio']['mean']:.3f}  "
            f"mean_reward={stats['rewards']['mean']:.2f}  "
            f"mean_attempts={stats['attempts']['mean']:.2f}"
        )

        return {
            "n": n,
            "n_movable": n_movable,
            "inference_epsilon": inference_epsilon,
            "completions_count": completions_count,
            "completions_ratio": completions_ratio,
            "attempts": attempts,
            "rewards": rewards,
            "stats": stats,
        }
    finally:
        # 恢复全局配置，避免污染训练侧语义
        Config.EARLY_STOP_THRESHOLD = saved_early_stop
        if env is not None:
            if saved_oob is not None:
                env.W_OUT_OF_BOUNDS_CURRENT = saved_oob
            env.close()
