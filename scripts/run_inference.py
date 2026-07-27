"""纯推理 Demo：使用合成 PCB 数据 + 随机 DQN 权重反复尝试布局。

不训练（epsilon=0），反复 episode 直到成功放置全部可移动元件。
追踪 reward 最高的布局，输出 state JSON + 可视化 PNG + 日志到 output/。
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- mock utils.kicad_utils 必须在 import env 之前 ----
from unittest import mock
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

import json
import logging
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from env.mock_pcb import create_env_from_json, MockKiCadUtils, MockComponent, MockNet
# 手动将 mock 子模块 attach 到 utils 并注入类（供 env.pcb_layout_env 中 from utils.kicad_utils import KiCadUtils 使用）
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_layout_env import Config
from dqn import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


# ==================== 状态快照 ====================

def snapshot_state(env) -> Dict:
    """提取 env 中所有元件的完整状态。

    Args:
        env: PCBLayoutEnv 实例（已执行若干 step）

    Returns:
        dict 含 board_bounds, movable_count, fixed_count, components 列表
    """
    (min_x, min_y), (max_x, max_y) = env.board_bounds

    components = []
    for comp_id, comp in env.kicad.components.items():
        components.append({
            "id": comp_id,
            "footprint": getattr(comp, "footprint", ""),
            "package_type": comp.package_type,
            "bbox": list(comp.bbox),
            "rotation": comp.rotation,
            "layer": comp.layer,
            "is_fixed": comp.is_fixed,
            "is_placed": comp.is_placed,
            "nets": list(comp.nets),
            "footprint_w": comp.footprint_w,
            "footprint_h": comp.footprint_h,
        })

    return {
        "board_bounds": {
            "min_x": min_x, "min_y": min_y,
            "max_x": max_x, "max_y": max_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
        },
        "grid": {
            "width": env.grid_width,
            "height": env.grid_height,
            "cell_size": Config.CELL_SIZE,
        },
        "movable_count": env.N_movable,
        "fixed_count": len(env.fixed_list),
        "components": components,
        "nets": {nid: {"is_power": net.is_power, "connections": net.connections}
                 for nid, net in env.kicad.nets.items()},
    }


def _snapshot_to_viz_components(state_snap: Dict) -> list:
    """将 snapshot_state 输出转为 draw_layout 可用的元件对象列表。

    Args:
        state_snap: snapshot_state() 的返回值

    Returns:
        VizComp 对象列表（含 id/bbox/is_fixed/is_placed/nets 属性）
    """
    components = []
    for comp_data in state_snap["components"]:
        # 跳过未放置的 movable 元件（仅显示已放置 + 固定元件）
        if not comp_data["is_placed"] and not comp_data["is_fixed"]:
            continue
        class VizComp:
            pass
        vc = VizComp()
        vc.id = comp_data["id"]
        vc.bbox = tuple(comp_data["bbox"])
        vc.is_fixed = comp_data["is_fixed"]
        vc.is_placed = comp_data["is_placed"]
        vc.nets = comp_data["nets"]
        components.append(vc)
    return components


# ==================== 推理 Episode ====================

def run_inference_episode(
    env, agent: DQNAgent, step_img_dir: Optional[str] = None
) -> Tuple[float, int, int, list]:
    """执行 1 个纯推理 episode（epsilon=0，不训练，不写 buffer）。

    Args:
        env: PCBLayoutEnv 实例
        agent: DQNAgent（随机权重，仅用于 select_action）
        step_img_dir: 若不为 None，每步后保存中间布局图到此目录

    Returns:
        (episode_reward, step_count, placed_count, action_history)
        action_history: [(comp_id, gx1, gy1, rot, lay), ...]
    """
    env.reset()
    env._reset_pending()
    done = False
    episode_reward = 0.0
    step_count = 0
    placed_count = 0
    action_history = []

    while not done:
        # ---- 5 步自回归（纯推理：epsilon=0） ----
        for step_id in range(5):
            mask = env._build_mask_for_step(step_id)
            obs = env._get_obs_with_pending(step_id)
            action = agent.select_action(obs, mask, step_id, epsilon=0.0)
            env._update_pending(action, step_id)

        # ---- env.step() ----
        full_action = env._assemble_action()
        obs_next, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        placed_count = info.get("placed_count", 0)

        # 记录行动历史
        comp = env.movable_list[full_action[0]]
        action_history.append({
            "comp_id": comp.id,
            "gx1": full_action[1],
            "gy1": full_action[2],
            "gx2": full_action[3],
            "gy2": full_action[4],
            "rot": full_action[5],
            "lay": full_action[6],
            "reward": reward,
            "is_placed": info.get("hard_overlap", 0) == 0 and info.get("out_of_bounds", 0) == 0,
        })

        env._reset_pending()

        # ---- 中间步骤可视化（每放完一个元件拍一张） ----
        if step_img_dir is not None:
            state_snap = snapshot_state(env)
            viz_comps = _snapshot_to_viz_components(state_snap)
            board_bounds = (
                (state_snap["board_bounds"]["min_x"], state_snap["board_bounds"]["min_y"]),
                (state_snap["board_bounds"]["max_x"], state_snap["board_bounds"]["max_y"]),
            )
            pad = len(str(env.N_movable))
            step_path = os.path.join(step_img_dir, f"step_{step_count:0{pad}d}.png")
            title = f"Step {step_count} — {comp.id}  |  Reward={reward:.1f}"
            from utils.visualization import draw_layout
            draw_layout(viz_comps, board_bounds, step_path, title=title)

    return episode_reward, step_count, placed_count, action_history


def _replay_best_episode(env, agent, best_actions: list, step_img_dir: str):
    """重放最佳 episode 的动作序列，每步生成一张步骤图。

    先 env.reset() 再按 best_actions 逐步 env.step()，
    每步后 snapshot → draw_layout 保存 step_NN.png。

    Args:
        env: PCBLayoutEnv 实例（会被 reset 并重放，副作用：env 停在终态）
        agent: DQNAgent（未使用，保留兼容）
        best_actions: run_inference_episode 返回的 action_history
        step_img_dir: 步骤图输出目录
    """
    from utils.visualization import draw_layout

    env.reset()
    env._reset_pending()

    # comp_id → movable_list 索引映射
    comp_id_to_idx = {c.id: i for i, c in enumerate(env.movable_list)}

    pad = len(str(env.N_movable))

    for i, act in enumerate(best_actions):
        comp_idx = comp_id_to_idx[act["comp_id"]]
        full_action = (
            comp_idx,
            act["gx1"], act["gy1"], act["gx2"], act["gy2"],
            act["rot"], act["lay"],
        )

        _, reward, _, _, _ = env.step(full_action)
        env._reset_pending()

        # 生成步骤图
        state_snap = snapshot_state(env)
        viz_comps = _snapshot_to_viz_components(state_snap)
        board_bounds = (
            (state_snap["board_bounds"]["min_x"], state_snap["board_bounds"]["min_y"]),
            (state_snap["board_bounds"]["max_x"], state_snap["board_bounds"]["max_y"]),
        )
        step_path = os.path.join(step_img_dir, f"step_{i+1:0{pad}d}.png")
        title = f"Step {i+1} — {act['comp_id']}  |  Reward={reward:.1f}"
        draw_layout(viz_comps, board_bounds, step_path, title=title)

    logger.info(f"[Replay] {len(best_actions)} step images saved to {step_img_dir}")


# ==================== 主流程 ====================

def main():
    """主推理流程：反复尝试直到全部放置成功。

    支持 --data 参数指定 JSON 数据文件，否则使用合成默认 PCB。
    所有输出放入 output/<pcb_name>/ 目录。
    """
    import argparse

    parser = argparse.ArgumentParser(description="PCB Layout Inference Demo")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to PCB JSON data file (e.g. data/模拟/pcb_001.json).",
    )
    args = parser.parse_args()

    # ---- 确定 PCB 名称与加载数据 ----
    pcb_name = os.path.splitext(os.path.basename(args.data))[0]
    logger.info(f"Loading PCB from JSON: {args.data}")
    env = create_env_from_json(os.path.abspath(args.data))

    # ---- 创建输出目录 ----
    pcb_output_dir = os.path.join(OUTPUT_DIR, pcb_name)
    os.makedirs(pcb_output_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 日志双写（stdout + 文件） ----
    log_path = os.path.join(pcb_output_dir, "inference_log.txt")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"PCB Layout Inference Demo — {pcb_name}")
    logger.info("=" * 60)

    obs_dim = env.N_movable * 8 + env.N_movable + 4

    (min_x, min_y), (max_x, max_y) = env.board_bounds
    logger.info(
        f"Board: {max_x - min_x:.0f}×{max_y - min_y:.0f}mm, "
        f"N_movable={env.N_movable}, grid=({env.grid_width},{env.grid_height})"
    )

    # ---- 创建 Agent（随机权重） ----
    device = torch.device("cpu")
    agent = DQNAgent(
        obs_dim=obs_dim, n_movable=env.N_movable,
        grid_w=env.grid_width, grid_h=env.grid_height,
        device=device,
        lr=Config.LEARNING_RATE, gamma=Config.GAMMA,
    )
    logger.info(f"Agent created (random weights): obs_dim={obs_dim}")

    # ---- 推理循环（搜索阶段，不生成步骤图） ----
    MAX_ATTEMPTS = 500
    best_reward = float("-inf")
    best_state = None
    best_episode = -1
    best_actions = []
    all_completed = False

    start_time = time.time()

    for ep in range(1, MAX_ATTEMPTS + 1):
        env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

        ep_reward, ep_steps, ep_placed, actions = run_inference_episode(env, agent)

        # 追踪最佳布局（按 reward 最高）
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_state = snapshot_state(env)
            best_episode = ep
            best_actions = actions

        # 日志（每 50 轮或关键事件）
        if ep % 50 == 0 or ep_placed == env.N_movable or ep == 1:
            status = "[COMPLETE]" if ep_placed == env.N_movable else ""
            logger.info(
                f"[Ep {ep:4d}] R={ep_reward:8.2f}  "
                f"placed={ep_placed}/{env.N_movable}  "
                f"steps={ep_steps:3d}  {status}"
            )

        # 全部放置成功
        if ep_placed == env.N_movable:
            all_completed = True
            logger.info(f"\n[OK] All {env.N_movable} components placed at episode {ep}!")
            break

    elapsed = time.time() - start_time

    if not all_completed:
        actual_placed = sum(
            1 for c in best_state["components"]
            if c["is_placed"] and not c["is_fixed"]
        ) if best_state else 0
        logger.warning(
            f"\n[WARN] Could not place all components in {MAX_ATTEMPTS} attempts. "
            f"Best: {actual_placed}/{best_state['movable_count']} placed (ep {best_episode})"
        )

    # ---- 输出（全部放入 output/<pcb_name>/ 目录） ----

    # 4a. 重放最佳 episode 生成步骤图
    if best_actions:
        _replay_best_episode(env, agent, best_actions, pcb_output_dir)
    else:
        logger.warning("[WARN] No actions recorded, skipping step images.")

    # 4b. State JSON
    json_path = os.path.join(pcb_output_dir, "best_layout_state.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pcb_name": pcb_name,
            "data_file": os.path.abspath(args.data),
            "episode": best_episode,
            "reward": best_reward,
            "all_placed": all_completed,
            "elapsed_seconds": round(elapsed, 2),
            "total_attempts": best_episode if all_completed else MAX_ATTEMPTS,
            **best_state,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"[FILE] State saved: {json_path}")

    # 4c. 日志汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"Inference Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  PCB name          : {pcb_name}")
    logger.info(f"  Total attempts    : {best_episode if all_completed else MAX_ATTEMPTS}")
    logger.info(f"  All placed        : {all_completed}")
    logger.info(f"  Best episode      : {best_episode}")
    logger.info(f"  Best reward       : {best_reward:.2f}")
    logger.info(f"  Elapsed           : {elapsed:.2f}s")
    logger.info(f"  Output directory  : {pcb_output_dir}")

    # 4d. 最终布局可视化
    if best_state is not None:
        from utils.visualization import draw_layout

        components_for_viz = _snapshot_to_viz_components(best_state)
        board_bounds = (
            (best_state["board_bounds"]["min_x"], best_state["board_bounds"]["min_y"]),
            (best_state["board_bounds"]["max_x"], best_state["board_bounds"]["max_y"]),
        )
        title = (
            f"Best Layout — {pcb_name}  Ep {best_episode}  "
            f"Reward={best_reward:.0f}"
        )
        png_path = os.path.join(pcb_output_dir, "best_layout.png")
        draw_layout(components_for_viz, board_bounds, png_path, title=title)
        logger.info(f"[FILE] Layout visualization saved: {png_path}")

    logger.info(f"\n[OK] Inference demo finished. See output/{pcb_name}/ for results.")

    # 关闭文件 handler
    logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()
