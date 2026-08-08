"""纯推理 Demo：使用合成 PCB 数据 + 随机 DQN 权重反复尝试布局。

支持 --data（JSON mock 数据）和 --pcb（真实 .kicad_pcb 文件）两种输入。
不训练（epsilon=0），反复 episode 直到成功放置全部可移动元件。
追踪 reward 最高的布局，输出 state JSON + 可视化 PNG + 日志到 output/。
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
import time
from typing import List, Tuple, Dict, Optional
from unittest import mock

import numpy as np
import torch
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
    from env.pcb_layout_env import Config

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


def _restore_state(env, state_snap: Dict):
    """将最佳布局快照回填到 env 的实时元件，供导出 best_layout 使用。

    Args:
        env: PCBLayoutEnv 实例（其 kicad.components 会被就地修改）
        state_snap: snapshot_state() 的返回值
    """
    for comp_data in state_snap["components"]:
        comp = env.kicad.components.get(comp_data["id"])
        if comp is None:
            continue
        comp.bbox = tuple(comp_data["bbox"])
        comp.rotation = comp_data["rotation"]
        comp.layer = comp_data["layer"]
        comp.is_placed = comp_data["is_placed"]


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


# ==================== 推理保底：C3-5 clamp ====================

def _clamp_action_to_board(env, comp_idx: int, gx1: int, gy1: int, gx2: int, gy2: int,
                           rot: int, lay: int) -> Tuple[int, int, int, int, int, int, int]:
    """推理阶段安全网：将网络输出的网格坐标 clamp 至 valid_area 内。

    保底机制：
      1. 网格坐标 → mm 坐标
      2. 调用 PCBUtils.clamp_to_board() 将 AABB 推回板框内（保持宽高不变，仅平移）
      3. mm 坐标 → 网格坐标

    无论模型输出多离谱的坐标，元件最终都会落在板框有效区域内，
    避免 inference 中因出界导致元件放置失败 → 浪费一步重试。

    Args:
        env: PCBLayoutEnv 实例
        comp_idx: 元件在 movable_list 中的索引
        gx1, gy1, gx2, gy2: 原始网格坐标
        rot, lay: 旋转和层（不透传 clamp，原样返回）

    Returns:
        (comp_idx, gx1, gy1, gx2, gy2, rot, lay) 保底后的 7 元动作
    """
    from env.pcb_utils import PCBUtils
    from env.pcb_layout_env import Config

    comp = env.movable_list[comp_idx]

    # 1. 网格 → mm
    x1 = env.valid_area[0] + gx1 * Config.CELL_SIZE
    y1 = env.valid_area[1] + gy1 * Config.CELL_SIZE
    x2 = env.valid_area[0] + gx2 * Config.CELL_SIZE
    y2 = env.valid_area[1] + gy2 * Config.CELL_SIZE

    # 2. Clamp 保底（宽高不变，仅平移）
    x1, y1, x2, y2 = PCBUtils.clamp_to_board(x1, y1, x2, y2, env.valid_area)

    # 3. mm → 网格
    new_gx1 = max(0, int(round((x1 - env.valid_area[0]) / Config.CELL_SIZE)))
    new_gy1 = max(0, int(round((y1 - env.valid_area[1]) / Config.CELL_SIZE)))
    # gx2/gy2 从 footprint + 新原点重新推导，保持宽高一致
    gw = max(1, int(comp.footprint_w / Config.CELL_SIZE))
    gh = max(1, int(comp.footprint_h / Config.CELL_SIZE))
    new_gx2 = new_gx1 + gw
    new_gy2 = new_gy1 + gh

    if (new_gx1, new_gy1) != (gx1, gy1):
        logger.debug(
            f"[CLAMP] {comp.id}: grid ({gx1},{gy1})→({new_gx1},{new_gy1}), "
            f"mm ({x1:.1f},{y1:.1f})"
        )

    return (comp_idx, new_gx1, new_gy1, new_gx2, new_gy2, rot, lay)


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
    from env.pcb_layout_env import Config

    env.reset()
    env._reset_pending()
    env.current_episode = 99999  # 推理时使用完整版 reward（quality_scale=1.0）
    done = False
    episode_reward = 0.0
    step_count = 0
    placed_count = 0
    action_history = []

    # 微量 epsilon 打破确定性死锁：
    # 未放置元件的 slot 全为零 → slot_encoder 输出相同 → argmax 总选索引最小的元件。
    # 若该位置与已放置元件冲突 → obs 不变 → 相同决策 → 无限循环直到 MAX_STEPS。
    # 训练 epsilon_min=0.15 掩盖了此问题，推理 epsilon=0 完全暴露。
    INFERENCE_EPSILON = 0.01

    while not done:
        # ---- 5 步自回归（推理：微量 epsilon 打破死锁） ----
        for step_id in range(5):
            mask = env._build_mask_for_step(step_id)
            obs = env._get_obs_with_pending(step_id)
            action = agent.select_action(obs, mask, step_id, epsilon=INFERENCE_EPSILON)
            env._update_pending(action, step_id)

        # ---- env.step()（C3-5 保底：clamp 坐标到板框内） ----
        full_action = env._assemble_action()
        full_action = _clamp_action_to_board(env, *full_action)
        prev_placed = placed_count
        obs_next, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        placed_count = info.get("placed_count", 0)
        is_placed = placed_count > prev_placed  # 准确：placed_list 增加了才算成功

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
            "is_placed": is_placed,
        })

        env._reset_pending()

        # ---- 中间步骤可视化（仅成功放置时拍一张） ----
        if step_img_dir is not None and is_placed:
            state_snap = snapshot_state(env)
            viz_comps = _snapshot_to_viz_components(state_snap)
            board_bounds = (
                (state_snap["board_bounds"]["min_x"], state_snap["board_bounds"]["min_y"]),
                (state_snap["board_bounds"]["max_x"], state_snap["board_bounds"]["max_y"]),
            )
            step_path = os.path.join(step_img_dir, f"step_{placed_count:02d}.png")
            title = f"Step {placed_count} — {comp.id}  |  Reward={episode_reward:.1f}"
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
    cumulative_reward = 0.0
    placed_num = 0  # 只对成功放置的步骤计数

    for act in best_actions:
        comp_idx = comp_id_to_idx[act["comp_id"]]
        full_action = (
            comp_idx,
            act["gx1"], act["gy1"], act["gx2"], act["gy2"],
            act["rot"], act["lay"],
        )
        # C3-5 保底：重放时同样 clamp 坐标
        full_action = _clamp_action_to_board(env, *full_action)

        _, reward, _, _, _ = env.step(full_action)
        env._reset_pending()
        cumulative_reward += reward

        # 仅成功放置时生成步骤图
        if not act.get("is_placed", False):
            continue

        placed_num += 1
        state_snap = snapshot_state(env)
        viz_comps = _snapshot_to_viz_components(state_snap)
        board_bounds = (
            (state_snap["board_bounds"]["min_x"], state_snap["board_bounds"]["min_y"]),
            (state_snap["board_bounds"]["max_x"], state_snap["board_bounds"]["max_y"]),
        )
        step_path = os.path.join(step_img_dir, f"step_{placed_num:0{pad}d}.png")
        title = f"Step {placed_num} — {act['comp_id']}  |  Reward={cumulative_reward:.1f}"
        draw_layout(viz_comps, board_bounds, step_path, title=title)

    logger.info(f"[Replay] {placed_num} step images saved to {step_img_dir}")


# ==================== 主流程 ====================

def main():
    """主推理流程：反复尝试直到全部放置成功。

    支持 --data（JSON mock 数据）和 --pcb（真实 .kicad_pcb 文件）两种输入。
    所有输出放入 output/<pcb_name>/ 目录。
    """
    import argparse

    parser = argparse.ArgumentParser(description="PCB Layout Inference Demo")
    parser.add_argument(
        "--data", type=str, default=None,
        help="PCB JSON 数据文件 (e.g. data/模拟/pcb_001.json).",
    )
    parser.add_argument(
        "--pcb", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "kicad-native-batch",
            "ran-pang-airlink-86231b34--开关小板-883285b3",
            "ran-pang-airlink-86231b34--开关小板-883285b3.kicad_pcb",
        )),
        help=".kicad_pcb 文件路径 (默认：开关小板).",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="导出最佳布局为 .kicad_pcb 文件路径（默认自动导出）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="输出步骤图、布局图、state JSON 等调试文件",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=500,
        help="最大尝试 episode 数（默认 500）",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="训练好的模型权重路径（.pt），不指定则使用随机权重",
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    if args.data:
        # JSON mock 路径：注入 mock KiCad
        _m = mock.MagicMock()
        sys.modules["utils.kicad_utils"] = _m

        from env.mock_pcb import create_env_from_json, MockKiCadUtils, MockComponent, MockNet
        import utils as _utils
        _utils.kicad_utils = sys.modules["utils.kicad_utils"]
        _utils.kicad_utils.KiCadUtils = MockKiCadUtils
        _utils.kicad_utils.Component = MockComponent
        _utils.kicad_utils.Net = MockNet
        from env.pcb_layout_env import Config as _C

        pcb_name = os.path.splitext(os.path.basename(args.data))[0]
        logger.info(f"Loading PCB from JSON: {args.data}")
        env = create_env_from_json(os.path.abspath(args.data))
        source_path = args.data

    else:  # --pcb
        # 真实 KiCad 路径：不 mock
        from env.pcb_layout_env import PCBLayoutEnv, Config as _C

        pcb_name = os.path.splitext(os.path.basename(args.pcb))[0]
        logger.info(f"Loading PCB from: {args.pcb}")
        env = PCBLayoutEnv(os.path.abspath(args.pcb))
        source_path = args.pcb

    Config = _C

    # ---- 清理所有旧 output ----
    import shutil
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        logger.info(f"Cleaned old output: {OUTPUT_DIR}")

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

    (min_x, min_y), (max_x, max_y) = env.board_bounds
    logger.info(
        f"Board: {max_x - min_x:.0f}×{max_y - min_y:.0f}mm, "
        f"N_movable={env.N_movable}, grid=({env.grid_width},{env.grid_height})"
    )

    # ---- 创建 Agent（固定维度架构：Head 维度 = MAX 常量，与训练一致） ----
    device = torch.device("cpu")
    agent = DQNAgent(
        n_movable=Config.MAX_MOVABLE,
        grid_w=Config.GRID_W,
        grid_h=Config.GRID_H,
        device=device,
        lr=Config.LEARNING_RATE, gamma=Config.GAMMA,
    )
    if args.model:
        agent.load(args.model)
        logger.info(f"Agent created + loaded weights from: {args.model}")
    else:
        logger.info(
            f"Agent created (random weights): "
            f"slots={Config.MAX_MOVABLE}×8, grid=({Config.GRID_W},{Config.GRID_H})"
        )

    # ---- 推理循环（搜索阶段，不生成步骤图） ----
    MAX_ATTEMPTS = args.max_attempts
    best_reward = float("-inf")
    best_state = None
    best_episode = -1
    best_actions = []
    all_completed = False

    start_time = time.time()

    # 推理阶段全局配置（只设置一次，避免在每 episode 内反复修改全局状态）
    Config.EARLY_STOP_THRESHOLD = float("-inf")  # 推理不早停
    env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

    for ep in range(1, MAX_ATTEMPTS + 1):
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

    # ④ 导出两份 PCB：先 final（活 env = 最后一集），后 best（回填 best_state）。
    # 顺序不能反：final 必须用活 env 导出，回填后再导 best。
    logger.info(
        f"Best: episode={best_episode}, reward={best_reward:.1f}, all_placed={all_completed}"
    )

    final_path = os.path.abspath(args.export) if args.export else \
        os.path.join(pcb_output_dir, "final_output.kicad_pcb")
    logger.info(f"Exporting final layout (last episode) to: {final_path}")
    env.kicad.export_layout(final_path, env.kicad.components)

    if best_state is not None:
        # ⑤ 回填 best 快照后导出 best_layout
        _restore_state(env, best_state)
        best_path = os.path.join(pcb_output_dir, "best_layout.kicad_pcb")
        logger.info(f"Exporting best layout to: {best_path}")
        env.kicad.export_layout(best_path, env.kicad.components)

    # ---- 以下为调试输出（仅在 --debug 时生成） ----
    if args.debug:
        # 重放最佳 episode 生成步骤图
        if best_actions:
            _replay_best_episode(env, agent, best_actions, pcb_output_dir)
        else:
            logger.warning("[WARN] No actions recorded, skipping step images.")

        # State JSON
        json_path = os.path.join(pcb_output_dir, "best_layout_state.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "pcb_name": pcb_name,
                "data_file": os.path.abspath(source_path),
                "episode": best_episode,
                "reward": best_reward,
                "all_placed": all_completed,
                "elapsed_seconds": round(elapsed, 2),
                "total_attempts": best_episode if all_completed else MAX_ATTEMPTS,
                **best_state,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"[FILE] State saved: {json_path}")

        # 最终布局可视化
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
