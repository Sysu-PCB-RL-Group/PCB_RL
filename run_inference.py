"""纯推理 Demo：使用真实 KiCad PCB 数据 + 训练好的 DQN 权重反复尝试布局。

不训练（epsilon=0），反复 episode 直到成功放置全部可移动元件。
追踪 reward 最高的布局，输出 state JSON + 可视化 PNG + 日志到 output/。
"""

import sys
import os
import glob

import json
import logging
import math
import time
from typing import List, Tuple, Dict, Optional

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

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))


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
    # gx2/gy2 从目标旋转的真实尺寸 + 新原点重新推导，保持宽高一致。
    # 宽高随旋转交换（dims_at_rotation），网格占用向上取整 ceil（不能 int 缩小）
    target_w, target_h = comp.dims_at_rotation(rot)
    gw = max(1, math.ceil(target_w / Config.CELL_SIZE))
    gh = max(1, math.ceil(target_h / Config.CELL_SIZE))
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
    env, agent: DQNAgent, step_img_dir: Optional[str] = None,
    inference_epsilon: float = 0.01,
) -> Tuple[float, int, int, list]:
    """执行 1 个纯推理 episode（epsilon=0，不训练，不写 buffer）。

    Args:
        env: PCBLayoutEnv 实例
        agent: DQNAgent（随机权重，仅用于 select_action）
        step_img_dir: 若不为 None，每步后保存中间布局图到此目录

    Returns:
        (episode_reward, step_count, placed_count, action_history, final_layout_reward)
        episode_reward: 仅成功放置步累加的 reward(干净质量分)
        step_count: 本 episode 总步数(含失败重试，作诊断)
        placed_count: 成功放置的元件数
        action_history: [(comp_id, gx1, gy1, rot, lay), ...]
        final_layout_reward: evaluate_final_layout 终局独立质量评估(quality_scale=1.0)
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
    # 训练 epsilon_min=0.15 掩盖了此问题，推理默认 epsilon=0.01（可用 --inference-epsilon 调高以公平评估）。
    while not done:
        # ---- 5 步自回归（推理：微量 epsilon 打破死锁） ----
        for step_id in range(5):
            mask = env._build_mask_for_step(step_id)
            obs = env._get_obs_with_pending(step_id)
            action = agent.select_action(obs, mask, step_id, epsilon=inference_epsilon)
            env._update_pending(action, step_id)

        # ---- env.step()（C3-5 保底：clamp 坐标到板框内） ----
        full_action = env._assemble_action()
        full_action = _clamp_action_to_board(env, *full_action)
        prev_placed = placed_count
        obs_next, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated
        placed_count = info.get("placed_count", 0)
        is_placed = placed_count > prev_placed  # 准确：placed_list 增加了才算成功
        # 仅累加成功放置步的 reward：失败重试步(死亡循环)只扣惩罚、不给品质项，
        # 若全步累加会被大量负惩罚主导，导致 episode_reward 失真(≈ -7755 而非真实质量分)。
        if is_placed:
            episode_reward += reward
        step_count += 1
        # 全部可移动元件放置完成即提前终止，省去后续无意义(或触发截断的)无效步
        if placed_count >= env.N_movable:
            break

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

    # 终局独立质量评估(quality_scale=1.0，与重试/失败步彻底解耦)，作为主报告值
    final_layout_reward = evaluate_final_layout(env)
    return episode_reward, step_count, placed_count, action_history, final_layout_reward


def evaluate_final_layout(env) -> float:
    """终局独立质量评估：在全部元件放置完成后的终态，以 quality_scale=1.0 一次性汇总
    布局质量分，与重试/失败步彻底解耦。不修改 env 状态（不影响后续 .kicad_pcb 导出）。

    计算口径与 env._calculate_reward 一致，但为「终局单次」语义：
      - 全局项（net_aggregation / congestion）在终态全集合上只算一次；
      - 顺序项（decoupling 按元件、large_comp 按 placed_list 顺序）逐元件累加；
      - 完成奖励 W_COMPLETION 全放完一次性发放；
      - 惩罚项（硬重叠 / 软间距 / 出界）按终态各已放置元件碰撞量求和（合法布局应≈0）。
    """
    from env.pcb_layout_env import Config
    from env.pcb_utils import PCBUtils

    quality_scale = 1.0
    placed_ids = list(env.placed_list)  # 放置顺序
    if not placed_ids:
        return 0.0

    reward = 0.0

    # ---- 全局品质项：Net 聚合度（终态全集合算一次） ----
    agg_reward = 0.0
    for net_id, net in env.nets.items():
        comp_bboxes = []
        seen = set()
        for conn_ref, _ in net.connections:
            if conn_ref in env.kicad.components and conn_ref not in seen:
                c = env.kicad.components[conn_ref]
                if c.is_placed:
                    seen.add(conn_ref)
                    comp_bboxes.append(c.bbox)
        if len(comp_bboxes) >= 2:
            agg = PCBUtils.net_aggregation(comp_bboxes)
            agg_score = max(0.0, 1.0 - agg / max(env.board_diagonal, 0.01))
            agg_reward += agg_score * Config.W_NET_AGGREGATION
    reward += agg_reward * quality_scale

    # ---- 全局品质项：拥塞均衡度（终态全集合算一次） ----
    placed_comps = [env.kicad.components[rid] for rid in placed_ids]
    cong_std = PCBUtils.congestion_std(
        placed_comps, env.nets, env.valid_area, Config.CONGESTION_CELL_SIZE
    )
    max_std = max(1.0, np.sqrt(env.N_movable))
    cong_score = max(0.0, 1.0 - cong_std / max_std)
    reward += cong_score * Config.W_CONGESTION * quality_scale

    # ---- 按元件品质项：去耦电容距离 ----
    decap_reward = 0.0
    for rid in placed_ids:
        comp = env.kicad.components[rid]
        for cap_id in env.decoupling_map.get(comp.id, ()):
            cap = env.kicad.components.get(cap_id)
            if cap is not None and cap.is_placed:
                dist = PCBUtils.min_pin_to_cap_distance(comp, cap)
                decap_reward += max(
                    0.0, 1.0 - dist / Config.DECOUPLING_MAX_DIST
                ) * Config.W_DECOUPLING
    reward += decap_reward * quality_scale

    # ---- 按元件品质项：大元件排名（按 placed_list 顺序，order_factor 用真实放置序） ----
    # 注意：env 中 _calculate_reward 在 placed_list.append 之后调用，故放置序为 1-based(order+1)
    large_comp_reward = 0.0
    for order, rid in enumerate(placed_ids):
        comp = env.kicad.components[rid]
        if comp.id in env.area_ranks:
            rank = env.area_ranks[comp.id]
            size_factor = max(0.0, 1.0 - rank / env.N_movable)
            order_factor = max(0.0, 1.0 - (order + 1) / env.N_movable)
            large_comp_reward += size_factor * order_factor * Config.W_LARGE_COMP
    reward += large_comp_reward * quality_scale

    # ---- 完成奖励（全放完一次性发放） ----
    reward += Config.W_COMPLETION

    # ---- 惩罚项（终态残余碰撞/出界，合法布局应≈0） ----
    total_hard = total_soft = total_oob = 0.0
    all_placed = [env.kicad.components[rid] for rid in placed_ids]
    for comp in all_placed:
        for other in all_placed + env.fixed_list:
            if other.id == comp.id:
                continue
            # 异层元件物理被板材隔开，不计碰撞（修复：碰撞前提为同层）
            if comp.layer != other.layer:
                continue
            threshold = Config.spacing_threshold_for(comp.package_type, other.package_type)
            hard, soft = PCBUtils.bbox_overlap_penalty(comp.bbox, other.bbox, threshold)
            total_hard += hard
            total_soft += soft
        total_oob += PCBUtils.out_of_bounds_area(comp.bbox, env.valid_area)
    reward -= Config.W_OVERLAP_HARD * total_hard
    reward -= Config.W_OVERLAP_SOFT * total_soft
    reward -= env.W_OUT_OF_BOUNDS_CURRENT * total_oob

    return float(reward)


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
        # 仅累加成功放置步的 reward，与 run_inference_episode 口径一致
        if act.get("is_placed", False):
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

def _find_pcb_files(data_dir: str) -> list:
    """递归扫描目录获取所有 .kicad_pcb 文件（委托到 utils.pcb_scan）。"""
    from utils.pcb_scan import find_pcb_files
    return find_pcb_files(data_dir)


def _run_single_board_inference(args, pcb_path: str, agent: "DQNAgent"):
    """对一个板子执行完整推理流程。

    Args:
        args: CLI 参数
        pcb_path: .kicad_pcb 文件路径
        agent: DQNAgent（已加载权重）
    """
    from env.pcb_layout_env import PCBLayoutEnv, Config

    pcb_name = os.path.splitext(os.path.basename(pcb_path))[0]
    logger.info(f"Loading PCB from: {pcb_path}")

    try:
        env = PCBLayoutEnv(os.path.abspath(pcb_path))
    except (ValueError, RuntimeError) as e:
        logger.warning(f"跳过无法加载的板: {pcb_name}: {e}")
        return

    file_handler = None
    try:
        # 创建输出目录
        pcb_output_dir = os.path.join(OUTPUT_DIR, pcb_name)
        os.makedirs(pcb_output_dir, exist_ok=True)

        # 日志双写
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

        # 推理配置：关闭早停 + 宽松出界惩罚
        saved_early_stop = Config.EARLY_STOP_THRESHOLD
        Config.EARLY_STOP_THRESHOLD = float("-inf")
        env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

        # 推理循环
        MAX_ATTEMPTS = args.max_attempts
        best_reward = float("-inf")
        best_state = None
        best_episode = -1
        best_actions = []
        best_steps = -1
        all_completed = False

        start_time = time.time()

        for ep in range(1, MAX_ATTEMPTS + 1):
            ep_reward, ep_steps, ep_placed, actions, final_layout_reward = run_inference_episode(
                env, agent, inference_epsilon=args.inference_epsilon)

            # 最佳选择改用终局独立质量分(与重试次数解耦)，而非被失败步污染的 ep_reward
            if final_layout_reward > best_reward:
                best_reward = final_layout_reward
                best_state = snapshot_state(env)
                best_episode = ep
                best_actions = actions
                best_steps = ep_steps

            if ep % 50 == 0 or ep_placed == env.N_movable or ep == 1:
                status = "[COMPLETE]" if ep_placed == env.N_movable else ""
                logger.info(
                    f"[Ep {ep:4d}] Quality={final_layout_reward:8.2f}  "
                    f"placed={ep_placed}/{env.N_movable}  "
                    f"attempts={ep_steps:3d}  stepR={ep_reward:8.2f}  {status}"
                )

            if ep_placed == env.N_movable:
                all_completed = True
                logger.info(f"\n[OK] All {env.N_movable} components placed at episode {ep}!")
                break

        elapsed = time.time() - start_time

        if not all_completed and best_state:
            actual_placed = sum(
                1 for c in best_state["components"]
                if c["is_placed"] and not c["is_fixed"]
            )
            logger.warning(
                f"\n[WARN] Could not place all components in {MAX_ATTEMPTS} attempts. "
                f"Best: {actual_placed}/{best_state['movable_count']} placed (ep {best_episode})"
            )

        # 导出
        logger.info(
            f"Best: episode={best_episode}, quality={best_reward:.1f}, "
            f"attempts={best_steps}, all_placed={all_completed}"
        )

        final_path = os.path.abspath(args.export) if args.export else \
            os.path.join(pcb_output_dir, "final_output.kicad_pcb")
        logger.info(f"Exporting final layout to: {final_path}")
        env.kicad.export_layout(final_path, env.kicad.components)

        if best_state is not None:
            _restore_state(env, best_state)
            best_path = os.path.join(pcb_output_dir, "best_layout.kicad_pcb")
            logger.info(f"Exporting best layout to: {best_path}")
            env.kicad.export_layout(best_path, env.kicad.components)

        # Debug 输出
        if args.debug:
            if best_actions:
                _replay_best_episode(env, agent, best_actions, pcb_output_dir)
            else:
                logger.warning("[WARN] No actions recorded, skipping step images.")

            json_path = os.path.join(pcb_output_dir, "best_layout_state.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pcb_name": pcb_name,
                    "data_file": os.path.abspath(pcb_path),
                    "episode": best_episode,
                    "layout_quality": best_reward,
                    "reward": best_reward,
                    "attempt_steps": best_steps,
                    "all_placed": all_completed,
                    "elapsed_seconds": round(elapsed, 2),
                    "total_attempts": best_episode if all_completed else MAX_ATTEMPTS,
                    **(best_state or {}),
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"[FILE] State saved: {json_path}")

            if best_state is not None:
                from utils.visualization import draw_layout
                components_for_viz = _snapshot_to_viz_components(best_state)
                board_bounds = (
                    (best_state["board_bounds"]["min_x"], best_state["board_bounds"]["min_y"]),
                    (best_state["board_bounds"]["max_x"], best_state["board_bounds"]["max_y"]),
                )
                title = (
                    f"Best Layout — {pcb_name}  Ep {best_episode}  "
                    f"Quality={best_reward:.0f}  attempts={best_steps}"
                )
                png_path = os.path.join(pcb_output_dir, "best_layout.png")
                draw_layout(components_for_viz, board_bounds, png_path, title=title)
                logger.info(f"[FILE] Layout visualization saved: {png_path}")

        # 恢复全局配置
        Config.EARLY_STOP_THRESHOLD = saved_early_stop

        logger.info(f"[OK] Inference finished for {pcb_name}. See output/{pcb_name}/ for results.\n")

    finally:
        env.close()
        if file_handler is not None:
            logger.removeHandler(file_handler)


def main():
    """主推理流程：支持单文件或批量目录模式。

    - 默认（单文件）：python run_inference.py
        只推理默认单板（utils.pcb_scan.DEFAULT_BOARD = 唯一兼容当前固定维度框架的板）
    - 单文件（指定）：python run_inference.py --pcb path/to/board.kicad_pcb
    - 批量：python run_inference.py --pcb data/kicad-native-batch/
    """
    import argparse

    from utils.pcb_scan import DEFAULT_BOARD, find_pcb_files

    # 默认单板：只推理默认单板（其余 19 块板超 MAX_MOVABLE/GRID 上限，无法完成推理）；
    # 显式传目录即切回批量模式（原有行为）。
    _batch_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data", "kicad-native-batch",
    ))
    _default_pcb = next(
        (p for p in find_pcb_files(_batch_dir)
         if DEFAULT_BOARD in os.path.basename(p)),
        _batch_dir,
    )

    parser = argparse.ArgumentParser(description="PCB Layout Inference Demo")
    parser.add_argument(
        "--pcb", type=str,
        default=_default_pcb,
        help=f".kicad_pcb 文件路径或 batch 根目录 "
             f"(默认：默认单板 '{DEFAULT_BOARD}'；传目录则批量推理).",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="导出最佳布局为 .kicad_pcb 文件路径（批量模式自动忽略，默认按板子名导出）",
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
    parser.add_argument(
        "--inference-epsilon", type=float, default=0.01,
        help="贪心推理时的探索率（打破确定性死锁，默认 0.01；调高如 0.15 可公平评估策略）",
    )
    parser.add_argument(
        "--deadlock-preview", action="store_true",
        help="开启 1c 防死锁前瞻预演（forward checking）：落子前检查剩余元件仍可行",
    )
    parser.add_argument(
        "--mrv-order", action="store_true",
        help="路线3：开启 MRV 顺序约束（step0 只放行剩余合法落子最少的元件；"
             "需与训练同开，仅用于对应重训模型的推理）",
    )
    args = parser.parse_args()

    # ---- 确定 PCB 文件列表 ----
    if os.path.isdir(args.pcb):
        pcb_file_list = _find_pcb_files(args.pcb)
        if not pcb_file_list:
            logger.error(f"No .kicad_pcb files found in {args.pcb}")
            return
        logger.info(f"Batch mode: found {len(pcb_file_list)} PCB files")
        for f in pcb_file_list:
            logger.info(f"  {f}")
    elif os.path.isfile(args.pcb):
        pcb_file_list = [args.pcb]
    else:
        logger.error(f"Invalid --pcb path: {args.pcb}")
        return

    # ---- 加载模型/创建 Agent（全局一份，所有板子共用） ----
    from env.pcb_layout_env import Config

    # 加载训练超参与固定维度参数（argument.txt，含 MAX_MOVABLE/GRID_W/GRID_H），
    # 保证推理的 reward 口径与训练一致（如 W_COMPLETION 等权重）
    _config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
    Config.load_from_file(os.path.join(_config_dir, "argument.txt"))
    # feat 引入的推理端开关：1c 防死锁前瞻 / MRV 顺序约束（维度参数已并入 argument.txt，
    # 不再单独 load admin.txt）
    if getattr(args, "deadlock_preview", False):
        Config.ENABLE_DEADLOCK_PREVIEW = True
        logger.info("1c 防死锁前瞻预演: ON")
    if getattr(args, "mrv_order", False):
        Config.ENABLE_MRV_ORDER = True
        logger.info("路线3 MRV 顺序约束: ON")

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
        logger.info(f"Agent loaded weights from: {args.model}")
    else:
        logger.info(
            f"Agent created (random weights): "
            f"slots={Config.MAX_MOVABLE}×8, grid=({Config.GRID_W},{Config.GRID_H})"
        )

    # ---- 旧 output 轮转备份（最多 5 份） ----
    import shutil
    from datetime import datetime
    if os.path.isdir(OUTPUT_DIR):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(OUTPUT_DIR, "..", "output_backup")
        os.makedirs(backup_dir, exist_ok=True)

        backup_name = f"output_{timestamp}"
        backup_dst = os.path.join(backup_dir, backup_name)
        shutil.move(OUTPUT_DIR, backup_dst)
        logger.info(f"Backed up old output → output_backup/{backup_name}")

        # 清理旧备份：超过 5 份删最旧的
        all_backups = sorted(glob.glob(os.path.join(backup_dir, "output_*")))
        while len(all_backups) > 5:
            oldest = all_backups[0]
            try:
                shutil.rmtree(oldest)
                logger.info(f"Removed old output backup: output_backup/{os.path.basename(oldest)}")
            except OSError:
                pass
            del all_backups[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 逐个板子推理 ----
    for pcb_path in pcb_file_list:
        _run_single_board_inference(args, pcb_path, agent)

    logger.info(f"\n[DONE] All {len(pcb_file_list)} boards processed. See output/ for results.")


if __name__ == "__main__":
    main()
