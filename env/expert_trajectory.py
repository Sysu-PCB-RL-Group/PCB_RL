"""专家轨迹生成模块：从 KiCad 源数据的完美布局反向拆解伪造正向轨迹。

核心思路：
  1. 从 KiCad 源数据的完整布局出发，按"先小后大"策略逐件反向拆解
  2. 拆解过程中施加微量变异（位置/旋转），生成多样化轨迹
  3. 反转拆解链路得到正向轨迹，按 5 步子步骤拆为 transition
  4. 所有 transition 序列化保存到 data/expert_transitions/

基于 SPEC：反向拆解 → 伪造完美轨迹 → DQN BC 预训练 + RL 注入。
"""

import copy
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.pcb_layout_env import Config
from env.pcb_utils import PCBUtils

logger = logging.getLogger(__name__)

# ===================== 默认超参数（可被 argument.txt 覆盖） =====================

DEFAULT_TOTAL_TRANSITIONS = 20000
DEFAULT_MAX_TRAJECTORIES = 5000
DEFAULT_EPSILON_DECOMPOSE = 0.5
DEFAULT_POS_NOISE_PROB = 0.2
DEFAULT_ROT_NOISE_PROB = 0.1

# 专家轨迹与训练侧 run.py 共用的 5 步 reward 拆分权重（唯一事实来源）。
# 禁止在本地重新硬编码，否则两者标准会静默分叉（见 Req2 一致性审查）。
EXPERT_SHARE_WEIGHTS = [0.2, 0.3, 0.3, 0.1, 0.1]


# ===================== 专家轨迹生成器 =====================

class ExpertTrajectoryGenerator:
    """从 KiCad 源数据生成伪造完美布局轨迹。

    通过反向拆解 + 反转策略，将 KiCad 中已完成的专家布局转换为
    多条可用于 DQN 训练的自回归 5 步子步骤 transition 序列。

    Attributes:
        total_transitions: 目标 transition 总数（生成直到达到即停）
        max_trajectories: 最大轨迹数安全上限（防死循环）
        epsilon_decompose: 拆解随机化概率 ε
        pos_noise_prob: 位置变异概率
        rot_noise_prob: 旋转变异概率
    """

    def __init__(
        self,
        total_transitions: int = DEFAULT_TOTAL_TRANSITIONS,
        max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
        epsilon_decompose: float = DEFAULT_EPSILON_DECOMPOSE,
        pos_noise_prob: float = DEFAULT_POS_NOISE_PROB,
        rot_noise_prob: float = DEFAULT_ROT_NOISE_PROB,
    ):
        """初始化生成器。

        Args:
            total_transitions: 目标 transition 总数（生成直到达到即停）
            max_trajectories: 最大轨迹数安全上限（防死循环）
            epsilon_decompose: 拆解随机化概率（50% 随机选 / 50% 选最小面积）
            pos_noise_prob: 位置变异概率（±1~2 格偏移）
            rot_noise_prob: 旋转变异概率（全 4 种旋转）
        """
        self.total_transitions = total_transitions
        self.max_trajectories = max_trajectories
        self.epsilon_decompose = epsilon_decompose
        self.pos_noise_prob = pos_noise_prob
        self.rot_noise_prob = rot_noise_prob

    # ==================== 核心：反向拆解 ====================

    def decompose_backward(
        self,
        env,
        shift: Tuple[float, float] = (0.0, 0.0),
    ) -> List[Tuple[str, dict]]:
        """从完整布局反向拆解，生成一条回溯链。

        每次选择一个元件拆掉，记录拆掉前的完整元件状态快照。
        选择策略：ε 概率随机选，1-ε 概率选面积最小的。

        Args:
            env: 已加载 KiCad 源数据的 PCBLayoutEnv（元件全部 is_placed=True）
            shift: 整体平移偏移 (dx, dy) mm，在 reset 后对所有 movable 元件
                bbox 平移，用于生成几何等价布局（调用方保证平移后不越界）。

        Returns:
            回溯链列表 [(comp_id, snapshot), ...]，按拆解顺序排列。
            snapshot 为 {comp_id: {bbox, rotation, layer, is_placed, ...}} 完整快照。
            第一个元素 = 拆掉第一个元件前的完整布局状态。
            最后一个元素 = 拆到只剩最后一个可动元件时的状态。
        """
        # 初始状态：所有 movable 元件均已放置
        env.reset()
        # 几何平移：必须在 reset（restore_initial_state）之后应用，否则被冲掉
        if shift != (0.0, 0.0):
            dx, dy = shift
            for comp in env.movable_list:
                x1, y1, x2, y2 = comp.bbox
                comp.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        for comp in env.movable_list:
            comp.is_placed = True
        env.placed_list = [c.id for c in env.movable_list]

        chain = []
        movable_pool = list(env.movable_list)

        while len(movable_pool) > 0:
            # 快照当前全部元件的状态
            snapshot = self._snapshot_all_components(env)

            # 选择一个元件拆掉
            chosen = self._select_component_to_remove(movable_pool)

            # 记录（拆掉前状态 + 拆掉的元件 ID）
            chain.append((chosen.id, snapshot))

            # 从 pool 中移除
            movable_pool = [c for c in movable_pool if c.id != chosen.id]
            # 更新 env 中的状态
            env.placed_list.remove(chosen.id)
            chosen.is_placed = False

        return chain

    def _select_component_to_remove(self, pool: List) -> object:
        """按 ε 策略从待拆解池中选择一个元件。

        Args:
            pool: 当前可拆解的元件列表（Component 对象）

        Returns:
            被选中的元件对象
        """
        if random.random() < self.epsilon_decompose:
            # ε 概率：随机选
            return random.choice(pool)
        else:
            # 1-ε 概率：选面积最小的
            return min(pool, key=lambda c: c.footprint_area)

    # ==================== 微量变异 ====================

    def _mutate_component_state(
        self,
        comp_id: str,
        snapshot: dict,
        env,
    ) -> dict:
        """对元件状态施加微量变异，变异后做合法性检查（越界/重叠则回退）。

        20% 概率位置 ±1~2 格偏移，10% 概率旋转变为其它 3 种之一，
        其余情况严格继承。层数始终严格继承。

        变异后的位置若越出 valid_area、或与 snapshot 中其它已放置元件
        发生 AABB 硬重叠，则放弃本次变异、回退到原始状态，保证专家
        轨迹绝不包含非法布局。

        Args:
            comp_id: 元件 ID
            snapshot: 当前快照 {comp_id: {bbox, rotation, layer, ...}}
            env: PCBLayoutEnv 实例（用于获取 valid_area 边界）

        Returns:
            变异后的元件状态 dict {bbox, rotation, layer}（仅包含位置/旋转/层字段）
        """
        comp_state = snapshot[comp_id]
        bbox = list(comp_state["bbox"])  # [x1, y1, x2, y2]
        rotation = comp_state["rotation"]
        layer = comp_state["layer"]

        # 层数始终严格继承，不变异

        # 位置变异：20% 概率 ±1~2 格（x 或 y 单方向）
        if random.random() < self.pos_noise_prob:
            cell_size = Config.CELL_SIZE
            direction = random.choice(["x", "y"])
            offset = random.choice([-2, -1, 1, 2]) * cell_size

            if direction == "x":
                bbox[0] += offset
                bbox[2] += offset
            else:
                bbox[1] += offset
                bbox[3] += offset

        # 旋转变异：10% 概率变为其它 3 种旋转之一（覆盖全 4 种旋转）
        if random.random() < self.rot_noise_prob:
            rotation = random.choice([r for r in range(4) if r != rotation])
            # 旋转改变后 AABB 宽高可能交换：保持 bbox 中心不变重设尺寸，
            # 否则 bbox 尺寸与目标姿态不符（碰撞/reward 用错几何）
            src_rot = comp_state.get("source_rotation", comp_state["rotation"])
            if (rotation - src_rot) % 4 % 2 == 1:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                bbox[0] = cx - h / 2.0
                bbox[1] = cy - w / 2.0
                bbox[2] = cx + h / 2.0
                bbox[3] = cy + w / 2.0

        # 合法性检查：越界或硬重叠则回退到原始状态
        mutated_bbox = tuple(bbox)
        if not self._is_legal_placement(mutated_bbox, comp_id, snapshot, env.valid_area, env=env):
            return {
                "bbox": tuple(comp_state["bbox"]),
                "rotation": comp_state["rotation"],
                "layer": layer,
            }

        return {
            "bbox": mutated_bbox,
            "rotation": rotation,
            "layer": layer,
        }

    @staticmethod
    def _is_legal_placement(
        bbox: Tuple[float, float, float, float],
        comp_id: str,
        snapshot: dict,
        valid_area: Tuple[float, float, float, float],
        env=None,
    ) -> bool:
        """检查 bbox 是否合法：不越界、且与快照中其它全部元件无硬重叠。

        复用 PCBUtils.bbox_overlap_penalty（与 env._detect_collisions 同一套
        碰撞判定）。**全局检查**：遍历 snapshot 中所有元件（含尚未放置者），
        只要与任一其它元件发生硬重叠（hard > 0）即视为非法并回退。

        为什么是全局而非仅"已放置"：正向逐件放置时，先放置的元件只会被
        检查 vs 已放置邻居，若变异后侵入"稍后才放置"元件的空间则漏检，
        导致后续元件 reward 为负（with_expert 组崩坏的根因之一）。改为全局
        检查后，任何方向的重叠都会被最先移动的元件在自身放置时就拦截并回退，
        保证最终布局零硬重叠。

        只拦截硬重叠（对应 has_violations 的判定），**不拦截软间距**：
        软间距仅产生轻微 reward 惩罚、不构成违规；且量化后完美布局在
        0.2mm 边界可能出现软间距，若一并拦截会导致 base 回退失败（无法放置）。

        Args:
            bbox: 待检查的 AABB (x1, y1, x2, y2)
            comp_id: 当前元件 ID（检查时跳过自身）
            snapshot: 完整布局快照 {comp_id: {bbox, is_placed, package_type, ...}}
            valid_area: 板框有效区域 (x_min, y_min, x_max, y_max)
            env: 可选 PCBLayoutEnv，用于 keepout 检查

        Returns:
            True 表示合法（可接受该变异），False 表示越界或硬重叠
        """
        x1, y1, x2, y2 = bbox
        va_xmin, va_ymin, va_xmax, va_ymax = valid_area
        if x1 < va_xmin or y1 < va_ymin or x2 > va_xmax or y2 > va_ymax:
            return False

        # keepout 检查（可选，板框内缩禁止区域）
        keepout_zones = getattr(env, "keepout_zones", None) if env is not None else None
        if keepout_zones:
            if PCBUtils.check_keepout(bbox, keepout_zones):
                return False

        self_pkg = snapshot[comp_id].get("package_type", 0)
        self_layer = snapshot[comp_id]["layer"]  # 必须存在；缺失即抛 KeyError 暴露数据错误

        # 全局检查：遍历所有元件（含尚未放置者），跳过自身
        for other_id, other_state in snapshot.items():
            if other_id == comp_id:
                continue
            # 异层元件物理被板材隔开，不计碰撞（修复：碰撞前提为同层）
            if not PCBUtils.layers_conflict(self_layer, other_state["layer"], Config.ENABLE_DUAL_SIDED):
                continue
            other_bbox = other_state["bbox"]
            other_pkg = other_state.get("package_type", 0)
            threshold = Config.spacing_threshold_for(self_pkg, other_pkg)
            hard, soft = PCBUtils.bbox_overlap_penalty(bbox, other_bbox, threshold)
            # 只拦截硬重叠（has_violations 依据）；软间距不在此拦截
            if hard > 0.0:
                return False
        return True

    # ==================== 轨迹 → Transition 转换 ====================

    def trajectory_to_transitions(
        self,
        env,
        reverse_chain: List[Tuple[str, dict]],
        shift: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[List[Tuple], dict]:
        """将反向拆解链反转为正向轨迹，并按 5 步子步骤拆为 transition。

        Args:
            env: PCBLayoutEnv 实例
            reverse_chain: decompose_backward() 返回的回溯链
                [(comp_id, snapshot_full), ...]，第一个是最先拆掉的元件

        Returns:
            transition 列表，每条为 (obs, action, reward, next_obs, mask, done, step_id, next_step_id)
        """
        # 反转：从空板到完整布局的正向放置顺序
        forward_order = list(reversed(reverse_chain))
        # forward_order[i] = (comp_id, snapshot_after_placement)
        # 即第 i 步放置 comp_id 后，得到 snapshot_after_placement 状态

        transitions = []
        raw_trajectory = {
            "shift": [float(shift[0]), float(shift[1])],
            "steps": [],
            "total_reward": 0.0,
        }

        # 重置环境到空板状态
        env.reset()
        for comp in env.movable_list:
            comp.is_placed = False
        env.placed_list = []

        # 构建"实际布局快照"：记录每个元件的实际（变异后）状态。
        # 初始 bbox 取 target_snapshot 里的平移后原始布局；movable 元件
        # is_placed=False、fixed 元件恒 True。每放置一个元件，更新其 bbox
        # 为变异后值并置 is_placed=True，使后续元件的变异检查基于"已放置
        # 元件的实际位置"而非原始布局，避免多个元件同时变异后互相靠得过近。
        # 关键：movable 元件 bbox 必须量化对齐到网格——reward 计算与 next_obs
        # 都用网格量化坐标，若快照存连续坐标，后续 _is_legal_placement 复查
        # 和 _detect_collisions 会因 ±0.1mm 量化误差在软间距阈值边界翻转违规
        # 判定，产生"reward≈1~4"的坏专家样本。fixed 元件保持原坐标（真实位置）。
        actual_snapshot = {}
        if forward_order:
            first_snap = forward_order[0][1]
            for comp_id, comp_state in first_snap.items():
                is_fixed = comp_state.get("is_fixed", False)
                bbox = comp_state["bbox"]
                if not is_fixed:
                    _, _, bbox = self._quantize_bbox_to_grid(env, bbox)
                actual_snapshot[comp_id] = {
                    "bbox": bbox,
                    "rotation": comp_state["rotation"],
                    "layer": comp_state["layer"],
                    "is_fixed": is_fixed,
                    "is_placed": comp_state.get("is_fixed", False),
                    "package_type": comp_state.get("package_type", 0),
                }

        for step_idx, (comp_id, target_snapshot) in enumerate(forward_order):
            # 找到该元件在 movable_list 中的索引
            comp_idx = None
            for i, c in enumerate(env.movable_list):
                if c.id == comp_id:
                    comp_idx = i
                    break
            if comp_idx is None:
                continue

            # 获取专家目标状态（从 target_snapshot 中提取该元件的目标）
            comp_target = target_snapshot[comp_id]
            target_bbox = comp_target["bbox"]
            target_rotation = comp_target["rotation"]
            target_layer = comp_target["layer"]
            # 记录无变异基准（用于量化后违规回退）
            base_bbox = comp_target["bbox"]
            base_rotation = comp_target["rotation"]

            # 施加微量变异（检查 vs 已放置元件的实际变异后位置）
            mutated = self._mutate_component_state(comp_id, actual_snapshot, env)
            target_bbox = mutated["bbox"]
            target_rotation = mutated["rotation"]
            target_layer = mutated["layer"]

            # 关键修复：变异产生连续坐标 bbox，而 reward/next_obs 用量化坐标。
            # 必须把目标 bbox 量化对齐到网格，否则 ±0.1mm 量化误差会在软间距
            # 阈值 0.2 边界翻转违规判定，产生 reward≈1~4 的坏专家样本。
            gx1, gy1, quantized_bbox = self._quantize_bbox_to_grid(env, target_bbox)

            # 量化后复查合法性：量化可能使间距 < 软间距阈值（连续坐标下恰好合法、
            # 量化后翻转），此时回退到无变异基准位置的量化坐标，保证专家轨迹合法。
            if not self._is_legal_placement(
                quantized_bbox, comp_id, actual_snapshot, env.valid_area, env=env
            ):
                gx1, gy1, quantized_bbox = self._quantize_bbox_to_grid(env, base_bbox)
                target_rotation = base_rotation
                target_layer = comp_target["layer"]

            # 全程统一使用量化坐标
            target_bbox = quantized_bbox

            # 专家动作序列（5 步子步骤）
            expert_actions = [
                comp_idx,           # step 0: 选元件
                gx1,                # step 1: x1
                gy1,                # step 2: y1
                target_rotation,    # step 3: rotation
                target_layer,       # step 4: layer
            ]

            # 生成 5 步子步骤 transition
            step_transitions, expert_reward = self._build_step_transitions(
                env, comp_idx, expert_actions, step_idx,
            )

            transitions.extend(step_transitions)

            # ---- 记录 raw 轨迹步骤（未变异基准 / 变异类型 / 拆分前完整 reward） ----
            _, _, base_quant_bbox = self._quantize_bbox_to_grid(env, base_bbox)
            pos_mutated = target_bbox != base_quant_bbox
            rot_mutated = target_rotation != base_rotation
            if pos_mutated and rot_mutated:
                mut_type = "both"
            elif pos_mutated:
                mut_type = "pos"
            elif rot_mutated:
                mut_type = "rot"
            else:
                mut_type = "none"
            raw_trajectory["steps"].append({
                "comp_id": comp_id,
                "target": {
                    "bbox": [float(v) for v in target_bbox],
                    "rotation": int(target_rotation),
                    "layer": int(target_layer),
                },
                "base": {
                    "bbox": [float(v) for v in base_quant_bbox],
                    "rotation": int(base_rotation),
                    "layer": int(comp_target["layer"]),
                },
                "mutated": mut_type != "none",
                "mut_type": mut_type,
                "reward_full": float(expert_reward),
            })
            raw_trajectory["total_reward"] += float(expert_reward)

            # 更新环境状态：放置该元件（量化坐标，与 reward/next_obs 一致）
            comp = env.movable_list[comp_idx]
            comp.bbox = target_bbox
            comp.rotation = target_rotation
            comp.layer = target_layer
            comp.is_placed = True
            env.placed_list.append(comp.id)

            # 同步更新实际布局快照（供后续元件的变异检查使用）
            actual_snapshot[comp_id]["bbox"] = target_bbox
            actual_snapshot[comp_id]["rotation"] = target_rotation
            actual_snapshot[comp_id]["layer"] = target_layer
            actual_snapshot[comp_id]["is_placed"] = True

        return transitions, raw_trajectory

    def _build_step_transitions(
        self,
        env,
        comp_idx: int,
        expert_actions: List[int],
        placement_step: int,
    ) -> List[Tuple]:
        """为一次元件放置构建 5 步子步骤 transition。

        Args:
            env: PCBLayoutEnv 实例（当前状态为放置前的状态）
            comp_idx: 元件在 movable_list 中的索引
            expert_actions: [comp_idx, gx1, gy1, rotation, layer]
            placement_step: 放置序号（0-based，第几个被放置的元件）

        Returns:
            5 条 transition 的列表
        """
        transitions = []

        # 清除 pending 状态
        env._reset_pending()

        # Step 0: 选元件
        mask0 = env._build_mask_for_step(0)
        obs0 = env._get_obs_with_pending(0)
        action0 = expert_actions[0]
        env._update_pending(action0, 0)

        # Step 1: x1
        mask1 = env._build_mask_for_step(1)
        obs1 = env._get_obs_with_pending(1)
        action1 = expert_actions[1]
        env._update_pending(action1, 1)

        # Step 2: y1
        mask2 = env._build_mask_for_step(2)
        obs2 = env._get_obs_with_pending(2)
        action2 = expert_actions[2]
        env._update_pending(action2, 2)

        # Step 3: rotation
        mask3 = env._build_mask_for_step(3)
        obs3 = env._get_obs_with_pending(3)
        action3 = expert_actions[3]
        env._update_pending(action3, 3)

        # Step 4: layer
        mask4 = env._build_mask_for_step(4)
        obs4 = env._get_obs_with_pending(4)
        action4 = expert_actions[4]

        # 确定 done 状态
        is_last_placement = (len(env.placed_list) + 1 == env.N_movable)
        done = is_last_placement

        # 计算专家 reward（复用环境完整 reward，含品质项，标准与训练一致）
        expert_reward = self._compute_expert_reward(env, comp_idx, expert_actions)

        # 按 share_weights 分配到各子步骤（与 run.py 共用 EXPERT_SHARE_WEIGHTS）
        share_weights = EXPERT_SHARE_WEIGHTS
        obs_list = [obs0, obs1, obs2, obs3, obs4]
        actions = [action0, action1, action2, action3, action4]
        masks = [mask0, mask1, mask2, mask3, mask4]

        for i in range(5):
            if i < 4:
                next_obs = obs_list[i + 1]
                next_sid = i + 1
                step_done = False
            else:
                # step 4: next_obs 是放置后的完整状态
                env._reset_pending()
                # 临时设置元件状态来计算 next_obs
                comp = env.movable_list[comp_idx]
                saved_state = (comp.bbox, comp.rotation, comp.layer, comp.is_placed, list(env.placed_list))
                # 用 grid 坐标 + 实际 footprint 重建 bbox（而非 1×1 cell）
                cell_size = Config.CELL_SIZE
                x1 = env.valid_area[0] + expert_actions[1] * cell_size
                y1 = env.valid_area[1] + expert_actions[2] * cell_size
                # 按目标旋转的真实尺寸重建 bbox（宽高随旋转交换）
                tw, th = comp.dims_at_rotation(expert_actions[3])
                comp.bbox = (x1, y1, x1 + tw, y1 + th)
                comp.rotation = expert_actions[3]
                comp.layer = expert_actions[4]
                comp.is_placed = True
                env.placed_list.append(comp.id)
                next_obs = env._get_obs_with_pending(0)
                # 恢复
                comp.bbox, comp.rotation, comp.layer, comp.is_placed, env.placed_list = saved_state
                next_sid = 0 if not done else -1
                step_done = done

            step_reward = expert_reward * share_weights[i]
            transitions.append((
                self._copy_obs(obs_list[i]),
                actions[i],
                step_reward,
                self._copy_obs(next_obs),
                masks[i],
                step_done,
                i,
                next_sid,
            ))

        return transitions, expert_reward

    def _compute_expert_reward(self, env, comp_idx: int, expert_actions: List[int]) -> float:
        """计算专家放置的完整 reward（与训练环境 _calculate_reward 完全一致）。

        专家轨迹代表理想最终布局，因此 quality_scale 强制为 1.0（品质项完全
        生效），复用 env._detect_collisions + env._calculate_reward，保证专家
        注入的 reward 标签与训练 reward 标准同步（含品质项，而非仅完成奖励）。

        Args:
            env: PCBLayoutEnv 实例
            comp_idx: 元件在 movable_list 中的索引
            expert_actions: [comp_idx, gx1, gy1, rotation, layer]

        Returns:
            完整 reward 值
        """
        comp = env.movable_list[comp_idx]

        # 保存当前元件状态，临时设为专家目标位置（与 _build_step_transitions
        # 中 next_obs 的重建方式一致），保持 is_placed=False、不加入 placed_list，
        # 与 env.step() 调用 _calculate_reward 时的状态约定一致。
        saved = (comp.bbox, comp.rotation, comp.layer)
        cell_size = Config.CELL_SIZE
        x1 = env.valid_area[0] + expert_actions[1] * cell_size
        y1 = env.valid_area[1] + expert_actions[2] * cell_size
        # 按目标旋转的真实尺寸重建 bbox（宽高随旋转交换）
        tw, th = comp.dims_at_rotation(expert_actions[3])
        comp.bbox = (x1, y1, x1 + tw, y1 + th)
        comp.rotation = expert_actions[3]
        comp.layer = expert_actions[4]

        try:
            total_hard_overlap, total_soft_spacing, total_out_of_bounds, in_keepout = \
                env._detect_collisions(comp)
            has_violations = (
                total_hard_overlap > 0.0 or total_out_of_bounds > 0.0 or in_keepout
            )
            return env._calculate_reward(
                comp,
                total_hard_overlap,
                total_soft_spacing,
                total_out_of_bounds,
                has_violations,
                quality_scale=1.0,
            )
        finally:
            comp.bbox, comp.rotation, comp.layer = saved

    def _grid_to_bbox(
        self,
        env,
        gx1: int,
        gy1: int,
    ) -> Tuple[float, float, float, float]:
        """将网格坐标转为 mm 坐标 AABB。

        Args:
            env: PCBLayoutEnv 实例
            gx1: 网格 x 坐标
            gy1: 网格 y 坐标

        Returns:
            (x1, y1, x2, y2) AABB
        """
        cell_size = Config.CELL_SIZE
        x1 = env.valid_area[0] + gx1 * cell_size
        y1 = env.valid_area[1] + gy1 * cell_size
        # 使用 1×1 默认尺寸，实际在 transition 构造中会被覆盖
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        return (x1, y1, x2, y2)

    def _quantize_bbox_to_grid(
        self,
        env,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[int, int, Tuple[float, float, float, float]]:
        """将连续坐标 bbox 量化对齐到网格，返回量化后的网格坐标与 bbox。

        关键修复：变异产生的 target_bbox 是连续坐标（含非网格对齐的小数位），
        而 reward 计算（_compute_expert_reward）与 next_obs 重建（_build_step_transitions）
        都用网格量化坐标（valid_area[0] + gx1*CELL_SIZE）。若不量化，连续 bbox
        与量化 bbox 之间存在 ±0.1mm 误差，在软间距阈值 0.2 边界会翻转违规判定，
        产生"reward≈1~4"的坏专家样本（污染专家注入、拖垮 with_expert 训练）。
        本方法保证 env 状态与 reward/next_obs 全程使用同一套量化坐标。

        Args:
            env: PCBLayoutEnv 实例（含 valid_area、grid_width/grid_height）
            bbox: 连续坐标 (x1, y1, x2, y2) AABB

        Returns:
            (gx1, gy1, quantized_bbox) 元组，其中 quantized_bbox 与
            _compute_expert_reward 的坐标重建公式完全一致
        """
        cell_size = Config.CELL_SIZE
        gx1 = int(round((bbox[0] - env.valid_area[0]) / cell_size))
        gy1 = int(round((bbox[1] - env.valid_area[1]) / cell_size))
        # clamp 到网格范围
        gx1 = max(0, min(gx1, env.grid_width - 1))
        gy1 = max(0, min(gy1, env.grid_height - 1))

        x1 = env.valid_area[0] + gx1 * cell_size
        y1 = env.valid_area[1] + gy1 * cell_size
        # 只把原点对齐到网格，宽高保持传入 bbox 不变（由调用方用
        # dims_at_rotation(target_rot) 重建，保证与 reward/next_obs 一致）
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (gx1, gy1, (x1, y1, x1 + w, y1 + h))

    def _copy_obs(self, obs: dict) -> dict:
        """深拷贝 dict 观测（component_slots 为 numpy array）。

        Args:
            obs: Dict 观测 {"component_slots": (M,8), "global_features": (8,)}

        Returns:
            深拷贝的观测 dict
        """
        return {
            "component_slots": obs["component_slots"].copy(),
            "global_features": obs["global_features"].copy(),
        }

    # ==================== 快照工具 ====================

    @staticmethod
    def _snapshot_all_components(env) -> dict:
        """快照当前环境中所有元件的完整状态。

        Args:
            env: PCBLayoutEnv 实例

        Returns:
            {comp_id: {id, footprint, package_type, bbox, rotation, layer,
                       is_fixed, is_placed, nets, footprint_w, footprint_h}}
        """
        snapshot = {}
        for comp_id, comp in env.kicad.components.items():
            snapshot[comp_id] = {
                "id": comp_id,
                "footprint": getattr(comp, "footprint", ""),
                "package_type": comp.package_type,
                "bbox": tuple(comp.bbox),
                "rotation": comp.rotation,
                "layer": comp.layer,
                "is_fixed": comp.is_fixed,
                "is_placed": comp.is_placed,
                "nets": list(comp.nets),
                "footprint_w": comp.footprint_w,
                "footprint_h": comp.footprint_h,
                "source_rotation": getattr(comp, "source_rotation", comp.rotation),
            }
        return snapshot

    # ==================== 批量生成 & 序列化 ====================

    def _compute_layout_bounds(self, env) -> Tuple[float, float, float, float]:
        """计算所有 movable 元件的 bbox 包围盒。

        需在 env.reset() 之后调用，此时 bbox 为原始（未平移）布局。

        Args:
            env: PCBLayoutEnv 实例

        Returns:
            (min_x, min_y, max_x, max_y)；无 movable 元件时返回全零
        """
        if not env.movable_list:
            return (0.0, 0.0, 0.0, 0.0)
        min_x = min(c.bbox[0] for c in env.movable_list)
        min_y = min(c.bbox[1] for c in env.movable_list)
        max_x = max(c.bbox[2] for c in env.movable_list)
        max_y = max(c.bbox[3] for c in env.movable_list)
        return (min_x, min_y, max_x, max_y)

    def _sample_legal_shift(self, env) -> Tuple[float, float]:
        """在合法范围内随机采样一个整数格平移偏移 (dx, dy)。

        合法范围 = valid_area 减去布局包围盒。布局占满板框时余量为 0，
        退化为 (0, 0)。偏移取 CELL_SIZE 的整数倍，保证平移后仍落在网格上，
        且所有 movable 元件仍在 valid_area 内（不越界）。

        Args:
            env: PCBLayoutEnv 实例（已 reset，movable bbox 为原始布局）

        Returns:
            (dx, dy) 平移偏移，单位 mm
        """
        min_x, min_y, max_x, max_y = self._compute_layout_bounds(env)
        va_xmin, va_ymin, va_xmax, va_ymax = env.valid_area
        cell = Config.CELL_SIZE

        # dx 下界：min_x + dx >= va_xmin → dx >= va_xmin - min_x，向上取整
        # dx 上界：max_x + dx <= va_xmax → dx <= va_xmax - max_x，向下取整
        lo_x = int(np.ceil((va_xmin - min_x) / cell))
        hi_x = int(np.floor((va_xmax - max_x) / cell))
        lo_y = int(np.ceil((va_ymin - min_y) / cell))
        hi_y = int(np.floor((va_ymax - max_y) / cell))

        # 防御：布局异常（本身越界，如布局比板框还大）时 lo > hi，收敛为单点
        lo_x = min(lo_x, hi_x)
        lo_y = min(lo_y, hi_y)

        dx = random.randint(lo_x, hi_x) * cell
        dy = random.randint(lo_y, hi_y) * cell
        return (dx, dy)

    def generate_all_expert_transitions(
        self,
        env,
        pcb_name: str,
    ) -> List[Tuple]:
        """为一块 PCB 生成专家 transition，直到总数达到目标阈值。

        每次循环：随机采样合法平移偏移 → 反向拆解（带平移）→ 变异 →
        拆为 5 步子步骤 transition → 累加，直到
        `len(transitions) >= total_transitions` 或达到 `max_trajectories` 上限。

        Args:
            env: 已加载 KiCad 源数据的 PCBLayoutEnv
            pcb_name: PCB 文件名

        Returns:
            所有 transition 的列表（可能略超 total_transitions）
        """
        all_transitions = []
        all_raw = []
        traj_count = 0

        # 先 reset 一次，确保 bbox 为原始布局，用于计算合法平移范围
        env.reset()

        while len(all_transitions) < self.total_transitions and \
                traj_count < self.max_trajectories:
            # 关键修复：trajectory_to_transitions() 末尾会把 env 元件 bbox 停留在
            # "shift 后" 状态且不恢复。若不在此 reset，_sample_legal_shift() 会基于
            # 上一轮残留的 shift 后布局计算合法范围，而 decompose_backward() 内部
            # 又对"原始布局"应用该 shift → 采样基准与应用基准不一致 → 布局越界，
            # 且越界会逐轮正反馈放大（小批量正常、大批量 gx 聚集 0/35 的根因）。
            # 每轮采样前强制 reset，保证 _sample_legal_shift 始终以原始布局为基准。
            env.reset()

            # 几何平移：随机采样一个不越界的整体偏移
            shift = self._sample_legal_shift(env)

            # 反向拆解（带平移）
            chain = self.decompose_backward(env, shift)

            if len(chain) < 1:
                traj_count += 1
                logger.warning(f"[{pcb_name}] traj {traj_count}: empty chain, skipping")
                continue

            # 反转 + 拆为 transition（同时产出 raw 轨迹，含 shift/基准/变异/完整 reward）
            transitions, raw_traj = self.trajectory_to_transitions(env, chain, shift)
            all_transitions.extend(transitions)
            all_raw.append(raw_traj)
            traj_count += 1

        buffer_capacity = Config.EXPERT_BUFFER_MULTIPLIER * len(all_transitions)
        logger.info(
            f"[{pcb_name}] Generated {traj_count} trajectories → "
            f"{len(all_transitions)} expert transitions "
            f"(target={self.total_transitions}, buffer_capacity={buffer_capacity})"
        )
        return all_transitions, all_raw

    def save_transitions(self, transitions: List[Tuple], output_dir: str, pcb_name: str):
        """序列化专家 transition 到磁盘。

        Args:
            transitions: transition 列表
            output_dir: 输出目录
            pcb_name: PCB 文件名（用于命名）
        """
        os.makedirs(output_dir, exist_ok=True)

        serialized = []
        for t in transitions:
            obs, action, reward, next_obs, mask, done, step_id, next_step_id = t
            serialized.append({
                "obs": {
                    "component_slots": obs["component_slots"].tolist(),
                    "global_features": obs["global_features"].tolist(),
                },
                "action": int(action),
                "reward": float(reward),
                "next_obs": {
                    "component_slots": next_obs["component_slots"].tolist(),
                    "global_features": next_obs["global_features"].tolist(),
                },
                "mask": mask.tolist() if mask is not None else None,
                "done": bool(done),
                "step_id": int(step_id),
                "next_step_id": int(next_step_id),
            })

        pcb_short = pcb_name.replace(".kicad_pcb", "")
        filepath = os.path.join(output_dir, f"{pcb_short}_expert.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False)

        logger.info(f"Saved {len(serialized)} expert transitions to {filepath}")

    @staticmethod
    def load_transitions(filepath: str) -> List[Tuple]:
        """从磁盘加载专家 transition。

        Args:
            filepath: JSON 文件路径

        Returns:
            transition 列表，格式与 ReplayBuffer.add() 兼容
        """
        with open(filepath, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        transitions = []
        for item in serialized:
            obs = {
                "component_slots": np.array(item["obs"]["component_slots"], dtype=np.float32),
                "global_features": np.array(item["obs"]["global_features"], dtype=np.float32),
            }
            next_obs = {
                "component_slots": np.array(item["next_obs"]["component_slots"], dtype=np.float32),
                "global_features": np.array(item["next_obs"]["global_features"], dtype=np.float32),
            }
            mask = np.array(item["mask"], dtype=bool) if item["mask"] is not None else None
            transitions.append((
                obs, item["action"], item["reward"], next_obs,
                mask, item["done"], item["step_id"], item["next_step_id"],
            ))

        logger.info(f"Loaded {len(transitions)} expert transitions from {filepath}")
        return transitions

    def save_raw_trajectories(self, raw_trajectories: List[dict], output_dir: str, pcb_name: str) -> str:
        """序列化完整 6 元器件 raw 轨迹到磁盘（与 save_transitions 同目录）。

        raw 轨迹每步含：comp_id、目标 bbox/旋转/层、未变异基准 base、
        是否变异及类型（pos/rot/both/none）、拆分前完整 reward；轨迹级含全局 shift。

        Args:
            raw_trajectories: generate_all_expert_transitions 产出的 raw 轨迹列表
            output_dir: 输出目录
            pcb_name: PCB 文件名（用于命名）

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        pcb_short = pcb_name.replace(".kicad_pcb", "")
        filepath = os.path.join(output_dir, f"{pcb_short}_expert_raw.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(raw_trajectories, f, ensure_ascii=False)
        logger.info(f"Saved {len(raw_trajectories)} raw trajectories to {filepath}")
        return filepath

    @staticmethod
    def load_raw_trajectories(filepath: str) -> List[dict]:
        """从磁盘加载 raw 轨迹。

        Args:
            filepath: JSON 文件路径

        Returns:
            raw 轨迹列表（dict 结构，见 save_raw_trajectories）
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


# ==================== 便捷函数 ====================

def generate_and_save_expert_transitions(
    env,
    pcb_name: str,
    output_dir: str = "data/expert_transitions",
    total_transitions: int = DEFAULT_TOTAL_TRANSITIONS,
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
    epsilon_decompose: float = DEFAULT_EPSILON_DECOMPOSE,
    pos_noise_prob: float = DEFAULT_POS_NOISE_PROB,
    rot_noise_prob: float = DEFAULT_ROT_NOISE_PROB,
) -> Tuple[List[Tuple], str]:
    """便捷函数：生成并保存一块 PCB 的专家 transition。

    Args:
        env: 已加载 KiCad 源数据的 PCBLayoutEnv
        pcb_name: PCB 文件名
        output_dir: 输出目录
        total_transitions: 目标 transition 总数（达到即停）
        max_trajectories: 最大轨迹数安全上限（防死循环）
        epsilon_decompose: 拆解随机化概率
        pos_noise_prob: 位置变异概率
        rot_noise_prob: 旋转变异概率

    Returns:
        (transitions, filepath) 元组
    """
    generator = ExpertTrajectoryGenerator(
        total_transitions=total_transitions,
        max_trajectories=max_trajectories,
        epsilon_decompose=epsilon_decompose,
        pos_noise_prob=pos_noise_prob,
        rot_noise_prob=rot_noise_prob,
    )

    transitions, all_raw = generator.generate_all_expert_transitions(env, pcb_name)
    generator.save_transitions(transitions, output_dir, pcb_name)
    generator.save_raw_trajectories(all_raw, output_dir, pcb_name)

    pcb_short = pcb_name.replace(".kicad_pcb", "")
    filepath = os.path.join(output_dir, f"{pcb_short}_expert.json")

    return transitions, filepath
