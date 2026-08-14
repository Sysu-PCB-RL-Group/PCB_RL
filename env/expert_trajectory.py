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

logger = logging.getLogger(__name__)

# ===================== 默认超参数（可被 argument.txt 覆盖） =====================

DEFAULT_TRAJECTORIES_PER_BOARD = 20
DEFAULT_EPSILON_DECOMPOSE = 0.5
DEFAULT_POS_NOISE_PROB = 0.2
DEFAULT_ROT_NOISE_PROB = 0.1


# ===================== 专家轨迹生成器 =====================

class ExpertTrajectoryGenerator:
    """从 KiCad 源数据生成伪造完美布局轨迹。

    通过反向拆解 + 反转策略，将 KiCad 中已完成的专家布局转换为
    多条可用于 DQN 训练的自回归 5 步子步骤 transition 序列。

    Attributes:
        trajectories_per_board: 每板生成轨迹数
        epsilon_decompose: 拆解随机化概率 ε
        pos_noise_prob: 位置变异概率
        rot_noise_prob: 旋转变异概率
    """

    def __init__(
        self,
        trajectories_per_board: int = DEFAULT_TRAJECTORIES_PER_BOARD,
        epsilon_decompose: float = DEFAULT_EPSILON_DECOMPOSE,
        pos_noise_prob: float = DEFAULT_POS_NOISE_PROB,
        rot_noise_prob: float = DEFAULT_ROT_NOISE_PROB,
    ):
        """初始化生成器。

        Args:
            trajectories_per_board: 每板生成轨迹数
            epsilon_decompose: 拆解随机化概率（50% 随机选 / 50% 选最小面积）
            pos_noise_prob: 位置变异概率（±1 格偏移）
            rot_noise_prob: 旋转变异概率（±90°）
        """
        self.trajectories_per_board = trajectories_per_board
        self.epsilon_decompose = epsilon_decompose
        self.pos_noise_prob = pos_noise_prob
        self.rot_noise_prob = rot_noise_prob

    # ==================== 核心：反向拆解 ====================

    def decompose_backward(
        self,
        env,
    ) -> List[Tuple[str, dict]]:
        """从完整布局反向拆解，生成一条回溯链。

        每次选择一个元件拆掉，记录拆掉前的完整元件状态快照。
        选择策略：ε 概率随机选，1-ε 概率选面积最小的。

        Args:
            env: 已加载 KiCad 源数据的 PCBLayoutEnv（元件全部 is_placed=True）

        Returns:
            回溯链列表 [(comp_id, snapshot), ...]，按拆解顺序排列。
            snapshot 为 {comp_id: {bbox, rotation, layer, is_placed, ...}} 完整快照。
            第一个元素 = 拆掉第一个元件前的完整布局状态。
            最后一个元素 = 拆到只剩最后一个可动元件时的状态。
        """
        # 初始状态：所有 movable 元件均已放置
        env.reset()
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
        """对元件状态施加微量变异。

        20% 概率位置 ±1 格偏移，10% 概率 rotation ±90°，70% 概率严格继承。
        层数始终严格继承。

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

        # 位置变异：20% 概率 ±1 格
        if random.random() < self.pos_noise_prob:
            cell_size = 1.0  # Config.CELL_SIZE = 1.0
            direction = random.choice(["x", "y"])
            offset = random.choice([-1, 1]) * cell_size

            if direction == "x":
                bbox[0] += offset
                bbox[2] += offset
            else:
                bbox[1] += offset
                bbox[3] += offset

            # Clamp 到 valid_area 内
            va_xmin, va_ymin, va_xmax, va_ymax = env.valid_area
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox[0] = max(va_xmin, min(bbox[0], va_xmax - w))
            bbox[1] = max(va_ymin, min(bbox[1], va_ymax - h))
            bbox[2] = bbox[0] + w
            bbox[3] = bbox[1] + h

        # 旋转变异：10% 概率 ±90°
        if random.random() < self.rot_noise_prob:
            delta = random.choice([-1, 1])
            rotation = (rotation + delta) % 4

        return {
            "bbox": tuple(bbox),
            "rotation": rotation,
            "layer": layer,
        }

    # ==================== 轨迹 → Transition 转换 ====================

    def trajectory_to_transitions(
        self,
        env,
        reverse_chain: List[Tuple[str, dict]],
    ) -> List[Tuple]:
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

        # 重置环境到空板状态
        env.reset()
        for comp in env.movable_list:
            comp.is_placed = False
        env.placed_list = []

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

            # 施加微量变异
            mutated = self._mutate_component_state(comp_id, target_snapshot, env)
            target_bbox = mutated["bbox"]
            target_rotation = mutated["rotation"]
            target_layer = mutated["layer"]

            # 计算网格坐标
            gx1 = int(round((target_bbox[0] - env.valid_area[0]) / 1.0))  # CELL_SIZE=1.0
            gy1 = int(round((target_bbox[1] - env.valid_area[1]) / 1.0))
            # clamp 到网格范围
            gx1 = max(0, min(gx1, env.grid_width - 1))
            gy1 = max(0, min(gy1, env.grid_height - 1))

            # 专家动作序列（5 步子步骤）
            expert_actions = [
                comp_idx,           # step 0: 选元件
                gx1,                # step 1: x1
                gy1,                # step 2: y1
                target_rotation,    # step 3: rotation
                target_layer,       # step 4: layer
            ]

            # 生成 5 步子步骤 transition
            step_transitions = self._build_step_transitions(
                env, comp_idx, expert_actions, step_idx,
            )

            transitions.extend(step_transitions)

            # 更新环境状态：放置该元件
            comp = env.movable_list[comp_idx]
            comp.bbox = target_bbox
            comp.rotation = target_rotation
            comp.layer = target_layer
            comp.is_placed = True
            env.placed_list.append(comp.id)

        return transitions

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

        # 计算专家 reward（全部正确放置，无违规）
        # 使用简化的正向 reward 计算
        expert_reward = self._compute_expert_reward(env, placement_step)

        # 按 share_weights 分配到各子步骤
        share_weights = [0.2, 0.3, 0.3, 0.1, 0.1]
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
                cell_size = 1.0
                x1 = env.valid_area[0] + expert_actions[1] * cell_size
                y1 = env.valid_area[1] + expert_actions[2] * cell_size
                comp.bbox = (x1, y1, x1 + comp.footprint_w, y1 + comp.footprint_h)
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

        return transitions

    def _compute_expert_reward(self, env, placement_step: int) -> float:
        """计算专家放置的正向 reward。

        使用简化的 reward 计算：基础完成奖励 + 品质奖励。

        Args:
            env: PCBLayoutEnv 实例
            placement_step: 放置序号

        Returns:
            正向 reward 值
        """
        from env.pcb_layout_env import Config

        reward = 0.0

        # 完成奖励：每个元件成功放置 = W_COMPLETION / N_movable
        completion_partial = Config.W_COMPLETION / max(env.N_movable, 1)
        reward += completion_partial

        # 最后一个元件额外奖励
        if placement_step + 1 == env.N_movable:
            completion_bonus = Config.W_COMPLETION - completion_partial * env.N_movable
            reward += completion_bonus

        return reward

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
        cell_size = 1.0  # Config.CELL_SIZE
        x1 = env.valid_area[0] + gx1 * cell_size
        y1 = env.valid_area[1] + gy1 * cell_size
        # 使用 1×1 默认尺寸，实际在 transition 构造中会被覆盖
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        return (x1, y1, x2, y2)

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
            }
        return snapshot

    # ==================== 批量生成 & 序列化 ====================

    def generate_all_expert_transitions(
        self,
        env,
        pcb_name: str,
    ) -> List[Tuple]:
        """为一块 PCB 生成全部专家 transition。

        生成 N 条随机化轨迹 → 每条反转 → 拆为 5 步子步骤 transition。

        Args:
            env: 已加载 KiCad 源数据的 PCBLayoutEnv
            pcb_name: PCB 文件名

        Returns:
            所有 transition 的列表
        """
        all_transitions = []

        for traj_idx in range(self.trajectories_per_board):
            # 反向拆解
            chain = self.decompose_backward(env)

            if len(chain) < 1:
                logger.warning(f"[{pcb_name}] traj {traj_idx}: empty chain, skipping")
                continue

            # 反转 + 拆为 transition
            transitions = self.trajectory_to_transitions(env, chain)
            all_transitions.extend(transitions)

        logger.info(
            f"[{pcb_name}] Generated {self.trajectories_per_board} trajectories "
            f"→ {len(all_transitions)} expert transitions"
        )
        return all_transitions

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


# ==================== 便捷函数 ====================

def generate_and_save_expert_transitions(
    env,
    pcb_name: str,
    output_dir: str = "data/expert_transitions",
    trajectories_per_board: int = DEFAULT_TRAJECTORIES_PER_BOARD,
    epsilon_decompose: float = DEFAULT_EPSILON_DECOMPOSE,
    pos_noise_prob: float = DEFAULT_POS_NOISE_PROB,
    rot_noise_prob: float = DEFAULT_ROT_NOISE_PROB,
) -> Tuple[List[Tuple], str]:
    """便捷函数：生成并保存一块 PCB 的专家 transition。

    Args:
        env: 已加载 KiCad 源数据的 PCBLayoutEnv
        pcb_name: PCB 文件名
        output_dir: 输出目录
        trajectories_per_board: 每板轨迹数
        epsilon_decompose: 拆解随机化概率
        pos_noise_prob: 位置变异概率
        rot_noise_prob: 旋转变异概率

    Returns:
        (transitions, filepath) 元组
    """
    generator = ExpertTrajectoryGenerator(
        trajectories_per_board=trajectories_per_board,
        epsilon_decompose=epsilon_decompose,
        pos_noise_prob=pos_noise_prob,
        rot_noise_prob=rot_noise_prob,
    )

    transitions = generator.generate_all_expert_transitions(env, pcb_name)
    generator.save_transitions(transitions, output_dir, pcb_name)

    pcb_short = pcb_name.replace(".kicad_pcb", "")
    filepath = os.path.join(output_dir, f"{pcb_short}_expert.json")

    return transitions, filepath
