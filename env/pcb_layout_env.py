"""PCB 布局 Gymnasium 环境（AABB-based，自回归 5 步动作分解）。

基于 SPEC v2.2 完整实现。支持 pending 状态观测扩展与每步 1D 掩码。
"""
import logging
from typing import Tuple, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.kicad_utils import KiCadUtils
from .pcb_utils import PCBUtils

logger = logging.getLogger(__name__)


# ===================== 默认超参数（可被 argument.txt 覆盖） =====================

class Config:
    """默认配置常量"""
    # Reward 权重
    W_OVERLAP_HARD: float = 1000.0
    W_OVERLAP_SOFT: float = 50.0
    W_OUT_OF_BOUNDS_INITIAL: float = 100.0
    W_OUT_OF_BOUNDS_FINAL: float = 1000.0
    W_NET_AGGREGATION: float = 1.0
    W_CONGESTION: float = 1.0
    W_DECOUPLING: float = 1.0
    W_LARGE_COMP: float = 1.0
    W_COMPLETION: float = 500.0

    # 训练参数
    GAMMA: float = 0.99
    MAX_STEPS: int = 2000
    TOTAL_EPISODES: int = 10000
    SAVE_INTERVAL: int = 500
    LOG_INTERVAL: int = 100
    EARLY_STOP_THRESHOLD: float = -500.0
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY_EPISODES: int = 10000
    CURRICULUM_START_EP: int = 0
    CURRICULUM_END_EP: int = 500

    # DQN 训练参数
    BATCH_SIZE: int = 64
    REPLAY_BUFFER_SIZE: int = 10000
    LEARNING_RATE: float = 0.0001
    TARGET_UPDATE_FREQ: int = 100

    # 几何阈值
    BOARD_MARGIN: float = 1.0
    DECOUPLING_MAX_DIST: float = 5.0
    CELL_SIZE: float = 1.0
    CONGESTION_CELL_SIZE: float = 2.0

    @classmethod
    def load_from_file(cls, path: str):
        """从 argument.txt 加载配置，覆盖默认值。

        Args:
            path: argument.txt 文件路径，key=value 格式
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    # 剥离行尾注释（# 之后的内容），再 strip 一次去尾部空白
                    val_str = val.split("#")[0].strip()
                    if not val_str:
                        continue
                    try:
                        # 先尝试整数解析（避免浮点精度问题），失败则 float
                        val = int(val_str)
                    except ValueError:
                        try:
                            val = float(val_str)
                        except ValueError:
                            continue
                    if hasattr(cls, key):
                        setattr(cls, key, val)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")

    @classmethod
    def spacing_threshold_for(cls, type_a: int, type_b: int) -> float:
        """查表获取两个元件之间的分类间距阈值（取较大值）。

        Args:
            type_a: 元件 A 的封装类型编码 (0-4)
            type_b: 元件 B 的封装类型编码 (0-4)

        Returns:
            两元件间距阈值 (mm)，取两者阈值之 max
        """
        thresholds = [0.2, 1.25, 2.5, 5.0, 2.0]
        return max(thresholds[type_a], thresholds[type_b])


# ===================== PCBLayoutEnv =====================

class PCBLayoutEnv(gym.Env):
    """PCB 布局 Gymnasium 环境"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, pcb_path: str, config_path: Optional[str] = None):
        """初始化 PCB 布局环境。

        加载 KiCad PCB 文件，构建动作空间、观测空间，初始化所有缓存状态。

        Args:
            pcb_path: PCB 文件路径（.kicad_pcb 格式）
            config_path: argument.txt 配置文件路径（可选）
        """
        super().__init__()

        # 加载配置
        if config_path:
            Config.load_from_file(config_path)

        self._pcb_path = pcb_path  # 保存路径，供 reset() 重新加载

        # KiCad 底层
        self.kicad = KiCadUtils()
        if not self.kicad.load_board(pcb_path):
            raise RuntimeError(f"Failed to load PCB: {pcb_path}")

        # 板框几何
        self.board_bounds = self.kicad.get_board_bounds()
        (min_x, min_y), (max_x, max_y) = self.board_bounds
        self.valid_area: Tuple[float, float, float, float] = (
            min_x + Config.BOARD_MARGIN,
            min_y + Config.BOARD_MARGIN,
            max_x - Config.BOARD_MARGIN,
            max_y - Config.BOARD_MARGIN,
        )

        # 网格离散化
        self.grid_width = max(1, int((self.valid_area[2] - self.valid_area[0]) / Config.CELL_SIZE))
        self.grid_height = max(1, int((self.valid_area[3] - self.valid_area[1]) / Config.CELL_SIZE))

        # 元件分类
        self.fixed_list = [c for c in self.kicad.components.values() if c.is_fixed]
        self.movable_list = [c for c in self.kicad.components.values() if not c.is_fixed]
        self.N_movable = len(self.movable_list)

        # 动作空间: 7 元 Tuple
        #   (component_id, x1, y1, x2, y2, rotation, layer)
        self.action_space = spaces.Tuple([
            spaces.Discrete(self.N_movable),          # component_id
            spaces.Discrete(self.grid_width),         # x1
            spaces.Discrete(self.grid_height),        # y1
            spaces.Discrete(self.grid_width),         # x2
            spaces.Discrete(self.grid_height),        # y2
            spaces.Discrete(4),                       # rotation {0,1,2,3}
            spaces.Discrete(2),                       # layer {0,1}
        ])

        # 观测空间: raw(N_movable×8) + pending(N_movable + 4)
        obs_dim = self.N_movable * 8 + self.N_movable + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(
                max_x - min_x,
                max_y - min_y,
                self.N_movable,     # net_count 上界
            ),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 训练动态参数
        self.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

        # 内部状态（reset 时初始化）
        self.placed_list: List[str] = []
        self.current_step: int = 0
        self.cumulative_reward: float = 0.0
        self.decoupling_map: Dict[str, List[str]] = {}
        self.area_ranks: Dict[str, int] = {}
        self.nets = self.kicad.nets
        self.keepout_zones = self.kicad.get_keepout_zones()

        # Pending 状态（自回归 5 步中间变量）
        self._pending_comp_idx: int = -1
        self._pending_gx1: int = -1
        self._pending_gy1: int = -1
        self._pending_rot: int = 0
        self._pending_lay: int = 0

    # ==================== Gymnasium 接口 ====================

    def reset(self, seed=None, options=None):
        """重置环境。

        重新加载 PCB 文件、重建元件列表、重置步数和已放置列表，
        构建去耦电容映射和面积排名。

        Args:
            seed: 随机种子（Gymnasium 标准接口）
            options: 额外选项（Gymnasium 标准接口）

        Returns:
            (obs, info) 元组，info 含 action_mask
        """
        super().reset(seed=seed)

        # 从内存缓存恢复原始布局（无需重新读取磁盘文件）
        self.kicad.restore_initial_state()

        # 重建 movable/fixed 列表
        self.fixed_list = [c for c in self.kicad.components.values() if c.is_fixed]
        self.movable_list = [c for c in self.kicad.components.values() if not c.is_fixed]
        self.N_movable = len(self.movable_list)

        # 重置全局状态
        self.placed_list = []
        self.current_step = 0
        self.cumulative_reward = 0.0

        # 构建辅助映射
        self.decoupling_map = PCBUtils.identify_decoupling_pairs(
            self.nets, self.kicad.components
        )
        self.area_ranks = PCBUtils.build_area_ranks(self.movable_list)
        self.keepout_zones = self.kicad.get_keepout_zones()

        # 计算板框对角线（归一化用）
        (min_x, min_y), (max_x, max_y) = self.board_bounds
        self.board_diagonal = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

        # 重置 pending 状态
        self._reset_pending()

        obs = self._get_obs_with_pending(0)
        info = {"action_mask": self._build_mask_for_step(0)}
        return obs, info

    def step(self, action: Tuple[int, ...]):
        """执行一步放置操作。

        解析 7 元动作 → 校验合法性 → 约束检测 → 计算 reward → 更新状态。

        Args:
            action: 7 元动作 (comp_idx, gx1, gy1, gx2, gy2, rot_idx, lay_idx)

        Returns:
            (obs, reward, terminated, truncated, info) 五元组
            terminated: 全部放置成功
            truncated: 超步数或 reward 低于阈值

        Raises:
            无显式异常，非法动作通过 _invalid_action 内部消化。
        """
        # 1. 解析 7 元动作
        comp_idx, gx1, gy1, gx2, gy2, rot_idx, lay_idx = action

        # 2. 检查 component_id 合法性
        if comp_idx < 0 or comp_idx >= self.N_movable:
            return self._invalid_action("component_id out of range")

        component = self.movable_list[comp_idx]

        # C1-2 检查：已放置不可再选
        if component.id in self.placed_list:
            return self._invalid_action("component already placed")

        # 3. 网格坐标 → mm 坐标
        x1 = self.valid_area[0] + gx1 * Config.CELL_SIZE
        y1 = self.valid_area[1] + gy1 * Config.CELL_SIZE
        x2 = self.valid_area[0] + gx2 * Config.CELL_SIZE
        y2 = self.valid_area[1] + gy2 * Config.CELL_SIZE

        # 确保 x2 >= x1, y2 >= y1
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # 4. AABB 尺寸校验
        # 容差 = CELL_SIZE：网格离散化最大误差 ≈ CELL_SIZE，必须不小于此值
        # 否则非整数 footprint（如 2.9、1.6mm）始终无法通过校验
        placed_w = x2 - x1
        placed_h = y2 - y1
        if abs(placed_w - component.footprint_w) > Config.CELL_SIZE or \
           abs(placed_h - component.footprint_h) > Config.CELL_SIZE:
            return self._invalid_action("AABB size mismatch with footprint")

        # 5. 更新元件状态
        component.bbox = (x1, y1, x2, y2)
        component.rotation = rot_idx
        component.layer = lay_idx

        # 6. 约束检测
        total_hard_overlap = 0.0
        total_soft_spacing = 0.0

        # C2: 与 fixed_list + placed_list 中元件的碰撞检测
        all_placed = [self.kicad.components[rid] for rid in self.placed_list]
        all_fixed_comps = self.fixed_list + all_placed

        for other in all_fixed_comps:
            if other.id == component.id:
                continue
            threshold = Config.spacing_threshold_for(
                component.package_type, other.package_type
            )
            hard, soft = PCBUtils.bbox_overlap_penalty(
                component.bbox, other.bbox, threshold
            )
            total_hard_overlap += hard
            total_soft_spacing += soft

        # C3: 出界检测
        total_out_of_bounds = PCBUtils.out_of_bounds_area(
            component.bbox, self.valid_area
        )

        # 禁止区域
        in_keepout = PCBUtils.check_keepout(component.bbox, self.keepout_zones)

        # 7. 计算 reward
        has_violations = (total_hard_overlap > 0.0 or total_out_of_bounds > 0.0 or in_keepout)
        reward = self._calculate_reward(
            component, total_hard_overlap, total_soft_spacing,
            total_out_of_bounds, has_violations
        )

        # 8. 更新 placed_list
        component.is_placed = not has_violations
        if component.is_placed:
            self.placed_list.append(component.id)

        # 9. 判定 done
        self.current_step += 1
        self.cumulative_reward += reward

        done_success = (len(self.placed_list) == self.N_movable) and not has_violations
        done_max_steps = self.current_step >= Config.MAX_STEPS
        done_early_stop = self.cumulative_reward < Config.EARLY_STOP_THRESHOLD

        terminated = done_success
        truncated = done_max_steps or done_early_stop

        # 10. 构建返回
        obs = self._get_obs_with_pending(0)  # raw obs + zero pending
        info = {
            "step": self.current_step,
            "placed_count": len(self.placed_list),
            "total_movable": self.N_movable,
            "hard_overlap": total_hard_overlap,
            "soft_spacing": total_soft_spacing,
            "out_of_bounds": total_out_of_bounds,
            "in_keepout": in_keepout,
            "decoupling_violations": self._last_reward_breakdown.get("decoupling_violations", 0),
            "action_mask": self._build_mask_for_step(0),
            "reward_breakdown": getattr(self, "_last_reward_breakdown", {}),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """文本模式渲染，打印当前步数和各元件状态。

        Args:
            mode: 渲染模式，默认 "human"
        """
        print(f"\n===== Step {self.current_step} =====")
        print(f"Placed: {len(self.placed_list)} / {self.N_movable}")
        print(f"Cumulative Reward: {self.cumulative_reward:.2f}")
        for comp in self.movable_list:
            status = "[placed]" if comp.is_placed else "[unplaced]"
            if comp.is_placed:
                x1, y1, x2, y2 = comp.bbox
                print(f"  [{status}] {comp.id:6s}  bbox=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"
                      f"  rot={comp.rotation*90}°  layer={'TOP' if comp.layer==0 else 'BOT'}")
            else:
                print(f"  [{status}] {comp.id:6s}  -- not placed --")
        print("=" * 40)

    # ==================== 内部方法 ====================

    def _get_obs(self) -> np.ndarray:
        """构建 fixed-slot 观测向量 (N_movable × 8)。

        每个元件 8 维：placed_flag, x1, y1, x2, y2, rotation, layer, net_count。

        Returns:
            shape=(N_movable*8,) 的 float32 向量
        """
        obs = np.zeros(self.N_movable * 8, dtype=np.float32)

        # 建立 comp_id → slot_idx 映射
        for slot_idx, comp in enumerate(self.movable_list):
            if not comp.is_placed:
                continue
            base = slot_idx * 8
            obs[base + 0] = 1.0                        # placed_flag
            obs[base + 1] = comp.bbox[0]               # x1
            obs[base + 2] = comp.bbox[1]               # y1
            obs[base + 3] = comp.bbox[2]               # x2
            obs[base + 4] = comp.bbox[3]               # y2
            obs[base + 5] = float(comp.rotation)       # rotation
            obs[base + 6] = float(comp.layer)          # layer
            obs[base + 7] = float(len(comp.nets))      # net_count

        return obs

    # ---- 自回归 5 步接口 ----

    def _reset_pending(self):
        """重置 5 步自回归的中间 pending 状态。

        在每次新的元件放置前调用，将所有 pending 字段清零。
        """
        self._pending_comp_idx = -1
        self._pending_gx1 = -1
        self._pending_gy1 = -1
        self._pending_rot = 0
        self._pending_lay = 0

    def _update_pending(self, action: int, step_id: int):
        """将当前子步骤的动作写入 pending 状态。

        Args:
            action: 当前子步骤选择的离散动作索引
            step_id: 子步骤编号 0=comp, 1=x1, 2=y1, 3=rot, 4=lay
        """
        if step_id == 0:
            self._pending_comp_idx = action
        elif step_id == 1:
            self._pending_gx1 = action
        elif step_id == 2:
            self._pending_gy1 = action
        elif step_id == 3:
            self._pending_rot = action
        elif step_id == 4:
            self._pending_lay = action

    def _assemble_action(self) -> Tuple[int, ...]:
        """将 pending 状态组装为完整的 7 元动作。

        根据已选的 comp/gx1/gy1/rot/lay，自动推导 gx2/gy2。
        调用方保证 5 步已完成。

        Returns:
            7 元动作 (comp_idx, gx1, gy1, gx2, gy2, rot, lay)
        """
        comp = self.movable_list[self._pending_comp_idx]
        expected_w = max(1, int(comp.footprint_w / Config.CELL_SIZE))
        expected_h = max(1, int(comp.footprint_h / Config.CELL_SIZE))
        gx2 = self._pending_gx1 + expected_w
        gy2 = self._pending_gy1 + expected_h
        return (
            self._pending_comp_idx,
            self._pending_gx1,
            self._pending_gy1,
            gx2,
            gy2,
            self._pending_rot,
            self._pending_lay,
        )

    def _get_obs_with_pending(self, step_id: int) -> np.ndarray:
        """构建带 pending 状态的观测向量。

        在 raw fixed-slot 观测后追加 N_movable 维 one-hot 元件选择
        + 4 维 (gx1, gy1, rot, lay)。未选中的维度填 0。

        Args:
            step_id: 当前子步骤 0-4，决定哪些 pending 字段已填充

        Returns:
            shape=(N_movable*8 + N_movable + 4,) 的 float32 向量
        """
        raw_obs = self._get_obs()  # shape=(N_movable*8,)

        # pending_comp: one-hot of selected component
        pending_comp = np.zeros(self.N_movable, dtype=np.float32)
        pending_others = np.zeros(4, dtype=np.float32)

        # 按 step_id 填充已选择的字段；仅在元件已选后才填充后续 pending 值
        # （避免 _pending_comp_idx == -1 时错误地将无效坐标写入观测）
        comp_ok = self._pending_comp_idx >= 0

        if step_id >= 1 and comp_ok:
            pending_comp[self._pending_comp_idx] = 1.0
        if step_id >= 2 and comp_ok:
            pending_others[0] = float(self._pending_gx1)
        if step_id >= 3 and comp_ok:
            pending_others[1] = float(self._pending_gy1)
        if step_id >= 4 and comp_ok:
            pending_others[2] = float(self._pending_rot)
        if step_id >= 5 and comp_ok:
            pending_others[3] = float(self._pending_lay)

        return np.concatenate([raw_obs, pending_comp, pending_others])

    def _build_mask_for_step(self, step_id: int) -> np.ndarray:
        """为指定子步骤构建 1D bool 动作掩码。

        Step 0 (comp): 已放置/固定 → mask=False
        Step 1 (x1): 超出右边界 → mask=False
        Step 2 (y1): 超出上边界 → mask=False
        Step 3 (rot): 全 True
        Step 4 (lay): 全 True

        Args:
            step_id: 子步骤编号 0-4

        Returns:
            1D bool 向量，合法动作为 True
        """
        if step_id == 0:
            # 元件选择掩码: 已放置/固定不可选
            mask = np.ones(self.N_movable, dtype=bool)
            for i, comp in enumerate(self.movable_list):
                if comp.is_placed or comp.is_fixed:
                    mask[i] = False
            return mask

        elif step_id == 1:
            # x1 掩码: 需已知选中的元件
            if self._pending_comp_idx < 0:
                return np.zeros(self.grid_width, dtype=bool)
            comp = self.movable_list[self._pending_comp_idx]
            expected_w = max(1, int(comp.footprint_w / Config.CELL_SIZE))
            mask = np.ones(self.grid_width, dtype=bool)
            for gx1 in range(self.grid_width):
                gx2 = gx1 + expected_w
                x1_mm = self.valid_area[0] + gx1 * Config.CELL_SIZE
                if gx2 >= self.grid_width or x1_mm + comp.footprint_w > self.valid_area[2]:
                    mask[gx1] = False
            return mask

        elif step_id == 2:
            # y1 掩码: 需已知选中的元件
            if self._pending_comp_idx < 0:
                return np.zeros(self.grid_height, dtype=bool)
            comp = self.movable_list[self._pending_comp_idx]
            expected_h = max(1, int(comp.footprint_h / Config.CELL_SIZE))
            mask = np.ones(self.grid_height, dtype=bool)
            for gy1 in range(self.grid_height):
                gy2 = gy1 + expected_h
                y1_mm = self.valid_area[1] + gy1 * Config.CELL_SIZE
                if gy2 >= self.grid_height or y1_mm + comp.footprint_h > self.valid_area[3]:
                    mask[gy1] = False
            return mask

        elif step_id == 3:
            # 旋转: 无掩码
            return np.ones(4, dtype=bool)

        elif step_id == 4:
            # 层: 无掩码
            return np.ones(2, dtype=bool)

        else:
            raise ValueError(f"Invalid step_id: {step_id}, must be 0-4")

    def _calculate_reward(
        self,
        component,
        total_hard_overlap: float,
        total_soft_spacing: float,
        total_out_of_bounds: float,
        has_violations: bool,
    ) -> float:
        """计算完整 6 项 reward。

        惩罚项：硬重叠 + 软间距 + 出界。
        奖励项：Net 聚合度 + 拥塞均衡度 + 去耦电容距离 + 大元件排名 + 完成奖励。

        Args:
            component: 当前放置的元件
            total_hard_overlap: 硬重叠累计面积
            total_soft_spacing: 软间距累计惩罚
            total_out_of_bounds: 出界累计面积
            has_violations: 是否存在任何违规

        Returns:
            标量 reward 值
        """
        reward = 0.0
        breakdown = {}

        # ========== 惩罚项 ==========

        # 硬重叠
        penalty_hard = Config.W_OVERLAP_HARD * total_hard_overlap
        reward -= penalty_hard
        breakdown["overlap_hard"] = -penalty_hard

        # 软间距
        penalty_soft = Config.W_OVERLAP_SOFT * total_soft_spacing
        reward -= penalty_soft
        breakdown["overlap_soft"] = -penalty_soft

        # 出界
        penalty_oob = self.W_OUT_OF_BOUNDS_CURRENT * total_out_of_bounds
        reward -= penalty_oob
        breakdown["out_of_bounds"] = -penalty_oob

        # ========== 奖励项 ==========

        # Net 聚合度
        agg_reward = 0.0
        for net_id in component.nets:
            net = self.nets[net_id]
            comp_bboxes = []
            for conn_ref, _ in net.connections:
                if conn_ref in self.kicad.components:
                    c = self.kicad.components[conn_ref]
                    if c.is_placed:
                        comp_bboxes.append(c.bbox)
            if len(comp_bboxes) >= 2:
                agg = PCBUtils.net_aggregation(comp_bboxes)
                agg_score = max(0.0, 1.0 - agg / max(self.board_diagonal, 0.01))
                agg_reward += agg_score * Config.W_NET_AGGREGATION
        reward += agg_reward
        breakdown["net_aggregation"] = agg_reward

        # 拥塞均衡度
        placed_comps = [self.kicad.components[rid] for rid in self.placed_list]
        if has_violations:
            placed_comps.append(component)
        cong_std = PCBUtils.congestion_std(
            placed_comps, self.nets, self.valid_area,
            Config.CONGESTION_CELL_SIZE
        )
        # 归一化：MAX_CONGESTION_STD 设为 sqrt(N_movable) 作为上界参考
        max_std = max(1.0, np.sqrt(self.N_movable))
        cong_score = max(0.0, 1.0 - cong_std / max_std)
        cong_reward = cong_score * Config.W_CONGESTION
        reward += cong_reward
        breakdown["congestion"] = cong_reward

        # 去耦电容距离
        decap_reward = 0.0
        decap_violations = 0
        if component.id in self.decoupling_map:
            for cap_id in self.decoupling_map[component.id]:
                if cap_id in self.kicad.components:
                    cap = self.kicad.components[cap_id]
                    if cap.is_placed:
                        dist = PCBUtils.min_pin_to_cap_distance(component, cap)
                        decap_reward += max(
                            0.0, 1.0 - dist / Config.DECOUPLING_MAX_DIST
                        ) * Config.W_DECOUPLING
                        if dist > Config.DECOUPLING_MAX_DIST:
                            decap_violations += 1
        reward += decap_reward
        breakdown["decoupling"] = decap_reward
        breakdown["decoupling_violations"] = decap_violations

        # 大元件排名
        if component.id in self.area_ranks:
            rank = self.area_ranks[component.id]
            large_comp_reward = max(0.0, 1.0 - rank / self.N_movable) * Config.W_LARGE_COMP
            reward += large_comp_reward
            breakdown["large_comp"] = large_comp_reward

        # 完成奖励
        if not has_violations:
            # 每放一个给一小部分
            completion_partial = Config.W_COMPLETION / max(self.N_movable, 1)
            reward += completion_partial
            breakdown["completion_partial"] = completion_partial
            if len(self.placed_list) + 1 == self.N_movable:
                # 最后一个：额外给剩余部分
                completion_bonus = Config.W_COMPLETION - completion_partial * self.N_movable
                reward += completion_bonus
                breakdown["completion_full"] = completion_bonus

        self._last_reward_breakdown = breakdown
        return reward

    def _invalid_action(self, reason: str = "") -> Tuple:
        """处理非法动作：返回当前观测 + 0 reward + 不终止。

        Args:
            reason: 非法动作的原因描述

        Returns:
            (obs, 0.0, False, truncated, info) 元组
        """
        obs = self._get_obs_with_pending(0)
        self.current_step += 1
        truncated = self.current_step >= Config.MAX_STEPS \
                    or self.cumulative_reward < Config.EARLY_STOP_THRESHOLD
        info = {
            "step": self.current_step,
            "invalid_action": reason,
            "action_mask": self._build_mask_for_step(0),
        }
        return obs, 0.0, False, truncated, info

    def close(self):
        """释放环境资源。

        KiCad board 对象无显式释放需求，预留接口以兼容 Runner 的 finally 块调用。
        """
        pass

    # ==================== 对外工具 ====================

    def get_all_files(self) -> List[str]:
        """扫描 data/ 目录下所有 PCB 文件。

        Returns:
            排序后的 PCB 文件名列表（不含路径前缀）
        """
        import os
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        )
        if not os.path.isdir(data_dir):
            return []
        return sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".kicad_pcb", ".pcb"))
        ])

    @staticmethod
    def get_all_pcb_files(data_dir: str) -> List[str]:
        """扫描指定目录下所有 PCB 文件。

        Args:
            data_dir: 要扫描的目录路径

        Returns:
            排序后的 PCB 文件名列表（不含路径前缀）
        """
        import os
        if not os.path.isdir(data_dir):
            return []
        return sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".kicad_pcb", ".pcb"))
        ])
