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
    W_OVERLAP_HARD: float = 20.0
    W_OVERLAP_SOFT: float = 10.0
    W_OUT_OF_BOUNDS_INITIAL: float = 2.0
    W_OUT_OF_BOUNDS_FINAL: float = 20.0
    W_NET_AGGREGATION: float = 10.0
    W_CONGESTION: float = 5.0
    W_DECOUPLING: float = 10.0
    W_LARGE_COMP: float = 5.0
    W_COMPLETION: float = 200.0
    # 面积归一化惩罚的最小分母 (mm²)：防止极小元件导致重叠/出界比例被放大
    MIN_NORM_AREA: float = 1.0

    # Reward Shaping：品质项时序控制
    QUALITY_START_EP: int = 10000
    QUALITY_RAMP_EP: int = 2000

    # 训练参数
    GAMMA: float = 0.99
    MAX_STEPS: int = 2000
    TOTAL_EPISODES: int = 10000
    SAVE_INTERVAL: int = 500
    LOG_INTERVAL: int = 100
    EARLY_STOP_THRESHOLD: float = -150.0
    EPSILON_START: float = 0.6
    EPSILON_MIN: float = 0.1
    EPSILON_DECAY_EPISODES: int = 10000  # ε 衰减最短收敛区间（下限），总 epi 超过此值时按实际总 epi 收敛
    CURRICULUM_START_EP: int = 0
    CURRICULUM_END_EP: int = 500

    # DQN 训练参数
    BATCH_SIZE: int = 64
    REPLAY_BUFFER_SIZE: int = 10000
    LEARNING_RATE: float = 0.0001
    TARGET_UPDATE_FREQ: int = 20   # 每 N 个 episode 同步一次 target 网络
    WARMUP_EPISODES: int = 500

    # 几何阈值
    BOARD_MARGIN: float = 0.0
    DECOUPLING_MAX_DIST: float = 5.0
    CELL_SIZE: float = 0.2       # 网格离散化单位（0.2mm/格，细化 5 倍以表达真实 0.24~0.48mm 间距）
    CONGESTION_CELL_SIZE: float = 2.0

    # 固定维度上限（支持不同 PCB 使用同一网络）
    # 统一维度 100×100：罩住全 5 块板在 0.2mm 下的最大覆盖率（约 80×80），预留余量
    # 切换到大板时需恢复: MAX_MOVABLE=384, GRID_W=512, GRID_H=512
    MAX_MOVABLE: int = 18        # 最大可移动元件数
    GRID_W: int = 100            # 最大网格宽度（统一维度上限；掩码/网络 head 输出均读此值）
    GRID_H: int = 100            # 最大网格高度（统一维度上限；掩码/网络 head 输出均读此值）

    # 观测归一化开关（默认关闭以兼容已有 checkpoint）
    # 开启后坐标除以 board_diagonal，rotation 除以 4.0，
    # 对齐 global_features 中的归一化尺度。需重新训练。
    NORMALIZE_OBS: bool = False

    # ---- 贪心能力评估（best_model 选择基建，合并自 main） ----
    GREEDY_EVAL_INTERVAL: int = 500   # 每隔多少 episode 跑一次贪心推理评估
    GREEDY_EVAL_K: int = 10           # 每次评估的贪心 episode 数（eps=0.01）

    # 层动作约束开关（任务级）
    # True  = 锁定所有元件到原始层（单面板/复刻任务，堵"翻面钻空子"reward hacking）
    # False = 按 footprint 物理层约束（SMD 锁焊盘面，PTH 可双面；未来生成任务）
    SINGLE_LAYER: bool = True

    # ---- 1c 防死锁前瞻预演（forward checking，默认关，评测/推理时开启） ----
    # 1b 只保证"这一步放下去自己合法"；1c 额外保证"放下去后，剩余未放置元件
    # 每个都还至少有一个合法位置"，把"顺序死锁"拦在发生之前。
    # 默认 False：不改变现有训练/推理基线；需要时由评测脚本 --deadlock-preview 开启。
    ENABLE_DEADLOCK_PREVIEW: bool = False
    # 只在剩余元件 ≤ 此数时启用 1c（死锁集中发生在收尾阶段，前期空区大零开销）
    DEADLOCK_PREVIEW_MAX_REMAINING: int = 3

    # ---- 路线3：MRV 顺序约束（Minimum Remaining Values，默认关） ----
    # 死锁元凶 = "最后才放位置最少的元件"（顺序死锁）。MRV：step0 只放行
    # 当前剩余合法落子最少的未放元件，把最受约束的元件提前处理，从顺序上
    # 消除"最后一个被堵死"。判定口径与 1b/1c 预演一致（按 (位置, 层) 计数）。
    # 注意：MRV 改变训练/推理的 step0 动作分布（自由选元件 → 排他顺序），
    # 必须训练与推理同步开启并重训（推理端突然开启会与训练分布失配）。
    ENABLE_MRV_ORDER: bool = False

    # ---- 专家轨迹注入 ----
    EXPERT_TOTAL_BUFFER: int = 100000       # 专家组 buffer 总容量（专家区+探索区）
    EXPERT_TOTAL_TRANSITIONS: int = 20000   # 目标专家 transition 总数（达到即停）
    EXPERT_MAX_TRAJECTORIES: int = 5000     # 最大轨迹数安全上限（防死循环）
    EXPERT_EPSILON_DECOMPOSE: float = 0.5
    EXPERT_POS_NOISE_PROB: float = 0.2
    EXPERT_ROT_NOISE_PROB: float = 0.1
    EXPERT_BC_EPOCHS: int = 50
    EXPERT_APPLY_PENALTIES: bool = True  # 专家 reward 是否携带惩罚项（True=与 RL 环境一致；False=仅品质奖励，实验 D 对照用）
    EXPERT_BUFFER_RATIO: float = 0.7
    EXPERT_BUFFER_MULTIPLIER: int = 5
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    BC_REG_LAMBDA: float = 10.0   # BC 正则化强度：RL loss += λ·KL(当前策略‖冻结BC策略)，锚定好策略防学坏
    BC_REG_TEMP: float = 1.0      # BC 正则化 softmax 温度（策略蒸馏温度）
    BC_REG_VALUE_LAMBDA: float = 0.5  # BC 值正则化强度：RL loss += λv·MSE(Q当前, Q_BC)，锚定 Q 值幅度防发散
    EXPERT_TRANSITIONS_DIR: str = "data/expert_transitions"

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
                            # 支持 bool（True/False），用于 EXPERT_APPLY_PENALTIES 等开关参数
                            if val_str.lower() == "true":
                                val = True
                            elif val_str.lower() == "false":
                                val = False
                            else:
                                continue
                    # 数值型字段的 "true/false" 字符串不覆盖为 bool（防误写污染数值配置）
                    cur = getattr(cls, key, None)
                    if (isinstance(val, bool)
                            and isinstance(cur, (int, float))
                            and not isinstance(cur, bool)):
                        logger.warning(
                            f"Ignored non-numeric value '{key} = {val_str}' "
                            f"(field is numeric, keep {cur})"
                        )
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
        # 统一软间距阈值 0.2mm（颗粒度 CELL_SIZE=0.2mm 下，覆盖真实元件间距 0.24~0.48mm，
        # 确保完美布局不因阈值过严而恒为负分；分类间距差异在现阶段 5 块板范围内不做区分）
        thresholds = [0.2, 0.2, 0.2, 0.2, 0.2]
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

        import os as _os

        # 加载训练超参（argument.txt），保证 reward 口径与训练一致。
        # config_path 未指定时默认加载项目 config/argument.txt，与 admin.txt
        # 对称，避免直接实例化 env（如命令行生成专家轨迹）时漏加载超参，
        # 导致 W_COMPLETION 等用类默认值（如 200）而非 argument.txt 的 100。
        if config_path is None:
            config_path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "..", "config", "argument.txt",
            )
        Config.load_from_file(config_path)
        # 固定维度参数已并入 argument.txt（MAX_MOVABLE/GRID_W/GRID_H），admin.txt 时代移除。

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
        # 用 round 而非 int 截断：0.2 无法精确用二进制浮点表示，7.2/0.2 会得到
        # 35.9999...，int 截断丢最右一格（grid 从 36 变 35），round 修正此下溢。
        self.grid_width = max(1, int(round((self.valid_area[2] - self.valid_area[0]) / Config.CELL_SIZE)))
        self.grid_height = max(1, int(round((self.valid_area[3] - self.valid_area[1]) / Config.CELL_SIZE)))

        # 元件分类
        self.fixed_list = [c for c in self.kicad.components.values() if c.is_fixed]
        self.movable_list = [c for c in self.kicad.components.values() if not c.is_fixed]
        self.N_movable = len(self.movable_list)

        # 空板/缺陷板（0 可动元件）：不参与训练，抛可捕获异常由 run.py 跳过。
        # 否则下方 spaces.Discrete(self.N_movable) 会因 Discrete(0) 崩溃。
        if self.N_movable < 1:
            raise ValueError(f"Non-circuit board (0 movable components): {pcb_path}")

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

        # 观测空间: Dict 结构（固定维度，不同 PCB 使用同一网络）
        #   component_slots: (MAX_MOVABLE, 8) — 每元件 [placed_flag,x1,y1,x2,y2,rot,layer,net_score]
        #   global_features: (8,)           — 板框 + 放置进度 + pending 编码
        self.observation_space = spaces.Dict({
            "component_slots": spaces.Box(
                low=-np.inf, high=np.inf, shape=(Config.MAX_MOVABLE, 8), dtype=np.float32
            ),
            "global_features": spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            ),
        })

        # 训练动态参数
        self.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL
        self.current_episode: int = 0  # 由 run.py 每 episode 设置

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

        # 4.5 层物理合法性校验（SINGLE_LAYER / footprint 推导）
        if lay_idx not in self._allowed_layers_of(component):
            return self._invalid_action(
                f"layer {lay_idx} not allowed for {component.id}"
            )

        # 5. 更新元件状态
        component.bbox = (x1, y1, x2, y2)
        component.rotation = rot_idx
        component.layer = lay_idx

        # 6. 约束检测（复用 _detect_collisions，与专家轨迹生成共享同一套碰撞标准）
        total_hard_overlap, total_soft_spacing, total_out_of_bounds, in_keepout = \
            self._detect_collisions(component)

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
        """构建 fixed-slot 观测矩阵 (MAX_MOVABLE, 8)。

        每个 slot 8 维：placed_flag, x1, y1, x2, y2, rotation, layer, net_count。
        slot 索引 = movable_list 中的位置，padding slot 全 0。

        Returns:
            shape=(MAX_MOVABLE, 8) 的 float32 矩阵
        """
        obs = np.zeros((Config.MAX_MOVABLE, 8), dtype=np.float32)

        n_slots = min(self.N_movable, Config.MAX_MOVABLE)
        if self.N_movable > Config.MAX_MOVABLE:
            logger.warning(
                f"N_movable={self.N_movable} > MAX_MOVABLE={Config.MAX_MOVABLE}, "
                f"超出部分元件将被忽略"
            )
        for slot_idx, comp in enumerate(self.movable_list[:n_slots]):
            if not comp.is_placed:
                continue
            obs[slot_idx, 0] = 1.0                        # placed_flag
            if Config.NORMALIZE_OBS:
                # 归一化坐标：提升跨尺度泛化能力，对齐其他归一化特征
                diag = max(self.board_diagonal, 0.01)
                obs[slot_idx, 1] = comp.bbox[0] / diag        # x1
                obs[slot_idx, 2] = comp.bbox[1] / diag        # y1
                obs[slot_idx, 3] = comp.bbox[2] / diag        # x2
                obs[slot_idx, 4] = comp.bbox[3] / diag        # y2
                obs[slot_idx, 5] = float(comp.rotation) / 4.0 # rotation
            else:
                # 旧行为：原始 mm 值（与已有 checkpoint 兼容）
                obs[slot_idx, 1] = comp.bbox[0]               # x1
                obs[slot_idx, 2] = comp.bbox[1]               # y1
                obs[slot_idx, 3] = comp.bbox[2]               # x2
                obs[slot_idx, 4] = comp.bbox[3]               # y2
                obs[slot_idx, 5] = float(comp.rotation)       # rotation
            obs[slot_idx, 6] = float(comp.layer)          # layer
            obs[slot_idx, 7] = float(len(comp.nets))      # net_count

        return obs

    def _compute_global_features(self) -> np.ndarray:
        """计算 8 维全局特征。

        Returns:
            shape=(8,) 的 float32 向量
            [0] board_w_norm  = grid_width / GRID_W
            [1] board_h_norm  = grid_height / GRID_H
            [2] placement_ratio = len(placed_list) / N_movable
            [3] pending_comp_norm = _pending_comp_idx / MAX_MOVABLE  (-1 表示未选)
            [4] pending_gx1_norm  = _pending_gx1 / max(grid_width, 1)  (-1 表示未选)
            [5] pending_gy1_norm  = _pending_gy1 / max(grid_height, 1)  (-1 表示未选)
            [6] pending_rot_norm  = _pending_rot / 4.0
            [7] pending_lay       = float(_pending_lay)
        """
        feats = np.zeros(8, dtype=np.float32)
        feats[0] = self.grid_width / max(Config.GRID_W, 1)
        feats[1] = self.grid_height / max(Config.GRID_H, 1)
        feats[2] = len(self.placed_list) / max(self.N_movable, 1)
        feats[3] = self._pending_comp_idx / max(Config.MAX_MOVABLE, 1) \
            if self._pending_comp_idx >= 0 else -1.0
        feats[4] = self._pending_gx1 / max(self.grid_width, 1) \
            if self._pending_gx1 >= 0 else -1.0
        feats[5] = self._pending_gy1 / max(self.grid_height, 1) \
            if self._pending_gy1 >= 0 else -1.0
        feats[6] = self._pending_rot / 4.0
        feats[7] = float(self._pending_lay)
        return feats

    # ---- 自回归 5 步接口 ----

    def _reset_pending(self):
        """重置 5 步自回归的中间 pending 状态。"""
        self._pending_comp_idx = -1
        self._pending_gx1 = -1
        self._pending_gy1 = -1
        self._pending_rot = 0
        self._pending_lay = 0

    def _update_pending(self, action: int, step_id: int):
        """将当前子步骤的动作写入 pending 状态。"""
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
        """将 pending 状态组装为完整的 7 元动作。"""
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

    def _get_obs_with_pending(self, step_id: int) -> dict:
        """构建带 pending 状态的 Dict 观测。

        Returns:
            Dict: {"component_slots": (MAX_MOVABLE, 8), "global_features": (8,)}
        """
        return {
            "component_slots": self._get_obs(),
            "global_features": self._compute_global_features(),
        }

    def _build_mask_for_step(self, step_id: int) -> Optional[np.ndarray]:
        """为指定子步骤构建固定维度 1D bool 动作掩码。

        Step 0 (comp): index >= N_movable 或已放置/固定 → mask=False
        Step 1 (x1):   grid_x >= grid_width → mask=False（越界坐标硬屏蔽）
        Step 2 (y1):   grid_y >= grid_height → mask=False（越界坐标硬屏蔽）
        Step 3~4:      无掩码，所有候选全合法

        Args:
            step_id: 子步骤编号 0-4

        Returns:
            step 0: 固定长度 (MAX_MOVABLE,) bool 向量
            step 1: (GRID_W,) bool 向量
            step 2: (GRID_H,) bool 向量
            step 3~4: None
        """
        if step_id == 0:
            mask = np.zeros(Config.MAX_MOVABLE, dtype=bool)
            n_valid = min(self.N_movable, Config.MAX_MOVABLE)
            for i, comp in enumerate(self.movable_list[:n_valid]):
                if not comp.is_placed and not comp.is_fixed:
                    mask[i] = True
            # 路线3：MRV 顺序约束（默认关，训练与推理需同步开启并重训）
            # step0 只放行"当前剩余合法落子最少"的未放元件（tie 全放行），
            # 把最受约束的元件提前处理，消除"最后才放最挤元件"的顺序死锁。
            if (getattr(Config, "ENABLE_MRV_ORDER", False) and mask.any()):
                mrv_mask = self._mrv_step0_mask()
                if mrv_mask is not None and mrv_mask.any():
                    return mrv_mask
            return mask

        if step_id == 1:
            # 1a: x2 出界硬屏蔽（gx1 + w_cell 越过板框右缘的坐标直接排除）
            base = (np.arange(Config.GRID_W) < self.grid_width) & (
                np.arange(Config.GRID_W) + self._pending_w_cell() <= self.grid_width
            )
            # 1b: 碰撞预演——排除"两层都无可行 y"的 x 候选（必败坐标）
            if self._pending_comp_idx >= 0 and base.any():
                comp = self.movable_list[self._pending_comp_idx]
                w_cell = self._pending_w_cell()
                h_cell = self._pending_h_cell()
                mask = base.copy()
                for gx1 in np.nonzero(mask)[0]:
                    if not self._preview_has_any_layer(comp, int(gx1), w_cell, h_cell):
                        mask[gx1] = False
                # 防御：全被挡 → 回退到只出界 mask（宁可失败一次，不可无动作可选）
                if mask.any():
                    return mask
            return base

        if step_id == 2:
            # 1a: y2 出界硬屏蔽（gy1 + h_cell 越过板框下缘的坐标直接排除）
            base = (np.arange(Config.GRID_H) < self.grid_height) & (
                np.arange(Config.GRID_H) + self._pending_h_cell() <= self.grid_height
            )
            # 1b: 碰撞预演——排除两层都与 fixed/已放置同层硬重叠的 y 候选
            if self._pending_comp_idx >= 0 and self._pending_gx1 >= 0 and base.any():
                comp = self.movable_list[self._pending_comp_idx]
                w_cell = self._pending_w_cell()
                h_cell = self._pending_h_cell()
                mask = base.copy()
                for gy1 in np.nonzero(mask)[0]:
                    if not self._preview_position_any_layer(
                        comp, self._pending_gx1, int(gy1), w_cell, h_cell
                    ):
                        mask[gy1] = False
                # 1c: 防死锁前瞻——1b 只保证"这一步自己合法"，这里再加
                # "放下去后剩余元件仍可行"检查（forward checking）。死锁集中
                # 在收尾阶段（剩余 ≤ DEADLOCK_PREVIEW_MAX_REMAINING）才启用，
                # 前期空区大、无死锁风险，避免无谓扫描。
                if (getattr(Config, "ENABLE_DEADLOCK_PREVIEW", False)
                        and mask.any()):
                    remaining = sum(1 for c in self.movable_list
                                    if not c.is_placed)
                    if remaining <= getattr(Config,
                                            "DEADLOCK_PREVIEW_MAX_REMAINING", 3):
                        area = self.valid_area
                        spots = self._collect_other_free_spots(
                            self._pending_comp_idx)
                        if spots:
                            dl_mask = mask.copy()
                            for gy1 in np.nonzero(mask)[0]:
                                px1 = area[0] + self._pending_gx1 * Config.CELL_SIZE
                                py1 = area[1] + int(gy1) * Config.CELL_SIZE
                                px2 = px1 + w_cell * Config.CELL_SIZE
                                py2 = py1 + h_cell * Config.CELL_SIZE
                                if not self._preview_place_keeps_others_feasible(
                                        self._pending_comp_idx,
                                        px1, py1, px2, py2, spots):
                                    dl_mask[gy1] = False
                            # 防御：1c 把合法位置全挡掉（当前元件只能放在
                            # 会堵死别人的位置）→ 回退 1b mask，宁放行一次
                            # 也不无动作可选（与 1b 的防御回退同哲学）。
                            if dl_mask.any():
                                mask = dl_mask
                if mask.any():
                    return mask
            return base

        if step_id == 4:
            # 1b 补全：层预演——step1/2 保证位置至少一层合法，此处把 mask
            # 缩到"当前位置实际合法的层"，避免选到必败层（探索期 lay 随机是
            # 残留失败的唯一来源）。
            if (self._pending_comp_idx >= 0 and self._pending_gx1 >= 0
                    and self._pending_gy1 >= 0):
                comp = self.movable_list[self._pending_comp_idx]
                legal = self._preview_layer_legal(
                    comp, self._pending_gx1, self._pending_gy1,
                    self._pending_w_cell(), self._pending_h_cell(),
                )
                if any(legal):
                    return np.array(legal, dtype=bool)
            return None

        return None

    # ---- 层动作约束（任务级 SINGLE_LAYER + footprint 物理层） ----

    def _allowed_layers_of(self, comp) -> Tuple[int, ...]:
        """元件的可行动作层集。

        SINGLE_LAYER=True：锁定到原始层（探索/复刻任务，堵 reward hacking）；
        SINGLE_LAYER=False：按 footprint 推导的物理允许层（未来生成任务）。

        未来「从需求出板」迁移点：SINGLE_LAYER 读需求输入，allowed_layers
        读封装库 footprint 的 pad 层，本函数接口不变。
        """
        if Config.SINGLE_LAYER:
            return (comp.layer,)
        return tuple(comp.allowed_layers)

    # ---- 碰撞预演辅助（step1/2/4 的 mask 过滤用，与 _detect_collisions 同口径） ----

    def _pending_w_cell(self) -> int:
        """当前 pending 元件的网格宽度（与 _assemble_action 同口径）。"""
        if self._pending_comp_idx < 0:
            return 1
        comp = self.movable_list[self._pending_comp_idx]
        return max(1, int(comp.footprint_w / Config.CELL_SIZE))

    def _pending_h_cell(self) -> int:
        """当前 pending 元件的网格高度（与 _assemble_action 同口径）。"""
        if self._pending_comp_idx < 0:
            return 1
        comp = self.movable_list[self._pending_comp_idx]
        return max(1, int(comp.footprint_h / Config.CELL_SIZE))

    def _preview_others(self, comp, lay: int) -> List:
        """预演碰撞对象：与指定层相同的 fixed + 已放置元件。"""
        others = list(self.fixed_list)
        others += [self.kicad.components[rid] for rid in self.placed_list]
        return [o for o in others if o.layer == lay and o.id != comp.id]

    def _preview_blocks_any(self, comp, blockers, x1, y1, x2, y2) -> bool:
        """预演：bbox 是否与任一 blocker 硬重叠（口径与 _detect_collisions 一致）。"""
        for o in blockers:
            threshold = Config.spacing_threshold_for(
                comp.package_type, o.package_type
            )
            hard, _ = PCBUtils.bbox_overlap_penalty(
                (x1, y1, x2, y2), o.bbox, threshold
            )
            if hard > 0.0:
                return True
        return False

    def _preview_position_any_layer(self, comp, gx1, gy1, w_cell, h_cell) -> bool:
        """预演：网格位置 (gx1,gy1) 在 Top/Bot 至少一层无碰撞且不越界。

        层是第 5 步子动作才决定，故对可行动作集逐层验证：最终所选层若在
        该位置合法即放行（零误杀）；全部可行动作集全非法才判定为必败位置。
        """
        area = self.valid_area
        x1 = area[0] + gx1 * Config.CELL_SIZE
        y1 = area[1] + gy1 * Config.CELL_SIZE
        x2 = x1 + w_cell * Config.CELL_SIZE
        y2 = y1 + h_cell * Config.CELL_SIZE
        # 出界/keepout 判定与 env.step 完全同口径（mm 坐标，避免网格 round 误差）
        if PCBUtils.out_of_bounds_area((x1, y1, x2, y2), area) > 0.0:
            return False
        if PCBUtils.check_keepout((x1, y1, x2, y2), self.keepout_zones):
            return False
        for lay in self._allowed_layers_of(comp):
            blockers = [o for o in self._preview_others(comp, lay)
                        if o.bbox[2] > x1 and o.bbox[0] < x2
                        and o.bbox[3] > y1 and o.bbox[1] < y2]
            if not self._preview_blocks_any(comp, blockers, x1, y1, x2, y2):
                return True
        return False

    def _preview_layer_legal(self, comp, gx1, gy1, w_cell, h_cell) -> List[bool]:
        """预演：网格位置 (gx1,gy1) 在可行动作层各层是否无碰撞（step4 层 mask）。

        固定返回长度 2（对应 Discrete(2) 层动作）：非物理允许层恒 False，
        由 _allowed_layers_of（SINGLE_LAYER / footprint 推导）过滤。
        """
        area = self.valid_area
        x1 = area[0] + gx1 * Config.CELL_SIZE
        y1 = area[1] + gy1 * Config.CELL_SIZE
        x2 = x1 + w_cell * Config.CELL_SIZE
        y2 = y1 + h_cell * Config.CELL_SIZE
        if PCBUtils.out_of_bounds_area((x1, y1, x2, y2), area) > 0.0:
            return [False, False]
        if PCBUtils.check_keepout((x1, y1, x2, y2), self.keepout_zones):
            return [False, False]
        legal = [False, False]
        for lay in self._allowed_layers_of(comp):
            blockers = [o for o in self._preview_others(comp, lay)
                        if o.bbox[2] > x1 and o.bbox[0] < x2
                        and o.bbox[3] > y1 and o.bbox[1] < y2]
            legal[lay] = not self._preview_blocks_any(comp, blockers, x1, y1, x2, y2)
        return legal

    def _preview_has_any_layer(self, comp, gx1, w_cell, h_cell) -> bool:
        """预演：x1=gx1 时，可行动作层至少一层存在无碰撞且不越界的 y1。"""
        area = self.valid_area
        x1 = area[0] + gx1 * Config.CELL_SIZE
        x2 = x1 + w_cell * Config.CELL_SIZE
        for lay in self._allowed_layers_of(comp):
            blockers = [o for o in self._preview_others(comp, lay)
                        if o.bbox[2] > x1 and o.bbox[0] < x2]
            for gy1 in range(self.grid_height):
                y1 = area[1] + gy1 * Config.CELL_SIZE
                y2 = y1 + h_cell * Config.CELL_SIZE
                if PCBUtils.out_of_bounds_area((x1, y1, x2, y2), area) > 0.0:
                    continue
                if PCBUtils.check_keepout((x1, y1, x2, y2), self.keepout_zones):
                    continue
                if not self._preview_blocks_any(comp, blockers, x1, y1, x2, y2):
                    return True
        return False

    # ---- 1c 防死锁前瞻预演（forward checking） ----
    # 与 1b 同口径的"位置合法"判定，但多查一层：pending 元件占据某位置后，
    # 其余每个未放置元件是否仍至少有一个合法落点。若某个元件被彻底堵死，
    # 该候选就是"必死位置"（放下去必然死锁）→ 从 mask 剔除。

    def _collect_other_free_spots(self, exclude_comp_idx: int):
        """收集"其它未放置元件"各自的全部合法落子（当前占用下）。

        Returns:
            {comp_id: [(x1, y1, x2, y2, lay), ...]} —— 仅含未放置且非
            exclude_comp 的元件。空 dict 表示没有需要守护的元件。
        """
        area = self.valid_area
        cell = Config.CELL_SIZE
        spots = {}
        for o in self.movable_list:
            if o.is_placed or o.id == exclude_comp_idx:
                continue
            ow = max(1, int(o.footprint_w / cell))
            oh = max(1, int(o.footprint_h / cell))
            o_list = []
            for lay in self._allowed_layers_of(o):
                blockers = self._preview_others(o, lay)
                for gx1 in range(self.grid_width):
                    x1 = area[0] + gx1 * cell
                    x2 = x1 + ow * cell
                    if x2 > area[2]:          # x 越界：gx1 递增后只会更越界
                        break
                    for gy1 in range(self.grid_height):
                        y1 = area[1] + gy1 * cell
                        y2 = y1 + oh * cell
                        if y2 > area[3]:
                            break
                        if PCBUtils.check_keepout((x1, y1, x2, y2),
                                                  self.keepout_zones):
                            continue
                        if self._preview_blocks_any(o, blockers,
                                                    x1, y1, x2, y2):
                            continue
                        o_list.append((x1, y1, x2, y2, lay))
            spots[o.id] = o_list
        return spots

    def _preview_place_keeps_others_feasible(self, comp_idx: int,
                                             x1: float, y1: float,
                                             x2: float, y2: float,
                                             spots) -> bool:
        """1c 判定：comp 占据 (x1,y1)-(x2,y2) 后，其余未放置元件仍可行？

        层是第 5 步子动作才决定：对 comp 的可行动作层逐层尝试，只要存在
        一层选择使放后所有其它元件都还有落点 → 放行（零误杀，与 1b 同哲学）。
        """
        comp = self.movable_list[comp_idx]
        others = [(cid, self.kicad.components[cid]) for cid, lst in spots.items()]
        for lay_c in self._allowed_layers_of(comp):
            all_ok = True
            for cid, o in others:
                ok = False
                th = Config.spacing_threshold_for(comp.package_type,
                                                  o.package_type)
                for q in spots[cid]:
                    qx1, qy1, qx2, qy2, qlay = q
                    if (qlay == lay_c and qx1 < x2 and qx2 > x1
                            and qy1 < y2 and qy2 > y1):
                        hard, _ = PCBUtils.bbox_overlap_penalty(
                            (qx1, qy1, qx2, qy2), (x1, y1, x2, y2), th)
                        if hard > 0.0:
                            continue   # 该落点被 P 占掉，试 o 的下一个落点
                    ok = True
                    break
                if not ok:
                    all_ok = False     # 某元件被 P 彻底堵死 → 这层选择不行
                    break
            if all_ok:
                return True
        return False

    # ---- 路线3：MRV 顺序约束（Minimum Remaining Values，默认关） ----
    # 死锁元凶 = "最后才放位置最少的元件"。MRV：step0 优先放当前剩余合法
    # 落子最少的未放元件，把最危险的元件提前处理。判定口径与 1b/1c 预演
    # 完全一致（mm bbox + 越界 + keepout + bbox_overlap_penalty），按
    # (位置, 层) 对计数，保证"合法"定义与 env.step 同口径。

    def _count_free_spot_all(self, comp) -> int:
        """统计 comp 在当前局面的全部合法落子数（(网格位置, 层) 对计数）。

        Args:
            comp: movable_list 中的元件（未放置）

        Returns:
            合法 (x1, y1, x2, y2, layer) 落子总数；0 = 已被彻底堵死。
        """
        area = self.valid_area
        cell = Config.CELL_SIZE
        ow = max(1, int(comp.footprint_w / cell))
        oh = max(1, int(comp.footprint_h / cell))
        count = 0
        for lay in self._allowed_layers_of(comp):
            blockers = self._preview_others(comp, lay)
            for gx1 in range(self.grid_width):
                x1 = area[0] + gx1 * cell
                x2 = x1 + ow * cell
                if x2 > area[2]:          # x 越界：gx1 递增后只会更越界
                    break
                for gy1 in range(self.grid_height):
                    y1 = area[1] + gy1 * cell
                    y2 = y1 + oh * cell
                    if y2 > area[3]:
                        break
                    if PCBUtils.check_keepout((x1, y1, x2, y2),
                                              self.keepout_zones):
                        continue
                    if self._preview_blocks_any(comp, blockers,
                                                x1, y1, x2, y2):
                        continue
                    count += 1
        return count

    def _mrv_step0_mask(self) -> Optional[np.ndarray]:
        """MRV step0 排他掩码：只放行"剩余合法落子最少"的未放元件。

        Returns:
            固定长度 (MAX_MOVABLE,) bool 向量；返回 None 表示回退 base mask：
              - 剩余候选 ≤ 1：无选择余地，MRV 无意义；
              - 存在合法落子 = 0 的元件：死锁已成定局（该元件以后只会更挤，
                其它元件放完也救不回）→ 回退 base，宁失败一次不可无动作可选
                （与 1b/1c 防御回退同哲学）。
        """
        n_valid = min(self.N_movable, Config.MAX_MOVABLE)
        candidates = [i for i in range(n_valid)
                      if not self.movable_list[i].is_placed
                      and not self.movable_list[i].is_fixed]
        if len(candidates) <= 1:
            return None
        counts = {i: self._count_free_spot_all(self.movable_list[i])
                  for i in candidates}
        min_c = min(counts.values())
        if min_c <= 0:
            return None
        mask = np.zeros(Config.MAX_MOVABLE, dtype=bool)
        for i in candidates:
            if counts[i] == min_c:
                mask[i] = True
        return mask

    def _get_quality_scale(self) -> float:
        """品质项权重缩放系数，基于 current_episode 线性升温。

        Episode < QUALITY_START_EP                 → 0.0（纯精简版 reward）
        Episode in [START, START+RAMP]              → 0.0→1.0 线性
        Episode >= QUALITY_START_EP + QUALITY_RAMP_EP → 1.0（完整版 reward）

        Returns:
            quality_scale ∈ [0.0, 1.0]
        """
        start = Config.QUALITY_START_EP
        ramp = max(Config.QUALITY_RAMP_EP, 1)
        # getattr 兼容 mock env：mock 绕过 __init__ 不会设置 current_episode，回退到 0
        ep = getattr(self, "current_episode", 0)

        if ep < start:
            return 0.0
        if ep >= start + ramp:
            return 1.0
        return (ep - start) / ramp

    def _detect_collisions(self, component) -> Tuple[float, float, float, bool]:
        """检测当前元件与已放置/固定元件的碰撞、出界与禁止区情况。

        供 env.step() 与专家轨迹生成共享，保证两处使用完全一致的
        碰撞判定标准（硬重叠、软间距、出界、keepout）。

        Args:
            component: 当前放置的元件（bbox/rotation/layer 已更新为目标位置，
                is_placed 尚未置 True，不在 placed_list 中）

        Returns:
            (total_hard_overlap, total_soft_spacing, total_out_of_bounds, in_keepout)
        """
        total_hard_overlap = 0.0
        total_soft_spacing = 0.0

        # C2: 与 fixed_list + placed_list 中元件的碰撞检测
        all_placed = [self.kicad.components[rid] for rid in self.placed_list]
        all_fixed_comps = self.fixed_list + all_placed

        for other in all_fixed_comps:
            if other.id == component.id:
                continue
            # 层感知修复：不同层（Top/Bottom）元件处于不同物理面，不产生
            # 硬重叠/软间距冲突。双面 PCB 中 U1(Top) 与 CN3(Bottom) 可叠放。
            if other.layer != component.layer:
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

        return total_hard_overlap, total_soft_spacing, total_out_of_bounds, in_keepout

    def _calculate_reward(
        self,
        component,
        total_hard_overlap: float,
        total_soft_spacing: float,
        total_out_of_bounds: float,
        has_violations: bool,
        quality_scale: Optional[float] = None,
        apply_penalties: bool = True,
    ) -> float:
        """计算 reward。

        惩罚项（apply_penalties=True 时生效）：硬重叠 + 软间距 + 出界。
        基础奖励（始终生效）：完成奖励。
        品质奖励（quality_scale 升温控制）：Net 聚合度 + 拥塞均衡度 + 去耦电容 + 大元件排名。

        Args:
            component: 当前放置的元件
            total_hard_overlap: 硬重叠累计面积
            total_soft_spacing: 软间距累计惩罚
            total_out_of_bounds: 出界累计面积
            has_violations: 是否存在任何违规
            quality_scale: 品质项权重缩放系数；None 时按 current_episode 升温
                （训练路径），否则用传入值（专家轨迹传 1.0 表示理想布局满分）
            apply_penalties: 是否应用硬重叠/软间距/出界惩罚项。训练路径始终
                True；专家轨迹生成时由 EXPERT_APPLY_PENALTIES 控制（False=
                仅品质奖励，用于对照实验 D 验证 reward 尺度错配假说）。

        Returns:
            标量 reward 值
        """
        if quality_scale is None:
            quality_scale = self._get_quality_scale()
        reward = 0.0
        breakdown = {}

        # ========== 惩罚项（apply_penalties=True 时生效） ==========
        # 归一化修复：面积类惩罚（硬重叠、出界）原本直接乘面积(mm²)，量级可达
        # -1000+（20 × 40mm²），而单步成功奖励上限仅 ~W_COMPLETION=100，
        # 惩罚比奖励大一个数量级，导致极端负值主导梯度、训练发散不收敛。
        # 现将面积惩罚归一化为"比例惩罚"（重叠/出界面积 ÷ 元件自身面积 ∈[0,1]），
        # 单步惩罚上限 = 权重本身（如 -20），与奖励同量级，梯度稳定。

        # 元件自身面积（归一化分母）；bbox 总有正面积，安全
        comp_area = max(0.0, (component.bbox[2] - component.bbox[0])) * max(
            0.0, (component.bbox[3] - component.bbox[1])
        )
        norm_denom = max(comp_area, Config.MIN_NORM_AREA)

        # 硬重叠：重叠面积占比
        overlap_ratio = total_hard_overlap / norm_denom
        penalty_hard = Config.W_OVERLAP_HARD * min(overlap_ratio, 1.0)
        if apply_penalties:
            reward -= penalty_hard
        breakdown["overlap_hard"] = -penalty_hard

        # 软间距：连续惩罚，量级本就较小，保留但以阈值为界封顶，避免极端
        penalty_soft = Config.W_OVERLAP_SOFT * total_soft_spacing
        penalty_soft = min(penalty_soft, Config.W_OVERLAP_SOFT)
        if apply_penalties:
            reward -= penalty_soft
        breakdown["overlap_soft"] = -penalty_soft

        # 出界：出界面积占比
        oob_ratio = total_out_of_bounds / norm_denom
        penalty_oob = self.W_OUT_OF_BOUNDS_CURRENT * min(oob_ratio, 1.0)
        if apply_penalties:
            reward -= penalty_oob
        breakdown["out_of_bounds"] = -penalty_oob

        # ========== 品质奖励（quality_scale 升温控制） ==========
        # 品质项仅在合法放置（无违规）时发放：失败 step 只扣 hard/soft/oob 惩罚，
        # 不给任何品质奖励。否则 agent 会通过反复"轻微失败"刷品质项（拥塞/大元件
        # 等固定奖励）导致 episode reward 虚高（实测 baseline 刷到 250~1000+，
        # 远超完美布局理论上限 ~158）。

        if not has_violations:
            # Net 聚合度
            agg_reward = 0.0
            for net_id in component.nets:
                net = self.nets[net_id]
                comp_bboxes = []
                # net.connections 不去重（一个 footprint 多 pad 连同一 net 有多个条目），
                # 需按 ref 去重，避免同一 footprint 的 bbox 被重复计入导致聚合度失真
                seen_refs = set()
                for conn_ref, _ in net.connections:
                    if conn_ref in self.kicad.components and conn_ref not in seen_refs:
                        seen_refs.add(conn_ref)
                        c = self.kicad.components[conn_ref]
                        if c.is_placed:
                            comp_bboxes.append(c.bbox)
                # 当前元件刚放置，bbox 已更新但 is_placed 尚未置 True（step() 在 reward 计算后更新），
                # 需显式计入，否则每个 net 少算一个元件，2 元件 net 永远拿不到聚合奖励
                comp_bboxes.append(component.bbox)
                if len(comp_bboxes) >= 2:
                    agg = PCBUtils.net_aggregation(comp_bboxes)
                    agg_score = max(0.0, 1.0 - agg / max(self.board_diagonal, 0.01))
                    agg_reward += agg_score * Config.W_NET_AGGREGATION
            reward += agg_reward * quality_scale
            breakdown["net_aggregation"] = agg_reward * quality_scale

            # 拥塞均衡度：统计已放置元件 + 当前元件的 net 分布均匀度
            placed_comps = [self.kicad.components[rid] for rid in self.placed_list]
            placed_comps.append(component)
            cong_std = PCBUtils.congestion_std(
                placed_comps, self.nets, self.valid_area,
                Config.CONGESTION_CELL_SIZE
            )
            max_std = max(1.0, np.sqrt(self.N_movable))
            cong_score = max(0.0, 1.0 - cong_std / max_std)
            cong_reward = cong_score * Config.W_CONGESTION
            reward += cong_reward * quality_scale
            breakdown["congestion"] = cong_reward * quality_scale

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
            reward += decap_reward * quality_scale
            breakdown["decoupling"] = decap_reward * quality_scale
            breakdown["decoupling_violations"] = decap_violations

            # 大元件排名：面积因子 × 早放因子，鼓励"大元件越早放越好"
            large_comp_reward = 0.0
            if component.id in self.area_ranks:
                rank = self.area_ranks[component.id]
                size_factor = max(0.0, 1.0 - rank / self.N_movable)
                # placement_step = len(placed_list)：当前元件是第几个被放置的（0-based）
                order_factor = max(0.0, 1.0 - len(self.placed_list) / self.N_movable)
                large_comp_reward = size_factor * order_factor * Config.W_LARGE_COMP
            reward += large_comp_reward * quality_scale
            breakdown["large_comp"] = large_comp_reward * quality_scale
        else:
            # 失败 step：品质项全部归零，仅记录到 breakdown 便于诊断
            breakdown["net_aggregation"] = 0.0
            breakdown["congestion"] = 0.0
            breakdown["decoupling"] = 0.0
            breakdown["decoupling_violations"] = 0
            breakdown["large_comp"] = 0.0

        # ========== 完成奖励（始终生效） ==========

        if not has_violations:
            completion_partial = Config.W_COMPLETION / max(self.N_movable, 1)
            reward += completion_partial
            breakdown["completion_partial"] = completion_partial
            if len(self.placed_list) + 1 == self.N_movable:
                completion_bonus = Config.W_COMPLETION - completion_partial * self.N_movable
                reward += completion_bonus
                breakdown["completion_full"] = completion_bonus

        breakdown["quality_scale"] = quality_scale
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
        """递归扫描 data/ 目录下所有 PCB 文件。

        Returns:
            排序后的 PCB 文件相对路径列表（相对于 data/ 根目录）
        """
        import os
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        )
        if not os.path.isdir(data_dir):
            return []
        files = []
        for root, _dirs, fnames in os.walk(data_dir):
            for f in fnames:
                if f.lower().endswith((".kicad_pcb", ".pcb")):
                    files.append(os.path.relpath(os.path.join(root, f), data_dir))
        return sorted(files)

    @staticmethod
    def get_all_pcb_files(data_dir: str) -> List[str]:
        """递归扫描指定目录下所有 PCB 文件。

        Args:
            data_dir: 要扫描的目录路径

        Returns:
            排序后的 PCB 文件相对路径列表（相对于 data_dir）
        """
        import os
        if not os.path.isdir(data_dir):
            return []
        files = []
        for root, _dirs, fnames in os.walk(data_dir):
            for f in fnames:
                if f.lower().endswith((".kicad_pcb", ".pcb")):
                    files.append(os.path.relpath(os.path.join(root, f), data_dir))
        return sorted(files)
