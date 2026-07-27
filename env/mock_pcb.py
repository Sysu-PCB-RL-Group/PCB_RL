"""合成 PCB 数据与 Mock 工具（无需 KiCad）。

将 tests/test_dqn_pipeline.py 中的 MockComponent/MockNet/MockKiCadUtils
和 build_synthetic_pcb()/create_patched_env() 抽取为独立模块，
供 tests 和 scripts 等复用。
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from unittest import mock

import numpy as np

logger = logging.getLogger(__name__)

# ==================== 模型层 — Mock 数据结构 ====================

@dataclass
class MockComponent:
    """最小化 Component mock，属性集与 kicad_utils.Component 兼容。"""

    id: str
    footprint: str = ""
    package_type: int = 4
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    rotation: int = 0
    layer: int = 0
    is_fixed: bool = False
    is_placed: bool = False
    nets: List[str] = field(default_factory=list)
    pins: List[Tuple[str, float, float]] = field(default_factory=list)
    # 默认值对齐 kicad_utils.Component（0.0），避免遗漏字段时产生假数据
    footprint_w: float = 0.0
    footprint_h: float = 0.0

    @property
    def footprint_area(self) -> float:
        return self.footprint_w * self.footprint_h

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )


@dataclass
class MockNet:
    """最小化 Net mock。"""

    net_id: str
    connections: List[Tuple[str, str]] = field(default_factory=list)
    is_power: bool = False


class MockKiCadUtils:
    """替换真实的 KiCadUtils，提供合成 PCB 数据。"""

    def __init__(self):
        self.board = None
        self.components: Dict[str, MockComponent] = {}
        self.nets: Dict[str, MockNet] = {}
        self.board_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0, 0), (0, 0)
        )
        self._initial_state: Optional[Dict] = None  # 供 load_board() 恢复

    def save_initial_state(self):
        """保存当前所有元件的状态快照，供 load_board() 在 reset 时恢复。

        在工厂函数填充完所有元件后调用一次即可。
        bbox/rotation/layer/is_placed 均为不可变值（tuple/int/bool），
        浅拷贝安全；若未来 bbox 改为 list 需改用深拷贝。
        """
        self._initial_state = {
            comp_id: {
                "bbox": comp.bbox,
                "rotation": comp.rotation,
                "layer": comp.layer,
                "is_placed": comp.is_placed,
            }
            for comp_id, comp in self.components.items()
        }

    def load_board(self, path: str) -> bool:
        """加载/重置 PCB 数据。对于 mock，从保存的初始状态恢复所有元件。

        若未调用过 save_initial_state()，则不重置（向后兼容 old mock 用法）。
        """
        if self._initial_state is not None:
            for comp_id, state in self._initial_state.items():
                if comp_id in self.components:
                    comp = self.components[comp_id]
                    comp.bbox = state["bbox"]
                    comp.rotation = state["rotation"]
                    comp.layer = state["layer"]
                    comp.is_placed = state["is_placed"]
        return True

    def get_board_bounds(self):
        return self.board_bounds

    def get_keepout_zones(self):
        return []


# ==================== 拦截 utils.kicad_utils 导入 ====================

def _install_mock_kicad():
    """注入 MockKiCadUtils 到 sys.modules，防 DLL 报错。

    在 import env.pcb_layout_env 之前调用即可。
    幂等：已安装则跳过。
    """
    if "utils.kicad_utils" in sys.modules:
        return
    _mock_module = mock.MagicMock()
    _mock_module.KiCadUtils = MockKiCadUtils
    _mock_module.Component = MockComponent
    _mock_module.Net = MockNet
    sys.modules["utils.kicad_utils"] = _mock_module


# ==================== 合成 PCB 数据 ====================

BOARD_W, BOARD_H = 100.0, 80.0
BOARD_BOUNDS = ((0.0, 0.0), (BOARD_W, BOARD_H))


def build_synthetic_pcb() -> MockKiCadUtils:
    """生成一块简单的合成 PCB。

    板框 100×80mm，5 个可移动元件 + 2 个固定元件，3 条网络。
    """
    kicad = MockKiCadUtils()
    kicad.board_bounds = BOARD_BOUNDS

    # ---- 可移动元件（5 个，各有不同尺寸/类型） ----
    u1 = MockComponent(
        id="U1", footprint="QFP-44", package_type=2,
        bbox=(0, 0, 10, 10), footprint_w=10.0, footprint_h=10.0,
        nets=["N1", "N2"], is_fixed=False, is_placed=False,
        pins=[("1", 0, 5), ("44", 10, 5)],
    )
    u2 = MockComponent(
        id="U2", footprint="SOIC-8", package_type=1,
        bbox=(0, 0, 6, 4), footprint_w=6.0, footprint_h=4.0,
        nets=["N1", "N3"], is_fixed=False, is_placed=False,
        pins=[("1", 0, 2), ("8", 6, 2)],
    )
    c1 = MockComponent(
        id="C1", footprint="C_0603", package_type=0,
        bbox=(0, 0, 2, 1), footprint_w=2.0, footprint_h=1.0,
        nets=["N2"], is_fixed=False, is_placed=False,
        pins=[("1", 0, 0.5), ("2", 2, 0.5)],
    )
    c2 = MockComponent(
        id="C2", footprint="C_0603", package_type=0,
        bbox=(0, 0, 2, 1), footprint_w=2.0, footprint_h=1.0,
        nets=["N3"], is_fixed=False, is_placed=False,
        pins=[("1", 0, 0.5), ("2", 2, 0.5)],
    )
    r1 = MockComponent(
        id="R1", footprint="R_0805", package_type=0,
        bbox=(0, 0, 3, 1.5), footprint_w=3.0, footprint_h=1.5,
        nets=["N1"], is_fixed=False, is_placed=False,
        pins=[("1", 0, 0.75), ("2", 3, 0.75)],
    )

    # ---- 固定元件（2 个） ----
    j1 = MockComponent(
        id="J1", footprint="CONN_4", package_type=4,
        bbox=(0, 30, 10, 40), footprint_w=10.0, footprint_h=10.0,
        nets=["N1", "N2", "N3"], is_fixed=True, is_placed=True,
    )
    j2 = MockComponent(
        id="J2", footprint="CONN_4", package_type=4,
        bbox=(85, 30, 95, 40), footprint_w=10.0, footprint_h=10.0,
        nets=["N1", "N2", "N3"], is_fixed=True, is_placed=True,
    )

    for comp in [u1, u2, c1, c2, r1, j1, j2]:
        kicad.components[comp.id] = comp

    # ---- 网络（3 条） ----
    kicad.nets["N1"] = MockNet(
        net_id="N1",
        connections=[("U1", "1"), ("U2", "1"), ("R1", "1"),
                     ("J1", "1"), ("J2", "1")],
        is_power=False,
    )
    kicad.nets["N2"] = MockNet(
        net_id="N2",
        connections=[("U1", "2"), ("C1", "1"), ("J1", "2"), ("J2", "2")],
        is_power=True,
    )
    kicad.nets["N3"] = MockNet(
        net_id="N3",
        connections=[("U2", "2"), ("C2", "1"), ("J1", "3"), ("J2", "3")],
        is_power=True,
    )

    return kicad


# ==================== 环境工厂 ====================

def _init_env_from_kicad(env, kicad: MockKiCadUtils, pcb_path: str):
    """将 MockKiCadUtils 数据注入已 __new__ 的 PCBLayoutEnv 实例。

    复用 PCBLayoutEnv.__init__ 的核心初始化逻辑，消除 create_patched_env
    与 create_env_from_json 之间约 150 行重复代码。

    Args:
        env: PCBLayoutEnv.__new__ 创建的裸实例
        kicad: 已填充 components/nets/board_bounds 的 MockKiCadUtils
        pcb_path: 模拟的 PCB 路径（用于 _pcb_path 和日志标识）
    """
    from env.pcb_layout_env import Config
    from env.pcb_utils import PCBUtils
    from gymnasium import spaces

    env._pcb_path = pcb_path
    env.kicad = kicad

    (min_x, min_y), (max_x, max_y) = kicad.board_bounds
    env.board_bounds = kicad.board_bounds
    env.valid_area = (
        min_x + Config.BOARD_MARGIN, min_y + Config.BOARD_MARGIN,
        max_x - Config.BOARD_MARGIN, max_y - Config.BOARD_MARGIN,
    )
    env.grid_width = max(1, int((env.valid_area[2] - env.valid_area[0]) / Config.CELL_SIZE))
    env.grid_height = max(1, int((env.valid_area[3] - env.valid_area[1]) / Config.CELL_SIZE))

    env.fixed_list = [c for c in kicad.components.values() if c.is_fixed]
    env.movable_list = [c for c in kicad.components.values() if not c.is_fixed]
    env.N_movable = len(env.movable_list)

    env.placed_list = []
    env.current_step = 0
    env.cumulative_reward = 0.0
    env.nets = kicad.nets
    env.board_diagonal = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    env._pending_comp_idx = -1
    env._pending_gx1 = -1
    env._pending_gy1 = -1
    env._pending_rot = 0
    env._pending_lay = 0

    env.decoupling_map = PCBUtils.identify_decoupling_pairs(env.nets, kicad.components)
    env.area_ranks = PCBUtils.build_area_ranks(env.movable_list)
    env.keepout_zones = kicad.get_keepout_zones()

    env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL
    env._last_reward_breakdown = {}

    obs_dim = env.N_movable * 8 + env.N_movable + 4
    env.observation_space = spaces.Box(
        low=0.0,
        high=float(max(max_x - min_x, max_y - min_y, env.N_movable)),
        shape=(obs_dim,),
        dtype=np.float32,
    )


def create_patched_env():
    """创建使用合成数据的 PCBLayoutEnv（无需 KiCad）。

    Returns:
        就绪的 PCBLayoutEnv 实例（无 KiCad 依赖）
    """
    _install_mock_kicad()
    from env.pcb_layout_env import PCBLayoutEnv

    env = PCBLayoutEnv.__new__(PCBLayoutEnv)
    kicad = build_synthetic_pcb()
    kicad.save_initial_state()
    _init_env_from_kicad(env, kicad, "mock.kicad_pcb")

    logger.info(
        f"Mock env created: board={BOARD_W}×{BOARD_H}mm, "
        f"N_movable={env.N_movable}, grid=({env.grid_width},{env.grid_height}), "
        f"fixed={len(env.fixed_list)}"
    )
    return env


# ==================== JSON 数据加载器 ====================

def load_pcb_from_json(json_path: str) -> MockKiCadUtils:
    """从 JSON 文件加载合成 PCB 数据。

    Args:
        json_path: JSON 文件路径（由 scripts/generate_mock_pcbs.py 生成）

    Returns:
        MockKiCadUtils 实例，含完整的 components/nets/board_bounds
    """
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kicad = MockKiCadUtils()

    if "board_bounds" not in data:
        raise ValueError(f"JSON missing required field 'board_bounds': {json_path}")
    kicad.board_bounds = (
        tuple(data["board_bounds"][0]),
        tuple(data["board_bounds"][1]),
    )

    for i, comp_data in enumerate(data.get("components", [])):
        if "id" not in comp_data:
            raise ValueError(
                f"Component at index {i} missing required field 'id' in {json_path}"
            )
        comp = MockComponent(
            id=comp_data["id"],
            footprint=comp_data.get("footprint", ""),
            package_type=comp_data.get("package_type", 4),
            bbox=tuple(comp_data["bbox"]),
            rotation=comp_data.get("rotation", 0),
            layer=comp_data.get("layer", 0),
            is_fixed=comp_data.get("is_fixed", False),
            is_placed=comp_data.get("is_placed", False),
            nets=comp_data.get("nets", []),
            pins=[tuple(p) for p in comp_data.get("pins", [])],
            footprint_w=comp_data.get("footprint_w", 5.0),
            footprint_h=comp_data.get("footprint_h", 5.0),
        )
        kicad.components[comp.id] = comp

    for net_data in data.get("nets", []):
        net = MockNet(
            net_id=net_data["net_id"],
            connections=[tuple(c) for c in net_data.get("connections", [])],
            is_power=net_data.get("is_power", False),
        )
        kicad.nets[net.net_id] = net

    return kicad


def create_env_from_json(json_path: str):
    """从 JSON 文件创建 PCBLayoutEnv（无需 KiCad）。

    Args:
        json_path: PCB JSON 文件路径

    Returns:
        就绪的 PCBLayoutEnv 实例
    """
    _install_mock_kicad()
    from env.pcb_layout_env import PCBLayoutEnv

    kicad = load_pcb_from_json(json_path)
    kicad.save_initial_state()

    env = PCBLayoutEnv.__new__(PCBLayoutEnv)
    _init_env_from_kicad(env, kicad, json_path)

    (min_x, min_y), (max_x, max_y) = kicad.board_bounds
    logger.info(
        f"Mock env loaded from JSON: board={max_x - min_x:.0f}×{max_y - min_y:.0f}mm, "
        f"N_movable={env.N_movable}, grid=({env.grid_width},{env.grid_height}), "
        f"fixed={len(env.fixed_list)}"
    )
    return env
