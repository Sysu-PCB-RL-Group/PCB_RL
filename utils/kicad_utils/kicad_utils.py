"""KiCad PCB 文件底层操作模块。

封装 KiCad 9.0 Python API，提供 PCB 数据导入（.kicad_pcb → Component/Net）
和布局导出（RL 优化结果 → .kicad_pcb）功能。

维护方：KiCad 导入导出负责同学。
接口契约详见 docs/kicad_interface.md。
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

KICAD_PATH = os.environ.get("KICAD_PATH", "")

KICAD_AVAILABLE = False
if not KICAD_PATH:
    print("[INFO] KICAD_PATH 未设置，跳过 KiCad 加载（Mock 模式可用）")
    print("       设置方法: set KICAD_PATH=E:\\Kicad   （指向 KiCad 安装目录）")
else:
    try:
        os.add_dll_directory(os.path.join(KICAD_PATH, "bin"))
        sys.path.insert(0, os.path.join(KICAD_PATH, "bin", "Lib", "site-packages"))
        sys.path.insert(0, os.path.join(KICAD_PATH, "bin"))
        import pcbnew  # noqa: F401
        KICAD_AVAILABLE = True
    except (OSError, ImportError) as e:
        print(f"[WARN] KiCad 加载失败 ({KICAD_PATH}): {e}")
        print("       Mock 模式仍可用，训练前请确认 KICAD_PATH 正确")

# ===================== 固定元件启发式 =====================

# 立创/EasyEDA 导出的 footprint 从不置 lock，RL 无锚点。
# 固定判定（见 _cache_components）：
#   1. 文件锁标记 IsLocked()               —— 真实锁定，最高优先
#   2. 0 网络连接（机械件：螺丝/铜柱/定位点）—— 数据实锤，无电气功能
#   3. 封装名关键词                         —— 默认仅机械锚点
#
# 连接器(HDR/CONN/USB/PIN) 和大 IC(BGA/QFP) 默认【不固定】——
# 它们是 RL 该优化放置的对象（reward 里还有"大元件优先"）。
# 如需固定，把对应开关改为 True 即可。
_FIXED_FOOTPRINT_KEYWORDS = ("HOLE", "MOUNT", "MECH", "FIDUCIAL")
_FIX_CONNECTORS = False          # True → 额外固定连接器 HDR/CONN/PIN/SOCKET/USB
_FIX_LARGE_ICS = False           # True → 额外固定大 IC BGA/QFP
_CONNECTOR_KEYWORDS = ("HDR", "CONN", "PIN", "SOCKET", "USB")
_LARGE_IC_KEYWORDS = ("BGA", "QFP")


def _is_fixed_footprint(footprint_name: str) -> bool:
    """按封装名关键词判定元件是否应固定为锚点。

    默认仅机械锚点（HOLE/MOUNT/MECH/FIDUCIAL）；
    连接器和大 IC 默认可动，通过 _FIX_CONNECTORS/_FIX_LARGE_ICS 开启。
    """
    upper = footprint_name.upper()
    if any(k in upper for k in _FIXED_FOOTPRINT_KEYWORDS):
        return True
    if _FIX_CONNECTORS and any(k in upper for k in _CONNECTOR_KEYWORDS):
        return True
    if _FIX_LARGE_ICS and any(k in upper for k in _LARGE_IC_KEYWORDS):
        return True
    return False


# ===================== 数据结构定义 =====================

@dataclass
class Component:
    """元器件核心数据结构（AABB-based）"""
    id: str                                          # 参考编号（如 R1、U2）
    footprint: str                                   # 封装名称
    package_type: int                                # 封装类型编码 0-4
    bbox: Tuple[float, float, float, float]          # 当前位置 AABB (x1, y1, x2, y2) mm
    rotation: int                                    # 当前旋转 0=0°, 1=90°, 2=180°, 3=270°
    layer: int                                       # 0=TOP, 1=BOTTOM
    is_fixed: bool                                   # 是否固定（不可移动）
    is_placed: bool                                  # 是否已放置（C1 追踪）
    nets: List[str] = field(default_factory=list)    # 连接的 net_id 列表
    pins: List[Tuple[str, float, float]] = field(default_factory=list)
    # pins: [(pin_number, rel_x, rel_y), ...] 相对【源姿态】bbox 左上角的引脚坐标 (mm)
    #       源姿态 = 源文件当前旋转状态，非 0°；旋转后需用 pin_abs_positions() 取绝对坐标
    footprint_w: float = 0.0                         # 源姿态下的 AABB 宽度 mm（非 0° 尺寸）
    footprint_h: float = 0.0                         # 源姿态下的 AABB 高度 mm
    source_rotation: int = 0                         # 源文件中的旋转（0-3），footprint_w/h 的测量姿态
    source_bbox: Tuple[float, float, float, float] = field(default_factory=tuple)
    # source_bbox: 源姿态 AABB（导入时），与 footprint_w/h 一致（不可变）
    pin_offsets_by_rotation: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    # pin_offsets_by_rotation[r]: [(ox, oy), ...] 旋转 r 下各 pad 相对该旋转 bbox 中心的偏移 (mm)
    #   顺序与 pins 一致；由 _cache_components 用真实 KiCad 几何预计算（含旋转）

    @property
    def footprint_area(self) -> float:
        """封装面积 (mm²)（源姿态下的面积，旋转不改变面积）"""
        return self.footprint_w * self.footprint_h

    @property
    def center(self) -> Tuple[float, float]:
        """从 AABB 推导中心点 (mm)"""
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)

    def dims_at_rotation(self, target_rotation: int) -> Tuple[float, float]:
        """返回目标旋转下的 AABB 宽高 (w, h)。

        footprint_w/h 是源姿态（source_rotation）下的 AABB 尺寸；
        目标旋转相对源旋转的 delta 为奇数（90°/270°）时宽高交换，
        偶数（0°/180°）时不交换。纯计算，不修改任何状态。
        """
        delta = (target_rotation - self.source_rotation) % 4
        if delta % 2 == 0:
            return self.footprint_w, self.footprint_h
        return self.footprint_h, self.footprint_w

    def pin_abs_positions(self, rotation: Optional[int] = None) -> List[Tuple[str, float, float]]:
        """返回指定（或当前）旋转下各引脚的绝对坐标。

        绝对位置 = 当前 bbox 中心 + 该旋转预计算的 pad 偏移。
        引脚随旋转同步更新，与 KiCad 真实几何一致。

        Args:
            rotation: 目标旋转（0-3），默认使用 self.rotation

        Returns:
            [(pin_number, abs_x, abs_y), ...]，与 pins 顺序一致
        """
        if rotation is None:
            rotation = self.rotation
        cx, cy = self.center
        offsets = self.pin_offsets_by_rotation.get(rotation, [])
        return [
            (num, cx + ox, cy + oy)
            for (num, _rel_x, _rel_y), (ox, oy) in zip(self.pins, offsets)
        ]


@dataclass
class Net:
    """网络数据结构（精简版，无 routing 字段）"""
    net_id: str                                      # 网络名称（如 VCC、GND）
    connections: List[Tuple[str, str]]               # [(comp_id, pin_number), ...]
    is_power: bool = False                           # 是否电源网络（GND/VCC/VDD/VSS/PWR）

    # 电源网络判定：精确匹配（高置信度）+ 前缀匹配（中等置信度）
    _POWER_NET_EXACT = {"VCC", "VDD", "VSS", "GND", "PWR", "VEE", "VPP", "VREF",
                        "AGND", "DGND", "AVCC", "AVDD", "DVCC", "DVDD"}
    _POWER_NET_PREFIXES = ("VCC", "VDD", "VSS", "VEE",
                           "+5V", "+3.3V", "+3V3", "+12V", "+1.8V",
                           "VIN", "VBUS", "VBAT", "VDDC", "VDDA",
                           "VCCINT", "VCCIO", "VCCPLL")

    @staticmethod
    def is_power_net(net_name: str) -> bool:
        """判定是否为电源网络（大小写不敏感）。

        先做精确白名单匹配，再按前缀模糊匹配覆盖常见命名变体。
        """
        upper = net_name.upper()
        if upper in Net._POWER_NET_EXACT:
            return True
        return any(upper.startswith(p) for p in Net._POWER_NET_PREFIXES)


# ===================== KiCad 核心工具类 =====================

class KiCadUtils:
    """KiCad PCB 底层操作工具类。

    RL 层（PCBLayoutEnv / PCBUtils）仅通过以下接口访问 KiCad 数据：
    - 属性：components, nets, board_bounds
    - 导入：load_board(), get_board_bounds(), get_keepout_zones()
    - 导出：export_layout()

    导入逻辑和导出逻辑均由 KiCad 同学负责实现和维护。
    接口契约详见 docs/kicad_interface.md。
    """

    def __init__(self):
        """初始化 KiCadUtils。

        创建空的缓存结构，待 load_board() 填充实际数据。
        """
        self.board: Optional[pcbnew.BOARD] = None
        self.footprints: Dict[str, pcbnew.FOOTPRINT] = {}
        self._source_path: Optional[str] = None      # load_board 的源 PCB 路径（导出时重载模板用）

        self.components: Dict[str, Component] = {}
        self.nets: Dict[str, Net] = {}
        self.board_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 0), (0, 0))

    # -------------------- 基础加载 / 缓存 --------------------

    @staticmethod
    def _preflight(pcb_path: str) -> bool:
        """子进程预检，挡住会让 LoadBoard 无界阻塞的板。

        返回 True = 可以继续加载。

        刻意做成惰性导入 + 全异常降级：
          * kicad_utils.py 会被 tests/test_mock_geometry.py 用
            spec_from_file_location 以【非包身份】加载，相对导入会失败，
            故只用绝对导入；
          * 那些测试还会把 sys.modules["utils.kicad_utils"] 换成 MagicMock，
            导入到的东西可能不是真函数；
          * 预检本身绝不应该成为加载失败的新原因——任何意外都放行，
            交回给原路径处理（最坏情况退化成修复前的行为）。
        """
        if not KICAD_AVAILABLE:
            return True
        try:
            from utils.kicad_utils.board_preflight import (
                preflight_enabled, probe_board, format_skip_line,
            )
            if not preflight_enabled():
                return True
            result = probe_board(pcb_path)
            if result.ok:
                return True
            print(format_skip_line(result))
            return False
        except Exception as e:                      # noqa: BLE001
            print(f"[WARN] 预检不可用，直接加载: {e}")
            return True

    def load_board(self, pcb_path: str) -> bool:
        """加载 PCB 文件并缓存所有核心数据。

        依次执行 footprint 缓存、元器件信息提取、网络连接解析和板框边界计算，
        最后打印加载统计日志。

        加载前先做一次子进程预检（见 utils/kicad_utils/board_preflight.py）：
        pcbnew.LoadBoard 在文本引用了本机未安装的字体会**无界阻塞**
        （实测某板阻塞 168 分钟仍未返回、CPU≈0%），线程与 signal 都打不断，
        因此在主进程里直接调用可能让整个训练永久挂死。预检用独立子进程 +
        15s 硬超时把这种板挡在外面（PCB_LOAD_TIMEOUT / PCB_PREFLIGHT 可调）。

        Args:
            pcb_path: PCB 文件路径（.kicad_pcb 格式）

        Returns:
            True 加载成功，False 加载失败或被预检跳过
        """
        if not self._preflight(pcb_path):
            return False

        try:
            self.board = pcbnew.LoadBoard(pcb_path)
            if self.board is None:
                raise RuntimeError(f"pcbnew.LoadBoard returned None (file not found or invalid format)")

            self._source_path = os.path.abspath(pcb_path)

            self.components.clear()
            self.nets.clear()

            self._cache_footprints()
            self._cache_components()
            self._cache_nets()
            self._calc_board_bounds()

            self._print_load_log()
            print(f"[OK] load pcb file: {pcb_path}")
            return True
        except Exception as e:
            print(f"[FAIL] load pcb file: {e}")
            return False

    @staticmethod
    def _index_footprints_by_ref(board) -> Dict[str, Any]:
        """按参考编号建立 footprint 索引，处理空 Reference 兜底与重名去重。

        与 _cache_footprints / export_layout 共享同一套 ref 修正逻辑，
        保证导出时从重载模板索引到的 ref 与 load_board 时一致。
        """
        result: Dict[str, pcbnew.FOOTPRINT] = {}
        counter = 0
        for fp in board.Footprints():
            ref = fp.GetReference()
            if not ref:
                # EasyEDA 转换文件 Reference 为空 → 用库项名兜底
                try:
                    ref = str(fp.GetFPID().GetLibItemName())
                except Exception:
                    counter += 1
                    ref = f"UNK_{counter}"
            # 确保唯一（多个同型号元件 LibraryItemName 可能重复）
            base = ref
            while ref in result:
                counter += 1
                ref = f"{base}_{counter}"
            result[ref] = fp
        return result

    def _cache_footprints(self):
        """缓存所有 footprint 对象。

        遍历 board.Footprints()，以参考编号为 key 存入 self.footprints 字典。
        EasyEDA 转换文件 Reference 可能为空，此时用库项名兜底并去重。
        同时存一份按遍历顺序的 ref 列表，供 _cache_nets 索引匹配。
        """
        self.footprints = self._index_footprints_by_ref(self.board)
        self._fp_refs = list(self.footprints.keys())  # 按遍历顺序保存修正后的 ref

    def _cache_components(self):
        """缓存所有元器件信息（AABB-based，含封装类型编码）。

        从 KiCad footprint 提取：参考编号、位置坐标、旋转角度、所在层、锁定状态、
        封装 AABB 边界、封装类型编码、引脚信息，构造 Component 对象存入 self.components。
        固定元件（is_fixed=True）初始 is_placed 也设为 True。
        """
        for ref, fp in self.footprints.items():
            reference = ref  # 直接使用 dict key（由 _cache_footprints 确保唯一）

            # 基础信息
            pos_nm = fp.GetPosition()
            pos_mm = (pcbnew.ToMM(pos_nm.x), pcbnew.ToMM(pos_nm.y))
            rotation_deg = fp.GetOrientationDegrees()
            rotation = int(round(rotation_deg / 90.0)) % 4      # 0/1/2/3
            layer_name = pcbnew.LayerName(fp.GetLayer())
            # 层编码：0=TOP(F.Cu), 1=BOTTOM(B.Cu), 2=内层/未知
            # 多层板内层（In*.Cu）暂不支持元件放置，归为 2 并在日志中警告
            if layer_name == "F.Cu":
                layer = 0
            elif layer_name == "B.Cu":
                layer = 1
            else:
                layer = 2
                print(f"⚠️  元件 {reference} 位于非标准层 {layer_name}，编码为 2")
            footprint_name = str(fp.GetFPID().GetLibItemName())
            # 固定判定：文件锁 或 机械关键词 或 0 网络（数据实锤的机械件）
            is_fixed = fp.IsLocked() or _is_fixed_footprint(footprint_name)
            if not is_fixed:
                # 0 网络连接 = 机械件（螺丝/铜柱/定位点/无 pad 图形件），固定零损失
                pad_nets = {pad.GetNetname() for pad in fp.Pads() if pad.GetNetname()}
                if not pad_nets:
                    is_fixed = True

            # 封装尺寸（bbox）——统一用 GetBoundingHull().BBox()（含旋转、与导出层一致）。
            # 注意：GetBoundingBox() 是缓存，SetOrientationDegrees 后不刷新；
            #       GetBoundingHull().BBox() 实时反映旋转后的真实几何（export 亦用它定位）。
            # AABB: (x1=left, y1=top, x2=right, y2=bottom)
            # 注意：KiCad 坐标系 Y 轴向下，GetTop() 的 y 值 < GetBottom()，
            # 为满足下游 bbox 统一约定 y1 < y2，这里 y1 取 GetTop（小值）、
            # y2 取 GetBottom（大值）。
            hull = fp.GetBoundingHull().BBox()
            fp_w = pcbnew.ToMM(hull.GetWidth())
            fp_h = pcbnew.ToMM(hull.GetHeight())
            bbox_mm = (
                pcbnew.ToMM(hull.GetLeft()),
                pcbnew.ToMM(hull.GetTop()),
                pcbnew.ToMM(hull.GetRight()),
                pcbnew.ToMM(hull.GetBottom()),
            )

            # 封装类型编码
            package_type = self._classify_package_type(footprint_name)

            # 引脚信息（相对源姿态 bbox 左上角：x 基准=left，y 基准=top）。
            # 下游取绝对坐标须经 pin_abs_positions()（随旋转更新），不要直接用
            # bbox[0]+rel_x——元件旋转后 bbox 尺寸/位置已变，旧公式会失真。
            pins = []
            for pad in fp.Pads():
                pad_pos_nm = pad.GetPosition()
                pad_pos_mm = (
                    pcbnew.ToMM(pad_pos_nm.x),
                    pcbnew.ToMM(pad_pos_nm.y),
                )
                rel_x = pad_pos_mm[0] - bbox_mm[0]
                rel_y = pad_pos_mm[1] - bbox_mm[1]
                pins.append((pad.GetNumber(), rel_x, rel_y))

            # 预计算 4 个旋转下各 pad 相对该旋转 bbox 中心的偏移（真实 KiCad 几何）。
            # 方法：临时设置 footprint 旋转 → 读 hull bbox 中心与 pad 世界坐标
            #       （PAD::GetPosition() 随父 footprint 变换更新）→ 计算偏移 →
            #       finally 恢复源姿态。源姿态的 bbox/旋转在恢复后不变。
            pin_offsets_by_rotation = {}
            _saved_rot = fp.GetOrientationDegrees()
            _saved_pos = fp.GetPosition()
            _saved_flipped = bool(fp.IsFlipped())
            try:
                for r in range(4):
                    fp.SetOrientationDegrees(float(r * 90))
                    h = fp.GetBoundingHull().BBox()
                    # Y 轴向下，GetBottom() > GetTop()；中心取两者中值即可
                    hcx_nm = (h.GetLeft() + h.GetRight()) / 2.0
                    hcy_nm = (h.GetTop() + h.GetBottom()) / 2.0
                    offsets = []
                    for pad in fp.Pads():
                        p = pad.GetPosition()
                        offsets.append((pcbnew.ToMM(p.x - hcx_nm),
                                        pcbnew.ToMM(p.y - hcy_nm)))
                    pin_offsets_by_rotation[r] = offsets
            finally:
                # 恢复必须用 Flip，不能再用 SetLayer：SetLayer 只改父层，撤销不了
                # Flip 对 pad 子层与几何做的镜像。本循环目前只改旋转，故这是 no-op；
                # 但只要循环内出现 Flip（预计算底面姿态时必然出现），用 SetLayer
                # 恢复就会静默污染 self.board / self.footprints 缓存
                # （反向对照实测：污染症状为 "pad 子层被污染"）。
                # 顺序：先复原面，再写绝对角与位置（Flip 会改写这两者）。
                if bool(fp.IsFlipped()) != _saved_flipped:
                    fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
                fp.SetOrientationDegrees(_saved_rot)
                fp.SetPosition(_saved_pos)

            self.components[reference] = Component(
                id=reference,
                footprint=footprint_name,
                package_type=package_type,
                bbox=bbox_mm,
                rotation=rotation,
                layer=layer,
                is_fixed=is_fixed,
                is_placed=is_fixed,      # 固定元件初始视为"已放置"
                nets=[],                  # 在 _cache_nets 中回填
                pins=pins,
                footprint_w=fp_w,
                footprint_h=fp_h,
                source_rotation=rotation,
                source_bbox=bbox_mm,
                pin_offsets_by_rotation=pin_offsets_by_rotation,
            )

    @staticmethod
    def _classify_package_type(footprint_name: str) -> int:
        """根据封装名称字符串匹配封装类型编码 0-4。

        通过关键字匹配判断封装类别，用于间距阈值查表。

        Args:
            footprint_name: KiCad 封装库项名称（如 "SOIC-8_3.9x4.9mm"）

        Returns:
            封装类型编码：0=Chip, 1=SOIC/SOT, 2=QFP, 3=BGA, 4=IC 通用
        """
        name_upper = footprint_name.upper()
        if any(kw in name_upper for kw in ("0402", "0603", "0805", "1206", "1210", "2512")):
            return 0    # Chip
        if any(kw in name_upper for kw in ("SOIC", "SOT", "SSOP", "TSSOP",
                                            "DFN", "QFN", "WSON", "MSOP")):
            return 1    # SOIC/SOT/QFN 类
        if any(kw in name_upper for kw in ("QFP", "TQFP", "LQFP", "PLCC")):
            return 2    # QFP 类
        if any(kw in name_upper for kw in ("BGA", "LGA", "CSP", "WLP")):
            return 3    # BGA/阵列类
        return 4        # IC 通用（如 DIP、SIP、自定义封装）

    def _cache_nets(self):
        """缓存网络连接信息，并回填 Component.nets。

        遍历所有网络，收集每个网络的连接 (comp_id, pin_number) 列表，
        同时建立 net_name → component_refs 映射用于回填 Component.nets 字段。
        """
        # 先建立 net → components 映射
        net_map: Dict[str, List[str]] = {}

        for net in self.board.GetNetsByName().values():
            net_name = net.GetNetname()
            if not net_name:
                continue

            connections = []
            target_net_code = net.GetNetCode()
            # 直接遍历已缓存的 footprints dict，避免依赖两次 Footprints() 迭代顺序一致
            for ref, fp in self.footprints.items():
                for pad in fp.Pads():
                    if pad.GetNetCode() == target_net_code:
                        connections.append((ref, pad.GetNumber()))
                        net_map.setdefault(net_name, [])
                        if ref not in net_map[net_name]:
                            net_map[net_name].append(ref)

            is_power = Net.is_power_net(net_name)
            self.nets[net_name] = Net(
                net_id=net_name,
                connections=connections,
                is_power=is_power,
            )

        # 回填 Component.nets
        for net_name, comp_refs in net_map.items():
            for ref in comp_refs:
                if ref in self.components:
                    self.components[ref].nets.append(net_name)

    def _calc_board_bounds(self):
        """计算电路板框边界 ((min_x, min_y), (max_x, max_y)) mm。

        优先从 Edge.Cuts 层读取线条/圆弧/多边形顶点计算包围盒；
        若无 Edge.Cuts 层则使用所有元器件边界替代；
        两者都无则回退到默认值 (0, 0) ~ (100, 80)。
        """
        edge_cuts_id = self.board.GetLayerID("Edge.Cuts")
        if edge_cuts_id == -1:
            # 无 Edge.Cuts 层 → 用元器件边界替代
            if not self.components:
                print("[WARN] 无 Edge.Cuts 且无元件，板框回退默认值 (0,0)~(100,80)")
                self.board_bounds = ((0, 0), (100, 80))
                return
            bboxes = [c.bbox for c in self.components.values()]
            xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
            ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
            self.board_bounds = ((min(xs), min(ys)), (max(xs), max(ys)))
            return

        # 从 Edge.Cuts 层计算边界
        all_vertices = []
        for drawing in self.board.GetDrawings():
            if drawing.GetLayer() != edge_cuts_id:
                continue
            if not isinstance(drawing, pcbnew.PCB_SHAPE):
                continue

            shape_type = drawing.GetShapeStr()
            if shape_type == "Line":
                s = drawing.GetStart()
                e = drawing.GetEnd()
                all_vertices.append((pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)))
                all_vertices.append((pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)))
            elif shape_type == "Circle":
                c = drawing.GetCenter()
                r = pcbnew.ToMM(drawing.GetRadius())
                all_vertices.append((pcbnew.ToMM(c.x) - r, pcbnew.ToMM(c.y) - r))
                all_vertices.append((pcbnew.ToMM(c.x) + r, pcbnew.ToMM(c.y) + r))
            elif shape_type == "Arc":
                s = drawing.GetStart()
                e = drawing.GetEnd()
                all_vertices.append((pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)))
                all_vertices.append((pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)))
            elif shape_type == "Rect":
                # 矩形：取对角两点作为边界顶点（此前被跳过导致假板框）
                s = drawing.GetStart()
                e = drawing.GetEnd()
                all_vertices.append((pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)))
                all_vertices.append((pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)))
            elif shape_type in ("Bezier", "Curve"):
                # 贝塞尔曲线：控制点 + 端点共同构成边界框
                s = drawing.GetStart()
                e = drawing.GetEnd()
                c1 = drawing.GetBezControl1()
                c2 = drawing.GetBezControl2()
                all_vertices.append((pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)))
                all_vertices.append((pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)))
                all_vertices.append((pcbnew.ToMM(c1.x), pcbnew.ToMM(c1.y)))
                all_vertices.append((pcbnew.ToMM(c2.x), pcbnew.ToMM(c2.y)))
            elif shape_type == "Polygon":
                poly = drawing.GetPolyShape()
                for oi in range(poly.OutlineCount()):
                    outline = poly.Outline(oi)
                    for i in range(outline.PointCount()):
                        pt = outline.CPoint(i)
                        all_vertices.append((pcbnew.ToMM(pt.x), pcbnew.ToMM(pt.y)))

        if not all_vertices:
            print("[WARN] Edge.Cuts 无有效形状，板框回退默认值 (0,0)~(100,80)")
            self.board_bounds = ((0, 0), (100, 80))
            return
        xs = [p[0] for p in all_vertices]
        ys = [p[1] for p in all_vertices]
        self.board_bounds = ((min(xs), min(ys)), (max(xs), max(ys)))

    # -------------------- 对外核心接口 --------------------

    def get_board_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """获取电路板边界。

        Returns:
            ((min_x, min_y), (max_x, max_y)) 单位 mm
        """
        return self.board_bounds

    def get_keepout_zones(self) -> List[Tuple[float, float, float, float]]:
        """从 KiCad Zone / Rule Area 读取禁止区域。

        遍历 board.Zones()，筛选规则区域（GetIsRuleArea），提取 AABB 包围盒。

        Returns:
            List[(x1, y1, x2, y2)] 禁止区域的 AABB 列表，单位 mm
        """
        keepouts = []
        if self.board is None:
            return keepouts
        try:
            for zone in self.board.Zones():
                if hasattr(zone, "GetIsRuleArea") and zone.GetIsRuleArea():
                    bbox = zone.GetBoundingBox()
                    # 与 _cache_components 一致：y1 取 GetTop（小值）、
                    # y2 取 GetBottom（大值），满足下游 y1 < y2 约定
                    keepouts.append((
                        pcbnew.ToMM(bbox.GetLeft()),
                        pcbnew.ToMM(bbox.GetTop()),
                        pcbnew.ToMM(bbox.GetRight()),
                        pcbnew.ToMM(bbox.GetBottom()),
                    ))
        except Exception:
            pass
        return keepouts

    # -------------------- 导出 --------------------

    def export_layout(
        self,
        output_path: str,
        components: Dict[str, Component],
        template_path: Optional[str] = None,
    ) -> bool:
        """将 RL 优化后的布局导出为 KiCad PCB 文件。

        **每次导出都重新加载一份独立源模板**，应用 RL 状态后保存，绝不修改
        缓存的 self.board。这样 final/best 多次导出互不污染，相同输入恒得
        相同输出，restore_initial_state() 也不依赖导出的副作用。

        只改位置/旋转/层，其余（走线、敷铜、丝印、3D 模型等）全部保留。

        Args:
            output_path: 输出 PCB 文件路径（.kicad_pcb 格式）
            components: RL 优化后的元件数据 {ref: Component}
            template_path: 模板路径，None 则使用 load_board 时的源 PCB 路径

        Returns:
            True 导出成功，False 导出失败
        """
        try:
            source = template_path or getattr(self, "_source_path", None)
            if source is None:
                print("[FAIL] Layout export error: no source path (call load_board first)")
                return False
            board = pcbnew.LoadBoard(source)
            if board is None:
                print(f"[FAIL] Layout export error: failed to load template {source}")
                return False
            # 与 load_board 相同的 ref 修正/去重逻辑，确保索引到正确 footprint
            footprints = self._index_footprints_by_ref(board)

            for ref, comp in components.items():
                # ① 只写 RL 放过的元件：固定 + 未放置都保持源坐标原封不动
                if comp.is_fixed or not comp.is_placed:
                    continue
                fp = footprints.get(ref)
                if fp is None:
                    continue

                # ② 层守卫：非标准层（内层等）不移动、不写层
                if comp.layer not in (0, 1):
                    print(f"[WARN] {ref} 在非标准层 {comp.layer}，跳过导出")
                    continue

                # AABB → 目标中心点坐标 (mm)
                cx = (comp.bbox[0] + comp.bbox[2]) / 2.0
                cy = (comp.bbox[1] + comp.bbox[3]) / 2.0

                # ③ 先定面：跨面必须用 Flip()，绝不能用 SetLayer()。
                # SetLayer 只改 footprint 父对象层，pad/钢网/阻焊/丝印/Fab/CrtYd
                # 全部留在原面，导出结构不一致的 footprint（实测 KiCad 10.0.4：
                # SetLayer(B.Cu) 后 IsFlipped()==True 但 pad 仍在 F.Cu）。
                # Flip() 同时完成：父层 + pad 子层(F/B.Paste/Mask) + 丝印/Fab/CrtYd
                # + 内部几何镜像 + 焊盘坐标 + 文本镜像 + 3D 模型正反面。
                # 关于方向：实测 TOP_BOTTOM 与 LEFT_RIGHT 仅在「中间态存储角」上
                # 差 180°，随后 SetOrientationDegrees 写绝对角即抹平，两者产出的
                # 存盘角度与 pad 坐标逐位一致（见 verification_output/
                # verify_flip_direction_equivalence.py [2]），故方向选择不影响结果。
                want_bottom = (comp.layer == 1)
                if bool(fp.IsFlipped()) != want_bottom:
                    # 以自身位置为镜像中心 → 位置不变；否则位置会被镜像
                    fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)

                # ④ 翻面之后才写最终绝对角度（顺序不可换：Flip 会改写角度）
                fp.SetOrientationDegrees(float(comp.rotation * 90))

                # ⑤ 位置写入：实测偏移反推参考点（非对称封装不产生系统偏移）
                # 必须在 Flip + SetOrientationDegrees 之后重新量：实测非对称封装
                # rot=0 时 hull 中心相对锚点由 (+0.5,+1.5) 变为 (+0.5,-1.5)，
                # rot=90 由 (+1.5,-0.5) 变为 (-1.5,-0.5)。翻面前算的偏移会错位。
                # 注意：GetBoundingBox() 是缓存，SetOrientationDegrees 后不刷新；
                #       必须用 GetBoundingHull().BBox()（实时反映旋转后的真实几何）
                bbox = fp.GetBoundingHull().BBox()
                pos = fp.GetPosition()
                off_x = (bbox.GetLeft() + bbox.GetRight()) / 2 - pos.x
                off_y = (bbox.GetBottom() + bbox.GetTop()) / 2 - pos.y
                fp.SetPosition(pcbnew.VECTOR2I(
                    pcbnew.FromMM(cx) - round(off_x),
                    pcbnew.FromMM(cy) - round(off_y),
                ))

            board.Save(output_path)
            print(f"[OK] Layout exported: {output_path}")
            return True

        except Exception as e:
            print(f"[FAIL] Layout export error: {e}")
            return False

    # -------------------- 辅助 --------------------

    def _print_load_log(self):
        """打印 PCB 加载统计日志。

        输出元器件数量（固定/可移动）、网络数量、板框边界范围。
        """
        print("=" * 60)
        print("PCB 加载统计:")
        fixed_count = sum(1 for c in self.components.values() if c.is_fixed)
        movable_count = len(self.components) - fixed_count
        print(f"  - 元器件: {len(self.components)} (固定: {fixed_count}, 可移动: {movable_count})")
        print(f"  - 网络: {len(self.nets)}")
        print(f"  - 板框边界: {self.board_bounds[0]} ~ {self.board_bounds[1]} mm")
        print("=" * 60)

    # -------------------- 状态快照 / 恢复 --------------------

    def restore_initial_state(self):
        """从内存缓存恢复元件和网络的初始状态。

        不重新读取磁盘文件，直接基于已缓存的 footprints/board 重建
        components 和 nets，使所有元件回到 load_board() 时的原始位置。

        RL 训练/推理中每个 episode 的 reset() 均可使用此方法替代 load_board()，
        避免反复解析 PCB 文件。
        """
        self.components.clear()
        self.nets.clear()
        self._cache_components()
        self._cache_nets()
