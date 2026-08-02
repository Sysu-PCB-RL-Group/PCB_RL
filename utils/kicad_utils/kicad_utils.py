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

# ===================== 数据结构定义 =====================

@dataclass
class Component:
    """元器件核心数据结构（AABB-based）"""
    id: str                                          # 参考编号（如 R1、U2）
    footprint: str                                   # 封装名称
    package_type: int                                # 封装类型编码 0-4
    bbox: Tuple[float, float, float, float]          # 当前位置 AABB (x1, y1, x2, y2) mm
    rotation: int                                    # 0=0°, 1=90°, 2=180°, 3=270°
    layer: int                                       # 0=TOP, 1=BOTTOM
    is_fixed: bool                                   # 是否固定（不可移动）
    is_placed: bool                                  # 是否已放置（C1 追踪）
    nets: List[str] = field(default_factory=list)    # 连接的 net_id 列表
    pins: List[Tuple[str, float, float]] = field(default_factory=list)
    # pins: [(pin_number, rel_x, rel_y), ...] 相对 bbox 左下角的引脚坐标 (mm)
    footprint_w: float = 0.0                         # 封装原始宽度 mm（0° rotation）
    footprint_h: float = 0.0                         # 封装原始高度 mm（0° rotation）

    @property
    def footprint_area(self) -> float:
        """封装面积 (mm²)"""
        return self.footprint_w * self.footprint_h

    @property
    def center(self) -> Tuple[float, float]:
        """从 AABB 推导中心点 (mm)"""
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)


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

        self.components: Dict[str, Component] = {}
        self.nets: Dict[str, Net] = {}
        self.board_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 0), (0, 0))

    # -------------------- 基础加载 / 缓存 --------------------

    def load_board(self, pcb_path: str) -> bool:
        """加载 PCB 文件并缓存所有核心数据。

        依次执行 footprint 缓存、元器件信息提取、网络连接解析和板框边界计算，
        最后打印加载统计日志。

        Args:
            pcb_path: PCB 文件路径（.kicad_pcb 格式）

        Returns:
            True 加载成功，False 加载失败
        """
        try:
            self.board = pcbnew.LoadBoard(pcb_path)
            if self.board is None:
                raise RuntimeError(f"pcbnew.LoadBoard returned None (file not found or invalid format)")

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

    def _cache_footprints(self):
        """缓存所有 footprint 对象。

        遍历 board.Footprints()，以参考编号为 key 存入 self.footprints 字典。
        EasyEDA 转换文件 Reference 可能为空，此时用库项名兜底并去重。
        同时存一份按遍历顺序的 ref 列表，供 _cache_nets 索引匹配。
        """
        self.footprints.clear()
        self._fp_refs = []  # 按遍历顺序保存修正后的 ref
        counter = 0
        for fp in self.board.Footprints():
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
            while ref in self.footprints:
                counter += 1
                ref = f"{base}_{counter}"
            self.footprints[ref] = fp
            self._fp_refs.append(ref)

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
            is_fixed = fp.IsLocked()

            # 封装尺寸（bbox）
            bbox = fp.GetBoundingBox()                           # pcbnew.BOX2I
            fp_w = pcbnew.ToMM(bbox.GetWidth())
            fp_h = pcbnew.ToMM(bbox.GetHeight())
            # AABB: (x1=left, y1=bottom, x2=right, y2=top)
            bbox_mm = (
                pcbnew.ToMM(bbox.GetLeft()),
                pcbnew.ToMM(bbox.GetBottom()),
                pcbnew.ToMM(bbox.GetRight()),
                pcbnew.ToMM(bbox.GetTop()),
            )

            # 封装类型编码
            footprint_name = str(fp.GetFPID().GetLibItemName())
            package_type = self._classify_package_type(footprint_name)

            # 引脚信息（相对 AABB 左下角）
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
            for idx, footprint in enumerate(self.board.Footprints()):
                # 使用 _cache_footprints 修正后的 ref，而非原始的 fp.GetReference()
                ref = self._fp_refs[idx] if idx < len(self._fp_refs) else footprint.GetReference()
                for pad in footprint.Pads():
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
                    keepouts.append((
                        pcbnew.ToMM(bbox.GetLeft()),
                        pcbnew.ToMM(bbox.GetBottom()),
                        pcbnew.ToMM(bbox.GetRight()),
                        pcbnew.ToMM(bbox.GetTop()),
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

        从 self.board（缓存的完整 PCB）中找到对应 footprint，
        只改位置/旋转/层，其余（走线、敷铜、丝印、3D 模型等）全部保留。

        Args:
            output_path: 输出 PCB 文件路径（.kicad_pcb 格式）
            components: RL 优化后的元件数据 {ref: Component}
            template_path: 模板路径，None 则使用 self.board

        Returns:
            True 导出成功，False 导出失败
        """
        try:
            board = self.board
            if template_path is not None:
                board = pcbnew.LoadBoard(template_path)
                # 重新缓存 footprint，用于查找
                self.footprints.clear()
                for fp in board.Footprints():
                    self.footprints[fp.GetReference()] = fp

            for ref, comp in components.items():
                if comp.is_fixed:
                    continue
                fp = self.footprints.get(ref)
                if fp is None:
                    continue

                # AABB → 中心点坐标 (mm → nm)
                cx = (comp.bbox[0] + comp.bbox[2]) / 2.0
                cy = (comp.bbox[1] + comp.bbox[3]) / 2.0
                fp.SetPosition(pcbnew.VECTOR2I(
                    pcbnew.FromMM(cx), pcbnew.FromMM(cy)
                ))

                # 旋转
                fp.SetOrientationDegrees(float(comp.rotation * 90))

                # 层
                layer_name = "F.Cu" if comp.layer == 0 else "B.Cu"
                fp.SetLayer(board.GetLayerID(layer_name))

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
