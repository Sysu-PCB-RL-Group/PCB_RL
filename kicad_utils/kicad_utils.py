import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

KICAD_PATH = os.path.normpath(r"D:\Program Files\KiCad\9.0") 

os.add_dll_directory(os.path.join(KICAD_PATH, "bin"))
sys.path.insert(0, os.path.join(KICAD_PATH, "bin", "Lib", "site-packages"))
sys.path.insert(0, os.path.join(KICAD_PATH, "bin"))

import pcbnew
KICAD_AVAILABLE = True

# ===================== 数据结构定义 =====================
@dataclass
class Component:
    """元器件核心数据结构"""
    reference: str            # 参考编号（如R1、U2）
    value: str                # 器件值（如10k、0805）
    footprint: str            # 封装名称
    position: Tuple[float, float]  # 位置 (x, y) mm
    rotation: float           # 旋转角度（度）
    layer: str                # 所在层（F.Cu/B.Cu）
    pad_count: int            # 焊盘数量
    is_fixed: bool            # 是否固定（不可移动）
    pins: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # 引脚编号: 坐标

@dataclass
class BoardBorder:
    """电路板框数据结构"""
    border_type: str          # 边框类型（Line/Circle/Arc/Polygon）
    vertices: List[Tuple[float, float]]  # 顶点坐标 mm
    layer: str = "Edge.Cuts"  # 默认板框层

@dataclass
class Net:
    """网络数据结构"""
    net_id: str               # 网络名称（如VCC、GND）
    connections: List[Tuple[str, str]]  # (器件ref, 引脚号) 列表
    is_routed: bool = False   # 是否已布线
    trace_width: float = 0.2  # 推荐线宽 mm

# ===================== KiCad 核心工具类 =====================
class KiCadUtils:
    """KiCad PCB底层操作工具类"""
    
    def __init__(self):
        # 初始化缓存
        self.board: Optional[pcbnew.BOARD] = None
        self.footprints: Dict[str, pcbnew.FOOTPRINT] = {}  # ref -> footprint
        
        self.components: Dict[str, Component] = {}  # ref -> Component
        self.borders: List[BoardBorder] = []        # 板框缓存
        self.nets: Dict[str, Net] = {}              # net_id -> Net
        self.board_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0,0), (0,0))  # 板框边界
    
    # -------------------- 基础加载/缓存 --------------------
    def load_board(self, pcb_path: str) -> bool:
        """加载PCB文件并缓存所有核心数据"""
        try:
            # 加载PCB
            self.board = pcbnew.LoadBoard(pcb_path)
            print(f"✅ load pcb file: {pcb_path}")
            
            # 清空旧缓存
            self.components.clear()
            self.borders.clear()
            self.nets.clear()
            
            # 缓存核心数据
            self._cache_footprints()

            self._cache_components()
            self._cache_board_border()
            self._cache_nets()
            self._calc_board_bounds()
            
            # 打印加载日志
            self._print_load_log()
            return True
        except Exception as e:
            print(f"❌ fail to load pcb file: {e}")
            return False
        
    def _cache_footprints(self):
        """缓存所有元器件信息"""
        self.footprints.clear()
        for fp in self.board.Footprints():
            ref = fp.GetReference()
            self.footprints[ref] = fp

    def _cache_components(self):
        """缓存所有元器件信息（含固定属性识别）"""
        for ref in self.footprints.keys():
            fp = self.footprints[ref]

            reference = fp.GetReference()
            # 基础信息
            pos_nm = fp.GetPosition()
            pos_mm = (pcbnew.ToMM(pos_nm.x), pcbnew.ToMM(pos_nm.y))
            rotation = fp.GetOrientation() / 10.0  # KiCad内部单位：0.1度
            layer = pcbnew.LayerName(fp.GetLayer())
            is_fixed = fp.IsLocked()  # 识别是否固定（Locked属性）
            
            # 引脚信息
            pins = {}
            for pad in fp.Pads():
                pad_pos_nm = pad.GetPosition()
                pad_pos_mm = (pcbnew.ToMM(pad_pos_nm.x), pcbnew.ToMM(pad_pos_nm.y))
                pins[pad.GetNumber()] = pad_pos_mm
            
            # 封装为Component对象
            self.components[reference] = Component(
                reference=reference,
                value=fp.GetValue(),
                footprint=fp.GetFPID().GetLibItemName(),
                position=pos_mm,
                rotation=rotation,
                layer=layer,
                pad_count=len(pins),
                is_fixed=is_fixed,
                pins=pins
            )
    
    def _cache_board_border(self):
        """缓存电路板框（参考kicad_draw逻辑）"""
        edge_cuts_id = self.board.GetLayerID("Edge.Cuts")
        if edge_cuts_id == -1:
            print("⚠️ layer 'Edge.Cuts' not found")
            return
        
        # 遍历所有绘图元素
        for drawing in self.board.GetDrawings():
            if drawing.GetLayer() != edge_cuts_id:
                continue
            
            if isinstance(drawing, pcbnew.PCB_SHAPE):
                shape_type = drawing.GetShapeStr()
                vertices = []
                
                # 线段
                if shape_type == "Line":
                    start = (pcbnew.ToMM(drawing.GetStart().x), pcbnew.ToMM(drawing.GetStart().y))
                    end = (pcbnew.ToMM(drawing.GetEnd().x), pcbnew.ToMM(drawing.GetEnd().y))
                    vertices = [start, end]
                
                # 圆形
                elif shape_type == "Circle":
                    center = (pcbnew.ToMM(drawing.GetCenter().x), pcbnew.ToMM(drawing.GetCenter().y))
                    radius = pcbnew.ToMM(drawing.GetRadius())
                    # 简化为正方形顶点（便于边界计算）
                    vertices = [
                        (center[0]-radius, center[1]-radius),
                        (center[0]+radius, center[1]-radius),
                        (center[0]+radius, center[1]+radius),
                        (center[0]-radius, center[1]+radius)
                    ]
                
                # 圆弧（简化为起止点）
                elif shape_type == "Arc":
                    start = (pcbnew.ToMM(drawing.GetStart().x), pcbnew.ToMM(drawing.GetStart().y))
                    end = (pcbnew.ToMM(drawing.GetEnd().x), pcbnew.ToMM(drawing.GetEnd().y))
                    vertices = [start, end]
                
                # 多边形
                elif shape_type == "Polygon":
                    poly = drawing.GetPolyShape()
                    if poly.OutlineCount() > 0:
                        outline = poly.Outline(0)
                        for i in range(outline.PointCount()):
                            pt = outline.CPoint(i)
                            vertices.append((pcbnew.ToMM(pt.x), pcbnew.ToMM(pt.y)))
                
                # 添加到板框缓存
                self.borders.append(BoardBorder(
                    border_type=shape_type,
                    vertices=vertices
                ))
    
    def _cache_nets(self):
        """缓存网络连接信息（适配KiCad 9.0 API）"""
        for net in self.board.GetNetsByName().values():
            net_name = net.GetNetname()
            if not net_name:
                continue
            
            connections = []
            target_net_code = net.GetNetCode()
            for footprint in self.board.Footprints():
                for pad in footprint.Pads():
                    if pad.GetNetCode() == target_net_code:
                        connections.append((footprint.GetReference(), pad.GetNumber()))
            
            # 创建Net对象
            self.nets[net_name] = Net(
                net_id=net_name,
                connections=connections,
                is_routed=False,
                trace_width=0.2
            )
    
    def _calc_board_bounds(self):
        """计算电路板框边界（min_x, min_y）, (max_x, max_y)"""
        if not self.borders:
            # 无板框时用元器件边界替代
            comp_pos = [c.pos for c in self.components.values()]
            if not comp_pos:
                self.board_bounds = ((0,0), (100, 80))  # 默认边界
                return
            xs = [p[0] for p in comp_pos]
            ys = [p[1] for p in comp_pos]
            self.board_bounds = ((min(xs), min(ys)), (max(xs), max(ys)))
        else:
            # 从板框计算边界
            all_vertices = []
            for border in self.borders:
                all_vertices.extend(border.vertices)
            xs = [p[0] for p in all_vertices]
            ys = [p[1] for p in all_vertices]
            self.board_bounds = ((min(xs), min(ys)), (max(xs), max(ys)))
    
    # -------------------- 辅助工具函数 --------------------
    
    def _print_load_log(self):
        """打印加载日志（测试用）"""
        print("="*60)
        print(f"📊 PCB加载统计:")
        print(f"  - 元器件数量: {len(self.components)} (固定: {sum(1 for c in self.components.values() if c.is_fixed)})")
        print(f"  - 板框元素数量: {len(self.borders)}")
        print(f"  - 网络数量: {len(self.nets)}")
        print(f"  - 电路板边界: {self.board_bounds[0]} ~ {self.board_bounds[1]} mm")
        print("="*60)
    
    # -------------------- 对外核心接口 --------------------
    def get_component_list(self) -> List[Component]:
        """获取所有元器件列表"""
        return list(self.components.values())
    
    def get_component(self, ref: str) -> Optional[Component]:
        """通过参考编号获取单个元器件"""
        return self.components.get(ref)
    
    def get_board_border(self) -> List[BoardBorder]:
        """获取电路板框列表"""
        return self.borders.copy()
    
    def get_board_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """获取电路板边界"""
        return self.board_bounds
    
    def get_netlist(self) -> List[Net]:
        """获取所有网络列表"""
        return list(self.nets.values())
    
    def get_net(self, net_id: str) -> Optional[Net]:
        """通过网络名称获取网络"""
        return self.nets.get(net_id)
    
    def check_component_spacing(self, ref1: str, ref2: str, min_spacing: float = 0.2) -> bool:
        """检查两个元器件间距是否合规"""
        c1 = self.get_component(ref1)
        c2 = self.get_component(ref2)
        if not c1 or not c2:
            return False
        dist = np.linalg.norm(np.array(c1.position) - np.array(c2.position))
        return dist >= min_spacing
