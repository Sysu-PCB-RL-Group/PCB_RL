import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
from kicad_utils import KiCadUtils

class PCBUtils:
    """PCB RL环境专用工具类（无JSON配置依赖）"""
    
    def __init__(self):
        """初始化（关联KiCad底层工具）"""
        self.kicad = KiCadUtils()
        self.min_spacing = 0.2  # 默认最小元器件间距 mm
        self.forbidden_areas: List[Tuple[float, float, float, float]] = []  # (x1,y1,x2,y2)
    
    def load_pcb(self, pcb_path: str):
        """加载PCB文件并缓存核心数据"""
        self.kicad.load_board(pcb_path)

    # -------------------- 元器件相关 --------------------
    def get_component_state(self, ref: str) -> Dict[str, Any]:
        """获取元器件状态（RL环境用）"""
        c = self.kicad.get_component(ref)
        if not c:
            return {}
        
        return {
            "id": ref,
            "type": ref[0],  # R/C/L/U等类型
            "position": c.position,
            "rotation": c.rotation,
            "layer": 1 if c.layer == "F.Cu" else 0,  # 数值化图层
            "fixed": 1 if c.is_fixed else 0,         # 数值化固定状态
            "pins": len(c.pins)
        }
    
    def get_all_component_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有元器件状态"""
        states = {}
        for ref in self.kicad.components:
            states[ref] = self.get_component_state(ref)
        return states
    
    def check_layout_constraints(self, component_positions: Optional[Dict[str, Tuple[float, float]]] = None) -> List[str]:
        """检查布局约束（间距/板框/禁止区域）"""
        violations = []
        # 使用传入的位置或缓存位置
        positions = component_positions or {ref: c.position for ref, c in self.kicad.components.items()}
        
        # 1. 检查元器件间距
        refs = list(positions.keys())
        for i, ref1 in enumerate(refs):
            for ref2 in refs[i+1:]:
                if not self.kicad.check_component_spacing(ref1, ref2, self.min_spacing):
                    violations.append(f"spacing_violation: {ref1}-{ref2}")
        
        # 2. 检查是否在板框内
        for ref, pos in positions.items():
            min_x, min_y = self.kicad.get_board_bounds()[0]
            max_x, max_y = self.kicad.get_board_bounds()[1]
            if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
                violations.append(f"out_of_border: {ref}")
        
        # 3. 检查禁止区域
        for ref, pos in positions.items():
            for (x1, y1, x2, y2) in self.forbidden_areas:
                if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2:
                    violations.append(f"forbidden_area: {ref}")
        
        return violations
    
    def calculate_layout_compactness(self, component_positions: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
        """计算布局紧凑度（0-1）"""
        positions = component_positions or {ref: c.position for ref, c in self.kicad.components.items()}
        pos_list = np.array(list(positions.values()))
        if len(pos_list) == 0:
            return 0.0
        
        # 计算中心位置
        center = np.mean(pos_list, axis=0)
        # 计算平均到中心的距离
        avg_dist = np.mean(np.linalg.norm(pos_list - center, axis=1))
        # 计算最大可能距离（板框对角线）
        (min_x, min_y), (max_x, max_y) = self.kicad.get_board_bounds()
        max_dist = np.linalg.norm([max_x - min_x, max_y - min_y])
        
        # 紧凑度 = 1 - (平均距离/最大距离)
        compactness = 1 - (avg_dist / max_dist) if max_dist > 0 else 0.0
        return max(0.0, min(1.0, compactness))
    
    # -------------------- 网络/布线相关 --------------------
    def get_net_state(self, net_id: str) -> Dict[str, Any]:
        """获取网络状态（RL环境用）"""
        net = self.kicad.get_net(net_id)
        if not net:
            return {}
        
        return {
            "id": net_id,
            "connections": net.connections,
            "routed": 1 if net.is_routed else 0,
            "trace_width": net.trace_width,
            "unrouted_pins": len(net.connections)  # 简化：未布线引脚数
        }
    
    def get_all_net_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有网络状态"""
        states = {}
        for net in self.kicad.get_netlist():
            states[net.net_id] = self.get_net_state(net.net_id)
        return states
    
    def check_routing_violation(self, net_id: str, path: List[Tuple[float, float]]) -> List[str]:
        """检查布线违规（简化版）"""
        violations = []
        # 1. 检查布线是否超出板框
        (min_x, min_y), (max_x, max_y) = self.kicad.get_board_bounds()
        for (x, y) in path:
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                violations.append(f"routing_out_of_border: {net_id} ({x},{y})")
                break
        
        # 2. 检查布线与元器件间距（简化）
        comp_pos = {ref: c.position for ref, c in self.kicad.components.items()}
        for (x, y) in path:
            for ref, (cx, cy) in comp_pos.items():
                dist = np.linalg.norm([x - cx, y - cy])
                if dist < self.min_spacing / 2:
                    violations.append(f"routing_comp_collision: {net_id}-{ref}")
                    break
        
        return violations
    
    # -------------------- 板框相关 --------------------
    def get_board_info(self) -> Dict[str, Any]:
        """获取电路板核心信息"""
        bounds = self.kicad.get_board_bounds()
        borders = self.kicad.get_board_border()
        
        return {
            "bounds": bounds,
            "border_count": len(borders),
            "width": bounds[1][0] - bounds[0][0],
            "height": bounds[1][1] - bounds[0][1],
            "area": (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
        }
