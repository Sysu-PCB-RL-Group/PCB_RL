"""PCB RL 环境专用工具类（全静态，无 KiCad 依赖）。

所有方法为 @staticmethod，数据由 PCBLayoutEnv 在调用时传入。
负责约束检测和 reward 各分项计算。
"""
import math
from typing import List, Tuple, Dict

import numpy as np


class PCBUtils:
    """PCB RL 环境工具类（纯计算层）"""

    # ---- 去耦电容识别阈值 ----
    _IC_MIN_PACKAGE_TYPE: int = 2   # SMD_IC=2, TH_IC=3, CONN=4 均视为 IC
    _CAP_MAX_PACKAGE_TYPE: int = 1  # 被动元件 (SMD_PASSIVE=0, TH_PASSIVE=1)

    # ========== C2：AABB 碰撞检测 ==========

    @staticmethod
    def bbox_overlap_penalty(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        spacing_threshold: float,
    ) -> Tuple[float, float]:
        """计算两个 AABB 之间的碰撞惩罚。

        同时返回硬重叠面积和软间距惩罚。

        Args:
            a: (x1, y1, x2, y2) AABB 包围盒
            b: (x1, y1, x2, y2) AABB 包围盒
            spacing_threshold: 分类间距阈值 (mm)，由调用方从两个元件的
                               package_type 查表后取 max 传入

        Returns:
            (hard_overlap, soft_spacing):
            hard_overlap: 物理交叉面积 (mm²)，>0 表示硬重叠
            soft_spacing: 间隙 < spacing_threshold 时的连续惩罚值，0 表示不惩罚
        """
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])

        if dx > 0.0 and dy > 0.0:
            # 硬重叠：返回交叉面积
            return (dx * dy, 0.0)

        # 计算间隙和共享边长度
        if dx > 0.0 >= dy:
            gap, edge = -dy, dx
        elif dy > 0.0 >= dx:
            gap, edge = -dx, dy
        else:
            # 角对角：零惩罚
            return (0.0, 0.0)

        if gap >= spacing_threshold:
            return (0.0, 0.0)

        # 软间距连续惩罚：(threshold - gap)² × edge
        return (0.0, (spacing_threshold - gap) ** 2 * edge)

    # ========== C3：板框约束 ==========

    @staticmethod
    def out_of_bounds_area(
        bbox: Tuple[float, float, float, float],
        valid_area: Tuple[float, float, float, float],
    ) -> float:
        """计算 bbox 超出 valid_area 的面积 (mm²)。

        Args:
            bbox: (x1, y1, x2, y2) 元件 AABB
            valid_area: (x_min, y_min, x_max, y_max) 板框内缩后的有效区域

        Returns:
            超出面积 (mm²)，完全在内部时返回 0.0
        """
        bxmin, bymin, bxmax, bymax = bbox
        oxmin, oymin, oxmax, oymax = valid_area

        bbox_area = max(0.0, bxmax - bxmin) * max(0.0, bymax - bymin)
        if bbox_area == 0.0:
            return 0.0

        ix = max(0.0, min(bxmax, oxmax) - max(bxmin, oxmin))
        iy = max(0.0, min(bymax, oymax) - max(bymin, oymin))

        return max(0.0, bbox_area - ix * iy)

    @staticmethod
    def clamp_to_board(
        x1: float, y1: float, x2: float, y2: float,
        valid_area: Tuple[float, float, float, float],
        margin: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """推理阶段：将 AABB clamp 至 valid_area 内（保留 margin 裕度）。

        保持宽高不变，仅平移。

        Args:
            x1: AABB 左边界 (mm)
            y1: AABB 下边界 (mm)
            x2: AABB 右边界 (mm)
            y2: AABB 上边界 (mm)
            valid_area: (x_min, y_min, x_max, y_max) 板框内缩后的有效区域
            margin: 板框内留白裕度 (mm)，默认 0.0

        Returns:
            (x1, y1, x2, y2) clamp 后的 AABB
        """
        w, h = x2 - x1, y2 - y1
        oxmin, oymin, oxmax, oymax = valid_area

        x1 = max(oxmin + margin, min(x1, oxmax - w - margin))
        y1 = max(oymin + margin, min(y1, oymax - h - margin))
        return (x1, y1, x1 + w, y1 + h)

    # ========== C6：Net 聚合度 ==========

    @staticmethod
    def net_aggregation(
        comp_bboxes: List[Tuple[float, float, float, float]],
    ) -> float:
        """计算同一 net 内所有已放置元件的聚合度。

        Args:
            comp_bboxes: 同一 net 内已放置元件的 AABB 列表

        Returns:
            各元件中心到 net 中心的平均距离 (mm)，越小越好。
            < 2 个元件时返回 0.0。
        """
        if len(comp_bboxes) < 2:
            return 0.0

        # 从 AABB 推导中心点
        cxs = [(b[0] + b[2]) / 2.0 for b in comp_bboxes]
        cys = [(b[1] + b[3]) / 2.0 for b in comp_bboxes]
        cx_avg = sum(cxs) / len(cxs)
        cy_avg = sum(cys) / len(cys)
        dists = [math.sqrt((x - cx_avg) ** 2 + (y - cy_avg) ** 2)
                 for x, y in zip(cxs, cys)]
        return sum(dists) / len(dists)

    # ========== C6：去耦电容 ==========

    @staticmethod
    def identify_decoupling_pairs(
        nets: Dict[str, object],       # net_id -> Net (has .is_power, .connections)
        components: Dict[str, object], # comp_id -> Component
    ) -> Dict[str, List[str]]:
        """识别去耦电容配对关系。

        遍历 netlist → IC 含电源引脚 → 同网络查找 MLCC 电容。

        Args:
            nets: net_id → Net 映射
            components: comp_id → Component 映射

        Returns:
            {ic_id: [cap_id, ...]} 每个 IC 对应的去耦电容列表
        """
        decoupling_map: Dict[str, List[str]] = {}

        for net_id, net in nets.items():
            if not net.is_power:
                continue

            # 该网络上所有元件
            net_comps = set()
            for conn_ref, _ in net.connections:
                net_comps.add(conn_ref)

            # 找 IC（package_type >= _IC_MIN_PACKAGE_TYPE）
            ics = [r for r in net_comps
                   if r in components
                   and components[r].package_type >= PCBUtils._IC_MIN_PACKAGE_TYPE]

            for ic_id in ics:
                for candidate_id in net_comps:
                    if candidate_id == ic_id:
                        continue
                    if candidate_id not in components:
                        continue
                    comp = components[candidate_id]
                    # MLCC 电容 heuristic: 被动元件 + footprint 名含 "C"
                    if (comp.package_type <= PCBUtils._CAP_MAX_PACKAGE_TYPE
                            and "C" in comp.footprint.upper()):
                        decoupling_map.setdefault(ic_id, []).append(candidate_id)

        return decoupling_map

    @staticmethod
    def min_pin_to_cap_distance(
        ic: object,          # Component with .pins, .center
        cap: object,         # Component with .center
        power_net_ids: List[str] = None,
    ) -> float:
        """计算电容到 IC 电源引脚组的最小欧氏距离 (mm)。

        Args:
            ic: IC 元件 (Component)
            cap: 电容元件 (Component)
            power_net_ids: IC 的电源网络 ID 列表（可选，默认使用 ic.nets）

        Returns:
            最小距离 (mm)。没有引脚数据时退化为 cap.center 到 ic.center 的距离。
        """
        ic_cx, ic_cy = ic.center
        cap_cx, cap_cy = cap.center

        # 收集 IC 所有引脚的绝对坐标。
        # 必须用 ic.pin_abs_positions()（相对当前 bbox 中心 + 该【层+旋转】的预计算
        # 偏移），不能用 bbox[0]+rel_x——元件旋转后 bbox 尺寸/位置已变，旧公式会失真；
        # 元件在背面时正/背面引脚布局还不同，只看旋转也会算错。
        # TODO: 待 kicad_utils 支持 pin→net 映射后，改为仅过滤电源网络引脚
        pin_abs_positions = [(abs_x, abs_y)
                             for _pin_num, abs_x, abs_y in ic.pin_abs_positions()]

        if not pin_abs_positions:
            # 无法获取引脚坐标 → 退化为中心点距离
            return math.sqrt((cap_cx - ic_cx) ** 2 + (cap_cy - ic_cy) ** 2)

        return min(
            math.sqrt((cap_cx - px) ** 2 + (cap_cy - py) ** 2)
            for px, py in pin_abs_positions
        )

    # ========== S1：拥塞密度 ==========

    @staticmethod
    def congestion_std(
        placed_components: List[object],  # List[Component] with .nets, .bbox
        nets: Dict[str, object],          # net_id -> Net with .connections
        valid_area: Tuple[float, float, float, float],
        cell_size: float,
    ) -> float:
        """统计拥塞密度网格的标准差。

        将 valid_area 划分为 cell_size × cell_size 网格，
        统计每个格子中涉及的 net 数量（通过 AABB 与格子交集判断），
        返回所有格子 net 计数的标准差。越小越均匀。

        Args:
            placed_components: 已放置元件列表
            nets: 全局网络列表
            valid_area: 有效区域 (x_min, y_min, x_max, y_max) mm
            cell_size: 网格单元大小 (mm)，默认 CONGESTION_CELL_SIZE = 2.0

        Returns:
            net 计数的标准差。无已放置元件时返回 0.0。
        """
        if not placed_components:
            return 0.0

        oxmin, oymin, oxmax, oymax = valid_area
        cols = max(1, int((oxmax - oxmin) / cell_size))
        rows = max(1, int((oymax - oymin) / cell_size))

        # 每个格子的 net 集合
        cell_nets: List[set] = [set() for _ in range(cols * rows)]

        for comp in placed_components:
            bx1, by1, bx2, by2 = comp.bbox
            # 元件覆盖的格子范围
            col_start = max(0, int((bx1 - oxmin) / cell_size))
            col_end = min(cols - 1, int((bx2 - oxmin) / cell_size))
            row_start = max(0, int((by1 - oymin) / cell_size))
            row_end = min(rows - 1, int((by2 - oymin) / cell_size))

            for col in range(col_start, col_end + 1):
                for row in range(row_start, row_end + 1):
                    idx = row * cols + col
                    for net_id in comp.nets:
                        cell_nets[idx].add(net_id)

        counts = [len(s) for s in cell_nets]
        return float(np.std(counts)) if counts else 0.0

    # ========== S3：大元件排名 ==========

    @staticmethod
    def build_area_ranks(
        movable_list: List[object],  # List[Component] with .footprint_area, .id
    ) -> Dict[str, int]:
        """按 footprint_area 降序排列 movable 元件。

        Args:
            movable_list: 可移动元件列表

        Returns:
            {comp_id: rank}，rank 从 1 开始（最大元件 rank=1）
        """
        sorted_comps = sorted(
            movable_list,
            key=lambda c: c.footprint_area,
            reverse=True,
        )
        return {comp.id: rank + 1 for rank, comp in enumerate(sorted_comps)}

    # ========== 禁止区域 ==========

    @staticmethod
    def check_keepout(
        bbox: Tuple[float, float, float, float],
        keepout_zones: List[Tuple[float, float, float, float]],
    ) -> bool:
        """检查 bbox 是否与任何 keepout zone 有交集。

        Args:
            bbox: (x1, y1, x2, y2) 元件 AABB
            keepout_zones: List[(x1, y1, x2, y2)]

        Returns:
            True 表示违规（进入禁止区域）
        """
        for kz in keepout_zones:
            dx = min(bbox[2], kz[2]) - max(bbox[0], kz[0])
            dy = min(bbox[3], kz[3]) - max(bbox[1], kz[1])
            if dx > 0.0 and dy > 0.0:
                return True
        return False
