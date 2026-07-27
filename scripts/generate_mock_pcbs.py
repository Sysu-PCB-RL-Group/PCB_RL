"""生成 10 块合成 PCB 的 JSON 数据文件。

输出到 data/模拟/pcb_001.json ~ pcb_010.json。
统一板框 120×80mm，grid=(118,78)，CELL_SIZE=1mm。
（若重新生成后需 clamp 固定元件，运行 fix_board_bounds.py）
"""

import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "模拟")

# ---- 元器件尺寸表 (footprint_w, footprint_h) mm ----
FOOTPRINTS = {
    "R_0603": (1.6, 0.8),
    "R_0805": (2.0, 1.2),
    "C_0603": (1.6, 0.8),
    "C_0805": (2.0, 1.25),
    "C_1206": (3.2, 1.6),
    "LED_0805": (2.0, 1.25),
    "SOT-23": (2.9, 1.3),
    "SOIC-8": (5.0, 4.0),
    "SOIC-14": (8.7, 4.0),
    "TSSOP-20": (6.5, 4.4),
    "QFP-44": (10.0, 10.0),
    "QFP-64": (14.0, 14.0),
    "BGA-100": (15.0, 15.0),
    "CONN-4": (10.0, 8.0),
    "CONN-6": (14.0, 8.0),
    "CONN-10": (25.0, 8.0),
    "Hole": (3.0, 3.0),
}

# ---- 封装类型编码 ----
PKG_TYPE = {
    "R_0603": 0, "R_0805": 0, "C_0603": 0, "C_0805": 0, "C_1206": 0,
    "LED_0805": 0, "SOT-23": 1, "SOIC-8": 1, "SOIC-14": 1,
    "TSSOP-20": 2, "QFP-44": 2, "QFP-64": 2, "BGA-100": 3,
    "CONN-4": 4, "CONN-6": 4, "CONN-10": 4, "Hole": 4,
}


def make_component(cid, footprint, nets, is_fixed=False, bbox=None,
                   pins=None, layer=0, rotation=0):
    """创建单个元件 dict。"""
    w, h = FOOTPRINTS[footprint]
    if bbox is None:
        bbox = [0.0, 0.0, w, h]
    if pins is None:
        # 简化 pin 生成：左边缘 + 右边缘各 1 pin（真实元件 pin 数更多）
        # 仅用于 mock 测试，不影响主训练流程（训练使用真实 KiCad 数据）
        pins = [["1", 0.0, h / 2], [str(max(2, len(nets))), w, h / 2]]
    return {
        "id": cid,
        "footprint": footprint,
        "package_type": PKG_TYPE[footprint],
        "bbox": list(bbox),
        "rotation": rotation,
        "layer": layer,
        "is_fixed": is_fixed,
        "is_placed": is_fixed,
        "nets": list(nets),
        "pins": [list(p) for p in pins],
        "footprint_w": w,
        "footprint_h": h,
    }


def make_net(nid, connections, is_power=False):
    """创建单个 Net dict。"""
    return {
        "net_id": nid,
        "connections": list(connections),
        "is_power": is_power,
    }


# ==================== 10 块 PCB 定义 ====================

def pcb_001():
    """小电源板：80x60mm, SOT-23 稳压 + 滤波电容 + 输入输出连接器"""
    return {
        "name": "模拟PCB-001 小电源板",
        "board_bounds": [[0, 0], [80, 60]],
        "components": [
            make_component("U1", "SOT-23", ["VIN", "GND", "VOUT"],
                           pins=[["1", 0, 0.65], ["2", 2.9, 0.65], ["3", 1.45, 0.65]]),
            make_component("C1", "C_0805", ["VIN", "GND"]),
            make_component("C2", "C_1206", ["VOUT", "GND"]),
            make_component("R1", "R_0805", ["VOUT", "FB"]),
            make_component("D1", "LED_0805", ["VOUT", "GND"]),
            make_component("J1", "CONN-4", ["VIN", "GND"], is_fixed=True,
                           bbox=[2, 20, 12, 28]),
            make_component("J2", "CONN-4", ["VOUT", "GND"], is_fixed=True,
                           bbox=[66, 20, 76, 28]),
        ],
        "nets": [
            make_net("VIN", [("U1", "1"), ("C1", "1"), ("J1", "1")]),
            make_net("GND", [("U1", "2"), ("C1", "2"), ("C2", "2"),
                             ("D1", "2"), ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("VOUT", [("U1", "3"), ("C2", "1"), ("R1", "1"),
                              ("D1", "1"), ("J2", "1")]),
            make_net("FB", [("R1", "2")]),
        ],
    }


def pcb_002():
    """传感器板：100x60mm, SOIC-8 传感器 + 滤波 + MCU"""
    return {
        "name": "模拟PCB-002 传感器板",
        "board_bounds": [[0, 0], [100, 60]],
        "components": [
            make_component("U1", "SOIC-8", ["VCC", "GND", "SDA", "SCL"],
                           pins=[["1", 0, 2], ["4", 0, 0], ["5", 5, 0],
                                 ["8", 5, 2]]),
            make_component("U2", "SOIC-8", ["VCC", "GND", "SDA", "SCL", "INT"],
                           pins=[["1", 0, 2], ["4", 0, 0], ["5", 5, 0],
                                 ["8", 5, 2]]),
            make_component("C1", "C_0603", ["VCC", "GND"]),
            make_component("C2", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["VCC", "SDA"]),
            make_component("R2", "R_0603", ["VCC", "SCL"]),
            make_component("J1", "CONN-6", ["VCC", "GND", "SDA", "SCL", "INT"],
                           is_fixed=True, bbox=[2, 20, 16, 28]),
            make_component("H1", "Hole", [], is_fixed=True,
                           bbox=[95, 2, 98, 5]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("C1", "1"),
                             ("C2", "1"), ("R1", "1"), ("R2", "1"),
                             ("J1", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("C1", "2"),
                             ("C2", "2"), ("J1", "2")], is_power=True),
            make_net("SDA", [("U1", "3"), ("U2", "3"), ("R1", "2"), ("J1", "3")]),
            make_net("SCL", [("U1", "4"), ("U2", "4"), ("R2", "2"), ("J1", "4")]),
            make_net("INT", [("U2", "5"), ("J1", "5")]),
        ],
    }


def pcb_003():
    """数字逻辑板：100x80mm, QFP-44 MCU + 外围"""
    return {
        "name": "模拟PCB-003 数字逻辑板",
        "board_bounds": [[0, 0], [100, 80]],
        "components": [
            make_component("U1", "QFP-44", ["VCC", "GND", "N1", "N2", "N3", "N4"],
                           pins=[("1", 0, 5), ("11", 5, 10), ("22", 10, 5),
                                 ("33", 5, 0), ("44", 10, 5)]),
            make_component("U2", "SOIC-14", ["VCC", "GND", "N1", "N5"]),
            make_component("U3", "SOIC-8", ["VCC", "GND", "N2", "N6"]),
            make_component("C1", "C_0805", ["VCC", "GND"]),
            make_component("C2", "C_0805", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0805", ["N3"]),
            make_component("R2", "R_0805", ["N4"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "N5", "N6"],
                           is_fixed=True, bbox=[2, 30, 27, 38]),
            make_component("J2", "CONN-4", ["VCC", "GND"], is_fixed=True,
                           bbox=[88, 30, 98, 38]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("C1", "1"), ("C2", "1"), ("C3", "1"),
                             ("J1", "1"), ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("C1", "2"), ("C2", "2"), ("C3", "2"),
                             ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("N1", [("U1", "3"), ("U2", "3")]),
            make_net("N2", [("U1", "4"), ("U3", "3")]),
            make_net("N3", [("U1", "5"), ("R1", "1")]),
            make_net("N4", [("U1", "6"), ("R2", "1")]),
            make_net("N5", [("U2", "4"), ("J1", "3")]),
            make_net("N6", [("U3", "4"), ("J1", "4")]),
        ],
    }


def pcb_004():
    """模拟放大板：120x60mm, SOIC + 电阻电容网络"""
    return {
        "name": "模拟PCB-004 模拟放大板",
        "board_bounds": [[0, 0], [120, 60]],
        "components": [
            make_component("U1", "SOIC-8", ["VCC", "GND", "IN+", "IN-", "OUT"],
                           pins=[("1", 0, 2), ("2", 0, 0), ("3", 5, 0),
                                 ("5", 5, 2), ("8", 5, 2)]),
            make_component("U2", "SOIC-8", ["VCC", "GND", "OUT", "FB"]),
            make_component("R1", "R_0805", ["IN+", "VREF"]),
            make_component("R2", "R_0805", ["IN-", "OUT"]),
            make_component("R3", "R_0805", ["IN-", "GND"]),
            make_component("R4", "R_0805", ["OUT", "FB"]),
            make_component("C1", "C_0603", ["VCC", "GND"]),
            make_component("C2", "C_0603", ["VCC", "GND"]),
            make_component("C3", "C_0805", ["IN+", "GND"]),
            make_component("J1", "CONN-6", ["VCC", "GND", "IN+", "OUT"],
                           is_fixed=True, bbox=[2, 20, 16, 28]),
            make_component("J2", "CONN-4", ["VCC", "GND"], is_fixed=True,
                           bbox=[108, 20, 118, 28]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[57, 2, 60, 5]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("C1", "1"),
                             ("C2", "1"), ("J1", "1"), ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("C1", "2"),
                             ("C2", "2"), ("C3", "2"), ("R3", "2"),
                             ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("IN+", [("U1", "3"), ("R1", "1"), ("C3", "1"), ("J1", "3")]),
            make_net("IN-", [("U1", "4"), ("R2", "2"), ("R3", "1")]),
            make_net("OUT", [("U1", "5"), ("U2", "3"), ("R2", "1"),
                             ("R4", "1"), ("J1", "4")]),
            make_net("VREF", [("R1", "2")]),
            make_net("FB", [("U2", "4"), ("R4", "2")]),
        ],
    }


def pcb_005():
    """通信接口板：120x80mm, QFP + 收发器 + 隔离"""
    return {
        "name": "模拟PCB-005 通信接口板",
        "board_bounds": [[0, 0], [120, 80]],
        "components": [
            make_component("U1", "QFP-64", ["VCC", "GND", "TX", "RX", "N1", "N2"],
                           pins=[("1", 0, 7), ("16", 7, 14), ("32", 14, 7),
                                 ("48", 7, 0), ("64", 14, 7)]),
            make_component("U2", "SOIC-14", ["VCC", "GND", "TX", "RX"]),
            make_component("U3", "TSSOP-20", ["VCC", "GND", "N1", "N3"]),
            make_component("C1", "C_0805", ["VCC", "GND"]),
            make_component("C2", "C_0603", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["TX"]),
            make_component("R2", "R_0603", ["RX"]),
            make_component("D1", "LED_0805", ["VCC", "N3"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "TX", "RX", "N3"],
                           is_fixed=True, bbox=[2, 30, 27, 38]),
            make_component("J2", "CONN-6", ["VCC", "GND"], is_fixed=True,
                           bbox=[104, 30, 118, 38]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("C1", "1"), ("C2", "1"), ("C3", "1"),
                             ("D1", "1"), ("J1", "1"), ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("C1", "2"), ("C2", "2"), ("C3", "2"),
                             ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("TX", [("U1", "3"), ("U2", "3"), ("R1", "1"), ("J1", "3")]),
            make_net("RX", [("U1", "4"), ("U2", "4"), ("R2", "1"), ("J1", "4")]),
            make_net("N1", [("U1", "5"), ("U3", "3")]),
            make_net("N2", [("U1", "6")]),
            make_net("N3", [("U3", "4"), ("D1", "2"), ("J1", "5")]),
        ],
    }


def pcb_006():
    """混合信号板：100x100mm, BGA + QFP + 模拟前端"""
    return {
        "name": "模拟PCB-006 混合信号板",
        "board_bounds": [[0, 0], [100, 100]],
        "components": [
            make_component("U1", "BGA-100", ["VCC", "GND", "ADC1", "ADC2",
                                              "N1", "N2", "N3"],
                           pins=[("A1", 0, 0), ("A10", 15, 0),
                                 ("J1", 0, 15), ("J10", 15, 15)]),
            make_component("U2", "QFP-44", ["VCC", "GND", "N1", "N4", "N5"]),
            make_component("U3", "SOIC-8", ["VCC", "GND", "ADC1", "ADC2"]),
            make_component("C1", "C_0805", ["VCC", "GND"]),
            make_component("C2", "C_0603", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("C4", "C_0805", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["N2"]),
            make_component("R2", "R_0603", ["N3"]),
            make_component("D1", "LED_0805", ["VCC", "N5"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "N4", "N5"],
                           is_fixed=True, bbox=[2, 40, 27, 48]),
            make_component("J2", "CONN-6", ["VCC", "GND"], is_fixed=True,
                           bbox=[84, 40, 98, 48]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[2, 2, 5, 5]),
            make_component("H2", "Hole", [], is_fixed=True, bbox=[95, 2, 98, 5]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("C1", "1"), ("C2", "1"), ("C3", "1"),
                             ("C4", "1"), ("D1", "1"), ("J1", "1"),
                             ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("C1", "2"), ("C2", "2"), ("C3", "2"),
                             ("C4", "2"), ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("ADC1", [("U1", "3"), ("U3", "3")]),
            make_net("ADC2", [("U1", "4"), ("U3", "4")]),
            make_net("N1", [("U1", "5"), ("U2", "3")]),
            make_net("N2", [("U1", "6"), ("R1", "1")]),
            make_net("N3", [("U1", "7"), ("R2", "1")]),
            make_net("N4", [("U2", "4"), ("J1", "3")]),
            make_net("N5", [("U2", "5"), ("D1", "2"), ("J1", "4")]),
        ],
    }


def pcb_007():
    """功率驱动板：130x80mm, 大封装功率管 + 散热"""
    return {
        "name": "模拟PCB-007 功率驱动板",
        "board_bounds": [[0, 0], [130, 80]],
        "components": [
            make_component("U1", "SOIC-14", ["VCC", "GND", "PWM1", "PWM2", "PWM3",
                                              "PWM4"]),
            make_component("Q1", "SOT-23", ["PWM1", "GND", "VOUT1"]),
            make_component("Q2", "SOT-23", ["PWM2", "GND", "VOUT2"]),
            make_component("Q3", "SOT-23", ["PWM3", "GND", "VOUT3"]),
            make_component("Q4", "SOT-23", ["PWM4", "GND", "VOUT4"]),
            make_component("C1", "C_1206", ["VCC", "GND"]),
            make_component("C2", "C_0805", ["VOUT1", "GND"]),
            make_component("C3", "C_0805", ["VOUT2", "GND"]),
            make_component("C4", "C_0805", ["VOUT3", "GND"]),
            make_component("C5", "C_0805", ["VOUT4", "GND"]),
            make_component("D1", "LED_0805", ["VCC", "GND"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "PWM1", "PWM2"],
                           is_fixed=True, bbox=[2, 30, 27, 38]),
            make_component("J2", "CONN-10", ["VOUT1", "VOUT2", "VOUT3", "VOUT4",
                                              "GND"], is_fixed=True,
                           bbox=[103, 30, 128, 38]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[2, 2, 5, 5]),
            make_component("H2", "Hole", [], is_fixed=True, bbox=[125, 2, 128, 5]),
            make_component("H3", "Hole", [], is_fixed=True, bbox=[2, 75, 5, 78]),
            make_component("H4", "Hole", [], is_fixed=True, bbox=[125, 75, 128, 78]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("C1", "1"), ("D1", "1"),
                             ("J1", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("Q1", "2"), ("Q2", "2"),
                             ("Q3", "2"), ("Q4", "2"), ("C1", "2"),
                             ("C2", "2"), ("C3", "2"), ("C4", "2"),
                             ("C5", "2"), ("D1", "2"), ("J1", "2"),
                             ("J2", "6")], is_power=True),
            make_net("PWM1", [("U1", "3"), ("Q1", "1"), ("J1", "3")]),
            make_net("PWM2", [("U1", "4"), ("Q2", "1"), ("J1", "4")]),
            make_net("PWM3", [("U1", "5"), ("Q3", "1")]),
            make_net("PWM4", [("U1", "6"), ("Q4", "1")]),
            make_net("VOUT1", [("Q1", "3"), ("C2", "1"), ("J2", "1")]),
            make_net("VOUT2", [("Q2", "3"), ("C3", "1"), ("J2", "2")]),
            make_net("VOUT3", [("Q3", "3"), ("C4", "1"), ("J2", "3")]),
            make_net("VOUT4", [("Q4", "3"), ("C5", "1"), ("J2", "4")]),
        ],
    }


def pcb_008():
    """MCU 核心板：120x90mm, BGA MCU + 时钟 + 复位 + 调试"""
    return {
        "name": "模拟PCB-008 MCU核心板",
        "board_bounds": [[0, 0], [120, 90]],
        "components": [
            make_component("U1", "BGA-100", ["VCC", "GND", "N1", "N2", "N3",
                                              "CLK", "RST", "SWD"]),
            make_component("U2", "SOIC-8", ["VCC", "GND", "N1", "N4"]),
            make_component("U3", "TSSOP-20", ["VCC", "GND", "N2", "N5"]),
            make_component("X1", "C_0805", ["CLK", "GND"]),
            make_component("C1", "C_0603", ["VCC", "GND"]),
            make_component("C2", "C_0603", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["RST"]),
            make_component("R2", "R_0603", ["N3"]),
            make_component("D1", "LED_0805", ["VCC", "N5"]),
            make_component("J1", "CONN-6", ["VCC", "GND", "SWD", "RST"],
                           is_fixed=True, bbox=[2, 35, 16, 43]),
            make_component("J2", "CONN-10", ["VCC", "GND", "N4", "N5"],
                           is_fixed=True, bbox=[100, 30, 125, 38]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[2, 2, 5, 5]),
            make_component("H2", "Hole", [], is_fixed=True, bbox=[115, 2, 118, 5]),
            make_component("H3", "Hole", [], is_fixed=True, bbox=[2, 85, 5, 88]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("C1", "1"), ("C2", "1"), ("C3", "1"),
                             ("D1", "1"), ("J1", "1"), ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("C1", "2"), ("C2", "2"), ("C3", "2"),
                             ("X1", "2"), ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("N1", [("U1", "3"), ("U2", "3")]),
            make_net("N2", [("U1", "4"), ("U3", "3")]),
            make_net("N3", [("U1", "5"), ("R2", "1")]),
            make_net("N4", [("U2", "4"), ("J2", "3")]),
            make_net("N5", [("U3", "4"), ("D1", "2"), ("J2", "4")]),
            make_net("CLK", [("U1", "6"), ("X1", "1")]),
            make_net("RST", [("U1", "7"), ("R1", "1"), ("J1", "4")]),
            make_net("SWD", [("U1", "8"), ("J1", "3")]),
        ],
    }


def pcb_009():
    """接口扩展板：140x80mm, 多 QFP 协同 + 电平转换"""
    return {
        "name": "模拟PCB-009 接口扩展板",
        "board_bounds": [[0, 0], [140, 80]],
        "components": [
            make_component("U1", "QFP-64", ["VCC", "GND", "BUS1", "BUS2", "N1", "N2"]),
            make_component("U2", "QFP-44", ["VCC", "GND", "BUS1", "N3"]),
            make_component("U3", "QFP-44", ["VCC", "GND", "BUS2", "N4"]),
            make_component("U4", "TSSOP-20", ["VCC", "GND", "N1", "N3", "N5"]),
            make_component("U5", "TSSOP-20", ["VCC", "GND", "N2", "N4", "N6"]),
            make_component("C1", "C_0805", ["VCC", "GND"]),
            make_component("C2", "C_0805", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("C4", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["N5"]),
            make_component("R2", "R_0603", ["N6"]),
            make_component("D1", "LED_0805", ["VCC", "N5"]),
            make_component("D2", "LED_0805", ["VCC", "N6"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "BUS1", "BUS2"],
                           is_fixed=True, bbox=[2, 30, 27, 38]),
            make_component("J2", "CONN-10", ["VCC", "GND", "N5", "N6"],
                           is_fixed=True, bbox=[113, 30, 138, 38]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[2, 2, 5, 5]),
            make_component("H2", "Hole", [], is_fixed=True, bbox=[135, 2, 138, 5]),
            make_component("H3", "Hole", [], is_fixed=True, bbox=[2, 75, 5, 78]),
            make_component("H4", "Hole", [], is_fixed=True, bbox=[135, 75, 138, 78]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("U4", "1"), ("U5", "1"), ("C1", "1"),
                             ("C2", "1"), ("C3", "1"), ("C4", "1"),
                             ("D1", "1"), ("D2", "1"), ("J1", "1"),
                             ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("U4", "2"), ("U5", "2"), ("C1", "2"),
                             ("C2", "2"), ("C3", "2"), ("C4", "2"),
                             ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("BUS1", [("U1", "3"), ("U2", "3"), ("J1", "3")]),
            make_net("BUS2", [("U1", "4"), ("U3", "3"), ("J1", "4")]),
            make_net("N1", [("U1", "5"), ("U4", "3")]),
            make_net("N2", [("U1", "6"), ("U5", "3")]),
            make_net("N3", [("U2", "4"), ("U4", "4")]),
            make_net("N4", [("U3", "4"), ("U5", "4")]),
            make_net("N5", [("U4", "5"), ("R1", "1"), ("D1", "2"), ("J2", "3")]),
            make_net("N6", [("U5", "5"), ("R2", "1"), ("D2", "2"), ("J2", "4")]),
        ],
    }


def pcb_010():
    """系统主板：150x100mm, BGA SoC + QFP + 多外设 + 电源管理"""
    return {
        "name": "模拟PCB-010 系统主板",
        "board_bounds": [[0, 0], [150, 100]],
        "components": [
            make_component("U1", "BGA-100", ["VCC", "GND", "D0", "D1", "D2",
                                              "D3", "CLK", "RST"]),
            make_component("U2", "QFP-64", ["VCC", "GND", "D0", "A0", "A1"]),
            make_component("U3", "QFP-44", ["VCC", "GND", "D1", "A0", "A2"]),
            make_component("U4", "SOIC-14", ["VCC", "GND", "D2", "D3"]),
            make_component("U5", "TSSOP-20", ["VCC", "GND", "A1", "A2", "INT"]),
            make_component("C1", "C_1206", ["VCC", "GND"]),
            make_component("C2", "C_0805", ["VCC", "GND"]),
            make_component("C3", "C_0603", ["VCC", "GND"]),
            make_component("C4", "C_0603", ["VCC", "GND"]),
            make_component("C5", "C_0603", ["VCC", "GND"]),
            make_component("R1", "R_0603", ["CLK"]),
            make_component("R2", "R_0603", ["RST"]),
            make_component("R3", "R_0805", ["A0"]),
            make_component("R4", "R_0805", ["A1"]),
            make_component("D1", "LED_0805", ["VCC", "INT"]),
            make_component("D2", "LED_0805", ["VCC", "GND"]),
            make_component("X1", "C_0603", ["CLK", "GND"]),
            make_component("J1", "CONN-10", ["VCC", "GND", "D0", "D1", "D2", "D3"],
                           is_fixed=True, bbox=[2, 40, 27, 48]),
            make_component("J2", "CONN-6", ["VCC", "GND", "INT", "RST"],
                           is_fixed=True, bbox=[134, 40, 148, 48]),
            make_component("H1", "Hole", [], is_fixed=True, bbox=[2, 2, 5, 5]),
            make_component("H2", "Hole", [], is_fixed=True, bbox=[145, 2, 148, 5]),
            make_component("H3", "Hole", [], is_fixed=True, bbox=[2, 95, 5, 98]),
            make_component("H4", "Hole", [], is_fixed=True, bbox=[145, 95, 148, 98]),
        ],
        "nets": [
            make_net("VCC", [("U1", "1"), ("U2", "1"), ("U3", "1"),
                             ("U4", "1"), ("U5", "1"), ("C1", "1"),
                             ("C2", "1"), ("C3", "1"), ("C4", "1"),
                             ("C5", "1"), ("D1", "1"), ("D2", "1"),
                             ("J1", "1"), ("J2", "1")], is_power=True),
            make_net("GND", [("U1", "2"), ("U2", "2"), ("U3", "2"),
                             ("U4", "2"), ("U5", "2"), ("C1", "2"),
                             ("C2", "2"), ("C3", "2"), ("C4", "2"),
                             ("C5", "2"), ("D2", "2"), ("X1", "2"),
                             ("J1", "2"), ("J2", "2")], is_power=True),
            make_net("D0", [("U1", "3"), ("U2", "3"), ("J1", "3")]),
            make_net("D1", [("U1", "4"), ("U3", "3"), ("J1", "4")]),
            make_net("D2", [("U1", "5"), ("U4", "3"), ("J1", "5")]),
            make_net("D3", [("U1", "6"), ("U4", "4"), ("J1", "6")]),
            make_net("A0", [("U2", "4"), ("U3", "4"), ("R3", "1")]),
            make_net("A1", [("U2", "5"), ("U5", "3"), ("R4", "1")]),
            make_net("A2", [("U3", "5"), ("U5", "4")]),
            make_net("CLK", [("U1", "7"), ("R1", "1"), ("X1", "1")]),
            make_net("RST", [("U1", "8"), ("R2", "1"), ("J2", "4")]),
            make_net("INT", [("U5", "5"), ("D1", "2"), ("J2", "3")]),
        ],
    }

# ---- 全部 PCB 定义 ----
ALL_PCBS = [
    pcb_001, pcb_002, pcb_003, pcb_004, pcb_005,
    pcb_006, pcb_007, pcb_008, pcb_009, pcb_010,
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, pcb_fn in enumerate(ALL_PCBS, 1):
        pcb = pcb_fn()
        fname = f"pcb_{i:03d}.json"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(pcb, f, indent=2, ensure_ascii=False)

        n_mov = sum(1 for c in pcb["components"] if not c["is_fixed"])
        n_fix = sum(1 for c in pcb["components"] if c["is_fixed"])
        w = pcb["board_bounds"][1][0]
        h = pcb["board_bounds"][1][1]
        print(f"  [{fname}] {pcb['name']}: {w:.0f}x{h:.0f}mm, "
              f"{len(pcb['components'])} comp ({n_mov}mov + {n_fix}fix), "
              f"{len(pcb['nets'])} nets")

    print(f"\nDone: {len(ALL_PCBS)} PCBs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
