#!/usr/bin/env python3
"""
KiCad 接口测试 —— 第三轮：导出功能（T19-T27）
============================================
测试从零生成 PCB 和模板修改导出的正确性。

测试内容：
  T19: export_board_from_scratch() 返回 True
  T20: 导出的 PCB 包含正确数量的元件
  T21: 元件位置与传入的一致
  T22: 包含板框 (gr_line Edge.Cuts)
  T23: 包含网络定义 (net ...)
  T24: 包含走线 (segment ...)
  T25: 包含过孔 (via ...)
  T26: 导出的 PCB 通过全部格式检查（T1-T6）
  T27: 导出的 PCB 能被 KiCad 打开

使用方法：
  python test_kicad_export.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from kicad_export_helper import export_board_from_scratch
from test_pcbnew import kicad_load_board

import subprocess

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ================================================================
# 配置
# ================================================================
OUTPUT_DIR = r"D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b"
OUTPUT_PCB = os.path.join(OUTPUT_DIR, "测试三_export_test.kicad_pcb")
OUTPUT_PCB2 = os.path.join(OUTPUT_DIR, "测试三_from_scratch.kicad_pcb")

# ================================================================
# 测试数据生成
# ================================================================

def make_test_components():
    """生成测试用的元件数据"""
    return [
        {'ref': 'R1', 'footprint': 'R0603', 'size': (1.6, 0.8),
         'pos': (130.0, 100.0), 'rotation': 0, 'pin_count': 2,
         'pins': [('1', -0.8, 0), ('2', 0.8, 0)]},
        {'ref': 'R2', 'footprint': 'R0603', 'size': (1.6, 0.8),
         'pos': (135.0, 100.0), 'rotation': 90, 'pin_count': 2,
         'pins': [('1', -0.8, 0), ('2', 0.8, 0)]},
        {'ref': 'C1', 'footprint': 'C0603', 'size': (1.6, 0.8),
         'pos': (130.0, 95.0), 'rotation': 0, 'pin_count': 2},
        {'ref': 'U1', 'footprint': 'LQFP-48', 'size': (7.0, 7.0),
         'pos': (145.0, 110.0), 'rotation': 0, 'pin_count': 48},
        {'ref': 'U2', 'footprint': 'SOP-8', 'size': (5.0, 4.0),
         'pos': (160.0, 105.0), 'rotation': 0, 'pin_count': 8},
        {'ref': 'J1', 'footprint': 'HDR-4', 'size': (10.0, 2.5),
         'pos': (120.0, 110.0), 'rotation': 0, 'pin_count': 4},
    ]


def make_test_nets():
    return ['GND', '3V3', 'ADC0', 'SCL', 'SDA', 'UART_TX', 'UART_RX']


def make_test_outline():
    return [(120, 90), (170, 90), (170, 120), (120, 120), (120, 90)]


def make_test_traces():
    return {
        'GND': [((130, 100), (145, 110), 0.2)],
        '3V3': [((135, 100), (145, 110), 0.2)],
        'SCL': [((130, 95), (160, 105), 0.2)],
    }


def make_test_vias():
    return [
        {'pos': (145, 100), 'drill': 0.3, 'diameter': 0.6},
        {'pos': (150, 100), 'drill': 0.3, 'diameter': 0.6},
    ]


# ================================================================
# 格式检查（复用第一轮的检查函数）
# ================================================================

def check_format(filepath):
    """运行 T1-T5 格式检查"""
    checks = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    checks['no_graphics'] = '(graphics' not in content
    checks['no_nets_wrapper'] = len(__import__('re').findall(r'\(nets\s', content)) == 0
    checks['paren_balance'] = content.count('(') == content.count(')')
    checks['starts_correct'] = content.strip().startswith('(kicad_pcb')
    checks['ends_correct'] = content.strip().endswith(')')

    return checks


# ================================================================
# 测试工具
# ================================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.details = []

    def add(self, ok, msg=""):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.details.append((ok, msg))

    def print(self):
        status = "✅" if self.failed == 0 else "❌"
        total = self.passed + self.failed
        print(f"\n  {status} {self.name}: {self.passed}/{total}")
        for ok, msg in self.details:
            icon = "✅" if ok else "❌"
            print(f"     {icon} {msg}")


# ================================================================
# 主测试
# ================================================================

def main():
    print("=" * 60)
    print("  KiCad 接口测试 —— 第三轮：导出功能（T19-T27）")
    print("=" * 60)

    all_results = []

    # ============================================================
    # 测试一：export_board_from_scratch 基础功能
    # ============================================================
    print("\n📤 从零生成测试")
    print("  " + "-" * 50)

    comps = make_test_components()
    nets = make_test_nets()
    outline = make_test_outline()
    traces = make_test_traces()
    vias_list = make_test_vias()

    # T19: 返回值
    r = TestResult("T19: export_board_from_scratch 返回 True")
    result = export_board_from_scratch(OUTPUT_PCB, comps, nets, outline, traces, vias_list)
    r.add(result, f"导出到 {os.path.basename(OUTPUT_PCB)}")
    all_results.append(r)

    if not result or not os.path.exists(OUTPUT_PCB):
        print("  ❌ 导出失败，跳过后续测试")
        for name in ["T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27"]:
            r = TestResult(f"{name}: 跳过")
            r.add(False, "导出失败")
            all_results.append(r)
        # 汇总
        print("\n" + "=" * 60)
        print("  汇总")
        print("=" * 60)
        for r in all_results:
            icon = "✅" if r.failed == 0 else "❌"
            print(f"  {icon} {r.name}: {r.passed}/{r.passed + r.failed}")
        return

    # 读取导出的文件内容
    with open(OUTPUT_PCB, 'r', encoding='utf-8') as f:
        pcb_text = f.read()

    # T20: 元件数量
    import re
    fp_count = len(re.findall(r'^\t\(footprint ', pcb_text, re.MULTILINE))
    r = TestResult(f"T20: 导出元件数量 == {len(comps)}")
    r.add(fp_count == len(comps), f"实际: {fp_count}, 期望: {len(comps)}")
    all_results.append(r)

    # T21: 位置验证（抽检R1和U1）
    r = TestResult("T21: 导出位置与传入一致")
    for comp in [comps[0], comps[3]]:
        ref = comp['ref']
        exp_x, exp_y = comp['pos']
        # 在文件中查找
        pattern = rf'\(footprint "RL:{ref}".*?\(at {exp_x}\.?\d* {exp_y}\.?\d*'
        found = re.search(pattern, pcb_text, re.DOTALL)
        r.add(bool(found), f"{ref}: 期望位置 ({exp_x}, {exp_y}) {'✅ 找到' if found else '❌ 未找到'}")
    all_results.append(r)

    # T22: 包含板框
    gr_lines = len(re.findall(r'\(gr_line .*Edge\.Cuts', pcb_text))
    r = TestResult("T22: 包含板框 (gr_line Edge.Cuts)")
    r.add(gr_lines >= 4, f"gr_line Edge.Cuts: {gr_lines} 条")
    all_results.append(r)

    # T23: 包含网络定义
    net_defs = len(re.findall(r'^\t\(net \d+ "', pcb_text, re.MULTILINE))
    r = TestResult(f"T23: 包含网络定义 ({len(nets)} 个)")
    r.add(net_defs == len(nets), f"net定义: {net_defs}, 期望: {len(nets)}")
    all_results.append(r)

    # T24: 包含走线
    seg_count = len(re.findall(r'^\t\(segment ', pcb_text, re.MULTILINE))
    total_segs = sum(len(s) for s in traces.values())
    r = TestResult(f"T24: 包含走线 ({total_segs} 段)")
    r.add(seg_count == total_segs, f"segment: {seg_count}, 期望: {total_segs}")
    all_results.append(r)

    # T25: 包含过孔
    via_count = len(re.findall(r'^\t\(via ', pcb_text, re.MULTILINE))
    r = TestResult(f"T25: 包含过孔 ({len(vias_list)} 个)")
    r.add(via_count == len(vias_list), f"via: {via_count}, 期望: {len(vias_list)}")
    all_results.append(r)

    # T26: 格式检查
    fmt = check_format(OUTPUT_PCB)
    r = TestResult("T26: 格式检查全部通过")
    for check_name, ok in fmt.items():
        r.add(ok, f"{check_name}: {'✅' if ok else '❌'}")
    all_results.append(r)

    # T27: KiCad 能打开
    r = TestResult("T27: KiCad 能打开导出的 PCB")
    kicad_result = kicad_load_board(OUTPUT_PCB)
    r.add(kicad_result.get('ok', False),
          f"KiCad {kicad_result.get('version','?')} 加载: "
          f"{'✅' if kicad_result.get('ok') else '❌'}")
    if kicad_result.get('footprints'):
        r.add(kicad_result['footprints'] == len(comps),
              f"footprint 数量: {kicad_result['footprints']} (期望 {len(comps)})")
    all_results.append(r)

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)

    total_ok = 0
    total_all = 0
    for r in all_results:
        icon = "✅" if r.failed == 0 else "❌"
        total = r.passed + r.failed
        total_ok += r.passed
        total_all += total
        print(f"  {icon} {r.name}: {r.passed}/{total}")

    print(f"\n  {'✅' if total_ok == total_all else '❌'} 总计: {total_ok}/{total_all} 通过")
    print(f"\n  输出文件:")
    print(f"    {OUTPUT_PCB}")
    print()


if __name__ == '__main__':
    main()
