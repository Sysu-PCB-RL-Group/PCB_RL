#!/usr/bin/env python3
"""
KiCad 接口测试 —— 第四轮：数据完整性 Round-trip（T28-T30）
========================================================
测试数据从文件→内存→文件→内存 不丢失。

测试内容：
  T28: 原版PCB → 导出 → 再导入，元件数量不变
  T29: 原版PCB → 导出 → 再导入，网络数量不变
  T30: 手动构建 → 导出 → 再导入，数据一致

使用方法：
  python test_kicad_roundtrip.py
"""

import os
import sys
import json
import re

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
DATA_DIR = r"D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b"
ORIGINAL_PCB = os.path.join(DATA_DIR, "90i0-quan-cai-shi-zhong-ac67228b.kicad_pcb")
OUTPUT_DIR = DATA_DIR
ROUNDTRIP_PCB = os.path.join(OUTPUT_DIR, "测试四_roundtrip.kicad_pcb")

KICAD_PYTHON = r'D:/pcb大模型/kicad/bin/python.exe'
SITE_PKG = r'D:/pcb大模型/kicad/bin/Lib/site-packages'

# ================================================================
# KiCad 数据提取脚本
# ================================================================

EXTRACT_SCRIPT = r"""
import sys, json
sys.path.append({SITE_PKG!r})
import pcbnew

board = pcbnew.LoadBoard({path!r})

# 元件
components = []
for fp in board.Footprints():
    try:
        ref = fp.GetReference()
        fp_name = fp.GetFPID().GetLibItemName() or ""
        bbox = fp.GetBoundingBox()
        components.append({{
            'ref': str(ref),
            'footprint': str(fp_name),
            'bbox': [round(pcbnew.ToMM(bbox.GetLeft()), 3),
                     round(pcbnew.ToMM(bbox.GetTop()), 3),
                     round(pcbnew.ToMM(bbox.GetRight()), 3),
                     round(pcbnew.ToMM(bbox.GetBottom()), 3)],
            'pad_count': len(fp.Pads()),
        }})
    except:
        pass

# 网络
nets = []
for net in board.GetNetsByName().values():
    try:
        name = net.GetNetname()
        if name:
            nets.append(str(name))
    except:
        pass

# 走线
segments = 0
for item in board.GetTracks():
    try:
        if item.GetClass() == 'PCB_TRACK':
            segments += 1
    except:
        pass

# 过孔
vias = 0
for item in board.GetTracks():
    try:
        if item.GetClass() == 'PCB_VIA':
            vias += 1
    except:
        pass

print(json.dumps({{
    'components_count': len(components),
    'components': components[:5],
    'nets_count': len(nets),
    'nets': nets,
    'segments': segments,
    'vias': vias,
}}, ensure_ascii=True))
"""


def extract_board_data(pcb_path):
    """用 KiCad Python 提取 PCB 数据"""
    script = EXTRACT_SCRIPT.format(SITE_PKG=SITE_PKG, path=pcb_path)
    proc = subprocess.run([KICAD_PYTHON, '-c', script],
                         capture_output=True, text=True, timeout=60)
    for line in proc.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ================================================================
# 手动构建数据（用于 T30）
# ================================================================

def make_components():
    return [
        {'ref': 'R1', 'footprint': 'R0603', 'size': (1.6, 0.8),
         'pos': (130.0, 100.0), 'rotation': 0, 'pin_count': 2},
        {'ref': 'C1', 'footprint': 'C0603', 'size': (1.6, 0.8),
         'pos': (135.0, 95.0), 'rotation': 90, 'pin_count': 2},
        {'ref': 'U1', 'footprint': 'LQFP-48', 'size': (7.0, 7.0),
         'pos': (148.0, 108.0), 'rotation': 0, 'pin_count': 48},
    ]


def make_nets():
    return ['GND', '3V3', 'ADC0', 'SCL', 'SDA']


def make_outline():
    return [(120, 85), (175, 85), (175, 125), (120, 125), (120, 85)]


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
    print("  KiCad 接口测试 —— 第四轮：Round-trip（T28-T30）")
    print("=" * 60)

    all_results = []

    # ============================================================
    # T28+T29: 原版PCB → 导出 → 再导入
    # ============================================================
    print("\n🔄 Round-trip 测试（原版PCB）")
    print("  " + "-" * 50)

    # 第一步：提取原版数据
    print("  [1/3] 提取原版数据...")
    orig_data = extract_board_data(ORIGINAL_PCB)
    if not orig_data:
        print("  ❌ 提取原版数据失败")
        for name in ["T28", "T29", "T30"]:
            r = TestResult(f"{name}: 跳过")
            r.add(False, "数据提取失败")
            all_results.append(r)
        return

    print(f"  原版: {orig_data['components_count']} 元件, "
          f"{orig_data['nets_count']} 网络, "
          f"{orig_data['segments']} 走线, "
          f"{orig_data['vias']} 过孔")

    # 第二步：用提取的数据构建导出
    # 从原版 PCB 提取元件位置
    extract_positions_script = f"""
import sys, json
sys.path.append({SITE_PKG!r})
import pcbnew
board = pcbnew.LoadBoard({ORIGINAL_PCB!r})

comps = []
for fp in board.Footprints():
    try:
        ref = fp.GetReference()
        pos = fp.GetPosition()
        rot = fp.GetOrientationDegrees()
        fp_name = fp.GetFPID().GetLibItemName() or ""
        bbox = fp.GetBoundingBox()
        pin_count = len(fp.Pads())
        w = pcbnew.ToMM(bbox.GetWidth())
        h = pcbnew.ToMM(bbox.GetHeight())
        if w < 0.1: w = 3.0
        if h < 0.1: h = 3.0
        comps.append({{
            'ref': str(ref),
            'footprint': str(fp_name),
            'size': [round(w, 3), round(h, 3)],
            'pos': [round(pcbnew.ToMM(pos.x), 3), round(pcbnew.ToMM(pos.y), 3)],
            'rotation': float(rot),
            'pin_count': pin_count,
        }})
    except:
        pass

print(json.dumps({{'components': comps}}))
"""
    proc = subprocess.run([KICAD_PYTHON, '-c', extract_positions_script],
                         capture_output=True, text=True, timeout=60)
    pos_data = None
    for line in proc.stdout.strip().split('\n'):
        if line.strip().startswith('{'):
            pos_data = json.loads(line)
            break

    if not pos_data:
        print("  ❌ 提取元件位置失败")
        for name in ["T28", "T29"]:
            r = TestResult(f"{name}: 跳过")
            r.add(False, "位置提取失败")
            all_results.append(r)
    else:
        # 用提取的数据重建
        export_comps = []
        for c in pos_data['components']:
            export_comps.append({
                'ref': c['ref'],
                'footprint': c['footprint'],
                'size': tuple(c['size']),
                'pos': tuple(c['pos']),
                'rotation': c['rotation'],
                'pin_count': c['pin_count'],
            })

        net_names = [{'name': n} for n in orig_data['nets']]

        # 提取板框
        outline_script = f"""
import sys, json
sys.path.append({SITE_PKG!r})
import pcbnew
board = pcbnew.LoadBoard({ORIGINAL_PCB!r})
pts = []
for d in board.GetDrawings():
    try:
        layer = pcbnew.LayerName(d.GetLayer())
        if 'Edge.Cuts' in layer:
            shape = str(d.GetShapeStr())
            if shape.lower() == 'rect':
                s, e = d.GetStart(), d.GetEnd()
                pts = [
                    [round(pcbnew.ToMM(s.x), 3), round(pcbnew.ToMM(s.y), 3)],
                    [round(pcbnew.ToMM(e.x), 3), round(pcbnew.ToMM(e.y), 3)],
                    [round(pcbnew.ToMM(e.x), 3), round(pcbnew.ToMM(s.y), 3)],
                    [round(pcbnew.ToMM(s.x), 3), round(pcbnew.ToMM(e.y), 3)],
                    [round(pcbnew.ToMM(s.x), 3), round(pcbnew.ToMM(s.y), 3)],
                ]
    except:
        pass
print(json.dumps({{'outline': pts}}))
"""
        proc = subprocess.run([KICAD_PYTHON, '-c', outline_script],
                             capture_output=True, text=True, timeout=60)
        outline_data = None
        for line in proc.stdout.strip().split('\n'):
            if line.strip().startswith('{'):
                outline_data = json.loads(line)
                break

        outline = outline_data['outline'] if outline_data else make_outline()

        # 导出
        print("  [2/3] 导出为测试PCB...")
        export_ok = export_board_from_scratch(
            ROUNDTRIP_PCB, export_comps, net_names, outline
        )
        print(f"  导出: {'✅' if export_ok else '❌'}")

        # 再导入验证
        print("  [3/3] 再导入验证...")
        reimport_data = extract_board_data(ROUNDTRIP_PCB)

        # T28: 元件数量一致
        r = TestResult("T28: Round-trip 元件数量不变")
        orig_count = orig_data['components_count']
        reimport_count = reimport_data['components_count'] if reimport_data else 0
        r.add(orig_count == reimport_count,
              f"原版 {orig_count} → 再导入 {reimport_count}")
        all_results.append(r)

        # T29: 网络数量一致
        r = TestResult("T29: Round-trip 网络数量不变")
        orig_nets = orig_data['nets_count']
        reimport_nets = reimport_data['nets_count'] if reimport_data else 0
        r.add(orig_nets == reimport_nets,
              f"原版 {orig_nets} → 再导入 {reimport_nets}")
        all_results.append(r)

    # ============================================================
    # T30: 手动构建 → 导出 → 再导入
    # ============================================================
    print("\n🔄 Round-trip 测试（手动构建）")
    print("  " + "-" * 50)

    manual_comps = make_components()
    manual_nets = [{'name': n} for n in make_nets()]
    manual_outline = make_outline()
    manual_traces = {
        'GND': [((130, 100), (148, 108), 0.2)],
        '3V3': [((135, 95), (148, 108), 0.2)],
    }
    manual_vias = [{'pos': (140, 102), 'drill': 0.3, 'diameter': 0.6}]

    manual_pcb = os.path.join(OUTPUT_DIR, "测试四_manual.kicad_pcb")
    manual_ok = export_board_from_scratch(
        manual_pcb, manual_comps, manual_nets,
        manual_outline, manual_traces, manual_vias
    )

    r = TestResult("T30: 手动构建 → 导出 → 再导入一致性")
    if not manual_ok:
        r.add(False, "导出失败")
    else:
        reimport = extract_board_data(manual_pcb)
        if not reimport:
            r.add(False, "再导入失败")
        else:
            # 检查
            r.add(reimport['components_count'] == len(manual_comps),
                  f"元件数: {reimport['components_count']} == {len(manual_comps)}")
            r.add(reimport['nets_count'] == len(manual_nets),
                  f"网络数: {reimport['nets_count']} == {len(manual_nets)}")
            r.add(reimport['segments'] == 2,
                  f"走线段: {reimport['segments']} == 2")
            r.add(reimport['vias'] == 1,
                  f"过孔: {reimport['vias']} == 1")
            # KiCad 能打开
            from test_pcbnew import kicad_load_board
            kr = kicad_load_board(manual_pcb)
            r.add(kr.get('ok', False), f"KiCad 打开: {'✅' if kr.get('ok') else '❌'}")
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

    # 生成的文件
    print(f"\n  生成的文件:")
    if os.path.exists(ROUNDTRIP_PCB):
        print(f"    {ROUNDTRIP_PCB}")
    if os.path.exists(manual_pcb):
        print(f"    {manual_pcb}")
    print()


if __name__ == '__main__':
    main()
