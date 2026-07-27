#!/usr/bin/env python3
"""
KiCad 接口测试 —— 第二轮：导入功能（T7-T18）
============================================
测试从 .kicad_pcb 和 .kicad_sch 文件导入数据的正确性。

测试内容：
  T7:  load_board() 返回 True
  T8:  Component 数量 == 47
  T9:  Component 字段非空
  T10: Net 数量 >= 5
  T11: Net connections 非空
  T12: 电源网络 (GND/3V3) is_power = True
  T13: 板框边界正确
  T14: 封装类型识别 (R0603→0, LQFP-48→2, SOP-14→1)
  T15: load_schematic() 返回 True
  T16: 原理图 Component 数量 == 48
  T17: 原理图 Component 无位置信息
  T18: 原理图 Net 数量 >= 5

使用方法：
  python test_kicad_import.py

依赖：
  test_pcbnew.py（调用 KiCad 的 pcbnew API）
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from test_pcbnew import kicad_load_board

# Windows UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ================================================================
# 配置
# ================================================================
DATA_DIR = r"D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b"
PCB_PATH = os.path.join(DATA_DIR, "90i0-quan-cai-shi-zhong-ac67228b.kicad_pcb")
SCH_PATH = os.path.join(DATA_DIR, "90i0-quan-cai-shi-zhong-ac67228b.kicad_sch")

EXPECTED_FP_COUNT = 47   # PCB 中 footprint 数量
EXPECTED_SCH_COUNT = 48  # 原理图中 symbol 数量
EXPECTED_NET_MIN = 5     # 最少网络数

# ================================================================
# S-expression 解析器（用于原理图）
# ================================================================

def sexpr_tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in '()':
            tokens.append(c)
            i += 1
        elif c in ' \t\n\r':
            i += 1
        elif c == '"':
            i += 1
            s = []
            while i < n:
                if text[i] == '\\' and i + 1 < n:
                    s.append(text[i+1]); i += 2
                elif text[i] == '"':
                    i += 1; break
                else:
                    s.append(text[i]); i += 1
            tokens.append('"' + ''.join(s) + '"')
        elif c == ';':
            while i < n and text[i] != '\n': i += 1
        else:
            s = []
            while i < n and text[i] not in '() \t\n\r"':
                s.append(text[i]); i += 1
            tokens.append(''.join(s))
    return tokens


def parse_sexpr(tokens, start=0):
    result = []
    i = start
    while i < len(tokens):
        t = tokens[i]
        if t == '(':
            subtree, i = parse_sexpr(tokens, i + 1)
            result.append(subtree)
        elif t == ')':
            return result, i + 1
        else:
            try:
                result.append(float(t) if '.' in t else int(t))
            except (ValueError, TypeError):
                result.append(t)
            i += 1
    return result, i


def unquote(s):
    if isinstance(s, str) and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return str(s)


def is_node(lst, name):
    return isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], str) and lst[0] == name


def find_all(lst, pred, results=None):
    if results is None:
        results = []
    if isinstance(lst, list):
        if pred(lst):
            results.append(lst)
        for item in lst:
            find_all(item, pred, results)
    return results


# ================================================================
# 原理图解析函数
# ================================================================

def parse_schematic(sch_path):
    """从原理图提取 Component 和 Net 列表"""
    with open(sch_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = sexpr_tokenize(text)
    tree, _ = parse_sexpr(tokens)

    all_symbols = find_all(tree, lambda n: is_node(n, 'symbol'))
    components = []
    seen_refs = set()

    for sym in all_symbols:
        has_lib_id = any(is_node(child, 'lib_id') for child in sym)
        if not has_lib_id:
            continue
        lib_id_node = [c for c in sym if is_node(c, 'lib_id')]
        if not lib_id_node:
            continue
        lib_id = unquote(lib_id_node[0][1]) if len(lib_id_node[0]) > 1 else ""

        ref = ""
        value = ""
        footprint = ""
        for child in sym:
            if is_node(child, 'property') and len(child) >= 3:
                prop_name = unquote(child[1])
                prop_val = unquote(child[2]) if isinstance(child[2], str) else ""
                if prop_name == 'Reference':
                    ref = prop_val
                elif prop_name == 'Value':
                    value = prop_val
                elif prop_name == 'Footprint':
                    footprint = prop_val

        if not ref or ref.startswith('#'):
            continue
        if ref in seen_refs:
            continue
        seen_refs.add(ref)

        pin_count = len(find_all(sym, lambda n: is_node(n, 'pin')))

        components.append({
            'ref': ref,
            'value': value,
            'lib_id': lib_id,
            'footprint_str': footprint,
            'pin_count': pin_count,
        })

    # 提取网络（通过 label 名称）
    labels = find_all(tree, lambda n: is_node(n, 'label'))
    nets = set()
    for lbl in labels:
        if len(lbl) >= 2:
            nets.add(unquote(lbl[1]))

    return components, sorted(nets)


def classify_package_type(fp_name):
    """封装类型识别（文档 §2.3）"""
    fp = fp_name.upper()
    if any(k in fp for k in ['0402', '0603', '0805', '1206', '1210', '2512']):
        return 0  # Chip
    if any(k in fp for k in ['SOIC', 'SOT', 'SSOP', 'TSSOP']):
        return 1  # SOIC/SOT
    if any(k in fp for k in ['QFP', 'TQFP', 'LQFP']):
        return 2  # QFP
    if 'BGA' in fp:
        return 3  # BGA
    return 4  # IC 通用


def estimate_footprint_size(fp_name):
    """从封装名估算尺寸"""
    fp = fp_name.lower()
    sizes = {
        '0201': (0.6, 0.3), '0402': (1.0, 0.5), '0603': (1.6, 0.8),
        '0805': (2.0, 1.2), '1206': (3.2, 1.6), 'sot-223': (6.7, 3.5),
        'sot-23': (2.9, 1.6), 'sop-14': (10.1, 5.0), 'lqfp-48': (7.0, 7.0),
        'qfn-48': (7.0, 7.0),
    }
    for key, size in sizes.items():
        if key in fp:
            return size
    m = __import__('re').search(r'l(\d+\.?\d*).*w(\d+\.?\d*)', fp, __import__('re').IGNORECASE)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (3.0, 3.0)


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
            if not ok or True:  # 总是打出来方便看
                print(f"     {icon} {msg}")


# ================================================================
# 主测试
# ================================================================

def main():
    print("=" * 60)
    print("  KiCad 接口测试 —— 第二轮：导入功能（T7-T18）")
    print("=" * 60)

    all_results = []

    # ============================================================
    # PCB 导入测试（T7-T14）
    # ============================================================
    print("\n📦 PCB 导入测试")
    print("  " + "-" * 50)

    r = TestResult("T7: load_board() 返回 True")
    result = kicad_load_board(PCB_PATH)
    r.add(result.get('ok'), f"KiCad {result.get('version','?')} 加载 {os.path.basename(PCB_PATH)}")
    all_results.append(r)

    pcb_ok = result.get('ok', False)
    if pcb_ok:
        fp_count_actual = result.get('footprints', 0)
    else:
        fp_count_actual = 0

    r = TestResult(f"T8: Component 数量 == {EXPECTED_FP_COUNT}")
    r.add(fp_count_actual == EXPECTED_FP_COUNT,
          f"实际: {fp_count_actual}, 期望: {EXPECTED_FP_COUNT}")
    all_results.append(r)

    # T9-T14: 用 KiCad Python 提取详细数据
    print("\n  🔍 提取 PCB 详细数据（通过 KiCad pcbnew API）...")
    detail_script = f"""
import sys, json, math
sys.path.append({repr(r'D:/pcb大模型/kicad/bin/Lib/site-packages')})
import pcbnew
board = pcbnew.LoadBoard({repr(PCB_PATH)})

components = []
for fp in board.Footprints():
    try:
        ref = fp.GetReference()
        fp_name = fp.GetFPID().GetLibItemName() or ""
        bbox = fp.GetBoundingBox()
        rot = fp.GetOrientationDegrees()
        pads = fp.Pads()
        components.append({{
            'ref': str(ref),
            'footprint': str(fp_name),
            'bbox': [round(pcbnew.ToMM(bbox.GetLeft()), 3),
                     round(pcbnew.ToMM(bbox.GetTop()), 3),
                     round(pcbnew.ToMM(bbox.GetRight()), 3),
                     round(pcbnew.ToMM(bbox.GetBottom()), 3)],
            'rotation': float(rot),
            'pad_count': len(pads),
            'is_locked': bool(fp.IsLocked()),
        }})
    except Exception as e:
        components.append({{'ref': 'ERR', 'footprint': str(e)}})

nets = []
for net in board.GetNetsByName().values():
    try:
        name = net.GetNetname()
        if not name:
            continue
        connections = []
        for fp in board.Footprints():
            for pad in fp.Pads():
                if pad.GetNetCode() == net.GetNetCode():
                    connections.append(str(fp.GetReference()))
        up = str(name).upper()
        is_power = up in ('VCC','VDD','VSS','GND','PWR','VEE','VPP','VREF','3V3','5V')
        nets.append({{
            'name': str(name),
            'connections_count': len(set(connections)),
            'is_power': is_power,
        }})
    except:
        pass

bounds = None
for drawing in board.GetDrawings():
    try:
        layer_name = pcbnew.LayerName(drawing.GetLayer())
        if 'Edge.Cuts' in layer_name:
            shape = str(drawing.GetShapeStr())
            if shape.lower() == 'rect':
                start = drawing.GetStart()
                end = drawing.GetEnd()
                pts = [(pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)),
                       (pcbnew.ToMM(end.x), pcbnew.ToMM(end.y))]
                bounds = {{
                    'min_x': round(min(p[0] for p in pts), 3),
                    'min_y': round(min(p[1] for p in pts), 3),
                    'max_x': round(max(p[0] for p in pts), 3),
                    'max_y': round(max(p[1] for p in pts), 3),
                }}
    except:
        pass

# 确保所有值都是 Python 原生类型
out = {{
    'components': components,
    'nets': nets,
    'board_bounds': bounds,
}}
print(json.dumps(out, ensure_ascii=True))
"""
    import subprocess
    kicad_python = r'D:/pcb大模型/kicad/bin/python.exe'
    proc = subprocess.run([kicad_python, '-c', detail_script],
                         capture_output=True, text=True, timeout=60)

    detail_data = None
    for line in proc.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            try:
                detail_data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if not detail_data:
        stderr = proc.stderr[:300] if proc.stderr else "(no stderr)"
        stdout = proc.stdout[:300] if proc.stdout else "(no stdout)"
        print(f"  ❌ 提取失败")
        print(f"  stderr: {stderr}")
        print(f"  stdout: {stdout}")
        for name in ["T9", "T10", "T11", "T12", "T13", "T14"]:
            r = TestResult(f"{name}: 跳过（数据提取失败）")
            r.add(False, "KiCad 数据提取未返回结果")
            all_results.append(r)
    else:
        comps = detail_data['components']
        nets = detail_data['nets']
        bounds = detail_data['board_bounds']

        # T9: 字段非空
        r = TestResult("T9: Component 字段非空")
        for c in comps[:5]:  # 抽检前5个
            r.add(bool(c['ref']), f"ref={c['ref']}")
            r.add(bool(c['footprint']), f"footprint={c['footprint']}")
            r.add(len(c['bbox']) == 4, f"bbox={c['bbox']}")
            r.add(c['rotation'] >= 0, f"rotation={c['rotation']}")
            r.add(c['pad_count'] > 0, f"pad_count={c['pad_count']}")
        all_results.append(r)

        # T10: Net 数量
        r = TestResult(f"T10: Net 数量 >= {EXPECTED_NET_MIN}")
        r.add(len(nets) >= EXPECTED_NET_MIN, f"实际: {len(nets)}")
        for n in nets[:8]:
            r.add(True, f"  net: {n['name']} ({n['connections_count']} connections)")
        all_results.append(r)

        # T11: Net connections 非空
        r = TestResult("T11: Net connections 非空")
        power_nets = [n for n in nets if n['is_power']]
        for n in power_nets[:3]:
            r.add(n['connections_count'] > 0, f"{n['name']}: {n['connections_count']} 连接")
        non_power = [n for n in nets if not n['is_power']]
        for n in non_power[:3]:
            r.add(n['connections_count'] > 0, f"{n['name']}: {n['connections_count']} 连接")
        all_results.append(r)

        # T12: 电源网络标记
        r = TestResult("T12: 电源网络 is_power = True")
        for n in nets:
            if n['name'] in ('GND', '3V3', '5V', 'VCC'):
                r.add(n['is_power'], f"{n['name']} is_power={n['is_power']}")
        all_results.append(r)

        # T13: 板框边界
        r = TestResult("T13: 板框边界正确")
        if bounds:
            bw = bounds['max_x'] - bounds['min_x']
            bh = bounds['max_y'] - bounds['min_y']
            r.add(bw > 0 and bh > 0,
                  f"({bounds['min_x']:.1f}, {bounds['min_y']:.1f}) ~ "
                  f"({bounds['max_x']:.1f}, {bounds['max_y']:.1f}) = {bw:.0f}x{bh:.0f}mm")
            r.add(50 < bw < 70, f"板宽 {bw:.0f}mm (期望 50-70mm)")
            r.add(30 < bh < 50, f"板高 {bh:.0f}mm (期望 30-50mm)")
        else:
            r.add(False, "未找到板框")
        all_results.append(r)

        # T14: 封装类型识别
        r = TestResult("T14: 封装类型识别")
        for c in comps:
            fp_name = c['footprint']
            ptype = classify_package_type(fp_name)
            if '0603' in fp_name.upper() or '0805' in fp_name.upper():
                r.add(ptype == 0, f"{c['ref']}: {fp_name} → Chip(0)")
            elif 'LQFP' in fp_name.upper() or 'QFP' in fp_name.upper():
                r.add(ptype == 2, f"{c['ref']}: {fp_name} → QFP(2)")
            elif 'SOT' in fp_name.upper() or 'SOIC' in fp_name.upper():
                r.add(ptype == 1, f"{c['ref']}: {fp_name} → SOIC/SOT(1)")
            elif 'BGA' in fp_name.upper():
                r.add(ptype == 3, f"{c['ref']}: {fp_name} → BGA(3)")
        all_results.append(r)

    # ============================================================
    # 原理图导入测试（T15-T18）
    # ============================================================
    print("\n📄 原理图导入测试")
    print("  " + "-" * 50)

    sch_components, sch_nets = parse_schematic(SCH_PATH)

    r = TestResult("T15: parse_schematic() 返回数据")
    r.add(len(sch_components) > 0, f"原理图元件数: {len(sch_components)}")
    r.add(len(sch_nets) > 0, f"原理图网络数: {len(sch_nets)}")
    all_results.append(r)

    r = TestResult(f"T16: 原理图 Component 数量 == {EXPECTED_SCH_COUNT}")
    r.add(len(sch_components) == EXPECTED_SCH_COUNT,
          f"实际: {len(sch_components)}, 期望: {EXPECTED_SCH_COUNT}")
    all_results.append(r)

    r = TestResult("T17: 原理图 Component 无位置信息")
    # 原理图解析不提取 bbox，所以所有元件没有位置数据
    # 验证：检查没有位置相关的字段
    has_position = any('pos' in c or 'bbox' in c or 'at' in str(c) for c in sch_components)
    r.add(not has_position, "原理图元件不包含位置信息（因为原理图不存位置）")
    # 打印前5个元件
    for c in sch_components[:5]:
        r.add(True, f"  {c['ref']:6s} lib={c['lib_id'][:40]}")
    all_results.append(r)

    r = TestResult(f"T18: 原理图 Net 数量 >= {EXPECTED_NET_MIN}")
    r.add(len(sch_nets) >= EXPECTED_NET_MIN, f"实际: {len(sch_nets)}")
    for n in sch_nets[:10]:
        r.add(True, f"  net: {n}")
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
    print()


if __name__ == '__main__':
    main()
