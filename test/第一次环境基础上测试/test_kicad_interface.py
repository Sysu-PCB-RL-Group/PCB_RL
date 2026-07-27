#!/usr/bin/env python3
"""
KiCad 接口测试 —— 第一轮：格式基础（T1-T6）
============================================
测试生成的 .kicad_pcb 文件是否满足 KiCad 格式要求。

测试内容：
  T1: 无 (graphics 包装 —— 错误token
  T2: 无 (nets 包装   —— 错误token
  T3: 括号平衡         —— ( 和 ) 数量相等
  T4: 文件开头是 (kicad_pcb
  T5: 文件结尾是 )
  T6: KiCad 能打开     —— pcbnew.LoadBoard() 不报异常

使用方法：
  python test_kicad_interface.py

输出：
  每项测试打印 ✅ 或 ❌，最后汇总
"""

import os
import sys
import re

# Windows UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ================================================================
# 配置：要测试的文件
# ================================================================
TEST_DIR = r"D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b"

FILES_TO_TEST = [
    # (显示名, 路径)
    ("原版答案PCB（基准）",
     os.path.join(TEST_DIR, "90i0-quan-cai-shi-zhong-ac67228b.kicad_pcb")),
    ("测试二（RL生成）",
     os.path.join(TEST_DIR, "测试二.kicad_pcb")),
]

# 自动发现所有 step 中间文件
for f in sorted(os.listdir(TEST_DIR)):
    if f.startswith("测试二_step") and f.endswith(".kicad_pcb"):
        FILES_TO_TEST.append((f, os.path.join(TEST_DIR, f)))

# ================================================================
# 测试实现
# ================================================================

class TestResult:
    """记录单个测试结果"""
    def __init__(self, name):
        self.name = name
        self.results = {}  # {filename: (passed, detail)}

    def add(self, filename, passed, detail=""):
        self.results[filename] = (passed, detail)

    def all_passed(self):
        return all(v[0] for v in self.results.values())

    def passed_count(self):
        return sum(1 for v in self.results.values() if v[0])

    def total_count(self):
        return len(self.results)

    def print(self):
        status = "✅" if self.all_passed() else "❌"
        print(f"\n  {status} {self.name}")
        print(f"     {'='*50}")
        for fname, (passed, detail) in sorted(self.results.items()):
            icon = "✅" if passed else "❌"
            short = fname if len(fname) <= 45 else fname[:42] + "..."
            print(f"     {icon} {short}")
            if detail and not passed:
                print(f"         └─ {detail}")


# ---- T1: 无 graphics 包装 ----
def test_no_graphics_wrapper(filepath):
    """T1: 检查文件中没有 (graphics 这个非法 token"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    count = content.count('(graphics')
    return count == 0, f"找到 {count} 处 (graphics" if count > 0 else ""


# ---- T2: 无 nets 包装 ----
def test_no_nets_wrapper(filepath):
    """T2: 检查文件中没有 (nets 这个非法 token（注意 (net 合法，(nets 非法）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 找 (nets 后面跟空格/换行/制表符 —— 这是非法包装
    # 而 (net 数字 "名称" 是合法的网络定义
    count = len(re.findall(r'\(nets\s', content))
    return count == 0, f"找到 {count} 处 (nets 包装" if count > 0 else ""


# ---- T3: 括号平衡 ----
def test_paren_balance(filepath):
    """T3: ( 和 ) 数量相等"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    open_p = content.count('(')
    close_p = content.count(')')
    if open_p == close_p:
        return True, f"({open_p} = {close_p})"
    else:
        return False, f"({open_p} != {close_p})"


# ---- T4: 文件开头 ----
def test_starts_with_kicad_pcb(filepath):
    """T4: 文件以 (kicad_pcb 开头"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    stripped = content.strip()
    if stripped.startswith('(kicad_pcb'):
        return True, ""
    else:
        return False, f"实际开头: {stripped[:50]}"


# ---- T5: 文件结尾 ----
def test_ends_with_proper_close(filepath):
    """T5: 最后一个非空字符是 )"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    stripped = content.strip()
    if stripped.endswith(')'):
        return True, ""
    else:
        return False, f"实际结尾: {stripped[-50:]}"


# ---- T6: KiCad 能打开 ----
def test_kicad_opens(filepath):
    """T6: pcbnew.LoadBoard() 能正常加载，不报异常"""
    try:
        # 优先用 KiCad 自带的 Python 调用 pcbnew
        from test_pcbnew import kicad_load_board
        result = kicad_load_board(filepath)
        if result.get('ok'):
            return True, f"KiCad {result['version']} 加载成功，{result['footprints']} 个 footprint"
        else:
            return False, result.get('error', '未知错误')
    except ImportError:
        return False, "未找到 test_pcbnew 模块"
    except Exception as e:
        return False, f"测试异常: {str(e)}"


# ================================================================
# 主程序
# ================================================================

def main():
    print("=" * 60)
    print("  KiCad 接口测试 —— 第一轮：格式基础（T1-T6）")
    print("=" * 60)
    print(f"\n  测试文件数: {len(FILES_TO_TEST)}")
    for name, path in FILES_TO_TEST:
        exists = "✅存在" if os.path.exists(path) else "❌不存在"
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"    {exists} {name} ({size:,} 字节)")

    # 所有测试定义
    tests = [
        ("T1 无 (graphics 包装", test_no_graphics_wrapper),
        ("T2 无 (nets 包装",    test_no_nets_wrapper),
        ("T3 括号平衡",         test_paren_balance),
        ("T4 以 (kicad_pcb 开头", test_starts_with_kicad_pcb),
        ("T5 以 ) 结尾",         test_ends_with_proper_close),
        ("T6 KiCad 能打开",      test_kicad_opens),
    ]

    all_results = []

    for test_name, test_func in tests:
        result = TestResult(test_name)
        for display_name, filepath in FILES_TO_TEST:
            if not os.path.exists(filepath):
                result.add(display_name, False, "文件不存在")
                continue

            try:
                passed, detail = test_func(filepath)
                result.add(display_name, passed, detail)
            except Exception as e:
                result.add(display_name, False, f"测试异常: {str(e)}")

        result.print()
        all_results.append(result)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)

    all_pass = True
    for r in all_results:
        icon = "✅" if r.all_passed() else "❌"
        print(f"  {icon} {r.name}: {r.passed_count()}/{r.total_count()} 通过")
        if not r.all_passed():
            all_pass = False

    print()
    if all_pass:
        print("  ✅✅✅ 全部通过！格式基础验证 OK")
    else:
        print("  ❌❌❌ 存在失败项，请检查上方的 ❌ 标记")

    print(f"\n  测试文件目录: {TEST_DIR}")
    print()


if __name__ == '__main__':
    main()
