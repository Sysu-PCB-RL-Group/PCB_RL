"""KiCad pcbnew 测试辅助：使用 KiCad 自带的 Python 3.11 调用 pcbnew"""
import os
import sys
import subprocess
import json

KICAD_PYTHON = r'D:/pcb大模型/kicad/bin/python.exe'
SITE_PKG = r'D:/pcb大模型/kicad/bin/Lib/site-packages'


def kicad_load_board(pcb_path: str) -> dict:
    """用 KiCad 自带的 Python 调用 pcbnew.LoadBoard，返回结果"""
    script = f"""
import sys, json
sys.path.append({SITE_PKG!r})
import pcbnew
try:
    board = pcbnew.LoadBoard({pcb_path!r})
    fp_count = len(list(board.Footprints()))
    net_count = len(list(board.GetNetsByName().values()))
    print(json.dumps({{'ok': True, 'version': pcbnew.Version(), 'footprints': fp_count, 'nets': net_count}}))
except Exception as e:
    print(json.dumps({{'ok': False, 'error': str(e)}}))
"""
    result = subprocess.run(
        [KICAD_PYTHON, '-c', script],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.strip().split('\n'):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {'ok': False, 'error': result.stderr or 'no JSON output'}


if __name__ == '__main__':
    # 测试文件
    test_dir = r'D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b'
    files = [
        ('原版答案', os.path.join(test_dir, '90i0-quan-cai-shi-zhong-ac67228b.kicad_pcb')),
        ('测试二', os.path.join(test_dir, '测试二.kicad_pcb')),
    ]
    # 自动发现 step 文件
    for f in sorted(os.listdir(test_dir)):
        if f.startswith('测试二_step') and f.endswith('.kicad_pcb'):
            files.append((f, os.path.join(test_dir, f)))

    all_ok = True
    for name, path in files:
        if not os.path.exists(path):
            print(f'  ❌ {name}: 文件不存在')
            all_ok = False
            continue
        result = kicad_load_board(path)
        if result.get('ok'):
            print(f'  ✅ {name}: {result["footprints"]}个footprint, {result["nets"]}个net (KiCad {result["version"]})')
        else:
            print(f'  ❌ {name}: {result.get("error", "未知错误")}')
            all_ok = False

    print()
    print('✅ 全部通过' if all_ok else '❌ 有失败')
