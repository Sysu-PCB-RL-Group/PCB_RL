"""PCB 文件扫描工具：递归扫描目录获取 .kicad_pcb 文件列表。

供 run.py、run_inference.py 等多处共用，统一过滤逻辑；
并集中定义「默认单板」常量，供训练/推理/专家生成/审查等入口复用同一口径。
"""

import os

# ---------------------------------------------------------------------------
# 默认单板（对 .kicad_pcb 文件名做子串匹配）——训练/推理的默认目标板。
# 完整目录名：jackt-tong-yong-xiao-che-kong-zhi-qi-zhuan-li-948f377a--4.IMU小板v1_1-09db0370
# 选它的原因：它是 data/kicad-native-batch 下 20 块候选中【唯一】兼容当前固定维度框架的板
#   （MAX_MOVABLE=18 / GRID_W=GRID_H=100 / CELL_SIZE=0.2mm → 板框须 ≤20×20mm 且可动元件 ≤18）。
#   实测：15/20 板可动元件超 18、19/20 板网格超 100，详见 result/kicad_board_scan/report.md。
# 用纯 ASCII 子串，避免 PowerShell 中文参数 GBK 乱码；改默认板时只需改这一处。
# ---------------------------------------------------------------------------
DEFAULT_BOARD = "IMU"


def find_pcb_files(data_dir: str) -> list:
    """递归扫描目录获取所有 .kicad_pcb 文件，过滤衍生/测试文件。

    Args:
        data_dir: 扫描根目录

    Returns:
        排序后的 PCB 文件绝对路径列表
    """
    extensions = (".kicad_pcb", ".pcb")
    files = []
    if not os.path.isdir(data_dir):
        return files
    for root, _dirs, fnames in os.walk(data_dir):
        for f in fnames:
            if not f.lower().endswith(extensions):
                continue
            # 过滤衍生文件（测试、导出版本、手动布局等）
            if any(kw in f for kw in ("测试", "test", "_step", "_export",
                                        "_manual", "_roundtrip", "movecall")):
                continue
            files.append(os.path.join(root, f))
    return sorted(files)
