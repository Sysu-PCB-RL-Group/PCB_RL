"""PCB 文件扫描工具：递归扫描目录获取 .kicad_pcb 文件列表。

供 run.py、run_inference.py 等多处共用，统一过滤逻辑。
"""

import os


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
