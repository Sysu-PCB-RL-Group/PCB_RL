"""KiCad 数据集扫描 CLI（阶段 A scan / 阶段 B report）。

用法：
  阶段 A（需 KiCad DLL，建议在用户自己的终端执行）：
    $env:KICAD_PATH="E:\Kicad"; cd E:\PCB_AI\PCB_RL
    & E:\miniconda\envs\pcb_rl_311\python.exe scripts/kicad_board_scan.py scan \
        --src "E:\\PCB_AI\\环境重构\\kicad-native-batch_extracted\\kicad-native-batch" \
        --out result/kicad_board_scan

  阶段 B（无需 KiCad，Agent 可直接执行）：
    & E:\miniconda\envs\pcb_rl_311\python.exe scripts/kicad_board_scan.py report --out result/kicad_board_scan

说明：源目录只读；所有产物写入 --out 目录。
"""

import argparse
import os
import sys

# 允许直接从仓库根目录运行（import utils.kicad_board_scan）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.kicad_board_scan import (  # noqa: E402
    load_rows,
    render_report,
    scan_dir,
    select_and_save,
)


def _cmd_scan(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.src):
        print(f"[scan] 源目录不存在: {args.src}")
        return 1
    meta = scan_dir(args.src, args.out, progress_every=args.progress_every,
                    round_timeout=args.timeout)
    if meta["failed"]:
        print(f"[scan] 注意：{meta['failed']} 块加载失败，见 {os.path.join(args.out, 'load_errors.csv')}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    json_path = os.path.join(args.out, "scan_all.json")
    if not os.path.exists(json_path):
        print(f"[report] 未找到 {json_path}，请先执行 scan")
        return 1
    meta, rows = load_rows(args.out)
    if not rows:
        print("[report] scan_all.json 为空")
        return 1
    candidates = select_and_save(rows, args.out)
    render_report(meta, rows, args.out, candidates=candidates)
    print(f"[report] 全部 {len(rows)} 块；通过硬条件 {sum(1 for r in rows if r.get('pass'))} 块；"
          f"候选 {len(candidates)} 块。产物写入 {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kicad_board_scan",
        description="KiCad 数据集扫描 / 筛选报告（源目录只读）。",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="阶段 A：全量扫描（需 KiCad DLL，LoadBoard）")
    p_scan.add_argument("--src", required=True, help="源数据集根目录（只读）")
    p_scan.add_argument("--out", default="result/kicad_board_scan", help="输出目录")
    p_scan.add_argument("--progress-every", type=int, default=10)
    p_scan.add_argument("--timeout", type=float, default=1800.0,
                        help="单轮扫描超时秒数（超时自动续扫；默认 30 分钟）")
    p_scan.set_defaults(func=_cmd_scan)

    p_report = sub.add_parser("report", help="阶段 B：读缓存出筛选/分档/报告（无需 KiCad）")
    p_report.add_argument("--out", default="result/kicad_board_scan", help="scan 输出目录")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
