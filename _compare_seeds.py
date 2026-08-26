"""对比多个 seed 训练结果。

用法（在仓库根目录）:
    python _compare_seeds.py            # 自动扫描 result 下 real_board_seed*
    python _compare_seeds.py dir1 dir2  # 手动指定多个 save_dir（相对 result 或绝对路径均可）
"""
import glob
import json
import os
import sys

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")


def collect(save_dir: str):
    """读取一个 save_dir 下所有板子的 summary，返回 {seed名: {board: summary}}。"""
    out = {}
    name = os.path.basename(save_dir.rstrip("\\/"))
    sfile = os.path.join(save_dir, "training_summary.json")
    if not os.path.isdir(save_dir) or not os.path.exists(sfile):
        return out
    with open(sfile, encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        board = os.path.basename(entry.get("pcb", "")).replace(".kicad_pcb", "")
        out.setdefault(name, {})[board] = entry
    return out


def rolling_avg(seq, w=1000):
    """滚动平均，返回与输入等长序列。"""
    res, s = [], 0.0
    for i, v in enumerate(seq):
        s += v
        if i >= w:
            s -= seq[i - w]
        res.append(s / min(i + 1, w))
    return res


def main():
    if len(sys.argv) > 1:
        dirs = [d if os.path.isabs(d) else os.path.join(RESULT, d) for d in sys.argv[1:]]
    else:
        dirs = sorted(glob.glob(os.path.join(RESULT, "real_board_seed*")))
    if not dirs:
        print("未找到任何 seed 结果目录，请先运行训练或手动指定 save_dir。")
        return

    print(f"发现 {len(dirs)} 个 seed 结果目录:\n")
    for d in dirs:
        print(f"  - {os.path.basename(d)}")

    all_data = {}
    for d in dirs:
        all_data.update(collect(d))

    # 汇总表
    print("\n" + "=" * 100)
    print("【各 seed 汇总对比】")
    print("=" * 100)
    header = f"{'seed':<16}{'板子':<44}{'first_avg_r':>12}{'last_avg_r':>12}{'best_R':>10}{'best_ep':>9}{'episodes':>10}"
    print(header)
    print("-" * 100)
    for seed_name in sorted(all_data):
        for board, e in sorted(all_data[seed_name].items()):
            print(
                f"{seed_name:<16}{board[:42]:<44}"
                f"{e['first_avg_r']:>12.2f}{e['last_avg_r']:>12.2f}"
                f"{e['best_reward']:>10.2f}{e['best_episode']:>9d}{e['episodes']:>10d}"
            )

    # reward 曲线对比（同一板子的各 seed 滚动平均）
    print("\n" + "=" * 100)
    print("【reward 滚动平均对比（每 1000 episode）】")
    print("=" * 100)
    boards = set()
    for m in all_data.values():
        boards.update(m.keys())
    for board in sorted(boards):
        print(f"\n--- {board} ---")
        print(f"{'seed':<16}" + "".join(f"{i:>9d}" for i in range(0, 10001, 1000)))
        for seed_name in sorted(all_data):
            if board not in all_data[seed_name]:
                continue
            rh_file = os.path.join(
                RESULT, seed_name, board, "reward_history.json"
            )
            if not os.path.exists(rh_file):
                rh_file = os.path.join(RESULT, seed_name, "reward_history.json")
            if not os.path.exists(rh_file):
                print(f"{seed_name:<16}  (无 reward_history)")
                continue
            with open(rh_file, encoding="utf-8") as f:
                rh = json.load(f)
            vals = [x[1] for x in rh] if rh and isinstance(rh[0], list) else rh
            rv = rolling_avg(vals)
            row = [f"{seed_name:<16}"]
            for i in range(0, min(len(rv), 10001), 1000):
                row.append(f"{rv[i]:>9.1f}")
            print("".join(row))


if __name__ == "__main__":
    main()
