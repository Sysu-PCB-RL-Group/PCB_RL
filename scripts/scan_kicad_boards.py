# -*- coding: utf-8 -*-
"""Scan KiCad 10 PCB files - count footprints and unique nets."""
import os, re

SCAN_ROOT = r"E:\PCB_AI\环境重构\kicad-native-batch_extracted\kicad-native-batch"
OUTPUT_FILE = r"e:\PCB_AI\PCB_RL\data\scan_kicad_boards_result.txt"

def parse_kicad_pcb(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return None, str(e)

    # KiCad 10 footprints: (footprint "lib:name" ...)
    fp_count = content.count('(footprint "')
    
    # Nets are embedded in pads as (net "NAME") - count unique net names
    net_names = set(re.findall(r'\(net\s+"([^"]+)"', content))
    net_count = len(net_names)

    return fp_count, net_count

def main():
    results = []
    if not os.path.isdir(SCAN_ROOT):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
            out.write("ERROR: path not found\n")
        return

    for proj_name in sorted(os.listdir(SCAN_ROOT)):
        proj_path = os.path.join(SCAN_ROOT, proj_name)
        if not os.path.isdir(proj_path):
            continue
        for fname in os.listdir(proj_path):
            if fname.endswith(".kicad_pcb"):
                fpath = os.path.join(proj_path, fname)
                fp, net = parse_kicad_pcb(fpath)
                if fp is None:
                    results.append((proj_name, fname, "ERR", str(net)))
                else:
                    results.append((proj_name, fname, fp, net))
                break

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("=" * 120 + "\n")
        out.write("SIMPLE BOARDS (footprints 1-9, nets < 5)\n")
        out.write("=" * 120 + "\n")
        out.write(f"{'Project':<68} {'FP':>5} {'Nets':>5}  Notes\n")
        out.write("-" * 120 + "\n")

        simple_boards = []
        all_boards = []
        for proj, fname, fp, net in results:
            if isinstance(fp, int):
                all_boards.append((proj, fname, fp, net))
                if 0 < fp < 10 and net < 5:
                    simple_boards.append((proj, fname, fp, net))
                    out.write(f"{proj:<68} {fp:>5} {net:>5}\n")

        out.write("\n" + "=" * 120 + "\n")
        out.write(f"Simple boards: {len(simple_boards)} (0 < fp < 10, net < 5)\n")
        out.write(f"Total projects: {len(results)}\n\n")

        if all_boards:
            fps = [r[2] for r in all_boards]
            nets = [r[3] for r in all_boards]
            out.write(f"FP range: {min(fps)} ~ {max(fps)}, median: {sorted(fps)[len(fps)//2]}\n")
            out.write(f"Net range: {min(nets)} ~ {max(nets)}, median: {sorted(nets)[len(nets)//2]}\n")

            # Count by net brackets
            nb_0 = sum(1 for r in all_boards if r[3] == 0)
            nb_1_4 = sum(1 for r in all_boards if 1 <= r[3] <= 4)
            nb_5_9 = sum(1 for r in all_boards if 5 <= r[3] <= 9)
            nb_10_19 = sum(1 for r in all_boards if 10 <= r[3] <= 19)
            nb_20p = sum(1 for r in all_boards if r[3] >= 20)
            out.write(f"\nNet distribution: 0={nb_0}, 1-4={nb_1_4}, 5-9={nb_5_9}, 10-19={nb_10_19}, 20+={nb_20p}\n")

        # Full listing sorted by fp
        out.write(f"\n=== ALL BOARDS (sorted by fp) ===\n")
        out.write(f"{'Project':<68} {'FP':>5} {'Nets':>5}\n")
        out.write("-" * 80 + "\n")
        for proj, fname, fp, net in sorted(all_boards, key=lambda x: (x[2], x[3])):
            out.write(f"{proj:<68} {fp:>5} {net:>5}\n")

    print(f"Done. {len(simple_boards)} simple boards. Results: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
