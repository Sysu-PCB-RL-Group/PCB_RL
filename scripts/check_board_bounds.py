# -*- coding: utf-8 -*-
import os, re

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"

for proj in sorted(os.listdir(TARGET)):
    d = os.path.join(TARGET, proj)
    if not os.path.isdir(d):
        continue
    for fname in os.listdir(d):
        if not fname.endswith(".kicad_pcb"):
            continue
        c = open(os.path.join(d, fname), encoding="utf-8", errors="replace").read()

        pts = []
        for m in re.finditer(r"Edge\.Cuts", c):
            pos = m.start()
            window = c[max(0, pos - 800):pos + 200]
            for coord_m in re.finditer(
                r"\((start|end|center|mid)\s+([-\d.]+)\s+([-\d.]+)\)", window
            ):
                tag = coord_m.group(1)
                x, y = float(coord_m.group(2)), float(coord_m.group(3))
                if coord_m.start() < len(window) - 200:
                    pts.append((x, y))

        fp = c.count('(footprint "')
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            gw = max(1, int((bw - 2) / 1.0))
            gh = max(1, int((bh - 2) / 1.0))
            ok = "OK" if gw <= 21 and gh <= 21 else "FAIL"
            print(f"{ok:>4} | FP={fp:>3} | {bw:>7.1f}x{bh:<7.1f}mm | grid {gw:>3}x{gh:<3} | {proj[:50]}")
        else:
            print(f"N/A  | FP={fp:>3} | NO EDGE DATA  | {proj[:50]}")
        break
