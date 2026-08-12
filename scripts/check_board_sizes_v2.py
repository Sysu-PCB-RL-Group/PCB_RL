# -*- coding: utf-8 -*-
"""Check board sizes using KiCad's own board parsing."""
import os, sys

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"

# 使用 kicad_utils 自身的解析能力 (最准)
# 但需要 mock 以避免在非 KiCad 环境崩溃
# 这里直接用 pcbnew 解析或用纯 Python 正则
# 改用更宽泛的 Edge.Cuts 解析

import re

def parse_all_edge_cuts_points(filepath):
    """Parse all Edge.Cuts geometry - gr_line, gr_arc, gr_circle, gr_rect, gr_poly."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    points = []
    
    # Find all geometry items and check if layer is "Edge.Cuts"
    # gr_line: (gr_line (start x y) (end x y) ... (layer "Edge.Cuts") ...)
    # gr_circle: (gr_circle (center x y) (end x y) ... (layer "Edge.Cuts") ...)
    # gr_arc: (gr_arc (start x y) ... (end x y) ... (layer "Edge.Cuts") ...)
    
    # Strategy: split s-expressions, find (gr_* ... "Edge.Cuts" ...) blocks
    
    # Simplest approach: find all coordinates near "Edge.Cuts"
    # Scan for "Edge.Cuts" and look for surrounding (start...), (end...), (center...), (pts...)
    
    # Use a block-based approach
    depth = 0
    current_block = []
    blocks = []
    for ch in content:
        if ch == '(':
            if depth == 0:
                current_block = []
            current_block.append(ch)
            depth += 1
        elif ch == ')':
            depth -= 1
            current_block.append(ch)
            if depth == 0:
                blocks.append(''.join(current_block))
        else:
            if depth > 0:
                current_block.append(ch)
    
    edge_blocks = [b for b in blocks if '"Edge.Cuts"' in b]
    
    for block in edge_blocks:
        # Extract all coordinate-like tokens: (start x y), (end x y), (center x y), (pts ...)
        coords = re.findall(r'([-\d.]+)\s+([-\d.]+)', block)
        for x, y in coords:
            points.append((float(x), float(y)))
    
    if not points:
        return None, None, None, None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def main():
    CELL = 1.0
    MARGIN = 1.0
    GRID_LIMIT = 21
    
    results = []
    for proj_name in sorted(os.listdir(TARGET)):
        proj_path = os.path.join(TARGET, proj_name)
        if not os.path.isdir(proj_path):
            continue
        for fname in os.listdir(proj_path):
            if fname.endswith(".kicad_pcb"):
                fpath = os.path.join(proj_path, fname)
                xmin, ymin, xmax, ymax = parse_all_edge_cuts_points(fpath)
                if xmin is None:
                    results.append((proj_name, 0, 0, 0, 0, "NO_EDGE"))
                    break
                w = xmax - xmin
                h = ymax - ymin
                # grid = actual valid area (minus 2*margin) / cell
                gw = max(1, int((w - 2*MARGIN) / CELL))
                gh = max(1, int((h - 2*MARGIN) / CELL))
                ok = gw <= GRID_LIMIT and gh <= GRID_LIMIT
                results.append((proj_name, w, h, gw, gh, "OK" if ok else "FAIL"))
                break
    
    # Also count footprints
    def count_fp(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            c = f.read()
        return c.count('(footprint "')
    
    for proj_name in sorted(os.listdir(TARGET)):
        proj_path = os.path.join(TARGET, proj_name)
        if not os.path.isdir(proj_path):
            continue
        for fname in os.listdir(proj_path):
            if fname.endswith(".kicad_pcb"):
                fpath = os.path.join(proj_path, fname)
                fp_count = count_fp(fpath)
                break
        # update the result
        for i, r in enumerate(results):
            if r[0] == proj_name:
                results[i] = (proj_name, r[1], r[2], r[3], r[4], r[5], fp_count)
                break
    
    print(f"{'Board':<65} {'W':>7} {'H':>7} {'GW':>5} {'GH':>5} {'FP':>4} {'Status'}")
    print("-" * 105)
    ok_count = 0
    fail_count = 0
    for r in results:
        name, w, h, gw, gh, status, fp = r
        flag = "OK" if status == "OK" else "FAIL"
        if status == "OK":
            ok_count += 1
        else:
            fail_count += 1
        print(f"{name:<65} {w:>7.1f} {h:>7.1f} {gw:>5} {gh:>5} {fp:>4} {flag}")
    
    print("-" * 105)
    print(f"OK={ok_count}, FAIL={fail_count}")
    print(f"Grid limit: {GRID_LIMIT}x{GRID_LIMIT}, MAX_MOVABLE=18, CELL_SIZE={CELL}mm")

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
