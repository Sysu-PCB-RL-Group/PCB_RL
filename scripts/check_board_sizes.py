# -*- coding: utf-8 -*-
"""Check board sizes and grid requirements for the 10 training boards."""
import os, re

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"

def parse_board_bounds(filepath):
    """Extract board outline bounds from KiCad PCB file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # CRITICAL: In KiCad 10, board outline is inside (gr_line ...) with "Edge.Cuts" layer
    # We need to extract (gr_line ...) and (gr_arc ...) on "Edge.Cuts", compute overall bbox.
    # Simpler approach: look for all (gr_line ... (start x y) (end x y) (layer "Edge.Cuts") )
    
    edge_points = []
    
    # gr_line on Edge.Cuts
    for m in re.finditer(r'\(gr_line[^)]*?\(start\s+([-\d.]+)\s+([-\d.]+)\)[^)]*?\(end\s+([-\d.]+)\s+([-\d.]+)\)', content):
        x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        # Check if layer is Edge.Cuts near this line
        start = m.start()
        # Check ~300 chars after for Edge.Cuts
        tail = content[start:start+400]
        if '"Edge.Cuts"' in tail:
            edge_points.append((x1, y1))
            edge_points.append((x2, y2))
    
    # Fallback: try to find any (gr_arc or (gr_line with Edge.Cuts
    if not edge_points:
        # Parse more broadly - find all gr_line with Edge.Cuts layer
        pattern = r'\(gr_line\s.*?\(start\s+([-\d.]+)\s+([-\d.]+)\).*?\(end\s+([-\d.]+)\s+([-\d.]+)\).*?(?:"Edge\.Cuts").*?\)'
        for m in re.finditer(pattern, content, re.DOTALL):
            x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            edge_points.append((x1, y1))
            edge_points.append((x2, y2))
    
    if not edge_points:
        return None, None, None, None, "no_edge_points"
    
    xs = [p[0] for p in edge_points]
    ys = [p[1] for p in edge_points]
    return min(xs), min(ys), max(xs), max(ys), None

def main():
    results = []
    for proj_name in sorted(os.listdir(TARGET)):
        proj_path = os.path.join(TARGET, proj_name)
        if not os.path.isdir(proj_path):
            continue
        for fname in os.listdir(proj_path):
            if fname.endswith(".kicad_pcb"):
                fpath = os.path.join(proj_path, fname)
                xmin, ymin, xmax, ymax, err = parse_board_bounds(fpath)
                w = xmax - xmin if xmin is not None else 0
                h = ymax - ymin if ymin is not None else 0
                
                # grid cells needed
                cell = 1.0
                grid_w = max(1, int(w / cell))
                grid_h = max(1, int(h / cell))
                
                results.append((proj_name, w, h, grid_w, grid_h, err))
                break
    
    print(f"{'Board':<60} {'W(mm)':>8} {'H(mm)':>8} {'GridW':>6} {'GridH':>6} {'Status'}")
    print("-" * 100)
    
    max_w = 0
    max_h = 0
    ok = 0
    fail = 0
    for name, w, h, gw, gh, err in results:
        status = "OK" if (gw <= 21 and gh <= 21) else "FAIL"
        if status == "OK":
            ok += 1
        else:
            fail += 1
        max_w = max(max_w, gw)
        max_h = max(max_h, gh)
        err_str = f" [ERR: {err}]" if err else ""
        print(f"{name:<60} {w:>8.1f} {h:>8.1f} {gw:>6} {gh:>6} {status}{err_str}")
    
    print("-" * 100)
    print(f"OK={ok}, FAIL={fail}, Max grid needed: {max_w}x{max_h}")
    print(f"Current MAX_MOVABLE=18, GRID_W=21, GRID_H=21")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
