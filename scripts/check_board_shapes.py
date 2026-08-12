"""分析 5 个板子的板框形状 —— 规则矩形 vs 异形"""
import os, re

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"

for proj in sorted(os.listdir(TARGET)):
    d = os.path.join(TARGET, proj)
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        if not f.endswith('.kicad_pcb'): continue
        c = open(os.path.join(d, f), encoding='utf-8', errors='replace').read()

        # 收集 Edge.Cuts 中的 gr_line / gr_circle / gr_arc
        lines = []   # (start_x, start_y, end_x, end_y, angle_deg)
        circles = [] # (center_x, center_y, end_x, end_y)
        arcs = []    # (start, mid, end)

        for m in re.finditer(r'Edge\.Cuts', c):
            pos = m.start()
            window = c[max(0, pos - 1000):pos + 200]
            # 按出现顺序解析几何体
            geo_pattern = re.compile(
                r'\((gr_line|gr_circle|gr_arc)\s+'
                r'\(start\s+([-\d.]+)\s+([-\d.]+)\)\s+'
                r'\((end|mid)\s+([-\d.]+)\s+([-\d.]+)\)'
            )
            for gm in geo_pattern.finditer(window):
                gtype = gm.group(1)
                x1, y1 = float(gm.group(2)), float(gm.group(3))
                tag2 = gm.group(4)
                x2, y2 = float(gm.group(5)), float(gm.group(6))
                if gtype == 'gr_line':
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = (360 + 180 * (0 if dx == 0 else (1 if dx == 0 else (1 if dy/dx > 0 else -1)))) % 360
                    # compute actual angle
                    import math
                    angle = math.degrees(math.atan2(dy, dx))
                    lines.append((x1, y1, x2, y2, angle))
                elif gtype == 'gr_circle':
                    circles.append((x1, y1, x2, y2))
                elif gtype == 'gr_arc':
                    arcs.append((x1, y1, x2, y2, tag2))

        # 判断形状
        print(f"\n{'='*60}")
        print(f"  {proj[:55]}")
        print(f"{'='*60}")
        print(f"  gr_line: {len(lines)}  |  gr_circle: {len(circles)}  |  gr_arc: {len(arcs)}")

        if len(lines) == 4 and len(circles) == 0 and len(arcs) == 0:
            # 4 条线段 → 可能是矩形
            angles = set()
            for l in lines:
                ang = round(l[4] % 180, 1)  # 线段方向 mod 180
                if ang > 90: ang = 180 - ang
                angles.add(ang)
            if angles == {0.0, 90.0} or angles == {90.0}:
                print(f"  >> 形状: 矩形 (4条正交线段)")
                xs = [p[0] for p in lines for p in [[p[0],p[1]] for p in [(l[0],l[1]) for l in lines]] + [[l[2],l[3]] for l in lines]]
                print(f"  >>    尺寸: {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} mm")
            else:
                print(f"  >> 形状: 四边形(不规则) 角度: {angles}")
        elif len(lines) > 4 and len(circles) == 0 and len(arcs) == 0:
            print(f"  >> 形状: 多边形 ({len(lines)} 边)")
            xs = [l[0] for l in lines] + [l[2] for l in lines]
            ys = [l[1] for l in lines] + [l[3] for l in lines]
            print(f"  >>    尺寸: {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} mm")
        elif len(circles) > 0 and len(lines) == 0:
            for c in circles:
                r = ((c[2]-c[0])**2 + (c[3]-c[1])**2)**0.5
                print(f"  >> 形状: 圆形 (圆心 {c[0]:.1f},{c[1]:.1f}, 半径 {r:.1f}mm)")
        elif len(arcs) > 0:
            print(f"  >> 形状: 含圆弧的异形")
        else:
            print(f"  >> 形状: 混合/不规则 ({len(lines)}线 + {len(circles)}圆 + {len(arcs)}弧)")

        # 打印每条线段的端点
        if lines:
            print(f"  --- 线段端点 ---")
            for l in lines:
                print(f"    ({l[0]:.3f},{l[1]:.3f}) -> ({l[2]:.3f},{l[3]:.3f})  angle={l[4]:.1f}°")
        if circles:
            for c in circles:
                r = ((c[2]-c[0])**2 + (c[3]-c[1])**2)**0.5
                print(f"    圆心({c[0]:.3f},{c[1]:.3f}) 半径={r:.3f}mm")

        break
