"""
KiCad PCB 导出辅助函数
根据 kicad_interface_完整版.md 的 §4.4 实现 export_board_from_scratch
"""
import os


def export_board_from_scratch(
    output_path: str,
    components: list,
    nets: list,
    board_outline: list,
    traces: dict = None,
    vias: list = None,
) -> bool:
    """从零生成 KiCad PCB 文件（文档 §4.4）

    Args:
        output_path: 输出路径
        components: [{'ref','footprint','size':(w,h),'pos':(x,y),'rotation':float,'pins':[...]}, ...]
        nets: [{'name':str}, ...]
        board_outline: [(x,y), ...] 板框多边形
        traces: {net_name: [((x1,y1),(x2,y2),width), ...]}
        vias: [{'pos':(x,y), 'drill':float, 'diameter':float}]

    Returns:
        True 成功, False 失败
    """
    try:
        lines = []
        lines.append('(kicad_pcb')
        lines.append('\t(version 20260206)')
        lines.append('\t(generator "pcbnew")')
        lines.append('\t(generator_version "10.0")')
        lines.append('\t(general')
        lines.append('\t\t(thickness 1.6)')
        lines.append('\t\t(legacy_teardrops no)')
        lines.append('\t)')
        lines.append('\t(paper "A4")')

        # 层定义
        lines.append('\t(layers')
        lines.append('\t\t(0 "F.Cu" signal "Top Layer")')
        lines.append('\t\t(2 "B.Cu" signal "Bottom Layer")')
        for lid, lname in [(31, "F.CrtYd"), (29, "B.CrtYd"), (35, "F.Fab"), (33, "B.Fab"),
                           (5, "F.SilkS"), (7, "B.SilkS"), (1, "F.Mask"), (3, "B.Mask")]:
            lines.append(f'\t\t({lid} "{lname}" user)')
        lines.append('\t\t(25 "Edge.Cuts" user "Multi-Layer")')
        lines.append('\t)')

        # 板框（直接顶层 gr_line，不能包 graphics）
        if board_outline:
            for i in range(len(board_outline) - 1):
                x1, y1 = board_outline[i]
                x2, y2 = board_outline[i + 1]
                lines.append(f'\t(gr_line (start {x1} {y1}) (end {x2} {y2}) '
                            f'(layer "Edge.Cuts") (width 0.05))')

        # 网络定义（直接顶层 net，不能包 nets）
        for i, net in enumerate(nets):
            name = net['name'] if isinstance(net, dict) else net
            lines.append(f'\t(net {i+1} "{name}")')

        # 元件
        net_map = {}
        for i, net in enumerate(nets):
            name = net['name'] if isinstance(net, dict) else net
            net_map[name] = i + 1

        for comp in components:
            ref = comp.get('ref', 'U?')
            pos = comp.get('pos', (0, 0))
            rot = comp.get('rotation', 0)
            w, h = comp.get('size', (3, 3))
            hw, hh = w / 2, h / 2
            pin_count = comp.get('pin_count', len(comp.get('pins', [])) or 2)

            lines.append(f'\t(footprint "RL:{ref}"')
            lines.append(f'\t\t(layer "F.Cu")')
            lines.append(f'\t\t(at {pos[0]} {pos[1]} {rot})')
            lines.append(f'\t\t(descr "RL generated {ref}")')
            lines.append(f'\t\t(tags "")')

            # Reference
            lines.append(f'\t\t(property "Reference" "{ref}"')
            lines.append(f'\t\t\t(at {pos[0]} {pos[1] + hh + 1} 0)')
            lines.append(f'\t\t\t(effects (font (size 0.8 0.8) (thickness 0.12)) (justify left))')
            lines.append(f'\t\t)')

            # Value
            lines.append(f'\t\t(property "Value" "{ref}"')
            lines.append(f'\t\t\t(at {pos[0]} {pos[1] - hh - 1} 0)')
            lines.append(f'\t\t\t(effects (font (size 0.8 0.8) (thickness 0.12)) (justify left))')
            lines.append(f'\t\t)')

            # 丝印
            lines.append(f'\t\t(fp_rect (start {-hw} {-hh}) (end {hw} {hh})')
            lines.append(f'\t\t\t(stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS")')
            lines.append(f'\t\t)')

            # 焊盘
            if pin_count <= 4:
                gap = max(w * 0.25, 0.3)
                pw, ph = min(0.6, w * 0.3), min(0.8, h * 0.4)
                lines.append(f'\t\t(pad "1" smd rect (at {-gap} 0 {rot}) '
                            f'(size {pw} {ph}) (layers "F.Cu" "F.Paste" "F.Mask"))')
                lines.append(f'\t\t(pad "2" smd rect (at {gap} 0 {rot}) '
                            f'(size {pw} {ph}) (layers "F.Cu" "F.Paste" "F.Mask"))')
            elif pin_count <= 16:
                pitch = h / (pin_count // 2 + 1)
                pw, ph = 0.4, 1.0
                for side in range(2):
                    for p in range(pin_count // 2):
                        yn = -h * 0.35 + p * pitch
                        xn = -w * 0.45 if side == 0 else w * 0.45
                        pn = p + 1 + side * (pin_count // 2)
                        lines.append(f'\t\t(pad "{pn}" smd rect (at {xn} {yn} 0) '
                                    f'(size {pw} {ph}) (layers "F.Cu" "F.Paste" "F.Mask"))')
            else:
                pps = pin_count // 4
                pitch = w * 0.7 / pps
                pw, ph = 0.3, 1.2
                for p in range(pps):
                    lines.append(f'\t\t(pad "{p+1}" smd rect (at {-w*0.35+p*pitch} {h*0.45} 0) '
                                f'(size {ph} {pw}) (layers "F.Cu" "F.Paste" "F.Mask"))')
                    lines.append(f'\t\t(pad "{p+1+pps}" smd rect (at {w*0.45} {h*0.35-p*pitch} 0) '
                                f'(size {pw} {ph}) (layers "F.Cu" "F.Paste" "F.Mask"))')
                    lines.append(f'\t\t(pad "{p+1+pps*2}" smd rect (at {w*0.35-p*pitch} {-h*0.45} 0) '
                                f'(size {ph} {pw}) (layers "F.Cu" "F.Paste" "F.Mask"))')
                    lines.append(f'\t\t(pad "{p+1+pps*3}" smd rect (at {-w*0.45} {-h*0.35+p*pitch} 0) '
                                f'(size {pw} {ph}) (layers "F.Cu" "F.Paste" "F.Mask"))')
            lines.append('\t\t)')

        # 走线
        if traces:
            net_id = 1
            for net_name in ([n['name'] for n in nets] if isinstance(nets[0], dict) else nets):
                segs = traces.get(net_name, [])
                for seg in segs:
                    if len(seg) == 3:
                        (x1, y1), (x2, y2), w = seg
                        lines.append(f'\t(segment (start {x1:.3f} {y1:.3f}) '
                                    f'(end {x2:.3f} {y2:.3f}) '
                                    f'(width {w}) (layer "F.Cu") (net {net_id}))')
                net_id += 1

        # 过孔
        if vias:
            for v in vias:
                vx, vy = v.get('pos', (0, 0))
                drill = v.get('drill', 0.3)
                dia = v.get('diameter', 0.6)
                lines.append(f'\t(via (at {vx} {vy}) (size {dia}) (drill {drill}) '
                            f'(layers "F.Cu" "B.Cu") (net 0))')

        lines.append('\t(embedded_fonts no)')
        lines.append(')')

        content = '\n'.join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    except Exception as e:
        print(f"导出失败: {e}")
        return False
