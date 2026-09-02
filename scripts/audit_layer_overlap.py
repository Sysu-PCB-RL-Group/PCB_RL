"""一次性审计脚本：检查真实 PCB 板是否存在「异层(TOP/BOTTOM)正对重叠」的元件对。

目的
----
修复「碰撞判定忽略 layer」前，需要确认：真实人工完美布局里是否存在
「TOP 面元件与 BOTTOM 面元件在 XY 平面完全/部分重叠」的情况。
- 若不存在：旧代码(忽略层)与修复后(同层才判)对完美布局的判定一致 → 现有专家缓存有效，无需重生成。
- 若存在：旧代码会把完美布局误判为硬重叠 → 专家 reward 标签已污染 → 必须重生成 + 重训 BC。

实现说明
--------
- 本环境无 pcbnew(KiCad Mock 模式)，故用纯文本 s-expression 解析 .kicad_pcb。
- 元件层：footprint 节点的 (layer ...) 字段，F.Cu→0 / B.Cu→1 / 其它→2(内层/非铜层，不计入可放置元件)。
- 元件 bbox：以 footprint 的 (at x y [rot]) 为原点，按 footprint 旋转角对 pad 偏移做坐标变换，
  用 pad 的 (size w h)/直径 ± 旋转后轴对齐外接矩形半边，得到近似包围盒。足以检测粗粒度重叠。
- 「异层重叠对」：两元件 layer 不同(且均∈{0,1})且 XY 硬重叠面积 > 0。

输出
----
- 屏幕打印每块的层分布与异层重叠对数。
- 结果同时写入 scripts/audit_layer_overlap_result.txt。
"""

import os
import sys
import glob
import math

# ---- 极简 s-expression 解析（仅用于读取，不修改任何文件）----

def tokenize(s: str):
    tokens = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "(":
            tokens.append("(")
            i += 1
            continue
        if c == ")":
            tokens.append(")")
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and s[j] != '"':
                buf.append(s[j])
                j += 1
            tokens.append('"' + "".join(buf) + '"')
            i = j + 1
            continue
        j = i
        while j < n and s[j] not in " \t\r\n()":
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def parse(tokens):
    pos = 0

    def parse_expr():
        nonlocal pos
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            lst = []
            while tokens[pos] != ")":
                lst.append(parse_expr())
            pos += 1
            return lst
        else:
            pos += 1
            return tok

    return parse_expr()


def get_child(node, name):
    for child in node:
        if isinstance(child, list) and child and child[0] == name:
            return child
    return None


def all_footprints(node, out):
    if isinstance(node, list) and node and node[0] == "footprint":
        out.append(node)
        return
    if isinstance(node, list):
        for child in node:
            all_footprints(child, out)


def layer_code(layer_name: str) -> int:
    if layer_name == "F.Cu":
        return 0
    if layer_name == "B.Cu":
        return 1
    return 2  # 内层/非铜层


def footprint_bbox(fp) -> tuple:
    """返回 (layer_code, (x1,y1,x2,y2) 近似包围盒) 或 None(非可放置元件)。"""
    layer_node = get_child(fp, "layer")
    if layer_node is None or len(layer_node) < 2:
        return None
    layer_name = layer_node[1].strip('"')
    lc = layer_code(layer_name)
    if lc == 2:
        return None  # 只统计 TOP/BOTTOM 可放置元件

    at = get_child(fp, "at")
    if at is None or len(at) < 3:
        return None
    fx, fy = float(at[1]), float(at[2])

    pads = [c for c in fp if isinstance(c, list) and c and c[0] == "pad"]
    if not pads:
        # 无 pad：用原点 ± 0.5mm 兜底
        return (lc, (fx - 0.5, fy - 0.5, fx + 0.5, fy + 0.5))

    # footprint 旋转角（at 第三项为旋转，单位：度）
    rot = float(at[3]) if len(at) >= 4 else 0.0
    ang = math.radians(rot)
    cos_a, sin_a = math.cos(ang), math.sin(ang)

    xmin = xmax = fx
    ymin = ymax = fy
    for pad in pads:
        pat = get_child(pad, "at")
        if pat is None or len(pat) < 3:
            continue
        ox = float(pat[1])
        oy = float(pat[2])
        px = fx + (ox * cos_a - oy * sin_a)
        py = fy + (ox * sin_a + oy * cos_a)
        size = get_child(pad, "size")
        if size is not None and len(size) >= 3:
            w = float(size[1])
            h = float(size[2])
        elif size is not None and len(size) == 2:
            w = h = float(size[1])
        else:
            w = h = 1.0
        # 旋转后轴对齐外接矩形半边
        hw = (abs(cos_a) * w + abs(sin_a) * h) / 2
        hh = (abs(sin_a) * w + abs(cos_a) * h) / 2
        xmin = min(xmin, px - hw)
        xmax = max(xmax, px + hw)
        ymin = min(ymin, py - hh)
        ymax = max(ymax, py + hh)
    return (lc, (xmin, ymin, xmax, ymax))


def overlap_area(a, b) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx > 0 and dy > 0:
        return dx * dy
    return 0.0


def audit_board(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    tokens = tokenize(text)
    tree = parse(tokens)
    fps = []
    all_footprints(tree, fps)

    comps = []  # (ref, layer_code, bbox)
    for fp in fps:
        # ref 取 (property "Reference" "R1") 或首层 tstamp；退化为 index
        ref = None
        for child in fp:
            if isinstance(child, list) and len(child) >= 3 and child[0] == "property" and child[1].strip('"') in ("Reference", "Reference designator"):
                ref = child[2].strip('"')
        bbox = footprint_bbox(fp)
        if bbox is None:
            continue
        if ref is None:
            ref = f"fp{len(comps)}"
        comps.append((ref, bbox[0], bbox[1]))

    top = [c for c in comps if c[1] == 0]
    bot = [c for c in comps if c[1] == 1]
    other = [c for c in comps if c[1] == 2]

    cross_pairs = []
    for i in range(len(comps)):
        li, bi = comps[i][1], comps[i][2]
        for j in range(i + 1, len(comps)):
            lj, bj = comps[j][1], comps[j][2]
            if li == lj:
                continue  # 仅统计异层
            if overlap_area(bi, bj) > 0:
                cross_pairs.append((comps[i][0], li, comps[j][0], lj, round(overlap_area(bi, bj), 3)))

    name = os.path.basename(os.path.dirname(path))
    return {
        "board": name,
        "total": len(comps),
        "top": len(top),
        "bottom": len(bot),
        "other_layer": len(other),
        "cross_layer_overlap_pairs": cross_pairs,
    }


def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kicad-native-batch")
    if not os.path.isdir(base):
        print(f"[ERR] 未找到目录 {base}")
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(base, "**", "*.kicad_pcb"), recursive=True))
    results = []
    lines = []
    lines.append("=" * 70)
    lines.append("真实板「异层重叠」审计（layer 修复前置检查）")
    lines.append("=" * 70)
    contaminated = False
    for p in files:
        r = audit_board(p)
        results.append(r)
        lines.append("")
        lines.append(f"板: {r['board']}")
        lines.append(f"  可放置元件: {r['total']}  (TOP={r['top']}, BOTTOM={r['bottom']}, 其它层={r['other_layer']})")
        if r["cross_layer_overlap_pairs"]:
            contaminated = True
            lines.append(f"  [WARN] 异层重叠对: {len(r['cross_layer_overlap_pairs'])} 个")
            for ref1, l1, ref2, l2, area in r["cross_layer_overlap_pairs"][:10]:
                lines.append(f"     {ref1}({'TOP' if l1==0 else 'BOT'}) vs {ref2}({'TOP' if l2==0 else 'BOT'}) 重叠={area}mm2")
        else:
            lines.append("  [OK] 无异层重叠对")
    lines.append("")
    lines.append("=" * 70)
    if contaminated:
        lines.append("结论: 存在异层重叠对 => 旧代码曾误判完美布局 => 专家缓存可能污染，修复后需重生成。")
    else:
        lines.append("结论: 所有板均无异层重叠对 → 修复前后对完美布局判定一致 → 专家缓存有效，无需重生成。")
    lines.append("=" * 70)
    out_text = "\n".join(lines)
    print(out_text)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_layer_overlap_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"\n[OK] 结果已写入 {out_path}")


if __name__ == "__main__":
    main()
