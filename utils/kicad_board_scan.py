"""KiCad 数据集批量扫描与筛选模块（对只读源目录做统计分析）。

用途：对 `kicad-native-batch` 之类的大批量 .kicad_pcb 数据集做全量指标提取，
按 RL 数据集要求筛出「大致矩形板框 + 可动元件 5-100 且非纯 pad + 有效 net>3」的板，
并在 5-100 全区间分档均衡选出不超过 20 块的候选清单。

设计要点：
- **纯逻辑 / pcbnew 惰性分离**：模块顶层不 import pcbnew，便于离线 report 与纯函数单测。
  `analyze_pcb()` 内部按需惰性导入（此时必须已配置 KICAD_PATH 并启用 KiCad 环境）。
- 固定件判定口径与 `utils/kicad_utils/kicad_utils.py` 保持一致：
  `IsLocked()` 或封装名含 HOLE/MOUNT/MECH/FIDUCIAL 或 0 网络连接 → fixed。
  为避免测试环境里 kicad_utils 被 Mock 污染，此处独立实现同一逻辑（常量复制，注释互指）。
- 两阶段：
  阶段 A `scan_dir()`：遍历 + LoadBoard + 全量指标落盘 scan_all.json/csv（需 KiCad DLL，用户终端跑）。
  阶段 B `load_rows()` → `select_candidates()` → `render_report()`：读缓存，离线筛选/分档/出报告。

输出产物（写入 out_dir，不写源目录）：
  scan_all.json / scan_all.csv    全量逐板指标
  load_errors.csv                 加载失败清单
  candidates.json / candidates.csv 最终候选（<=20）
  report.md                       筛选说明与候选报告
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# ===================== 可调阈值常量区（报告会逐条列出） =====================

# —— 板框矩形度 ——
RECT_RATIO_MIN = 0.75          # aligned_len / 总周长 >= 此值 → 大致矩形（允许圆角/倒角）
EDGE_TOL_MM = 0.15             # 线段贴齐 bbox 边线的容差 (mm)
ARC_STEP_MM = 0.4              # 弧线离散最大步长 (mm)

# —— 元件 ——
MOVABLE_MIN, MOVABLE_MAX = 5, 100      # 非固定元件数范围
CONV_ABS_MIN = 3                       # 常规可动元件绝对下限
CONV_RATIO_MIN = 0.3                   # 常规可动元件 / 可动元件 比例下限
PAD_LIKE_REF_PREFIXES = ("TP", "PAD", "MP", "MTP", "TEST")

# —— 网络 ——
NET_MIN = 4                       # 有效 net（>=2 pad）数量下限：要求 >3

# —— 分档（5-100 均衡覆盖）——
TIERS = [(5, 12), (13, 25), (26, 45), (46, 70), (71, 100)]
TIER_QUOTA = 4                    # 每档配额，5 档 * 4 = 20
MAX_CANDIDATES = 20

# 标准常规元件位号前缀（IPC/常见前缀；连接器 J/JP/P/JP 等算常规元件）
CONVENTIONAL_TOKENS = frozenset({
    "R", "C", "U", "IC", "L", "Q", "D", "CR", "ZD", "VR", "RV",
    "J", "JP", "P", "CON", "CN", "SW", "K", "FB", "F", "BT", "B",
    "Y", "X", "RN", "LED", "DS", "T", "M", "S", "Z", "G", "MK",
    "LS", "BUZ", "SP", "FS", "RX", "TX", "O", "A", "W", "E",
})

# 固定件封装关键词（与 kicad_utils._FIXED_FOOTPRINT_KEYWORDS 同步）
_FIXED_KEYWORDS = ("HOLE", "MOUNT", "MECH", "FIDUCIAL")

# KICAD_PATH 默认兜底（本项目 KiCad 安装路径）
_DEFAULT_KICAD_PATH = r"E:\Kicad"


# ===================== 纯函数：位号 / 元件分类 =====================

_REF_PREFIX_RE = re.compile(r"^([A-Za-z]+)")


def ref_alpha_prefix(ref: str) -> str:
    """提取位号前的字母前缀并大写。'R21'->'R'，'JP1'->'JP'，空串->''。"""
    if not ref:
        return ""
    m = _REF_PREFIX_RE.match(ref.strip())
    return m.group(1).upper() if m else ""


def _is_pad_like_name(footprint_name_upper: str) -> bool:
    """按封装名判断是否纯焊盘/测试点（Pad_eXXX / TESTPOINT 等）。"""
    n = footprint_name_upper
    if n.startswith("PAD"):
        return True
    if "TESTPOINT" in n or ("TEST" in n and "POINT" in n):
        return True
    return False


def classify_ref(ref: str, footprint_name: str) -> str:
    """将 footprint 归类为 conventional / pad_like / other。

    规则（优先级从高到低）：
      1. Reference 为空：封装名像 pad → pad_like；否则 other（未知元件，不算常规）；
      2. 位号前缀命中 pad-like 集合（TP/PAD/MP/TEST...）→ pad_like；
      3. 位号前缀命中常规元件集合 → conventional；
      4. 其余 → other。
    """
    name_u = (footprint_name or "").upper()
    token = ref_alpha_prefix(ref)
    if not ref or not ref.strip():
        return "pad_like" if _is_pad_like_name(name_u) else "other"
    if token in PAD_LIKE_REF_PREFIXES:
        return "pad_like"
    if token in CONVENTIONAL_TOKENS:
        return "conventional"
    return "other"


def _seg_len(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ===================== 纯函数：曲线 → 折线段 =====================

def _wrap_angle(rad: float) -> float:
    """归一到 (-pi, pi]。"""
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad <= -math.pi:
        rad += 2 * math.pi
    return rad


def polyline_arc(start, mid, end, max_step: float = ARC_STEP_MM) -> List[Tuple[float, float]]:
    """由三点（弧上 start/mid/end，mm）重建圆弧并离散为折线点列。

    Returns:
        [(x, y), ...]，包含首尾；仅一个点时返回空列表。单位 mm。
    """
    s = (float(start[0]), float(start[1]))
    m = (float(mid[0]), float(mid[1]))
    e = (float(end[0]), float(end[1]))

    def _circle_center(p, q, r):
        # 三条弦的垂直平分线交点（圆心）
        ax, ay = q[0] - p[0], q[1] - p[1]
        bx, by = r[0] - q[0], r[1] - q[1]
        denom = 2.0 * (ax * by - ay * bx)
        if abs(denom) < 1e-9:
            return None
        # p + 0.5*(q-p) 的垂线：方向 (-ay, ax)；q + 0.5*(r-q) 的垂线：(-by, bx)
        px = (p[0] + q[0]) / 2.0
        py = (p[1] + q[1]) / 2.0
        qx = (q[0] + r[0]) / 2.0
        qy = (q[1] + r[1]) / 2.0
        # px + t*(-ay, ax) == qx + u*(-by, bx)
        ux, uy = -ay, ax
        vx, vy = -by, bx
        t = ((qx - px) * vy - (qy - py) * vx) / (ux * vy - uy * vx)
        return (px + t * ux, py + t * uy)

    c = _circle_center(s, m, e)
    if c is None:
        return []
    r = _seg_len(c, s)
    if r < 1e-9:
        return []
    a_s = math.atan2(s[1] - c[1], s[0] - c[0])
    a_m = math.atan2(m[1] - c[1], m[0] - c[0])
    a_e = math.atan2(e[1] - c[1], e[0] - c[0])
    d_sm = _wrap_angle(a_m - a_s)
    d_me = _wrap_angle(a_e - a_m)
    # 统一方向（mid 夹在 start..end 之间）
    if d_sm * d_me >= 0 and abs(d_sm) > 1e-6:
        total = d_sm + d_me
    else:
        total = _wrap_angle(a_e - a_s)
    total = _wrap_angle(total)
    if abs(total) < 1e-6:
        return []
    arc_len = abs(total) * r
    n = int(math.ceil(arc_len / max_step))
    n = max(4, min(n, 400))
    pts = []
    for i in range(n + 1):
        ang = a_s + total * (i / n)
        pts.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)))
    # 确保首尾精确落在输入端点
    pts[0] = s
    pts[-1] = e
    return pts


def polyline_circle(center, radius: float, max_step: float = ARC_STEP_MM) -> List[Tuple[float, float]]:
    """整圆离散为折线点列（含闭合首尾重复点）。"""
    r = float(radius)
    if r <= 0:
        return []
    circ = 2 * math.pi * r
    n = max(12, int(math.ceil(circ / max_step)))
    pts = []
    for i in range(n + 1):
        ang = 2 * math.pi * (i / n)
        pts.append((float(center[0]) + r * math.cos(ang), float(center[1]) + r * math.sin(ang)))
    return pts


def polyline_bezier(start, c1, c2, end, n: int = 24) -> List[Tuple[float, float]]:
    """三次贝塞尔离散为折线点列（mm）。"""
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (mt ** 3) * start[0] + 3 * (mt ** 2) * t * c1[0] + 3 * mt * (t ** 2) * c2[0] + (t ** 3) * end[0]
        y = (mt ** 3) * start[1] + 3 * (mt ** 2) * t * c1[1] + 3 * mt * (t ** 2) * c2[1] + (t ** 3) * end[1]
        pts.append((x, y))
    return pts


def _polyline_to_segments(pts: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segs = []
    for a, b in zip(pts, pts[1:]):
        if _seg_len(a, b) > 1e-9:
            segs.append((a, b))
    return segs


# ===================== 纯函数：矩形度 =====================

def rect_score(segments_mm):
    """评估板框折线段与自身 bbox 边线的贴合度。

    Args:
        segments_mm: [((x1,y1),(x2,y2)), ...] 单位 mm（Edge.Cuts 离散后）。

    Returns:
        (rect_ratio, stats)：rect_ratio ∈ [0,1]，1≈完美矩形；
        stats = {"perimeter_mm", "aligned_mm", "w_mm", "h_mm", "outline_present"}
    """
    stats = {"perimeter_mm": 0.0, "aligned_mm": 0.0, "w_mm": 0.0, "h_mm": 0.0,
             "outline_present": bool(segments_mm)}
    if not segments_mm:
        return 0.0, stats

    xs = []
    ys = []
    for (a, b) in segments_mm:
        xs.extend((a[0], b[0]))
        ys.extend((a[1], b[1]))
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    stats["w_mm"], stats["h_mm"] = w, h
    if w < 1e-6 or h < 1e-6:
        return 0.0, stats

    tol = EDGE_TOL_MM
    perimeter = 0.0
    aligned = 0.0
    for (a, b) in segments_mm:
        length = _seg_len(a, b)
        perimeter += length
        # 水平段：贴合上/下边线
        if abs(a[1] - b[1]) <= 1e-6:
            y = a[1]
            if abs(y - ymin) <= tol or abs(y - ymax) <= tol:
                aligned += length
        # 竖直段：贴合左/右边线
        elif abs(a[0] - b[0]) <= 1e-6:
            x = a[0]
            if abs(x - xmin) <= tol or abs(x - xmax) <= tol:
                aligned += length
    stats["perimeter_mm"] = perimeter
    stats["aligned_mm"] = aligned
    if perimeter <= 0:
        return 0.0, stats
    return aligned / perimeter, stats


# ===================== pcbnew 运行时分析（惰性导入 KiCad） =====================

_pcbnew = None


def _get_pcbnew():
    """惰性导入 pcbnew，自动按 KICAD_PATH 注入 DLL 目录与 site-packages。

    与 utils/kicad_utils 的注入逻辑一致（此处独立实现，避免模块级 Mock 干扰）；
    未配置 KICAD_PATH 或加载失败时给出明确报错。仅 Windows 场景（add_dll_directory）。
    """
    global _pcbnew
    if _pcbnew is not None:
        return _pcbnew
    kicad_path = os.environ.get("KICAD_PATH", "")
    if not kicad_path and os.path.isdir(_DEFAULT_KICAD_PATH):
        kicad_path = _DEFAULT_KICAD_PATH
    if not kicad_path:
        raise RuntimeError(
            "KICAD_PATH 未设置：请先执行 `$env:KICAD_PATH=\"E:\\Kicad\"` 再运行 scan"
        )
    try:
        os.add_dll_directory(os.path.join(kicad_path, "bin"))
    except Exception:
        pass
    for p in (os.path.join(kicad_path, "bin", "Lib", "site-packages"),
              os.path.join(kicad_path, "bin")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import pcbnew  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"pcbnew 加载失败 (KICAD_PATH={kicad_path}): {exc}") from exc
    _pcbnew = pcbnew
    return _pcbnew


def _pt_mm(pn, vec) -> Tuple[float, float]:
    return (pn.ToMM(vec.x), pn.ToMM(vec.y))


def _is_fixed(fp, footprint_name_upper: str, pn) -> bool:
    """固定件判定，与 kicad_utils._cache_components 同口径。"""
    try:
        if fp.IsLocked():
            return True
    except Exception:
        pass
    if any(k in footprint_name_upper for k in _FIXED_KEYWORDS):
        return True
    # 0 网络连接 = 机械件（螺丝/铜柱/定位点/无 pad 图形件）
    try:
        pad_nets = {pad.GetNetname() for pad in fp.Pads() if pad.GetNetname()}
        if not pad_nets:
            return True
    except Exception:
        pass
    return False


def _shape_kind_of(item, pn) -> Optional[str]:
    """返回图形的规范种类：line/arc/circle/rect/polygon/curve/None。

    优先用 SHAPE_T 枚举常量比较；旧版本无常量或异常时退回 GetShapeStr() 名称。
    """
    try:
        st = item.GetShape()
        try:
            if st == pn.S_SEGMENT:
                return "line"
            if st == pn.S_ARC:
                return "arc"
            if st == pn.S_CIRCLE:
                return "circle"
            if st == pn.S_RECT:
                return "rect"
            if st == pn.S_POLYGON:
                return "polygon"
            if st == pn.S_CURVE:
                return "curve"
        except AttributeError:
            pass
    except Exception:
        pass
    try:
        name = str(item.GetShapeStr())
    except Exception:
        return None
    upper = name.upper()
    if upper in ("SEGMENT", "LINE"):
        return "line"
    if upper in ("ARC",):
        return "arc"
    if upper in ("CIRCLE",):
        return "circle"
    if upper in ("RECT", "RECTANGLE"):
        return "rect"
    if upper in ("POLYGON",):
        return "polygon"
    if upper in ("BEZIER", "CURVE"):
        return "curve"
    return None


def _collect_edge_cuts_segments(board, pn) -> Tuple[List, Dict[str, Any]]:
    """提取 Edge.Cuts 层的全部图形并离散为 mm 折线段。

    Returns:
        (segments, shape_stats)；shape_stats 记录图形构成，便于人工复核
        （line/arc/circle/rect/polygon/curve 数量与弧长占比）。
    """
    segments: List = []
    counts: Dict[str, int] = {"line": 0, "arc": 0, "circle": 0, "rect": 0,
                              "polygon": 0, "curve": 0, "other": 0, "skip": 0}
    curved_len = 0.0
    edge_id = board.GetLayerID("Edge.Cuts")
    if edge_id < 0:
        return segments, {"counts": counts, "curved_len_mm": 0.0}
    for item in board.GetDrawings():
        cls = item.GetClass()
        if "SHAPE" not in cls.upper():
            continue
        if item.GetLayer() != edge_id:
            continue
        kind = _shape_kind_of(item, pn)
        try:
            if kind == "line":
                a = _pt_mm(pn, item.GetStart())
                b = _pt_mm(pn, item.GetEnd())
                seg = (a, b)
                if _seg_len(*seg) > 1e-9:
                    segments.append(seg)
                counts["line"] += 1
            elif kind == "arc":
                s = _pt_mm(pn, item.GetStart())
                e = _pt_mm(pn, item.GetEnd())
                m = _pt_mm(pn, item.GetArcMid())
                pts = polyline_arc(s, m, e)
                segs = _polyline_to_segments(pts)
                segments.extend(segs)
                curved_len += sum(_seg_len(a, b) for a, b in segs)
                counts["arc"] += 1
            elif kind == "circle":
                c = _pt_mm(pn, item.GetCenter())
                r = pn.ToMM(item.GetRadius())
                pts = polyline_circle(c, r)
                segs = _polyline_to_segments(pts)
                segments.extend(segs)
                curved_len += sum(_seg_len(a, b) for a, b in segs)
                counts["circle"] += 1
            elif kind == "rect":
                try:
                    corners = [_pt_mm(pn, v) for v in item.GetRectCorners()]
                except Exception:
                    corners = None
                if not corners or len(corners) < 4:
                    s = _pt_mm(pn, item.GetStart())
                    e = _pt_mm(pn, item.GetEnd())
                    corners = [(s[0], s[1]), (e[0], s[1]), (e[0], e[1]), (s[0], e[1])]
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    seg = (a, b)
                    if _seg_len(*seg) > 1e-9:
                        segments.append(seg)
                counts["rect"] += 1
            elif kind == "polygon":
                poly = item.GetPolyShape()
                n_out = poly.OutlineCount()
                for oi in range(n_out):
                    outline = poly.Outline(oi)
                    n_pt = outline.PointCount()
                    if n_pt < 3:
                        continue
                    for i in range(n_pt):
                        p = outline.CPoint(i)
                        q = outline.CPoint((i + 1) % n_pt)
                        a = (pn.ToMM(p.x), pn.ToMM(p.y))
                        b = (pn.ToMM(q.x), pn.ToMM(q.y))
                        seg = (a, b)
                        if _seg_len(*seg) > 1e-9:
                            segments.append(seg)
                counts["polygon"] += 1
            elif kind == "curve":
                s = _pt_mm(pn, item.GetStart())
                e = _pt_mm(pn, item.GetEnd())
                c1 = _pt_mm(pn, item.GetBezierC1())
                c2 = _pt_mm(pn, item.GetBezierC2())
                pts = polyline_bezier(s, c1, c2, e)
                segs = _polyline_to_segments(pts)
                segments.extend(segs)
                curved_len += sum(_seg_len(a, b) for a, b in segs)
                counts["curve"] += 1
            else:
                counts["other"] += 1
        except Exception:
            counts["skip"] += 1
    return segments, {"counts": counts, "curved_len_mm": curved_len}


def analyze_pcb(pcb_path: str) -> Dict[str, Any]:
    """加载单个 .kicad_pcb 并产出全部扫描指标。

    Args:
        pcb_path: 文件绝对路径

    Returns:
        指标 dict；加载失败会向上抛出异常，由 scan_dir 捕获记为 load_error。
    """
    pn = _get_pcbnew()
    board = pn.LoadBoard(pcb_path)
    if board is None:
        raise RuntimeError("pcbnew.LoadBoard returned None")

    # ---- footprint 层：位号 / pad / net ----
    net_pad_count: Dict[str, int] = {}
    fp_total = 0
    fixed_cnt = 0
    movable = 0
    conv_cnt = 0
    padlike_cnt = 0
    other_cnt = 0
    for fp in board.Footprints():
        fp_total += 1
        try:
            fpid = fp.GetFPID()
            fname = str(fpid.GetLibItemName())
        except Exception:
            fname = ""
        try:
            ref = fp.GetReference()
        except Exception:
            ref = ""
        try:
            is_fixed = _is_fixed(fp, fname.upper(), pn)
        except Exception:
            is_fixed = False
        # net 统计（含固定件，保证不漏 pad）
        try:
            seen = set()
            for pad in fp.Pads():
                nm = pad.GetNetname()
                if nm:
                    if nm not in seen:
                        seen.add(nm)
                        net_pad_count[nm] = net_pad_count.get(nm, 0) + 1
        except Exception:
            pass
        if is_fixed:
            fixed_cnt += 1
            continue
        movable += 1
        kind = classify_ref(ref, fname)
        if kind == "conventional":
            conv_cnt += 1
        elif kind == "pad_like":
            padlike_cnt += 1
        else:
            other_cnt += 1

    net_ge2 = sum(1 for c in net_pad_count.values() if c >= 2)
    net_total = len(net_pad_count)

    # ---- Edge.Cuts：矩形度 ----
    segs, shape_stats = _collect_edge_cuts_segments(board, pn)
    rect_ratio, rst = rect_score(segs)
    edge_total_len = rst["perimeter_mm"]
    curved_frac = (shape_stats["curved_len_mm"] / edge_total_len) if edge_total_len > 0 else 0.0

    return {
        "fp_total": fp_total,
        "fixed_cnt": fixed_cnt,
        "movable_cnt": movable,
        "conv_cnt": conv_cnt,
        "padlike_cnt": padlike_cnt,
        "other_cnt": other_cnt,
        "net_total": net_total,
        "net_ge2_cnt": net_ge2,
        "has_edge_cuts": bool(segs),
        "edge_shape_cnt": sum(shape_stats["counts"].values()),
        "rect_ratio": round(rect_ratio, 4),
        "edge_perimeter_mm": round(edge_total_len, 3),
        "aligned_len_mm": round(rst["aligned_mm"], 3),
        "bbox_w_mm": round(rst["w_mm"], 3),
        "bbox_h_mm": round(rst["h_mm"], 3),
        "curved_len_mm": round(shape_stats["curved_len_mm"], 3),
        "curved_frac": round(curved_frac, 4),
        "shape_kind": shape_stats["counts"],
    }


# ===================== 纯文本 S-expression 解析（无需 pcbnew） =====================
#
# 用途：只读扫描时替代 analyze_pcb（pcbnew.LoadBoard）。理由：
#   1. 快 ~100 倍（纯文本，无需 KiCad DLL），436 板全量 1-3 分钟；
#   2. 零崩溃——个别板会触发 KiCad C++ 层 assert/卡死（坐标转换溢出），
#      pcbnew 版不得不分轮子进程+看门狗；文本解析对这些板正常出指标；
#   3. 扫描需要的全部字段（footprint 名/Reference/locked/pad 的 net 名、
#      Edge.Cuts 的 gr_* 几何）在文本里齐全，坐标本身就是 mm 浮点。
# 与 pcbnew 版输出字段完全一致（用 pcbnew 冒烟板做一致性校验）。

def _sexp_tokens(text: str):
    """S-expr 词法：产出 '(' ')' 或原子（字符串原子已去引号）。"""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            yield c
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n:
                ch = text[j]
                if ch == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if ch == '"':
                    j += 1
                    break
                buf.append(ch)
                j += 1
            yield "".join(buf)
            i = j
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            yield text[i:j]
            i = j


def _sexp_parse(text: str):
    """解析整文件为嵌套 list，返回根节点 ['kicad_pcb', ...]。"""
    stack = [[]]
    for tok in _sexp_tokens(text):
        if tok == "(":
            new = []
            stack[-1].append(new)
            stack.append(new)
        elif tok == ")":
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].append(tok)
    return stack[0][0] if stack[0] else []


def _to_float(s) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _xy(node) -> Optional[Tuple[float, float]]:
    """node like ['start','139.8','101.2'] → (139.8, 101.2) mm。"""
    if not isinstance(node, list) or len(node) < 3:
        return None
    x, y = _to_float(node[1]), _to_float(node[2])
    if x is None or y is None:
        return None
    return (x, y)


def analyze_pcb_text(pcb_path: str) -> Dict[str, Any]:
    """纯文本解析单个 .kicad_pcb，产出与 analyze_pcb 完全一致的指标 dict。

    固定件判定与 kicad_utils 同口径（locked/机械关键词/0 网络）；
    net 统计与 pcbnew 版同口径（每 footprint 每 net 计 1，net_ge2 = 该 net
    被 ≥2 个 footprint 连接）。
    """
    with open(pcb_path, "r", encoding="utf-8", errors="replace") as fh:
        root = _sexp_parse(fh.read())
    if not isinstance(root, list) or not root or root[0] != "kicad_pcb":
        raise RuntimeError("不是有效的 .kicad_pcb（根节点异常）")

    net_pad_count: Dict[str, int] = {}
    fp_total = 0
    fixed_cnt = 0
    movable = 0
    conv_cnt = 0
    padlike_cnt = 0
    other_cnt = 0
    segments: List = []
    counts: Dict[str, int] = {"line": 0, "arc": 0, "circle": 0, "rect": 0,
                              "polygon": 0, "curve": 0, "other": 0, "skip": 0}
    curved_len = 0.0

    _EDGE_TAGS = {"gr_line", "gr_arc", "gr_circle", "gr_rect", "gr_poly",
                  "gr_bezier", "gr_curve"}

    for node in root[1:]:
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            continue
        tag = node[0]

        # ---------- footprint ----------
        if tag == "footprint":
            fp_total += 1
            fname = node[1] if len(node) > 1 else ""
            ref = ""
            locked = False
            pad_nets: List[str] = []
            for sub in node[2:]:
                if not isinstance(sub, list) or not sub:
                    continue
                t = sub[0]
                if t == "property" and len(sub) >= 3 and sub[1] == "Reference":
                    ref = sub[2]
                elif t == "locked":
                    locked = (len(sub) == 1) or (len(sub) > 1 and sub[1] == "yes")
                elif t == "pad":
                    for p in sub[1:]:
                        if not (isinstance(p, list) and p and p[0] == "net"):
                            continue
                        # 两种格式：(net 42 "GND") 原生 / (net "GND") EasyEDA
                        if len(p) >= 3:
                            nm = p[2]
                        elif len(p) == 2:
                            nm = p[1] if not str(p[1]).isdigit() else ""
                        else:
                            nm = ""
                        if nm:
                            pad_nets.append(nm)
            fname_u = fname.upper()
            is_fixed = locked or any(k in fname_u for k in _FIXED_KEYWORDS) \
                or (len(set(pad_nets)) == 0)
            for nm in set(pad_nets):
                net_pad_count[nm] = net_pad_count.get(nm, 0) + 1
            if is_fixed:
                fixed_cnt += 1
                continue
            movable += 1
            kind = classify_ref(ref, fname)
            if kind == "conventional":
                conv_cnt += 1
            elif kind == "pad_like":
                padlike_cnt += 1
            else:
                other_cnt += 1

        # ---------- Edge.Cuts 图形 ----------
        elif tag in _EDGE_TAGS:
            layer = None
            for sub in node[1:]:
                if isinstance(sub, list) and sub and sub[0] == "layer" and len(sub) > 1:
                    layer = sub[1]
                    break
            if layer != "Edge.Cuts":
                continue
            try:
                if tag == "gr_line":
                    s, e = _get_named(node, "start"), _get_named(node, "end")
                    if s and e and _seg_len(s, e) > 1e-9:
                        segments.append((s, e))
                    counts["line"] += 1
                elif tag == "gr_arc":
                    s, e = _get_named(node, "start"), _get_named(node, "end")
                    m = _get_named(node, "mid")
                    if s and e and m:
                        segs = _polyline_to_segments(polyline_arc(s, m, e))
                        segments.extend(segs)
                        curved_len += sum(_seg_len(a, b) for a, b in segs)
                    counts["arc"] += 1
                elif tag == "gr_circle":
                    c = _get_named(node, "center")
                    e = _get_named(node, "end")
                    if c and e:
                        segs = _polyline_to_segments(polyline_circle(c, _seg_len(c, e)))
                        segments.extend(segs)
                        curved_len += sum(_seg_len(a, b) for a, b in segs)
                    counts["circle"] += 1
                elif tag == "gr_rect":
                    s, e = _get_named(node, "start"), _get_named(node, "end")
                    if s and e:
                        corners = [(s[0], s[1]), (e[0], s[1]), (e[0], e[1]), (s[0], e[1])]
                        for i in range(4):
                            a, b = corners[i], corners[(i + 1) % 4]
                            if _seg_len(a, b) > 1e-9:
                                segments.append((a, b))
                    counts["rect"] += 1
                elif tag in ("gr_poly", "gr_bezier", "gr_curve"):
                    pts = []
                    for sub in node[1:]:
                        if isinstance(sub, list) and sub and sub[0] == "pts":
                            for xy in sub[1:]:
                                if (isinstance(xy, list) and xy and xy[0] == "xy"):
                                    p = _xy(xy)
                                    if p is not None:
                                        pts.append(p)
                    if tag == "gr_poly":
                        if len(pts) >= 3:
                            for i in range(len(pts)):
                                a, b = pts[i], pts[(i + 1) % len(pts)]
                                if _seg_len(a, b) > 1e-9:
                                    segments.append((a, b))
                        counts["polygon"] += 1
                    elif tag == "gr_bezier":
                        if len(pts) >= 4:
                            segs = _polyline_to_segments(polyline_bezier(*pts[:4]))
                            segments.extend(segs)
                            curved_len += sum(_seg_len(a, b) for a, b in segs)
                        counts["curve"] += 1
                    else:  # gr_curve：二次贝塞尔，折线近似即可（极少见）
                        if len(pts) >= 2:
                            segments.extend(_polyline_to_segments(pts))
                        counts["curve"] += 1
            except Exception:  # noqa: BLE001
                counts["skip"] += 1

    net_ge2 = sum(1 for c in net_pad_count.values() if c >= 2)
    net_total = len(net_pad_count)
    rect_ratio, rst = rect_score(segments)
    edge_total_len = rst["perimeter_mm"]
    curved_frac = (curved_len / edge_total_len) if edge_total_len > 0 else 0.0

    return {
        "fp_total": fp_total,
        "fixed_cnt": fixed_cnt,
        "movable_cnt": movable,
        "conv_cnt": conv_cnt,
        "padlike_cnt": padlike_cnt,
        "other_cnt": other_cnt,
        "net_total": net_total,
        "net_ge2_cnt": net_ge2,
        "has_edge_cuts": bool(segments),
        "edge_shape_cnt": sum(counts.values()),
        "rect_ratio": round(rect_ratio, 4),
        "edge_perimeter_mm": round(edge_total_len, 3),
        "aligned_len_mm": round(rst["aligned_mm"], 3),
        "bbox_w_mm": round(rst["w_mm"], 3),
        "bbox_h_mm": round(rst["h_mm"], 3),
        "curved_len_mm": round(curved_len, 3),
        "curved_frac": round(curved_frac, 4),
        "shape_kind": counts,
    }


def _get_named(node, name: str) -> Optional[Tuple[float, float]]:
    """在 gr_* 节点里找 (name x y) 子项 → (x, y) mm。"""
    for sub in node[1:]:
        if isinstance(sub, list) and sub and sub[0] == name:
            return _xy(sub)
    return None


# ===================== 条件判定 / 分档 =====================

def passes(metric: Dict[str, Any]) -> bool:
    """是否同时满足四项硬条件。"""
    m = metric
    ok_range = MOVABLE_MIN <= m["movable_cnt"] <= MOVABLE_MAX
    ok_conv = m["conv_cnt"] >= CONV_ABS_MIN and m["conv_cnt"] >= CONV_RATIO_MIN * m["movable_cnt"]
    ok_net = m["net_ge2_cnt"] >= NET_MIN
    ok_rect = bool(m["has_edge_cuts"]) and m["rect_ratio"] >= RECT_RATIO_MIN
    return ok_range and ok_conv and ok_net and ok_rect


def pass_flags(metric: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "ok_range": MOVABLE_MIN <= metric["movable_cnt"] <= MOVABLE_MAX,
        "ok_conv": metric["conv_cnt"] >= CONV_ABS_MIN
        and metric["conv_cnt"] >= CONV_RATIO_MIN * metric["movable_cnt"],
        "ok_net": metric["net_ge2_cnt"] >= NET_MIN,
        "ok_rect": bool(metric["has_edge_cuts"]) and metric["rect_ratio"] >= RECT_RATIO_MIN,
    }


def tier_of(movable_cnt: int) -> Optional[Tuple[int, int]]:
    for lo, hi in TIERS:
        if lo <= movable_cnt <= hi:
            return (lo, hi)
    return None


def select_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """分档 + 源项目去重贪心选取，总上限 MAX_CANDIDATES。

    分 3 阶段：
      1) 每档内优先取「不同源项目」的最优板，取满档配额（TIER_QUOTA）；
      2) 有档未满时，从该档剩余候选中补足配额（放宽项目去重，但仍优先新项目）；
      3) 总数未满 20 时，按质量全局补足。
    源项目分组键 = 路径首段目录中首个 `--` 前缀。
    """
    passing = [r for r in rows if passes(r)]

    def quality(r):
        # 质量：矩形度 > 常规元件数 > 有效 net 数（越高越优）
        return (r["rect_ratio"], r["conv_cnt"], r["net_ge2_cnt"])

    def tier_rows(lo, hi):
        return sorted([r for r in passing if lo <= r["movable_cnt"] <= hi],
                      key=quality, reverse=True)

    chosen: List[Dict[str, Any]] = []
    chosen_per_tier: Dict[Tuple[int, int], int] = {}

    def _take(r):
        chosen.append(r)
        key = tier_of(r["movable_cnt"])
        chosen_per_tier[key] = chosen_per_tier.get(key, 0) + 1

    # 阶段 1：每档取「每源项目最优 1 块」，直到满配额或档内耗尽
    for lo, hi in TIERS:
        rows_tier = tier_rows(lo, hi)
        by_group: Dict[str, List] = {}
        for r in rows_tier:
            by_group.setdefault(r["src_group"], []).append(r)
        # 组按组内最优排序
        for g in sorted(by_group, key=lambda g: quality(by_group[g][0]), reverse=True):
            if chosen_per_tier.get((lo, hi), 0) >= TIER_QUOTA or len(chosen) >= MAX_CANDIDATES:
                break
            _take(by_group[g][0])

    # 阶段 2：配额未满的档，从剩余板里补足（优先未出现过的项目，其次高质量）
    if len(chosen) < MAX_CANDIDATES:
        for lo, hi in TIERS:
            if chosen_per_tier.get((lo, hi), 0) >= TIER_QUOTA:
                continue
            used = {r["src_group"] for r in chosen}
            rest = [r for r in tier_rows(lo, hi) if r not in chosen]
            rest.sort(key=lambda r: (r["src_group"] not in used, quality(r)), reverse=True)
            for r in rest:
                if chosen_per_tier.get((lo, hi), 0) >= TIER_QUOTA or len(chosen) >= MAX_CANDIDATES:
                    break
                _take(r)

    # 阶段 3：仍不足 20，全局按质量 + 项目新颖度补足
    if len(chosen) < MAX_CANDIDATES:
        used = {r["src_group"] for r in chosen}
        rest = [r for r in passing if r not in chosen]
        rest.sort(key=lambda r: (r["src_group"] not in used, quality(r)), reverse=True)
        for r in rest:
            if len(chosen) >= MAX_CANDIDATES:
                break
            _take(r)
    return chosen


def src_group_of(file_abs_path: str, src_dir: str) -> str:
    """源项目分组键：取相对路径首段目录中首个 `--` 前缀；否则整个首段目录名。"""
    try:
        rel = os.path.relpath(file_abs_path, src_dir)
    except ValueError:
        rel = os.path.basename(file_abs_path)
    first_dir = rel.split(os.sep)[0]
    return first_dir.split("--")[0]


def _row_to_public(metric: Dict[str, Any], src_dir: str) -> Dict[str, Any]:
    """把分析结果转成对外行记录（含文件路径 / 条件标记）。"""
    row = dict(metric)
    row["file"] = metric["_path"]
    row["rel"] = metric["_rel"]
    row["src_group"] = src_group_of(metric["_path"], src_dir)
    row["name"] = os.path.basename(metric["_path"])
    flags = pass_flags(metric)
    row.update(flags)
    row["pass"] = passes(metric)
    for k in ("_path", "_rel"):
        row.pop(k, None)
    return row


# ===================== 阶段 A：全量扫描 =====================

_SCAN_FIELDS = [
    "file", "rel", "src_group", "name",
    "fp_total", "fixed_cnt", "movable_cnt", "conv_cnt", "padlike_cnt", "other_cnt",
    "net_total", "net_ge2_cnt",
    "has_edge_cuts", "edge_shape_cnt", "rect_ratio", "edge_perimeter_mm",
    "aligned_len_mm", "bbox_w_mm", "bbox_h_mm", "curved_len_mm", "curved_frac",
    "ok_range", "ok_conv", "ok_net", "ok_rect", "pass",
]


def _default_shape_kind():
    return {"line": 0, "arc": 0, "circle": 0, "rect": 0, "polygon": 0, "curve": 0,
            "other": 0, "skip": 0}


# ---- 分轮子进程（每轮一次 pcbnew 导入，崩溃自动重跑续扫）----

_PART_ROWS = "scan_partial_rows.jsonl"
_PART_ERRS = "scan_partial_errors.jsonl"
_CURRENT_FILE = "scan_current.json"
_ROUND_DONE = "__KBS_ROUND_DONE__"


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_partial(out_dir: str):
    """读取断点续扫文件，返回 (rows, errors, done_rels, err_files)。"""
    rows = []
    errors = []
    done_rels = set()
    err_files = set()
    p_rows = os.path.join(out_dir, _PART_ROWS)
    p_errs = os.path.join(out_dir, _PART_ERRS)
    if os.path.exists(p_rows):
        with open(p_rows, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(obj)
                key = obj.get("rel") or obj.get("file")
                if key:
                    done_rels.add(key)
    if os.path.exists(p_errs):
        with open(p_errs, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                errors.append(obj)
                if obj.get("file"):
                    err_files.add(obj["file"])
    return rows, errors, done_rels, err_files


def _write_current(out_dir: str, path: str) -> None:
    """记录正在处理的文件（用于崩溃后定位罪魁祸首）。"""
    with open(os.path.join(out_dir, _CURRENT_FILE), "w", encoding="utf-8") as fh:
        json.dump({"file": path}, fh, ensure_ascii=False)


def _clear_current(out_dir: str) -> None:
    p = os.path.join(out_dir, _CURRENT_FILE)
    if os.path.exists(p):
        os.remove(p)


def _read_current(out_dir: str) -> Optional[str]:
    p = os.path.join(out_dir, _CURRENT_FILE)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("file")
    except Exception:  # noqa: BLE001
        _clear_current(out_dir)
        return None


def _round_command(root_dir: str) -> List[str]:
    """构造一轮扫描的子进程命令（该轮内只导入一次 pcbnew）。"""
    snippet = (
        "import sys;"
        f"sys.path.insert(0, {root_dir!r});"
        "from utils.kicad_board_scan import scan_round_entry;"
        "sys.exit(scan_round_entry(sys.argv[1], sys.argv[2], int(sys.argv[3])))"
    )
    return [sys.executable, "-u", "-c", snippet]


def scan_round_entry(src_dir: str, out_dir: str, progress_every: int = 10) -> int:
    """单轮进程内扫描（可被 KiCad 原生 assert 终止，由父进程 orchestrate 续扫）。

    本函数只负责：读完 partial → 逐个 LoadBoard → 成功/失败都即时追加 jsonl。
    崩溃（进程被杀）前正在处理的文件会留在 scan_current.json，由父进程标记跳过。
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for root, _dirs, fnames in os.walk(src_dir):
        for f in fnames:
            if f.lower().endswith(".kicad_pcb"):
                files.append(os.path.join(root, f))
    files.sort()
    total = len(files)
    rows, errors, done_rels, err_files = _load_partial(out_dir)
    p_rows = os.path.join(out_dir, _PART_ROWS)
    p_errs = os.path.join(out_dir, _PART_ERRS)

    ok_count = len(rows)
    for i, path in enumerate(files, 1):
        try:
            rel = os.path.relpath(path, src_dir)
        except ValueError:
            rel = path
        if rel in done_rels or path in err_files:
            continue
        _write_current(out_dir, path)
        try:
            metric = analyze_pcb(path)
        except Exception as exc:  # noqa: BLE001（普通 python 异常，非崩溃）
            err = {"file": path, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(err)
            _append_jsonl(p_errs, err)
            _clear_current(out_dir)
            continue
        metric["_path"] = path
        metric["_rel"] = rel
        row = _row_to_public(metric, src_dir)
        rows.append(row)
        ok_count += 1
        _append_jsonl(p_rows, row)
        _clear_current(out_dir)
        if i % progress_every == 0 or i == total:
            print(f"[scan] {i}/{total}  ok={ok_count} failed={len(errors)}", flush=True)
    print(_ROUND_DONE, flush=True)
    return 0


def _wait_with_stall_watch(proc: subprocess.Popen, out_dir: str,
                           stall_s: float, poll_s: float = 2.0) -> Tuple[Optional[int], bool]:
    """等待轮子进程；若 current 标记同一文件停留超 stall_s 判定为卡死。

    注意：current 标记不存在（pcbnew 导入期 / 轮收尾期）时不计停滞，
    避免把正常的初始化耗时误判为卡死导致无限循环。

    Returns:
        (returncode, stalled)：stalled=True 表示因单文件停滞被本函数 kill。
    """
    marker_p = os.path.join(out_dir, _CURRENT_FILE)
    last_key = None
    last_change = None
    import time as _time
    while True:
        try:
            rc = proc.wait(timeout=poll_s)
            return rc, False
        except subprocess.TimeoutExpired:
            pass
        key = None
        if os.path.exists(marker_p):
            try:
                with open(marker_p, encoding="utf-8") as fh:
                    f = json.load(fh).get("file")
                key = (f, os.path.getmtime(marker_p))
            except Exception:  # noqa: BLE001
                key = None
        if key is None:
            # 无 current（导入/收尾阶段）：不算停滞，等待首个文件标记
            last_key = None
            last_change = None
            continue
        if key != last_key:
            last_key = key
            last_change = _time.time()
        elif last_change is not None and _time.time() - last_change > stall_s:
            proc.kill()
            proc.wait()
            return None, True


def scan_dir(src_dir: str, out_dir: str, progress_every: int = 25,
             **_: Any) -> Dict[str, int]:
    """全量扫描（**纯文本 S-expression 解析**，无需 KiCad，默认路线）。

    与 scan_dir_pcbnew 等效输出，但：
      - 快 ~100 倍：纯文本解析 436 板约 1-3 分钟，无需 KiCad DLL / 用户终端；
      - 零崩溃：不会触发 KiCad C++ assert/卡死，坏文件普通 try/except 记错；
      - 含 partial 断点续扫（中断重跑跳过已完成文件）。
    输出写入 out_dir（不写源目录）。
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for root, _dirs, fnames in os.walk(src_dir):
        for f in fnames:
            if f.lower().endswith(".kicad_pcb"):
                files.append(os.path.join(root, f))
    files.sort()
    total = len(files)

    rows, errors, done_rels, err_files = _load_partial(out_dir)
    p_rows = os.path.join(out_dir, _PART_ROWS)
    p_errs = os.path.join(out_dir, _PART_ERRS)

    ok_count = len(rows)
    for i, path in enumerate(files, 1):
        try:
            rel = os.path.relpath(path, src_dir)
        except ValueError:
            rel = path
        if rel in done_rels or path in err_files:
            continue
        try:
            metric = analyze_pcb_text(path)
        except Exception as exc:  # noqa: BLE001
            err = {"file": path, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(err)
            _append_jsonl(p_errs, err)
            continue
        metric["_path"] = path
        metric["_rel"] = rel
        row = _row_to_public(metric, src_dir)
        rows.append(row)
        ok_count += 1
        _append_jsonl(p_rows, row)
        if i % progress_every == 0 or i == total:
            print(f"[scan] {i}/{total}  ok={ok_count} failed={len(errors)}", flush=True)

    rows.sort(key=lambda r: r["rel"])  # 确定性输出顺序（断点续扫可能乱序）
    meta = {
        "src_dir": src_dir,
        "total": total,
        "ok": ok_count,
        "failed": len(errors),
        "engine": "sexp-text",
        "thresholds": {
            "RECT_RATIO_MIN": RECT_RATIO_MIN,
            "EDGE_TOL_MM": EDGE_TOL_MM,
            "MOVABLE_MIN": MOVABLE_MIN,
            "MOVABLE_MAX": MOVABLE_MAX,
            "CONV_ABS_MIN": CONV_ABS_MIN,
            "CONV_RATIO_MIN": CONV_RATIO_MIN,
            "NET_MIN": NET_MIN,
            "TIERS": list(TIERS),
            "TIER_QUOTA": TIER_QUOTA,
            "MAX_CANDIDATES": MAX_CANDIDATES,
        },
    }
    _write_scan_outputs(meta, rows, errors, out_dir)
    for f in (_PART_ROWS, _PART_ERRS, _CURRENT_FILE):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    print(f"[scan] done: total={total} ok={meta['ok']} failed={meta['failed']}", flush=True)
    return meta


def scan_dir_pcbnew(src_dir: str, out_dir: str, progress_every: int = 10,
                    round_timeout: float = 1800.0, max_rounds: int = 200,
                    stall_s: float = 90.0) -> Dict[str, int]:
    """全量扫描（pcbnew 版，备用/对照）：「分轮子进程 + 单文件停滞看门狗」。

    背景：个别异常板（坐标超 int 范围）会触发 KiCad 原生 assert / 内部卡死，
    无法被 python 捕获。策略：
      1. 每轮是一个独立子进程（一轮只导入一次 pcbnew，避免逐板导入的巨额开销）；
      2. 每块处理前写 scan_current.json、成功/失败即时追加 partial jsonl；
      3. 看门狗监视 current 标记：同一文件停留超过 stall_s 秒 → kill 该轮，
         把该文件记 load_error 后自动起下一轮续扫；
      4. 子进程异常退出且未打印完成哨兵 → 同样定位 current 记 load_error 续扫；
         直到某轮打印哨兵 = 全量完成。
    输出写入 out_dir（不写源目录）。子进程缺省 KICAD_PATH 回退 E:\\Kicad。
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for root, _dirs, fnames in os.walk(src_dir):
        for f in fnames:
            if f.lower().endswith(".kicad_pcb"):
                files.append(os.path.join(root, f))
    files.sort()
    total = len(files)

    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    cmd = _round_command(root_dir)
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    if not child_env.get("KICAD_PATH") and os.path.isdir(_DEFAULT_KICAD_PATH):
        child_env["KICAD_PATH"] = _DEFAULT_KICAD_PATH
    p_errs = os.path.join(out_dir, _PART_ERRS)
    log_p = os.path.join(out_dir, "_round.log")

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        rc = None
        stalled = False
        with open(log_p, "w", encoding="utf-8", errors="replace") as fout:
            proc = subprocess.Popen(  # noqa: S603
                cmd + [src_dir, out_dir, str(progress_every)],
                stdout=fout,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
            try:
                rc, stalled = _wait_with_stall_watch(proc, out_dir, stall_s=stall_s)
            except Exception:  # noqa: BLE001
                proc.kill()
                proc.wait()
        try:
            with open(log_p, encoding="utf-8", errors="replace") as fh:
                log_text = fh.read()
        except Exception:  # noqa: BLE001
            log_text = ""
        if os.path.exists(log_p):
            try:
                os.remove(log_p)
            except OSError:
                pass
        if not stalled and rc == 0 and _ROUND_DONE in log_text:
            break  # 正常完成一轮
        # 异常：崩溃或卡死 → 定位 current 文件并记为 load_error 后续扫
        cur = _read_current(out_dir)
        reason = "单文件停滞超过看门狗阈值(killed)" if stalled else f"异常退出(exit={rc})"
        if cur:
            err = {"file": cur, "error": reason + "：KiCad 原生崩溃/卡死（LoadBoard 阶段）"}
            _append_jsonl(p_errs, err)
            print(f"[scan] 第 {rounds} 轮{reason}，隔离文件: {os.path.basename(cur)}", flush=True)
        else:
            print(f"[scan] 第 {rounds} 轮{reason}，自动续扫…", flush=True)
        _clear_current(out_dir)
        # 中间态进度
        try:
            rows_now, errs_now, _, _ = _load_partial(out_dir)
            print(f"[scan] partial ok={len(rows_now)} failed={len(errs_now)} / {total}", flush=True)
        except Exception:  # noqa: BLE001
            pass
    else:
        print(f"[scan] 达到轮次上限 {max_rounds}，强制结束（partial 可续扫）", flush=True)

    rows, errors, _, _ = _load_partial(out_dir)
    meta = {
        "src_dir": src_dir,
        "total": total,
        "ok": len(rows),
        "failed": len(errors),
        "rounds": rounds,
        "thresholds": {
            "RECT_RATIO_MIN": RECT_RATIO_MIN,
            "EDGE_TOL_MM": EDGE_TOL_MM,
            "MOVABLE_MIN": MOVABLE_MIN,
            "MOVABLE_MAX": MOVABLE_MAX,
            "CONV_ABS_MIN": CONV_ABS_MIN,
            "CONV_RATIO_MIN": CONV_RATIO_MIN,
            "NET_MIN": NET_MIN,
            "TIERS": list(TIERS),
            "TIER_QUOTA": TIER_QUOTA,
            "MAX_CANDIDATES": MAX_CANDIDATES,
        },
    }
    _write_scan_outputs(meta, rows, errors, out_dir)
    # 清理断点续扫临时文件
    for f in (_PART_ROWS, _PART_ERRS, _CURRENT_FILE):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    print(f"[scan] done: total={total} ok={meta['ok']} failed={meta['failed']} rounds={rounds}", flush=True)
    return meta


def _write_scan_outputs(meta: Dict[str, Any], rows: List[Dict[str, Any]],
                        errors: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scan_all.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "scan_all.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SCAN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(os.path.join(out_dir, "load_errors.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "error"])
        writer.writeheader()
        for e in errors:
            writer.writerow(e)


# ===================== 阶段 B：加载 / 报告 =====================

def load_rows(out_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """读取 scan_all.json，返回 (meta, rows)。"""
    with open(os.path.join(out_dir, "scan_all.json"), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("meta", {}), data.get("rows", [])


def rationale_of(r: Dict[str, Any]) -> str:
    """单块候选的入选理由（供报告/CSV 复用）。"""
    t = tier_of(r["movable_cnt"])
    tier_s = _fmt_tier(t) if t else "-"
    return (
        f"可动 {r['movable_cnt']} 在分档 {tier_s} 均衡覆盖内；"
        f"常规元件 {r['conv_cnt']}(占比 {r['conv_cnt'] / r['movable_cnt']:.0%})、"
        f"有效 net {r['net_ge2_cnt']}、矩形度 {r['rect_ratio']:.2f}"
    )


def select_and_save(rows: List[Dict[str, Any]], out_dir: str) -> List[Dict[str, Any]]:
    candidates = select_candidates(rows)
    for r in candidates:
        r["rationale"] = rationale_of(r)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, ensure_ascii=False, indent=1)
    _CAND_FIELDS = _SCAN_FIELDS + ["rationale"]
    with open(os.path.join(out_dir, "candidates.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CAND_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in candidates:
            writer.writerow(r)
    return candidates


def _fmt_tier(lo_hi) -> str:
    return f"{lo_hi[0]}-{lo_hi[1]}"


def render_report(meta: Dict[str, Any], rows: List[Dict[str, Any]],
                  out_dir: str, candidates: Optional[List[Dict[str, Any]]] = None) -> str:
    """渲染 report.md 并落盘；返回报告文本。"""
    os.makedirs(out_dir, exist_ok=True)
    total = len(rows)
    failed_rows = []
    try:
        with open(os.path.join(out_dir, "load_errors.csv"), "r", encoding="utf-8-sig") as fh:
            failed_rows = list(csv.DictReader(fh))
    except Exception:
        failed_rows = []

    passing = [r for r in rows if r.get("pass")]
    cands = candidates if candidates is not None else []

    # 单项通过率
    def count(flag):
        return sum(1 for r in rows if r.get(flag))

    n_range, n_conv, n_net, n_rect = count("ok_range"), count("ok_conv"), count("ok_net"), count("ok_rect")

    lines = []
    lines.append("# KiCad 数据集板卡筛选报告")
    lines.append("")
    lines.append("## 0. 总览")
    lines.append("")
    lines.append(f"- 扫描目录：`{meta.get('src_dir', '')}`")
    lines.append(f"- 扫描到的 `.kicad_pcb` 总数：**{meta.get('total', total)}**；加载成功 **{meta.get('ok', total)}**；加载失败 **{meta.get('failed', len(failed_rows))}**")
    lines.append(f"- 全量指标落盘：`scan_all.csv` / `scan_all.json`；加载失败清单：`load_errors.csv`")
    lines.append(f"- 通过全部硬条件（可作 RL 数据集候选）的板：**{len(passing)}**")
    lines.append(f"- 分档均衡选取候选：**{len(cands)}** 块（上限 {MAX_CANDIDATES}）")
    lines.append("")

    lines.append("## 1. 判定阈值（可调，改后重跑 report 即可）")
    lines.append("")
    lines.append("| 阈值 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| RECT_RATIO_MIN | {RECT_RATIO_MIN} | 板框矩形度下限（对齐边长/总周长），允许圆角倒角 |")
    lines.append(f"| EDGE_TOL_MM | {EDGE_TOL_MM} mm | 线段贴齐 bbox 边线容差 |")
    lines.append(f"| MOVABLE_MIN / MAX | {MOVABLE_MIN} / {MOVABLE_MAX} | 非固定元件数区间 |")
    lines.append(f"| CONV_ABS_MIN | {CONV_ABS_MIN} | 常规可动元件绝对下限 |")
    lines.append(f"| CONV_RATIO_MIN | {CONV_RATIO_MIN} | 常规可动元件占比下限 |")
    lines.append(f"| NET_MIN | {NET_MIN} | 有效 net(≥2 pad) 下限（要求 >3） |")
    lines.append(f"| TIERS | {'; '.join(_fmt_tier(t) for t in TIERS)} | 元件数分档（均衡覆盖） |")
    lines.append(f"| TIER_QUOTA / MAX_CANDIDATES | {TIER_QUOTA} / {MAX_CANDIDATES} | 每档配额 / 候选上限 |")
    lines.append("")

    lines.append("## 2. 各条件淘汰情况（单条件通过数）")
    lines.append("")
    lines.append(f"- ① 板框大致矩形（rect_ratio ≥ {RECT_RATIO_MIN} 且存在 Edge.Cuts）：**{n_rect} / {total}**")
    lines.append(f"- ② 非固定元件 ∈ [{MOVABLE_MIN}, {MOVABLE_MAX}]：**{n_range} / {total}**")
    lines.append(f"- ②' 含常规元件（≥{CONV_ABS_MIN} 且占比 ≥{CONV_RATIO_MIN}）：**{n_conv} / {total}**")
    lines.append(f"- ③ 有效 net > 3：**{n_net} / {total}**")
    lines.append(f"- 四项全过：**{len(passing)} / {total}**")
    lines.append("")

    # 矩形度分布
    lines.append("### 矩形度 (rect_ratio) 分布")
    lines.append("")
    lines.append("| rect_ratio 区间 | 数量 |")
    lines.append("|---|---|")
    for lo, hi in [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 0.97), (0.97, 1.001)]:
        c = sum(1 for r in rows if lo <= r.get("rect_ratio", 0) < hi)
        lines.append(f"| {lo:.2f} ~ {hi:.2f} | {c} |")
    lines.append("")

    # 元件数分布
    lines.append("### 可动元件数分布（全部板 / 通过板）")
    lines.append("")
    lines.append("| 区间 | 全部 | 通过 |")
    lines.append("|---|---|---|")
    for lo, hi in TIERS:
        a = sum(1 for r in rows if lo <= r.get("movable_cnt", 0) <= hi)
        b = sum(1 for r in passing if lo <= r.get("movable_cnt", 0) <= hi)
        lines.append(f"| {lo}-{hi} | {a} | {b} |")
    lines.append("")

    lines.append("## 3. 候选清单（≤ 20 块）")
    lines.append("")
    if not cands:
        lines.append("（无可通过全部条件的板，请放宽阈值后重跑 report。）")
    else:
        lines.append("| # | 板名 | 元件数(可动/固定/总) | 常规可动 | rect_ratio | net(≥2) | 尺寸 mm | 分档 | 源项目 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(cands, 1):
            tier = _fmt_tier(tier_of(r["movable_cnt"])) if tier_of(r["movable_cnt"]) else "-"
            lines.append(
                f"| {i} | {r['name']} | {r['movable_cnt']}/{r['fixed_cnt']}/{r['fp_total']} "
                f"| {r['conv_cnt']} | {r['rect_ratio']:.3f} | {r['net_ge2_cnt']} "
                f"| {r['bbox_w_mm']:.1f}×{r['bbox_h_mm']:.1f} | {tier} | {r['src_group']} |"
            )
        lines.append("")
        lines.append("### 每块入选理由 / 关键指标")
        lines.append("")
        for i, r in enumerate(cands, 1):
            tier = _fmt_tier(tier_of(r["movable_cnt"])) if tier_of(r["movable_cnt"]) else "-"
            lines.append(f"{i}. **{r['name']}**")
            lines.append(f"   - 路径：`{r['rel']}`")
            lines.append(f"   - 元件：可动 {r['movable_cnt']}（常规 {r['conv_cnt']} / pad-like {r['padlike_cnt']} / other {r['other_cnt']}），固定 {r['fixed_cnt']}，总 {r['fp_total']}")
            lines.append(f"   - net：去重 {r['net_total']}，有效(≥2 pad) {r['net_ge2_cnt']}")
            lines.append(f"   - 板框：Edge.Cuts 周长 {r['edge_perimeter_mm']:.1f} mm，bbox {r['bbox_w_mm']:.1f}×{r['bbox_h_mm']:.1f} mm，矩形度 {r['rect_ratio']:.3f}（弧长占比 {r['curved_frac']:.2%}）")
            lines.append(f"   - 分档：{tier}（均衡覆盖，源项目 {r['src_group']}）")
            lines.append("")

    report_text = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report_text)
    return report_text
