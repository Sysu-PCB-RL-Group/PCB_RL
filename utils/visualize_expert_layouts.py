"""专家布局可视化脚本（训练前预览真实 KiCad 板）。

将 `data/kicad-native-batch/` 下的真实 KiCad 板渲染为可视化图片，每块板输出
**一张** PNG（左=顶层 F.Cu、右=底层 B.Cu 左右并排），便于在训练前直观查看
专家布局信息。

渲染元素与配色（由需求指定）：
    - 板框      ：矩形描边（Edge.Cuts 层包围盒）
    - 元件本体  ：浅蓝色矩形（可移动元件）/ 浅黄色矩形（固定元件）
                  本体语义 = **component 本身的矩形尺寸**：
                  `GetBoundingHull().BBox()`，与训练/碰撞检测的
                  `Component.bbox` 完全同口径（不含 Courtyard 的装配余量）
    - 焊盘      ：浅绿色矩形（**从属焊盘**，即属于某个元器件的 pad）；
                  **实心 = 本层专属**（仅该层有铜）、**空心 = 贯穿焊盘 PTH**（两层都有铜）
    - 禁布区    ：红色矩形（rule area，有则画；**按层分流**——F.Cu 的只画顶层、
                  B.Cu 的只画底层、Edge.Cuts 等非铜层两图都画）
    - 过孔      ：红色矩形（按直径近似，有则画）
    - 元件编号  ：黑色细字标注

独立焊盘（EasyEDA/立创 EDA 的"自由焊盘"，转换后成为匿名 footprint：Reference 为空
且 `GraphicalItems()` 为空）按**固定元件本体**渲染（浅黄色）；其自身焊盘不再重复画
绿色框——因为它"焊盘即本体"，画绿框会盖住黄框、丢失"固定"语义。

坐标轴按 KiCad 原生坐标（Y 轴向下），次刻度 0.1mm，主刻度按板尺寸自适应。
板框与过孔为跨层元素，两个子图均绘制；元件、焊盘、禁布区均按所在层分流
（PTH 焊盘两层均出现；`Edge.Cuts` 层的板边禁布区两层都显示）。
图例统一置于整图底部（图外），避免遮挡板内元件。

设计说明：
    本脚本**直接依赖 pcbnew** 读取 .kicad_pcb，不走 `utils/kicad_utils`。
    原因：kicad_utils 的 Component 结构只提供元件整体 AABB（含焊盘）与焊盘
    中心点，无法满足「非焊盘本体矩形」「焊盘尺寸」「过孔」这三项本次所需数据。

用法（需在已配置 KiCad 的终端运行）：
    python utils/visualize_expert_layouts.py --limit 2      # 先行验证前 2 块
    python utils/visualize_expert_layouts.py                # 全量
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# 项目根加入 sys.path：脚本位于 utils/ 子目录，需保证 `import utils.*` 可用
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger("visualize_expert_layouts")

# ==================== 配色方案（需求指定） ====================

COLOR_COMP_MOVABLE_FACE = "#BBDEFB"   # 可移动元件本体：浅蓝
COLOR_COMP_MOVABLE_EDGE = "#1E88E5"
COLOR_COMP_FIXED_FACE = "#FFF9C4"     # 固定元件本体：浅黄
COLOR_COMP_FIXED_EDGE = "#F9A825"

# 本体填充不透明度：可动元件（浅蓝）调低，使色块更淡，
# 避免大面积蓝色压制焊盘与编号的可读性；固定元件保持原有观感
_MOVABLE_FACE_ALPHA = 0.45
_FIXED_FACE_ALPHA = 0.9
COLOR_PAD_FACE = "#C8E6C9"            # 焊盘：浅绿
COLOR_PAD_EDGE = "#43A047"
COLOR_FORBID_EDGE = "#E53935"         # 禁布区 / 过孔：红
COLOR_FORBID_FACE = "#FFCDD2"

COLOR_BOARD = "#333333"               # 板框描边
COLOR_BOARD_BG = "#FAFAF7"            # 板框底色
COLOR_GRID_MAJOR = "#D5D5CC"
COLOR_GRID_MINOR = "#EEEEE8"
COLOR_ID = "#000000"                  # 元件编号：黑色细字

# 坐标轴次刻度固定 0.1mm（需求指定的"颗粒度"）
_TICK_MINOR_STEP = 0.1

# 固定元件判定关键词（与 utils/kicad_utils 的启发式保持一致）
_FIXED_FOOTPRINT_KEYWORDS = ("HOLE", "MOUNT", "MECH", "FIDUCIAL")

# ==================== 中文字体设置 ====================

_FONT_SET = False


def _setup_chinese_font():
    """全局设置 matplotlib 中文字体（仅首次调用生效）。

    自带实现而非复用 utils.visualization，避免经其顶部
    `from env.pcb_layout_env import Config` 拉起整个 env 包（含 KiCad 依赖链），
    使本脚本保持"独立、仅依赖 pcbnew"的定位。
    """
    global _FONT_SET
    if _FONT_SET:
        return
    # 一次性把全部候选塞进列表，由 matplotlib 按序回退到第一个可用字体。
    # 注意：不能写成"逐个赋值 + try/except"——rcParams 赋值不校验字体是否存在、
    # 也不会抛异常，那种写法会在首次迭代就返回，后续候选永远不会被尝试。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_SET = True


# ==================== pcbnew 延迟加载 ====================

_pcbnew = None


def _load_pcbnew():
    """按 KICAD_PATH 配置 DLL 搜索路径后导入 pcbnew。

    Returns:
        已导入的 pcbnew 模块。

    Raises:
        RuntimeError: KICAD_PATH 未设置或 pcbnew 导入失败。
    """
    kicad_path = os.environ.get("KICAD_PATH", "")
    if not kicad_path:
        raise RuntimeError(
            "环境变量 KICAD_PATH 未设置。请先设置，例如：\n"
            r'    $env:KICAD_PATH="E:\Kicad"'
        )
    bin_dir = os.path.join(kicad_path, "bin")
    if hasattr(os, "add_dll_directory") and os.path.isdir(bin_dir):
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            pass
    for extra in (os.path.join(bin_dir, "Lib", "site-packages"), bin_dir):
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.insert(0, extra)
    import pcbnew  # noqa: F401  (延迟导入：必须先配好 DLL 路径)
    _silence_wx_log()
    return pcbnew


def _silence_wx_log():
    """把 wxWidgets 日志重定向到 stderr，避免缺字体时弹出**模态对话框**。

    背景：部分 EasyEDA/立创导出的板子里，文本对象指定了系统不存在的字体
    （如 "Arial Bold"）。wxWidgets 会以 wxLogWarning 报
    "Font ... not found; substituting ..."；在默认 GUI 日志目标下这会弹出
    模态对话框并**阻塞批处理**（每块板都可能弹一次，需人工点"确定"）。

    KiCad 的 pcbnew 与 Python 的 wx 绑定共用同一份 wxWidgets 运行时，
    因此改动全局 wxLog 目标即可同时影响 KiCad 内部的日志输出。
    """
    try:
        import wx
        wx.Log.SetActiveTarget(wx.LogStderr())
        # 字体替换属 Warning 级噪音，过滤掉只保留 Error
        wx.Log.SetLogLevel(wx.LOG_Error)
    except Exception:
        # wx 不可用（非 KiCad 自带解释器）时静默跳过，不影响主流程
        pass


def _pcb():
    """返回已加载的 pcbnew 模块（惰性初始化）。"""
    global _pcbnew
    if _pcbnew is None:
        _pcbnew = _load_pcbnew()
    return _pcbnew


# ==================== 元素提取 ====================


def _merge_boxes(
    boxes: List[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float, float, float]]:
    """合并多个 AABB 为最小外接盒；空列表返回 None。"""
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _shape_bbox_mm(item) -> Optional[Tuple[float, float, float, float]]:
    """PCB_SHAPE 的包围盒 → mm 四元组；零尺寸（退化）返回 None。"""
    pcb = _pcb()
    try:
        box = item.GetBoundingBox()
        b = (
            pcb.ToMM(box.GetLeft()),
            pcb.ToMM(box.GetTop()),
            pcb.ToMM(box.GetRight()),
            pcb.ToMM(box.GetBottom()),
        )
    except Exception:
        return None
    # 退化盒（点/零长线）无绘制意义，跳过
    if b[2] - b[0] <= 1e-9 and b[3] - b[1] <= 1e-9:
        return None
    return b


def _iter_items(obj, names: Tuple[str, ...]) -> list:
    """按候选 API 名依次尝试取子元素列表（跨 KiCad 版本兼容）。

    obj 可为 FOOTPRINT（Pads / GraphicalItems）或 BOARD（Footprints）。
    """
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return list(fn())
        except Exception:
            continue
    return []


def get_board_rect(board) -> Optional[Tuple[float, float, float, float]]:
    """取板框矩形：Edge.Cuts 层所有图形的最小外接盒。

    局限：这是**外接矩形近似**。本批 20 块板框均为矩形，误差为 0；
    若板内有开槽/挖孔/异形轮廓，图上的板框会填满为一个完整矩形，
    丢失"该处为空"的信息（那一带同样不可放件）。

    Args:
        board: pcbnew.BOARD 实例。

    Returns:
        (x1, y1, x2, y2) mm；无 Edge.Cuts 图形时返回 None。
    """
    pcb = _pcb()
    edge_id = board.GetLayerID("Edge.Cuts")
    if edge_id < 0:
        return None
    boxes = []
    for item in board.GetDrawings():
        try:
            if item.GetLayer() != edge_id:
                continue
        except Exception:
            continue
        b = _shape_bbox_mm(item)
        if b:
            boxes.append(b)
    return _merge_boxes(boxes)


def get_footprint_body_bbox(
    fp,
) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    """取元件「本体」矩形 = `GetBoundingHull().BBox()`，与训练口径完全一致。

    项目口径（2026-09-11 定稿）：蓝/黄框表示 **component 本身的矩形尺寸**，
    即 footprint 凸包包围盒——它与训练/碰撞检测用的 `Component.bbox` 是
    **同一个值**，保证"图上看到的框"就是"训练判定所用的框"。

    说明：
    - 不含 Courtyard 的装配余量（那是占位区，比元件矩形大一圈）
    - 从属焊盘另以绿框叠加在本体框内；独立焊盘则以黄框直接表示
      （其 hull 就是焊盘本身）

    Args:
        fp: pcbnew.FOOTPRINT 实例。

    Returns:
        ((x1, y1, x2, y2), 层来源说明 "hull")；提取失败时 bbox 为 None。
    """
    pcb = _pcb()
    try:
        hull = fp.GetBoundingHull().BBox()
        box = (
            pcb.ToMM(hull.GetLeft()),
            pcb.ToMM(hull.GetTop()),
            pcb.ToMM(hull.GetRight()),
            pcb.ToMM(hull.GetBottom()),
        )
        return box, "hull"
    except Exception:
        return None, "none"


def _pad_on_layer(pad, layer_id: int) -> bool:
    """判断焊盘是否位于指定层（跨版本兼容，解析失败时宽松返回 True）。"""
    if layer_id < 0:
        return False
    try:
        return bool(pad.IsOnLayer(layer_id))
    except Exception:
        pass
    try:
        return bool(pad.GetLayerSet().Contains(layer_id))
    except Exception:
        # 返回 False 而非 True：否则 on_top/on_bot 会同时为真，
        # 普通 SMD 焊盘将被误判为"贯穿 PTH"（空心样式）并污染层统计
        return False


def get_pads(fp, board) -> List[Tuple[float, float, float, float, bool, bool]]:
    """提取 footprint 全部焊盘的包围盒及其层归属。

    Returns:
        [(x1, y1, x2, y2, on_top, on_bottom), ...]；坐标为 mm。
        PTH 通孔焊盘 on_top 与 on_bottom 同时为 True（两层图都画）。
    """
    pcb = _pcb()
    f_cu = board.GetLayerID("F.Cu")
    b_cu = board.GetLayerID("B.Cu")
    out = []
    for pad in _iter_items(fp, ("Pads",)):
        try:
            box = pad.GetBoundingBox()
            b = (
                pcb.ToMM(box.GetLeft()),
                pcb.ToMM(box.GetTop()),
                pcb.ToMM(box.GetRight()),
                pcb.ToMM(box.GetBottom()),
            )
        except Exception:
            continue
        out.append((b[0], b[1], b[2], b[3],
                    _pad_on_layer(pad, f_cu), _pad_on_layer(pad, b_cu)))
    return out


def is_standalone_pad(fp) -> bool:
    """独立焊盘判定：无 Reference 且无任何图形项。

    EasyEDA/立创 EDA 的"自由焊盘"（不属于任何元件符号）转成 KiCad 时会被包成
    一个匿名 footprint：Reference 为空、库名形如 `Pad_e131`、仅 1 个 pad、
    `GraphicalItems()` 为空。

    这类对象的位置由布线（走线/敷铜连接点）钉死，按项目口径视为**固定元件**，
    不可与其他元器件重叠。判据用双条件以降低误判：实测 20 块板 858 个
    footprint 中，恰好 25 个满足该条件，且均为独立焊盘。

    Args:
        fp: pcbnew.FOOTPRINT 实例。

    Returns:
        True 表示这是独立焊盘（应按固定元件本体渲染）。
    """
    try:
        if fp.GetReference():
            return False
    except Exception:
        return False
    return not _iter_items(fp, ("GraphicalItems", "GetGraphicalItems"))


def is_fixed(fp, standalone: Optional[bool] = None) -> bool:
    """固定元件判定（kicad_utils 启发式 + 独立焊盘规则）。

    判定顺序：独立焊盘（无 Reference 且无图形项）→ 文件锁 IsLocked()
    → 封装名机械关键词 → 0 网络连接（机械件）。
    连接器与大 IC 默认【不固定】（它们是 RL 应优化放置的对象）。

    Args:
        fp: pcbnew.FOOTPRINT 实例。
        standalone: 预计算的独立焊盘判定结果；None 表示函数内部自行计算。
            调用方若已算过则传入，可避免重复枚举 GraphicalItems。

    Returns:
        True 表示该元件不可移动（含独立焊盘）。
    """
    # 独立焊盘优先级最高：它既不是可布局对象，也不应被其他元件压占
    if standalone is None:
        standalone = is_standalone_pad(fp)
    if standalone:
        return True

    try:
        if fp.IsLocked():
            return True
    except Exception:
        pass

    try:
        name = str(fp.GetFPID().GetLibItemName()).upper()
    except Exception:
        name = ""
    if any(k in name for k in _FIXED_FOOTPRINT_KEYWORDS):
        return True

    # 0 网络连接 = 机械件（螺丝/铜柱/定位点/无 pad 图形件）
    try:
        pad_nets = {p.GetNetname() for p in _iter_items(fp, ("Pads",)) if p.GetNetname()}
    except Exception:
        pad_nets = set()
    return not pad_nets


def get_ref(fp) -> str:
    """取元件参考编号；为空时用封装库名兜底。"""
    try:
        ref = fp.GetReference()
    except Exception:
        ref = ""
    if not ref:
        try:
            ref = str(fp.GetFPID().GetLibItemName())
        except Exception:
            ref = "?"
    return str(ref)


def get_keepouts(
    board,
) -> List[Tuple[Tuple[float, float, float, float], bool, bool]]:
    """提取禁布区（rule area）包围盒及其**层归属**。

    pcbnew 里没有名为 "keepout" 的 API，其本体是 **Rule Area**
    （`ZONE` + `GetIsRuleArea()==True`，即 KiCad 界面 "Add Rule Area" 的产物）。
    一块板的 Zone 绝大多数是普通敷铜，故按该标志筛选。
    几何取 `zone.GetBoundingBox()`（外接矩形），实测本批数据误差 ≈ 0
    （rule area 全为近矩形轮廓）。

    层归属判定（禁布区**是分层的**）：
        - 含 `F.Cu` → 仅顶层生效
        - 含 `B.Cu` → 仅底层生效
        - 不含任何铜层（如 `Edge.Cuts` 的板边约束区）→ 视为跨层，两图都画

    Returns:
        [((x1, y1, x2, y2), on_top, on_bottom), ...]，坐标为 mm；无则空列表。
    """
    pcb = _pcb()
    out = []
    try:
        zones = list(board.Zones())
    except Exception:
        return out
    for zone in zones:
        try:
            if not (hasattr(zone, "GetIsRuleArea") and zone.GetIsRuleArea()):
                continue
            box = zone.GetBoundingBox()
        except Exception:
            continue
        bbox = (
            pcb.ToMM(box.GetLeft()),
            pcb.ToMM(box.GetTop()),
            pcb.ToMM(box.GetRight()),
            pcb.ToMM(box.GetBottom()),
        )

        # 解析该 zone 的层集合（跨版本兼容：优先 LayerSet，退回单层）
        names = set()
        try:
            names = {pcb.LayerName(lid) for lid in zone.GetLayerSet().Seq()}
        except Exception:
            try:
                names = {pcb.LayerName(zone.GetLayer())}
            except Exception:
                pass
        is_copper = any(n.endswith(".Cu") for n in names)
        on_top = ("F.Cu" in names) or (not is_copper)
        on_bottom = ("B.Cu" in names) or (not is_copper)

        out.append((bbox, on_top, on_bottom))
    return out


def _iter_tracks(board) -> list:
    """按候选 API 名取 board 的走线/过孔集合（跨版本兼容）。"""
    for name in ("GetTracks", "Tracks"):
        fn = getattr(board, name, None)
        if fn is None:
            continue
        try:
            return list(fn())
        except Exception:
            continue
    return []


def _is_via(track) -> bool:
    """判断 track 是否为过孔（PCB_VIA）。"""
    pcb = _pcb()
    via_cls = getattr(pcb, "PCB_VIA", None)
    if via_cls is not None:
        try:
            return isinstance(track, via_cls)
        except Exception:
            pass
    try:
        return track.GetClass() == "PCB_VIA"
    except Exception:
        return False


def get_vias(board) -> List[Tuple[float, float, float]]:
    """提取过孔：中心坐标与直径。

    Returns:
        [(cx, cy, diameter), ...] mm。需求要求以矩形框表示，故仅返回
        中心+直径，由绘图侧换算成外接正方形。
    """
    pcb = _pcb()
    out = []
    for track in _iter_tracks(board):
        if not _is_via(track):
            continue
        try:
            pos = track.GetPosition()
            # KiCad 10 的 PCB_VIA::GetWidth() 无参调用会触发 assert 并中止进程
            # （新签名要求传入层参数）。改用包围盒长边作为过孔直径，跨版本更稳。
            box = track.GetBoundingBox()
            diameter = max(pcb.ToMM(box.GetWidth()), pcb.ToMM(box.GetHeight()))
        except Exception:
            continue
        if diameter <= 0:
            continue
        out.append((pcb.ToMM(pos.x), pcb.ToMM(pos.y), diameter))
    return out


# ==================== 绘图 ====================


def _pick_label_step(span: float) -> float:
    """按板尺寸选主刻度步长（mm），保证标签不重叠（约 10 个主刻度）。

    Args:
        span: 板框较长边的长度（mm）。

    Returns:
        1-2-5 进制量级的步长，最小 0.1mm。
    """
    target = max(span, 1.0) / 10.0
    for step in (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1):
        if step <= target:
            return step
    return _TICK_MINOR_STEP


def _font_size_for(span: float) -> float:
    """按板尺寸自适应 ID 字号：板越大字越小，避免文字糊成一片。"""
    if span < 30:
        return 9.0
    if span < 60:
        return 7.0
    if span < 120:
        return 5.5
    return 4.5


def _pick_minor_step(span: float) -> float:
    """次刻度步长：默认 0.1mm，仅当板跨度过大时按 2 倍逐级放宽。

    0.1mm 是本项目的既定颗粒度；但 matplotlib 有 tick 数上限
    `Locator.MAXTICKS = 1000`——165mm 的板会产生 1655 个 tick，
    触发告警且渲染成无意义的密集纹理，故仅在超限时放宽。

    Args:
        span: 板框较长边长度（mm）。

    Returns:
        次刻度步长（mm），≥ 0.1。
    """
    step = _TICK_MINOR_STEP
    while span / step > 900:
        step *= 2
    return step


def _draw_layer_elements(
    ax,
    board_rect: Tuple[float, float, float, float],
    comps: List[Tuple[Tuple[float, float, float, float], bool, str]],
    pads: List[Tuple[float, float, float, float, bool]],
    keepouts: List[Tuple[float, float, float, float]],
    vias: List[Tuple[float, float, float]],
    font_size: float,
    label_step: float,
):
    """在给定 axes 上绘制单层元素（板框 / 禁布区 / 元件本体 / 焊盘 / 过孔）并配置坐标轴。

    Args:
        ax: matplotlib Axes。
        board_rect: 板框 (x1, y1, x2, y2) mm。
        comps: 元件本体 [(bbox, is_fixed, ref), ...]（已按该层过滤）。
        pads: 焊盘 [(x1, y1, x2, y2, is_pth), ...]（已按该层过滤）。
        keepouts: 该层生效的禁布区 bbox 列表（由上层按层分流后传入）。
        vias: 过孔 [(cx, cy, diameter), ...]（跨层元素，两子图均绘制）。
        font_size: 元件编号字号（按板尺寸自适应）。
        label_step: 坐标轴主刻度步长（mm）。
    """
    x1, y1, x2, y2 = board_rect
    board_w = abs(x2 - x1)
    board_h = abs(y2 - y1)
    span = max(board_w, board_h)
    margin = span * 0.05

    # ---- 板框（底色 + 描边，置于最底层） ----
    ax.add_patch(mpatches.Rectangle(
        (min(x1, x2), min(y1, y2)), board_w, board_h,
        linewidth=2.5, edgecolor=COLOR_BOARD, facecolor=COLOR_BOARD_BG,
        zorder=1,
    ))

    # ---- 禁布区（红；先于元件绘制，避免遮挡元件信息） ----
    for kx1, ky1, kx2, ky2 in keepouts:
        kw, kh = kx2 - kx1, ky2 - ky1
        if kw <= 0 or kh <= 0:
            continue
        ax.add_patch(mpatches.Rectangle(
            (kx1, ky1), kw, kh,
            linewidth=1.0, edgecolor=COLOR_FORBID_EDGE,
            facecolor=COLOR_FORBID_FACE, alpha=0.35, zorder=3,
        ))

    # ---- 元件本体（浅蓝=可动 / 浅黄=固定） ----
    for bbox, fixed, ref in comps:
        bx1, by1, bx2, by2 = bbox
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            continue
        face = COLOR_COMP_FIXED_FACE if fixed else COLOR_COMP_MOVABLE_FACE
        edge = COLOR_COMP_FIXED_EDGE if fixed else COLOR_COMP_MOVABLE_EDGE
        # alpha 只作用于**填充**（不传给 patch.alpha，否则边框会一起变淡），
        # 这样可动元件的浅蓝更淡，而轮廓线依然清晰
        face_alpha = _FIXED_FACE_ALPHA if fixed else _MOVABLE_FACE_ALPHA
        ax.add_patch(mpatches.Rectangle(
            (bx1, by1), bw, bh,
            linewidth=0.9, edgecolor=edge,
            facecolor=mcolors.to_rgba(face, face_alpha), zorder=5,
        ))
        # 元件编号：黑色细字，居中
        ax.text(
            (bx1 + bx2) / 2.0, (by1 + by2) / 2.0, ref,
            ha="center", va="center", fontsize=font_size,
            color=COLOR_ID, fontweight="normal", zorder=12,
            clip_on=True,
        )

    # ---- 焊盘（浅绿，覆盖在本体之上） ----
    # 本层专属焊盘 = 实心；贯穿焊盘(PTH) = 空心描边。
    # PTH 两层铜都有、会在两张图上同时出现，用空心样式可避免被误读为"本层焊盘"。
    for px1, py1, px2, py2, is_pth in pads:
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            continue
        if is_pth:
            ax.add_patch(mpatches.Rectangle(
                (px1, py1), pw, ph,
                linewidth=0.7, edgecolor=COLOR_PAD_EDGE,
                facecolor="none", zorder=7,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (px1, py1), pw, ph,
                linewidth=0.5, edgecolor=COLOR_PAD_EDGE,
                facecolor=COLOR_PAD_FACE, alpha=0.9, zorder=7,
            ))

    # ---- 过孔（红，按直径的外接正方形表示） ----
    for cx, cy, diameter in vias:
        r = diameter / 2.0
        ax.add_patch(mpatches.Rectangle(
            (cx - r, cy - r), diameter, diameter,
            linewidth=0.6, edgecolor=COLOR_FORBID_EDGE,
            facecolor="none", zorder=9,
        ))

    # ---- 坐标轴（KiCad 原生方向：Y 轴向下） ----
    ax.set_xlim(min(x1, x2) - margin, max(x1, x2) + margin)
    ax.set_ylim(min(y1, y2) - margin, max(y1, y2) + margin)
    ax.invert_yaxis()   # 与 KiCad 视图一致：y 小值在上方
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(label_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(label_step))
    # 次刻度默认 0.1mm（需求指定的颗粒度），只画线不标注，避免标签重叠；
    # 超大板会自动放宽步长（见 _pick_minor_step），避免 tick 数超上限
    minor_step = _pick_minor_step(span)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
    ax.grid(True, which="major", color=COLOR_GRID_MAJOR, linewidth=0.4, alpha=0.9)
    ax.grid(True, which="minor", color=COLOR_GRID_MINOR, linewidth=0.15, alpha=0.6)

    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm, KiCad 方向)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    # 隐藏 matplotlib 坐标轴自带的边框（spine），避免与 Edge.Cuts 板框矩形
    # 叠加形成"内外双圈"——图上只保留真正的板框
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_board(
    board_name: str,
    board_rect: Tuple[float, float, float, float],
    top_comps: List[Tuple[Tuple[float, float, float, float], bool, str]],
    top_pads: List[Tuple[float, float, float, float, bool]],
    top_keepouts: List[Tuple[float, float, float, float]],
    bot_comps: List[Tuple[Tuple[float, float, float, float], bool, str]],
    bot_pads: List[Tuple[float, float, float, float, bool]],
    bot_keepouts: List[Tuple[float, float, float, float]],
    vias: List[Tuple[float, float, float]],
    output_path: str,
    dpi: int = 150,
):
    """绘制合并图并保存：一张 PNG 内左右并排（左=顶层，右=底层），图例置于图外底部。

    Args:
        board_name: 板名（用于总标题）。
        board_rect: 板框 (x1, y1, x2, y2) mm。
        top_comps: 顶层元件本体 [(bbox, is_fixed, ref), ...]。
        top_pads: 顶层焊盘 [(x1, y1, x2, y2, is_pth), ...]。
        top_keepouts: 顶层生效的禁布区（F.Cu 层 + 跨层的非铜层区）。
        bot_comps: 底层元件本体。
        bot_pads: 底层焊盘 [(x1, y1, x2, y2, is_pth), ...]。
        bot_keepouts: 底层生效的禁布区（B.Cu 层 + 跨层的非铜层区）。
        vias: 过孔（跨层，两子图均绘制）。
        output_path: 输出 PNG 路径。
        dpi: 分辨率。
    """
    _setup_chinese_font()

    x1, y1, x2, y2 = board_rect
    board_w = abs(x2 - x1)
    board_h = abs(y2 - y1)
    span = max(board_w, board_h)
    font_size = _font_size_for(span)
    label_step = _pick_label_step(span)
    minor_step = _pick_minor_step(span)

    # 左右两个面板并排；单面板尺寸按板框比例自适应（最小 6 英寸）
    panel_w = max(6.0, board_w / 10.0)
    panel_h = max(6.0, board_h / 10.0)
    fig, (ax_top, ax_bot) = plt.subplots(
        1, 2, figsize=(panel_w * 2.0 + 1.0, panel_h)
    )
    fig.patch.set_facecolor("white")

    _draw_layer_elements(ax_top, board_rect, top_comps, top_pads,
                         top_keepouts, vias, font_size, label_step)
    _draw_layer_elements(ax_bot, board_rect, bot_comps, bot_pads,
                         bot_keepouts, vias, font_size, label_step)
    ax_top.set_title("顶层 (F.Cu)", fontsize=11)
    ax_bot.set_title("底层 (B.Cu)", fontsize=11)

    fig.suptitle(
        f"{board_name}   [次刻度 {minor_step:g}mm / 主刻度 {label_step:g}mm]",
        fontsize=13, fontweight="bold",
    )

    # ---- 图例：置于图外底部横排，避免遮挡板内元件 ----
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_COMP_MOVABLE_FACE, edgecolor=COLOR_COMP_MOVABLE_EDGE,
                       label="元件本体（可移动）"),
        mpatches.Patch(facecolor=COLOR_COMP_FIXED_FACE, edgecolor=COLOR_COMP_FIXED_EDGE,
                       label="元件本体（固定，含独立焊盘）"),
        mpatches.Patch(facecolor=COLOR_PAD_FACE, edgecolor=COLOR_PAD_EDGE,
                       label="焊盘（本层）"),
        mpatches.Patch(facecolor="none", edgecolor=COLOR_PAD_EDGE,
                       label="贯穿焊盘 PTH"),
        mpatches.Patch(facecolor=COLOR_FORBID_FACE, edgecolor=COLOR_FORBID_EDGE,
                       label="禁布区 / 过孔"),
        mpatches.Patch(facecolor=COLOR_BOARD_BG, edgecolor=COLOR_BOARD, label="板框"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    # 底部预留图例空间、顶部预留总标题空间
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.94])

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==================== 单板处理 ====================


def process_board(pcb_path: str, out_dir: str, dpi: int = 150) -> Dict[str, object]:
    """处理单块板：提取元素 → 绘制合并图（左顶层/右底层）PNG。

    Args:
        pcb_path: .kicad_pcb 文件路径。
        out_dir: 输出目录。
        dpi: 输出分辨率。

    Returns:
        统计信息字典（板名、各元素数量、输出路径、本体层命中情况）。

    Raises:
        RuntimeError: 板文件加载失败或板框无法确定。
    """
    pcb = _pcb()
    board = pcb.LoadBoard(pcb_path)
    if board is None:
        raise RuntimeError(f"LoadBoard 失败（文件损坏或格式不支持）: {pcb_path}")

    board_rect = get_board_rect(board)
    if board_rect is None:
        # 无 Edge.Cuts：用整块板的包围盒兜底，保证仍能出图
        try:
            bb = board.GetBoardEdgesBoundingBox()
            board_rect = (
                pcb.ToMM(bb.GetLeft()), pcb.ToMM(bb.GetTop()),
                pcb.ToMM(bb.GetRight()), pcb.ToMM(bb.GetBottom()),
            )
            logger.warning("无 Edge.Cuts 板框，使用板级包围盒兜底")
        except Exception as e:
            raise RuntimeError(f"无法确定板框: {e}") from e

    keepouts = get_keepouts(board)
    # 禁布区按层分流：F.Cu 的只进顶层图、B.Cu 的只进底层图、
    # 非铜层（Edge.Cuts 板边约束）两层都进
    keepouts_top = [b for b, on_t, _ in keepouts if on_t]
    keepouts_bot = [b for b, _, on_b in keepouts if on_b]
    vias = get_vias(board)

    top_comps, bot_comps = [], []
    top_pads, bot_pads = [], []
    body_layer_hits: Dict[str, int] = {}

    standalone_cnt = 0
    for fp in _iter_items(board, ("Footprints",)):
        standalone = is_standalone_pad(fp)
        fixed = is_fixed(fp, standalone)   # 复用上面的判定，避免重复枚举图形项
        ref = get_ref(fp)
        body, layer_src = get_footprint_body_bbox(fp)
        body_layer_hits[layer_src] = body_layer_hits.get(layer_src, 0) + 1
        if standalone:
            standalone_cnt += 1

        try:
            is_bottom = fp.GetLayer() == board.GetLayerID("B.Cu")
        except Exception:
            is_bottom = False

        if body:
            (bot_comps if is_bottom else top_comps).append((body, fixed, ref))

        # 独立焊盘的"本体即焊盘"，已在上方以固定元件本体渲染；
        # 不再重复画从属焊盘绿框（否则绿框会盖住黄框、丢失"固定"语义）
        if standalone:
            continue

        for px1, py1, px2, py2, on_top, on_bot in get_pads(fp, board):
            # 两层铜都有 = 贯穿焊盘(PTH)：它在两张图上都会出现，
            # 故携带该标记，供绘图侧用不同样式与"本层专属焊盘"区分开
            is_pth = on_top and on_bot
            if on_top:
                top_pads.append((px1, py1, px2, py2, is_pth))
            if on_bot:
                bot_pads.append((px1, py1, px2, py2, is_pth))

    board_name = os.path.splitext(os.path.basename(pcb_path))[0]
    png_path = os.path.join(out_dir, f"{board_name}.png")

    draw_board(board_name, board_rect, top_comps, top_pads, keepouts_top,
               bot_comps, bot_pads, keepouts_bot, vias, png_path, dpi=dpi)

    stats = {
        "board": board_name,
        "comp_top": len(top_comps),
        "comp_bottom": len(bot_comps),
        "fixed": sum(1 for _, is_f, _ in (top_comps + bot_comps) if is_f),
        "pad_top": len(top_pads),
        "pad_bottom": len(bot_pads),
        "keepout": len(keepouts),
        "keepout_top": len(keepouts_top),
        "keepout_bottom": len(keepouts_bot),
        "via": len(vias),
        "standalone_pad": standalone_cnt,
        "body_layers": body_layer_hits,
        "png": png_path,
    }
    return stats


# ==================== 入口 ====================


def _clean_legacy_outputs(out_dir: str):
    """清理旧版产物（每板两张 `*_top.png` / `*_bottom.png`）。

    合并为单图后旧文件已无意义。**前提**：输出目录为本脚本专用，
    该命名模式不会被其他工具占用，否则同名文件会被误删。
    仅按该模式删除，不触碰目录内其他文件。

    Args:
        out_dir: 输出目录。
    """
    if not os.path.isdir(out_dir):
        return
    removed = 0
    for fname in os.listdir(out_dir):
        if not (fname.endswith("_top.png") or fname.endswith("_bottom.png")):
            continue
        try:
            os.remove(os.path.join(out_dir, fname))
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info(f"已清理旧版分层产物 {removed} 个（*_top.png / *_bottom.png）")


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="将真实 KiCad 专家布局渲染为顶层/底层可视化 PNG。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "kicad-native-batch"),
        help="PCB 数据根目录（递归扫描 .kicad_pcb）",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "kicad-native-batch-vision"),
        help="图片输出目录",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="只处理前 N 块板（0 = 全部）；用于先行验证效果",
    )
    parser.add_argument(
        "--board", default=None,
        help="只处理文件名包含该关键字的板（如 IMU）",
    )
    parser.add_argument("--dpi", type=int, default=150, help="输出图片分辨率")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """脚本入口：扫描板子 → 逐板提取 → 绘制「左顶层/右底层」合并图 PNG。

    Args:
        argv: 命令行参数（None 时取 sys.argv[1:]）。

    Returns:
        进程退出码：0 全部成功（或部分成功但无板可跑时为 1）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _build_parser().parse_args(argv)

    try:
        _pcb()   # 提前校验 KiCad 环境，避免逐板重复报错
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    from utils.pcb_scan import find_pcb_files

    files = find_pcb_files(args.data_dir)
    if args.board:
        files = [f for f in files if args.board in os.path.basename(f)]
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        logger.error(f"未找到可处理的 .kicad_pcb 文件（data-dir={args.data_dir}）")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    _clean_legacy_outputs(args.out_dir)
    logger.info(f"共 {len(files)} 块板待处理，输出目录: {args.out_dir}")

    ok, failed = 0, []
    for idx, pcb_path in enumerate(files, 1):
        name = os.path.basename(pcb_path)
        logger.info(f"[{idx}/{len(files)}] 处理中: {name}")
        try:
            stats = process_board(pcb_path, args.out_dir, dpi=args.dpi)
        except Exception as e:   # 单板失败不中断整体
            logger.error(f"[{idx}/{len(files)}] 失败: {name}: {e}")
            failed.append((name, str(e)))
            continue
        ok += 1
        logger.info(
            f"[{idx}/{len(files)}] 完成: {stats['board']} | "
            f"元件 顶{stats['comp_top']}/底{stats['comp_bottom']}"
            f"(固定{stats['fixed']}) | "
            f"焊盘 顶{stats['pad_top']}/底{stats['pad_bottom']} | "
            f"过孔 {stats['via']} | "
            f"禁布区 {stats['keepout']}"
            f"(顶{stats['keepout_top']}/底{stats['keepout_bottom']}) | "
            f"独立焊盘(fixed) {stats['standalone_pad']} | "
            f"本体层命中 {stats['body_layers']}"
        )
        logger.info(f"            → {stats['png']}")

    logger.info("=" * 60)
    logger.info(f"处理完成：成功 {ok} / 失败 {len(failed)}")
    if failed:
        for name, err in failed:
            logger.warning(f"  失败板: {name} → {err}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
