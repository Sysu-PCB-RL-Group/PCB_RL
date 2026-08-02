"""PCB 布局可视化（matplotlib）。

绘制板框矩形 + 1mm 颗粒度网格 + 各元件矩形 + ID/net 标注。
支持中文标题，元件过小时 ID 自动外移避免遮挡。
"""

import logging
import os
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# ==================== 字体设置（支持中文） ====================

_FONT_SET = False


def _setup_chinese_font():
    """全局设置 matplotlib 中文字体（仅在首次调用时执行一次）。"""
    global _FONT_SET
    if _FONT_SET:
        return
    try:
        # 常见含中文的字体优先级：微软雅黑 > SimHei > Arial Unicode MS
        for name in ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]:
            try:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                _FONT_SET = True
                logger.debug(f"Matplotlib font set to: {name}")
                return
            except Exception:
                continue
    except Exception:
        pass


# ==================== 配色方案 ====================

COMP_COLORS = {
    "movable": {
        "face": "#C8E6C9",      # 淡绿色
        "edge": "#43A047",      # 深绿色边框
    },
    "fixed": {
        "face": "#BDBDBD",      # 灰色
        "edge": "#757575",
    },
}

BOARD_COLOR = "#333333"
BOARD_BG = "#F5F5F0"
GRID_COLOR = "#E0E0D8"

COLOR_ID = "#D32F2F"           # 元件 ID：红色
COLOR_NET = "#333333"          # Net 信息：黑色

# ==================== 小元件文字外移判定 ====================


def _is_small_bbox(bbox: Tuple[float, float, float, float]) -> bool:
    """面积 < 6mm² 或宽/高 < 2mm 视为小元件，ID 需外移。"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w * h < 6.0 or w < 2.0 or h < 2.0


def _auto_tick_step(board_w: float, board_h: float) -> float:
    """自动计算坐标轴主刻度步长。

    使得板框范围内有大约 5-10 根刻度线，结果取整到 1/5/10。
    """
    target = max(board_w, board_h) / 6.0
    for step in [10.0, 5.0, 2.0, 1.0]:
        if step <= target * 1.5:
            return step
    return 1.0


# ==================== 主绘图函数 ====================


def draw_layout(
    components: list,
    board_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    output_path: str,
    title: Optional[str] = None,
    show_grid: bool = True,
    dpi: int = 150,
):
    """绘制 PCB 布局图并保存。

    Args:
        components: 元件对象列表，每个需有 id/bbox/is_fixed/is_placed/nets 属性
        board_bounds: ((min_x, min_y), (max_x, max_y)) mm
        output_path: 输出 PNG 路径
        title: 图片标题（可选，支持中文）
        show_grid: 是否显示网格（默认 True）
        dpi: 分辨率（默认 150）
    """
    _setup_chinese_font()

    (min_x, min_y), (max_x, max_y) = board_bounds
    board_w = max_x - min_x
    board_h = max_y - min_y

    # 画布尺寸：根据板框比例自适应，最小 4 英寸
    fig_w = max(4.0, board_w / 12)
    fig_h = max(4.0, board_h / 12)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # ---- 板框 ----
    board_rect = mpatches.Rectangle(
        (min_x, min_y), board_w, board_h,
        linewidth=3, edgecolor=BOARD_COLOR, facecolor=BOARD_BG,
        zorder=1,
    )
    ax.add_patch(board_rect)

    # ---- 1mm 网格 ----
    if show_grid:
        for x in np.arange(min_x, max_x + 1.0, 1.0):
            ax.axvline(x, color=GRID_COLOR, linewidth=0.2, zorder=0)
        for y in np.arange(min_y, max_y + 1.0, 1.0):
            ax.axhline(y, color=GRID_COLOR, linewidth=0.2, zorder=0)

    # ---- 各元件 ----
    for comp in components:
        x1, y1, x2, y2 = comp.bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        style = COMP_COLORS["fixed"] if comp.is_fixed else COMP_COLORS["movable"]

        # 淡绿色矩形
        rect = mpatches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5, edgecolor=style["edge"],
            facecolor=style["face"], alpha=0.85,
            zorder=5,
        )
        ax.add_patch(rect)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        small = _is_small_bbox((x1, y1, x2, y2))

        # ---- ID（红色） ----
        if small:
            # 小元件：外移到右侧
            id_x = x2 + board_w * 0.02
            id_y = cy
            ha, va = "left", "center"
        else:
            # 正常大小：居中
            id_x, id_y = cx, cy
            ha, va = "center", "center"

        ax.text(id_x, id_y, comp.id,
                ha=ha, va=va, fontsize=8, fontweight="bold",
                color=COLOR_ID, zorder=10)

        # ---- Net 信息（黑色） ----
        if hasattr(comp, "nets") and comp.nets:
            net_text = ", ".join(comp.nets[:3])
            if len(comp.nets) > 3:
                net_text += "..."

            if small:
                nx, ny = id_x, id_y - board_h * 0.015
                ha_net, va_net = "left", "top"
            else:
                nx, ny = cx, y2 + board_h * 0.005
                ha_net, va_net = "center", "bottom"

            ax.text(nx, ny, net_text,
                    ha=ha_net, va=va_net, fontsize=5.5,
                    color=COLOR_NET, style="italic", zorder=10)

    # ---- 图例 ----
    legend_elements = [
        mpatches.Patch(facecolor=COMP_COLORS["movable"]["face"], alpha=0.85,
                       edgecolor=COMP_COLORS["movable"]["edge"], label="Movable"),
        mpatches.Patch(facecolor=COMP_COLORS["fixed"]["face"], alpha=0.85,
                       edgecolor=COMP_COLORS["fixed"]["edge"], label="Fixed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC")

    # ---- 坐标轴 ----
    tick_step = _auto_tick_step(board_w, board_h)
    margin = max(board_w, board_h) * 0.10
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug(f"Layout saved to: {output_path}")


# ==================== Loss-Episode 曲线 ====================


def plot_loss_curve(
    loss_history: List[Tuple[int, float]],
    output_path: str,
    pcb_name: str = "",
    window: int = 100,
    dpi: int = 150,
):
    """绘制 Loss-Episode 训练曲线。

    原始 loss 按 {window} episode 分组取中位数平滑，避免单点异常抖动。

    Args:
        loss_history: [(episode_num, avg_loss), ...] 列表
        output_path: 输出 PNG 路径
        pcb_name: PCB 名称（用于标题）
        window: 平滑窗口大小（episode 数）
        dpi: 分辨率
    """
    _setup_chinese_font()

    if len(loss_history) < window:
        return  # 数据不够，不画

    # 提取原始数据
    raw_eps = [ep for ep, _ in loss_history]
    raw_losses = [loss for _, loss in loss_history]

    # ---- 按 window 分块取中位数 + min/max 带 ----
    smoothed_eps = []
    smoothed_losses = []
    smoothed_mins = []
    smoothed_maxs = []
    for i in range(0, len(raw_losses), window):
        chunk = raw_losses[i:i + window]
        if not chunk:
            continue
        arr = np.array(chunk)
        smoothed_eps.append(raw_eps[i + len(chunk) // 2])
        smoothed_losses.append(float(np.median(arr)))
        smoothed_mins.append(float(np.min(arr)))
        smoothed_maxs.append(float(np.max(arr)))

    # ---- 绘图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 子图 1：原始 loss（log 轴 + 小散点）
    ax1.scatter(raw_eps, raw_losses, s=1, alpha=0.12, color="#1565C0", label="原始 loss")
    ax1.plot(smoothed_eps, smoothed_losses, linewidth=2, color="#E53935",
             label=f"中位数 ({window}ep 窗口)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (log)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 子图 2：平滑 loss + min/max 带
    ax2.fill_between(smoothed_eps, smoothed_mins, smoothed_maxs,
                     alpha=0.15, color="#1565C0", label=f"min~max ({window}ep 窗口)")
    ax2.plot(smoothed_eps, smoothed_losses, linewidth=2.5, color="#1565C0",
             label=f"中位数 ({window}ep)")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Loss (中位数)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 标题
    title = f"Loss-Episode 训练曲线 — {pcb_name}" if pcb_name else "Loss-Episode 训练曲线"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Loss curve saved to: {output_path}")
