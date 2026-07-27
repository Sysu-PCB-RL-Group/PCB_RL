"""PCB 布局可视化（matplotlib）。

绘制板框矩形 + 各元件矩形 + ID/net 标注，替代 KiCad 导出。
"""

import logging
import os
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # 无头模式，无需 GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)


# ---- 配色方案 ----

COMP_COLORS = {
    "movable": {
        "face": "#4A90D9",      # 蓝色
        "edge": "#2C5F8A",
        "text": "#FFFFFF",
    },
    "fixed": {
        "face": "#AAAAAA",      # 灰色
        "edge": "#666666",
        "text": "#FFFFFF",
    },
}

BOARD_COLOR = "#333333"
BOARD_BG = "#F5F5F0"
GRID_COLOR = "#E0E0D8"


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
        title: 图片标题（可选）
        show_grid: 是否显示网格（默认 True）
        dpi: 分辨率（默认 150）
    """
    (min_x, min_y), (max_x, max_y) = board_bounds
    board_w = max_x - min_x
    board_h = max_y - min_y

    # 创建画布
    fig, ax = plt.subplots(figsize=(board_w / 12, board_h / 12))
    fig.patch.set_facecolor("white")

    # ---- 绘制板框 ----
    board_rect = mpatches.Rectangle(
        (min_x, min_y), board_w, board_h,
        linewidth=3, edgecolor=BOARD_COLOR, facecolor=BOARD_BG,
        zorder=1,
    )
    ax.add_patch(board_rect)

    # 板框尺寸标注
    ax.text(
        min_x + board_w / 2, min_y - board_h * 0.03,
        f"{board_w:.0f} × {board_h:.0f} mm",
        ha="center", va="top", fontsize=10, fontweight="bold",
        color=BOARD_COLOR, zorder=10,
    )

    # ---- 网格 ----
    if show_grid:
        step = 5.0  # 每 5mm
        for x in np.arange(min_x, max_x + step, step):
            ax.axvline(x, color=GRID_COLOR, linewidth=0.3, zorder=0)
        for y in np.arange(min_y, max_y + step, step):
            ax.axhline(y, color=GRID_COLOR, linewidth=0.3, zorder=0)

    # ---- 绘制各元件 ----
    for comp in components:
        x1, y1, x2, y2 = comp.bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue  # 未放置或无效 bbox

        style = COMP_COLORS["fixed"] if comp.is_fixed else COMP_COLORS["movable"]

        # 元件矩形
        rect = mpatches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5, edgecolor=style["edge"],
            facecolor=style["face"], alpha=0.75,
            zorder=5,
        )
        ax.add_patch(rect)

        # 元件 ID（居中，粗体）
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            cx, cy, comp.id,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=style["text"], zorder=10,
        )

        # Net 信息（上方小字）
        if hasattr(comp, "nets") and comp.nets:
            net_text = ", ".join(comp.nets[:3])  # 最多显示 3 个 net
            if len(comp.nets) > 3:
                net_text += "..."
            ax.text(
                cx, y2 + board_h * 0.005,
                net_text,
                ha="center", va="bottom", fontsize=6,
                color="#555555", style="italic", zorder=10,
            )

    # ---- 图例 ----
    legend_elements = [
        mpatches.Patch(facecolor=COMP_COLORS["movable"]["face"], alpha=0.75,
                       edgecolor=COMP_COLORS["movable"]["edge"],
                       label="Movable"),
        mpatches.Patch(facecolor=COMP_COLORS["fixed"]["face"], alpha=0.75,
                       edgecolor=COMP_COLORS["fixed"]["edge"],
                       label="Fixed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC")

    # ---- 坐标轴设置 ----
    margin = max(board_w, board_h) * 0.08
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # 紧凑布局
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug(f"Layout saved to: {output_path}")
