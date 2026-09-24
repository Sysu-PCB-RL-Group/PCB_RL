# -*- coding: utf-8 -*-
"""跨板指标对比图（§10.2 那 4 个指标的图面表达）—— 3 张"与专家组比较"的百分比柱状图。

Owner 2026-09-23 拍：每张 **16 根柱**（横轴一板一根）｜**纵轴 0 居中**（向上 = 优于专家组、向下 = 差）
｜**不标具体数值**（数值看 `inference.log` / `summary.json`）。

三张图的方向统一为"**正 = 好**"：
```text
metric_hpwl_drop.png     HPWL 降低 %    （向上 = 线长比专家短）
metric_sigma_gain.png    σ 提升 %       （向上 = 离散度比专家大）
metric_reward_gain.png   reward 提升 %  （向上 = 一局回报比专家高）
```
只依赖 matplotlib + `utils/eval_metrics`（**不需要 KiCad**）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

from utils.viz_layout import setup_chinese_font   # noqa: E402

__all__ = ["CHART_SPECS", "draw_metric_bar", "draw_all_metrics"]

setup_chinese_font()

C_POS = "#43A047"   # 优于专家组
C_NEG = "#E53935"   # 差于专家组
C_NA = "#BDBDBD"    # 无数据（分母为 0 ⇒ 指标记 None）
C_ZERO = "#333333"

#: 3 张图的规格：`(指标键, 文件名, 标题, y 轴名)`（顺序 = 日志里 ②③④ 的顺序）
CHART_SPECS = (
    ("hpwl_drop_pct", "metric_hpwl_drop.png", "HPWL 降低百分比（对比专家组）",
     "HPWL 降低 %（向上 = 优于专家）"),
    ("sigma_gain_pct", "metric_sigma_gain.png", "σ 提升百分比（对比专家组）",
     "σ 提升 %（向上 = 优于专家）"),
    ("reward_gain_pct", "metric_reward_gain.png", "reward 提升百分比（对比专家组）",
     "reward 提升 %（向上 = 优于专家）"),
)


def _labels(rows: Sequence[Mapping[str, Any]]) -> list:
    """横轴标签：`序号.板名前 16 字符`（16 板太长，缩写 + 序号便于和 log 对照）。"""
    out = []
    for i, r in enumerate(rows):
        name = str(r.get("board") or r.get("name") or "")
        out.append(f"{i + 1}.{name[:16]}")
    return out


def draw_metric_bar(labels: Sequence[str], values: Sequence[Optional[float]], *,
                    title: str, ylabel: str, out_path: str, dpi: int = 110) -> Dict[str, Any]:
    """一张百分比柱状图：**0 居中**、正绿负红、无数据灰；**不写任何数值标注**。"""
    vals = [0.0 if v is None else float(v) for v in values]
    colors = [C_NA if v is None else (C_POS if float(v) >= 0.0 else C_NEG) for v in values]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * max(1, len(vals)) + 3.0), 4.6), dpi=int(dpi))
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="#37474F", linewidth=0.4, zorder=3)
    ax.axhline(0.0, color=C_ZERO, linewidth=0.9, zorder=4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(f"{title}（正 = 比专家组好｜负 = 差｜灰 = 无数据）", fontsize=11)
    ax.grid(axis="y", color="#E0E0D8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "n": len(vals),
            "n_pos": int(sum(1 for v in vals if v > 0.0)),
            "n_neg": int(sum(1 for v in vals if v < 0.0)),
            "min": (min(vals) if vals else 0.0), "max": (max(vals) if vals else 0.0)}


def draw_all_metrics(rows: Sequence[Mapping[str, Any]], out_dir: str, *,
                     dpi: int = 110) -> Dict[str, Dict[str, Any]]:
    """出全部 3 张图（`CHART_SPECS` 的顺序）→ `{指标键: 图信息}`。"""
    labels = _labels(rows)
    out: Dict[str, Dict[str, Any]] = {}
    for key, fname, title, yname in CHART_SPECS:
        out[key] = draw_metric_bar(labels, [r.get(key) for r in rows], title=title, ylabel=yname,
                                   out_path=os.path.join(out_dir, fname), dpi=dpi)
    return out
