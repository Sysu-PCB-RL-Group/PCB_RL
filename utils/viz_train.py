# -*- coding: utf-8 -*-
"""训练效果图：随 episode 的 **loss** / **一局累计 reward** / **每局步数** 曲线（Owner 2026-09-24）。

```text
落点：训练结束时写 `<out>/train_curves.png` —— **一张图三个子图**（上 loss｜中 reward｜下 每局步数）
数据：训练循环内存里的逐局记录（`kind="episode"` 的 `loss` / `ret` / `steps`），**不依赖日志文件**
风格：与 `utils/viz_metrics.py` 一致（简洁、不写数值标注、中文标题）；**不需要 KiCad**（Agg 后端）
```

三条曲线都画"原始（细淡）+ 滑动平均（粗实）"：逐局值抖得厉害，单看原始线读不出趋势。
`loss` 为 `None` 的局（`--update-every 0` 或该局没做更新）**跳过不画**（不拿 0 冒充）。

⚠️ 三个量的读法（勿误读）：
* `loss` 只是 TD 误差的平方（量级地板 ≈ 0.006），**不用来判好坏**（判好坏看评估指标）；
* `一局累计 reward` 含 `β·n` 进度项 ⇒ 跨板不可直接比（每板 β 不同）；
* `每局步数` = **该局放下的件数** = `件数 × 完成度` ⇒ 它其实在反映"**这一局抽到了哪块板**"
  （板件数 10~97，差 10 倍）—— 用来确认"采样没跑偏"，不是"策略变好"。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence

__all__ = ["draw_training_curves", "SMOOTH_WINDOW", "DEFAULT_NAME"]

#: 滑动平均窗口（局）—— 与 train_dqn 的日志节奏（每 10 局一行）同量级，肉眼可读
SMOOTH_WINDOW = 20

#: 产物文件名（放 `--out` 目录下）
DEFAULT_NAME = "train_curves.png"

C_RAW = "#B0BEC5"      # 原始逐局值（淡灰）
C_LOSS = "#E53935"     # loss 滑动平均（红）
C_RET = "#1E88E5"      # reward 滑动平均（蓝）
C_STEP = "#8E24AA"     # 每局步数滑动平均（紫）
C_GRID = "#E0E0E0"


def _moving_average(xs: Sequence[float], window: int) -> list:
    """长度与 `xs` 相同的滑动均值（前 window−1 个点用"到目前为止"的均值 ⇒ 不丢开头）。"""
    w = max(1, int(window))
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += float(x)
        if i >= w:
            acc -= float(xs[i - w])
        out.append(acc / min(i + 1, w))
    return out


def draw_training_curves(history: Sequence[Mapping[str, Any]], out_path: str, *,
                         dpi: int = 110, window: int = SMOOTH_WINDOW) -> Dict[str, Any]:
    """`history`（逐局 `{"episode", "loss", "ret", "steps"}`）⇒ 写一张 PNG，返回统计字典。

    Args:
        history: 逐局记录序列（`train_dqn.train()` 内存里的那一份）。
        out_path: 输出 PNG 路径（父目录不存在会自动建）。
        dpi / window: 分辨率、滑动平均窗口。

    Returns:
        `{"path", "episodes", "dpi", "window", "n_loss",
          "loss_last", "loss_ma_last", "loss_min",
          "ret_last", "ret_ma_last", "ret_max",
          "steps_last", "steps_ma_last", "steps_max", "env_steps"}`。
    """
    import matplotlib
    matplotlib.use("Agg")                      # 无人值守：绝不弹窗
    import matplotlib.pyplot as plt

    from utils.viz_layout import setup_chinese_font
    setup_chinese_font()

    eps = [int(r["episode"]) for r in history]
    rets = [float(r["ret"]) for r in history]
    steps = [int(r.get("steps") or 0) for r in history]
    loss_pairs = [(int(r["episode"]), float(r["loss"]))
                  for r in history if r.get("loss") is not None]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11.0, 7.8), sharex=True)
    for ax in (ax0, ax1, ax2):
        ax.grid(True, color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # ---- 上：loss ----
    if loss_pairs:
        lx = [e for e, _ in loss_pairs]
        ly = [v for _, v in loss_pairs]
        ax0.plot(lx, ly, color=C_RAW, linewidth=0.8, alpha=0.7, label="逐局")
        ax0.plot(lx, _moving_average(ly, window), color=C_LOSS, linewidth=1.8,
                 label=f"{window} 局滑动平均")
        ax0.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax0.set_ylabel("loss（SmoothL1）")
        ax0.set_title(f"训练 loss（有 {len(lx)} 局做了梯度更新；量级地板 ≈ 0.006，不用来判好坏）")
    else:
        ax0.text(0.5, 0.5, "本次训练没有梯度更新（--update-every 0 ?）",
                 ha="center", va="center", transform=ax0.transAxes, fontsize=11, color="#666666")
        ax0.set_ylabel("loss（SmoothL1）")
        ax0.set_title("训练 loss")

    # ---- 中：一局累计 reward ----
    ax1.plot(eps, rets, color=C_RAW, linewidth=0.8, alpha=0.7, label="逐局")
    ax1.plot(eps, _moving_average(rets, window), color=C_RET, linewidth=1.8,
             label=f"{window} 局滑动平均")
    ax1.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.set_ylabel("一局累计 reward")
    ax1.set_title("一局累计 reward（含 β·n 进度项，跨板不可直接比）")

    # ---- 下：每局步数（= 该局放下的件数 = 件数 × 完成度） ----
    ax2.plot(eps, steps, color=C_RAW, linewidth=0.8, alpha=0.7, label="逐局")
    ax2.plot(eps, _moving_average(steps, window), color=C_STEP, linewidth=1.8,
             label=f"{window} 局滑动平均")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("每局步数")
    ax2.set_title(f"每局步数（= 放下的件数；反映抽到哪块板。累计 env 步 {sum(steps)}）")

    fig.suptitle("训练效果曲线", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

    loss_ma = _moving_average([v for _, v in loss_pairs], window) if loss_pairs else []
    ret_ma = _moving_average(rets, window)
    step_ma = _moving_average(steps, window) if steps else []
    return {"path": out_path, "episodes": len(eps), "dpi": int(dpi), "window": int(window),
            "n_loss": len(loss_pairs),
            "loss_last": (loss_pairs[-1][1] if loss_pairs else None),
            "loss_ma_last": (loss_ma[-1] if loss_ma else None),
            "loss_min": (min(v for _, v in loss_pairs) if loss_pairs else None),
            "ret_last": (rets[-1] if rets else None),
            "ret_ma_last": (ret_ma[-1] if rets else None),
            "ret_max": (max(rets) if rets else None),
            "steps_last": (steps[-1] if steps else None),
            "steps_ma_last": (step_ma[-1] if step_ma else None),
            "steps_max": (max(steps) if steps else None),
            "env_steps": (sum(steps) if steps else 0)}
