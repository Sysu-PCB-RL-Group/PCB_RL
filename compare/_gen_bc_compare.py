# -*- coding: utf-8 -*-
"""生成 方案B+C 前后对比图与对比表（result1 = 改进前，result2 = 改进后）"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"c:/Users/26059/PCB_RL"
OUT = os.path.join(ROOT, "compare")
os.makedirs(OUT, exist_ok=True)

BASELINE_DIR = os.path.join(ROOT, "result1")   # 改进前 8/15
NEW_DIR = os.path.join(ROOT, "result2")        # 改进后 B+C 8/16


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_name(pcb):
    """dimsmary-...--Button-xxx.kicad_pcb -> Button"""
    name = os.path.basename(pcb)
    if "--" in name:
        return name.split("--")[-1].replace(".kicad_pcb", "")
    return name.replace(".kicad_pcb", "")


def hist_path(base_dir, pcb, fname):
    board_dir = os.path.join(base_dir, os.path.basename(pcb).replace(".kicad_pcb", ""))
    return os.path.join(board_dir, fname)


def smooth(arr, win=20):
    if len(arr) < win:
        return arr
    return np.convolve(arr, np.ones(win) / win, mode="valid")


# ---------------- 1. 汇总数据 ----------------
r1 = load_json(os.path.join(BASELINE_DIR, "training_summary.json"))
r2 = load_json(os.path.join(NEW_DIR, "training_summary.json"))
d1 = {e["pcb"]: e for e in r1}
d2 = {e["pcb"]: e for e in r2}
pcbs = list(d1.keys())
labels = [short_name(p) for p in pcbs]

rows = []
for p in pcbs:
    e1, e2 = d1[p], d2[p]
    diff = e2["last_avg_placed"] - e1["last_avg_placed"]
    pct = (diff / e1["last_avg_placed"] * 100) if e1["last_avg_placed"] else float("inf")
    rows.append({
        "board": short_name(p), "n_movable": e1["n_movable"],
        "pl1": e1["last_avg_placed"], "pl2": e2["last_avg_placed"],
        "diff": diff, "pct": pct,
        "r1": e1["last_avg_r"], "r2": e2["last_avg_r"],
        "be1": e1["best_episode"], "be2": e2["best_episode"],
    })

# ---------------- 2. placed 对比条形图 ----------------
fig, ax = plt.subplots(figsize=(9.5, 5))
x = np.arange(len(pcbs)); w = 0.35
n_mov = [d1[p]["n_movable"] for p in pcbs]
pl1 = [r["pl1"] for r in rows]; pl2 = [r["pl2"] for r in rows]
ax.bar(x - w / 2, pl1, w, label="改进前 (8/15)", color="#c86a6a")
ax.bar(x + w / 2, pl2, w, label="改进后 B+C (8/16)", color="#4d8f6f")
for i, (a, b) in enumerate(zip(pl1, pl2)):
    ax.text(i - w / 2, a + 0.08, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.08, f"{b:.2f}", ha="center", fontsize=9)
    nm = n_mov[i]
    ax.text(i, max(a, b) + 0.5, f"目标 {nm} 件", ha="center", fontsize=8, color="gray")
    if b - a > 0:
        ax.annotate(f"+{b-a:.2f}", xy=(i + w / 2, b + 0.35), ha="center", fontsize=9, color="#2d6a4f")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("训练后期平均放置数 (last_avg_placed)")
ax.set_ylim(0, max(max(pl1), max(pl2)) + 1.6)
ax.set_title("方案 B+C 前后各板放置数对比（客观指标）")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "B_C_placed_compare.png"), dpi=150)
plt.close(fig)

# ---------------- 3. reward / loss 曲线对比 ----------------
for metric in ("reward", "loss"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    for i, p in enumerate(pcbs):
        ax = axes[i]
        for base, label, color in ((BASELINE_DIR, "改进前", "#c86a6a"), (NEW_DIR, "改进后 B+C", "#2d6a4f")):
            hp = hist_path(base, p, f"{metric}_history.json")
            if not os.path.exists(hp):
                continue
            data = load_json(hp)  # [[ep, value], ...]
            eps = np.array([d[0] for d in data])
            vals = np.array([d[1] for d in data])
            ax.plot(eps, vals, alpha=0.15, color=color)
            ax.plot(eps[:len(smooth(vals))] if len(smooth(vals)) < len(eps) else eps, smooth(vals),
                    color=color, label=label, linewidth=1.5)
        ax.set_title(f"{labels[i]}（{n_mov[i]}件）")
        ax.set_xlabel("episode"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    ylabel = "reward（滑窗20平均）" if metric == "reward" else "TD loss（滑窗20平均）"
    fig.suptitle(f"{ylabel}：改进前后对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"B_C_{metric}_curves.png"), dpi=150)
    plt.close(fig)

# ---------------- 4. 对比表 md ----------------
lines = [
    "# 方案 B+C 前后结果对比（result1 改进前 vs result2 改进后）\n",
    "| 板子 | 元件数 | 改进前 placed | 改进后 placed | 变化 | 完成率 | 改进前后期 R | 改进后后期 R |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    pct = f"+{r['pct']:.0f}%" if r["pct"] > 0 else (f"{r['pct']:.0f}%" if r["pct"] < 0 else "持平")
    lines.append(
        f"| {r['board']} | {r['n_movable']} | {r['pl1']:.2f} | {r['pl2']:.2f} | {pct} "
        f"| {r['pl2']/r['n_movable']*100:.0f}% | {r['r1']:.0f} | {r['r2']:.0f} |"
    )
lines.append("")
lines.append("> 客观指标：placed（训练后期平均放置数）；完成率 = 改进后 placed / 元件数。")
lines.append("> 说明：reward 被 clip 过，跨版本只能看趋势，判断改进是否有效一律以 placed 为准。")
with open(os.path.join(OUT, "B_C_对比表.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("生成完成：")
for fn in os.listdir(OUT):
    if fn.startswith("B_C_"):
        print(" -", os.path.join(OUT, fn))
