# -*- coding: utf-8 -*-
"""推理入口 —— 指定权重，**16 板各推理 1 次**，默认出可视化（台账 §9.2 推理块 / §10 可视化）。

Owner 2026-09-23 拍定的口径
```text
· 动作筛选**只用 `L`**（推理档，§6.3）：`PcbEnv(use_lookahead=False)` ⇒ `action_masks() == L`，不做 lookahead 收缩
· **纯贪心**：动作 = `mask` 内 argmax Q（走 `policy_net.masked_argmax`，**不经 ε 分支、不消耗 rng**）
  ⇒ 同一权重跑两次逐位相同（可复现）
· **默认出可视化**（`--no-viz` 可关）：每板一张 **2×2 对比图**（上=专家源位姿、下=模型推理；左顶层/右底层）
· 图上**默认不画 HPWL 包围框**；加 `--net-boxes` 才画（浅黄线框 = 每个非电源 net 的
  `(max_x−min_x)+(max_y−min_y)` 那个框；Owner 2026-09-23：做成可选项、默认关）；
  不画 net 连线；不做逐帧放置图（只看最终效果）
· 推理产物放 **`result/`** 下；可视化图**按板子名放入各自文件夹**
· **评判标准（§10.2，Owner 2026-09-23 拍）**：① 完成度（**只看件数**）② HPWL **降低%** ③ σ **提升%**
  ④ reward **提升%**（reward = 一局累计，**与训练同式** `−HPWL/H_ref + α·σ/D + β·n`；分母 `|R_专家|`）
  四项统一成「**正 = 更好**」；单一实现 = `utils/eval_metrics.py`（日志/summary/图表都从它取）
· 默认额外出 **3 张指标对比图**（`metrics/`；`--no-charts` 关）
· **每次推理前先清空上次的推理产物**（`result/infer_*`；不动 `result/` 下的其它内容）
```

产物
```text
result/infer_<权重名>_ep<权重里的局号>/
├── inference.log        人类可读，**两张表**（按**显示宽度**对齐：中文按 2 列，不再错位）：
│                        【评判表】board｜n｜完成度｜HPWL降低%｜σ提升%｜reward提升%｜终止（+ 均值行）
│                        【原始值表】模型/专家 的 HPWL、σ、reward（+ 均值行）
├── summary.json         机器可读：`metrics` = 4 指标的 mean/median/min/max；逐板行含 4 指标 + 原始值
├── metrics/             **3 张对比图**：HPWL 降低% / σ 提升% / reward 提升%（0 居中、不标数值）
└── <板名>/
    ├── <板名>.png       **2×2 对比图**（HPWL 包围框可选：默认不画，`--net-boxes` 打开才画）
    └── <板名>.json       该板 4 指标 + 原始值 + 位姿表（ref → (x, y, rot)）+ 放置顺序
```

用法
```text
& python scripts/infer_dqn.py                                 # 用 output/best.pt，16 板各 1 次，出图 + 日志
& python scripts/infer_dqn.py --ckpt output/ckpt_ep50.pt       # 指定权重（"指定参数"）
& python scripts/infer_dqn.py --boards <板名> --no-viz         # 单板调试 / 只出日志
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env import board_state as bs                   # noqa: E402
from env import grid_render as gr                   # noqa: E402
from env import policy_net as pn                    # noqa: E402
from env import reward as rw                        # noqa: E402
from env.dqn_agent import DoubleDQNAgent            # noqa: E402
from env.pcb_env import PcbEnv                      # noqa: E402
from utils import eval_metrics as em                # noqa: E402
from utils import viz_layout                        # noqa: E402  （不缩写：`viz` 是下面的布尔参数）
from utils import viz_metrics                       # noqa: E402
from utils.kicad_utils import model_cache as mc     # noqa: E402

DEFAULTS: Dict[str, Any] = {
    "ckpt": os.path.join("output", "best.pt"),
    "out": "result",
    "boards": "",
    "dpi": 110,
    "viz": True,
    "net_boxes": False,      # ★ HPWL 包围框：**默认不画**（Owner 2026-09-23）；`--net-boxes` 打开
    "charts": True,          # ★ 3 张"与专家组比较"的百分比对比图（HPWL/σ/reward）；`--no-charts` 关
    "device": "cpu",
    "cache_root": None,
}


def _disp_width(text: Any) -> int:
    """字符串在**等宽终端**里的显示宽度（东亚宽/全角字符按 2 列算）。

    为什么需要它：Python 的 `ljust` / f-string 按**字符数**补白，而中文在终端占 **2** 列
    ⇒ 含中文表头的表格必然错位（Owner 2026-09-23 反馈"看得难受"就是这个原因）。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad(text: Any, width: int, *, right: bool = False) -> str:
    """按**显示宽度**补白到 `width` 列（`right=True` ⇒ 右对齐，用于数字列）。"""
    s = str(text)
    gap = max(0, int(width) - _disp_width(s))
    return (" " * gap + s) if right else (s + " " * gap)


def _num(v: Any, *, digits: int = 2, suffix: str = "", sign: bool = False) -> str:
    """`None` ⇒ `"—"`（**不猜成 0**）；否则固定小数位（`sign=True` ⇒ 强制带正负号）+ 后缀。"""
    if v is None:
        return "—"
    fmt = f"{{:+.{int(digits)}f}}" if sign else f"{{:.{int(digits)}f}}"
    return fmt.format(float(v)) + suffix


def all_boards(cache_root: Optional[str] = None) -> List[str]:
    return [e["board"] for e in mc.read_index(root=cache_root)["boards"]]


def tag_of(ckpt: str, episode: Optional[int]) -> str:
    """产物目录名：`<权重文件名去扩展名>_ep<局号>`（局号来自权重文件里存的 episode，没有就省略）。"""
    stem = os.path.splitext(os.path.basename(str(ckpt)))[0]
    return f"{stem}_ep{int(episode)}" if episode is not None else stem


def load_agent(ckpt: str, *, device: str = "cpu") -> Tuple[DoubleDQNAgent, Dict[str, Any]]:
    """读权重 → `(agent, meta)`；`meta` 里有 `episode`（权重存进去时的局号）与当时的评估指标。

    ⚠️ 推理**不需要回放**（不学习）⇒ `capacity=1, batch_size=1`，避免白白预分配 3.8 GB。
    """
    import torch
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "agent" not in blob:
        raise ValueError(f"权重文件格式不对（应含 'agent' 键，来自 scripts/train_dqn.py 的断点）：{ckpt}")
    agent = DoubleDQNAgent(capacity=1, batch_size=1, device=device)
    agent.load_state_dict(blob["agent"])
    agent.net.eval()
    meta = {"episode": blob.get("episode"), "eval": blob.get("eval"), "config": blob.get("config")}
    return agent, meta


def greedy_action(agent: DoubleDQNAgent, obs: Mapping[str, np.ndarray], mask: np.ndarray) -> Tuple[int, int, int]:
    """**纯贪心**（Owner 2026-09-23）＝ `mask` 内 argmax Q，走 `policy_net.masked_argmax`。

    ⚠️ 刻意**不走** `agent.act(..., explore=False)`：那条路虽在 `eps=0` 时也返回同一个动作，
    但仍经过 ε 分支（未来若有人改默认 ε / 改 `masked_epsilon_greedy`，推理就会悄悄沾上随机性）。
    这里直接调 `masked_argmax`：**没有 rng 参与 ⇒ 同一权重跑两次逐位相同**（可复现）。
    """
    return pn.masked_argmax(agent.q_values(obs)[0], mask)


def run_board(env: PcbEnv, agent: DoubleDQNAgent, board: str) -> Dict[str, Any]:
    """单板**纯贪心**推理（**mask = L**）→ 指标 + 位姿表。"""
    t0 = time.perf_counter()
    obs, info = env.reset(options={"board": board})
    n_degen = 0
    while True:
        mask = env.action_masks()
        if not mask.any():                        # 推理档终止：L 空（物理无位置）
            break
        obs, _r, term, trunc, info = env.step(greedy_action(agent, obs, mask))
        if env.frame is not None:
            n_degen += int(bool(env.frame.degenerate))
        if term or trunc:
            break

    st = env.state
    placed = set(st.placed_refs)
    hpwl_model = float(env.model.hpwl(refs=placed, poses=st.poses))     # "真实 HPWL"（模型布局，当前位姿）
    hpwl_expert = float(env.h_ref)                                      # 专家完整布局 HPWL（= H_ref）
    hpwl_expert_same = float(env.model.hpwl(refs=placed))               # 同件集下的专家 HPWL（公平对比用）

    # σ（跨 net 离散度，mm）：与 reward 项同一口径、同一个函数（§5.4 / §7.3）
    # ⚠️ `dispersion_of` 要求"位姿化克隆"⇒ 模型的用 `state.placed`，专家的走 `board_state.derive_placed`
    #    （同一实现，避免手抄位姿）
    expert_placed = bs.derive_placed(env.model, viz_layout.expert_poses_of(env.model))
    sigma_model = float(gr.dispersion_of(env.model, st.placed))                 # ★ 真实 σ
    sigma_expert = float(gr.dispersion_of(env.model, expert_placed))            # ★ 专家完整布局 σ
    sigma_expert_same = float(gr.dispersion_of(                          # 同件集专家 σ（放完时 == sigma_expert）
        env.model, tuple(f for f in expert_placed if f.ref in placed)))

    poses = {r: [float(st.poses[r][0]), float(st.poses[r][1]), int(st.poses[r][2])] for r in st.placed_refs}

    # ★ 项目评判标准（§10.2）的 4 个指标 —— 单一实现在 `utils/eval_metrics.py`
    #   reward 用**与训练完全相同的式子**（§7.3）：R = −HPWL/H_ref + α·σ/D + β·n
    met = em.board_metrics(hpwl_model=hpwl_model, hpwl_expert=hpwl_expert,
                           sigma_model=sigma_model, sigma_expert=sigma_expert,
                           h_ref=float(env.h_ref), d_board=float(env.d_board), beta=float(env.beta),
                           n=int(info["n_placed"]), alpha=rw.ALPHA, n_total=int(info["n_total"]))
    return {
        "board": board,
        "n_total": int(info["n_total"]), "n_placed": int(info["n_placed"]),
        "completion": float(info["completion"]),
        "hpwl_model": hpwl_model,                  # ★ 真实 HPWL
        "hpwl_expert": hpwl_expert,                # ★ 专家 HPWL
        "hpwl_expert_on_placed": hpwl_expert_same,  # 同件集专家 HPWL（放完时 == hpwl_expert）
        "hpwl_ratio": hpwl_model / max(hpwl_expert, 1e-9),
        "hpwl_ratio_same_set": hpwl_model / max(hpwl_expert_same, 1e-9),
        "sigma_model": sigma_model,                # ★ 真实 σ（mm）
        "sigma_expert": sigma_expert,              # ★ 专家 σ（mm）
        "sigma_expert_on_placed": sigma_expert_same,  # 同件集专家 σ（放完时 == sigma_expert）
        "sigma_ratio": sigma_model / max(sigma_expert_same, 1e-9),
        "sigma_over_d": sigma_model / max(float(env.d_board), 1e-9),  # σ/D（reward 里 σ 项的分母 ⇒ 跨板可比）
        "steps": int(info["n_placed"]),
        # ---- 项目评判标准（§10.2）：4 指标 ----
        "hpwl_drop_pct": met["hpwl_drop_pct"],        # ② HPWL 降低%（正 = 线长比专家短）
        "sigma_gain_pct": met["sigma_gain_pct"],      # ③ σ 提升%（正 = 离散度比专家大）
        "reward_model": met["reward_model"],          # ④ 一局 reward（与训练同式）
        "reward_expert": met["reward_expert"],
        "reward_delta": met["reward_delta"],
        "reward_gain_pct": met["reward_gain_pct"],    # ④ 提升%（分母 = |R_专家|）
        "stopped_reason": "all_placed" if int(info["n_placed"]) == int(info["n_total"]) else "L_empty",
        "deadlock_kind": info["deadlock_kind"], "at": info["at"],
        "degenerate_steps": n_degen,
        "seconds": round(time.perf_counter() - t0, 2),
        "poses": poses, "order": list(env.order),
    }


def clear_previous_inference(out: str) -> List[str]:
    """**每次推理前清空上次的推理产物**（Owner 2026-09-23）。

    删掉产物根目录（默认 `result/`）下**所有 `infer_*` 条目**（含上一次不同权重的目录）⇒ 不会新旧混在一起。
    ⚠️ **只动 `infer_*`**（我们自己建的）：`result/` 下的其它内容（如 `kicad_board_scan/` —— 数据集扫描工具的
    输出，**仍在使用**）**一律不动** —— 删了就得重跑工具才能恢复。
    ⚠️ 安全闸：产物根目录若就是仓库根 ⇒ 拒绝（防手滑 `--out .`）。

    Returns: 被删掉的条目名列表（写进日志，便于对账）。
    """
    r = os.path.abspath(str(out))
    if r == os.path.abspath(_ROOT):
        raise ValueError(f"拒绝清理 {r}：它是仓库根（只允许清理产物根目录下的 infer_* 条目）")
    if not os.path.isdir(r):
        os.makedirs(r, exist_ok=True)
        return []
    removed: List[str] = []
    for name in sorted(os.listdir(r)):
        if not name.startswith("infer_"):
            continue
        q = os.path.join(r, name)
        if os.path.isdir(q) and not os.path.islink(q):
            shutil.rmtree(q)
        else:
            os.remove(q)
        removed.append(name)
    return removed


def infer(*, ckpt: str = DEFAULTS["ckpt"], out: str = DEFAULTS["out"], boards: Optional[Sequence[str]] = None,
          viz: bool = DEFAULTS["viz"], net_boxes: bool = DEFAULTS["net_boxes"],
          charts: bool = DEFAULTS["charts"],
          dpi: int = DEFAULTS["dpi"], device: str = DEFAULTS["device"],
          cache_root: Optional[str] = None, agent: Optional[DoubleDQNAgent] = None,
          env: Optional[PcbEnv] = None, verbose: bool = True) -> Dict[str, Any]:
    """16 板各推理 1 次（**只用 L**）→ 出图 + 两份日志；返回汇总字典。"""
    meta: Dict[str, Any] = {}
    if agent is None:
        agent, meta = load_agent(ckpt, device=device)
    tag = tag_of(ckpt, meta.get("episode"))
    removed = clear_previous_inference(out)          # ★ 每次推理前清空上次的推理产物（只清 infer_*）
    root = os.path.join(out, f"infer_{tag}")
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, "inference.log")

    board_list = list(boards) if boards else all_boards(cache_root)
    env = env if env is not None else PcbEnv(boards=board_list, cache_root=cache_root,
                                             use_lookahead=False,        # ★ 只用 L（Owner 拍定）
                                             verify=True, with_contrib=False)

    tlog = open(log_path, "w", encoding="utf-8")

    def say(msg: str) -> None:
        tlog.write(msg + "\n")
        tlog.flush()
        if verbose:
            print(msg, flush=True)

    t0 = time.perf_counter()
    say(f"[ckpt] {os.path.abspath(ckpt)}"
        + (f"（权重里的局号 ep{meta.get('episode')}｜当时评估 {meta.get('eval')}）" if meta else "")
        + f"｜**mask = L only**（不做 lookahead 收缩）｜**纯贪心**（masked_argmax，不经 ε）"
          f"｜板 {len(board_list)} 块｜dpi {dpi}｜HPWL 包围框 {'开' if net_boxes else '关（默认）'}")
    say(f"[clean] 清空上次推理产物：{'、'.join(removed) if removed else '（无）'}"
        f"｜产物根 {os.path.abspath(out)}（只清 infer_*，历史内容不动）")
    say("**[HPWL] 非电源 net 的 (max_x−min_x)+(max_y−min_y) 求和；[σ] 跨 net 离散度 sqrt(mean|c|²−|mean c|²)"
        "（net 中心 = bbox 中心；两把尺都与专家组比较，专家 = 源位姿布局）**")
    # ---- 表格列宽：**按显示宽度对齐**（中文列名占 2 列，见 `_pad` / `_disp_width`）----
    W_B, W_N, W_SEP = 30, 8, 104
    say("=" * W_SEP)
    say("【评判表】与专家组比较（§10.2，四项都统一成「正 = 更好」）：HPWL **降低%**（正 = 线长比专家短）｜"
        "σ **提升%**（σ 越大越好）｜reward **提升%**（一局累计，**与训练同一式子**）")
    say(_pad("board", W_B) + _pad("n", W_N, right=True) + _pad("完成度", 9, right=True)
        + _pad("HPWL降低%", 12, right=True) + _pad("σ提升%", 10, right=True)
        + _pad("reward提升%", 13, right=True) + "  终止")
    say("-" * W_SEP)

    rows: List[Dict[str, Any]] = []
    lines_raw: List[str] = []                    # 原始值表（循环后一起打，避免两表交错）
    for b in board_list:
        r = run_board(env, agent, b)
        rows.append(r)
        say(_pad(b[:22], W_B) + _pad(f"{r['n_placed']}/{r['n_total']}", W_N, right=True)
            + _pad(f"{r['completion']:.4f}", 9, right=True)
            + _pad(_num(r["hpwl_drop_pct"], suffix="%"), 12, right=True)
            + _pad(_num(r["sigma_gain_pct"], suffix="%"), 10, right=True)
            + _pad(_num(r["reward_gain_pct"], suffix="%"), 13, right=True)
            + "  " + str(r["stopped_reason"]))
        lines_raw.append(_pad(b[:22], W_B)
                         + _pad(f"{r['hpwl_model']:.2f}", 10, right=True)
                         + _pad(f"{r['hpwl_expert']:.2f}", 10, right=True)
                         + _pad(f"{r['sigma_model']:.3f}", 9, right=True)
                         + _pad(f"{r['sigma_expert']:.3f}", 9, right=True)
                         + _pad(f"{r['reward_model']:+.3f}", 10, right=True)
                         + _pad(f"{r['reward_expert']:+.3f}", 10, right=True))
        if viz:
            board_dir = os.path.join(root, b)
            os.makedirs(board_dir, exist_ok=True)
            png = os.path.join(board_dir, f"{b}.png")
            stats = viz_layout.draw_compare(env.model, model_poses=r["poses"], out_path=png,
                                            keepouts_model=env.state.keepouts,
                                            show_net_boxes=net_boxes,        # ★ 默认 False = 不画
                                            subtitle=(f"真实 HPWL {r['hpwl_model']:.2f} mm"
                                                      f"｜专家 HPWL {r['hpwl_expert']:.2f} mm"
                                                      f"｜HPWL÷H_ref {r['hpwl_ratio']:.4f}"
                                                      f"｜完成度 {r['completion']:.4f}"),
                                            dpi=dpi, board=b)
            r["png"] = os.path.relpath(png, root).replace("\\", "/")
            r["net_boxes"] = {"expert": int(stats["n_net_boxes"]["expert"]),   # 图上画了多少个 HPWL 框
                              "model": int(stats["n_net_boxes"]["model"])}
            with open(os.path.join(board_dir, f"{b}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, ensure_ascii=False, indent=1)

    def agg(key: str) -> Dict[str, Optional[float]]:
        """逐板聚合：**跳过 `None`**（分母为 0 的指标）；全 `None` ⇒ 四项都记 `None`（不猜 0）。"""
        v = np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
        if v.size == 0:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max())}

    # ---- ★ 项目评判标准（§10.2）的 4 指标汇总 ----
    metrics = em.aggregate(rows)
    say("-" * W_SEP)
    say("[mean·评判] 完成度 " + _num(metrics["completion"]["mean"], digits=4)
        + "｜HPWL 降低 " + _num(metrics["hpwl_drop_pct"]["mean"], suffix="%")
        + "（中位 " + _num(metrics["hpwl_drop_pct"]["median"], suffix="%") + "）"
        + "｜σ 提升 " + _num(metrics["sigma_gain_pct"]["mean"], suffix="%")
        + "（中位 " + _num(metrics["sigma_gain_pct"]["median"], suffix="%") + "）"
        + "｜reward 提升 " + _num(metrics["reward_gain_pct"]["mean"], suffix="%")
        + "（中位 " + _num(metrics["reward_gain_pct"]["median"], suffix="%") + "）")

    # ---- ★ 3 张"与专家组比较"的百分比对比图（HPWL / σ / reward；0 居中、不标数值） ----
    chart_files: Dict[str, Any] = {}
    if charts:
        chart_files = viz_metrics.draw_all_metrics(rows, os.path.join(root, "metrics"), dpi=dpi)
        say("[图] 3 张对比图 → `metrics/`：" + "、".join(os.path.basename(v["path"])
                                                    for v in chart_files.values())
            + f"（每张 {len(rows)} 柱；**正 = 优于专家**）")

    elapsed = time.perf_counter() - t0
    summary = {
        "kind": "inference_summary",
        "ckpt": os.path.abspath(ckpt), "ckpt_episode": meta.get("episode"), "ckpt_metrics": meta.get("eval"),
        "mask_mode": "L_only", "lookahead": False, "policy": "pure_greedy(masked_argmax)", "dpi": dpi,
        "net_boxes": bool(net_boxes),          # 图上是否画了 HPWL 包围框（默认 False）
        "charts_on": bool(charts), "charts": chart_files,
        "cleared_previous": removed,
        "n_boards": len(rows), "elapsed_s": round(elapsed, 1),
        "boards": rows,
        # ★ 评判标准（§10.2）的 4 项汇总（`metrics` = 单一实现聚合；其余是原始值的分位数）
        "metrics": metrics,
        "hpwl_drop_pct": agg("hpwl_drop_pct"), "sigma_gain_pct": agg("sigma_gain_pct"),
        "reward_gain_pct": agg("reward_gain_pct"), "reward_delta": agg("reward_delta"),
        "completion": agg("completion"), "hpwl_model": agg("hpwl_model"),
        "hpwl_expert": agg("hpwl_expert"), "hpwl_ratio": agg("hpwl_ratio"),
        "sigma_model": agg("sigma_model"), "sigma_expert": agg("sigma_expert"),
        "sigma_ratio": agg("sigma_ratio"), "sigma_over_d": agg("sigma_over_d"),
        "reward_model": agg("reward_model"), "reward_expert": agg("reward_expert"),
        "deadlocks": int(sum(r["deadlock_kind"] is not None for r in rows)),
    }
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    # ---- 第二张表：原始值（模型 vs 专家；均值行取 summary 的分位数） ----
    say("=" * W_SEP)
    say("【原始值表】模型 vs 专家（HPWL / σ 单位 mm；reward = 一局累计 `−HPWL/H_ref + α·σ/D + β·n`，"
        "**与训练同式**，越大越好）")
    say(_pad("board", W_B) + _pad("模型HPWL", 10, right=True) + _pad("专家HPWL", 10, right=True)
        + _pad("模型σ", 9, right=True) + _pad("专家σ", 9, right=True)
        + _pad("模型reward", 11, right=True) + _pad("专家reward", 11, right=True))
    say("-" * W_SEP)
    for ln in lines_raw:
        say(ln)
    say("-" * W_SEP)
    say(_pad("（均值）", W_B)
        + _pad(_num(summary["hpwl_model"]["mean"]), 10, right=True)
        + _pad(_num(summary["hpwl_expert"]["mean"]), 10, right=True)
        + _pad(_num(summary["sigma_model"]["mean"], digits=3), 9, right=True)
        + _pad(_num(summary["sigma_expert"]["mean"], digits=3), 9, right=True)
        + _pad(_num(summary["reward_model"]["mean"], digits=3, sign=True), 11, right=True)
        + _pad(_num(summary["reward_expert"]["mean"], digits=3, sign=True), 11, right=True))

    # ---- 收尾汇总（按 reward 提升% 排序；综合指标 = reward） ----
    order = sorted(rows, key=lambda r: (r["reward_gain_pct"] is None, -(r["reward_gain_pct"] or 0.0)))
    say("=" * W_SEP)
    say("[mean] 真实 HPWL " + _num(summary["hpwl_model"]["mean"]) + " mm｜专家 HPWL "
        + _num(summary["hpwl_expert"]["mean"]) + " mm｜σ " + _num(summary["sigma_model"]["mean"], digits=3)
        + " / " + _num(summary["sigma_expert"]["mean"], digits=3) + " mm｜早终止 "
        + f"{summary['deadlocks']}/{len(rows)}｜用时 {elapsed:.1f} s")
    say("[最好 3·按 reward 提升%] " + "｜".join(
        f"{r['board'][:24]} {_num(r['reward_gain_pct'], suffix='%')}" for r in order[:3]))
    say("[最差 3·按 reward 提升%] " + "｜".join(
        f"{r['board'][:24]} {_num(r['reward_gain_pct'], suffix='%')}" for r in order[-3:]))
    say(f"[产物] {os.path.abspath(root)}（inference.log / summary.json / metrics/ 三张对比图 / "
        f"<板名>/<板名>.png + .json）")
    say("=" * W_SEP)
    tlog.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="推理入口（16 板各 1 次，只用 L，默认出可视化）")
    p.add_argument("--ckpt", default=DEFAULTS["ckpt"], help="权重文件（默认 output/best.pt）")
    p.add_argument("--out", default=DEFAULTS["out"], help="产物根目录（默认 result/）")
    p.add_argument("--boards", default=DEFAULTS["boards"], help="逗号分隔的板名（缺省 = 全部 16 板）")
    p.add_argument("--dpi", type=int, default=DEFAULTS["dpi"])
    p.add_argument("--no-viz", dest="viz", action="store_false", default=DEFAULTS["viz"],
                   help="关闭可视化（只出日志）")
    p.add_argument("--net-boxes", dest="net_boxes", action="store_true", default=DEFAULTS["net_boxes"],
                   help="在图上画 HPWL 包围框（浅黄线框；**默认不画**）")
    p.add_argument("--no-charts", dest="charts", action="store_false", default=DEFAULTS["charts"],
                   help="不出 3 张指标对比图（`metrics/` 目录）")
    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--cache-root", default=DEFAULTS["cache_root"])
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    boards = [b.strip() for b in args.boards.split(",") if b.strip()] or None
    summary = infer(ckpt=args.ckpt, out=args.out, boards=boards, viz=args.viz, dpi=args.dpi,
                    net_boxes=args.net_boxes, charts=args.charts,
                    device=args.device, cache_root=args.cache_root)
    print(json.dumps({"n_boards": summary["n_boards"], "elapsed_s": summary["elapsed_s"],
                      "hpwl_ratio_mean": summary["hpwl_ratio"]["mean"],
                      "completion_mean": summary["completion"]["mean"],
                      "deadlocks": summary["deadlocks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
