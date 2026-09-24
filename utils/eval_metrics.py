# -*- coding: utf-8 -*-
"""项目评判标准的**单一实现**（台账 §10.2）—— 4 个指标 + "与专家组比较"的百分比。

口径（Owner 2026-09-23 拍定）
---------------------------
```text
① 完成度        = 已放件数 / 总件数            （**只看件数**；专家 ≡ 1.0000）
② HPWL 降低%    = (专家 − 模型) / 专家 × 100    （HPWL **越小越好** ⇒ 降低为正）
③ σ   提升%     = (模型 − 专家) / 专家 × 100    （σ **越大越好**，与 reward 的 `+α·Δσ` 同向 ⇒ 提升为正）
④ reward 提升%  = (R_模型 − R_专家) / |R_专家| × 100
   R = 一局累计 reward，**与训练用的 reward 完全一致**（§7.3 定稿）：
       R = −HPWL_final/H_ref + α·σ_final/D + β·n        （n = 放下的件数 = 步数，α = 5.6、γ = 1）
   专家侧 = 把 (HPWL_专家, σ_专家, n) 代进去（专家 HPWL ≡ H_ref ⇒ 首项 = −1）
```

两条必须知道的性质
------------------
1. **ΔR 与 β 无关**：`ΔR = (H_ref − HPWL_模型)/H_ref + α·(σ_模型 − σ_专家)/D`
   （两边的 β·n 相同 ⇒ 抵消。β 只影响"绝对值"，不影响"与专家的差距"。）
2. **分母取绝对值**：`|R_专家|` —— 避免 R_专家 为负时百分比符号翻转（实测 16 板 R_专家 全为正）。
3. ⚠️ **百分比的分母 `|R_专家|` 里含 `β·n`** ⇒ 若改 β（唯一可在线调的旋钮），`reward_gain_pct` **会变**；
   而 `reward_delta`（绝对差）**不受 β 影响**。这是"主口径与训练 reward 完全一致"（Owner 2026-09-23 拍）的必然结果
   （测试 `test_reward_delta_ignores_beta_but_percentage_does_not` 钉住）。

本模块是**纯函数**（不 import `env`、不碰文件）：`alpha` 由调用方传入 `env.reward.ALPHA`
（常数单点定义在 `env/reward.py`，这里不复制第二份）。
"""
from __future__ import annotations

import statistics as st
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = ["METRIC_KEYS", "episode_reward_total", "board_metrics", "aggregate"]

#: 会被"与专家组比较"的 4 个指标（主指标在前）—— 供日志/图表/测试统一取用
METRIC_KEYS = ("completion", "hpwl_drop_pct", "sigma_gain_pct", "reward_gain_pct")

_EPS = 1e-12


def episode_reward_total(*, hpwl: float, sigma: float, h_ref: float, d_board: float,
                         beta: float, n: int, alpha: float) -> float:
    """一局累计 reward（**与训练 reward 同一公式**，§7.3）：`−HPWL/H_ref + α·σ/D + β·n`。

    ⚠️ 这里用的是**终局**的 HPWL/σ（两个势差项 telescoping 后恰好只剩终局值），不是逐步累加；
    与 `env.reward.step_reward` 的逐步形式在数学上等价。
    """
    if h_ref <= 0.0 or d_board <= 0.0:
        raise ValueError(f"H_ref / D 必须 > 0，收到 h_ref={h_ref}, d_board={d_board}")
    return (-float(hpwl) / float(h_ref) + float(alpha) * float(sigma) / float(d_board)
            + float(beta) * int(n))


def _pct(num: float, den: float) -> Optional[float]:
    """`num/den × 100`；分母为 0（或 None）⇒ **`None`**（不猜、不填 0）。"""
    if den is None or abs(float(den)) < _EPS:
        return None
    return float(num) / float(den) * 100.0


def board_metrics(*, hpwl_model: float, hpwl_expert: float,
                  sigma_model: float, sigma_expert: float,
                  h_ref: float, d_board: float, beta: float, n: int,
                  alpha: float, n_total: Optional[int] = None) -> Dict[str, Any]:
    """单板的 4 个评判指标（连带把算它们用的**原始值**一起返回，便于对账）。

    Args:
        hpwl_model / hpwl_expert: 模型布局 / 专家（源位姿完整布局）的 HPWL（mm）。
        sigma_model / sigma_expert: 同上口径的 σ（跨 net 离散度，mm）。
        h_ref / d_board: 该板的两把尺（`env.reward.h_ref_of` / `d_board_of`）。
        beta: 该板的 β（`env.reward.beta_of`）—— 只进 reward 的绝对值，不影响与专家的差距。
        n: 放下的件数（= 步数）；`n_total` 给了才算 `completion`（否则记 `None`）。
        alpha: `env.reward.ALPHA`（**必传**，避免第二处定义常数）。
    """
    rw_model = episode_reward_total(hpwl=hpwl_model, sigma=sigma_model, h_ref=h_ref,
                                    d_board=d_board, beta=beta, n=n, alpha=alpha)
    rw_expert = episode_reward_total(hpwl=hpwl_expert, sigma=sigma_expert, h_ref=h_ref,
                                     d_board=d_board, beta=beta, n=n, alpha=alpha)
    return {
        # ---- ① 完成度（只看件数） ----
        "completion": (None if n_total is None else (float(n) / float(n_total) if n_total else 1.0)),
        "n_placed": int(n), "n_total": (None if n_total is None else int(n_total)),
        # ---- ② HPWL：越小越好 ⇒ 降低% ----
        "hpwl_model": float(hpwl_model), "hpwl_expert": float(hpwl_expert),
        "hpwl_drop_pct": _pct(float(hpwl_expert) - float(hpwl_model), hpwl_expert),
        # ---- ③ σ：越大越好 ⇒ 提升% ----
        "sigma_model": float(sigma_model), "sigma_expert": float(sigma_expert),
        "sigma_gain_pct": _pct(float(sigma_model) - float(sigma_expert), sigma_expert),
        # ---- ④ reward（与训练同式）：越大越好 ⇒ 提升%（分母 = |R_专家|） ----
        "reward_model": rw_model, "reward_expert": rw_expert, "reward_delta": rw_model - rw_expert,
        "reward_gain_pct": _pct(rw_model - rw_expert, abs(rw_expert)),
    }


def aggregate(rows: Sequence[Mapping[str, Any]],
              keys: Sequence[str] = METRIC_KEYS) -> Dict[str, Dict[str, Optional[float]]]:
    """把逐板指标聚成 `{key: {mean, median, min, max}}`；**`None` 一律跳过**（全 `None` ⇒ 各项记 `None`）。"""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "median": None, "min": None, "max": None, "n": 0.0}
            continue
        out[k] = {"mean": float(st.mean(vals)), "median": float(st.median(vals)),
                  "min": float(min(vals)), "max": float(max(vals)), "n": float(len(vals))}
    return out
