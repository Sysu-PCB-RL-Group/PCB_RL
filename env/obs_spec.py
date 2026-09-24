# -*- coding: utf-8 -*-
"""obs 的**形状 / 通道序 / 标量归一化口径**（只放口径，不放 env 类）。

§5.5 定案（Owner 2026-09-22）：

```text
spatial = (8, 200, 200) float32     通道序 c = rot·2 + (0 = L, 1 = G)   ← 只在这一处定义
piece   = (3,) float32              [pad 数, 非电源 net 数, t/n] 的归一化值
```

**标量归一化（口径 + 实测依据 probe207）**

```text
pad_norm    = log1p(pad 数)        / log1p(PAD_MAX)      ∈ [0, 1]
np_net_norm = log1p(非电源 net 数)  / log1p(NET_MAX)      ∈ [0, 1]
t_norm      = index / n                                 ∈ [0, 1]（天然比例）
```

* **为什么 `log1p`**：两个量都**重尾**（pad 数 p50 = 2 / max = **80**；非电源 net 数 p50 = 1 / max = **28**）
  ⇒ 线性 `x/MAX` 会把中位件压成 **0.022**（等于不可见），`log1p` 版是 **0.250**。
  取 `log1p` 还保证 **`x = 0 ⇒ 0.000`、`x = MAX ⇒ 1.000` 精确命中两端**（不用裁剪）。
* **为什么分母用【实测 max】而不是 p99**（Owner 2026-09-22 拍）：让
  「**全部件的归一值 ∈ [0, 1]**」成为一条**可断言的不变量**（测试守护），且**不丢尾部排序**
  （p99 = 25 与 max = 80 仍分得开；若用 p99 再裁剪，最大的那批连接器/IC 会全塌成 1.0）。
* **为什么用【数据集固定常数】而不是"本板最大值"**：同一个件在任何板上必须编出**同一个数**
  （按本板归一 ⇒ 同一个 2-pad 件在 11-pad 板与 89-pad 板的编码会漂 1.8×）。
* ⚠️ **换数据集 ⇒ `PAD_MAX`/`NET_MAX` 必须重测**：`tests/kicad/test_obs_spec.py` 的守门断言
  （全数据集 `≤` 常数、且**存在件恰好 = 1.0**）会在超界时直接失败，逼我们重测而不是静默越界。
* 边界：`非电源 net 数 = 0` 的件（**153/668 = 22.9%**）归一后**精确 = 0.000**，
  与"有 net 但只 1 条"（= **0.206**）分得开 ⇒ **0 专门表示"这一步对目标无信号"**。
  超出 `MAX` 的值（换数据集后可能出现）**不裁剪**（保留"异常大"的信息）。
"""
from __future__ import annotations

import math
import os
from typing import Tuple

# ===================== spatial：通道序（与动作 [rot, X, Y] 同序：rot 在最前） =====================

N_ROT = 4                       # 旋转数（转换角 {0,1,2,3}）
SPATIAL_CHANNELS = 8            # 4 旋转 × (L, G)
_CH_OF = {"L": 0, "G": 1}       # 每个旋转组内的两个通道


def channel_of(rot: int, which: str = "L") -> int:
    """`(rot, 'L' | 'G')` → 通道下标 `c = rot·2 + (0 = L, 1 = G)`（**唯一实现**）。"""
    if int(rot) not in range(N_ROT):
        raise ValueError(f"rot 必须是 0~{N_ROT - 1}，收到 {rot!r}")
    key = str(which).upper()
    if key not in _CH_OF:
        raise ValueError(f"which 只能是 'L' 或 'G'，收到 {which!r}")
    return int(rot) * 2 + _CH_OF[key]


# ===================== piece 标量：归一化常数与函数 =====================

PAD_MAX = 80        # 实测（`scripts/dataset_report.py`，16 板 668 件）单件最大 pad 数（20 板时代 = 89）
NET_MAX = 28        # 实测（同上）单件最大【非电源】net 数（20 板时代 = 78；离群板已移入 data/bad_data）

PIECE_KEYS = ("pad", "np_net", "progress")     # piece (3,) 的语义顺序（写死）


def norm_count(x: int, hi: int) -> float:
    """`log1p(x) / log1p(hi)`：`x = 0 ⇒ 0.0`、`x = hi ⇒ 1.0`、单调不减；`x > hi ⇒ > 1`（不裁剪）。"""
    hi = int(hi)
    if hi <= 0:
        raise ValueError(f"归一化上界必须 > 0，收到 {hi}")
    return math.log1p(max(0, int(x))) / math.log1p(hi)


def norm_pad_count(pad_count: int) -> float:
    """pad 数 → `[0, 1]`（`PAD_MAX` 归一）。"""
    return norm_count(pad_count, PAD_MAX)


def norm_np_net_count(np_net_count: int) -> float:
    """非电源 net 数 → `[0, 1]`（`NET_MAX` 归一；**= 0 ⇒ 0.0，表示"这一步对目标无信号"**）。"""
    return norm_count(np_net_count, NET_MAX)


def norm_progress(index: int, n_total: int) -> float:
    """`t/n`（`t = index =` 已放件数）→ `[0, 1]`；`n_total <= 0` ⇒ **1.0**
    （与 `env/completion.py` 的边界口径一致：没有可放的东西 ⇒ 视为已完成）。"""
    n_total = int(n_total)
    if n_total <= 0:
        return 1.0
    return max(0.0, min(1.0, int(index) / n_total))


def piece_vector(pad_count: int, np_net_count: int, index: int, n_total: int
                 ) -> Tuple[float, float, float]:
    """`piece (3,)` = `(pad_norm, np_net_norm, t/n)` —— **唯一的组装入口**。"""
    return (norm_pad_count(pad_count),
            norm_np_net_count(np_net_count),
            norm_progress(index, n_total))


# ===================== G 的 −inf 净化（A3，Owner 2026-09-22 定案：非法格置 0） =====================

GAIN_FILL = 0.0                 # 非法格在"送网络的那一份 G"里填 0（Owner 2026-09-22 拍：0 = 无效/非法；
                                # 与合法格的 [0,1] 同域 ⇒ 不会被误读成高收益。不要改成 −1：BN 会放大非物理值）


def sanitize_gain(G, L, fill: float = GAIN_FILL):
    """把 `G` 的**非法格 `−inf`** 换成 `fill`（默认 **0.0**）⇒ 可安全送进网络。

    **为什么必须做**：`G` 的非法格是 `−inf`（§5.4 的契约，供 mask 用），而 `G` 又是 obs 输入
    ⇒ 直接进卷积 / BN / 归一化会**变成 NaN 并污染整批**（一个格就够）。
    非法格的信息已由 **`L` 通道**完全表达（两者成对出现，`c = rot·2 + (0=L, 1=G)`）
    ⇒ 这里把 `G` 置 0 **不丢信息**。

    **为什么填 0 而不是哨兵 −1**：0 与"合法格 `G ∈ [0,1]`"**同域**（0 = 该步最差），不会被误读成高收益；
    填 −1 会引入 `[−1, 0)` 的非物理值，BN 会把它放大。

    ⚠️ **`mask` 的计算必须继续用原始那份带 `−inf` 的 `G`**（§5.4 契约不变）；本函数只产出
    "送网络的那一份"。返回 float32 副本（不改入参）。
    """
    import numpy as np

    G = np.asarray(G)
    L = np.asarray(L, dtype=bool)
    if G.shape != L.shape:
        raise ValueError(f"G 与 L 形状不一致：{G.shape} vs {L.shape}")
    out = np.where(L, G, fill).astype(np.float32, copy=False)
    if os.environ.get("PCB_RL_AUDIT") == "1" and not np.isfinite(out).all():
        raise ValueError("净化后仍有非有限值（合法格里混进了 −inf/NaN）—— 这是上游 bug")
    return out
