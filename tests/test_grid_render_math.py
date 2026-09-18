# -*- coding: utf-8 -*-
"""`env/grid_render.py` 的解析数学守门测试（不依赖 pcbnew）。

钉死三件事（都是曾经被质疑/写错过的地方）：
1. **障碍要"向外长"（⊕）、不是"内切"（⊖）** —— 负号方向守门；
2. **区间是严格开区间**（`gap == soft` 恰好算合法）；
3. 板框走相反方向（"向内缩"），由闭区间函数承担。
"""
import math

import pytest

from env import grid_render as gr


def test_anchor_region_is_dilated_not_inscribed():
    """Owner 算例：障碍 [0.2,2.8]²、s=0.2 ⇒ G=[0,3]²；pad 2×2（半尺寸 1.0）。
    正确（⊕ 外长）：锚点/中心违规区 = (−1, 4)²。
    若误写成内切（⊖）：会得到 [1,2]² —— 本测试就是用来抓这个方向的。
    """
    spec = gr.GridSpec(-2.0, -2.0, 0.5, 0.5, 20)
    B = [(0.2, 0.2, 2.8, 2.8)]
    import numpy as np
    B = np.array(B, dtype=float)
    i0, i1, j0, j1 = gr._rect_idx(B, cx=0.0, hx=1.0, cy=0.0, hy=1.0, soft=0.2, spec=spec)
    # 格中心 = -2 + (i+0.5)*0.5；要求 ∈ (−1, 4) ⇒ i ∈ [2, 11]
    assert (i0, i1) == (2, 11), f"锚点违规区方向错了：得到 i∈[{i0},{i1}]，期望 [2,11]"
    assert (j0, j1) == (2, 11)

    # 与解析区间对照：正确 = (−1, 4)；内切 = [1, 2] ⇒ 二者必然不同
    lo_x = B[0, 0] - 0.2 - 0.0 - 1.0        # bx1 − s − cx − hx = −1.0
    hi_x = B[0, 2] + 0.2 - 0.0 + 1.0        # bx2 + s − cx + hx =  4.0
    assert lo_x == pytest.approx(-1.0) and hi_x == pytest.approx(4.0)
    insc_lo, insc_hi = B[0, 0] + 0.2 + 1.0, B[0, 2] - 0.2 - 1.0
    assert (insc_hi - insc_lo) < (hi_x - lo_x), \
        f"内切区间必然更窄（{insc_lo}~{insc_hi}）⇒ 与本实现的 ⊕ 区间（{lo_x}~{hi_x}）不同"


def test_open_interval_excludes_exact_boundary():
    """严格开区间：中心恰好落在边界上 ⇒ **不算**在区间内。"""
    # 中心序列（origin=0, cell=0.5）：0.25, 0.75, 1.25 …
    i0, i1 = gr._idx_open(0.25, 0.75, 0.0, 0.5, 10)
    assert (i0, i1) == (1, 0), f"开区间应排除两端（空集），得到 ({i0},{i1})"
    # 闭区间：两端都算 ⇒ 包含 0.25 与 0.75
    i0c, i1c = gr._idx_closed(0.25, 0.75, 0.0, 0.5, 10)
    assert (i0c, i1c) == (0, 1), f"闭区间应包含两端，得到 ({i0c},{i1c})"


def test_threshold_tie_is_conservatively_forbidden():
    """`gap` 恰好 = 0.2 的格：浮点上无唯一答案 ⇒ **保守判为禁放**（`_EPS` 外扩）。

    构造：障碍 x=[0,0.05]、pad 半宽 0.5、s=0.2 ⇒ `hi_x` 恰好 = 0.75 = 格 i=1 的中心。
    策略：把禁区向外扩 1nm（`_EPS`），使"任何 Judge 会判违规的位置"必然被禁 ⇒
          危险方向恒为 0；代价是此类平局格被保守禁掉。
    """
    import numpy as np

    spec = gr.GridSpec(0.0, 0.0, 0.5, 0.5, 10)
    B = np.array([[0.0, 0.0, 0.05, 0.05]], dtype=float)
    # 几何校验：该格中心恰好落在"gap = 0.2"的边界上
    assert B[0, 2] + 0.5 + 0.2 == pytest.approx(0.75)
    assert spec.mm_of(1, 0)[0] == pytest.approx(0.75)

    i0, i1, _j0, _j1 = gr._rect_idx(B, cx=0.0, hx=0.5, cy=0.0, hy=0.5, soft=0.2, spec=spec)
    assert i0 <= 1 <= i1, f"平局格（i=1）应被**保守**纳入禁区，得到 i∈[{i0},{i1}]"
    # 参考：纯数学的开区间函数本身不含 `_EPS`（仍排除精确端点）
    assert gr._idx_open(0.25, 0.75, 0.0, 0.5, 10) == (1, 0)
