# -*- coding: utf-8 -*-
"""单步预测掩膜（lookahead mask）—— 台账 §0.6 未决 n；口径由 Owner 2026-09-18 给定。

定义
----
对当前待放件 `comp_now` 的每个候选动作 `(k, i, j)`（须 `L_now[k][i][j] == True`）：

    把 `comp_now` **真的放在**该位姿上，看**下一个件**（固定顺序）是否仍有合法格；
    有 ⇒ 保留该动作；没有 ⇒ 该动作被 mask 掉（结果**必然比 `L` 更严格**）。

Owner 指定的算法形态（含单调性带来的关键简化）
--------------------------------------------
```text
1. 放 comp_now 之前，先算一次"下一个件"的 L_next0（== 没放 comp_now、但选中 comp_next 时的 L）
2. 遍历 comp_now 的每个合法候选动作：把 comp_now 新造成的"禁区"从 L_next0 里减掉
     L_next(放之后) = L_next0 \ 新覆盖的非法区        ← 增量压缩，不重算整张 L_next
   只要 L_next(放之后) 非空 ⇒ 该候选保留（早停：找到一个存活格就收工）
3. 遍历完得到 mask（⊆ L_now）
```
* 单调性依据：放置只**增加**障碍 ⇒ 后件合法格只减不增 ⇒ 增量压缩与"从零重算"**等价**。
* `L_next0` 每步只算一次（下一步本来也要用 ⇒ 可视为半价）。

边界（Owner 2026-09-18 确认）
---------------------------
`L_next0` 本身为空 ⇒ **全部 pass（保留 `L` 不变）**："下一个件当前就没位置"不是本候选的
过错（放 `comp_now` 只会更糟、永远救不回来），照算会把 mask 清空；"已死"由 env 终止逻辑
接管（台账 §5.4③）。

实现（2026-09-18，把"逐候选几何判定"换成"位移表 + 批量索引"）
----------------------------------------------------------
利用**平移不变性**：`comp_now@α` 与 `comp_next@β` 是否冲突**只取决于锚点差 α−β**。
而两个锚点都落在同一套格点上 ⇒ 差值等价于**整数格差** `(dI, dJ)` ⇒ 可以预先做一张
**位移表**（每个 `(k, kf)` 一张，用矩形 stamping 一次算出）：

```text
Tmap[dI][dJ] = True  ⟺  锚点相差 (dI·cell_x, dJ·cell_y) 时两者冲突
L_next(放之后) 非空  ⟺  ∃ c ∈ L_next0 : not Tmap[p_i − c_i][p_j − c_j]
```
于是每个候选只做 **|L_next0| 次整数查表**，且**对所有候选批量一次完成**（numpy fancy index），
不需要任何逐候选的几何判定、不需要"充分判据 + 回退"。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from utils.kicad_utils.pcb_model import Footprint, PcbModel, rotate_aabb, state_keepouts

from .grid_render import GRID_M, SOFT_GAP_MM, GridSpec, grid_spec, legal_masks

__all__ = ["lookahead_mask", "compress_legal", "killed_rects", "template_rects", "conflict_table"]


# --------------------------------------------------------------------------- 几何

def _layer_bits(layers: Sequence[int]) -> int:
    m = 0
    for L in layers:
        m |= 1 << int(L)
    return m


def _pads(comp: Footprint, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """`comp` 在旋转 k 下、**相对锚点**的 pad AABB `(n,4)` + 层掩码 `(n,)`。"""
    boxes, masks = [], []
    for pad in comp.pads:
        if not pad.local_aabb:
            continue
        boxes.append(rotate_aabb(pad.local_aabb, k * 90.0, (0.0, 0.0)))
        masks.append(_layer_bits(pad.layers))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    return np.asarray(boxes, dtype=np.float64), np.asarray(masks, dtype=np.int64)


def _slots(comp: Footprint, model: PcbModel, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """`comp` 自带禁布区（开槽）在旋转 k 下、**相对锚点**的 AABB（开槽**随件旋转**）。

    ⚠️ `comp` 必须是**未放置的源位姿对象**：本函数对 `model.keepouts` 里的**源位姿 AABB**
    做逆变换（`R(+θ_src)` 相对源锚点）再按 k 旋转 ⇒ 若传入已移动的克隆，逆变换前提不成立。
    这是"**待放件自己的槽**"（不是当前局面禁布区集合）—— 后者统一走 `state_keepouts`（§0.6 q-④）。
    """
    boxes, masks = [], []
    th = math.radians(comp.rotation_index * 90.0)
    c, s = math.cos(th), math.sin(th)
    for kp in model.keepouts:
        if kp.owner_ref != comp.ref:
            continue
        xs, ys = [], []
        for bx, by in ((kp.aabb[0], kp.aabb[1]), (kp.aabb[2], kp.aabb[1]),
                       (kp.aabb[2], kp.aabb[3]), (kp.aabb[0], kp.aabb[3])):
            dx, dy = bx - comp.anchor[0], by - comp.anchor[1]
            xs.append(dx * c - dy * s)
            ys.append(dx * s + dy * c)
        boxes.append(rotate_aabb((min(xs), min(ys), max(xs), max(ys)), k * 90.0, (0.0, 0.0)))
        masks.append(_layer_bits(kp.layers))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    return np.asarray(boxes, dtype=np.float64), np.asarray(masks, dtype=np.int64)


def _rects(a_boxes, a_masks, b_boxes, b_masks, px: float, py: float) -> np.ndarray:
    """A 放 `(px,py)` 时，使 **B 锚点**落在冲突区的矩形 `(n,4)`（B 锚点坐标系）。

    推导：A 的 pad（相对框 `a`）与 B 的 pad（相对框 `b`）冲突 ⟺ 四个 gap 项都 < `SOFT`
    ⇒ B 锚点坐标 `x ∈ (px + a.x1 − b.x2 − SOFT, px + a.x2 − b.x1 + SOFT)`（y 同理）。
    """
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return np.zeros((0, 4))
    ia, ib = np.where((a_masks[:, None] & b_masks[None, :]) != 0)
    if len(ia) == 0:
        return np.zeros((0, 4))
    a, b = a_boxes[ia], b_boxes[ib]
    x1 = px + a[:, 0] - b[:, 2] - SOFT_GAP_MM
    x2 = px + a[:, 2] - b[:, 0] + SOFT_GAP_MM
    y1 = py + a[:, 1] - b[:, 3] - SOFT_GAP_MM
    y2 = py + a[:, 3] - b[:, 1] + SOFT_GAP_MM
    keep = (x2 > x1) & (y2 > y1)
    if not keep.any():
        return np.zeros((0, 4))
    return np.stack([x1[keep], x2[keep], y1[keep], y2[keep]], axis=1)


def killed_rects(model: PcbModel, comp_now: Footprint, k: int,
                 comp_next: Footprint, kf: int, px: float, py: float) -> np.ndarray:
    """`comp_now` 放 `(k,px,py)` 后，`comp_next`（旋转 kf）锚点被禁的矩形集合。

    三组：`now.pad ↔ next.pad`、`now.slot ↔ next.pad`、`now.pad ↔ next.slot`。
    """
    an, am = _pads(comp_now, k)
    asl, asm = _slots(comp_now, model, k)
    bn, bm = _pads(comp_next, kf)
    # ⚠️ **不包含** `now.pad ↔ next.slot`：口径 (A) 下 `comp_next` 尚未放置 ⇒ 它的槽不在板上；
    #    且 `comp_next` 自己的合法格本就**豁免自家槽**（§2.4）⇒ 该对与"next 是否还有位置"无关
    #    （保留它会让增量比真实 `L_next(after)` 更严，破坏"增量 == 从零重算"，§0.6 p）。
    parts = [_rects(an, am, bn, bm, px, py),
             _rects(asl, asm, bn, bm, px, py)]
    parts = [x for x in parts if len(x)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 4))


def template_rects(model: PcbModel, comp_now: Footprint, k: int,
                   comp_next: Footprint, kf: int) -> np.ndarray:
    """`(k, kf)` 的**相对模板**（= 把 `comp_now` 放在原点时的被杀矩形）；候选处只是平移。"""
    return killed_rects(model, comp_now, k, comp_next, kf, 0.0, 0.0)


def conflict_table(model: PcbModel, comp_now: Footprint, k: int,
                   comp_next: Footprint, kf: int, spec: GridSpec):
    """`(k, kf)` 的紧凑位移表 → `(T, Ki, Kj)`。

    **索引约定（⚠️ 勿写反）**：`T[dI+Ki][dJ+Kj] = True` ⟺ `d_next − d_now = (dI, dJ)`（格）时冲突，
    即 `dI = next_i − now_i`（= 模板"把 now 放原点、B=next 锚点坐标系"的自然约定）。
    """
    return conflict_table_from(template_rects(model, comp_now, k, comp_next, kf), spec)


def conflict_table_from(tpl: np.ndarray, spec: GridSpec):
    """由**相对模板矩形**一次 stamping 出**紧凑**位移表 → `(T, Ki, Kj)`（向量化，无 Python 循环）。

    `T[dI+Ki][dJ+Kj] = True` ⟺ **`d_next − d_now = (dI, dJ)`**（格）时冲突
    （`dI = next_i − now_i`；模板 `template_rects` 就是"now 在原点时 B=next 锚点的禁区"，
    与 `killed_rects(..., px, py)` 的绝对口径互为平移）。

    ⚠️ **曾有符号写反**（查表用 `now − next`）⇒ 只在"精确路径"命中且模板不对称时漏杀，
    表现为**误留**（把下一个件堵死的动作被保留）；已由 `test_conflict_table_matches_geometry` 钉住。
    * 表范围**只覆盖模板外接框**（`|dI| ≤ Ki`），并留 1 圈 False ⇒ 查表时把 `dI` **clamp**
      到 `[-Ki, Ki]` 即可：越界者本来就不在模板内（必 False），clamp 后落在 False 圈上 ✓。
    * 区间取**开**（`x1 < dI·cell_x < x2`），与判据 `gap < SOFT` 严格一致。
    """
    if len(tpl) == 0:
        return np.zeros((1, 1), dtype=bool), 0, 0
    lo_i = int(math.floor(float(tpl[:, 0].min()) / spec.cell_x))
    hi_i = int(math.ceil(float(tpl[:, 1].max()) / spec.cell_x))
    lo_j = int(math.floor(float(tpl[:, 2].min()) / spec.cell_y))
    hi_j = int(math.ceil(float(tpl[:, 3].max()) / spec.cell_y))
    Ki = max(abs(lo_i), abs(hi_i)) + 1                 # +1 圈 False（clamp 保护区）
    Kj = max(abs(lo_j), abs(hi_j)) + 1
    N, M = 2 * Ki + 1, 2 * Kj + 1
    i0 = np.floor(tpl[:, 0] / spec.cell_x).astype(np.int64) + 1
    i1 = np.ceil(tpl[:, 1] / spec.cell_x).astype(np.int64) - 1
    j0 = np.floor(tpl[:, 2] / spec.cell_y).astype(np.int64) + 1
    j1 = np.ceil(tpl[:, 3] / spec.cell_y).astype(np.int64) - 1
    i0 = np.clip(i0, -Ki + 1, Ki - 1)
    i1 = np.clip(i1, -Ki + 1, Ki - 1)
    j0 = np.clip(j0, -Kj + 1, Kj - 1)
    j1 = np.clip(j1, -Kj + 1, Kj - 1)
    keep = (i1 >= i0) & (j1 >= j0)
    if not keep.any():
        return np.zeros((N, M), dtype=bool), Ki, Kj
    i0, i1 = i0[keep] + Ki, i1[keep] + Ki
    j0, j1 = j0[keep] + Kj, j1[keep] + Kj
    W = M + 1                                          # 多一列给 j1+1 落脚
    flat = (np.bincount(i0 * W + j0, minlength=(N + 1) * W)
            + np.bincount((i1 + 1) * W + (j1 + 1), minlength=(N + 1) * W)
            - np.bincount(i0 * W + (j1 + 1), minlength=(N + 1) * W)
            - np.bincount((i1 + 1) * W + j0, minlength=(N + 1) * W))
    diff = flat.reshape(N + 1, W).astype(np.int64)
    return diff.cumsum(0).cumsum(1)[:N, :M] > 0, Ki, Kj


# --------------------------------------------------------------------------- 增量压缩

def compress_legal(model: PcbModel, comp_now: Footprint, k: int, i: int, j: int,
                   comp_next: Footprint, legal_next: np.ndarray,
                   spec: Optional[GridSpec] = None, out: Optional[np.ndarray] = None) -> np.ndarray:
    """**物化版增量压缩**（Owner 口径的第 2 步，字面实现）：

    ```text
    L_next(after) = L_next0 \\ (comp_now 放在 (k,i,j) 后"新覆盖的非法区")
    ```

    与 `legal_masks(model, placed ∪ {comp_now@(k,i,j)}, comp_next, spec)` 等价（待 `legal_masks`
    支持"已放件位姿覆盖"后可对拍，见模块末尾 ⚠️）。

    用途：① 掩膜判定（等价于 `mask` 的那条谓词）；② **env 下一步直接用**——真实落子后
    下一个件的 `L` 就是它 ⇒ 省掉一次 `legal_masks`。

    Args:
        legal_next: `L_next0`（**没放** `comp_now` 时下一个件的 L；本函数不修改它）。
        out: 可复用的输出缓冲（`legal_next` 的副本；缺省新建）。
    """
    spec = spec or grid_spec(model)
    m = spec.m
    X = spec.x0 + (np.arange(m) + 0.5) * spec.cell_x
    Y = spec.y0 + (np.arange(m) + 0.5) * spec.cell_y
    px, py = spec.mm_of(i, j)
    if out is None:
        res = legal_next.copy()
    else:
        np.copyto(out, legal_next)
        res = out
    for kf in range(4):
        if not legal_next[kf].any():
            continue
        for x1, x2, y1, y2 in killed_rects(model, comp_now, k, comp_next, kf, px, py):
            a0 = int(np.searchsorted(X, x1, "right"))          # 开区间：X > x1
            a1 = int(np.searchsorted(X, x2, "left"))           # 开区间：X < x2
            b0 = int(np.searchsorted(Y, y1, "right"))
            b1 = int(np.searchsorted(Y, y2, "left"))
            if a1 > a0 and b1 > b0:
                res[kf][a0:a1, b0:b1] = False
    return res


# --------------------------------------------------------------------------- 掩膜

def lookahead_mask(model: PcbModel, placed: Sequence[Footprint], comp_now: Footprint,
                   comp_next: Footprint, spec: Optional[GridSpec] = None,
                   legal: Optional[np.ndarray] = None,
                   legal_next: Optional[np.ndarray] = None,
                   keepouts=None,
                   ) -> Tuple[np.ndarray, Dict[str, object]]:
    """单步预测掩膜 → `(mask, info)`；`mask ⊆ L_now`（比 `L` 更严格）。

    Args:
        placed: 已放件（**不含** `comp_now`）。
        comp_now: 当前待放件；`comp_next`: 固定顺序里的**下一个**件。
        legal / legal_next: 已知的 `L_now` / `L_next0`（省一次计算）；缺省自行算。

    Returns:
        `(mask, info)`；`info` 含 `dead_next`（下一个件本来就无位置 ⇒ 全部 pass）、
        `kept` / `cut` / `n_cells_next` / `resolved_at`（在第几个 kf 解决的数量）。
    """
    spec = spec or grid_spec(model, m=GRID_M)
    m = spec.m
    if keepouts is None:                                  # 口径 (A)：未放置件的槽不上板（§0.6 p）
        keepouts = state_keepouts(model, placed)
    L = legal if legal is not None else legal_masks(model, placed, comp_now, spec, keepouts=keepouts)
    Ln = (legal_next if legal_next is not None
          else legal_masks(model, placed, comp_next, spec, keepouts=keepouts))

    mask = np.zeros_like(L)
    total = int(L.sum())
    n_next = int(Ln.sum())
    if n_next == 0:                                   # 边界：已死 ⇒ 全部 pass（保留 L 不变）
        return L.copy(), dict(dead_next=True, kept=total, cut=0,
                              n_cells_next=0, resolved_at=[0, 0, 0, 0])

    X = spec.x0 + (np.arange(m) + 0.5) * spec.cell_x
    Y = spec.y0 + (np.arange(m) + 0.5) * spec.cell_y
    ps = []
    for kf in range(4):
        s = np.zeros((m + 1, m + 1), dtype=np.int64)
        s[1:, 1:] = np.cumsum(np.cumsum(Ln[kf].astype(np.int64), 0), 1)
        ps.append(s)
    # ---- 一次性缓存两件的 pad/slot 相对几何（避免 16×(k,kf) 里重复解析） ----
    now_pads = [_pads(comp_now, k) for k in range(4)]
    now_slots = [_slots(comp_now, model, k) for k in range(4)]
    nxt_pads = [_pads(comp_next, kf) for kf in range(4)]
    nxt_slots = [_slots(comp_next, model, kf) for kf in range(4)]

    def _tpl(k: int, kf: int) -> np.ndarray:
        parts = [_rects(now_pads[k][0], now_pads[k][1], nxt_pads[kf][0], nxt_pads[kf][1], 0.0, 0.0),
                 _rects(now_slots[k][0], now_slots[k][1], nxt_pads[kf][0], nxt_pads[kf][1], 0.0, 0.0),
                 _rects(now_pads[k][0], now_pads[k][1], nxt_slots[kf][0], nxt_slots[kf][1], 0.0, 0.0)]
        parts = [x for x in parts if len(x)]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 4))

    tpl_cache: Dict[Tuple[int, int], np.ndarray] = {}
    tab_cache: Dict[Tuple[int, int], object] = {}
    budget = 4_000_000                                 # 每批查表元素上限（内存安全）
    resolved_at = [0, 0, 0, 0]
    for k in range(4):
        cells = np.argwhere(L[k])                      # (n, 2) 候选格索引
        if not len(cells):
            continue
        px, py = spec.mm_of(cells[:, 0].astype(float), cells[:, 1].astype(float))
        unresolved = np.ones(len(cells), dtype=bool)
        for kf in range(4):
            need = np.flatnonzero(unresolved)
            if not len(need):
                break
            if not Ln[kf].any():                       # 该旋转本来就没位置（不算过错）
                continue
            key = (k, kf)                              # 位移表**按需懒构建**（不必 16 张全建）
            if key not in tpl_cache:
                t = _tpl(k, kf)
                tpl_cache[key] = t
                tab_cache[key] = conflict_table_from(t, spec) if len(t) else None
            tpl = tpl_cache[key]
            K = int(Ln[kf].sum())
            if len(tpl) == 0:                          # 没有任何冲突矩形 ⇒ 整张 L_next 都活
                mask[k][cells[need, 0], cells[need, 1]] = True
                unresolved[need] = False
                resolved_at[kf] += len(need)
                continue
            # ---- ① 外接框快路径：被杀矩形并的外接框之外还有 L_next0 格 ⇒ 必然存活 ----
            bx1, bx2 = float(tpl[:, 0].min()), float(tpl[:, 1].max())
            by1, by2 = float(tpl[:, 2].min()), float(tpl[:, 3].max())
            i0 = np.clip(np.searchsorted(X, px[need] + bx1, "left"), 0, m)
            i1 = np.clip(np.searchsorted(X, px[need] + bx2, "right"), 0, m)
            j0 = np.clip(np.searchsorted(Y, py[need] + by1, "left"), 0, m)
            j1 = np.clip(np.searchsorted(Y, py[need] + by2, "right"), 0, m)
            s = ps[kf]
            inside = s[i1, j1] - s[i0, j1] - s[i1, j0] + s[i0, j0]
            ok = inside < K                            # 框外有格 ⇒ 存活（O(1)/候选）
            if ok.any():
                idx = need[ok]
                mask[k][cells[idx, 0], cells[idx, 1]] = True
                unresolved[idx] = False
                resolved_at[kf] += int(ok.sum())
            rest = need[~ok]
            if not len(rest):
                continue
            # ---- ② 精确：紧凑位移表批量查表（分批，控内存） ----
            T, Ki, Kj = tab_cache[key]
            cn = np.argwhere(Ln[kf])                   # (c, 2) 后件合法格索引
            step = max(1, budget // max(1, len(cn)))
            for a in range(0, len(rest), step):
                chunk = rest[a:a + step]
                # ⚠️ 索引约定：**(next − now)**（模板的 B 锚点坐标系，见 `conflict_table`）
                dI = cn[None, :, 0] - cells[chunk, 0][:, None]
                dJ = cn[None, :, 1] - cells[chunk, 1][:, None]
                killed = T[np.clip(dI, -Ki, Ki) + Ki, np.clip(dJ, -Kj, Kj) + Kj]
                alive = ~killed.all(axis=1)
                if alive.any():
                    idx = chunk[alive]
                    mask[k][cells[idx, 0], cells[idx, 1]] = True
                    unresolved[idx] = False
                    resolved_at[kf] += int(alive.sum())
    kept = int(mask.sum())
    return mask, dict(dead_next=False, kept=kept, cut=total - kept,
                      n_cells_next=n_next, resolved_at=resolved_at)
