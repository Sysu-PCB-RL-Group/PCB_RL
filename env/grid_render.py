"""栅格渲染：**解析式**生成 L（合法位置矩阵）与 G（增益矩阵）。

G 的口径（Owner 2026-09-17 定稿，λ = 0.7 : 0.3）：

```text
s_hpwl = (−ΔHPWL)/D_board                 D_board = sqrt(W²+H²)
s_disp = (σ_after − σ_before)/D_board     σ = 各非电源 net 的 bbox 中心的标准差（越大越好）
z_*    = clip((s_* − p5)/(p95 − p5), 0, 1)     ← 4 张表算完后**共用一把标尺**（表从不合并）
G      = 0.7·z_hpwl + 0.3·z_disp ∈ [0,1]（合法格）；非法格 G = −inf
```

**两趟式，禁止逐旋转归一化**：先把 4 张表的原始量**全部算完**，再在**同一个样本池**上取一次 p5/p95（"共用一把尺"，**4 张表始终独立存在、从不合并**）
（各旋转各自归一化会把每张表的最优都拉成 1.0 ⇒ 跨旋转不可比）。
某分量在池内恒定（`p95−p5 < eps`）时该分量给常数 0.5（防 0/0）；两分量**同时**退化时
G ≡ 0.5 且 `degenerate=True`（= "这一步的位置对 HPWL 与离散度都无差别"，实测占 23.6%）。
`degenerate` 只作元数据：**不进 G 张量、不进网络输入**。

不逐格判定（16 万次 Judge/步 = 29~232 秒），而是把三层违规归结为**锚点空间的矩形并集**：

```text
违规 ⟺ 硬重叠 ∨ 软间距 ∨ 出界
硬重叠 ∨ 软间距 ⟺ 同一对对象 gap < 0.2        （§3.1~§3.5 逐条核对：三类对象硬/软过滤条件一致；
                                              keepout 阈值本项目的口径亦取 0.2）
⇒ 每个 (pad, 障碍) 对只需戳 **1 个矩形**；出界只需 **1 个矩形**
```

过滤规则（逐条对应台账 §3.1~§3.5，**最易写错的是 net 过滤**）：
  * pad↔pad      ：同层 ∧ 不看 net ∧ 不同 footprint（当前件未放置 ⇒ 天然成立）
  * pad↔via      ：同层 ∧ **双方 netcode≠0 ∧ 不相等**
  * pad↔keepout  ：同层 ∧ 排除 `owner_ref == 本件` 的禁布区
  * 出界         ：pad 的 AABB 越出板框外接矩形（不管 net、不豁免）

几何：全部用 **pad AABB**（与 `env.violations.Judge` 同口径），在候选位姿下由
`板坐标 = 锚点 + R(−θ)·局部` 现算（θ 为 90° 倍数 ⇒ 精确）；映射按 Owner 拍案
"板框铺满 M×M"（`mode="fill"`，cell 非方形），另提供 `mode="uniform"`（等比缩放、cell 方形）。

与 `Judge` 的关系：本模块是判定式的**等价快速形式**；两者必须**抽样对拍 0 偏差**
（`verify_legal`）。
"""
from __future__ import annotations

import math
import os
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from env.violations import Pose, pose_of
from utils.kicad_utils.pcb_model import AABB, Footprint, PcbModel, rotate_aabb, state_keepouts

GRID_M = 200          # 画布边长（Owner 2026-09-16 拍案）
SOFT_GAP_MM = 0.2     # 软间距阈值（= 硬∪软的合并阈值）

# G 的组合权重与稳健 min-max 分位（Owner 2026-09-17 拍案）
G_W_HPWL, G_W_DISP = 0.7, 0.3
G_P_LO, G_P_HI = 5.0, 95.0
_DEGEN_EPS = 1e-12    # `p_hi − p_lo` 小于此 ⇒ 该分量在池内恒定（"退化"）

# 边界数值容差 = 1e-9 mm（= 1e-3 nm = 1 pm）：用于消除"恰好等于阈值"的浮点末位差。
# ⚠️ 它不是 1nm（1nm = 1e-6 mm）——它兜的是浮点误差，不是 nm 量化平局；与"精度下限 1nm"不冲突。
# 方向是**保守**的：禁区向外扩 `_EPS`、板框允许区向内收 `_EPS` ⇒
# **任何被 Judge 判为违规的位置，都一定被 L 判为禁止**（危险方向恒为 0，可证）。
# 代价：极少数"gap 恰等于 0.2"的位置会被保守地禁掉（实测 coloreasypico_D2 17 格 / 160000）。
_EPS = 1e-9
AUDIT_ENV = "PCB_RL_AUDIT"   # 与 project 其它审计开关一致：置 1 ⇒ "本该不可达"的静默回退改为抛错
MODE_FILL = "fill"        # 板框铺满 M×M（按轴缩放；Owner 拍案口径）
MODE_UNIFORM = "uniform"  # 等比缩放（cell 方形；可见性同、几何不失真）


class GridSpec(NamedTuple):
    """栅格映射：`i = floor((x−x0)/cell_x)`；格中心 `X(i) = x0 + (i+0.5)·cell_x`。"""

    x0: float
    y0: float
    cell_x: float
    cell_y: float
    m: int = GRID_M

    def centers_x(self) -> np.ndarray:
        return self.x0 + (np.arange(self.m, dtype=np.float64) + 0.5) * self.cell_x

    def centers_y(self) -> np.ndarray:
        return self.y0 + (np.arange(self.m, dtype=np.float64) + 0.5) * self.cell_y

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor((x - self.x0) / self.cell_x)), int(math.floor((y - self.y0) / self.cell_y))

    def mm_of(self, i: int, j: int) -> Tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell_x, self.y0 + (j + 0.5) * self.cell_y


def grid_spec(model: PcbModel, m: int = GRID_M, mode: str = MODE_FILL) -> GridSpec:
    """由板框生成映射（`fill` = 铺满，`uniform` = 等比）。"""
    x0, y0, x1, y1 = model.board_outline.bbox
    w, h = x1 - x0, y1 - y0
    if mode == MODE_UNIFORM:
        c = max(w, h) / m
        return GridSpec(x0, y0, c, c, m)
    return GridSpec(x0, y0, w / m, h / m, m)


# ---------------- 索引换算（格中心语义，严格按 Judge 的开闭区间） ----------------

def _idx_open(v_lo: float, v_hi: float, origin: float, cell: float, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """闭/开区间 → 满足 `v_lo < center < v_hi`（严格）的格索引区间 [i0, i1]（可空）。"""
    u_lo = (v_lo - origin) / cell - 0.5
    u_hi = (v_hi - origin) / cell - 0.5
    i0 = np.floor(u_lo).astype(np.int64) + 1
    i1 = np.ceil(u_hi).astype(np.int64) - 1
    return np.clip(i0, 0, m), np.clip(i1, -1, m - 1)


def _idx_closed(v_lo: float, v_hi: float, origin: float, cell: float, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """闭区间 → 满足 `v_lo ≤ center ≤ v_hi` 的格索引区间 [i0, i1]。"""
    u_lo = (v_lo - origin) / cell - 0.5
    u_hi = (v_hi - origin) / cell - 0.5
    i0 = np.ceil(u_lo).astype(np.int64)
    i1 = np.floor(u_hi).astype(np.int64)
    return np.clip(i0, 0, m), np.clip(i1, -1, m - 1)


def _add_rects(diff: np.ndarray, i0, i1, j0, j1, m: int) -> None:
    """把一批矩形 [i0,i1]×[j0,j1] 加到差分数组（越界已裁剪；空矩形自动丢弃）。"""
    i0, i1, j0, j1 = (np.atleast_1d(a) for a in (i0, i1, j0, j1))
    keep = (i1 >= i0) & (j1 >= j0)
    if not keep.any():
        return
    i0, i1, j0, j1 = i0[keep], i1[keep], j0[keep], j1[keep]
    np.add.at(diff, (i0, j0), 1)
    np.add.at(diff, (i1 + 1, j0), -1)
    np.add.at(diff, (i0, j1 + 1), -1)
    np.add.at(diff, (i1 + 1, j1 + 1), 1)


# ---------------- 障碍（按层分组；net 过滤在 pad 层做） ----------------

def _obstacles(placed: Sequence[Footprint], model: PcbModel, comp: Footprint, keepouts):
    """返回 (pad_obs, via_obs, keep_obs)，每项按层分组：{(0,): arr, (1,): arr, (0,1): arr}。

    每个元素是 `(aabb(K,4) 数组, netcode 数组或 None)`；keepout 用 owner 名单单独处理。
    """
    def _group(items):
        buckets = {0: [], 1: []}
        for aabb, layers, extra in items:
            for lay in layers:
                buckets[lay].append((aabb, extra))
        out = {}
        for lay, arrs in buckets.items():
            if not arrs:
                out[lay] = (np.zeros((0, 4)), np.zeros(0, dtype=np.int64))
                continue
            out[lay] = (np.array([a for a, _ in arrs], dtype=np.float64),
                        np.array([e for _, e in arrs], dtype=np.int64))
        return out

    pads = [(p.aabb, p.layers, p.netcode) for f in placed for p in f.pads]
    vias = [(v.aabb, v.layers, v.netcode) for v in model.vias]
    ks = keepouts          # **必传**（口径 A 的 `state_keepouts` 结果）——不留 `model.keepouts` 兜底：
    #                        否则"源位姿槽"会从这条私有后门悄悄回来（§0.6 q-④）
    keeps = [(k.aabb, k.layers, i) for i, k in enumerate(ks) if k.owner_ref != comp.ref]
    return _group(pads), _group(vias), keeps, _group(keeps)


# ---------------- L ----------------

def legal_masks(model: PcbModel, placed: Sequence[Footprint], comp: Footprint,
                spec: Optional[GridSpec] = None, soft: float = SOFT_GAP_MM,
                keepouts=None) -> np.ndarray:
    """`L[k][i][j]`：把 `comp` 放到格中心 `(i,j)`、旋转 `k` 是否合法（True = 合法）。

    Args:
        placed: 已放置的 footprint（**不含** comp）。⚠️ **必须传"位姿更新后的对象"**：
            对象的 `pad.aabb`/`pad.center` 是导入时按**源位姿**算的，策略落子后用
            `pcb_model.place_at(fp, anchor, rot)` 生成克隆再传进来（台账 §0.6 o）。
        keepouts: 覆盖 `model.keepouts`。**缺省（推荐）= `pcb_model.state_keepouts(model, placed)`**
            （口径 A，唯一入口：槽随已放件、未放置件的槽不在板上，§0.6 p）；
            显式传入时由调用方保证**与 `placed` 同帧**（否则就是"过期几何"，§4.2-6）。
    """
    spec = spec or grid_spec(model)
    m = spec.m
    # 口径 (A)（§0.6 p）：未给 keepouts 时按"当前局面"推导 —— 未放置件的槽不在板上
    ks = keepouts if keepouts is not None else state_keepouts(model, placed)
    pad_obs, via_obs, _keeps, keep_by_layer = _obstacles(placed, model, comp, ks)
    va = model.board_outline.valid_area(0.0)
    out = np.zeros((4, m, m), dtype=bool)

    for k in range(4):
        diff = np.zeros((m + 1, m + 1), dtype=np.int32)
        # ① 出界：允许锚点区 = 所有 pad 的 [X1−px1, X2−px2] ∩ [Y1−py1, Y2−py2]（闭区间）
        lo_x, hi_x = -np.inf, np.inf
        lo_y, hi_y = -np.inf, np.inf
        for pad in comp.pads:
            if not pad.local_aabb:
                continue
            a = rotate_aabb(pad.local_aabb, k * 90.0, (0.0, 0.0))
            cx, hx = (a[0] + a[2]) / 2.0, (a[2] - a[0]) / 2.0
            cy, hy = (a[1] + a[3]) / 2.0, (a[3] - a[1]) / 2.0
            # 同一顺序：① 物体空间把板框**内缩** pad 半尺寸  ② 倒推锚点（平移 −(cx,cy)）
            # `_EPS`：允许区再向内收一点（保守 ⇒ 贴边平局判为出界）
            lo_x = max(lo_x, va[0] + hx - cx + _EPS); hi_x = min(hi_x, va[2] - hx - cx - _EPS)
            lo_y = max(lo_y, va[1] + hy - cy + _EPS); hi_y = min(hi_y, va[3] - hy - cy - _EPS)
        inb = np.zeros((m, m), dtype=bool)
        if np.isfinite(lo_x) and np.isfinite(lo_y):
            i0, i1 = _idx_closed(lo_x, hi_x, spec.x0, spec.cell_x, m)
            j0, j1 = _idx_closed(lo_y, hi_y, spec.y0, spec.cell_y, m)
            if i1 >= i0 and j1 >= j0:
                inb[i0:i1 + 1, j0:j1 + 1] = True

        # ②③④ 三类障碍：每个 (pad, 障碍) 一对 ⇒ 1 个矩形（gap < soft）
        for pad in comp.pads:
            if not pad.local_aabb:
                continue
            px1, py1, px2, py2 = rotate_aabb(pad.local_aabb, k * 90.0, (0.0, 0.0))
            cx, hx = (px1 + px2) / 2.0, (px2 - px1) / 2.0
            cy, hy = (py1 + py2) / 2.0, (py2 - py1) / 2.0
            layers = pad.layers
            for lay in layers:
                # pad↔pad（不看 net）
                B, _ = pad_obs[lay]
                if len(B):
                    _add_rects(diff, *_rect_idx(B, cx, hx, cy, hy, soft, spec), m)
                # pad↔via（双方 net≠0 ∧ 不等）
                B, code = via_obs[lay]
                if len(B) and pad.netcode != 0:
                    sel = (code != 0) & (code != pad.netcode)
                    if sel.any():
                        _add_rects(diff, *_rect_idx(B[sel], cx, hx, cy, hy, soft, spec), m)
                # pad↔keepout（已排除 owner 为本件）
                Bk, _ = keep_by_layer[lay]
                if len(Bk):
                    _add_rects(diff, *_rect_idx(Bk, cx, hx, cy, hy, soft, spec), m)

        forbidden = diff.cumsum(0).cumsum(1)[:m, :m] > 0
        out[k] = inb & ~forbidden
    return out


def _rect_idx(B: np.ndarray, cx: float, hx: float, cy: float, hy: float,
              soft: float, spec: GridSpec):
    """一批障碍 AABB → 禁忌锚点矩形的格索引区间（`gap < soft`，严格）。

    **三段顺序（Owner 2026-09-17 定式）**：
        ① 物体空间：障碍 ⊕ pad 半尺寸      → [bx1−hx, bx2+hx]
        ② 物体空间：再加 0.2 软间距环      → [bx1−hx−s, bx2+hx+s]
        ③ 倒推锚点：平移 −(cx, cy)         → [bx1−hx−s−cx, bx2+hx+s−cx]
    与"把 −cx、−hx、+s 混在一个式子里"数学恒等（Minkowski 和可交换可结合），
    但本顺序物理语义清晰（0.2 属于**形状之间**的间隙，不是锚点的属性）、
    实现上不易写错符号，且与"出界（先内缩板框、再倒推）"处理顺序一致。
    **若将来改用真实形状**（圆 via / 非正交件），必须保持本顺序并在**最后一步**才取 AABB
    （AABB 近似不满足交换律；先取近似再膨胀会偏小 = 危险方向）。
    """
    # ① 障碍 ⊕ pad 半尺寸
    gx1, gy1 = B[:, 0] - hx, B[:, 1] - hy
    gx2, gy2 = B[:, 2] + hx, B[:, 3] + hy
    # ② 再加 0.2 软间距环（+ `_EPS`：保守外扩，见模块顶部说明）
    gx1, gy1 = gx1 - soft - _EPS, gy1 - soft - _EPS
    gx2, gy2 = gx2 + soft + _EPS, gy2 + soft + _EPS
    # ③ 倒推锚点（平移 −(cx, cy)）
    lo_x, hi_x = gx1 - cx, gx2 - cx
    lo_y, hi_y = gy1 - cy, gy2 - cy
    i0, i1 = _idx_open(lo_x, hi_x, spec.x0, spec.cell_x, spec.m)
    j0, j1 = _idx_open(lo_y, hi_y, spec.y0, spec.cell_y, spec.m)
    return i0, i1, j0, j1


# ---------------- G ----------------

def hpwl_gain_maps(model: PcbModel, placed: Sequence[Footprint], comp: Footprint,
                   spec: Optional[GridSpec] = None) -> np.ndarray:
    """`H[k][i][j] = −ΔHPWL`（放该件后 HPWL 的减少量；越高越好）。

    ⚠️ 这是 **G 的 HPWL 分量**（原始量：未除对角线、未归一化、未加权）；完整 G 见 `gain_maps`。

    口径与台账 §7.1（HPWL）+ §2.5（电源网/连接点）一致：**排除电源/GND**、连接点 = pad 中心、不分层、未放置件不计。
    逐 net 解析：新 pad 的极值相对旧 bbox 的 `hinge` 之和。
    """
    spec = spec or grid_spec(model)
    X, Y = spec.centers_x(), spec.centers_y()
    placed_refs = {f.ref for f in placed}
    # ⚠️ 已放件的 pad 中心必须取【位姿化后】的对象：`net.pads[].center` 是**源位姿快照**
    pose_map = _pose_pad_index(model, placed)
    out = np.zeros((4, spec.m, spec.m), dtype=np.float32)
    for k in range(4):
        d = np.zeros((spec.m, spec.m), dtype=np.float64)
        const = 0.0                            # 与锚点无关的常量（"自 net"的内部跨度）
        for net in model.nets.values():
            if net.is_power:
                continue
            mine = [p for p in net.pads if p.ref == comp.ref and p.local_aabb]
            if not mine:
                continue
            others = [p for p in net.pads if p.ref in placed_refs]
            if not others:
                # 该 net 上还没有已放 pad：放上后若本件在该 net 有 ≥2 pad，
                # 该 net 由"无点（跳过）"变成"有跨度" ⇒ ΔHPWL 含一个常量（与锚点无关）
                if len(mine) >= 2:
                    cs = []
                    for p in mine:
                        a = rotate_aabb(p.local_aabb, k * 90.0, (0.0, 0.0))
                        cs.append(((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0))
                    xs = [c[0] for c in cs]; ys = [c[1] for c in cs]
                    const += (max(xs) - min(xs)) + (max(ys) - min(ys))
                continue
            oc = [_center_now(pose_map, p) for p in others]
            ox1 = min(c[0] for c in oc); ox2 = max(c[0] for c in oc)
            oy1 = min(c[1] for c in oc); oy2 = max(c[1] for c in oc)
            cxs, cys = [], []
            for p in mine:
                a = rotate_aabb(p.local_aabb, k * 90.0, (0.0, 0.0))
                cxs.append((a[0] + a[2]) / 2.0)
                cys.append((a[1] + a[3]) / 2.0)
            mx, mn, my, mny = max(cxs), min(cxs), max(cys), min(cys)
            dx = np.maximum(0.0, (X + mx) - ox2) + np.maximum(0.0, ox1 - (X + mn))
            dy = np.maximum(0.0, (Y + my) - oy2) + np.maximum(0.0, oy1 - (Y + mny))
            d += dx[:, None] + dy[None, :]
        out[k] = (-(d + const)).astype(np.float32)
    return out


# ---------------- G 的另一个分量：跨网络离散度 ----------------

def _pose_pad_index(model: PcbModel, placed: Sequence[Footprint]) -> dict:
    """`{ref: (位姿化后的 pads, {id(模型 pad): 下标})}`。

    用途：`net.pads` 里的 pad 是**模型对象**（`center` 是导入时源位姿快照）；要拿"当前位姿"
    的中心，必须映射到调用方传入的**位姿化对象**（`pcb_model.place_at` 保序 ⇒ 按下标对应）。

    ⚠️ **契约硬校验（2026-09-18，§0.6 q-⑤）**：本函数是"按下标取克隆 pad"的**唯一桥梁** ——
    一旦"克隆的 pad 序列 ≠ 模型源 pad 序列"，`_center_now` 会**静默取到错中心**
    （HPWL 与离散度都错、且不报错）⇒ 这里**硬失败**（这三条都是调用方契约，不是数据问题）：
      * `ref` 不在 `model.footprints` 里 ⇒ 抛（外来对象）；
      * `len(f.pads) != len(src.pads)` ⇒ 抛（下标会错位）；
      * 同下标 pad 的 `local_aabb` / `rel_offset`（**都与位姿无关**）不等 ⇒ 抛（同长度但不同 pad）。
    """
    byref = {f.ref: f for f in model.footprints}
    out = {}
    for f in placed:
        src = byref.get(f.ref)
        if src is None:
            raise ValueError(
                f"_pose_pad_index: ref {f.ref!r} 不在该模型里（外来对象？）——取点要求 `placed` 是"
                "【本模型 footprint 的位姿化克隆】（`pcb_model.place_at` 的产物）")
        if len(f.pads) != len(src.pads):
            raise ValueError(
                f"_pose_pad_index: {f.ref} 的 pad 数 {len(f.pads)} ≠ 模型源 {len(src.pads)} ——"
                "按下标映射会取到【错中心】（G 静默出错）；请用 `pcb_model.place_at` 生成克隆")
        for i, (a, b) in enumerate(zip(f.pads, src.pads)):
            if a.local_aabb != b.local_aabb or a.rel_offset != b.rel_offset:
                raise ValueError(
                    f"_pose_pad_index: {f.ref} 第 {i} 个 pad 与模型源不是同一个"
                    "（local_aabb / rel_offset 不等）—— 克隆必须来自 `pcb_model.place_at`（保序）")
        out[f.ref] = (f.pads, {id(q): i for i, q in enumerate(src.pads)})
    return out


def _center_now(pose_map: dict, p) -> Tuple[float, float]:
    """模型里的 pad `p` → **位姿化后**的中心。

    ⚠️ 映射不上时**回退 `p.center`（= 源位姿快照，即专家布局位置）**：正常情况下**不可达**
    （`_pose_pad_index` 已硬校验"克隆 == 源，同序"）；但一旦发生，G 会静默用专家位置
    ⇒ 置 `PCB_RL_AUDIT=1` 时改为**抛错**（§0.6 q-⑤）。
    """
    ent = pose_map.get(p.ref)
    if ent is not None:
        pads, idx = ent
        i = idx.get(id(p))
        if i is not None and i < len(pads):
            return pads[i].center
    if os.environ.get(AUDIT_ENV) == "1":
        raise ValueError(
            f"_center_now: 模型 pad {p.ref}/{p.number} 找不到位姿化映射 ——"
            "G 会退回【源位姿中心】（专家位置）⇒ 结果静默出错")
    return p.center


def _net_state(model: PcbModel, refs, pose_map: Optional[dict] = None) -> Tuple[dict, dict]:
    """→ `(cnt, rect)`：每个**非电源** net 的已放 pad 数 + bbox（无已放 pad 则不建 rect）。

    `pose_map`：来自 `_pose_pad_index` —— 已放件的 pad 中心按**当前位姿**取（否则用源位姿）。
    """
    cnt, rect = {}, {}
    pose_map = pose_map or {}
    for net in model.nets.values():
        if net.is_power:
            continue
        xs, ys = [], []
        for p in net.pads:
            if p.ref in refs:
                cx, cy = _center_now(pose_map, p)
                xs.append(cx)
                ys.append(cy)
        cnt[net.net_id] = len(xs)
        if xs:
            rect[net.net_id] = (min(xs), max(xs), min(ys), max(ys))
    return cnt, rect


def _sigma_of_centers(pts) -> float:
    """`σ = sqrt(mean|c|² − |mean c|²)`（net 的 bbox 中心的总体标准差）。"""
    n = len(pts)
    if n == 0:
        return 0.0
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    return math.sqrt(max(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / n, 0.0))


def dispersion_delta(model: PcbModel, placed: Sequence[Footprint], comp: Footprint,
                     spec: GridSpec, legal: np.ndarray) -> np.ndarray:
    """`Δσ[k][i][j] = σ_after − σ_before`（**G 的跨网络离散度分量**；未除对角线）。

    口径（与 HPWL 分量同源）：排除电源/GND、net 连接点 = pad 中心、net 中心 = **bbox 中心**、
    不分层；参与 net = "**放上当前件之后**该 net ≥2 个点"（含"1 个已放 + 本件 1 个"）。
    只在 `legal[k]` 的格上计算（其余格填 0，调用方不应读取）。

    实现：只对**被本件触及**的 net 逐格算中心（未触及的 net 贡献常量），
    故成本 O(触及 net 数 × 合法格数)，与"全部 net 数 × 格数"相比省一个量级。
    """
    refs = {f.ref for f in placed}
    cnt, rect = _net_state(model, refs, _pose_pad_index(model, placed))
    sig_before = _sigma_of_centers([((r[0] + r[1]) / 2.0, (r[2] + r[3]) / 2.0)
                                    for n, r in rect.items() if cnt[n] >= 2])

    out = np.zeros((4, spec.m, spec.m), dtype=np.float64)
    for k in range(4):
        mine = {}                                   # net_id -> [(cx, cy)]（本件在该 net 上的 pad 中心偏移）
        for p in comp.pads:
            if not p.local_aabb or p.net_id is None:
                continue
            net = model.nets.get(p.net_id)
            if net is None or net.is_power:          # 与 HPWL 同口径排除电源/GND
                continue
            a = rotate_aabb(p.local_aabb, k * 90.0, (0.0, 0.0))
            mine.setdefault(p.net_id, []).append(((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0))

        c = np.argwhere(legal[k])
        if not len(c):
            continue
        # 未触及且已活跃的 net：常量中心；被触及且放上后 ≥2 点：逐格算中心
        const_ctrs = [((rect[n][0] + rect[n][1]) / 2.0, (rect[n][2] + rect[n][3]) / 2.0)
                      for n in rect if cnt[n] >= 2 and n not in mine]
        touched = [n for n in mine if cnt.get(n, 0) + len(mine[n]) >= 2]
        n_after = len(const_ctrs) + len(touched)
        if n_after == 0:                             # σ_before 也必为 0 ⇒ Δσ ≡ 0
            continue

        X = spec.x0 + (c[:, 0] + 0.5) * spec.cell_x
        Y = spec.y0 + (c[:, 1] + 0.5) * spec.cell_y
        sx = np.full(len(X), sum(q[0] for q in const_ctrs))
        sy = np.full(len(X), sum(q[1] for q in const_ctrs))
        ss = np.full(len(X), sum(q[0] ** 2 + q[1] ** 2 for q in const_ctrs))
        for n in touched:
            off = np.asarray(mine[n], dtype=np.float64)
            px = X[:, None] + off[None, :, 0]
            py = Y[:, None] + off[None, :, 1]
            if n in rect:                            # net 已有已放点（含"仅 1 点"的退化矩形）
                x1, x2, y1, y2 = rect[n]
                bx1 = np.minimum(px, x1).min(axis=1); bx2 = np.maximum(px, x2).max(axis=1)
                by1 = np.minimum(py, y1).min(axis=1); by2 = np.maximum(py, y2).max(axis=1)
            else:                                    # 全新 net（本件自身 ≥2 pad）
                bx1 = px.min(axis=1); bx2 = px.max(axis=1)
                by1 = py.min(axis=1); by2 = py.max(axis=1)
            cxn = (bx1 + bx2) / 2.0
            cyn = (by1 + by2) / 2.0
            sx += cxn; sy += cyn; ss += cxn * cxn + cyn * cyn
        cbx, cby = sx / n_after, sy / n_after
        sig = np.sqrt(np.maximum(ss / n_after - (cbx * cbx + cby * cby), 0.0))
        out[k][c[:, 0], c[:, 1]] = sig - sig_before
    return out


# ---------------- G（组合层） ----------------

def gain_maps(model: PcbModel, placed: Sequence[Footprint], comp: Footprint,
              spec: Optional[GridSpec] = None, legal: Optional[np.ndarray] = None,
              w: Tuple[float, float] = (G_W_HPWL, G_W_DISP),
              p_lo: float = G_P_LO, p_hi: float = G_P_HI,
              eps: float = _DEGEN_EPS, keepouts=None) -> Tuple[np.ndarray, bool]:
    """`G[k][i][j] = w0·z_hpwl + w1·z_disp`（合法格；**非法格 = −inf**）+ 退化标志。

    **两趟式（Owner 要求，禁止逐旋转归一化）**：
      趟 1：`hpwl_gain_maps` 与 `dispersion_delta` 把 **4 个旋转的原始量全部算出**；
            与 `legal` 的 4 张表一并投进**同一个样本池**（池 = 4 旋转合法格的并集；**表从不合并**）。
      趟 2：池上各分量取一次 `p_lo/p_hi` 分位（**共用同一把标尺**），映射到 [0,1] 后加权。

    Args:
        legal: 已知的 L（省一次 `legal_masks`）；缺省自行计算（G 与 L 必须**同一帧**）。
        w: `(w_hpwl, w_disp)`，缺省 `(0.7, 0.3)`。
        p_lo/p_hi: 稳健 min-max 的分位（缺省 5/95）。
        eps: `p_hi − p_lo < eps` ⇒ 该分量在池内恒定 ⇒ 该分量给常数 0.5（防 0/0，见模块说明）。

    Returns:
        `(G, degenerate)`：`degenerate=True` 表示**两分量同时**退化（G ≡ 0.5，本步 G 无信息）。
        无合法格（死锁）时返回 `(全 −inf, True)`；`degenerate` 只作元数据，不进 G/不进网络输入。
    """
    spec = spec or grid_spec(model)
    m = spec.m
    L = legal if legal is not None else legal_masks(model, placed, comp, spec, keepouts=keepouts)
    H = hpwl_gain_maps(model, placed, comp, spec)
    disp = dispersion_delta(model, placed, comp, spec, L)

    ob = model.board_outline.bbox
    D = math.hypot(ob[2] - ob[0], ob[3] - ob[1])

    # —— 趟 1：4 个旋转的原始量全部取出（先算完，不逐旋转归一化） ——
    parts = []                                    # [(k, cells_k, s_hpwl_k, s_disp_k)]
    for k in range(4):
        c = np.argwhere(L[k])
        if not len(c):
            continue
        parts.append((k, c,
                      H[k][c[:, 0], c[:, 1]].astype(np.float64) / D,
                      disp[k][c[:, 0], c[:, 1]].astype(np.float64) / D))
    G = np.full((4, m, m), -np.inf, dtype=np.float32)
    if not parts:                                 # 死锁：无合法格
        return G, True

    # —— 趟 2：每个分量各用一次共用标尺（4 旋转共用同一池的分位） ——
    pool_h = np.concatenate([p[2] for p in parts])
    pool_d = np.concatenate([p[3] for p in parts])
    lo_h, hi_h = np.percentile(pool_h, p_lo), np.percentile(pool_h, p_hi)
    lo_d, hi_d = np.percentile(pool_d, p_lo), np.percentile(pool_d, p_hi)
    degen_h = bool(hi_h - lo_h < eps)
    degen_d = bool(hi_d - lo_d < eps)

    w0, w1 = w
    for k, c, s_h, s_d in parts:
        z_h = np.full_like(s_h, 0.5) if degen_h else np.clip((s_h - lo_h) / (hi_h - lo_h), 0.0, 1.0)
        z_d = np.full_like(s_d, 0.5) if degen_d else np.clip((s_d - lo_d) / (hi_d - lo_d), 0.0, 1.0)
        G[k][c[:, 0], c[:, 1]] = (w0 * z_h + w1 * z_d).astype(np.float32)
    return G, bool(degen_h and degen_d)


# ---------------- 对拍验收 ----------------

class LegalVerification(NamedTuple):
    """`verify_legal` 的结果（按**方向**分开统计）。"""

    checked: int
    danger: int                   # L 说可放但 Judge 判违规（必须为 0）
    over: int                     # L 说禁放但 Judge 判合法（应全部是"阈值平局"）
    danger_examples: list
    over_cells: list              # [(k, i, j)]，便于调用方逐个复核"是否平局"


def verify_legal(model: PcbModel, placed: Sequence[Footprint], comp: Footprint,
                 spec: Optional[GridSpec] = None, n: int = 400, seed: int = 0,
                 soft: float = SOFT_GAP_MM, keepouts=None) -> LegalVerification:
    """抽样对拍：`L` vs `Judge.judge(格中心)`（按**方向**分别统计）。

    ⚠️ **必须传"位姿化后的对象"**（`pcb_model.place_at` 的产物）+ 位姿化开槽：
    `L` 与 `Judge` 的 `others` 都由**同一批对象**推导 ⇒ 若对象是源位姿快照，**两边同源同错**
    ⇒ 对拍会给出"0 偏差"的**假绿灯**（台账 §0.6 o）。

    抽样 = 均匀随机 + **边界格**（L 的 4 邻域存在翻转的格，最容易暴露开闭区间写错）。
    """
    from env.violations import Judge

    spec = spec or grid_spec(model)
    m = spec.m
    ks = keepouts if keepouts is not None else state_keepouts(model, placed)
    L = legal_masks(model, placed, comp, spec, soft, keepouts=ks)
    judge = Judge(model, soft_gap=soft, keepouts=ks)
    others: Mapping[str, Pose] = {f.ref: pose_of(f) for f in placed}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, 2))
    edge = np.zeros((m, m), dtype=bool)
    for axis in (1, 2):                        # L 形状 (4, m, m)
        edge |= (np.roll(L, 1, axis=axis) != L).any(axis=0)
    ei, ej = np.nonzero(edge)
    if len(ei):
        pick = rng.integers(0, len(ei), size=(n,))
        idx = np.concatenate([idx, np.stack([ei[pick], ej[pick]], axis=1)], axis=0)

    danger_examples, over_cells = [], []
    n_danger = checked = 0
    for i, j in idx:
        i, j = int(i), int(j)
        x, y = spec.mm_of(i, j)
        for k in range(4):
            checked += 1
            res = judge.judge(comp.ref, Pose(x, y, k), others)
            viol = bool(res.is_violation)
            legal_by_L = bool(L[k][i][j])
            if legal_by_L and viol:                        # 危险方向
                n_danger += 1
                if len(danger_examples) < 12:
                    danger_examples.append((i, j, k, res.describe(), round(x, 4), round(y, 4)))
            elif (not legal_by_L) and (not viol):          # 过度封锁
                over_cells.append((k, i, j))
    return LegalVerification(checked, n_danger, len(over_cells), danger_examples, over_cells)
