# -*- coding: utf-8 -*-
"""布局对比可视化：**专家（源位姿） vs 模型推理布局** —— 每板一张 2×2 PNG。

为什么新写一个（不复用上一代的两个文件）
------------------------------------
* `utils/visualization.py::draw_layout` 顶层 `import env.pcb_layout_env` ⇒ 会拉起**上一代 KiCad env 链**；
  而且它的输入是旧 `Component`（`.bbox/.is_fixed/.nets`），新栈的几何在 `PcbModel`（`Footprint.hull_aabb`、
  `Pad.aabb`…）⇒ 字段对不上。
* `utils/visualize_expert_layouts.py` 直接 `import pcbnew`（要求 `KICAD_PATH`）⇒ 无 KiCad 的环境跑不了。
* 本模块**只依赖 matplotlib + `PcbModel`**（`pcb_model` 对 KiCad 是**软依赖**：`KICAD_PATH` 未设只打 `[WARN]`）
  ⇒ 与 P1"训练/推理不依赖 KiCad"同一条纪律：**哪台机器都能出图**。

图的样子（Owner 2026-09-23 拍：**2×2 并排**、**画 HPWL 包围框**、不画 net 连线、不做逐帧）
```text
                  顶层 (F.Cu)                     底层 (B.Cu)
上排       专家（源位姿）                  专家（源位姿）
下排       模型推理布局（mask = L）         模型推理布局（mask = L）
```
每格画的层级（**从下往上**）：板框底色（浅米白 `#FAFAF7`）｜网格（次刻度 0.1mm / 主刻度 1~2mm）
｜禁布区/开槽（红虚线）｜过孔（红方框）｜元件本体矩形（**浅蓝 `#BBDEFB` 填充 + `#1E88E5` 描边**）
｜焊盘（**浅绿 `#C8E6C9`** = 本层 SMD，空心 = 贯穿 PTH）｜元件 ref 文字（黑）｜板框（`#333333`）
｜**（可选）HPWL 包围框 `#FFD54F`**：**默认不画**，只有 `draw_compare(show_net_boxes=True)` 才画（那时在最上层，
图例与它联动）。
配色照 `utils/visualize_expert_layouts.py`（生成 `data/kicad-native-batch-vision/` 那套）：
**填充用浅色 + alpha 只作用在填充上**（本体 0.45 / 焊盘 0.9），描边保持不透明 ⇒ 色块淡、轮廓清楚
（Owner 2026-09-23：上一版填充做成实心深色"太深、难看"，已回退；HPWL 框改成**可选项、默认不画**）。
坐标轴 mm、**Y 轴向下**（KiCad 方向）、等比例。
⚠️ 包围框（**仅 `show_net_boxes=True` 时画**）与 `PcbModel.hpwl()` **同源同口径**（非电源 net、连接点 = pad 中心、
   **不分层**、net 点数 < 2 不算）⇒ ① 框周长的一半之和 = 该排布局的 HPWL；② 顶层/底层两格画的是**同一批**框。
⚠️ 与上一代的一处差异：新栈**全部 footprint 都参与放置**（`plan_order(comps=None)`）⇒
图例不再区分"可移动/固定"，只有一种"元件本体"。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")                      # 无显示环境也能出图
import matplotlib.colors as mcolors        # noqa: E402（填充色的 alpha 只作用在填充上）
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402
from matplotlib.ticker import MultipleLocator   # noqa: E402

from utils.kicad_utils import pcb_model as pm   # noqa: E402

__all__ = ["draw_compare", "expert_poses_of", "net_boxes_of", "setup_chinese_font"]

# ---- 配色（Owner 2026-09-23：参照 `utils/visualize_expert_layouts.py`（生成 `data/kicad-native-batch-vision/`
#      那 20 张图的那套）—— **浅色填充 + 只作用于填充的 alpha**，描边保持清晰；
#      上一版把填充做成实心深色 ⇒ 太深、难看，已回退。） ----
C_BODY = "#BBDEFB"            # 元件本体填充（浅蓝，同 vision）
C_BODY_A = 0.45               # ⚠️ 只作用于**填充**（不传 patch.alpha，否则描边一起变淡）
C_BODY_E = "#1E88E5"          # 本体描边（清晰）
C_PAD = "#C8E6C9"             # 本层 SMD 焊盘填充（浅绿，同 vision）
C_PAD_A = 0.9
C_PAD_E = "#43A047"           # 焊盘描边
C_KEEP = "#E53935"
C_VIA = "#E53935"
C_BOARD = "#333333"           # 板框描边（同 vision）
C_BOARD_BG = "#FAFAF7"        # 板框底色（同 vision：板内浅米白）
C_NETBOX = "#FFD54F"          # 浅黄：HPWL 的 net 包围框（Owner 2026-09-23；**画在最上层**）
C_GRID_M = "#D5D5CC"
C_GRID_S = "#EEEEE8"
C_LABEL = "#000000"           # 元件编号（黑细字，同 vision）

_LEGEND = (
    ("元件本体（全部件都参与放置）", C_BODY, C_BODY_E, True),
    ("焊盘（本层 SMD）", C_PAD, C_PAD_E, True),
    ("焊盘（贯穿 PTH）", "none", C_PAD_E, False),
    ("禁布区 / 开槽", "none", C_KEEP, False),
    ("过孔", "none", C_VIA, False),
    ("HPWL 包围框（非电源 net）", "none", C_NETBOX, False),
    ("板框", "none", C_BOARD, False),
)


def setup_chinese_font() -> None:
    """中文字体（与上一代同名单；缺字体就退化成默认字体，不报错）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


# ===================== 位姿表 =====================

def expert_poses_of(model: pm.PcbModel) -> Dict[str, Tuple[float, float, int]]:
    """专家/源位姿表 `ref → (x, y, rot)`（= 导入时的快照，**不需要 KiCad**）。"""
    return {f.ref: (float(f.anchor[0]), float(f.anchor[1]), int(f.rotation_index))
            for f in model.footprints}


def net_boxes_of(model: pm.PcbModel, poses: Mapping[str, Tuple[float, float, int]], *,
                 exclude_power: bool = True, min_points: int = 2) -> Dict[str, Tuple[float, float, float, float]]:
    """`{net_id: (x0, y0, x1, y1)}` —— **HPWL 的框**（Owner 2026-09-23：可视化里画出来）。

    与 `PcbModel.hpwl()` **同源同口径**（两边都走 `Net.connection_points`）：
      · 排除电源/地网（`exclude_power=True`，与 `hpwl` 默认一致）｜连接点 = pad 中心｜**不分层**
      · `len(点数) < min_points` 的 net 不算（`hpwl` 同样 `continue`）
      · **只取 `poses` 里的 ref**（未放置的件不能按源位姿混进来 —— 否则框会画在专家位置上）

    性质（测试守门）：对上这一批框，`Σ[(x1−x0) + (y1−y0)] == model.hpwl(refs=set(poses), poses=poses)`。
    """
    refs = set(poses)
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for net in model.nets.values():
        if exclude_power and net.is_power:
            continue
        pts = net.connection_points(refs, poses)
        if len(pts) < int(min_points):
            continue
        xs = [p[2] for p in pts]
        ys = [p[3] for p in pts]
        out[net.net_id] = (min(xs), min(ys), max(xs), max(ys))
    return out


def _tick_steps(w: float, h: float) -> Tuple[float, float]:
    """主/次刻度（mm）：主刻度按板尺寸走 1-2-5，次刻度 = 主刻度/5（上限 0.1mm 的说法与上一代一致）。"""
    span = max(w, h)
    major = 1.0
    for cand in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        major = cand
        if span / cand <= 14.0:
            break
    return major, major / 5.0


# ===================== 单格绘制 =====================

def _rect(ax, aabb: Sequence[float], *, face: str = "none", edge: str = C_BOARD,
          lw: float = 1.0, ls: str = "-", zorder: float = 2.0,
          face_alpha: Optional[float] = None) -> None:
    """画一个矩形。

    `face_alpha` 只作用在**填充**（`facecolor=mcolors.to_rgba(face, α)`），**不动描边** ——
    这是 vision 那套图的做法：色块淡、轮廓清楚（若改用 `patch.alpha` 会把描边一起变淡）。
    """
    x0, y0, x1, y1 = (float(v) for v in aabb)
    if x1 <= x0 or y1 <= y0:
        return
    fc: Any = face
    if face_alpha is not None and face != "none":
        fc = mcolors.to_rgba(face, float(face_alpha))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor=edge,
                           linewidth=lw, linestyle=ls, zorder=zorder))


def _draw_panel(ax, model: pm.PcbModel, poses: Mapping[str, Tuple[float, float, int]],
                keepouts: Optional[Sequence[pm.KeepoutRegion]], layer: int, *,
                panel_title: str, show_pads: bool, show_keepouts: bool, show_vias: bool,
                label_refs: bool, show_net_boxes: bool = False) -> Dict[str, int]:
    """画一格（某一层 + 某一套位姿）；返回画了多少个元素（进 summary 的诊断）。"""
    by_ref = {f.ref: f for f in model.footprints}
    bb = model.board_outline.bbox
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    major, minor = _tick_steps(w, h)

    # 网格（先把背景网格铺上，元素都画在它上面）
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(which="major", color=C_GRID_M, linewidth=0.6, zorder=0)
    ax.grid(which="minor", color=C_GRID_S, linewidth=0.35, zorder=0)

    n_body = n_pad = n_via = n_keep = n_box = 0

    # 板框底色（照 vision：板内铺一层浅米白；放在网格之下，网格仍可见）
    _rect(ax, bb, face=C_BOARD_BG, edge="none", lw=0.0, zorder=-1.0)

    if show_keepouts and keepouts:
        for kp in keepouts:
            _rect(ax, kp.aabb, edge=C_KEEP, lw=0.9, ls="--", zorder=3.0)
            n_keep += 1

    if show_vias:
        for v in model.vias:
            if layer not in v.layers:                    # 层不匹配的过孔不画
                continue
            _rect(ax, v.aabb, edge=C_VIA, lw=0.8, zorder=3.5)
            n_via += 1

    for ref, (x, y, k) in poses.items():
        fp = by_ref.get(ref)
        if fp is None or int(fp.layer) != int(layer):
            continue
        placed = pm.place_at(fp, (float(x), float(y)), int(k))      # 位姿化克隆：hull/pad 都在板坐标
        _rect(ax, placed.hull_aabb, face=C_BODY, face_alpha=C_BODY_A,
              edge=C_BODY_E, lw=0.9, zorder=4.0)
        n_body += 1
        if label_refs:
            cx = (placed.hull_aabb[0] + placed.hull_aabb[2]) / 2.0
            cy = (placed.hull_aabb[1] + placed.hull_aabb[3]) / 2.0
            ax.text(cx, cy, ref, ha="center", va="center", fontsize=_label_font(w, h),
                    color=C_LABEL, zorder=6.0)
        if show_pads:
            for p in placed.pads:
                if layer not in p.layers:
                    continue
                pth = len(p.layers) > 1                              # (0,1) ⇒ 贯穿
                _rect(ax, p.aabb, face=("none" if pth else C_PAD),
                      face_alpha=(None if pth else C_PAD_A), edge=C_PAD_E,
                      lw=0.6, zorder=5.0)
                n_pad += 1

    # 板框 + 坐标范围（Y 轴向下 = KiCad 方向）
    _rect(ax, bb, edge=C_BOARD, lw=1.6, zorder=7.0)

    # HPWL 的框（浅黄线框，**最后画 = 置顶**，Owner 2026-09-23）：与 `PcbModel.hpwl()` 同源
    # （不分层 ⇒ 两列画的是同一批框；`net_boxes_of` 只取传入 poses 的件）
    if show_net_boxes:
        for _net_id, box in net_boxes_of(model, poses).items():
            _rect(ax, box, edge=C_NETBOX, lw=1.0, zorder=9.0)
            n_box += 1
    mx, my = 0.05 * w, 0.05 * h
    ax.set_xlim(bb[0] - mx, bb[2] + mx)
    ax.set_ylim(bb[3] + my, bb[1] - my)                          # 注意：反向 ⇒ Y 向下
    ax.set_aspect("equal")
    ax.set_title(panel_title, fontsize=9.5)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm, KiCad 方向)", fontsize=8)
    ax.tick_params(labelsize=7)
    return {"body": n_body, "pad": n_pad, "via": n_via, "keepout": n_keep, "net_box": n_box}


def _label_font(w: float, h: float) -> float:
    """ref 字号随板尺寸走（小板大一点、大板小一点），落在 4.0~6.5。"""
    span = max(w, h)
    return float(min(6.5, max(4.0, 6.5 - (span - 20.0) / 60.0)))


# ===================== 对外主函数 =====================

def draw_compare(model: pm.PcbModel, *, model_poses: Mapping[str, Tuple[float, float, int]],
                 out_path: str, expert_poses: Optional[Mapping[str, Tuple[float, float, int]]] = None,
                 keepouts_expert: Optional[Sequence[pm.KeepoutRegion]] = None,
                 keepouts_model: Optional[Sequence[pm.KeepoutRegion]] = None,
                 subtitle: str = "", dpi: int = 110, show_pads: bool = True,
                 show_keepouts: bool = True, show_vias: bool = True, label_refs: bool = True,
                 show_net_boxes: bool = False, board: str = "") -> Dict[str, Any]:
    """出图：**上排 = 专家（源位姿）**、**下排 = 模型推理布局**；左右 = 顶层 / 底层。

    Args:
        model: 板子的几何（`PcbModel`）。
        model_poses: 模型布局位姿表 `ref → (x, y, rot)`。
        expert_poses: 专家位姿表；缺省 = `expert_poses_of(model)`（导入快照）。
        keepouts_expert / keepouts_model: 两排各自的禁布区（**同帧纪律**：专家排用源位姿口径、
            模型排用推理结束那一刻的 `state.keepouts`）；缺省用 `model.keepouts`。
        show_net_boxes: **是否画 HPWL 的框**（浅黄线框；每个非电源 net 一个；`net_boxes_of` 同源口径）。
            **默认 `False` = 不画**（Owner 2026-09-23：做成可选项）；置 `True` 时框画在**最上层**。
        out_path / dpi: 输出 PNG 与分辨率。
        subtitle: 写进大标题的补充信息（例如 `HPWL÷H_ref`、`σ`）。
    """
    expert_poses = dict(expert_poses) if expert_poses is not None else expert_poses_of(model)
    model_poses = dict(model_poses)
    bb = model.board_outline.bbox
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pw = float(min(9.0, max(4.5, w / 10.0)))
    ph = float(min(8.0, max(3.5, h / 10.0)))

    fig, axes = plt.subplots(2, 2, figsize=(2 * pw + 1.0, 2 * ph + 1.6), dpi=int(dpi))
    rows = (("专家（源位姿）", expert_poses, keepouts_expert if keepouts_expert is not None else model.keepouts),
            ("模型推理布局", model_poses, keepouts_model if keepouts_model is not None else model.keepouts))
    cols = ((0, "顶层 (F.Cu)"), (1, "底层 (B.Cu)"))

    stats: Dict[str, Any] = {}
    for r, (row_name, poses, kos) in enumerate(rows):
        for c, (layer, col_name) in enumerate(cols):
            stats[f"{r}{c}"] = _draw_panel(axes[r][c], model, poses, kos, layer,
                                           panel_title=f"{row_name} · {col_name}",
                                           show_pads=show_pads, show_keepouts=show_keepouts,
                                           show_vias=show_vias, label_refs=label_refs,
                                           show_net_boxes=show_net_boxes)

    # 大标题 + 底部横排图例（图例只做一次）
    head = f"{board or model.path or 'board'}｜上排 = 专家（源位姿）｜下排 = 模型推理布局（mask = L）"
    fig.suptitle(head + (f"\n{subtitle}" if subtitle else ""), fontsize=11.5)
    items = _LEGEND if show_net_boxes else tuple(  # 不画框时，图例里也不出现那一项（否则误导）
        it for it in _LEGEND if not it[0].startswith("HPWL"))
    handles = [Rectangle((0, 0), 1, 1, facecolor=(f if fill else "none"), edgecolor=e, linewidth=1.2)
               for _lab, f, e, fill in items]
    fig.legend(handles, [lab for lab, _f, _e, _fill in items], loc="lower center",
               ncol=len(items), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(out_path, dpi=int(dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "figsize": [round(2 * pw + 1.0, 2), round(2 * ph + 1.6, 2)],
            "panels": stats, "n_expert": len(expert_poses), "n_model": len(model_poses),
            "show_net_boxes": bool(show_net_boxes),
            # HPWL 框的数量 = **实际画出的**（默认不画 ⇒ 0；同排顶层/底层相同，因为 HPWL 口径不分层）
            "n_net_boxes": ({"expert": len(net_boxes_of(model, expert_poses)),
                             "model": len(net_boxes_of(model, model_poses))} if show_net_boxes
                            else {"expert": 0, "model": 0})}
