# -*- coding: utf-8 -*-
"""**开槽（keepout）口径 (A) 的回归测试** —— 台账 §0.6 p（2026-09-18 用 KiCad 本体裁决）。

口径 (A)：**开槽属于封装，随 footprint 移动/旋转**；未放置件的槽**不在板上**。
（事实依据 = `pcbnew` 行为：平移/旋转带槽 footprint 时，`GetBoardPolygonOutlines()` 报的
 板轮廓内孔**正好跟着动** ⇒ 见下方 `test_kicad_moves_slots_with_footprint`。）

检查项（共 **12 项**）：
 1. KiCad 权威：孔随件移动/旋转（外部权威，裁决口径）
 2. **空板：无任何 footprint 开槽被纳入**（静态禁布区仍在，且 L 不受"源位姿幽灵槽"影响）
 3. **放置后：只纳入【已放件】的槽，不多不少**（owner 集合与槽数精确相等；含"部分已放"的尖锐情形）
 4. **反复移动/旋转：槽跟随移动，且几何与判定无误**（与 KiCad 本体逐姿态对拍 + `Judge` 复核）
 5. **唯一入口守门**：`pose_updated_keepouts` / `state_keepouts_by_pose` **不得复活**；
    口径 A 只纳入已放件的槽；`legal_masks` 的默认 keepouts 必须就是 `state_keepouts(model, placed)`
 6. `state_keepouts` 行为：空板无槽；已放件 ⇒ 槽按位姿在场
 7. **`compress_legal` == 从零重算**（lr1388：`now=USBC1` **自带槽**）⇒ 逐格 0 不一致
 8. `comp_next` 也自带槽时同样 0 不一致（覆盖"去掉 `now.pad↔next.slot` 对"的改动）
"""
import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr                 # noqa: E402
from env import lookahead as la                   # noqa: E402
from env import violations as vio                 # noqa: E402
from utils.kicad_utils import pcb_model as pm     # noqa: E402

pytestmark = pytest.mark.require_kicad
# ⚠️ 必须走 importorskip，不能写成 `pcbnew = pm.pcbnew`：模块级硬取属性会在**收集期**抛
#    AttributeError ⇒ 没设 KICAD_PATH 的机器上 `pytest tests` 直接中断（0 用例跑）。
#    `require_kicad` 那条跳过走的是 runtest setup，管不到收集期。
pcbnew = pytest.importorskip("pcbnew", reason="需要 KiCad 环境（KICAD_PATH）")


def _board(sub, all_kicad_pcb_paths):
    hits = [p for p in all_kicad_pcb_paths if sub in os.path.basename(os.path.dirname(p))]
    assert hits, f"未找到板 {sub}"
    return hits[0]


def _scene(sub, all_kicad_pcb_paths, now_ref=None, next_ref=None):
    """构造局面：`placed` = 除 `now/next` 之外的全部（保持源位姿）。"""
    from env.order_plan import plan_order
    path = _board(sub, all_kicad_pcb_paths)
    model = pm.build_model(pcbnew.LoadBoard(path), path)
    byref = {f.ref: f for f in model.footprints}
    owners = sorted({k.owner_ref for k in model.keepouts if k.kind == "board_slot"})
    now = byref[now_ref] if now_ref else byref[owners[0]]
    if next_ref:
        nxt = byref[next_ref]
    else:                                   # 缺省：按固定顺序取第一个 ≠ now 的件
        order = [r for r in plan_order(model) if r in byref]
        nxt = byref[next(r for r in order if r != now.ref)]
    assert now.ref != nxt.ref, "now 与 next 不能是同一个件"
    placed = [f for f in model.footprints if f.ref not in {now.ref, nxt.ref}]
    return path, model, now, nxt, placed, owners


def _holes(brd):
    """板轮廓内孔 bbox（mm，已排序）。"""
    poly = pcbnew.SHAPE_POLY_SET()
    brd.GetBoardPolygonOutlines(poly, True)
    out = []
    for i in range(poly.OutlineCount()):
        for j in range(poly.HoleCount(i)):
            h = poly.CHole(i, j)
            xs = [pm._mm(h.CPoint(k).x) for k in range(h.PointCount())]
            ys = [pm._mm(h.CPoint(k).y) for k in range(h.PointCount())]
            out.append((round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)))
    return sorted(out)


def test_kicad_moves_slots_with_footprint(all_kicad_pcb_paths):
    """① KiCad 本体：平移/旋转带槽 footprint ⇒ 板轮廓内孔跟着动（外部权威 ⇒ 口径 (A)）。"""
    path = _board("bliysfyt", all_kicad_pcb_paths)
    model = pm.build_model(pcbnew.LoadBoard(path), path)
    owners = sorted({k.owner_ref for k in model.keepouts if k.kind == "board_slot"})
    assert owners, "该板没有自带开槽的件（样本为空）"

    brd = pcbnew.LoadBoard(path)
    builder = pm.PcbModelBuilder()
    canon = {builder._footprint_ref(fp, i): fp for i, fp in enumerate(brd.GetFootprints())}
    fp = canon[owners[0]]
    h0 = _holes(brd)
    assert len(h0) >= 2, "该板应有 ≥2 个内孔（两个定位槽）"

    dx, dy = 3.0, 2.0
    p = fp.GetPosition()
    fp.SetPosition(pcbnew.VECTOR2I(int(round((pm._mm(p.x) + dx) * 1e6)),
                                   int(round((pm._mm(p.y) + dy) * 1e6))))
    h1 = _holes(brd)
    dels = {(round(b[0] - a[0], 3), round(b[1] - a[1], 3)) for a, b in zip(h0, h1)}
    print(f"\n[bliysfyt/{owners[0]}] 平移 ({dx:+.1f},{dy:+.1f}) 后内孔位移集合 = {dels}")
    assert (dx, dy) in dels, "开槽没有跟着 footprint 平移 ⇒ 口径 (A) 不成立"
    assert (0.0, 0.0) not in dels, "该板两个孔都属于同一个件 ⇒ 应当全部跟着移动，不应有孔不动"

    fp.SetOrientationDegrees(90.0)
    h2 = _holes(brd)
    assert sorted(h2) != sorted(h1), "旋转 90° 后开槽没有跟着转 ⇒ 口径 (A) 不成立"


def _holes_raw(brd):
    """板轮廓内孔 bbox（mm，**不四舍五入**，按 bbox 排序）—— 用于与我们的 AABB 精确比对。"""
    poly = pcbnew.SHAPE_POLY_SET()
    brd.GetBoardPolygonOutlines(poly, True)
    out = []
    for i in range(poly.OutlineCount()):
        for j in range(poly.HoleCount(i)):
            h = poly.CHole(i, j)
            xs = [pm._mm(h.CPoint(k).x) for k in range(h.PointCount())]
            ys = [pm._mm(h.CPoint(k).y) for k in range(h.PointCount())]
            out.append((min(xs), min(ys), max(xs), max(ys)))
    return sorted(out)


def _keepout_key(k):
    """禁布区的"身份+几何"键（用于集合相等的严格比对）。"""
    return (k.kind, k.owner_ref, tuple(round(v, 9) for v in k.aabb), tuple(k.layers))


def _static_keepouts(model):
    """**静态**禁布区 = owner 不是"可动 footprint"的那些（如 `TT马达` hull）⇒ 恒在场。"""
    refs = {f.ref for f in model.footprints}
    return [k for k in model.keepouts if not k.owner_ref or k.owner_ref not in refs]


# ---------------- ② 空板：无 footprint 开槽被纳入 ----------------

@pytest.mark.parametrize("sub,comp_ref,golden_blocked", [
    ("bliysfyt", "U1", 1485),        # L(空局面)=8852 vs L(旧口径, 含源位姿槽)=7367
    ("wzh2814685", "U1", 3472),      # 影响最大的一块：3735 vs 263
])
def test_no_slots_on_empty_board(sub, comp_ref, golden_blocked, all_kicad_pcb_paths):
    """空板：**没有任何 footprint 的开槽**被纳入禁布区（静态项仍在），且 L 不受幽灵槽影响。"""
    path = _board(sub, all_kicad_pcb_paths)
    model = pm.build_model(pcbnew.LoadBoard(path), path)
    byref = {f.ref: f for f in model.footprints}
    spec = gr.grid_spec(model, m=gr.GRID_M)
    slots = [k for k in model.keepouts if k.kind == "board_slot"]
    assert slots, f"{sub}: 该板没有开槽（样本为空，纪律 2）"
    comp = byref[comp_ref]

    kp_empty = pm.state_keepouts(model, [])
    # ① 逐项相等：空局面禁布区 == 静态禁布区（独立枚举），且不含任何 board_slot
    assert sorted(map(_keepout_key, kp_empty)) == sorted(map(_keepout_key, _static_keepouts(model))), \
        "空局面的禁布区必须恰好等于【静态项】（不含任何 footprint 开槽）"
    assert not [k for k in kp_empty if k.kind == "board_slot"], "空板上不该有任何 board_slot"

    # ② L 层面：默认（口径 A）必须等于"显式传静态集合"，且**只可能比旧口径更宽松**
    L_a = gr.legal_masks(model, [], comp, spec)
    L_explicit = gr.legal_masks(model, [], comp, spec, keepouts=kp_empty)
    L_old = gr.legal_masks(model, [], comp, spec, keepouts=model.keepouts)   # 旧口径（含源位姿槽）
    assert np.array_equal(L_a, L_explicit), "默认 keepouts 与显式静态集合不一致"
    n_phantom_block = int((L_a & ~L_old).sum())  # 口径 A 合法、旧口径（源位姿槽）封掉 ⇒ 幽灵槽造成
    n_newly_block = int((~L_a & L_old).sum())    # 口径 A 封掉、旧口径合法 ⇒ 应为 0（去障碍只会更宽松）
    print(f"\n[{sub}/{comp_ref}] 槽 {len(slots)} 个（槽主 {sorted({k.owner_ref for k in slots})}）"
          f"｜L(口径A)={int(L_a.sum())}｜L(旧,含源位姿槽)={int(L_old.sum())}"
          f"｜幽灵槽多封 {n_phantom_block} 格｜反向差异 {n_newly_block}")
    assert n_newly_block == 0, "口径 A 只会比旧口径更宽松（去掉幽灵槽 ⇒ 不可能多封）"
    assert n_phantom_block == golden_blocked, \
        f"{sub}: 幽灵槽多封格数变了（{n_phantom_block} ≠ {golden_blocked}）"


# ---------------- ③ 放置后：只纳入已放件的槽，不多不少 ----------------

@pytest.mark.parametrize("sub", ["lr1388", "verysunshine"])
def test_only_placed_owners_slots_are_included(sub, all_kicad_pcb_paths):
    """放置后：纳入的槽 = **恰好**已放件所拥有的槽（owner 集合与槽数都精确相等）。"""
    path = _board(sub, all_kicad_pcb_paths)
    model = pm.build_model(pcbnew.LoadBoard(path), path)
    byref = {f.ref: f for f in model.footprints}
    slots = [k for k in model.keepouts if k.kind == "board_slot"]
    owners = sorted({k.owner_ref for k in slots})
    assert len(owners) >= 2, f"{sub}: 需要 ≥2 个槽主才能构造'部分已放'（样本为空，纪律 2）"
    allf = list(model.footprints)
    static_n = len(_static_keepouts(model))

    for tag, placed in (
        ("全部已放", allf),
        (f"只差 {owners[0]} 未放", [f for f in allf if f.ref != owners[0]]),
        (f"只差 {owners[1]} 未放", [f for f in allf if f.ref != owners[1]]),
        (f"{owners[0]},{owners[1]} 都未放", [f for f in allf if f.ref not in set(owners[:2])]),
        ("空板", []),
    ):
        placed_refs = {f.ref for f in placed}
        want = [k for k in slots if k.owner_ref in placed_refs]        # 独立枚举的期望集合
        kps = pm.state_keepouts(model, placed)
        got = [k for k in kps if k.kind == "board_slot"]
        print(f"  [{sub}] {tag:22s} 已放 {len(placed):3d} 件 ⇒ 纳入槽 {len(got)}（期望 {len(want)}）"
              f"｜槽主 {sorted({k.owner_ref for k in got})}")
        assert sorted(map(_keepout_key, got)) == sorted(map(_keepout_key, want)), \
            f"{sub}/{tag}: 纳入的槽与'已放件所拥有'不一致（不多不少没守住）"
        # 静态项必须在所有情形下都在场
        assert len([k for k in kps if k.kind != "board_slot"]) == static_n, \
            f"{sub}/{tag}: 静态禁布区数量变了"

    # L 消费一致性：默认（推导）== 显式传入独立枚举的集合
    placed = [f for f in allf if f.ref != owners[0]]
    comp = byref[owners[0]]                       # 唯一未放件 ⇒ 用它的 L 来看"另一槽主的槽是否被正确消费"
    spec = gr.grid_spec(model, m=gr.GRID_M)
    want = [k for k in slots if k.owner_ref in {f.ref for f in placed}]
    L_a = gr.legal_masks(model, placed, comp, spec)
    L_b = gr.legal_masks(model, placed, comp, spec, keepouts=tuple(want))
    print(f"  [{sub}] L 消费一致性：comp={comp.ref}｜L={int(L_a.sum())}｜与显式集合({'一致' if np.array_equal(L_a, L_b) else '不一致'})")
    assert np.array_equal(L_a, L_b), "L 实际消费的禁布区集合与'已放件的槽'不一致"


# ---------------- ④ 反复移动：槽跟随 + 计算无误（KiCad 权威 + Judge 复核） ----------------

def test_repeated_moves_slots_follow_and_judgement_ok(all_kicad_pcb_paths):
    """反复平移/旋转同一个带槽件：**槽跟着走**（与 KiCad 本体逐姿态对拍），且 L 与 `Judge` 一致。"""
    sub = "bliysfyt"                      # 该板只有 USB1 带槽（2 个）⇒ 内孔可与我们的槽一一配对
    path = _board(sub, all_kicad_pcb_paths)
    model = pm.build_model(pcbnew.LoadBoard(path), path)
    byref = {f.ref: f for f in model.footprints}
    spec = gr.grid_spec(model, m=gr.GRID_M)
    slots = [k for k in model.keepouts if k.kind == "board_slot"]
    owners = sorted({k.owner_ref for k in slots})
    assert len(owners) == 1, f"{sub}: 该用例要求'只有一个槽主'（当前 {owners}）"
    owner_ref = owners[0]
    src = byref[owner_ref]
    assert len(slots) == 2, f"{sub}: 期望 2 个槽（实际 {len(slots)}）"

    brd = pcbnew.LoadBoard(path)
    canon = {pm.PcbModelBuilder()._footprint_ref(fp, i): fp for i, fp in enumerate(brd.GetFootprints())}
    live = canon[owner_ref]
    comp = byref["U1"]                     # 用**非槽主**的件来看"槽移动是否被正确消费"
    others_src = [f for f in model.footprints if f.ref not in {owner_ref, "U1"}]

    ax, ay = src.anchor
    k0 = src.rotation_index
    # 候选位姿：只保留"开槽 bbox 仍在板框内"的（槽被推出板外时 KiCad 不生成内孔 ⇒ 无法对拍）
    bb = model.board_outline.bbox
    cand = [("源位姿", ax, ay, k0),
            ("平移 (+2,+1)", ax + 2.0, ay + 1.0, k0),
            ("再平移 (+1,−1)", ax + 3.0, ay, k0),
            ("转 90°", ax + 3.0, ay, 1),
            ("转 180°", ax + 3.0, ay, 2)]
    poses = []
    for tag, x, y, k in cand:
        mv = pm.place_at(src, (x, y), k)
        aa = [k2.aabb for k2 in pm.state_keepouts(model, others_src + [mv])
              if k2.owner_ref == owner_ref]
        if aa and all(bb[0] <= a[0] and a[2] <= bb[2] and bb[1] <= a[1] and a[3] <= bb[3]
                      for a in aa):
            poses.append((tag, x, y, k))
    assert len(poses) >= 3 and poses[0][0] == "源位姿", \
        f"可用位姿不足（{len(poses)}）⇒ 样本为空（纪律 2）"

    prev_kp, prev_xy, n_L, L_all = None, None, [], []
    for tag, x, y, k in poses:
        moved = pm.place_at(src, (x, y), k)
        placed = others_src + [moved]
        kps = pm.state_keepouts(model, placed)
        mine = sorted(k.aabb for k in kps if k.owner_ref == owner_ref)

        live.SetPosition(pcbnew.VECTOR2I(int(round(x * 1e6)), int(round(y * 1e6))))
        live.SetOrientationDegrees(k * 90.0)
        holes = _holes_raw(brd)
        assert len(holes) == len(mine) == 2, f"{tag}: 孔数 {len(holes)} vs 我们的槽数 {len(mine)}"
        dev = max(abs(h[t] - m[t]) for h, m in zip(holes, mine) for t in range(4))
        # 平移步：位移向量必须精确等于位姿差（"跟随移动"的最强形式，与绝对定义无关）
        d_msg = ""
        if prev_kp is not None and k == k0:
            d_kp = [(round(n[0] - o[0], 6), round(n[1] - o[1], 6)) for o, n in zip(prev_kp, mine)]
            d_msg = f"｜槽位移 {d_kp}"
            assert all(abs(d[0] - (x - prev_xy[0])) < 1e-6 and abs(d[1] - (y - prev_xy[1])) < 1e-6
                       for d in d_kp), f"{tag}: 槽没有跟随平移（{d_kp}）"

        # L 层面：用非槽主件看在"槽已跟随"的局面下的合法域，并用 Judge（同一套 keepouts）复核
        L = gr.legal_masks(model, placed, comp, spec)
        assert int(L.sum()) > 0, f"{tag}: 该件无合法格（样本为空，纪律 2）"
        judge = vio.Judge(model, keepouts=kps)
        poses_map = {f.ref: vio.pose_of(f) for f in placed}
        rng = np.random.default_rng(17)
        picks = []
        for kk in range(4):
            c = np.argwhere(L[kk])
            if len(c):
                picks += [(kk, int(a), int(b)) for a, b in
                          c[rng.choice(len(c), size=min(12, len(c)), replace=False)]]
        n_bad = 0
        for kk, i, j in picks:
            cx, cy = spec.mm_of(i, j)
            if judge.judge(comp.ref, vio.Pose(cx, cy, kk), poses_map).reasons:
                n_bad += 1
        n_L.append(int(L.sum()))
        L_all.append(L.copy())
        print(f"  [{sub}] {tag:16s} 槽 AABB 与 KiCad 孔最大偏差 {dev:.3e} mm｜L({comp.ref})={int(L.sum()):6d}"
              f"｜Judge 复核 {len(picks)} 格 ⇒ 违规 {n_bad}{d_msg}")
        assert dev < 1e-3, f"{tag}: 槽几何与 KiCad 本体偏差过大（{dev:.3e} mm）"
        assert n_bad == 0, f"{tag}: 槽跟随后的 L 与 Judge 不一致（{n_bad} 格）"
        prev_kp, prev_xy = mine, (x, y)

    # 判别力：槽移动后，该件的**合法域本身**必须发生变化（不只是计数）
    changed = [i for i in range(1, len(L_all)) if not np.array_equal(L_all[i], L_all[0])]
    assert len(set(n_L)) >= 2 or changed, \
        f"槽移动后 {comp.ref} 的合法域竟然没变 ⇒ 用例失去判别力（计数 {n_L}）"
    assert changed, f"槽移动后 {comp.ref} 的合法域逐格完全相同 ⇒ 用例失去判别力（计数 {n_L}）"


def test_state_keepouts_follows_placement(all_kicad_pcb_paths):
    """② `state_keepouts`：空板 ⇒ 无 `board_slot`；已放件 ⇒ 槽按位姿在场。"""
    # 让"自带开槽的件"留在 placed 里（故 now 取一个非槽主件）
    _path, model, now, _nxt, placed, owners = _scene("bliysfyt", all_kicad_pcb_paths,
                                                    now_ref="U1", next_ref="H1")
    slots_all = [k for k in model.keepouts if k.kind == "board_slot"]
    assert slots_all, "样本为空"

    kp_empty = pm.state_keepouts(model, [])
    assert not [k for k in kp_empty if k.kind == "board_slot"], \
        "空板上不应有任何 board_slot（未放置件的槽不在板上）"

    kp_placed = pm.state_keepouts(model, placed)
    got = [k for k in kp_placed if k.kind == "board_slot"]
    assert len(got) == len([k for k in slots_all if k.owner_ref in {f.ref for f in placed}])

    # 被挪动的 owner ⇒ 它的槽跟着位姿
    cand = [f for f in placed if f.ref in set(owners)]
    assert cand, "该场景的已放件里没有自带开槽的件（样本为空）"
    victim = cand[0]
    moved = pm.place_at(victim, (victim.anchor[0] + 3.0, victim.anchor[1] + 2.0),
                        victim.rotation_index)
    kp_moved = pm.state_keepouts(model, [moved if f.ref == victim.ref else f for f in placed])
    a = [k.aabb for k in kp_placed if k.owner_ref == victim.ref]
    b = [k.aabb for k in kp_moved if k.owner_ref == victim.ref]
    assert a and a != b, "已放件被挪动后其开槽没跟着走"
    assert all(abs((y[0] - x[0]) - 3.0) < 1e-9 and abs((y[1] - x[1]) - 2.0) < 1e-9
               for x, y in zip(sorted(a), sorted(b))), "开槽位移量不为 (3,2)"


# ---------------- ⑤ 唯一入口（2026-09-18 Owner："删掉两个没用的"） ----------------

def test_single_keepout_entry_point(all_kicad_pcb_paths):
    """口径 A **只保留 `state_keepouts` 一个入口**；"保留源位姿"的同义函数不得复活。"""
    for name in ("pose_updated_keepouts", "state_keepouts_by_pose"):
        assert not hasattr(pm, name), (
            f"`{name}` 已被删除：它的语义是「其余保持原样」⇒【未放置件】的槽会留在专家位（幽灵槽），"
            f"实测让待放件 U1 的 L 少 1485 格（bliysfyt，probe177）。口径 A 的唯一入口 = "
            f"`state_keepouts(model, placed)`；若确实需要「保留源位姿」的诊断行为，请显式构造 keepout 列表。")

    _path, model, now, _nxt, placed, owners = _scene("lr1388", all_kicad_pcb_paths)
    placed_refs = {f.ref for f in placed}
    unplaced_owners = [o for o in owners if o not in placed_refs]
    assert unplaced_owners, "该场景没有「未放置的槽主」（样本为空，纪律 2）"

    kps = pm.state_keepouts(model, placed)
    got_owners = {k.owner_ref for k in kps if k.kind == "board_slot"}
    print(f"\n[lr1388] 全部槽主 {owners}｜未放置 {unplaced_owners}｜"
          f"state_keepouts 纳入的槽主 {sorted(got_owners)}")
    assert got_owners == (set(owners) & placed_refs), "口径 A：只该纳入【已放件】的槽（不多不少）"

    spec = gr.grid_spec(model, m=gr.GRID_M)
    assert np.array_equal(
        gr.legal_masks(model, placed, now, spec),
        gr.legal_masks(model, placed, now, spec, keepouts=pm.state_keepouts(model, placed))), \
        "`legal_masks` 的默认 keepouts 必须就是 `state_keepouts(model, placed)`（唯一入口）"


def _increment_equivalence(model, now, nxt, placed, spec, n=40, seed=3):
    """核心对拍：`compress_legal(L_next0)` vs `真放下后从零重算` ⇒ 返回 (抽样数, 不一致候选数, 差异格数)。"""
    L = gr.legal_masks(model, placed, now, spec)
    Ln = gr.legal_masks(model, placed, nxt, spec)
    if not L.any() or not Ln.any():
        pytest.skip("该局面合法集为空（样本为空）")
    rng = np.random.default_rng(seed)
    picks = []
    for k in range(4):
        c = np.argwhere(L[k])
        if len(c):
            take = c[rng.choice(len(c), size=min(n, len(c)), replace=False)]
            picks += [(k, int(a), int(b)) for a, b in take]
    n_bad = n_cells = 0
    for k, i, j in picks:
        x, y = spec.mm_of(i, j)
        inc = la.compress_legal(model, now, k, i, j, nxt, Ln, spec)
        now_at = pm.place_at(now, (x, y), k)
        ref = gr.legal_masks(model, list(placed) + [now_at], nxt, spec)
        d = int((inc ^ ref).sum())
        if d:
            n_bad += 1
            n_cells += d
    return len(picks), n_bad, n_cells


@pytest.mark.parametrize("sub,now_ref,next_ref", [
    ("lr1388", "USBC1", None),      # now 自带 2 槽（SW8 作为 next）
    ("lr1388", "SW8", "USBC1"),     # **next 也自带槽**（覆盖"去掉 now.pad↔next.slot"的改动）
    ("bliysfyt", "USB1", None),     # USB1 自带 2 槽
    ("verysunshine", "SW11", None),  # SW11 自带 2 槽
])
def test_increment_equals_recompute_with_slots(sub, now_ref, next_ref, all_kicad_pcb_paths):
    """③④ 口径 (A) 下："增量压缩 == 从零重算" 必须逐格一致（这是 §0.6 p 的收官验证）。"""
    _path, model, now, nxt, placed, _owners = _scene(sub, all_kicad_pcb_paths, now_ref, next_ref)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    n, n_bad, n_cells = _increment_equivalence(model, now, nxt, placed, spec)
    print(f"\n[{sub}] now={now.ref}（槽 {sum(1 for k in model.keepouts if k.owner_ref == now.ref)}）"
          f" next={nxt.ref}（槽 {sum(1 for k in model.keepouts if k.owner_ref == nxt.ref)}）"
          f"｜抽样 {n}｜不一致候选 {n_bad}｜差异格 {n_cells}")
    assert n > 0, "样本为空（纪律 2）"
    assert n_bad == 0 and n_cells == 0, f"{sub}: 增量与从零重算不一致（{n_bad} 候选 / {n_cells} 格）"
