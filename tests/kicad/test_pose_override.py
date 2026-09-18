# -*- coding: utf-8 -*-
"""**位姿更新**回归测试（台账 §0.6 o；Owner 2026-09-18："不能拿原始 footprint 位置算，那是专家数据的"）。

背景
----
`pad.aabb` / `pad.center` / `model.keepouts[].aabb` 都是**导入时按源位姿**算好的快照，
而源位姿 = **专家布局的位置**。策略真实落子后（位姿 ≠ 源位姿），几何必须**按位姿反解**：

```text
place_at(fp, anchor, rot)                        # pad 的 aabb/center 实时反解（克隆）
state_keepouts(model, placed)                    # 禁布区唯一入口（口径 A：自带开槽跟着件走）
```

用 `Judge`（**认位姿**的权威实现）作真相（共 **8 项**）：
  A. `place_at` 与 `pcbnew` 实测一致（外部权威）
  B. 位姿更新后 `legal_masks` 与 `Judge` 一致（危险 = 0）
  C. **守门**：不给位姿更新（用源快照）时**确实会放过违规**（probe165 实测 verysunshine 4 格）
  D. 自带开槽的件被挪动时，槽必须跟着走（几何级 + 判定级）
  E. G 的两个分量（`hpwl_gain_maps` / `dispersion_delta` 与 `PcbModel.hpwl`）取点必须走"当前位姿"
  F. **取点桥契约（§0.6 q-⑤）**：`_pose_pad_index` 对坏克隆（pad 数不同 / 同长但不同 pad / 外来 ref）
     必须**硬失败**（否则 `_center_now` 会静默取错中心）
"""
import glob
import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import grid_render as gr                 # noqa: E402
from env import violations as vio                 # noqa: E402
from env.order_plan import plan_order             # noqa: E402
from utils.kicad_utils import pcb_model as pm     # noqa: E402

pytestmark = pytest.mark.require_kicad
MOVE = (2.0, 1.5)          # 板内小位移（挪出板外只会"多封"、不暴露危险方向）


def _mm_to_iu(v: float) -> int:
    return int(round(v * 1e6))                     # mm → nm（SWIG 要 int）


def _scene(sub, all_kicad_pcb_paths):
    hits = [p for p in all_kicad_pcb_paths if sub in os.path.basename(os.path.dirname(p))]
    assert hits, f"未找到板 {sub}"
    model = pm.build_model(pm.pcbnew.LoadBoard(hits[0]), hits[0])
    byref = {f.ref: f for f in model.footprints}
    order = [r for r in plan_order(model) if r in byref]
    now, nxt = byref[order[0]], byref[order[1]]
    placed = [f for f in model.footprints if f.ref not in {now.ref, nxt.ref}]
    return hits[0], model, now, nxt, placed


def _move_one(placed, victim, dx, dy):
    """把 `victim` 在板内挪 `(dx, dy)` ⇒ 返回 (poses, placed_ok)；`placed_ok` 用 `place_at` 克隆。"""
    vx, vy, vk = victim.anchor[0] + dx, victim.anchor[1] + dy, victim.rotation_index
    poses = {f.ref: vio.pose_of(f) for f in placed}
    poses[victim.ref] = vio.Pose(vx, vy, vk)
    placed_ok = [pm.place_at(f, (vx, vy), vk) if f.ref == victim.ref else f for f in placed]
    return (vx, vy, vk), poses, placed_ok


def _kp_tuple(poses):
    return {r: (p[0], p[1], p[2]) for r, p in poses.items()}


def _judge_danger(model, now, L, poses, spec, keepouts=None, cells=None):
    """对给定格（缺省 = `L` 的全部合法格）问 `Judge` ⇒ 返回"L 说合法但 Judge 判违规"的格数。"""
    judge = vio.Judge(model, keepouts=keepouts)
    n_bad = 0
    for k in range(4):
        it = np.argwhere(L[k]) if cells is None else np.argwhere(cells[k])
        for i, j in it:
            x, y = spec.mm_of(int(i), int(j))
            if judge.judge(now.ref, vio.Pose(x, y, k), poses).reasons:
                n_bad += 1
    return n_bad


# ---------------- A. place_at vs pcbnew（外部权威） ----------------

def test_place_at_matches_kicad(all_kicad_pcb_paths):
    path = [p for p in all_kicad_pcb_paths if "bliysfyt" in os.path.basename(os.path.dirname(p))][0]
    model = pm.build_model(pm.pcbnew.LoadBoard(path), path)
    brd = pm.pcbnew.LoadBoard(path)
    builder = pm.PcbModelBuilder()
    canon = {}
    for idx, fp in enumerate(brd.GetFootprints()):
        canon[builder._footprint_ref(fp, idx)] = fp

    checked = 0
    worst = 0.0
    poses = ((model.footprints[0].anchor[0] + 1.0, model.footprints[0].anchor[1] + 0.5, 0),
             (150.0, 100.0, 1), (152.0, 102.0, 2), (155.0, 105.0, 3))
    for f in sorted(model.footprints, key=lambda x: -len(x.pads))[:6]:
        fp = canon[f.ref]
        saved_mm = (pm._mm(fp.GetPosition().x), pm._mm(fp.GetPosition().y),
                    float(fp.GetOrientationDegrees()))
        try:
            for (ax, ay, rot) in poses:
                fp.SetPosition(pm.pcbnew.VECTOR2I(_mm_to_iu(ax), _mm_to_iu(ay)))
                fp.SetOrientationDegrees(rot * 90.0)
                clone = pm.place_at(f, (ax, ay), rot)
                for idx, (p_old, live) in enumerate(zip(f.pads, list(fp.Pads()))):
                    got = pm.PcbModelBuilder._aabb(live.GetBoundingBox())
                    exp = clone.pads[idx].aabb
                    worst = max(worst, max(abs(got[t] - exp[t]) for t in range(4)))
                    gc = (pm._mm(live.GetPosition().x), pm._mm(live.GetPosition().y))
                    ec = clone.pads[idx].center
                    worst = max(worst, abs(gc[0] - ec[0]), abs(gc[1] - ec[1]))
                    checked += 1
                    del p_old
        finally:
            fp.SetPosition(pm.pcbnew.VECTOR2I(_mm_to_iu(saved_mm[0]), _mm_to_iu(saved_mm[1])))
            fp.SetOrientationDegrees(saved_mm[2])
    assert checked > 0, "样本为空（纪律 2）"
    assert worst < 1e-9, f"place_at 与 pcbnew 实测差 {worst:.3e} mm"


# ---------------- B. 位姿更新后 L 与 Judge 一致 ----------------

@pytest.mark.parametrize("sub", ["verysunshine", "bliysfyt"])
def test_pose_updated_L_matches_judge(sub, all_kicad_pcb_paths):
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    victim = max(placed, key=lambda f: len(f.pads))
    (vx, vy, _vk), poses, placed_ok = _move_one(placed, victim, *MOVE)
    bb = model.board_outline.bbox
    assert bb[0] <= vx <= bb[2] and bb[1] <= vy <= bb[3], "位移须在板内"

    kps = pm.state_keepouts(model, placed_ok)
    L_ok = gr.legal_masks(model, placed_ok, now, spec, keepouts=kps)
    bad = _judge_danger(model, now, L_ok, poses, spec, keepouts=kps)
    print(f"\n[{sub}] 受害件 {victim.ref}（{len(victim.pads)} pad）｜位姿更新后 L={int(L_ok.sum())}"
          f"｜Judge 危险 = {bad}")
    assert int(L_ok.sum()) > 0, "样本为空（纪律 2）"
    assert bad == 0, f"{sub}: 位姿更新后 L 仍与 Judge 不一致（{bad} 格）"


# ---------------- C. 守门：用源快照确实会放过违规（verysunshine 已实测） ----------------

def test_stale_pose_geometry_is_dangerous(all_kicad_pcb_paths):
    """⚠️ 这条是**守门**：证明"不给位姿更新"真的错（否则修复本身没有意义）。"""
    sub = "verysunshine"
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    victim = max(placed, key=lambda f: len(f.pads))
    (vx, vy, _vk), poses, placed_ok = _move_one(placed, victim, *MOVE)
    kps = pm.state_keepouts(model, placed_ok)                                 # 口径 A 唯一入口

    L_ok = gr.legal_masks(model, placed_ok, now, spec, keepouts=kps)          # 正确
    L_bad = gr.legal_masks(model, placed, now, spec, keepouts=kps)            # 源快照（错）
    diff = L_bad & ~L_ok                     # 源快照说合法、正确几何说非法 ⇒ 潜在危险
    real_bad = _judge_danger(model, now, L_bad, poses, spec, keepouts=kps, cells=diff)
    print(f"\n[{sub}] 受害件 {victim.ref}｜源快照 L={int(L_bad.sum())} vs 正确 L={int(L_ok.sum())}"
          f"｜危险差集 {int(diff.sum())} 格｜Judge 确认违规 = {real_bad}")
    assert int(diff.sum()) > 0, "本次位移没造出差异 ⇒ 守门用例失去意义"
    assert real_bad > 0, "源快照居然没放过违规 ⇒ 守门用例失去意义"


# ---------------- D. 自带开槽必须跟着件走 ----------------

def test_moved_keepout_is_used(all_kicad_pcb_paths):
    """`verysunshine` 的 `SW11/SW13` 自带开槽：挪动它们时，槽必须一起走。"""
    sub = "verysunshine"
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    owners = [f for f in placed if any(k.owner_ref == f.ref for k in model.keepouts)]
    assert owners, "该场景里没有「自带开槽的已放件」（样本为空）"
    victim = owners[0]
    (vx, vy, vk), poses, placed_ok = _move_one(placed, victim, 3.0, 2.0)
    bb = model.board_outline.bbox
    assert bb[0] <= vx <= bb[2] and bb[1] <= vy <= bb[3]

    # D1：几何级——`keepout_at` 必须等于"先还原局部、再按新位姿旋转"
    own_slots = [k for k in model.keepouts if k.owner_ref == victim.ref]
    kp_new = pm.state_keepouts(model, placed_ok)
    new_slots = [k for k in kp_new if k.owner_ref == victim.ref]
    assert len(new_slots) == len(own_slots) > 0
    for k0, k1 in zip(own_slots, new_slots):
        local = pm.board_to_local_aabb(k0.aabb, victim.anchor, victim.rotation_index)
        assert np.allclose(k1.aabb, pm.rotate_aabb(local, vk * 90.0, (vx, vy)), atol=1e-9)
        assert k1.aabb != k0.aabb or (vx, vy) == victim.anchor

    # D2：判定级——用"槽跟随"的 keepouts，L 必须与 Judge（同样槽跟随）一致
    L_new = gr.legal_masks(model, placed_ok, now, spec, keepouts=kp_new)
    bad = _judge_danger(model, now, L_new, poses, spec, keepouts=kp_new)
    L_old = gr.legal_masks(model, placed_ok, now, spec, keepouts=model.keepouts)   # 槽留原处（错）

    print(f"\n[{sub}] 开槽件 {victim.ref}（{len(own_slots)} 槽）｜L(槽跟随)={int(L_new.sum())}"
          f"｜L(槽留原处)={int(L_old.sum())}｜差异 {int((L_new ^ L_old).sum())} 格"
          f"｜Judge 危险 = {bad}")
    assert bad == 0, f"位姿化开槽后 L 仍与 Judge 不一致（{bad} 格）"


# ---------------- E. G 的两个分量 / PcbModel.hpwl 必须用"当前位姿"的中心 ----------------

def _centers_at(model, now, placed_ok, k=0, x=0.0, y=0.0, include_now=False):
    """**独立构造**：每个非电源 net 的连接点中心（`placed_ok` 用其位姿化后的 `center`）。"""
    th = math.radians(k * 90.0)
    c, s = math.cos(th), math.sin(th)
    out = {}
    for net in model.nets.values():
        if net.is_power:
            continue
        pts = [p.center for f in placed_ok for p in f.pads if p.net_id == net.net_id]
        if include_now:
            for p in now.pads:
                if p.net_id == net.net_id and p.local_aabb:
                    lx, ly = p.rel_offset
                    pts.append((x + lx * c + ly * s, y - lx * s + ly * c))
        out[net.net_id] = pts
    return out


def _hpwl(centers):
    tot = 0.0
    for pts in centers.values():
        if len(pts) < 2:
            continue
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        tot += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return tot


def _sigma(centers):
    cs = []
    for pts in centers.values():
        if len(pts) < 2:
            continue
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        cs.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    n = len(cs)
    if n == 0:
        return 0.0
    cx = sum(q[0] for q in cs) / n
    cy = sum(q[1] for q in cs) / n
    return math.sqrt(max(sum((q[0] - cx) ** 2 + (q[1] - cy) ** 2 for q in cs) / n, 0.0))


def _victim_with_extreme(model, now, placed, need_shared_net=False):
    """选一个"其 pad 是某 net 的**极值点**"的已放件 + 一个"向外"的位移。

    ⚠️ 判别性有两层要求（都是"HPWL 的敏感性"决定的）：
      1. 移动必须落在**极值点**上（挪内部件不改变 net 包围盒）；
      2. 若目标是让 **ΔHPWL** 变化，该 net 还必须**同时被 `now` 占用**
         （`now` 没参与的 net 对 `H = −ΔHPWL` 毫无影响）。
    """
    refs = {f.ref for f in placed}
    bb = model.board_outline.bbox
    now_nets = {p.net_id for p in now.pads if p.net_id}
    for f in sorted(placed, key=lambda x: -len(x.pads)):
        for net in model.nets.values():
            if net.is_power:
                continue
            if need_shared_net and net.net_id not in now_nets:
                continue
            pts = [p for p in net.pads if p.ref in refs]
            if len(pts) < 2:
                continue
            xs = [p.center[0] for p in pts]
            ys = [p.center[1] for p in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            for p in pts:
                if p.ref != f.ref:
                    continue
                if not (abs(p.center[0] - x0) < 1e-9 or abs(p.center[0] - x1) < 1e-9
                        or abs(p.center[1] - y0) < 1e-9 or abs(p.center[1] - y1) < 1e-9):
                    continue
                dx = 5.0 if p.center[0] >= (x0 + x1) / 2 else -5.0
                dy = 5.0 if p.center[1] >= (y0 + y1) / 2 else -5.0
                if not (bb[0] <= f.anchor[0] + dx <= bb[2]):
                    dx = -dx
                if not (bb[1] <= f.anchor[1] + dy <= bb[3]):
                    dy = -dy
                return f, (dx, dy)
    return None, (0.0, 0.0)


def test_G_components_use_pose_updated_centers(all_kicad_pcb_paths):
    """`hpwl_gain_maps` / `dispersion_delta` 的**已放点中心**必须取当前位姿（不是源位姿快照）。"""
    sub = "verysunshine"
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    victim, (dx, dy) = _victim_with_extreme(model, now, placed, need_shared_net=True)
    assert victim is not None, "找不到「与 now 共享 net 且在该 net 上是极值点」的已放件"
    _v, poses, placed_ok = _move_one(placed, victim, dx, dy)
    kps = pm.state_keepouts(model, placed_ok)
    L = gr.legal_masks(model, placed_ok, now, spec, keepouts=kps)
    assert int(L.sum()) > 0, "样本为空（纪律 2）"

    H_ok = gr.hpwl_gain_maps(model, placed_ok, now, spec)
    D_ok = gr.dispersion_delta(model, placed_ok, now, spec, L)
    H_bad = gr.hpwl_gain_maps(model, placed, now, spec)          # 源快照（错）

    h0 = _hpwl(_centers_at(model, now, placed_ok))
    s0 = _sigma(_centers_at(model, now, placed_ok))
    rng = np.random.default_rng(5)
    cells = []
    for k in range(4):
        cc = np.argwhere(L[k])
        if len(cc):
            take = cc[rng.choice(len(cc), size=min(120, len(cc)), replace=False)]
            cells += [(k, int(a), int(b)) for a, b in take]
    assert cells, "样本为空（纪律 2）"

    eH = eD = eH_bad = 0.0
    for k, i, j in cells:
        x, y = spec.mm_of(i, j)
        ctr = _centers_at(model, now, placed_ok, k, x, y, include_now=True)
        act_h = h0 - _hpwl(ctr)
        eH = max(eH, abs(float(H_ok[k][i, j]) - act_h))
        eD = max(eD, abs(float(D_ok[k][i, j]) - (_sigma(ctr) - s0)))
        eH_bad = max(eH_bad, abs(float(H_bad[k][i, j]) - act_h))
    print(f"\n[{sub}] 受害件 {victim.ref}（持有极值点，挪 ({dx:+.1f},{dy:+.1f})）"
          f"｜位姿化 max|ΔHPWL|={eH:.2e}、max|Δσ|={eD:.2e}"
          f"｜源快照 max|ΔHPWL|={eH_bad:.2e}")
    assert eH < 1e-4, f"位姿化后 HPWL 分量仍与独立重算不一致（{eH:.3e}）"
    assert eD < 1e-10, f"位姿化后 Δσ 仍与独立重算不一致（{eD:.3e}）"
    assert eH_bad > 1e-3, "源快照居然和正确值一样 ⇒ 守门用例失去判别力"


def test_model_hpwl_supports_poses(all_kicad_pcb_paths):
    """`PcbModel.hpwl(refs, poses=...)` 必须按当前位姿取点（缺省 = 源位姿）。"""
    sub = "verysunshine"
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    victim, (dx, dy) = _victim_with_extreme(model, now, placed)
    assert victim is not None, "样本为空（纪律 2）"
    _v, poses, placed_ok = _move_one(placed, victim, dx, dy)
    refs = {f.ref for f in placed}
    ptu = _kp_tuple(poses)
    got = model.hpwl(refs, poses=ptu)
    exp = _hpwl(_centers_at(model, now, placed_ok))
    stale = model.hpwl(refs)
    # 连接点本身必须变（单位级、确定性）：取一条含该受害件的非电源 net 验证
    net = next((n for n in model.nets.values()
                if not n.is_power and any(p.ref == victim.ref for p in n.pads)
                and sum(1 for p in n.pads if p.ref in refs) >= 2), None)
    assert net is not None, "找不到可用于验证的连接点 net（样本为空）"
    assert net.connection_points(refs, ptu) != net.connection_points(refs), \
        "连接点没有随位姿变化 ⇒ 用例失去意义"
    print(f"\n[{sub}] PcbModel.hpwl：位姿化 {got:.6f} ｜独立重算 {exp:.6f} ｜源位姿 {stale:.6f}")
    assert abs(got - exp) < 1e-6, f"hpwl(poses=...) 与独立重算不一致（{got:.6f} vs {exp:.6f}）"
    assert abs(stale - got) > 1e-6, "源位姿与位姿化 HPWL 相同 ⇒ 守门用例失去判别力"


# ---------------- F. ⑤ 取点桥的契约硬校验（§0.6 q-⑤） ----------------

def test_pose_pad_index_contract(all_kicad_pcb_paths):
    """`_pose_pad_index` 对"坏克隆"必须**硬失败** —— 否则 `_center_now` 会静默取错中心
    （HPWL / 离散度都错、且不报错）。覆盖三类：pad 数不同 / 同长但不同 pad / 外来 ref。"""
    sub = "verysunshine"
    _path, model, now, _nxt, placed = _scene(sub, all_kicad_pcb_paths)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    victim = max(placed, key=lambda f: len(f.pads))
    _v, _poses, placed_ok = _move_one(placed, victim, *MOVE)
    t = next(i for i, f in enumerate(placed_ok) if f.ref == victim.ref)
    clone = placed_ok[t]
    assert len(clone.pads) > 1, "样本为空（纪律 2）"

    pmap = gr._pose_pad_index(model, placed_ok)               # ① 正常路径：不抛、映射覆盖完整
    assert victim.ref in pmap and len(pmap[victim.ref][1]) == len(victim.pads)
    H = gr.hpwl_gain_maps(model, placed_ok, now, spec)        # 正常路径：G 照常可算（回归）
    assert H.shape == (4, spec.m, spec.m)

    broken = [
        ("pad 数不同", replace(clone, pads=clone.pads[:-1])),
        ("同长但不同 pad",
         replace(clone, pads=(replace(clone.pads[0], rel_offset=(1.0, 1.0)),) + clone.pads[1:])),
        ("外来 ref", replace(clone, ref="__zzz_foreign__")),
    ]
    for tag, brk in broken:
        lst = list(placed_ok)
        lst[t] = brk
        with pytest.raises(ValueError) as ei:
            gr._pose_pad_index(model, lst)
        msg = str(ei.value)
        print(f"\n  [{tag}] 已拒绝：{msg[:92]}")
        assert "_pose_pad_index" in msg, f"{tag}: 报错信息不可定位（{msg}）"

    lst = list(placed_ok)
    lst[t] = broken[0][1]
    with pytest.raises(ValueError):
        gr.hpwl_gain_maps(model, lst, now, spec)              # ② 端到端：G 入口也被闸住
