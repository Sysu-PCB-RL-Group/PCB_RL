# -*- coding: utf-8 -*-
"""把 ChiPFormer 的布局**搬进我们的坐标体系**，再用**我们的口径**给它算一遍（Owner 2026-09-24）。

为什么
------
ChiPFormer 只产出 HPWL（它的净集、它的画布）；我们项目 §10.2 的 4 个指标里 **σ 与 reward 算不出来**，
它的"出界/重叠"统计也与我们的判据**不同名不同意**。
本脚本把它的布局逐件搬进**我们的板坐标系**，然后用**我们的函数**重算：
`PcbModel.hpwl` / `grid_render.dispersion_of` / `violations.Judge` / §10.2 的 reward 式
⇒ 4 个指标第一次能在**同一套尺**下与专家、与我们并列。

坐标换算（**与 CF 自己的 `pcb_eval/export_kicad.py` 同一公式**）
---------------------------------------------------------------
```text
new_anchor_nm = ll_placement_frame + origin_nm + offset_nm
   · ll_placement_frame = 该件 body 左下角在**平移后帧**里的位置
       - 模型放过的件（`chipformer.macros[].phys_lower_left_nm` = grid × ratio）
       - 未搬动的件（top 口径的背层锚点）  = body_ll_orig_nm − origin_nm
   · offset_nm = fp_at_orig_nm − body_ll_orig_nm（预存在 layout_map.json 的 nodes 里）
```
**两条自检**（跑完先看 `selfcheck`）：
  A. **人类布局重建**：用同一公式重建 `kicad` 侧（人类）布局，应逐件等于我们模型的源锚点；
  B. **HPWL 对拍**：重建后的 CF 布局用**我们**的 HPWL 算一遍，与 CF 自报的 `hpwl_nm` 比 ——
     在"两侧净集一致"的板上应几乎相等（其余板净集不同，只能看量级）。

固定口径（改口径 = 改这里 + 重跑）
----------------------------------
* 未搬动的件保持**源位姿**（CF 也没动它们，`fixed_names`）；**旋转一律不动**
  （CF 的动作空间只有 (x, y)，`export_kicad` 明确保持 rot）；
* 禁布区走**唯一入口** `pcb_model.state_keepouts(model, placed)`（口径 A：槽随件）；
* 判定 = `Judge`（硬重叠 ∪ 软间距 ∪ 出界）；**CF 与专家两侧走同一条代码路径**；
* reward 用 §10.2 的式子，`n` 取**板上件数**（与 RL 侧同一 n ⇒ 两侧的 `ΔR` 与百分比都能并列）。
  ⚠️ `ΔR` 本身**与 n、与 β 无关**（两边 β·n 抵消），所以 headline 用 `ΔR`；
  `reward_gain_pct` 才依赖分母里的 `β·n`。
* HPWL / σ 一律用**我们的净集口径**（排除电源网）——这正是"用我们的尺读它的解"的含义。

用法
```text
& python scripts/cf_metrics_in_our_frame.py
& python scripts/cf_metrics_in_our_frame.py --cf-root D:\\ChiPFormer --out result/xxx/cf_in_our_frame.json
```

产物：`<result-dir>/cf_in_our_frame.json`（逐板 + 聚合）。**只处理有 P1 缓存的 16 板**
（`pcb07/13/14/15` 已整板排除，跳过并在产物里记明）。

⚠️ 本脚本**不写** `compare.json`（那是 `gen_chipformer_result.py` 的产物）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env import board_state as bs                      # noqa: E402
from env import grid_render as gr                      # noqa: E402
from env import reward as rw                           # noqa: E402
from env import violations as vio                      # noqa: E402
from utils import viz_layout as viz                    # noqa: E402
from utils.kicad_utils import model_cache as mc        # noqa: E402
from utils.kicad_utils import pcb_model as pm          # noqa: E402

DEFAULTS: Dict[str, Any] = {
    "cf_root": r"E:\PCB_AI\ChiPFormer",
    "cf_bench": "converted_top",
    "result_dir": os.path.join("result", "chipformer_result"),
    "rl_run": os.path.join("result", "infer_best_ep350"),
    "out": None,                       # 默认 <result_dir>/cf_in_our_frame.json
}


def _norm_ref(ref: str) -> str:
    """CF 的 ref → 我们的 ref：去掉合成前缀 `@` 与后缀 `#<序号>`（实测规则，见 `gen_chipformer_result`）。"""
    return ref[1:].split("#", 1)[0] if ref.startswith("@") else ref


def _r(v: Optional[float], n: int = 6) -> Optional[float]:
    return None if v is None else round(float(v), n)


def map_refs(model: pm.PcbModel, nodes: Dict[str, Any]) -> Dict[str, str]:
    """`{oN: 我们的 ref}`；**必须单射且覆盖全板**，否则抛错（宁可停，不猜）。"""
    ours = {_norm_ref(f.ref): f.ref for f in model.footprints}
    out: Dict[str, str] = {}
    for name, nd in nodes.items():
        r = ours.get(nd["ref"])
        if r is None:
            raise KeyError("CF 的 ref 在我们模型里找不到：%s（版 %s）" % (nd["ref"], name))
        out[name] = r
    if len(set(out.values())) != len(out):
        raise ValueError("ref 映射不是单射（有两条 CF 记录指向同一件）")
    if set(out.values()) != {f.ref for f in model.footprints}:
        raise ValueError("ref 映射覆盖不全（有我们模型里的件没被映射）")
    return out


def anchor_of(node: Dict[str, Any], ll_placement: Sequence[float],
              origin: Sequence[float]) -> Tuple[float, float]:
    """`ll_placement + origin + offset` → 我们坐标系里的锚点（mm）。"""
    ox, oy = origin
    return ((ll_placement[0] + ox + node["offset_nm"][0]) / 1e6,
            (ll_placement[1] + oy + node["offset_nm"][1]) / 1e6)


def judge_poses(model: pm.PcbModel, poses: Dict[str, Tuple[float, float, int]]) -> vio.ViolationResult:
    """整板判定（两侧共用这一条路径）：禁布区按"已放件"位姿化，再逐件判定。"""
    placed = bs.derive_placed(model, poses)
    j = vio.Judge(model, keepouts=pm.state_keepouts(model, placed))
    return j.judge_all({r: (float(x), float(y), int(rot)) for r, (x, y, rot) in poses.items()})


def enumerate_violations(model: pm.PcbModel, poses: Dict[str, Tuple[float, float, int]]
                         ) -> List[Dict[str, Any]]:
    """**逐条**列出违规（`Judge` 按设计只给计数、不给细节 —— 见 `env/violations.py` 开头拍案 2）。

    判定式**逐字照** `Judge.judge` + `judge_all`（硬重叠 A/B/C、软间距 S1/S2/S3、出界；
    含"每对只计一次"的 `r > ref`）。⚠️ 这是与 `Judge` **平行的第二条路径** ⇒
    调用方**必须**用计数对拍（见 `check_violation_counts`），不一致就抛错。
    为简洁**跳过** Judge 里"外扩软间距"的粗筛（那只是剪枝，不改变结果）。
    """
    byref = {f.ref: f for f in model.footprints}
    keepouts = pm.state_keepouts(model, bs.derive_placed(model, poses))
    valid = model.board_outline.valid_area(0.0)

    def _out(aabb: pm.AABB) -> bool:
        return aabb[0] < valid[0] or aabb[1] < valid[1] or aabb[2] > valid[2] or aabb[3] > valid[3]

    def _over(aabb: pm.AABB) -> float:
        """越出板框的最大量（mm）—— 0 表示没越界。"""
        return max(valid[0] - aabb[0], valid[1] - aabb[1], aabb[2] - valid[2], aabb[3] - valid[3])

    cand_of: Dict[str, List[Tuple[pm.Pad, pm.AABB, Tuple[int, ...]]]] = {}
    for ref, (x, y, rot) in poses.items():
        cand_of[ref] = [(p, pm.rotate_aabb(p.local_aabb, rot * 90.0, (x, y)), p.layers)
                        for p in byref[ref].pads]

    items: List[Dict[str, Any]] = []
    for ref, (x, y, _rot) in poses.items():
        # ① 出界（逐 pad 记一次）
        for p, aabb, _L in cand_of[ref]:
            if _out(aabb):
                over = _over(aabb)
                items.append({"kind": vio.OUT_OF_BOUNDS, "a": "%s.%s" % (ref, p.number),
                              "gap_mm": None, "over_mm": _r(over, 4), "severity": _r(-over, 4)})
        # ② pad ↔ 别的 footprint 的 pad（同一对只算一次：`oref > ref`）
        for oref in poses:
            if not oref > ref:
                continue
            for p, aabb, L in cand_of[ref]:
                for op, oaabb, oL in cand_of[oref]:
                    if not (set(L) & set(oL)):
                        continue
                    g = vio.gap(aabb, oaabb)
                    if g < 0.0:
                        items.append({"kind": vio.HARD_PAD_PAD, "a": "%s.%s" % (ref, p.number),
                                      "b": "%s.%s" % (oref, op.number), "gap_mm": _r(g, 4),
                                      "severity": _r(g, 4)})
                    elif g < vio.SOFT_GAP_MM:
                        items.append({"kind": vio.SOFT_PAD_PAD, "a": "%s.%s" % (ref, p.number),
                                      "b": "%s.%s" % (oref, op.number), "gap_mm": _r(g, 4),
                                      "severity": _r(g, 4)})
        # ③ pad ↔ via（硬重叠 B / 软间距 S2）
        for p, aabb, L in cand_of[ref]:
            if not p.has_net:
                continue
            for v in model.vias:
                if not v.has_net or v.netcode == p.netcode:
                    continue
                if not (set(L) & set(v.layers)):
                    continue
                g = vio.gap(aabb, v.aabb)
                if g < vio.SOFT_GAP_MM:
                    items.append({"kind": (vio.HARD_PAD_VIA if g < 0.0 else vio.SOFT_PAD_VIA),
                                  "a": "%s.%s" % (ref, p.number),
                                  "b": "via@(%.3f,%.3f)" % v.center, "gap_mm": _r(g, 4),
                                  "severity": _r(g, 4)})
        # ④ pad ↔ 禁布区（硬重叠 C / 软间距 S3；排除禁布区项所属 footprint 自己）
        for k in keepouts:
            if k.owner_ref == ref:
                continue
            for p, aabb, L in cand_of[ref]:
                if not (set(L) & set(k.layers)):
                    continue
                g = vio.gap(aabb, k.aabb)
                if g < vio.SOFT_GAP_MM:
                    who = "%s%s" % (k.kind, ("(%s)" % k.owner_ref) if k.owner_ref else "")
                    items.append({"kind": (vio.HARD_PAD_KEEPOUT if g < 0.0 else vio.SOFT_PAD_KEEPOUT),
                                  "a": "%s.%s" % (ref, p.number), "b": who, "gap_mm": _r(g, 4),
                                  "severity": _r(g, 4)})
    return items


def check_violation_counts(items: Sequence[Dict[str, Any]], res: vio.ViolationResult) -> None:
    """**对拍**：逐条枚举的计数必须与 `Judge` 完全一致（不等 ⇒ 抛错，绝不放行）。"""
    got: Dict[str, int] = {}
    for it in items:
        got[it["kind"]] = got.get(it["kind"], 0) + 1
    want = {k: int(v) for k, v in res.counts.items()}
    if got != want:
        raise AssertionError("逐条枚举与 Judge 计数不一致：枚举=%s vs Judge=%s" % (got, want))


def board_metrics(model: pm.PcbModel, cf_poses: Dict[str, Tuple[float, float, int]],
                  exp_poses: Dict[str, Tuple[float, float, int]]) -> Dict[str, Any]:
    """同一把尺下算 CF 与专家两侧的 ①HPWL ②σ ③④reward + 判定计数。"""
    refs = set(cf_poses)
    h_ref, d_board = float(rw.h_ref_of(model)), float(rw.d_board_of(model))
    beta, n = float(rw.beta_of(model)), len(model.footprints)

    hpwl_cf = float(model.hpwl(refs=refs, poses=cf_poses))
    hpwl_exp = float(model.hpwl(refs=refs, poses=exp_poses))
    sig_cf = float(gr.dispersion_of(model, bs.derive_placed(model, cf_poses)))
    sig_exp = float(gr.dispersion_of(model, bs.derive_placed(model, exp_poses)))

    r_cf = -hpwl_cf / h_ref + rw.ALPHA * sig_cf / d_board + beta * n
    r_exp = -hpwl_exp / h_ref + rw.ALPHA * sig_exp / d_board + beta * n
    j_cf, j_exp = judge_poses(model, cf_poses), judge_poses(model, exp_poses)
    return {
        "h_ref": _r(h_ref), "d_board": _r(d_board), "beta": _r(beta, 6), "n": n,
        "cf": {"hpwl_mm": _r(hpwl_cf), "sigma_mm": _r(sig_cf), "reward": _r(r_cf),
               "hpwl_drop_pct": _r((h_ref - hpwl_cf) / h_ref * 100.0, 4),
               "sigma_gain_pct": _r((sig_cf - sig_exp) / sig_exp * 100.0, 4) if sig_exp else None,
               "reward_delta": _r(r_cf - r_exp),
               "reward_gain_pct": _r((r_cf - r_exp) / abs(r_exp) * 100.0, 4) if r_exp else None,
               "judge": {"counts": dict(j_cf.counts), "describe": j_cf.describe()}},
        "expert": {"hpwl_mm": _r(hpwl_exp), "sigma_mm": _r(sig_exp), "reward": _r(r_exp),
                   "judge": {"counts": dict(j_exp.counts), "describe": j_exp.describe()}},
    }


def selfcheck_poses(model: pm.PcbModel, lmap: Dict[str, Any], lay: Dict[str, Any],
                    refs: Dict[str, str]) -> float:
    """自检 A：用 `kicad` 侧（人类）几何重建锚点 ⇒ 必须等于我们模型的源锚点；返回最大偏差 mm。"""
    by = {f.ref: f for f in model.footprints}
    worst = 0.0
    for m in lay["kicad"]["macros"]:
        nd = lmap["nodes"][m["name"]]
        x, y = anchor_of(nd, m["phys_lower_left_nm"], lmap["origin_nm"])
        src = by[refs[m["name"]]]
        worst = max(worst, abs(x - src.anchor[0]), abs(y - src.anchor[1]))
    return worst


def run_board(pcb: str, kdir: str, *, cf_root: str, bench: str, layouts: str,
              rl_boards: Dict[str, Any], n_top: int = 0) -> Dict[str, Any]:
    """单板：读 CF 侧布局 → 搬进我们坐标系 → 算指标（`n_top>0` 时另列最严重的 `n_top` 条违规）。"""
    model = mc.load_model(kdir, allow_build=False)
    lmap = json.load(open(os.path.join(cf_root, bench, pcb, "layout_map.json"), encoding="utf-8"))
    lay = json.load(open(os.path.join(layouts, "%s_layout.json" % pcb), encoding="utf-8"))
    nodes, origin = lmap["nodes"], lmap["origin_nm"]
    refs = map_refs(model, nodes)
    by = {f.ref: f for f in model.footprints}
    fixed = set(lay["fixed_names"])
    moved = {m["name"]: m for m in lay["chipformer"]["macros"]}

    # ---- 位姿：搬过的用新锚点，没搬的保持源位姿（两者都走同一公式） ----
    cf_poses: Dict[str, Tuple[float, float, int]] = {}
    for name, nd in nodes.items():
        ref = refs[name]
        src = by[ref]
        rot = int(src.rotation_index)
        if name in fixed:
            ll = [nd["body_ll_orig_nm"][0] - origin[0], nd["body_ll_orig_nm"][1] - origin[1]]
        else:
            ll = moved[name]["phys_lower_left_nm"]
        x, y = anchor_of(nd, ll, origin)
        cf_poses[ref] = (x, y, rot)

    exp_poses = {f.ref: (float(f.anchor[0]), float(f.anchor[1]), int(f.rotation_index))
                 for f in model.footprints}
    met = board_metrics(model, cf_poses, exp_poses)

    # ---- 可选：逐条列出 CF 侧违规（与 Judge 计数对拍后才算数） ----
    viol_top: List[Dict[str, Any]] = []
    if n_top:
        items = enumerate_violations(model, cf_poses)
        check_violation_counts(items, judge_poses(model, cf_poses))     # ★ 不过拍就抛错
        viol_top = sorted(items, key=lambda d: d["severity"])[:int(n_top)]

    # ---- 自检 B：我们算的 HPWL vs CF 自报的 HPWL（净集一致时应当相等） ----
    cf_reported = lay["chipformer"]["hpwl_nm"] / 1e6
    delta_pct = (met["cf"]["hpwl_mm"] - cf_reported) / cf_reported * 100.0 if cf_reported else None
    rl_row = rl_boards.get(kdir)
    c_cnt = met["cf"]["judge"]["counts"]
    e_cnt = met["expert"]["judge"]["counts"]
    return {
        "board_id": pcb, "kicad_dir": kdir,
        "n_parts": len(nodes), "n_moved_by_cf": len(moved), "n_fixed": len(fixed),
        # CF 相对**专家**新增的违规（专家自己就有既存违规，必须减掉才算"CF 引入的"）
        "judge_delta_vs_expert": {k: int(c_cnt.get(k, 0) - e_cnt.get(k, 0))
                                  for k in sorted(set(c_cnt) | set(e_cnt))},
        # 搬进我们坐标系后的位姿表 `ref → [x, y, rot]`（供可视化直接用，免得下游重算映射）
        "cf_poses": {r: [float(x), float(y), int(k)] for r, (x, y, k) in cf_poses.items()},
        **({"violations_top": viol_top} if n_top else {}),
        "selfcheck": {"human_rebuild_max_abs_mm": _r(selfcheck_poses(model, lmap, lay, refs), 9),
                      "our_hpwl_vs_cf_reported_pct": _r(delta_pct, 4),
                      "net_set_same_as_ours": bool(rl_row and abs(
                          rl_row["hpwl_expert"] - lay["kicad"]["hpwl_nm"] / 1e6) <= 0.005)},
        "ours_frame": met,
        "cf_reported": {"hpwl_mm": _r(cf_reported),
                        "hpwl_drop_pct_their_scale": _r(
                            (lay["kicad"]["hpwl_nm"] - lay["chipformer"]["hpwl_nm"])
                            / lay["kicad"]["hpwl_nm"] * 100.0, 4),
                        "out_of_board_parts": len(lay["chipformer"]["out_of_board"]),
                        "overlap_pairs_frame": len(lay["chipformer"]["overlaps"])},
        "provenance": {"layout": "%s/%s_layout.json" % (layouts.replace(os.sep, "/"), pcb),
                       "layout_map": os.path.join(cf_root, bench, pcb, "layout_map.json")},
    }


def _agg(rows: Sequence[Dict[str, Any]], path: Sequence[str]) -> Dict[str, Any]:
    vals = []
    for r in rows:
        v: Any = r
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            vals.append(float(v))
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": _r(st.mean(vals), 4), "median": _r(st.median(vals), 4),
            "min": _r(min(vals), 4), "max": _r(max(vals), 4)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="把 CF 布局搬进我们的坐标系并用我们的口径重算")
    p.add_argument("--cf-root", default=DEFAULTS["cf_root"])
    p.add_argument("--cf-bench", default=DEFAULTS["cf_bench"])
    p.add_argument("--result-dir", default=DEFAULTS["result_dir"])
    p.add_argument("--rl-run", default=DEFAULTS["rl_run"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--violations", type=int, default=0,
                   help="额外**逐条**列出 CF 侧最严重的 N 条违规（进产物 + 打印；默认 0 = 不列）")
    args = p.parse_args(argv)

    cf_root = os.path.abspath(args.cf_root)
    result_dir = os.path.abspath(args.result_dir)
    layouts = os.path.join(result_dir, "source", "layouts")
    out = os.path.abspath(args.out or os.path.join(result_dir, "cf_in_our_frame.json"))

    batch: List[Tuple[str, str]] = []
    for ln in open(os.path.join(result_dir, "source", "batch_all20.txt"), encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            n, q = ln.split("|")
            batch.append((n, os.path.basename(os.path.dirname(q))))
    rl_boards = {r["board"]: r for r in
                 json.load(open(os.path.join(os.path.abspath(args.rl_run), "summary.json"),
                                encoding="utf-8"))["boards"]}

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for pcb, kdir in batch:
        if kdir not in rl_boards:                     # 无 P1 缓存 ⇒ 整板排除的 4 板
            skipped.append(pcb)
            continue
        rows.append(run_board(pcb, kdir, cf_root=cf_root, bench=args.cf_bench,
                              layouts=layouts, rl_boards=rl_boards,
                              n_top=int(args.violations)))

    worst = max(r["selfcheck"]["human_rebuild_max_abs_mm"] for r in rows)
    same = [r for r in rows if r["selfcheck"]["net_set_same_as_ours"]]
    doc = {
        "schema": "cf_in_our_frame/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "convention": {
            "anchor_formula": "ll_placement_frame + origin_nm + offset_nm（与 CF export_kicad 同式）",
            "rotation": "保持源旋转（CF 只动 (x, y)）",
            "net_filter": "pcb_model.is_power_net（四级判据）；HPWL/σ 均排除电源网",
            "keepouts": "pcb_model.state_keepouts(model, placed)（口径 A，槽随件）",
            "judge": "violations.Judge（硬重叠 ∪ 软间距 ∪ 出界）；CF 与专家同一路径",
            "reward_n": "n = 板上件数（与 RL 侧同 n）；ΔR 与 n、β 无关",
            "alpha": rw.ALPHA,
        },
        "selfcheck": {
            "human_rebuild_max_abs_mm": _r(worst, 9),
            "same_net_set_boards": [r["board_id"] for r in same],
            "hpwl_vs_cf_reported_pct_same_net": _agg(same, ["selfcheck", "our_hpwl_vs_cf_reported_pct"]),
            "hpwl_vs_cf_reported_pct_all": _agg(rows, ["selfcheck", "our_hpwl_vs_cf_reported_pct"]),
        },
        "n_boards": len(rows), "skipped": skipped,
        "boards": rows,
        "aggregate": {
            "cf_in_our_frame": {
                "hpwl_drop_pct": _agg(rows, ["ours_frame", "cf", "hpwl_drop_pct"]),
                "sigma_gain_pct": _agg(rows, ["ours_frame", "cf", "sigma_gain_pct"]),
                "reward_gain_pct": _agg(rows, ["ours_frame", "cf", "reward_gain_pct"]),
                "reward_delta": _agg(rows, ["ours_frame", "cf", "reward_delta"]),
            },
            "expert": {"judge_hit_boards": int(sum(
                1 for r in rows if r["ours_frame"]["expert"]["judge"]["counts"]))},
            "cf_judge_hit_boards": int(sum(
                1 for r in rows if r["ours_frame"]["cf"]["judge"]["counts"])),
            # 逐方向把"CF 相对专家新增的违规"求和（正 = CF 引入）
            "judge_delta_total": {k: int(sum(r["judge_delta_vs_expert"].get(k, 0) for r in rows))
                                  for k in sorted(set().union(
                                      *[set(r["judge_delta_vs_expert"]) for r in rows]))},
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)

    print("== 自检 A：人类布局重建最大偏差 = %.3e mm（应 ≈ 0）" % worst)
    print("== 自检 B：我们算的 HPWL vs CF 自报（同净集的 %d 板）mean=%s%% median=%s%%"
          % (len(same), doc["selfcheck"]["hpwl_vs_cf_reported_pct_same_net"]["mean"],
             doc["selfcheck"]["hpwl_vs_cf_reported_pct_same_net"]["median"]))
    print("%-18s %8s %8s %8s %9s %10s %s" % ("pcb", "d%_ours", "d%_cf", "s%_gain",
                                             "dR", "j_cf", "j_exp"))
    for r in rows:
        c = r["ours_frame"]["cf"]
        print("%-18s %8.2f %8.2f %8.2f %9.3f %10s %s"
              % (r["board_id"], c["hpwl_drop_pct"], r["cf_reported"]["hpwl_drop_pct_their_scale"],
                 c["sigma_gain_pct"], c["reward_delta"], c["judge"]["describe"],
                 r["ours_frame"]["expert"]["judge"]["describe"]))
    print("\n[聚合] CF 在我们尺下：HPWL 降低%% mean=%s 中位=%s｜σ 提升%% mean=%s｜ΔR mean=%s"
          % (doc["aggregate"]["cf_in_our_frame"]["hpwl_drop_pct"]["mean"],
             doc["aggregate"]["cf_in_our_frame"]["hpwl_drop_pct"]["median"],
             doc["aggregate"]["cf_in_our_frame"]["sigma_gain_pct"]["mean"],
             doc["aggregate"]["cf_in_our_frame"]["reward_delta"]["mean"]))
    print("[产物] %s（跳过 %d 板：%s）" % (out, len(skipped), ",".join(skipped)))

    if int(args.violations) > 0:
        print("\n== CF 侧最严重违规（逐条；与 `Judge` 计数已对拍通过）")
        print("  %-18s %-18s %-22s %-22s %s" % ("板", "类型", "对象 A", "对象 B", "量值"))
        for r in rows:
            for v in r.get("violations_top", []):
                val = ("gap %.4f mm" % v["gap_mm"] if v["gap_mm"] is not None
                       else "越出板框 %.4f mm" % v["over_mm"])
                print("  %-18s %-18s %-22s %-22s %s"
                      % (r["board_id"], v["kind"], v["a"], v.get("b", ""), val))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
