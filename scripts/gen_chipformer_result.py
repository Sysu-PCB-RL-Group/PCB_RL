# -*- coding: utf-8 -*-
"""跨项目对比产物：把 ChiPFormer（本地克隆）的结果搬成统一 JSON（Owner 2026-09-24）。

为什么有这条脚本
----------------
ChiPFormer 仓库就在本机（默认 `E:\\PCB_AI\\ChiPFormer`），**不需要人工跨仓库搬运**；
本脚本只读两侧产物 → 落 `<out>/compare.json` + 来源快照 `<out>/source/`（每个数字都能指回去）。

口径（**改口径就得改这里 + 重生成**）
------------------------------------
```text
· CF 侧：默认 route = top（背层件钉为锚点，模型只放顶层）；20 板；原生单位 nm
· RL 侧：`result/infer_<tag>/`（纯贪心 argmax，mask = L）；16 板；原生单位 mm
· 长度统一 mm（CF 侧 ÷1e6）；百分比统一「**正 = 更好**」（Owner 2026-09-23 拍）：
      HPWL 降低%   = (基线 − 模型) / 基线 × 100       基线 = 人类源位姿布局
      σ   提升%    = (模型 − 基线) / 基线 × 100
      reward 提升% = (模型 − 基线) / |基线| × 100     reward = 一局累计，与训练同式
· 板 ID = CF 侧 `pcbNN_slug`；join key = `converters/batch_all20.txt`（板 ID ↔ KiCad 目录名）
· RL 侧 4 板（pcb07/13/14/15）已整板排除 ⇒ 该侧全 null + `missing_reasons`
```

产物结构（写进 `compare.json`）
--------------------------------
```text
schema / generated_at / run_tag / board_set_main / units / notes
sides.rl / sides.cf     两侧运行信息（权重、批量、净过滤、原单位、转录项）
boards[20]              逐板：board_id / kicad_dir / n_movable / signal_net_cnt / ref_map
                              / sides.{rl,cf} / comparability{level, reasons}
  sides.<侧>            model / baseline / delta_pct / validity / missing_reasons / provenance
aggregate               paired_16（主口径）/ cf_20 / rl_16，每项 = n / mean / median / min / max
source/                 CF 侧来源文件**原样快照**（`provenance` 全部指向这里或 RL 产物目录）
```

⚠️ **读结论前必看**：两侧**净集口径不同**（实测 12/16 板的人类基线 HPWL 不一致，
比值 0.77~1.32）⇒ 两侧 `delta_pct` **不是同一把尺**，只能看方向与量级，**不可**直接对拼。
本脚本会把这件事逐板写进 `comparability.reasons`（不靠人记）：
`baseline_hpwl_mismatch` / `cf_top_route_pins_back_parts` / `validity_rule_mismatch`。

用法
```text
& python scripts/gen_chipformer_result.py
& python scripts/gen_chipformer_result.py --rl-run result/infer_best_ep100 --out result/xxx_vs_ep100
& python scripts/gen_chipformer_result.py --cf-bench converted_final --cf-eval pcb_eval/results_ref
```

⚠️ **`--cf-bench` / `--cf-eval` 必须指向同一 route 的完整产物**：
必需件 = `{bench}/baseline_hpwl.json`、`{eval}/zeroshot_results.json`、`{eval}/comparison.json`
（缺一件**直接报错**）；`{eval}/layouts/` 缺失则 `validity` 全 `null` + 原因（不猜 0）。
例：`pcb_eval/results_ref/` 目前只有 2 个文件 ⇒ **用不了**，换 route 得先在 CF 侧补齐产物。

⚠️ **本脚本不写 README.md**（人读说明是手写的，重生成不得覆盖）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.kicad_utils import model_cache as mc          # noqa: E402

DEFAULTS: Dict[str, Any] = {
    "cf_root": r"E:\PCB_AI\ChiPFormer",
    "cf_bench": "converted_top",                 # top 口径；换 ref 见 docstring
    "cf_eval": "pcb_eval/results_top",
    "rl_run": os.path.join("result", "infer_best_ep350"),
    "out": os.path.join("result", "chipformer_result"),
}

#: **必需**来源（缺一件 ⇒ 直接报错，不做"猜"）：`compare.json` 的骨架全靠这三份
SRC_ESSENTIAL: Dict[str, str] = {
    "batch_all20.txt": "converters/batch_all20.txt",
    "baseline_hpwl.json": "{bench}/baseline_hpwl.json",
    "zeroshot_results.json": "{eval}/zeroshot_results.json",
    "comparison.json": "{eval}/comparison.json",
}

#: **可选**来源（只进 `source/` 快照，不参与计算；缺了只告警）
SRC_OPTIONAL: Dict[str, str] = {
    "summary_final.json": "converted_final/summary_final.json",
    "power_filter.json": "converted/power_filter.json",
    "zeroshot_table.md": "{eval}/zeroshot_table.md",
    "comparison.md": "{eval}/comparison.md",
    "verification.md": "{eval}/layouts/verification.md",
    "README_eval.md": "pcb_eval/README_eval.md",
}

#: CF 运行信息里**CF 侧 JSON 没记、只能人工转录**的项（出处 = `pcb_eval/README_eval.md` §7 的命令行）
CF_TRANSCRIBED: Dict[str, Any] = {"repeat": 3, "device": "cpu", "seed": 42, "pyhashseed": 42}

_NULL_MODEL = {"hpwl": None, "sigma": None, "reward": None, "n_placed": None,
               "completion": None, "seconds": None, "stopped_reason": None}
_NULL_BASE = {"hpwl": None, "sigma": None, "reward": None, "n": None, "label": None}
_NULL_DELTA = {"hpwl": None, "sigma": None, "reward": None}
_NULL_VALIDITY = {"out_of_board": None, "hard_overlap": None, "soft_violation": None,
                  "checked_by": None}


def _r(v: Optional[float], n: int) -> Optional[float]:
    """四舍五入到 `n` 位（`None` 透传）。"""
    return None if v is None else round(float(v), n)


def _drop_pct(model: Optional[float], base: Optional[float]) -> Optional[float]:
    """HPWL 降低%（**正 = 模型更短**）；基线为 0 / 缺失 ⇒ `None`（不猜）。"""
    if not model or not base:
        return None
    return _r((float(base) - float(model)) / float(base) * 100.0, 4)


def _gain_pct(model: Optional[float], base: Optional[float]) -> Optional[float]:
    """σ / reward 提升%（**正 = 模型更大**）。"""
    if model is None or not base:
        return None
    return _r((float(model) - float(base)) / abs(float(base)) * 100.0, 4)


def _blank_side() -> Dict[str, Any]:
    """一个"这一侧没有数据"的 side 骨架（字段全留、值全 `None`）。"""
    return {"model": dict(_NULL_MODEL), "baseline": dict(_NULL_BASE),
            "delta_pct": dict(_NULL_DELTA), "validity": dict(_NULL_VALIDITY),
            "missing_reasons": {}, "provenance": {}}


def parse_batch(path: str) -> List[Tuple[str, str]]:
    """读 `batch_all20.txt` → `[(板 ID, kicad 目录名)]`（保序；`#` 注释跳过）。"""
    out: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            name, p = ln.split("|")
            out.append((name, os.path.basename(os.path.dirname(p))))
    return out


def copy_sources(cf_root: str, bench: str, evald: str, src: str) -> Tuple[int, List[str]]:
    """把 CF 侧来源**原样**拷进 `<out>/source/`（含逐板 `layouts/*_layout.json`）。

    **必需件缺一件就报错**（宁可停下，也不拿半份产物去算指标）；可选件缺了只告警。

    Returns: `(拷进来的 layouts 文件数, 缺失的可选件说明列表)`。
    """
    miss: List[str] = []
    os.makedirs(src, exist_ok=True)          # ★ 新 `--out` 时 `source/` 还不存在，必须先建
    for local, rel in SRC_ESSENTIAL.items():
        p = os.path.join(cf_root, rel.format(bench=bench, eval=evald).replace("/", os.sep))
        if not os.path.isfile(p):
            raise FileNotFoundError("CF 侧必需来源缺失：%s（带 --cf-bench/--cf-eval 指向完整产物目录）" % p)
        shutil.copy2(p, os.path.join(src, local))
    os.makedirs(os.path.join(src, "layouts"), exist_ok=True)
    for local, rel in SRC_OPTIONAL.items():
        p = os.path.join(cf_root, rel.format(bench=bench, eval=evald).replace("/", os.sep))
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(src, local))
        else:
            miss.append(rel.format(bench=bench, eval=evald))
    lay_dir = os.path.join(cf_root, evald.replace("/", os.sep), "layouts")
    n = 0
    if os.path.isdir(lay_dir):
        for f in sorted(os.listdir(lay_dir)):
            if f.endswith("_layout.json"):
                shutil.copy2(os.path.join(lay_dir, f), os.path.join(src, "layouts", f))
                n += 1
    else:
        miss.append(evald + "/layouts/（越界与重叠计数 ⇒ validity 全 null）")
    return n, miss


def rl_refs_by_dir() -> Dict[str, List[str]]:
    """RL 侧逐板 `ref` 列表（从 P1 缓存读，不需要 KiCad）—— 用来建 `ref_map`。"""
    out: Dict[str, List[str]] = {}
    for e in mc.read_index()["boards"]:
        out[e["board"]] = [f.ref for f in mc.load_model(e["board"], allow_build=False).footprints]
    return out


def build_ref_map(cf_refs: Sequence[str], rl_refs: Sequence[str]) -> Dict[str, str]:
    """CF `ref → RL ref` 的映射（两侧逐字相等 ⇒ `{}`）。

    实测规则：RL 对匿名件合成 `@<库名>#<序号>`，CF 直接用 `<库名>`
    ⇒ 去掉 `@` 前缀与 `#序号` 后缀即同一件；**必须双射**（否则返回 `{}` 并让人工看）。
    """
    if sorted(cf_refs) == sorted(rl_refs):
        return {}
    norm: Dict[str, List[str]] = {}
    for x in rl_refs:
        norm.setdefault(x[1:].split("#", 1)[0] if x.startswith("@") else x, []).append(x)
    m: Dict[str, str] = {}
    for c in cf_refs:
        v = norm.get(c, [])
        if len(v) == 1:
            m[c] = v[0]
    ok = len(m) == len(cf_refs) and len(set(m.values())) == len(m)
    return m if ok else {}


def cf_side(pcb: str, bl: Dict[str, Any], zs: Dict[str, Any], lay: Optional[Dict[str, Any]],
            cmp_: Dict[str, Any], route: str) -> Dict[str, Any]:
    """CF 单板 side（长度已 mm 化；百分比按落盘值算 ⇒ 自洽）。

    `lay is None`（CF 结果目录没有逐板 `layouts/*_layout.json`）⇒ `validity` 全 `null` + 原因，
    **不猜 0**。
    """
    base_mm = _r(bl["physical_hpwl_nm"] / 1e6, 6)
    cf_mm = _r(zs["hpwl_nm"] / 1e6, 6)
    s = _blank_side()
    s["model"] = {"hpwl": cf_mm, "sigma": None, "reward": None,
                  "n_placed": int(zs["placed_macros"]), "completion": None,
                  "seconds": _r(zs["seconds"], 1), "stopped_reason": "all_placed"}
    s["baseline"] = {"hpwl": base_mm, "sigma": None, "reward": None,
                     "n": int(bl["macro_cnt"]), "label": "kicad_human_source_pose"}
    s["delta_pct"] = {"hpwl": _drop_pct(cf_mm, base_mm), "sigma": None, "reward": None}
    if lay is None:
        s["validity"] = {"out_of_board": None, "hard_overlap": None, "soft_violation": None,
                         "checked_by": "CF：结果目录无 layouts/，未产出计数"}
    else:
        s["validity"] = {
            "out_of_board": len(lay["chipformer"]["out_of_board"]),
            "hard_overlap": len(lay["chipformer"]["overlaps"]),
            "soft_violation": None,
            "checked_by": "CF：元件外形框 ∩ Edge.Cuts / 外形框相交（**非 pad 级**）",
        }
    s["missing_reasons"] = {
        "sigma": "ChiPFormer 不产出 σ",
        "reward": "CF 的 rollout_score 与 RL reward 不同式，不可换算",
        "completion": "CF 画布无板框，不做完成度口径",
        "soft_violation": "CF 无软间距口径",
        "hard_overlap": "同名不同义：CF 用元件外形框，RL 用 pad AABB",
    }
    if lay is None:
        s["missing_reasons"]["out_of_board"] = "CF 结果目录无 layouts/，未产出越界计数"
        s["missing_reasons"]["hard_overlap"] = "CF 结果目录无 layouts/，未产出重叠计数"
    s["extras"] = {"layout_area_ratio": _r(zs["layout_area_ratio"], 6),
                   "rollout_score": _r(zs["rollout_score"], 6),
                   "action_cnt": int(zs["action_cnt"]),
                   "cf_movable_ref": int(cmp_["movable_ref"]),
                   "cf_movable_top": int(cmp_["movable_top"]),
                   "cf_back_parts": int(cmp_["back_parts"])}
    s["provenance"] = {
        "model": {"file": "source/zeroshot_results.json", "id_key": "board",
                  "field_map": {"hpwl": "hpwl_nm（÷1e6 → mm）", "n_placed": "placed_macros",
                                "seconds": "seconds"}},
        "baseline": {"file": "source/baseline_hpwl.json", "id_key": "board",
                     "field_map": {"hpwl": "physical_hpwl_nm（÷1e6 → mm）", "n": "macro_cnt"}},
        "validity": ({} if lay is None else
                     {"file": "source/layouts/%s_layout.json" % pcb, "id_key": "board",
                      "field_map": {"out_of_board": "chipformer.out_of_board（长度）",
                                    "hard_overlap": "chipformer.overlaps（长度）"}}),
    }
    s["route"] = route
    return s


def rl_side(row: Optional[Dict[str, Any]], kdir: str, reason: str) -> Dict[str, Any]:
    """RL 单板 side；`row is None`（该板不在 RL 数据集）⇒ 全 `None` + 一句原因。"""
    s = _blank_side()
    if row is None:
        s["missing_reasons"] = {"*": reason}
        return s
    h_m, h_b = _r(row["hpwl_model"], 6), _r(row["hpwl_expert"], 6)
    s_m, s_b = _r(row["sigma_model"], 6), _r(row["sigma_expert"], 6)
    rw_m, rw_b = _r(row["reward_model"], 6), _r(row["reward_expert"], 6)
    s["model"] = {"hpwl": h_m, "sigma": s_m, "reward": rw_m, "n_placed": int(row["n_placed"]),
                  "completion": _r(row["completion"], 6), "seconds": _r(row["seconds"], 1),
                  "stopped_reason": row["stopped_reason"]}
    s["baseline"] = {"hpwl": h_b, "sigma": s_b, "reward": rw_b, "n": int(row["n_total"]),
                     "label": "expert_source_pose"}
    s["delta_pct"] = {"hpwl": _drop_pct(h_m, h_b), "sigma": _gain_pct(s_m, s_b),
                      "reward": _gain_pct(rw_m, rw_b)}
    s["validity"] = {
        "out_of_board": None, "hard_overlap": None, "soft_violation": None,
        "checked_by": "RL：pad AABB 对板框 / pad↔pad 同层不同件（`env.Judge`；逐板计数未落盘）",
    }
    s["missing_reasons"] = {
        "validity.out_of_board": "推理产物只落汇总口径（16/16 零违规），逐板未落盘",
        "validity.hard_overlap": "同上",
        "validity.soft_violation": "同上",
    }
    s["provenance"] = {
        "model": {"file": "<rl_run>/%s/%s.json" % (kdir, kdir), "id_key": "board",
                  "field_map": {"hpwl": "hpwl_model", "sigma": "sigma_model",
                                "reward": "reward_model", "n_placed": "n_placed",
                                "completion": "completion", "seconds": "seconds"}},
        "baseline": {"file": "<rl_run>/%s/%s.json" % (kdir, kdir), "id_key": "board",
                     "field_map": {"hpwl": "hpwl_expert", "sigma": "sigma_expert",
                                   "reward": "reward_expert", "n": "n_total"}},
    }
    return s


def _agg(rows: Sequence[Dict[str, Any]], side: str, key: str) -> Dict[str, Optional[float]]:
    """逐板 `delta_pct` 的 n / mean / median / min / max（**先逐板算、再聚合**；`None` 跳过）。"""
    vals = []
    for r in rows:
        v = r["sides"][side]["delta_pct"][key] if key in ("hpwl", "sigma", "reward") \
            else r["sides"][side]["model"][key]
        if v is not None:
            vals.append(float(v))
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": _r(st.mean(vals), 4), "median": _r(st.median(vals), 4),
            "min": _r(min(vals), 4), "max": _r(max(vals), 4)}


def aggregate(boards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """`paired_16`（主口径）+ `cf_20` / `rl_16`（参考）；每集只出现真正有数据的侧。"""
    paired = [b for b in boards if b["sides"]["rl"]["missing_reasons"].get("*") is None]
    metrics = ("hpwl", "sigma", "reward")

    def _rl(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {**{("hpwl_delta_pct" if k == "hpwl" else k + "_gain_pct"): _agg(rows, "rl", k)
                   for k in metrics},
                "completion": _agg(rows, "rl", "completion")}

    return {
        "paired_16": {"rl": _rl(paired), "cf": {"hpwl_delta_pct": _agg(paired, "cf", "hpwl")}},
        "cf_20": {"cf": {"hpwl_delta_pct": _agg(boards, "cf", "hpwl")}},
        "rl_16": {"rl": _rl(boards)},
    }


def build(*, cf_root: str, cf_bench: str, cf_eval: str, rl_run: str, out: str,
          verbose: bool = True) -> Dict[str, Any]:
    """读两侧产物 → 写 `<out>/compare.json` + `<out>/source/`；返回落盘的那个 dict。"""
    cf_root, rl_run, out = os.path.abspath(cf_root), os.path.abspath(rl_run), os.path.abspath(out)
    src = os.path.join(out, "source")
    os.makedirs(out, exist_ok=True)
    n_lay, miss = copy_sources(cf_root, cf_bench, cf_eval, src)

    batch = parse_batch(os.path.join(src, "batch_all20.txt"))
    bl = json.load(open(os.path.join(src, "baseline_hpwl.json"), encoding="utf-8"))
    zs = {r["board"]: r for r in
          json.load(open(os.path.join(src, "zeroshot_results.json"), encoding="utf-8"))["results"]}
    cmp_ = {r["board"]: r for r in json.load(open(os.path.join(src, "comparison.json"),
                                                  encoding="utf-8"))}
    lay: Dict[str, Any] = {}
    for b, _ in batch:
        p = os.path.join(src, "layouts", "%s_layout.json" % b)
        if os.path.isfile(p):
            lay[b] = json.load(open(p, encoding="utf-8"))
    summary = json.load(open(os.path.join(rl_run, "summary.json"), encoding="utf-8"))
    rows = {r["board"]: r for r in summary["boards"]}
    refs = rl_refs_by_dir()
    route = "top" if cf_bench.endswith("top") else "ref"

    boards: List[Dict[str, Any]] = []
    n_paired = 0
    for pcb, kdir in batch:
        row = rows.get(kdir)
        in_rl = row is not None
        cf = cf_side(pcb, bl[pcb], zs[pcb], lay.get(pcb), cmp_[pcb], route)
        rl = rl_side(row, kdir, "RL 数据集已整板排除（4 块结构性不可放件板，见台账 §0.2a）")
        rl["provenance"] = {k: {"file": v["file"].replace("<rl_run>",
                                                           os.path.relpath(rl_run, _ROOT)
                                                           .replace(os.sep, "/")),
                                "id_key": v["id_key"], "field_map": v["field_map"]}
                            for k, v in (rl["provenance"] or {}).items()}

        reasons = ["validity_rule_mismatch"]
        if in_rl:
            n_paired += 1
            if abs(rl["baseline"]["hpwl"] - cf["baseline"]["hpwl"]) > 0.005:
                reasons.append("baseline_hpwl_mismatch")
            if cf["extras"]["cf_movable_top"] != cf["extras"]["cf_movable_ref"]:
                reasons.append("cf_top_route_pins_back_parts")
            level = "directional_only"
        else:
            reasons = ["rl_board_excluded_bad_data"]
            level = "not_comparable"

        n_rl = int(row["n_total"]) if in_rl else None
        if n_rl is not None and n_rl != int(bl[pcb]["macro_cnt"]):
            print("[WARN] 件数两侧不等：%s RL=%d CF=%d" % (pcb, n_rl, bl[pcb]["macro_cnt"]))
        boards.append({
            "board_id": pcb, "kicad_dir": kdir,
            "kicad_path": "%s/%s/%s.kicad_pcb" % (
                "data/kicad-native-batch" if in_rl else "data/bad_data", kdir, kdir),
            "in_rl_dataset": bool(in_rl),
            "n_movable": {"rl": n_rl, "cf": int(bl[pcb]["macro_cnt"])},
            "signal_net_cnt": {"rl": None, "cf": int(bl[pcb]["net_cnt"])},
            "ref_map": {"cf_to_rl": build_ref_map(
                [m["ref"] for m in lay[pcb]["kicad"]["macros"]] if pcb in lay else [],
                refs.get(kdir, []))},        # layouts 缺失 ⇒ 空映射（两侧 ref 无法逐件对齐）
            "sides": {"rl": rl, "cf": cf},
            "comparability": {"level": level, "reasons": reasons},
        })

    doc: Dict[str, Any] = {
        "schema": "pcb_compare/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": os.path.basename(out),
        "board_set_main": "paired_16",
        "units": {"length": "mm", "percent": "%"},
        "sides": {
            "rl": {"tool": "scripts/infer_dqn.py",
                   "ckpt": os.path.relpath(str(summary["ckpt"]), _ROOT).replace(os.sep, "/"),
                   "episode": summary.get("ckpt_episode"),
                   "n_boards": int(summary["n_boards"]),
                   "run_dir": os.path.relpath(rl_run, _ROOT).replace(os.sep, "/"),
                   "policy": summary.get("policy"), "mask_mode": summary.get("mask_mode"),
                   "elapsed_s": summary.get("elapsed_s"),
                   "dataset_root": "data/kicad-native-batch",
                   "cache_index": "data/model_cache/index.json",
                   "net_filter": "pcb_model.is_power_net（四级判据）",
                   "native_unit": "mm"},
            "cf": {"tool": "pcb_eval.run_zeroshot", "route": route,
                   "bench_root": cf_bench, "eval_dir": cf_eval,
                   "weight": json.load(open(os.path.join(src, "zeroshot_results.json"),
                                            encoding="utf-8")).get("weight"),
                   "n_boards": len(batch),
                   "net_filter": "converted/power_filter.json",
                   "native_unit": "nm",
                   "transcribed": sorted(CF_TRANSCRIBED),
                   "transcribed_from": "source/README_eval.md §7（CF 侧 JSON 未记，属人工转录）",
                   **CF_TRANSCRIBED},
        },
        "notes": "主口径 = paired_16；CF route=%s（背层件钉为锚点，只放顶层）。"
                 "两侧净集口径不同 ⇒ `delta_pct` 不可直接对拼，逐板见 comparability.reasons。"
                 % route,
        "boards": boards,
        "aggregate": aggregate(boards),
    }
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    if verbose:
        a = doc["aggregate"]
        print("boards=%d paired=%d source=%d+layouts=%d → %s"
              % (len(boards), n_paired, len(SRC_ESSENTIAL) + len(SRC_OPTIONAL), n_lay,
                 os.path.join(out, "compare.json")))
        for m in miss:
            print("  [WARN] 可选来源缺失（只影响 source/ 快照）：%s" % m)
        for k in ("paired_16", "cf_20", "rl_16"):
            for side in ("rl", "cf"):
                if side in a[k]:
                    print("  %-9s %-2s HPWL 降低%% mean=%s median=%s (n=%s)"
                          % (k, side, a[k][side]["hpwl_delta_pct"]["mean"],
                             a[k][side]["hpwl_delta_pct"]["median"],
                             a[k][side]["hpwl_delta_pct"]["n"]))
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="把 ChiPFormer 结果搬成统一对比 JSON（两侧逐板并列）")
    p.add_argument("--cf-root", default=DEFAULTS["cf_root"], help="ChiPFormer 克隆根目录")
    p.add_argument("--cf-bench", default=DEFAULTS["cf_bench"],
                   help="CF 数据集：converted_top（默认，route=top）/ converted_final（route=ref）")
    p.add_argument("--cf-eval", default=DEFAULTS["cf_eval"],
                   help="CF 结果目录（默认 results_top）；⚠️ 必须是**完整**产物目录："
                        "缺 comparison.json 会直接报错，缺 layouts/ 则 validity 全 null")
    p.add_argument("--rl-run", default=DEFAULTS["rl_run"], help="RL 推理产物目录（含 summary.json）")
    p.add_argument("--out", default=DEFAULTS["out"], help="产物目录")
    args = p.parse_args(argv)
    build(cf_root=args.cf_root, cf_bench=args.cf_bench, cf_eval=args.cf_eval,
          rl_run=args.rl_run, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
