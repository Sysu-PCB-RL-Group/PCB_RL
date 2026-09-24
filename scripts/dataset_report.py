# -*- coding: utf-8 -*-
"""数据集普查 / 检验脚本 —— **增板 / 删板后跑这一条**（Owner 2026-09-22 要求）。

用途：`data/kicad-native-batch/` 里**加板或删板**后，一条命令得到【全部会变的数字】，
      并与脚本内记录的 **BASELINE** 逐项对比（✅ 一致 / ⚠️ 变了）⇒ 据此更新金表、`verify_pcb_model.py`
      与台账 §0.2 / §0.4，而不用靠记忆。

用法（⚠️ 必须先设 `KICAD_PATH`）：

```text
$env:KICAD_PATH="E:\Kicad"
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dataset_report.py                 # 快档：A~D + G + Z
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dataset_report.py --full           # 再加 E(参照诊断)/F(结构性件)/H(边际贡献)
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dataset_report.py --json out.json  # 落 JSON（便于脚本化）
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dataset_report.py --data data/bad_data   # 也扫"被排除"目录
```

分节：

```text
A 对象计数（= scripts/verify_pcb_model.py 的口径）   E 参照诊断数字（§0.4：`Judge` 全量、源位姿）
B net / pad 统计                                     F 结构性不可放件（C1 空板即空 / C2 专家前缀即空）
C HPWL（排除电源网）＋ H_ref 分布                     G 顺序聚合指标（§6.2 的五个数）
D obs 标量归一化常数（PAD_MAX / NET_MAX）             H 边际贡献 c_t 的偏斜（probe131/206 口径）
Z 与【BASELINE】逐项对比
```
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.kicad_utils import pcb_model as pm          # noqa: E402  （触发 bootstrap：DLL + wx 抑制）
from env import grid_render as gr                      # noqa: E402
from env import order_plan as op                       # noqa: E402
from env import violations as vio                      # noqa: E402
from utils.kicad_utils.pcb_model import place_at       # noqa: E402

DEF_DATA = os.path.join("data", "kicad-native-batch")

# ===================== 基线（**增删板后按本脚本输出更新这里**） =====================
# 记录时间：2026-09-22（**16 板**；2026-09-22 把 4 块 C1/C2 板移入 `data/bad_data/` 之后重测）
# 20 板时代的值见台账 §0.2/§0.4 的历史记录（857 件 / 3780 pad / 3422 via / 46 禁布区 / HPWL 11355.1mm …）
BASELINE = {
    "boards": 16,
    # ---- A 对象计数 ----
    "A.footprints": 668,
    "A.zero_pad_footprints_in_model": 0,
    "A.synthetic_refs": 23,
    "A.pads_total": 2648,
    "A.pads_with_net": 2266,
    "A.pads_no_net": 382,
    "A.pads_negative_annular": 28,
    "A.vias_total": 2800,
    "A.vias_no_net": 24,
    "A.keepouts_total": 40,
    "A.keepouts_board_slot": 39,
    "A.keepouts_motor_hull": 1,
    "A.non_orthogonal_raw_angle": 8,
    "A.hpwl_total_mm": 7139.3,
    # ---- B net / pad ----
    "B.nets_power": 61,
    "B.nets_nonpower": 445,
    "B.nonpower_pads": 1278,
    # ---- C HPWL ----
    "C.hpwl_min": 35.5,
    "C.hpwl_p50": 505.2,
    "C.hpwl_max": 1477.8,
    "C.hpwl_total_mm": 7139.3,
    # ---- D obs 标量常数 ----
    "D.PAD_MAX": 80,
    "D.NET_MAX": 28,
    "D.zero_np_net_pieces": 153,
}


def q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def boards_of(data_dir):
    out = []
    for root, _d, files in os.walk(os.path.abspath(data_dir)):
        for f in sorted(files):
            if f.endswith(".kicad_pcb") and not f.endswith("best_layout.kicad_pcb"):
                out.append(os.path.join(root, f))
    return sorted(out)


def short(path):
    return os.path.basename(os.path.dirname(path))


class Report:
    def __init__(self, data_dir, full=False):
        self.data_dir = data_dir
        self.full = full
        self.rep = {}
        self.models = []            # [(path, model)]
        self._load()

    # ---------- 载入 ----------
    def _load(self):
        paths = boards_of(self.data_dir)
        if not paths:
            raise SystemExit(f"在 {self.data_dir} 下没找到 .kicad_pcb")
        print(f"== 扫描目录：{os.path.abspath(self.data_dir)}｜板数 {len(paths)} ==")
        for p in paths:
            board = pm.pcbnew.LoadBoard(p)
            self.models.append((p, pm.build_model(board, p)))
        self.rep["boards"] = len(paths)

    # ---------- A 对象计数 ----------
    def sec_a(self):
        fps = [f for _p, m in self.models for f in m.footprints]
        pads = [p for f in fps for p in f.pads]
        vias = [v for _p, m in self.models for v in m.vias]
        kos = [k for _p, m in self.models for k in m.keepouts]
        a = {
            "footprints": len(fps),
            "zero_pad_footprints_in_model": sum(1 for f in fps if not f.pads),
            "synthetic_refs": sum(1 for f in fps if f.ref_is_synthetic),
            "pads_total": len(pads),
            "pads_with_net": sum(1 for p in pads if p.has_net),
            "pads_no_net": sum(1 for p in pads if not p.has_net),
            # 负环宽：**单点定义在 `pcb_model.Pad.annular_ok`**（不要另写一份口径）
            "pads_negative_annular": sum(1 for p in pads if not p.annular_ok),
            "pads_layers_F": sum(1 for p in pads if p.layers == (0,)),
            "pads_layers_B": sum(1 for p in pads if p.layers == (1,)),
            "pads_layers_FB": sum(1 for p in pads if p.layers == (0, 1)),
            "vias_total": len(vias),
            "vias_no_net": sum(1 for v in vias if not v.net_id),
            "keepouts_total": len(kos),
            "keepouts_board_slot": sum(1 for k in kos if getattr(k, "kind", "") == "board_slot"),
            "keepouts_motor_hull": sum(1 for k in kos if getattr(k, "kind", "") == "motor_hull"),
            "non_orthogonal_raw_angle": sum(
                1 for f in fps
                if min(abs(f.rotation_deg_raw % 90), 90 - abs(f.rotation_deg_raw % 90)) > 1e-6),
        }
        hpwl_total = 0.0
        for _p, m in self.models:
            hpwl_total += m.hpwl(refs={f.ref for f in m.footprints})
        a["hpwl_total_mm"] = round(hpwl_total, 1)
        print("\n== A 对象计数（= verify_pcb_model 口径）==")
        for k, v in a.items():
            print(f"  {k:32s} {v}")
        self.rep.update({f"A.{k}": v for k, v in a.items()})

    # ---------- B net / pad ----------
    def sec_b(self):
        nets_p = nets_np = 0
        dist = {"1": 0, "2": 0, "3-5": 0, ">5": 0}
        np_pads = 0
        for _p, m in self.models:
            for net in m.nets.values():
                if net.is_power:
                    nets_p += 1
                    continue
                nets_np += 1
                k = len(net.pads)
                np_pads += k
                dist["1" if k <= 1 else "2" if k == 2 else "3-5" if k <= 5 else ">5"] += 1
        b = {"nets_power": nets_p, "nets_nonpower": nets_np, "nonpower_pads": np_pads}
        print("\n== B net / pad 统计 ==")
        print(f"  电源 net {nets_p}｜非电源 net {nets_np}｜非电源 net 的 pad 总数 {np_pads}")
        print(f"  非电源 net 的 pad 数分布（1 / 2 / 3-5 / >5）：{dist}")
        self.rep.update({f"B.{k}": v for k, v in b.items()})

    # ---------- C HPWL / H_ref ----------
    def sec_c(self):
        h = []
        for p, m in self.models:
            h.append((short(p), m.hpwl(refs={f.ref for f in m.footprints})))
        vals = [x for _n, x in h]
        c = {
            "hpwl_min": round(min(vals), 1), "hpwl_p50": round(q(vals, 0.5), 1),
            "hpwl_p95": round(q(vals, 0.95), 1), "hpwl_max": round(max(vals), 1),
            "hpwl_total_mm": round(sum(vals), 1),
        }
        print("\n== C HPWL（排除电源网，源位姿）==")
        print(f"  逐板 min {c['hpwl_min']}｜p50 {c['hpwl_p50']}｜p95 {c['hpwl_p95']}｜max {c['hpwl_max']}"
              f"｜合计 {c['hpwl_total_mm']} mm")
        self.rep.update({f"C.{k}": v for k, v in c.items()})

    # ---------- D obs 标量常数 ----------
    def sec_d(self):
        pads, nets, zero = [], [], 0
        for _p, m in self.models:
            for _ref, (npads, nnet) in op.piece_features(m).items():
                pads.append(npads)
                nets.append(nnet)
                zero += int(nnet == 0)
        d = {
            "pad_min": min(pads), "pad_p50": q(pads, 0.5), "pad_p95": q(pads, 0.95),
            "pad_p99": q(pads, 0.99), "PAD_MAX": max(pads),
            "net_min": min(nets), "net_p50": q(nets, 0.5), "net_p95": q(nets, 0.95),
            "net_p99": q(nets, 0.99), "NET_MAX": max(nets),
            "zero_np_net_pieces": zero,
            "zero_np_net_ratio": round(zero / len(nets), 4),
        }
        print("\n== D obs 标量：归一化常数（`env/obs_spec` 的 PAD_MAX / NET_MAX）==")
        print(f"  pad 数   min {d['pad_min']}｜p50 {d['pad_p50']}｜p95 {d['pad_p95']}｜p99 {d['pad_p99']}"
              f"｜**max {d['PAD_MAX']}**")
        print(f"  非电源 net 数 min {d['net_min']}｜p50 {d['net_p50']}｜p95 {d['net_p95']}｜p99 {d['net_p99']}"
              f"｜**max {d['NET_MAX']}**")
        print(f"  「非电源 net 数 = 0」的件 {zero} / {len(nets)} = {d['zero_np_net_ratio']:.1%}"
              f"（= 目标零贡献件；归一后精确 0.0）")
        self.rep.update({f"D.{k}": v for k, v in d.items()})

    # ---------- E 参照诊断（§0.4） ----------
    def sec_e(self):
        tot = {r: 0 for r in vio.ALL_REASONS}
        for _p, m in self.models:
            placements = {f.ref: (f.anchor[0], f.anchor[1], int(f.rotation_index))
                          for f in m.footprints}
            res = vio.Judge(m).judge_all(placements)
            for r in vio.ALL_REASONS:
                tot[r] += int(res.counts.get(r, 0))
        print("\n== E 参照诊断数字（§0.4；全件源位姿、AABB 口径）==")
        for r in vio.ALL_REASONS:
            print(f"  {r:20s} {tot[r]}")
        self.rep.update({f"E.{r}": v for r, v in tot.items()})

    # ---------- F 结构性不可放件 ----------
    def sec_f(self):
        c1all, c2all = [], []
        for p, m in self.models:
            order = tuple(op.plan_order(m))
            spec = gr.grid_spec(m, m=gr.GRID_M)
            byref = {f.ref: f for f in m.footprints}
            for t, ref in enumerate(order):
                if not gr.legal_masks(m, [], byref[ref], spec).any():
                    c1all.append(f"{short(p)}/{ref}")
            placed = []
            for t, ref in enumerate(order):
                if not gr.legal_masks(m, placed, byref[ref], spec).any():
                    c2all.append(f"{short(p)}/t={t}/{ref}")
                f = byref[ref]
                placed.append(place_at(f, f.anchor, f.rotation_index))
        print("\n== F 结构性不可放件 ==")
        print(f"  C1（空板即空，与策略无关）：{len(c1all)} 件 -> {c1all}")
        print(f"  C2（专家前缀即空）：{len(c2all)} 件 -> {c2all}")
        self.rep["F.C1_count"] = len(c1all)
        self.rep["F.C2_count"] = len(c2all)
        self.rep["F.C1_list"] = c1all
        self.rep["F.C2_list"] = c2all

    # ---------- G 顺序聚合 ----------
    def sec_g(self):
        conf, gaps, area25, pwr_pos = [], [], [], []
        for _p, m in self.models:
            order = tuple(op.plan_order(m))
            n = len(order)
            byref = {f.ref: f for f in m.footprints}
            np_nets = {r: {p.net_id for p in byref[r].pads
                           if p.has_net and not m.nets[p.net_id].is_power} for r in order}
            warm, cold = set(), 0
            for t, r in enumerate(order):
                # 迷茫步 = 该件【有】非电源 net、但与已放件涉及的 net 全不相交
                # （电源专属/无 net 件不计入 —— 它们对目标零贡献，计入会把比例虚高 3~4 倍）
                if np_nets[r] and not (np_nets[r] & warm):
                    cold += 1
                warm |= np_nets[r]
            conf.append(cold / n)
            areas = {r: byref[r].local_hull_area for r in order}
            area25.append(sum(areas[r] for r in order[: max(1, n // 4)]) / max(1e-12, sum(areas.values())))
            first = {}
            for t, r in enumerate(order):
                for net in np_nets[r]:
                    first.setdefault(net, []).append(t)
            for _net, ts in first.items():
                if len(ts) >= 2:
                    ts.sort()
                    gaps.append(ts[1] - ts[0])
            pos = [t / n for t, r in enumerate(order) if not np_nets[r]]
            if pos:
                pwr_pos.append(sum(pos) / len(pos))
        g = {
            "confused_step_ratio": round(st.median(conf), 4),
            "first_to_second_gap_median": round(st.median(gaps), 2) if gaps else None,
            "area_in_first_25pct": round(st.median(area25), 4),
            "power_only_piece_pos": round(st.median(pwr_pos), 4) if pwr_pos else None,
        }
        print("\n== G 顺序聚合指标（§6.2 口径；逐板取中位）==")
        for k, v in g.items():
            print(f"  {k:30s} {v}")
        self.rep.update({f"G.{k}": v for k, v in g.items()})

    # ---------- H 边际贡献偏斜 ----------
    def sec_h(self):
        cs = []
        zero = 0
        for _p, m in self.models:
            order = tuple(op.plan_order(m))
            allr = set(order)
            full = m.hpwl(refs=allr)
            for r in order:
                c = full - m.hpwl(refs=allr - {r})
                cs.append(c)
                zero += int(c == 0.0)
        print("\n== H 边际贡献 c_t（专家布局口径；probe131/206）==")
        print(f"  c 绝对值 min {min(cs):.3f}｜p50 {q(cs, .5):.3f}｜p95 {q(cs, .95):.3f}"
              f"｜p99 {q(cs, .99):.3f}｜max {max(cs):.3f} mm")
        print(f"  c = 0 的件 {zero} / {len(cs)} = {zero / len(cs):.1%}")
        self.rep.update({"H.c_p50": round(q(cs, .5), 3), "H.c_max": round(max(cs), 3),
                         "H.c_zero_count": zero})

    # ---------- Z 对比 ----------
    def sec_z(self):
        print("\n== Z 与基线对比（⚠️ = 变了 ⇒ 需要更新金表/台账）==")
        diff = 0
        for k, want in BASELINE.items():
            got = self.rep.get(k)
            if want is None:
                print(f"  {k:32s} {got}   (基线未记 ⇒ 请回填)")
                diff += 1
            elif got == want:
                print(f"  {k:32s} {got}   ✅")
            else:
                print(f"  {k:32s} {got}   ⚠️ 基线 {want}")
                diff += 1
        print(f"  ⇒ 需要处理的项：{diff}")

    def run(self):
        self.sec_a()
        self.sec_b()
        self.sec_c()
        self.sec_d()
        if self.full:
            self.sec_e()
            self.sec_f()
        self.sec_g()
        if self.full:
            self.sec_h()
        self.sec_z()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--full", action="store_true", help="加跑 E(参照诊断)/F(结构性件)/H(边际贡献)（慢）")
    ap.add_argument("--json", default=None, help="把结果落成 JSON")
    args = ap.parse_args()

    if not os.environ.get("KICAD_PATH"):
        print("⚠️ 建议先设 KICAD_PATH（否则 pcbnew 的 DLL 路径可能找不到）")
    r = Report(args.data, full=args.full)
    r.run()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r.rep, fh, ensure_ascii=False, indent=2)
        print(f"\n已写 {args.json}")


if __name__ == "__main__":
    main()
