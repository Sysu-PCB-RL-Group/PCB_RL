"""纯 JSON 统计：排查专家经验本身是否生成好（不 import env，不加载 KiCad）。

统计维度：
  1) 文件完整性 / 条数 / reward 是否损坏
  2) reward 标签健康度（专家不该负；完美布局 reward_full≈158.63）
  3) 变异多样性 mut_type 分布（是否几乎全是 none → 变异失效）
  4) 平移退化：unique shift 数（若≈1 说明平移采样退化）
  5) 动作覆盖：unique target (x1,y1,rot,layer) 组合数 / 总条数（过度集中=退化）
"""
import json
import glob
import os
from collections import Counter
import statistics

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "expert_transitions"))
PERFECT = 158.63  # 已知开关小板完美布局 reward 上限

def main():
    main_files = [f for f in glob.glob(os.path.join(D, "*_expert.json"))
                  if "_BACKUP" not in f and "_raw" not in f]
    raw_files = [f for f in glob.glob(os.path.join(D, "*_expert_raw.json"))]
    print(f"[scan] main={main_files}\n[scan] raw={raw_files}\n")

    for fp in main_files:
        data = json.load(open(fp, encoding="utf-8"))
        n = len(data)
        print(f"=== MAIN {os.path.basename(fp)} : {n} transitions ===")
        by_step = {}
        bad = 0
        for t in data:
            r = t.get("reward")
            if r is None or (isinstance(r, float) and r != r):
                bad += 1
            by_step.setdefault(t["step_id"], []).append(r if isinstance(r, (int, float)) else 0.0)
        for sid in sorted(by_step):
            rs = by_step[sid]
            print(f"  step{sid}: n={len(rs)} mean={statistics.mean(rs):.3f} "
                  f"min={min(rs):.3f} max={max(rs):.3f}")
        negs = [t for t in data if isinstance(t.get("reward"), (int, float)) and t["reward"] < 0]
        print(f"  NaN/None reward: {bad}  negative reward: {len(negs)}")

    for fp in raw_files:
        raw = json.load(open(fp, encoding="utf-8"))
        ntraj = len(raw)
        steps = [s for traj in raw for s in traj["steps"]]
        nstep = len(steps)
        rf = [s["reward_full"] for s in steps]
        print(f"\n=== RAW {os.path.basename(fp)} : {ntraj} traj / {nstep} steps ===")
        print(f"  reward_full: mean={statistics.mean(rf):.2f} min={min(rf):.2f} max={max(rf):.2f} "
              f"median={statistics.median(rf):.2f}")
        # 直方图
        buckets = [(-1e9, 0), (0, 100), (100, 130), (130, 150), (150, PERFECT), (PERFECT, 1e9)]
        for lo, hi in buckets:
            c = sum(1 for v in rf if lo <= v < hi)
            print(f"    [{lo:>7.2f},{hi:>7.2f}): {c}")
        below = sum(1 for v in rf if v < PERFECT - 1.0)
        print(f"  reward_full < {PERFECT-1:.2f} (即含惩罚/非完美): {below} "
              f"({100*below/nstep:.1f}%)")
        # 变异多样性
        mt = Counter(s["mut_type"] for s in steps)
        print(f"  mut_type: {dict(mt)}  (none 占比 {100*mt.get('none',0)/nstep:.1f}%)")
        # 平移退化
        shifts = Counter(tuple(traj["shift"]) for traj in raw)
        print(f"  unique shifts: {len(shifts)} / {ntraj}")
        if len(shifts) <= 5:
            print(f"    shift counts: {shifts.most_common(5)}")
        # 动作覆盖
        tgxy = Counter((round(s["target"]["bbox"][0], 3), round(s["target"]["bbox"][1], 3))
                        for s in steps)
        tgrl = Counter((s["target"]["rotation"], s["target"]["layer"]) for s in steps)
        print(f"  unique target (x1,y1): {len(tgxy)} / {nstep}  "
              f"(多样性比 {100*len(tgxy)/nstep:.1f}%)")
        print(f"  unique (rot,layer): {len(tgrl)}  top: {tgrl.most_common(6)}")
        print(f"  top target (x1,y1): {tgxy.most_common(8)}")

if __name__ == "__main__":
    main()
