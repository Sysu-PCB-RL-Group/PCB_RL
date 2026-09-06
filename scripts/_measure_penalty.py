"""临时标定：实测开关小板真实 bbox + 用"目前"公式/权重算真实罚分。
复用 env.pcb_utils.PCBUtils.bbox_overlap_penalty（当前公式），权重从 argument.txt 读。
"""
import os, re, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kicad_utils.kicad_utils import KiCadUtils
from env.pcb_utils import PCBUtils

PCB = (r"E:\PCB_AI\PCB_RL\data\kicad-native-batch\ran-pang-airlink-86231b34"
       r"--开关小板-883285b3\ran-pang-airlink-86231b34--开关小板-883285b3.kicad_pcb")
ARG = r"E:\PCB_AI\PCB_RL\config\argument.txt"
CELL = 0.2


def parse_args(path):
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*([-\d.eE]+)", line)
            if m:
                cfg[m.group(1)] = float(m.group(2))
    return cfg


cfg = parse_args(ARG)
W_HARD = cfg["W_OVERLAP_HARD"]
W_SOFT = cfg["W_OVERLAP_SOFT"]
W_OOB_INIT = cfg["W_OUT_OF_BOUNDS_INITIAL"]
W_OOB_FINAL = cfg["W_OUT_OF_BOUNDS_FINAL"]
W_COMP = cfg["W_COMPLETION"]
THR = 0.2  # 软间距阈值（2026-08-19 统一校准值）

print("=" * 70)
print("当前权重（来自 argument.txt）:",
      f"W_HARD={W_HARD} W_SOFT={W_SOFT} W_OOB_INIT={W_OOB_INIT} "
      f"W_OOB_FINAL={W_OOB_FINAL} W_COMP={W_COMP} THR={THR}")
print("=" * 70)

ku = KiCadUtils()
assert ku.load_board(PCB), "load_board 失败"
movable = [(ref, c) for ref, c in ku.components.items() if not c.is_fixed]
print(f"\n总元件={len(ku.components)} 可动={len(movable)} 固定={len(ku.components)-len(movable)}")

print("\n[表A] 可动元件真实 bbox（GetBoundingHull 真实几何）")
dims = []
for ref, c in movable:
    x1, y1, x2, y2 = c.bbox
    w, h = x2 - x1, y2 - y1
    d = max(w, h)
    print(f"  {ref:6s} {c.footprint:20s} w={w:.3f} h={h:.3f} "
          f"d=max={d:.3f}  d%0.2={d % CELL:.3f}  type={c.package_type}")
    dims.append((ref, w, h, d))

# 颗粒度适配：最小正 gap（0.2*ceil(d/0.2) - d）
print("\n[表B] 颗粒度适配：每元件最小正 gap（紧贴 gap=0 是否可达）")
min_pos_gaps = []
for ref, w, h, d in dims:
    k = math.ceil(d / CELL)
    min_pos = CELL * k - d
    reachable0 = abs(d % CELL) < 1e-9
    print(f"  {ref:6s} d={d:.3f} 最小正gap={min_pos:.3f}  gap=0可达={'是' if reachable0 else '否'}")
    min_pos_gaps.append(min_pos)
board_min_pos_gap = min(min_pos_gaps)
print(f"  >>> 全板最小正 gap = {board_min_pos_gap:.3f} mm （软间距唯一可达档；gap=0 不可达）")

# 枚举所有可动元件对的量化偏移，收集 soft/hard 真实面积
print("\n[表C] 用目前公式+权重枚举每对元件量化偏移(k,m=0..9)")
soft_records, hard_records = [], []
for i in range(len(movable)):
    for j in range(i + 1, len(movable)):
        refa, ca = movable[i]
        refb, cb = movable[j]
        wa, ha = ca.bbox[2] - ca.bbox[0], ca.bbox[3] - ca.bbox[1]
        wb, hb = cb.bbox[2] - cb.bbox[0], cb.bbox[3] - cb.bbox[1]
        for k in range(0, 10):
            for m in range(0, 10):
                ax, ay = CELL * k, CELL * m
                bbox_a = (ax, ay, ax + wa, ay + ha)
                bbox_b = (0, 0, wb, hb)
                hard, soft = PCBUtils.bbox_overlap_penalty(bbox_a, bbox_b, THR)
                if soft > 0:
                    soft_records.append(soft)
                if hard > 0:
                    hard_records.append((k, m, hard))

soft_areas = sorted(set(round(s, 5) for s in soft_records))
hard_areas = sorted(set((round(a, 5), k, m) for k, m, a in hard_records))
print(f"  软间距面积档(去重,{len(soft_areas)}个): 最轻={soft_areas[0]:.5f} 最坏={soft_areas[-1]:.5f}")
print(f"  硬重叠面积档(去重,{len(hard_areas)}个): 最小={hard_areas[0][0]:.5f} 完全={hard_areas[-1][0]:.5f}")
print("\n  软间距各档罚分(W_SOFT=%g):" % W_SOFT)
for sa in soft_areas:
    print(f"    area={sa:.5f} -> {sa*W_SOFT:+.4f}")
print("  硬重叠代表档罚分(W_HARD=%g):" % W_HARD)
for a, k, m in [hard_areas[0], hard_areas[len(hard_areas)//2], hard_areas[-1]]:
    print(f"    k={k} m={m} area={a:.5f} -> {a*W_HARD:+.3f}")

print("\n[表D] 目前罚分汇总（关键档位，负号=惩罚）")
soft_worst = soft_areas[-1] * W_SOFT
soft_light = soft_areas[0] * W_SOFT
hard_min = hard_areas[0][0] * W_HARD
hard_mid = hard_areas[len(hard_areas)//2][0] * W_HARD
hard_full = hard_areas[-1][0] * W_HARD
print(f"  软间距-紧贴正对(gap=0.076,area={soft_areas[-1]:.5f}): {soft_worst:+.3f}")
print(f"  软间距-角落最轻(area={soft_areas[0]:.5f}): {soft_light:+.3f}")
print(f"  硬重叠-角落轻微(area={hard_areas[0][0]:.5f}): {hard_min:+.3f}")
print(f"  硬重叠-中位(area={hard_areas[len(hard_areas)//2][0]:.5f}): {hard_mid:+.3f}")
print(f"  硬重叠-完全(area={hard_areas[-1][0]:.5f}): {hard_full:+.3f}")
area = dims[0][1] * dims[0][2]
print(f"  出界-完全(面积={area:.3f}): 初始{area*W_OOB_INIT:+.3f} 最终{area*W_OOB_FINAL:+.3f}")

print("\n[对照] 一次成功放置收益 ≈ +22.24（实测 133.41/6）")
print(f"  >>> 紧贴软间距罚 {soft_worst:+.2f}  vs 成功 +22.24  → 仅占 {soft_worst/22.24*100:.1f}%")
print(f"  >>> 最小硬重叠 {hard_min:+.2f} | 中位 {hard_mid:+.2f} | 完全 {hard_full:+.2f}")
print(f"  >>> 阈值: 失败罚分须 > 成功收益 +22.24 才有'亏本'信号")
