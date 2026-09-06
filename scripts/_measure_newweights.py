"""临时标定：按用户新权重（纯数值上调，公式不变）算开关小板真实罚分。
权重：W_SOFT=2000(10*200) W_HARD=200(20*10) W_OOB_INIT=10(2*5) W_OOB_FINAL=40(20*2)
软件间距公式保持不变 ((thr-gap)**2 * edge)，复用 PCBUtils.bbox_overlap_penalty。
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


cur = parse_args(ARG)
# 当前值（来自 argument.txt）
CUR = dict(W_HARD=cur["W_OVERLAP_HARD"], W_SOFT=cur["W_OVERLAP_SOFT"],
          W_OOB_INIT=cur["W_OUT_OF_BOUNDS_INITIAL"], W_OOB_FINAL=cur["W_OUT_OF_BOUNDS_FINAL"],
          EARLY=cur["EARLY_STOP_THRESHOLD"])
# 新值（用户方案：纯数值上调，公式不变）
NEW = dict(W_HARD=20*10, W_SOFT=10*200, W_OOB_INIT=2*5, W_OOB_FINAL=20*2)
THR = 0.2
SUCCESS = 22.24  # 一次成功放置收益估算

print("当前 EARLY_STOP_THRESHOLD =", CUR["EARLY"])
print(f"新权重: W_SOFT={NEW['W_SOFT']} W_HARD={NEW['W_HARD']} "
      f"W_OOB_INIT={NEW['W_OOB_INIT']} W_OOB_FINAL={NEW['W_OOB_FINAL']}  (公式保持平方不变)\n")

ku = KiCadUtils()
assert ku.load_board(PCB)
movable = [(ref, c) for ref, c in ku.components.items() if not c.is_fixed]

soft_areas, hard_areas, oob_area = [], [], None
for i in range(len(movable)):
    for j in range(i + 1, len(movable)):
        refa, ca = movable[i]; refb, cb = movable[j]
        wa, ha = ca.bbox[2]-ca.bbox[0], ca.bbox[3]-ca.bbox[1]
        wb, hb = cb.bbox[2]-cb.bbox[0], cb.bbox[3]-cb.bbox[1]
        for k in range(0, 10):
            for m in range(0, 10):
                ax, ay = CELL*k, CELL*m
                hard, soft = PCBUtils.bbox_overlap_penalty(
                    (ax, ay, ax+wa, ay+ha), (0, 0, wb, hb), THR)
                if soft > 0: soft_areas.append(soft)
                if hard > 0: hard_areas.append(hard)
oob_area = movable[0][1].bbox[2]-movable[0][1].bbox[0]
oob_area *= movable[0][1].bbox[3]-movable[0][1].bbox[1]

soft_areas = sorted(set(round(s, 5) for s in soft_areas))
hard_areas = sorted(set(round(a, 5) for a in hard_areas))

print("=" * 78)
print(f"{'档位':12s} | {'面积mm2':>9s} | {'目前罚分':>10s} | {'新罚分':>10s} | 新净收益(+22.24-罚)")
print("=" * 78)
def row(name, area, w_cur, w_new):
    pc = area*w_cur; pn = area*w_new
    print(f"{name:12s} | {area:9.5f} | {pc:+10.2f} | {pn:+10.2f} | {SUCCESS-pn:+10.2f}")

print("-- 软间距 (W_SOFT: 目前%d -> 新%d) --" % (CUR['W_SOFT'], NEW['W_SOFT']))
row("紧贴-正对", soft_areas[-1], CUR['W_SOFT'], NEW['W_SOFT'])
row("紧贴-角落", soft_areas[0], CUR['W_SOFT'], NEW['W_SOFT'])
print("-- 硬重叠 (W_HARD: 目前%d -> 新%d) --" % (CUR['W_HARD'], NEW['W_HARD']))
row("轻微-角落", hard_areas[0], CUR['W_HARD'], NEW['W_HARD'])
row("中位", hard_areas[len(hard_areas)//2], CUR['W_HARD'], NEW['W_HARD'])
row("完全", hard_areas[-1], CUR['W_HARD'], NEW['W_HARD'])
print("-- 出界 (初期 %d->%d, 后期 %d->%d) --" % (CUR['W_OOB_INIT'], NEW['W_OOB_INIT'], CUR['W_OOB_FINAL'], NEW['W_OOB_FINAL']))
row("完全-初期", oob_area, CUR['W_OOB_INIT'], NEW['W_OOB_INIT'])
row("完全-后期", oob_area, CUR['W_OOB_FINAL'], NEW['W_OOB_FINAL'])

print("\n[单调性检查] 净收益排序(高=好): 保持=+%.2f" % SUCCESS)
nets = {
    "保持间距": SUCCESS,
    "紧贴-正对": SUCCESS - soft_areas[-1]*NEW['W_SOFT'],
    "轻微重叠": SUCCESS - hard_areas[0]*NEW['W_HARD'],
    "中位重叠": SUCCESS - hard_areas[len(hard_areas)//2]*NEW['W_HARD'],
    "完全重叠": SUCCESS - hard_areas[-1]*NEW['W_HARD'],
}
for name, v in sorted(nets.items(), key=lambda x: -x[1]):
    print(f"  {name:10s} {v:+10.2f}")

print("\n[EARLY_STOP 溢出检查] 当前阈值=%.0f" % CUR['EARLY'])
for name, area, w in [("完全重叠", hard_areas[-1], NEW['W_HARD']),
                       ("出界-后期", oob_area, NEW['W_OOB_FINAL']),
                       ("紧贴-正对", soft_areas[-1], NEW['W_SOFT'])]:
    pen = area*w
    print(f"  {name:10s} 单步罚={pen:+.1f}  vs 阈值{CUR['EARLY']:.0f} -> "
          f"{'一步早停!' if pen < CUR['EARLY'] else '未溢出'}")
print("\n注: 新标准未调整 EARLY_STOP，也未加 flat cost，未改公式。")
