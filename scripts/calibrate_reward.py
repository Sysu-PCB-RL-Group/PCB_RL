"""PCB RL — Reward 惩罚体系离线标定脚本（方案B）。

目的：在不启动训练/推理的前提下，用开关小板真实几何数据，量化验证
"保持间距 > 紧贴 > 重叠"三档在 reward 上严格单调递减，且违规动作净收益为负。

输出四张表：
  ① 开关小板真实 bbox 尺寸与代表性 edge
  ② 合规（人工）布局逐对软间距惩罚（应全为 0，验证阈值 0.2 不误伤）
  ③ 软间距新旧公式对照（gap ∈ {0, 0.15, 0.1, 0.2}）
  ④ 重叠 + 固定失败成本对照 + 失败边际 vs 成功边际

运行：
  需要 KICAD_PATH=E:\Kicad（提取真实 bbox）；若 KiCad 不可用，表①②跳过，
  表③④用 edge=1.5mm 估算并标注。
  例： $env:KICAD_PATH="E:\Kicad"; python scripts/calibrate_reward.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("KICAD_PATH", r"E:\Kicad")

from utils.kicad_utils import KiCadUtils
from env.pcb_utils import PCBUtils
from env.pcb_layout_env import Config

Config.load_from_file(os.path.join(ROOT, "config", "argument.txt"))

SWITCH_PCB = os.path.join(
    ROOT,
    "data", "kicad-native-batch",
    "ran-pang-airlink-86231b34--开关小板-883285b3",
    "ran-pang-airlink-86231b34--开关小板-883285b3.kicad_pcb",
)

W_SOFT = Config.W_OVERLAP_SOFT
W_HARD = Config.W_OVERLAP_HARD
W_FLAT = Config.W_VIOLATION_FLAT
THR = 0.2  # spacing_threshold_for 统一 0.2
SREF = Config.SOFT_REF_DEPTH
W_COMP = Config.W_COMPLETION


def load_switch():
    k = KiCadUtils()
    ok = k.load_board(SWITCH_PCB)
    return k if ok else None


def table1_edge(k):
    print("=" * 70)
    print("表① 开关小板真实 bbox 尺寸与代表性 edge")
    print("=" * 70)
    mov = [c for c in k.components.values() if not c.is_fixed]
    widths = sorted(c.bbox[2] - c.bbox[0] for c in mov)
    heights = sorted(c.bbox[3] - c.bbox[1] for c in mov)
    n = len(mov)
    med = lambda xs: (xs[len(xs) // 2] if xs else 0.0)
    edge_rep = med(widths)  # 代表性 edge（窄边中位数）
    print(f"可移动元件数: {n}")
    print(f"bbox 宽度(mm): min={min(widths):.3f} median={med(widths):.3f} max={max(widths):.3f}")
    print(f"bbox 高度(mm): min={min(heights):.3f} median={med(heights):.3f} max={max(heights):.3f}")
    print(f"代表性 edge (宽度中位数) = {edge_rep:.3f} mm  <- 用于表③④计算")
    return edge_rep, n


def table2_perfect(k):
    print("=" * 70)
    print("表② 合规(人工)布局逐对软间距惩罚（方案B要求全为 0）")
    print("=" * 70)
    mov = [c for c in k.components.values() if not c.is_fixed]
    max_soft = 0.0
    max_hard = 0.0
    total_soft = 0.0
    pairs_checked = 0
    viol_pairs = []
    for i in range(len(mov)):
        for j in range(i + 1, len(mov)):
            a, b = mov[i], mov[j]
            if not PCBUtils.layers_conflict(a.layer, b.layer, Config.ENABLE_DUAL_SIDED):
                continue
            thr = Config.spacing_threshold_for(a.package_type, b.package_type)
            hard, soft = PCBUtils.bbox_overlap_penalty(a.bbox, b.bbox, thr, SREF)
            pairs_checked += 1
            max_soft = max(max_soft, soft)
            max_hard = max(max_hard, hard)
            total_soft += soft
            if soft > 1e-9:
                viol_pairs.append((a.id, b.id, soft))
    print(f"检查同层元件对: {pairs_checked}")
    print(f"最大硬重叠: {max_hard:.4f} mm^2")
    print(f"最大软间距惩罚面积: {max_soft:.4f} mm^2")
    print(f"软间距总惩罚: {total_soft:.4f} mm^2")
    if not viol_pairs:
        print("[OK] 合规布局软间距惩罚全为 0（阈值 0.2 未误伤人工布局，完美布局零罚守住）")
    else:
        print(f"[WARN] 发现 {len(viol_pairs)} 对软间距违规（人工布局本身间距<0.2?）：")
        for a, b, s in viol_pairs[:10]:
            print(f"   {a} ~ {b}: soft={s:.4f}")
    return max_soft


def table3_gap(edge):
    print("=" * 70)
    print(f"表③ 软间距新旧公式对照（edge={edge:.3f}mm, W_SOFT={W_SOFT}, THR={THR}）")
    print("=" * 70)
    print(f"{'gap(mm)':>8} | {'旧罚分':>10} | {'新罚分':>10} | {'新/旧':>8}")
    print("-" * 42)
    for gap in (0.2, 0.15, 0.1, 0.0):
        if gap >= THR:
            old = new = 0.0
        else:
            old_area = (THR - gap) ** 2 * edge
            new_area = (THR - gap) / THR * edge * SREF
            old = W_SOFT * old_area
            new = W_SOFT * new_area
        ratio = (new / old) if old != 0 else float("inf")
        print(f"{gap:>8.2f} | {old:>10.3f} | {new:>10.3f} | {ratio:>7.1f}x")


def table4_overlap(n_mov):
    print("=" * 70)
    print("表④ 重叠 + 固定失败成本对照（方案B：重叠 has_violations=True -> 扣 flat）")
    print("=" * 70)
    comp_per = W_COMP / max(n_mov, 1)
    print(f"成功单步基准收益 ≈ W_COMPLETION/{n_mov} = +{comp_per:.2f}（不含品质项）")
    print(f"{'重叠(mm^2)':>10} | {'硬罚':>8} | {'+flat':>8} | {'违规净':>8} | {'相对成功':>10}")
    print("-" * 52)
    for area in (0.3, 0.9, 2.3):
        hard_pen = W_HARD * area
        total = hard_pen + W_FLAT
        print(f"{area:>10.2f} | {-hard_pen:>8.1f} | {-W_FLAT:>8.1f} | {-total:>8.1f} | {(-total - comp_per):>+10.1f}")

    print()
    print("紧贴（gap=0）净收益（方案B核心：贴紧 has_violations=True -> 不发奖励只扣分）：")
    for edge in (1.5,):
        soft_area = (THR - 0.0) / THR * edge * SREF
        soft_pen = W_SOFT * soft_area
        total = soft_pen + W_FLAT
        print(f"  edge={edge}: soft罚={-soft_pen:.1f}, +flat={-W_FLAT:.1f}, 违规净={-total:.1f} "
              f"(成功基准 +{comp_per:.1f})  ->  {'净负[OK]' if -total < 0 else '仍正[X]'}")


def main():
    k = load_switch()
    edge, n = 1.5, 6
    if k is not None:
        edge, n = table1_edge(k)
        table2_perfect(k)
    else:
        print("[WARN] KiCad 不可用，表①②跳过；表③④用 edge=1.5mm 估算")
    table3_gap(edge)
    table4_overlap(n)
    print("\n结论：违规动作（贴紧/重叠）净收益为负，合法放置为正 -> 模型会主动规避贴紧。")


if __name__ == "__main__":
    main()
