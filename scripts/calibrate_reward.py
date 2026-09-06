"""PCB RL — Reward 惩罚体系离线标定脚本（feat 体例：二次软间距、S3=off、无 flat）。

目的：在不启动训练/推理的前提下，用开关小板真实几何数据，量化验证
reward 数值体例（合并决策 S3/S4/S5 采纳 feat）：
  - 硬重叠 / 出界 / keepout 计入 has_violations → 违规步不发完成/品质、不提交放置；
  - 软间距(gap<thr)不计违规，仅作连续惩罚 `min(W_OVERLAP_SOFT × (thr-gap)²×edge, W_OVERLAP_SOFT)`；
  - 面积惩罚为"比例惩罚"（÷自身面积，MIN_NORM_AREA 兜底），与 reward 同量级。

输出四张表：
  ① 开关小板真实 bbox 尺寸与代表性 edge
  ② 合规（人工）布局逐对软间距惩罚（理想应全为 0；>0 仅反映量化后固有紧贴，非违规）
  ③ 软间距二次公式单对罚分（gap ∈ {0.18, 0.124, 0.076, 0.0}）
  ④ 硬重叠比例罚分对照 + 失败边际 vs 成功边际

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
MIN_NORM_AREA = getattr(Config, "MIN_NORM_AREA", 1.0)
THR = 0.2  # spacing_threshold_for 统一 0.2
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
    edge_rep = med(widths)  # 代表性 edge（宽度中位数）
    print(f"可移动元件数: {n}")
    print(f"bbox 宽度(mm): min={min(widths):.3f} median={med(widths):.3f} max={max(widths):.3f}")
    print(f"bbox 高度(mm): min={min(heights):.3f} median={med(heights):.3f} max={max(heights):.3f}")
    print(f"代表性 edge (宽度中位数) = {edge_rep:.3f} mm  <- 用于表③④计算")
    return edge_rep, n


def table2_perfect(k):
    print("=" * 70)
    print("表② 合规(人工)布局逐对软间距（feat：S3=off，soft 不构成违规，理想全为 0）")
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
            if a.layer != b.layer:  # 同层才判（feat：碰撞前提为同层）
                continue
            thr = Config.spacing_threshold_for(a.package_type, b.package_type)
            hard, soft = PCBUtils.bbox_overlap_penalty(a.bbox, b.bbox, thr)
            pairs_checked += 1
            max_soft = max(max_soft, soft)
            max_hard = max(max_hard, hard)
            total_soft += soft
            if soft > 1e-9:
                viol_pairs.append((a.id, b.id, soft))
    print(f"检查同层元件对: {pairs_checked}")
    print(f"最大硬重叠: {max_hard:.4f} mm^2")
    print(f"最大软间距惩罚(二次 raw): {max_soft:.4f}")
    print(f"软间距总惩罚(二次 raw): {total_soft:.4f}")
    if not viol_pairs:
        print("[OK] 合规布局软间距全为 0（阈值 0.2 未误伤人工布局）")
    else:
        print(f"[WARN] {len(viol_pairs)} 对 soft>0（人工布局量化后固有紧贴，非违规；"
              f"对应单步软罚=min(W_SOFT×Σ, W_SOFT)）:")
        for a, b, s in viol_pairs[:10]:
            print(f"   {a} ~ {b}: soft={s:.4f} -> 单步罚 {min(W_SOFT * s, W_SOFT):.2f}")
    return max_soft


def table3_gap(edge):
    print("=" * 70)
    print(f"表③ 软间距二次公式单对罚分（edge={edge:.3f}mm, W_SOFT={W_SOFT}, THR={THR}）")
    print("=" * 70)
    print(f"罚分 = min(W_SOFT × (THR-gap)² × edge, W_SOFT)（单对；多邻累加仍封顶 W_SOFT）")
    print(f"{'gap(mm)':>8} | {'soft raw':>10} | {'单对罚':>10} | {'对比完成partial':>12}")
    print("-" * 50)
    comp_per = W_COMP / 6.0  # 示意：6 元件板 partial
    for gap in (0.18, 0.124, 0.076, 0.0):
        raw = (THR - gap) ** 2 * edge
        pen = min(W_SOFT * raw, W_SOFT)
        print(f"{gap:>8.3f} | {raw:>10.4f} | {-pen:>10.2f} | {comp_per:>+12.2f}")


def table4_overlap(n_mov):
    print("=" * 70)
    print("表④ 硬重叠比例罚分对照（feat：重叠 has_violations=True -> 无完成/品质）")
    print("=" * 70)
    comp_per = W_COMP / max(n_mov, 1)
    print(f"成功单步基准收益 ≈ W_COMPLETION/{n_mov} = +{comp_per:.2f}（不含品质项）")
    print(f"{'重叠(mm^2)':>10} | {'占比':>8} | {'硬罚(=权重×占比)':>16} | {'违规净':>8}")
    print("-" * 56)
    comp_area = max((1.524 ** 2), MIN_NORM_AREA)  # 示意：真实 bbox 约 1.524mm 见方
    for area in (0.3, 0.9, 2.3, 5.0):
        ratio = min(area / comp_area, 1.0)
        pen = W_HARD * ratio
        print(f"{area:>10.2f} | {ratio:>7.1%} | {-pen:>16.1f} | {-pen:>8.1f}")

    print()
    print("对照说明（S3=off）：贴紧(gap<thr)不判违规、放置照常提交并拿完成/品质，"
          "只扣软罚单对；")
    print("要使 0.076mm 档单对净负，需 W_SOFT 高到让罚分反超成功收益——这正是 W_SOFT=400 "
          "背景下‘贴边代价反超完成奖励’的标定依据。")


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
    print("\n结论（feat 体例）：硬重叠/出界/keepout 违规净负；软间距仅连续罚、不判违规，"
          "贴紧代价由 W_OVERLAP_SOFT 标定。")
    print("注：本脚本为离线诊断，随合并后 reward 体例更新（原方案B flat/SOFT_REF_DEPTH 已废除）。")


if __name__ == "__main__":
    main()
