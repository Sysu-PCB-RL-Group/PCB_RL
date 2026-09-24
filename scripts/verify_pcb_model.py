# -*- coding: utf-8 -*-
"""验收脚本：用 16 块板跑新的转换数据模型，逐项对照《导入信息重构》的目标数字。

用法：
    set KICAD_PATH=E:\\Kicad
    python scripts/verify_pcb_model.py [--data data/kicad-native-batch]

退出码：0 = 全部断言通过；1 = 有断言失败（便于当回归测试用）。
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kicad_utils.pcb_model import (  # noqa: E402
    KICAD_AVAILABLE,
    LAYER_B,
    LAYER_F,
    build_model,
    convert_rotation_index,
)

# 目标数字（docs/讨论结果/导入信息重构.md §0.1 / §1.1 / §2 / probe77）
EXPECTED = {
    "footprint": 668,
    "zero_pad_footprint": 1,
    "footprint_synthetic_ref": 23,   # 参考号为空、由我们合成 `@<库名>` 的匿名封装（EasyEDA 独立焊盘）
    "pad": 2648,
    "pad_net0": 382,
    "pad_negative_annular": 28,
    "via": 2800,
    "via_net0": 24,
    "keepout": 40,
    "keepout_slot": 39,
    "keepout_hull": 1,
    "standalone_pads": 0,
    # 语义闸门（§1.4 层 ∈ {F}/{B}/{F,B}；ref 必须板内唯一）
    "pad_layers_empty": 0,
    "footprint_ref_duplicate": 0,
}

# 非正交原件的转换角（§1.2 决策表）
ROT_TABLE = [(45.0, 0), (135.0, 1), (-45.0, 3), (-135.0, 2), (-0.62, 0), (90.0, 1), (0.0, 0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kicad-native-batch")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not KICAD_AVAILABLE:
        print("[FAIL] pcbnew 不可用：请先 set KICAD_PATH=E:\\Kicad")
        return 1

    from utils.kicad_utils.pcb_model import pcbnew  # noqa: E402  (延迟取，避免硬依赖)

    boards = sorted(glob.glob(os.path.join(args.data, "*", "*.kicad_pcb")))
    if not boards:
        print(f"[FAIL] 未找到板文件：{args.data}")
        return 1

    print(f"=== 转换数据模型验收：{len(boards)} 块板 ===")
    totals = {k: 0 for k in EXPECTED}
    layer_sig = {}
    non_orth = []
    slot_no_owner = 0
    net_names = set()
    power_nets = 0
    max_pads_in_fp = 0
    pads_with_net = 0
    pads_in_nets = 0
    hpwl_total = 0.0
    empty_nets = 0
    rev_err = 0.0
    rev_checked = 0
    owner_unmatched = 0
    keepout_shape_bad = 0

    for path in boards:
        board = pcbnew.LoadBoard(path)
        model = build_model(board, path)
        st = model.stats()
        for k in EXPECTED:
            if k in st:
                totals[k] += st[k]
            elif k == "standalone_pads":
                totals[k] += model.standalone_pads
        net_names |= set(model.nets.keys())
        power_nets += st["net_power"]
        # pin→pad 合并后的自检：net 上的 pad 数 == 有 net 的 pad 数；net 不为空；HPWL 可算
        pads_with_net += st["pad_with_net"]
        pads_in_nets += sum(len(n.pads) for n in model.nets.values())
        hpwl_total += model.hpwl()
        # 禁布区归属闸门：owner_ref 必须落在"footprint ref ∪ 零 pad 封装名"里
        # （否则"排除禁布区项所属 footprint 自己的 pad"会静默失效）
        _refs = {f.ref for f in model.footprints} | set(model.zero_pad_footprints)
        owner_unmatched += sum(1 for k in model.keepouts if k.owner_ref and k.owner_ref not in _refs)
        # 禁布区构成闸门：keepouts 必须恰好 = 板轮廓开槽 + 零 pad 封装 hull
        # （slots 与 keepouts 是同一批对象 ⇒ 多算/漏算都会在此暴露）
        if len(model.keepouts) != len(model.board_outline.slots) + len(model.zero_pad_footprints):
            keepout_shape_bad += 1
        if any(not n.pads for n in model.nets.values()):
            empty_nets += 1

        for fp in model.footprints:
            max_pads_in_fp = max(max_pads_in_fp, len(fp.pads))
            if abs(fp.rotation_deg_raw % 90.0) > 1e-6:
                non_orth.append((os.path.basename(os.path.dirname(path)), fp.ref,
                                 fp.rotation_deg_raw, fp.rotation_index))
            for pad in fp.pads:
                if pad.layers == (LAYER_F,):
                    sig = "F"
                elif pad.layers == (LAYER_B,):
                    sig = "B"
                elif set(pad.layers) == {LAYER_F, LAYER_B}:
                    sig = "FB"
                else:
                    sig = f"EMPTY/ODD{pad.layers}"      # 不再被 "FB" 掩盖（§1.4 语义闸门）
                layer_sig[sig] = layer_sig.get(sig, 0) + 1
                # 硬要求 5：pad 坐标可反解 —— 锚点 + R(−θ_转换)·rel_offset ≟ pad.center
                th = math.radians(fp.rotation_index * 90.0)
                c, s = math.cos(th), math.sin(th)
                lx, ly = pad.rel_offset
                bx = fp.anchor[0] + lx * c + ly * s
                by = fp.anchor[1] - lx * s + ly * c
                rev_err = max(rev_err, abs(bx - pad.center[0]), abs(by - pad.center[1]))
                rev_checked += 1
        for k in model.keepouts:
            if k.kind == "board_slot" and k.owner_ref is None:
                slot_no_owner += 1

        if not args.quiet:
            print(f"  {os.path.basename(os.path.dirname(path))[:44]:46s} "
                  f"fp={st['footprint']:3d} pad={st['pad']:4d}(net0 {st['pad_net0']:3d}) "
                  f"via={st['via']:4d}(net0 {st['via_net0']:2d}) "
                  f"keepout={st['keepout']:2d}(slot {st['keepout_slot']:2d}) "
                  f"net={st['net']:3d}")

    # ---------------- 板框：raw ↔ 转换 一致性（取第一块板） ----------------
    from utils.kicad_utils.kicad_utils import KiCadUtils  # noqa: E402

    board0 = pcbnew.LoadBoard(boards[0])
    raw = KiCadUtils().get_raw_board_outline(board0)
    conv = build_model(board0, boards[0]).board_outline
    rb = raw.get("polygon_bbox")
    bbox_same = rb is not None and all(abs(rb[i] - conv.bbox[i]) < 1e-6 for i in range(4))
    print(f"\n=== 板框：raw ↔ 转换（{os.path.basename(os.path.dirname(boards[0]))[:42]}） ===")
    print(f"  raw polygon_bbox        = {tuple(round(v, 3) for v in rb)}")
    print(f"  raw edges_bbox（含线宽）= {tuple(round(v, 3) for v in raw['edges_bbox'])}")
    print(f"  转换 BoardOutline.bbox  = {tuple(round(v, 3) for v in conv.bbox)}")
    print(f"  valid_area(margin=0)    = {tuple(round(v, 3) for v in conv.valid_area())}  ← 不内缩")
    print(f"  raw：板级 Edge.Cuts {len(raw['edges_items'])} 个 / 封装内嵌 {len(raw['footprint_edges'])} 个 / "
          f"外轮廓 {len(raw['polygon'])} 个 / inferred={raw['inferred']}")
    print(f"  转换：slots={len(conv.slots)} / outline_count={conv.outline_count} / inferred={conv.inferred}")
    if not bbox_same:
        failed.append(f"板框 AABB 不一致：raw {rb} vs 转换 {conv.bbox}")

    # ---------------- 断言 ----------------
    print("\n=== 对照目标数字 ===")
    failed = []
    for k, want in EXPECTED.items():
        got = totals[k]
        flag = "OK " if got == want else "FAIL"
        if got != want:
            failed.append(f"{k}: got {got}, want {want}")
        print(f"  [{flag}] {k:24s} got={got:5d}  want={want:5d}")

    print(f"\n=== 附加统计 ===")
    print(f"  pad 层签名：{layer_sig}（应无空层）")
    print(f"  pad 所在 footprint 内 pad 数最大值：{max_pads_in_fp}")
    print(f"  禁布区未匹配到归属的开槽：{slot_no_owner}（应尽量为 0）")
    print(f"  禁布区归属无法匹配到 footprint 的数量：{owner_unmatched}（必须为 0）")
    print(f"  net 总数：{len(net_names)}；其中电源/地网：{power_nets}；空 net 的板：{empty_nets}")
    print(f"  net→pad 覆盖：有 net 的 pad {pads_with_net} 个，落在 net 里的 {pads_in_nets} 个（应相等）")
    print(f"  20 板 HPWL（排除电源/地网，全部连接点）合计：{hpwl_total:.1f} mm")
    print(f"  反解自检（硬要求 5）：{rev_checked} 个 pad，最大误差 {rev_err:.6f} mm（含翻面件）")
    print(f"  非正交原件：{len(non_orth)} 个（应为 16）")
    for nm, ref, raw, idx in sorted(non_orth):
        print(f"      {nm[:36]:38s} {ref:5s} raw={raw:8.2f}° → idx={idx} ({idx * 90}°)")

    print("\n=== 转换角规则自检 ===")
    for deg, want in ROT_TABLE:
        got = convert_rotation_index(deg)
        flag = "OK " if got == want else "FAIL"
        if got != want:
            failed.append(f"rotation {deg}° → {got}, want {want}")
        print(f"  [{flag}] {deg:7.2f}° → {got} ({got * 90}°)  want {want}")

    if len(non_orth) != 8:
        failed.append(f"非正交原件数 got {len(non_orth)}, want 8")
    if pads_in_nets != pads_with_net:
        failed.append(f"net→pad 覆盖不一致：net 内 {pads_in_nets} ≠ 有 net 的 pad {pads_with_net}")
    if empty_nets:
        failed.append(f"存在空 net 的板：{empty_nets}")
    if slot_no_owner:
        failed.append(f"有 {slot_no_owner} 个开槽未匹配到归属 footprint")
    if owner_unmatched:
        failed.append(f"有 {owner_unmatched} 个禁布区 owner_ref 匹配不到任何 footprint（自家槽排除会失效）")
    if keepout_shape_bad:
        failed.append(f"有 {keepout_shape_bad} 块板的禁布区构成异常（≠ 开槽 + 零 pad 封装 hull）")
    if rev_checked != totals["pad"]:
        failed.append(f"反解自检覆盖不全：{rev_checked} ≠ pad {totals['pad']}")
    if rev_err > 1e-3:
        failed.append(f"pad 坐标反解误差过大：{rev_err:.6f} mm")
    if any(v == 0 for v in layer_sig.values()) or sum(layer_sig.values()) != totals["pad"]:
        failed.append(f"pad 层签名异常：{layer_sig}")

    print()
    if failed:
        print("❌ 验收失败：")
        for f in failed:
            print("   -", f)
        return 1
    print("✅ 验收通过：转换数据模型与《导入信息重构》的目标数字完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
