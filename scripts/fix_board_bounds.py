"""统一 data/模拟/ 下所有 PCB 板框为 120×80mm。

固定元件超出新边界的自动 clamp 到板内。
"""

import json
import os

TARGET_W, TARGET_H = 120.0, 80.0
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "模拟")

for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(SRC_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_w = data["board_bounds"][1][0]
    old_h = data["board_bounds"][1][1]

    data["board_bounds"] = [[0, 0], [TARGET_W, TARGET_H]]

    moved = 0
    for comp in data["components"]:
        if not comp["is_fixed"]:
            continue
        x1, y1, x2, y2 = comp["bbox"]
        w = x2 - x1
        h = y2 - y1

        # 超出右边界 → 贴右边缘
        if x2 > TARGET_W:
            x1 = TARGET_W - w
            x2 = TARGET_W
            moved += 1
        # 超出上边界 → 贴上边缘
        if y2 > TARGET_H:
            y1 = TARGET_H - h
            y2 = TARGET_H
            moved += 1
        # 负坐标也修正
        if x1 < 0:
            x1 = 0
            x2 = w
            moved += 1
        if y1 < 0:
            y1 = 0
            y2 = h
            moved += 1

        comp["bbox"] = [x1, y1, x2, y2]

    # 写入前自动备份到 .bak
    import shutil
    bak_path = fpath + ".bak"
    shutil.copy2(fpath, bak_path)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    name = data["name"]
    print(f"  [{fname}] {name}: {old_w:.0f}x{old_h:.0f} -> {TARGET_W}x{TARGET_H}, "
          f"{moved} fixed comps repositioned")

print(f"\nDone: all boards now {TARGET_W}x{TARGET_H}mm")
