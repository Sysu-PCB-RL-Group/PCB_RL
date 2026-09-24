# -*- coding: utf-8 -*-
"""把板子预处理成**离线模型缓存**（Owner 2026-09-22 拍案 P1）。

```text
$env:KICAD_PATH="E:\Kicad"
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dump_models.py            # dump 全部板
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dump_models.py --check     # 只体检（不写，有问题退出码 1）
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dump_models.py --boards jackt-... lr1388-...
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/dump_models.py --data data/bad_data --out /tmp/x
```

产物（默认 `data/model_cache/`）：

```text
index.json     板清单（按板名 sorted）+ 每板源路径/SHA1/缓存版本/对象计数/生成时间
<board>.pkl    {header, payload}；payload 只含 `PcbModel` 的构造字段（派生字段由 `__post_init__` 重建）
```

**为什么**：训练机（如 AutoDL）装不了 KiCad，而我们对 `pcbnew` 的依赖只有"读板那一次"。
dump 一次后，训练/推理只读缓存 ⇒ 不依赖 KiCad、几何口径被一次性冻结（本机 KiCad 10.0.4）。
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils.kicad_utils import pcb_model as pm                       # noqa: E402
from utils.kicad_utils import model_cache as mc                     # noqa: E402

DEF_DATA = os.path.join("data", "kicad-native-batch")


def boards_of(data_dir: str) -> list:
    """板目录名（sorted；确定性顺序 = 训练时的板序）。"""
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and mc.source_of(d, data_dir))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA, help="源板根目录（默认 data/kicad-native-batch）")
    ap.add_argument("--out", default=None, help="缓存目录（默认 data/model_cache）")
    ap.add_argument("--boards", nargs="*", default=None, help="只处理这些板（板目录名；默认全部）")
    ap.add_argument("--check", action="store_true", help="只体检：缓存是否存在/版本/源是否变了")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    if not os.path.isdir(data_dir):
        print(f"源目录不存在：{data_dir}")
        return 2

    if args.check:
        rows = mc.cache_status(root=args.out, data_dir=data_dir)
        if not rows:
            print("缓存索引为空 ⇒ 先跑一次 dump（不带 --check）")
            return 1
        bad = 0
        print(f"{'板':46s} {'缓存':>4s} {'版本':>4s} {'源已变':>6s}  计数")
        print("-" * 110)
        for r in rows:
            flag = "?" if r["stale"] is None else ("是" if r["stale"] else "否")
            ok = r["cached"] and r["cache_version"] == mc.CACHE_VERSION and r["stale"] is not True
            bad += 0 if ok else 1
            print(f"{r['board'][:46]:46s} {str(r['cached']):>4s} {str(r['cache_version']):>4s} "
                  f"{flag:>6s}  {r['counts']}   {'OK' if ok else '⚠️ 需重 dump'}")
        print(f"\n需要重 dump 的板：{bad} / {len(rows)}（版本要求 {mc.CACHE_VERSION}）")
        return 1 if bad else 0

    if not getattr(pm, "KICAD_AVAILABLE", False):
        print("⚠️ KiCad 不可用（KICAD_AVAILABLE=False）⇒ dump 必须在装有 KiCad 的机器上跑"
              "（设好 KICAD_PATH 后再试）")
        return 2

    boards = args.boards if args.boards else boards_of(data_dir)
    if not boards:
        print(f"{data_dir} 下没有可处理的板目录")
        return 2

    print(f"源目录 {data_dir}｜待处理 {len(boards)} 块板｜输出 {mc.cache_dir(args.out)}")
    entries, tot = [], {"footprints": 0, "pads": 0, "vias": 0, "keepouts": 0, "nets": 0}
    for i, board in enumerate(boards, 1):
        src = mc.source_of(board, data_dir)
        if not src:
            print(f"  [{i}/{len(boards)}] {board}: 跳过（找不到 .kicad_pcb）")
            continue
        model = pm.build_model(pm.pcbnew.LoadBoard(src), src)
        header = mc.dump_model(model, src, root=args.out)
        entries.append(header)
        for k, v in header["counts"].items():
            tot[k] += v
        print(f"  [{i}/{len(boards)}] {board[:52]:52s} fp={header['counts']['footprints']:3d} "
              f"pad={header['counts']['pads']:4d} via={header['counts']['vias']:4d} "
              f"ko={header['counts']['keepouts']:2d} net={header['counts']['nets']:3d}")
    idx = mc.write_index(entries, root=args.out)
    print(f"\n已写 {len(entries)} 个 .pkl + 索引 {idx}")
    print(f"合计：{tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
