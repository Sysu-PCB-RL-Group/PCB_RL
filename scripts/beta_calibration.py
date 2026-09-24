# -*- coding: utf-8 -*-
"""**β 的离线体检 / 对账**（不产生产物文件）。

```text
β[board] = BETA_MARGIN · β_crit(该板)
  β_crit = max over m = 0…n−1 [ (H_n − H_m)/H_ref − α·(σ_n − σ_m)/D ] / (n − m)   ← 沿【专家】曲线
  ⇒ β 由**板本身现算**（`env.reward.beta_crit_of`，~30 ms/板，纯 numpy）⇒ 无需离线表 / 无需 SHA1 守门

& python scripts/beta_calibration.py            # 逐板：β_crit / β / β·n / 与旧口径 k/n 的对比
& python scripts/beta_calibration.py --check     # 只判定：margin ≥ 1（某板要求若未被覆盖 ⇒ 退出码 1）
& python scripts/beta_calibration.py --margin 1.5
```

判据（β 的下界从哪来）：任意**能停**的前缀（`L = ∅`）都必须**劣于**把剩下的放完 ⇒
`β ≥ β_crit`；低于它会出现"把尾部堵死 ⇒ 提前终止 ⇒ HPWL 天然小 ⇒ 回报反而更高"。
`β_crit = 0` 的板（σ 收益已超过继续下去的 HPWL 代价）⇒ β = 0 是**合法结果**，不是 bug。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from env import order_plan as op                                      # noqa: E402
from env import reward as rw                                          # noqa: E402
from utils.kicad_utils import model_cache as mc                       # noqa: E402

OLD_K = 2.5          # 旧口径（全局经验式 β = OLD_K / n）—— 保留作对照


def board_row(board, margin):
    model = mc.load_model(board, allow_build=False)
    order = list(op.plan_order(model))
    n = len(order)
    H_ref, D = rw.h_ref_of(model), rw.d_board_of(model)
    crit, m_star = rw.beta_crit_of(model, order)
    beta = margin * crit
    return {"board": board, "n": n, "H_ref": H_ref, "D": D, "crit": crit, "m_star": m_star,
            "beta": beta, "beta_n": beta * n, "crit_n": crit * n, "old": OLD_K / n if n else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="β 的离线体检（HPWL 早停惩罚要求 vs 实际 β）")
    ap.add_argument("--margin", type=float, default=rw.BETA_MARGIN,
                    help=f"β 的余量（默认 env.reward.BETA_MARGIN = {rw.BETA_MARGIN}）")
    ap.add_argument("--cache-dir", default=None, help="模型缓存目录（默认 data/model_cache）")
    ap.add_argument("--boards", nargs="*", default=None, help="只看这些板（默认全部）")
    ap.add_argument("--check", action="store_true", help="只判定（margin < 1 或出现 NaN/负值 ⇒ 退出码 1）")
    args = ap.parse_args(argv)

    try:
        boards = args.boards if args.boards else [e["board"] for e in mc.read_index(root=args.cache_dir)["boards"]]
    except mc.CacheError as exc:
        print(f"{exc}")
        return 2
    if not boards:
        print("没有可体检的板（先跑 scripts/dump_models.py 生成缓存）")
        return 2

    rows = [board_row(b, args.margin) for b in boards]
    if not args.check:
        print(f"α = {rw.ALPHA}｜margin = {args.margin}｜β = margin · β_crit(该板)｜"
              f"板 {len(rows)} 块｜缓存 {mc.cache_dir(args.cache_dir)}")
        print(f"{'板':22s} {'n':>3s} {'H_ref':>9s} {'β_crit':>8s} {'m*':>4s} {'β':>8s} "
              f"{'β·n':>6s} {'旧口径 β=k/n':>12s} {'新/旧':>6s}")
        print("-" * 96)
        for r in rows:
            print(f"{r['board'][:22]:22s} {r['n']:3d} {r['H_ref']:9.2f} {r['crit']:8.4f} "
                  f"{str(r['m_star']):>4s} {r['beta']:8.4f} {r['beta_n']:6.3f} {r['old']:12.4f} "
                  f"{(r['beta'] / r['old'] if r['old'] > 0 else float('nan')):6.2f}")

    crits = np.array([r["crit"] for r in rows])
    betas = np.array([r["beta"] for r in rows])
    bn = np.array([r["beta_n"] for r in rows])
    bad = (not np.isfinite(betas).all()) or (betas < 0).any() or args.margin < 1.0
    print(f"\nβ_crit：min {crits.min():.4f}｜p50 {np.median(crits):.4f}｜max {crits.max():.4f}"
          f"（β_crit = 0 的板 {int((crits == 0).sum())} 块 ⇒ 合法）")
    print(f"β     ：min {betas.min():.4f}｜p50 {np.median(betas):.4f}｜max {betas.max():.4f}")
    print(f"β·n（完成项整局分量 / 目标）：min {bn.min():.3f}｜p50 {np.median(bn):.3f}｜max {bn.max():.3f}"
          f"   ← 旧口径恒为 {OLD_K}（HPWL 项的 2.5 倍）")
    print(f"margin = {args.margin} ⇒ {'✅ 每板 β ≥ 该板要求' if args.margin >= 1.0 else '❌ margin < 1'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
