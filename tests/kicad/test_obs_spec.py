# -*- coding: utf-8 -*-
"""obs 口径（台账 §5.5；Owner 2026-09-22 定案）的回归测试。

覆盖：
  A. `piece_features`：与**独立重算**逐件一致；覆盖 `plan_order` 全集；件数 / `非电源 net 数 = 0` 的件数
   = **数据集金值**（2026-09-22 排除 4 块 C1/C2 板后：**668 件 / 153 件**；旧 20 板为 857 / 201）
  B. 归一化：端点精确（`0 ⇒ 0.0`、`MAX ⇒ 1.0`）、单调、超界不裁剪、非法上界抛错；`t/n` 边界
  C. **守门（最重要）**：全 20 板全部件的两个归一值 ∈ [0,1]，且**实测 max 恰好 == 常数**
     （⇒ 换数据集/加板会直接失败，逼我们重测常数，而不是静默越界）
  D. `channel_of`：8 通道、rot 在最前、双射（两两不重叠）、非法入参抛错
"""
import os

import numpy as np
import pytest

from env import grid_render as gr
from env import obs_spec as osp
from env import order_plan as op

pytestmark = pytest.mark.require_kicad


def _pick(boards, sub):
    """按板目录名（唯一子串）取 `(path, board, model)`。"""
    hits = [(p, b, m) for p, b, m, _r in boards if sub in os.path.basename(os.path.dirname(p))]
    assert len(hits) == 1, f"板名 {sub!r} 匹配到 {len(hits)} 块（须唯一）"
    return hits[0]


# ---------------- A. piece_features 与独立重算 ----------------

def test_piece_features_matches_independent_recount(boards_with_models):
    """两个量必须与"直接用 model 数一遍"逐件一致；并钉住数据集金值。"""
    n_pieces = 0
    n_np_zero = 0
    for _p, _b, model, _r in boards_with_models:
        feats = op.piece_features(model)
        order = tuple(op.plan_order(model))
        assert set(feats) == set(order), "piece_features 的键集必须覆盖 plan_order 全集"

        for f in model.footprints:
            np_nets = {p.net_id for p in f.pads
                       if p.has_net and not model.nets[p.net_id].is_power}
            assert feats[f.ref] == (len(f.pads), len(np_nets)), \
                f"{f.ref}: 期望 {(len(f.pads), len(np_nets))}，得到 {feats[f.ref]}"
            n_np_zero += int(len(np_nets) == 0)
            n_pieces += 1

    # 金值（16 板；2026-09-22 把 4 块 C1/C2 板移入 data/bad_data 后由 scripts/dataset_report.py 重测）
    assert n_pieces == 668, f"数据集件数变了：{n_pieces}（金值 668）"
    assert n_np_zero == 153, (
        f"『非电源 net 数 = 0』的件数变了：{n_np_zero}（金值 153 = 目标零贡献件 22.9%）")


# ---------------- B. 归一化函数本身 ----------------

def test_norm_count_endpoints_monotonic_and_overflow():
    """端点精确命中、单调、超界不裁剪、非法上界抛错。"""
    assert osp.norm_count(0, osp.PAD_MAX) == 0.0
    assert osp.norm_count(osp.PAD_MAX, osp.PAD_MAX) == 1.0
    assert osp.norm_count(osp.NET_MAX, osp.NET_MAX) == 1.0

    xs = [0, 1, 2, 4, 8, 16, 32, 48, 64, osp.PAD_MAX]
    vals = [osp.norm_count(x, osp.PAD_MAX) for x in xs]
    assert all(vals[i] < vals[i + 1] for i in range(len(xs) - 1)), "必须严格单调"
    assert all(0.0 <= v <= 1.0 for v in vals)

    # 取 log1p 的理由（量化）：中位件（2 pad）看得见 —— 线性只有 2/80 = 0.025
    # （独立算：log1p(2)/log1p(80) = 1.0986/4.3944 = 0.2500）
    assert osp.norm_count(2, osp.PAD_MAX) == pytest.approx(0.250, abs=0.002)

    assert osp.norm_count(120, osp.PAD_MAX) > 1.0, "超界必须不裁剪（保留『异常大』的信息）"
    with pytest.raises(ValueError):
        osp.norm_count(1, 0)


def test_piece_vector_and_progress_boundaries():
    """`piece_vector` 的组装顺序与边界；`t/n` 与 `completion.py` 的边界口径一致。"""
    assert osp.PIECE_KEYS == ("pad", "np_net", "progress")
    v = osp.piece_vector(osp.PAD_MAX, osp.NET_MAX, 0, 10)
    assert v == (1.0, 1.0, 0.0)

    pad_n, net_n, t_n = osp.piece_vector(2, 0, 5, 10)
    assert pad_n == pytest.approx(0.250, abs=0.002)
    assert net_n == 0.0, "非电源 net 数 = 0 ⇒ 精确 0（′无信号′的专用编码）"
    assert t_n == pytest.approx(0.5)

    assert osp.norm_progress(10, 10) == 1.0
    assert osp.norm_progress(0, 0) == 1.0, "n = 0 ⇒ 1.0（与 completion.py 边界一致）"
    assert osp.norm_progress(99, 10) == 1.0, "上夹"
    assert osp.norm_progress(-1, 10) == 0.0, "下夹"


# ---------------- C. 守门：常数不能漂 ----------------

def test_guard_dataset_max_equals_constants(boards_with_models):
    """**守门断言**：全数据集归一值 ∈ [0,1]，且实测 max **恰好等于** `PAD_MAX`/`NET_MAX`。

    这条同时守两件事：① 换数据集/加板 ⇒ 超界 ⇒ 失败（逼重测常数）；
    ② 有人把常数无理由调大 ⇒ 失败（常数必须等于实测 max）。
    """
    max_pad = max_net = 0
    min_pad_norm = 1.0
    n_with_net = 0
    for _p, _b, model, _r in boards_with_models:
        for _ref, (npads, nnet) in op.piece_features(model).items():
            pn, nn = osp.norm_pad_count(npads), osp.norm_np_net_count(nnet)
            assert 0.0 <= pn <= 1.0, f"pad 数归一越界：{npads} ⇒ {pn}"
            assert 0.0 <= nn <= 1.0, f"非电源 net 数归一越界：{nnet} ⇒ {nn}"
            max_pad, max_net = max(max_pad, npads), max(max_net, nnet)
            min_pad_norm = min(min_pad_norm, pn)
            n_with_net += int(nnet > 0)

    assert max_pad == osp.PAD_MAX, f"实测最大 pad 数 {max_pad} ≠ PAD_MAX {osp.PAD_MAX} ⇒ 常数该重测"
    assert max_net == osp.NET_MAX, f"实测最大非电源 net 数 {max_net} ≠ NET_MAX {osp.NET_MAX} ⇒ 常数该重测"
    assert min_pad_norm > 0.0, "最小 pad 数 = 1 ⇒ 归一值 > 0（不会与 net 的 0 混淆）"
    assert n_with_net > 0 and n_with_net < 668, "样本须同时含『有 net』与『无 net』两端"


# ---------------- D. spatial 通道序 ----------------

def test_channel_order_is_rot_major_and_bijective():
    """通道序 `c = rot·2 + (0 = L, 1 = G)`：8 通道、rot 在最前、双射。"""
    assert osp.SPATIAL_CHANNELS == osp.N_ROT * 2 == 8
    seen = set()
    for rot in range(osp.N_ROT):
        for which in ("L", "G"):
            c = osp.channel_of(rot, which)
            assert 0 <= c < osp.SPATIAL_CHANNELS
            assert c not in seen, f"通道 {c} 被两个 (rot, which) 占用"
            seen.add(c)
    assert seen == set(range(osp.SPATIAL_CHANNELS))

    assert osp.channel_of(0, "L") == 0 and osp.channel_of(0, "G") == 1
    assert osp.channel_of(2, "l") == 4, "大小写宽容"
    assert osp.channel_of(3, "G") == 7

    for bad in ((-1, "L"), (4, "L"), (0, "X"), (0, "")):
        with pytest.raises(ValueError):
            osp.channel_of(*bad)


# ---------------- E. G 的 −inf 净化（A3） ----------------

def test_sanitize_gain_synthetic_and_shape_guard():
    """合成用例：非法格填 `GAIN_FILL`、合法格原样、不改入参、形状不符抛错。"""
    G = np.array([[0.5, -np.inf], [-np.inf, 0.25]], dtype=np.float32)
    L = np.array([[True, False], [False, True]])
    keep = G.copy()

    out = osp.sanitize_gain(G, L)
    assert out.tolist() == [[0.5, osp.GAIN_FILL], [osp.GAIN_FILL, 0.25]]
    assert out.dtype == np.float32
    assert np.array_equal(G, keep), "不得修改入参"
    assert osp.GAIN_FILL == 0.0, "填 0（与合法格 [0,1] 同域；填 −1 会被 BN 放大 ⇒ 不采用）"

    with pytest.raises(ValueError):
        osp.sanitize_gain(np.zeros((2, 3), dtype=np.float32), L)


def test_sanitize_gain_on_real_board_removes_all_nonfinite(boards_with_models):
    """真实板：`gain_maps` 的非法格确实是 `−inf`（前置样本非空）⇒ 净化后**全有限**、
    合法格一个不动、且 `L` 本身不被改动（mask 契约不受影响）。"""
    _p, _b, model = _pick(boards_with_models, "bliysfyt")
    order = tuple(op.plan_order(model))
    comp = {f.ref: f for f in model.footprints}[order[0]]
    spec = gr.grid_spec(model, m=gr.GRID_M)

    L = gr.legal_masks(model, [], comp, spec)
    G, _degen = gr.gain_maps(model, [], comp, spec)
    assert not np.isfinite(G).all(), "前置条件：G 的非法格里应当有 −inf（否则样本为空）"
    assert np.isinf(G[~L]).all(), "契约：G 的有限域必须严格等于 L（非法格 = −inf）"

    L_keep, G_keep = L.copy(), G.copy()
    out = osp.sanitize_gain(G, L)
    assert np.isfinite(out).all()
    assert np.array_equal(out[L], G[L]), "合法格的值必须原样保留"
    assert np.all(out[~L] == osp.GAIN_FILL)
    assert np.array_equal(L, L_keep) and np.array_equal(G, G_keep)
