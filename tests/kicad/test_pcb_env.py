# -*- coding: utf-8 -*-
"""`env/pcb_env.py`（环境层，台账 §9.1）的回归测试。

**不需要 KiCad**：板子一律走 P1 缓存（`data/model_cache/`）⇒ 本文件在训练机上也能跑
（这与 `test_no_kicad_runtime.py` 是同一立场：训练路径不得依赖 KiCad）。

覆盖：
  A. reset 契约：obs 形状/范围/通道序（= `sanitize(L,G)` 逐通道相等）｜`action_masks()` 与 `L` 同帧
     ｜推理档 `mask == L`、训练档 `mask ⊆ L`｜`info` 含 §9.1.6 键
  B. 单步 reward == 手算（独立重算 `hpwl` / `dispersion_of`，§7.3）
  C. **mask 外的动作被阻挡**（Owner 2026-09-22 ⑦）：L 外 / L 内 mask 外 两种都必须抛 `EnvActionError`
  D. 整局：终止两条分支（放完 ∨ mask 空）｜`truncated` 恒 False｜完成度 == `n_placed/n_total`
     ｜终止后 `step` 抛 `EnvDoneError` 且 `action_masks()` 全 False
  E. 末件（`next_comp is None`）⇒ `mask == L`（Owner 2026-09-22 ⑨）
  F. 执行复核（§5.7/§6.3）：`L` 说合法而 `Judge` 判违规 ⇒ `EnvContractError`；`verify=False` 时不拦
  G. 确定性 + 动作编码（`(k,i,j)` 与一维 `k·m²+i·m+j` 等价）
  H. 结构性死板（开局 `L` 就空）：默认重采（`skipped_dead_start` 有记录）；`resample_dead_start=False`
     ⇒ 如实报 `structural` + `step` 抛 `EnvDoneError`
"""
import time

import numpy as np
import pytest

from env import board_state as bs
from env import grid_render as gr
from env import obs_spec as osp
from env import order_plan as op
from env import pcb_env as pe
from env import reward as rw
from utils.kicad_utils import model_cache as mc

INFO_KEYS = ("completion", "completion_count", "completion_area", "completion_contrib",
             "n_placed", "n_total", "deadlock_kind", "at", "audit_failures",
             "board", "H_ref", "degenerate")


# ===================== 选板（一次扫描，全文件复用） =====================

_SCAN = None


def _scan():
    """扫缓存里的板：`[(board, L0 合法格数), …]`（L0 = 开局、顺序第一件的合法集大小）。"""
    global _SCAN
    if _SCAN is None:
        try:
            names = [e["board"] for e in mc.read_index()["boards"]]
        except mc.CacheError:
            pytest.skip("没有 data/model_cache/index.json（先跑 scripts/dump_models.py）")
        rows = []
        for b in names:
            model = mc.load_model(b, allow_build=False)
            st = bs.BoardState.start(model, tuple(op.plan_order(model)))
            if st.done:
                continue
            spec = gr.grid_spec(model, m=gr.GRID_M)
            L0 = gr.legal_masks(model, st.placed, st.comp, spec, keepouts=st.keepouts)
            rows.append((b, int(L0.sum())))
        _SCAN = rows
    return _SCAN


def _cheap() -> str:
    """`L0` 最小的板（`L` 越小 ⇒ 本文件的整局测试越快；不是"训练用板"的口径）。"""
    live = [r for r in _scan() if r[1] > 0]
    assert live, "缓存里所有板开局都没有合法位置"
    return min(live, key=lambda r: r[1])[0]


def _structural():
    dead = [r for r in _scan() if r[1] == 0]
    return dead[0][0] if dead else None


def _dense() -> str:
    """`L0` 最大的板（用于吞吐快照；`lookahead` 的成本 ∝ `|L|`，台账 §6.3 实测 428 ms/步那一档）。"""
    live = [r for r in _scan() if r[1] > 0]
    return max(live, key=lambda r: r[1])[0]


def _first_cell(mask) -> tuple:
    k, i, j = (int(v) for v in np.argwhere(mask)[0])
    return k, i, j


def _own_L(env) -> np.ndarray:
    """独立重算 env 当前局面的 `L`（同帧：同一 `placed` + 同一 `keepouts`）。"""
    st = env.state
    return gr.legal_masks(env.model, st.placed, st.comp, env.grid, keepouts=st.keepouts)


# ===================== A. reset 契约 =====================

def test_reset_obs_and_mask_contract():
    """obs = `sanitize(L, G)` 逐通道 + `piece`；推理档 `mask == L`；`info` 键齐。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=False)              # 推理档：mask = L（§6.3）
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs), "obs 越界（空间声明与实际不符）"
    assert set(obs) == {pe.OBS_SPATIAL, pe.OBS_PIECE}
    st, frame = env.state, env.frame

    L = _own_L(env)
    assert np.array_equal(frame.legal, L)
    assert np.array_equal(frame.mask, L) and not frame.lookahead_applied
    assert np.array_equal(env.action_masks(), L)
    assert env.action_masks().dtype == bool

    G, degen = gr.gain_maps(env.model, st.placed, st.comp, env.grid, legal=L, keepouts=st.keepouts)
    gs = osp.sanitize_gain(G, L)
    sp = obs[pe.OBS_SPATIAL]
    assert sp.shape == (osp.SPATIAL_CHANNELS, gr.GRID_M, gr.GRID_M) and sp.dtype == np.float32
    for k in range(osp.N_ROT):
        assert np.array_equal(sp[osp.channel_of(k, "L")], L[k].astype(np.float32))
        assert np.array_equal(sp[osp.channel_of(k, "G")], gs[k])
    assert np.isfinite(sp).all(), "spatial 里有非有限值（G 的 −inf 没被净化，§5.5/A3）"
    assert sp.min() >= 0.0 and sp.max() <= 1.0

    pad, np_net = env.features[st.comp.ref]
    assert np.array_equal(obs[pe.OBS_PIECE],
                          np.asarray(osp.piece_vector(pad, np_net, 0, st.n_total), np.float32))

    for key in INFO_KEYS:
        assert key in info, f"info 少了 §9.1.6 的键：{key}"
    assert info["board"] == b and info["n_placed"] == 0 and info["n_total"] == len(env.order)
    assert info["completion"] == 0.0
    assert info["H_ref"] == pytest.approx(rw.h_ref_of(env.model))
    assert info["deadlock_kind"] is None and info["at"] is None and info["terminated"] is False


def test_training_mask_is_subset_of_L_and_matches_lookahead_info():
    """训练档：`mask ⊆ L` 且与 `lookahead_mask` 自报的 `kept` 一致（末件除外）。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=True)
    env.reset(seed=0)
    frame = env.frame
    assert not (frame.mask & ~frame.legal).any(), "mask 必须 ⊆ L（§6.3）"
    if frame.lookahead_applied:
        info = frame.lookahead_info
        assert info["kept"] == int(frame.mask.sum())
        assert info["cut"] == int(frame.legal.sum()) - int(frame.mask.sum())


# ===================== B. reward 手算对拍 =====================

def test_step_reward_matches_manual_formula():
    """`r == (−ΔHPWL)/H_ref + α·Δσ/D + β`（全部独立重算；§7.3）。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=False)
    obs0, _ = env.reset(seed=0)
    st0 = env.state
    h0 = env.model.hpwl(refs=set(st0.placed_refs), poses=st0.poses)
    s0 = gr.dispersion_of(env.model, st0.placed)

    k, i, j = _first_cell(env.action_masks())
    obs1, r, terminated, truncated, info = env.step((k, i, j))
    st1 = env.state
    h1 = env.model.hpwl(refs=set(st1.placed_refs), poses=st1.poses)
    s1 = gr.dispersion_of(env.model, st1.placed)

    want = rw.step_reward(d_hpwl=h1 - h0, d_sigma=s1 - s0, h_ref=rw.h_ref_of(env.model),
                          d_board=rw.d_board_of(env.model),
                          beta=rw.beta_of(env.model, env.order))
    assert r == pytest.approx(want, rel=1e-12, abs=1e-12)
    assert truncated is False and isinstance(terminated, bool)
    assert info["n_placed"] == 1
    assert info["completion"] == pytest.approx(1.0 / len(env.order))
    assert obs0[pe.OBS_PIECE][2] == 0.0                                     # t/n：0 → 1/n
    pad, np_net = env.features[env.state.comp.ref]                          # 当前件已换 → 特征跟着换
    assert np.array_equal(obs1[pe.OBS_PIECE],
                          np.asarray(osp.piece_vector(pad, np_net, 1, len(env.order)), np.float32))
    assert not np.array_equal(obs0[pe.OBS_SPATIAL], obs1[pe.OBS_SPATIAL])   # 局面确实变了


# ===================== C. mask 外的动作被阻挡 =====================

def test_masked_actions_are_blocked():
    """Owner 2026-09-22 ⑦：落在 `mask` 之外的动作用 `EnvActionError` 挡住（不静默、不退化）。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=True)
    env.reset(seed=0)
    frame = env.frame

    outside_L = np.argwhere(~frame.legal)
    assert len(outside_L), "L 不该是满的"
    with pytest.raises(pe.EnvActionError, match="L 之外"):
        env.step(tuple(int(v) for v in outside_L[0]))

    cut = np.argwhere(frame.legal & ~frame.mask)               # L 内但被 lookahead 砍掉
    if len(cut):
        with pytest.raises(pe.EnvActionError, match="掩膜"):
            env.step(tuple(int(v) for v in cut[0]))

    k, i, j = _first_cell(env.action_masks())                  # 合法动作照常前进
    _, _r, _term, _trunc, info = env.step((k, i, j))
    assert info["n_placed"] == 1


# ===================== D. 整局 =====================

def test_full_episode_terminates_with_completion():
    """跑完整局：终止 ∈ {放完, mask 空}；`truncated` 恒 False；完成度自洽；终止后不许再 step。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=True)
    env.reset(seed=0)
    n = len(env.order)
    steps, total = 0, 0.0
    info = {}
    last_index = -1
    while True:
        mask = env.action_masks()
        assert mask.any(), "还没 terminated 但 mask 已空（终止判定漏了）"
        if env.state.next_comp is None:
            # 走到末件那一帧：Owner ⑨ ⇒ mask 必须退化为 L（见 E 的专项断言）
            assert np.array_equal(mask, _own_L(env))
        k, i, j = _first_cell(mask)
        last_index = int(env.state.index)
        _obs, r, terminated, truncated, info = env.step((k, i, j))
        steps += 1
        total += r
        assert truncated is False, "§9.1.5 ⑧：不加步数上限 ⇒ truncated 恒 False"
        if terminated:
            break
        assert steps <= n, "步数超过件数（每步必放一件）"

    assert steps <= n and np.isfinite(total)
    assert info["n_placed"] == steps and info["n_total"] == n
    assert info["completion"] == pytest.approx(steps / n)
    assert info["completion"] == info["completion_count"]

    at, kind = info["at"], info["deadlock_kind"]
    if at is None:                                             # 分支①：全部放完
        assert steps == n and kind is None
    else:
        assert at in ("L", "mask") and kind in pe.DEADLOCK_KINDS
        assert (kind == "structural") == (steps == 0), "只有「开局就无解」才叫 structural"
        if at == "L":
            assert kind in ("structural", "policy")
        else:
            assert kind == "order_locked"

    with pytest.raises(pe.EnvDoneError):
        env.step((0, 0, 0))
    assert not env.action_masks().any()
    assert last_index >= 0
    print(f"[env 实测] 整局 {b}: n={n} steps={steps} completion={info['completion']:.4f} "
          f"at={info['at']} kind={info['deadlock_kind']} Σr={total:.4f}")


# ===================== E. 末件 ⇒ mask = L =====================

class _LastPieceState:
    """把真实状态包一层、只把 `next_comp` 变成 `None`（专测 `_make_frame` 的末件分支）。"""

    def __init__(self, st):
        self._st = st

    comp = property(lambda self: self._st.comp)
    placed = property(lambda self: self._st.placed)
    keepouts = property(lambda self: self._st.keepouts)
    next_comp = None


def test_last_piece_mask_degrades_to_L():
    """Owner 2026-09-22 ⑨：末件没有"下一个件" ⇒ `mask = L`（不调 lookahead）。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=True)
    env.reset(seed=0)
    frame = env._make_frame(_LastPieceState(env.state))
    assert frame.lookahead_applied is False
    assert np.array_equal(frame.mask, frame.legal)
    assert np.array_equal(frame.mask, _own_L(env))


# ===================== F. 执行复核 =====================

class _BadJudge:
    """永远判违规的替身（只为验证"不一致即报错"这条纪律会被触发）。"""

    def __init__(self, *a, **kw):
        pass

    def judge(self, *a, **kw):
        class _R:
            is_violation = True

            @staticmethod
            def describe():
                return "stub：故意判违规"

        return _R()


# ===================== I. 吞吐快照（打印实测，不做性能断言） =====================

def test_throughput_snapshot_sparse_and_dense():
    """吞吐快照（台账 §9.1.5 ③ 的输入）：reset / step 实测耗时 + `lookahead` 的砍量。

    **只打印 + 给一个宽松上限哨兵**（不做性能断言；数字进台账/报告，用来决定"要不要优化"）。
    """
    for tag, b in (("稀疏", _cheap()), ("稠密", _dense())):
        env = pe.PcbEnv(board=b, use_lookahead=True)
        t0 = time.perf_counter()
        env.reset(seed=0)
        t_reset = time.perf_counter() - t0
        frame = env.frame
        cut = int(frame.legal.sum()) - int(frame.mask.sum())

        t1 = time.perf_counter()
        env.step(_first_cell(env.action_masks()))
        t_step = time.perf_counter() - t1

        n_legal, n_mask, degen = int(frame.legal.sum()), int(frame.mask.sum()), frame.degenerate
        print(f"[env 实测] {tag} {b}: n={len(env.order)} L0={n_legal} mask0={n_mask} "
              f"砍={cut} degenerate={degen} β={env.beta:.4f} H_ref={env.h_ref:.3f} "
              f"D={env.d_board:.2f} reset={t_reset:.2f}s step={t_step:.2f}s")

        assert n_mask > 0 and 0 <= cut
        assert not (frame.mask & ~frame.legal).any()
        assert t_reset < 60.0 and t_step < 30.0        # 哨兵上限（远超实测，不会误红）


# ===================== F. 执行复核 =====================

def test_verify_gate_raises_on_L_judge_disagreement(monkeypatch):
    """§6.3/§5.7：`L` 说合法而 `Judge` 判违规 ⇒ `EnvContractError`；`verify=False` 时不拦。"""
    b = _cheap()
    env = pe.PcbEnv(board=b, use_lookahead=False, verify=True)
    env.reset(seed=0)
    k, i, j = _first_cell(env.action_masks())
    monkeypatch.setattr(pe.vio, "Judge", _BadJudge)
    with pytest.raises(pe.EnvContractError, match="不一致"):
        env.step((k, i, j))

    env2 = pe.PcbEnv(board=b, use_lookahead=False, verify=False)
    env2.reset(seed=0)
    env2.step(_first_cell(env2.action_masks()))                 # verify=False ⇒ 照常走
    assert env2.state.index == 1


# ===================== G. 确定性 + 动作编码 =====================

def test_determinism_and_action_encoding():
    """同板同动作 ⇒ 逐位相同；`(k,i,j)` 与一维 `k·m²+i·m+j` 等价。"""
    b = _cheap()
    e1 = pe.PcbEnv(board=b, use_lookahead=False)
    e2 = pe.PcbEnv(board=b, use_lookahead=False)
    o1, _ = e1.reset(seed=0)
    o2, _ = e2.reset(seed=0)
    assert np.array_equal(o1[pe.OBS_SPATIAL], o2[pe.OBS_SPATIAL])

    for _ in range(3):
        k, i, j = _first_cell(e1.action_masks())
        s1, r1, t1, _, _ = e1.step((k, i, j))
        s2, r2, t2, _, _ = e2.step(k * gr.GRID_M * gr.GRID_M + i * gr.GRID_M + j)
        assert r1 == r2 and t1 == t2
        assert np.array_equal(s1[pe.OBS_SPATIAL], s2[pe.OBS_SPATIAL])
        assert np.array_equal(s1[pe.OBS_PIECE], s2[pe.OBS_PIECE])


# ===================== H. 结构性死板 =====================

def _no_legal(model, placed, comp, spec=None, **kw):
    """替身 `legal_masks`：永远"没有任何合法格"（制造开局即死，确定性）。"""
    m = gr.GRID_M
    return np.zeros((osp.N_ROT, m, m), dtype=bool)


def test_structural_dead_start_is_reported(monkeypatch):
    """开局 `L` 就空：`resample_dead_start=False` ⇒ 如实报 `structural`（不静默、不空转）。"""
    b = _cheap()
    monkeypatch.setattr(pe.gr, "legal_masks", _no_legal)

    env = pe.PcbEnv(board=b, resample_dead_start=False)
    _obs, info = env.reset(seed=0)
    assert info["deadlock_kind"] == "structural" and info["at"] == "L"
    assert info["terminated"] is True and info["n_placed"] == 0
    assert not env.action_masks().any()
    with pytest.raises(pe.EnvDoneError):
        env.step((0, 0, 0))

    only = pe.PcbEnv(board=b, boards=[b], resample_dead_start=True)   # 无处可重采 ⇒ 仍要报出来
    _o2, info2 = only.reset(seed=0)
    assert info2["deadlock_kind"] == "structural"

    res = pe.PcbEnv(board=b, resample_dead_start=True)                # 默认：换一块能跑的
    _o3, info3 = res.reset(seed=0)
    assert info3["skipped_dead_start"] and b in info3["skipped_dead_start"]
    assert res.board != b and info3["n_placed"] == 0

    # 真实数据里有没有这种板？只报事实（不 skip、不造数据）
    print(f"[env 实测] 缓存 16 板里「开局 L 就空」的板：{_structural() or '无'}")


def test_deadlock_branches_after_a_step(monkeypatch):
    """终止两条支路的**接线**（替身掩膜确定性触发）：`at`/`deadlock_kind` 必须如实记录。"""
    b = _cheap()

    # (a) L 非空、被单步预测掩膜清空 ⇒ at="mask"、kind="order_locked"
    env = pe.PcbEnv(board=b, use_lookahead=True)
    env.reset(seed=0)
    real = pe.la.lookahead_mask

    def _kill(model, placed, comp, nxt, spec=None, legal=None, **kw):
        L = legal if legal is not None else gr.legal_masks(model, placed, comp, spec)
        return np.zeros_like(L), {"dead_next": False, "kept": 0, "cut": int(L.sum()),
                                  "n_cells_next": 0, "resolved_at": [0, 0, 0, 0]}

    monkeypatch.setattr(pe.la, "lookahead_mask", _kill)
    _obs, _r, terminated, truncated, info = env.step(_first_cell(env.action_masks()))
    assert terminated is True and truncated is False
    assert info["at"] == "mask" and info["deadlock_kind"] == "order_locked"
    assert info["n_placed"] == 1 and info["completion"] == pytest.approx(1.0 / len(env.order))
    monkeypatch.setattr(pe.la, "lookahead_mask", real)

    # (b) L 全空（物理死）⇒ at="L"、kind="policy"（index > 0 ⇒ 不是开局无解）
    env2 = pe.PcbEnv(board=b, use_lookahead=True)
    env2.reset(seed=0)
    monkeypatch.setattr(pe.gr, "legal_masks", _no_legal)
    _o2, _r2, terminated2, _t2, info2 = env2.step(_first_cell(env2.action_masks()))
    assert terminated2 is True
    assert info2["at"] == "L" and info2["deadlock_kind"] == "policy"
    assert info2["n_placed"] == 1
