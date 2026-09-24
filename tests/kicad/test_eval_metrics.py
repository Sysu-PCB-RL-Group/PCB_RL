# -*- coding: utf-8 -*-
"""`utils/eval_metrics.py`（项目评判标准的单一实现，§10.2）的守门测试。

**不需要 KiCad、也不需要板**（纯函数，手算小例对拍）。
覆盖：
  A. 4 个指标的**符号约定**：HPWL 降低为正｜σ 提升为正｜reward 提升为正｜完成度只看件数
  B. reward 与**训练同一式子**：终局公式 == 用 `env.reward.step_reward` 逐步累加
  C. **专家侧恒等式**：模型 == 专家 ⇒ 三项百分比全 0、ΔR == 0
  D. `ΔR` 与 `β` 无关（β 只进绝对值）；而 `reward_gain_pct` 的分母含 β ⇒ 会随 β 变（如实钉住）
  E. 分母为 0 ⇒ 该指标记 `None`（**不猜 0**）；`aggregate` 跳过 `None`
"""
import pytest

from env import reward as rw                # noqa: E402
from utils import eval_metrics as em        # noqa: E402

_BASE = dict(h_ref=100.0, d_board=50.0, beta=0.01, n=5, alpha=rw.ALPHA)


def test_reward_total_matches_stepwise_training_reward():
    """**终局公式 == 逐步累加**：用 `env.reward.step_reward` 逐步算的和，必须等于 `episode_reward_total`。"""
    traj = [(0.0, 0.0), (30.0, 2.0), (60.0, 7.0), (80.0, 12.0)]        # 每步之后的 (HPWL, σ)
    stepwise = 0.0
    for (h0, s0), (h1, s1) in zip(traj, traj[1:]):
        stepwise += rw.step_reward(d_hpwl=h1 - h0, d_sigma=s1 - s0,
                                   h_ref=100.0, d_board=50.0, beta=0.01)
    total = em.episode_reward_total(hpwl=80.0, sigma=12.0, h_ref=100.0, d_board=50.0,
                                    beta=0.01, n=3, alpha=rw.ALPHA)
    assert total == pytest.approx(stepwise), "终局公式与逐步累加必须逐位一致（telescoping）"
    assert total == pytest.approx(-80.0 / 100.0 + rw.ALPHA * 12.0 / 50.0 + 0.01 * 3)


def test_four_metrics_sign_convention():
    """手算小例：HPWL 降低 20%｜σ 提升 20%｜reward 提升按 `ΔR/|R_专家|`。"""
    m = em.board_metrics(hpwl_model=80.0, hpwl_expert=100.0, sigma_model=12.0, sigma_expert=10.0,
                         n_total=5, **_BASE)
    assert m["completion"] == pytest.approx(1.0)
    assert m["hpwl_drop_pct"] == pytest.approx(20.0)          # 越小越好 ⇒ 降低为正
    assert m["sigma_gain_pct"] == pytest.approx(20.0)         # 越大越好 ⇒ 提升为正
    r_model = -80.0 / 100.0 + rw.ALPHA * 12.0 / 50.0 + 0.01 * 5
    r_expert = -100.0 / 100.0 + rw.ALPHA * 10.0 / 50.0 + 0.01 * 5
    assert m["reward_model"] == pytest.approx(r_model)
    assert m["reward_expert"] == pytest.approx(r_expert)
    assert m["reward_delta"] == pytest.approx(r_model - r_expert)
    assert m["reward_gain_pct"] == pytest.approx((r_model - r_expert) / abs(r_expert) * 100.0)
    assert m["reward_gain_pct"] > 0.0, "reward 更高 ⇒ 提升为正"

    # 反向：HPWL 更大、σ 更小 ⇒ 两个百分比都为负（可负数）
    worse = em.board_metrics(hpwl_model=120.0, hpwl_expert=100.0, sigma_model=8.0,
                             sigma_expert=10.0, n_total=5, **_BASE)
    assert worse["hpwl_drop_pct"] == pytest.approx(-20.0)
    assert worse["sigma_gain_pct"] == pytest.approx(-20.0)


def test_expert_equals_model_gives_all_zeros():
    """模型 == 专家 ⇒ 三项百分比 0、ΔR = 0（判据自洽）。"""
    z = em.board_metrics(hpwl_model=100.0, hpwl_expert=100.0, sigma_model=10.0, sigma_expert=10.0,
                         n_total=5, **_BASE)
    assert z["hpwl_drop_pct"] == pytest.approx(0.0)
    assert z["sigma_gain_pct"] == pytest.approx(0.0)
    assert z["reward_delta"] == pytest.approx(0.0)
    assert z["reward_gain_pct"] == pytest.approx(0.0)


def test_reward_delta_ignores_beta_but_percentage_does_not():
    """`ΔR` 与 β 无关（两边 β·n 抵消）；但 `reward_gain_pct` 的分母含 β·n ⇒ 会随 β 变（如实钉住）。"""
    kw = dict(hpwl_model=80.0, hpwl_expert=100.0, sigma_model=12.0, sigma_expert=10.0,
              h_ref=100.0, d_board=50.0, n=5, alpha=rw.ALPHA, n_total=5)
    a = em.board_metrics(beta=0.01, **kw)
    b = em.board_metrics(beta=10.0, **kw)
    assert a["reward_delta"] == pytest.approx(b["reward_delta"]), "ΔR 必须与 β 无关"
    assert abs(a["reward_gain_pct"] - b["reward_gain_pct"]) > 1.0, "分母含 β·n ⇒ 百分比会变"


def test_zero_denominator_records_none():
    """分母为 0 ⇒ 该指标记 `None`（不猜 0）。

    构造：`hpwl_expert = σ_expert = 0` 且 `β = 0` ⇒ `R_专家 = 0` ⇒ 三个百分比分母全为 0。
    """
    kw = {**_BASE, "beta": 0.0}
    m = em.board_metrics(hpwl_model=80.0, hpwl_expert=0.0, sigma_model=12.0, sigma_expert=0.0,
                         n_total=5, **kw)
    assert m["hpwl_drop_pct"] is None
    assert m["sigma_gain_pct"] is None
    assert m["reward_expert"] == pytest.approx(0.0)
    assert m["reward_gain_pct"] is None
    assert m["reward_delta"] is not None, "绝对差仍应给出"


def test_completion_is_count_only_and_handles_edges():
    """完成度**只看件数**（`n/n_total`）；`n_total` 缺省 ⇒ `None`；`n_total=0` ⇒ 记 1.0（没有可放的）。"""
    base = {k: v for k, v in _BASE.items() if k != "n"}         # `n` 由各断言自己给
    m = em.board_metrics(hpwl_model=1.0, hpwl_expert=2.0, sigma_model=1.0, sigma_expert=1.0,
                         n=3, n_total=6, **base)
    assert m["completion"] == pytest.approx(0.5)
    assert m["n_placed"] == 3 and m["n_total"] == 6

    assert em.board_metrics(hpwl_model=1.0, hpwl_expert=2.0, sigma_model=1.0, sigma_expert=1.0,
                            n=3, **base)["completion"] is None
    assert em.board_metrics(hpwl_model=1.0, hpwl_expert=2.0, sigma_model=1.0, sigma_expert=1.0,
                            n=0, n_total=0, **base)["completion"] == pytest.approx(1.0)


def test_aggregate_skips_none():
    rows = [{"completion": 1.0, "hpwl_drop_pct": 10.0, "sigma_gain_pct": None, "reward_gain_pct": 5.0},
            {"completion": 1.0, "hpwl_drop_pct": 20.0, "sigma_gain_pct": None, "reward_gain_pct": 15.0}]
    ag = em.aggregate(rows)
    assert ag["hpwl_drop_pct"]["mean"] == pytest.approx(15.0)
    assert ag["hpwl_drop_pct"]["median"] == pytest.approx(15.0)
    assert ag["hpwl_drop_pct"]["min"] == pytest.approx(10.0)
    assert ag["hpwl_drop_pct"]["max"] == pytest.approx(20.0)
    assert ag["reward_gain_pct"]["mean"] == pytest.approx(10.0)
    assert ag["sigma_gain_pct"]["mean"] is None, "全是 None ⇒ 记 None（不当 0）"
    assert set(em.METRIC_KEYS) == {"completion", "hpwl_drop_pct", "sigma_gain_pct", "reward_gain_pct"}


def test_bad_scales_raise():
    """`H_ref` / `D` 非正 ⇒ 直接抛（不让 NaN 悄悄流下去）。"""
    with pytest.raises(ValueError):
        em.episode_reward_total(hpwl=1.0, sigma=1.0, h_ref=0.0, d_board=10.0,
                                beta=0.0, n=1, alpha=rw.ALPHA)
    with pytest.raises(ValueError):
        em.episode_reward_total(hpwl=1.0, sigma=1.0, h_ref=10.0, d_board=0.0,
                                beta=0.0, n=1, alpha=rw.ALPHA)
