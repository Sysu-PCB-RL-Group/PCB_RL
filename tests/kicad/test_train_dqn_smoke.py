# -*- coding: utf-8 -*-
"""`scripts/train_dqn.py`（Double DQN 训练入口，台账 §9.2）的冒烟测试。

**不需要 KiCad**（板走 P1 缓存），但需要 `data/model_cache/`（没有就 skip）。
小配置直接调 `train()`（不经 CLI），跑 1~2 局：几秒钟，足以覆盖"接线是否通"。

覆盖：
  A. **默认值是 Owner 拍定的那组数**（1000 局 / 每 4 步更新 / batch 32 / 每 50 局评估）—— 防漂守门
  B. 主循环能跑通：落 `log.jsonl`（config / episode / eval 三类记录）+ `ckpt_ep*.pt`
  C. `--update-every` 真的按"每 K 个 env 步"控制梯度步数（`updates == steps // K`）
  D. 评估走**贪心**（`explore=False`）+ 报告"完成度 / HPWL÷H_ref / 早终止"
"""
import importlib.util
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")

from env import order_plan as op                    # noqa: E402
from env.dqn_agent import DoubleDQNAgent            # noqa: E402
from env.pcb_env import PcbEnv                      # noqa: E402
from utils.kicad_utils import model_cache as mc     # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("train_dqn", _ROOT / "scripts" / "train_dqn.py")
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)


# ---------------- 选一块最短的板（= 步数最少 ⇒ 测试最快） ----------------

_BOARD = None


def _cheapest_board() -> str:
    global _BOARD
    if _BOARD is None:
        try:
            names = [e["board"] for e in mc.read_index()["boards"]]
        except mc.CacheError:
            pytest.skip("没有 data/model_cache/index.json（先跑 scripts/dump_models.py）")
        best = None
        for b in names:
            n = len(op.plan_order(mc.load_model(b, allow_build=False)))
            if best is None or n < best[1]:
                best = (b, n)
        _BOARD = best[0]
    return _BOARD


def _tiny_agent(capacity=64, batch=4, **kw):
    return DoubleDQNAgent(capacity=capacity, batch_size=batch, eps_top=0.5, eps_bottom=0.5, **kw)


# ---------------- A. 默认值守门 ----------------

def test_defaults_match_the_agreed_config():
    """默认值 = Owner 2026-09-23 拍定；被改动 ⇒ 这条红（连同 CLI 默认）。"""
    assert td.DEFAULTS["episodes"] == 1000
    assert td.DEFAULTS["update_every"] == 4
    assert td.DEFAULTS["batch"] == 32
    assert td.DEFAULTS["eval_every_episodes"] == 50
    assert td.DEFAULTS["capacity"] == 10_000
    assert td.DEFAULTS["out"] == "output", "产物目录 = output/（Owner 2026-09-23）"
    assert td.DEFAULTS["seed"] == 114514, "固定种子 114514（Owner 2026-09-23）"
    assert td.DEFAULTS["target_every_episodes"] == 20, "目标网同步 100 → 20（Owner 2026-09-23）"

    ns = td.build_parser().parse_args([])
    assert (ns.episodes, ns.update_every, ns.batch, ns.eval_every_episodes) == (1000, 4, 32, 50)
    assert ns.out == "output" and ns.target_every_episodes == 20 and ns.seed == 114514


# ---------------- A2. 清零（写入前）与安全闸 ----------------

def test_output_is_cleared_before_writing(tmp_path):
    """Owner 2026-09-23：产物目录写入前清零（旧结果可被覆盖，但不允许"残留旧文件混在里面"）。"""
    out = tmp_path / "output"
    (out / "sub").mkdir(parents=True)
    (out / "stale.txt").write_text("old", encoding="utf-8")
    (out / "ckpt_ep999.pt").write_text("old", encoding="utf-8")

    b = _cheapest_board()
    td.train(episodes=1, update_every=0, eval_every_episodes=0, log_every=1, ckpt_every=0,
             batch=4, capacity=64, boards=[b], out=str(out),
             agent=_tiny_agent(), env=PcbEnv(boards=[b], use_lookahead=True), verbose=False)

    assert not (out / "stale.txt").exists() and not (out / "ckpt_ep999.pt").exists()
    assert not (out / "sub").exists()
    assert (out / "log.jsonl").is_file()


def test_clear_output_refuses_repo_like_dir(tmp_path):
    """安全闸：目标目录看起来像仓库（含 `data/` 等）⇒ 拒绝清零（防手滑删掉仓库/数据）。"""
    fake = tmp_path / "not_output"
    (fake / "data").mkdir(parents=True)
    (fake / "data" / "x.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝清零"):
        td.clear_output(str(fake))
    assert (fake / "data" / "x.txt").is_file(), "被拒绝时必须原封不动"
    assert td.clear_output(str(tmp_path / "fresh")) is None      # 不存在 ⇒ 新建，不报错


# ---------------- A3. best.pt 判据 ----------------

def test_better_rule_prefers_completion_then_hpwl():
    """`best.pt` 判据（§0.6 m 主指标 = 完成度）：完成度优先，打平看 `HPWL÷H_ref` 更小。"""
    lo = {"completion": 0.80, "h_ratio": 1.20}
    hi = {"completion": 0.90, "h_ratio": 1.90}
    assert td.better(hi, None) is True
    assert td.better(hi, lo) is True, "完成度更高者胜（哪怕 HPWL 更差）"
    assert td.better(lo, hi) is False
    assert td.better({"completion": 0.90, "h_ratio": 1.50}, {"completion": 0.90, "h_ratio": 1.60}) is True
    assert td.better({"completion": 0.90, "h_ratio": 1.70}, {"completion": 0.90, "h_ratio": 1.60}) is False


# ---------------- B. 主循环 + 落盘 ----------------

def test_train_smoke_writes_log_and_checkpoints(tmp_path):
    b = _cheapest_board()
    agent = _tiny_agent()
    env = PcbEnv(boards=[b], use_lookahead=True)
    out = str(tmp_path / "run")

    summary = td.train(episodes=2, update_every=0, eval_every_episodes=1, log_every=1,
                       ckpt_every=1, batch=4, capacity=64, boards=[b], out=out,
                       agent=agent, env=env, verbose=False)

    lines = [json.loads(x) for x in open(os.path.join(out, "log.jsonl"), encoding="utf-8")]
    kinds = [x["kind"] for x in lines]
    assert kinds[0] == "config" and kinds.count("episode") == 2 and kinds.count("eval") == 2
    assert summary["episodes"] == 2 and summary["updates"] == 0        # update_every=0 ⇒ 只收集不学
    assert summary["env_steps"] == sum(x["steps"] for x in lines if x["kind"] == "episode")
    assert summary["steps_per_s"] > 0

    ep0 = next(x for x in lines if x["kind"] == "episode")
    for key in ("steps", "ret", "eps", "completion", "deadlock_kind", "at", "wall_s"):
        assert key in ep0
    assert 0.0 <= ep0["completion"] <= 1.0

    ev = next(x for x in lines if x["kind"] == "eval")
    assert ev["n_boards"] == 1 and len(ev["board_rows"]) == 1
    row = ev["board_rows"][0]
    assert 0.0 <= row["completion"] <= 1.0 and row["h_ratio"] > 0.0
    assert row["n_placed"] <= row["n_total"]

    assert os.path.isfile(os.path.join(out, "ckpt_ep1.pt")) and os.path.isfile(os.path.join(out, "ckpt_ep2.pt"))

    # train.log：人类可读的完整日志（头信息 + 进度 + 评估 + 收尾汇总）
    tlog = open(os.path.join(out, "train.log"), encoding="utf-8").read()
    assert "[out]" in tlog and "（已清零）" in tlog
    assert "[eval]" in tlog and "HPWL÷H_ref" in tlog
    assert "[summary]" in tlog and "[done]" in tlog
    assert f"目标网每 {td.DEFAULTS['target_every_episodes']} 局同步" in tlog


# ---------------- C. 更新频率 ----------------

def test_update_every_controls_gradient_steps(tmp_path):
    """`--update-every K` ⇒ 梯度步数 == ⌊env 步 / K⌋（本例 K=4；不评估以省时）。"""
    b = _cheapest_board()
    agent = _tiny_agent()
    env = PcbEnv(boards=[b], use_lookahead=True)

    summary = td.train(episodes=1, update_every=4, eval_every_episodes=0, log_every=0,
                       ckpt_every=0, batch=4, capacity=64, boards=[b], out=str(tmp_path),
                       agent=agent, env=env, verbose=False)

    assert summary["env_steps"] == agent.steps > 0
    assert agent.updates == agent.steps // 4
    assert agent.updates > 0, "这局至少有 4 步（板最少 10 件）⇒ 应当更新过"


# ---------------- D. 评估是贪心 + 指标齐 ----------------

def test_state_record_and_best_pt(tmp_path):
    """`kind="state"` 记录训练状态（Owner 要求）；评估更优时覆盖保存 `best.pt`。"""
    b = _cheapest_board()
    agent = _tiny_agent()
    env = PcbEnv(boards=[b], use_lookahead=True)
    out = str(tmp_path / "output")

    summary = td.train(episodes=2, update_every=0, eval_every_episodes=1, log_every=1,
                       ckpt_every=0, batch=4, capacity=64, boards=[b], out=out,
                       agent=agent, env=env, verbose=False)

    lines = [json.loads(x) for x in open(os.path.join(out, "log.jsonl"), encoding="utf-8")]
    states = [x for x in lines if x["kind"] == "state"]
    assert len(states) == 2, "每 log_every 局应写一条训练状态"
    st = states[-1]
    for key in ("steps", "episodes_agent", "updates", "eps", "lr", "gamma", "target_syncs",
                "buffer_len", "buffer_capacity", "buffer_gb", "per_transition_bytes",
                "recent_ret", "recent_loss", "steps_per_s", "wall_s"):
        assert key in st, f"训练状态缺字段：{key}"
    assert st["lr"] == pytest.approx(1e-4)                      # Adam，Owner 拍
    assert st["gamma"] == pytest.approx(1.0)
    assert 0.1 <= st["eps"] <= 0.9                              # ε 曲线上下界
    assert st["buffer_len"] == min(agent.steps, agent.replay.capacity)
    assert st["buffer_gb"] > 0.0
    assert st["steps"] == agent.steps

    evals = [x for x in lines if x["kind"] == "eval"]
    assert evals and evals[0]["is_best"] is True, "第一次评估必然优于 None ⇒ 应存 best.pt"
    assert os.path.isfile(os.path.join(out, "best.pt"))
    assert summary["best"] is not None and summary["best"]["completion"] == evals[0]["completion"]
    assert summary["best_pt"].endswith("best.pt")


def test_fixed_seed_makes_runs_reproducible(tmp_path):
    """固定种子（114514）⇒ **连网络初始化也复现**：同一命令两次跑的日志逐条相同（去掉耗时字段）。

    这里**不注入 agent**（让 `train()` 自己建网）⇒ 才能验证"播种发生在建网之前"。
    """
    assert td.DEFAULTS["seed"] == 114514
    b = _cheapest_board()
    recs = []
    for k in range(2):
        out = str(tmp_path / f"run{k}")
        td.train(episodes=1, update_every=4, eval_every_episodes=0, log_every=1, ckpt_every=0,
                 batch=4, capacity=64, seed=114514, boards=[b], out=out, verbose=False)
        got = [json.loads(x) for x in open(os.path.join(out, "log.jsonl"), encoding="utf-8")]
        recs.append([{kk: vv for kk, vv in r.items() if kk not in ("wall_s", "steps_per_s")}
                     for r in got if r["kind"] != "config"])

    ep0 = next(r for r in recs[0] if r["kind"] == "episode")
    ep1 = next(r for r in recs[1] if r["kind"] == "episode")
    assert ep0 == ep1, f"同种子两跑的逐局记录应相同：\n{ep0}\n{ep1}"
    st0 = next(r for r in recs[0] if r["kind"] == "state")
    st1 = next(r for r in recs[1] if r["kind"] == "state")
    assert st0 == st1, "训练状态（含 loss ⇒ 依赖网络初始化）也应相同"
    assert st0["recent_loss"] is not None, "这局应当更新过（update_every=4，板最少 10 件）"


def test_evaluate_is_greedy_and_reports_metrics():
    b = _cheapest_board()
    agent = _tiny_agent()
    seen = []
    real_act = agent.act

    def spy(obs, mask, *, explore=True):
        seen.append(explore)
        return real_act(obs, mask, explore=explore)

    agent.act = spy                                              # type: ignore[assignment]
    ev = td.evaluate(agent, [b])

    assert seen and all(x is False for x in seen), "评估必须是贪心（explore=False）"
    assert ev["n_boards"] == 1 and ev["deadlocks"] in (0, 1)
    assert 0.0 <= ev["completion"] <= 1.0
    assert ev["h_ratio"] > 0.0
    row = ev["board_rows"][0]
    assert row["board"] == b and row["at"] in (None, "L", "mask")
