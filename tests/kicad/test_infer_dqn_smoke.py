# -*- coding: utf-8 -*-
"""`scripts/infer_dqn.py`（推理入口，台账 §9.2 推理块 / §10 可视化）的冒烟测试。

**不需要 KiCad**（板走 P1 缓存），权重在 `tmp_path` 里现造（**不碰正在训练的 `output/`**）。

覆盖：
  A. 默认值 = Owner 拍定（`output/best.pt` ｜ `result/` ｜ 默认出图）+ 清空安全闸（拒绝清仓库根）
  B. **每次推理前清空上次推理产物**：删 `result/infer_*`；`result/` 下其它历史内容**一律不动**
  C. 产物齐：`inference.log`（头 / 每板一行含真实与专家 HPWL / 均值 / 最好最差）+ `summary.json`
     + `<板名>/<板名>.png` + `<板名>/<板名>.json`
  D. 口径：`mask = L only`（`use_lookahead=False`）｜**纯贪心**（= `mask` 内 argmax Q，不消耗 rng）
     ｜**可复现**（同一权重两跑逐位相同）｜CLI 能跑通
"""
import copy
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from env import board_state as bs                       # noqa: E402
from env import grid_render as gr                       # noqa: E402
from env import order_plan as op                        # noqa: E402
from env import policy_net as pn                        # noqa: E402
from env import reward as rw                            # noqa: E402
from env.dqn_agent import DoubleDQNAgent                # noqa: E402
from env.pcb_env import PcbEnv                          # noqa: E402
from utils import viz_layout as vz                      # noqa: E402
from utils.kicad_utils import model_cache as mc          # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("infer_dqn", _ROOT / "scripts" / "infer_dqn.py")
inf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf)


# ---------------- 选一块最短的板（步数最少 ⇒ 测试最快）+ 造一个临时权重 ----------------

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


def _d_board(board: str) -> float:
    """`D` = 板框对角线（mm）—— 与 `env.reward.d_board_of` 同一实现（σ 的归一化分母）。"""
    return float(rw.d_board_of(mc.load_model(board, allow_build=False)))


def _tiny_agent():
    """容量 1 的小 agent（只用来跑推理，不需要回放 ⇒ 不预分配大数组）。"""
    return DoubleDQNAgent(capacity=1, batch_size=1, eps_top=0.5, eps_bottom=0.5, seed=0)


def _ckpt(tmp_path, *, episode: int = 7, name: str = "best.pt") -> str:
    """临时权重（结构同 `train_dqn.py` 的断点：含 `agent`/`episode`/`eval`/`config`）。"""
    agent = _tiny_agent()
    p = str(tmp_path / name)
    torch.save({"agent": agent.state_dict(), "episode": episode,
                "eval": {"completion": 1.0, "h_ratio": 1.5, "deadlocks": 0},
                "config": {"lr": 1e-4}}, p)
    return p


# ---------------- A. 默认值 + 清空安全闸 ----------------

def test_defaults_are_the_agreed_ones():
    """默认值 = Owner 2026-09-23 拍定：`output/best.pt` / `result/` / 默认出可视化。"""
    assert inf.DEFAULTS["ckpt"] == os.path.join("output", "best.pt")
    assert inf.DEFAULTS["out"] == "result"
    assert inf.DEFAULTS["viz"] is True and inf.DEFAULTS["dpi"] == 110
    assert inf.DEFAULTS["net_boxes"] is False, "HPWL 包围框默认不画（Owner 2026-09-23）"

    ns = inf.build_parser().parse_args([])
    assert ns.ckpt.endswith("best.pt") and ns.out == "result" and ns.viz is True
    assert inf.build_parser().parse_args(["--no-viz"]).viz is False
    assert inf.build_parser().parse_args([]).net_boxes is False
    assert inf.build_parser().parse_args(["--net-boxes"]).net_boxes is True
    assert inf.DEFAULTS["charts"] is True, "3 张指标对比图默认出（§10.2）"
    assert inf.build_parser().parse_args([]).charts is True
    assert inf.build_parser().parse_args(["--no-charts"]).charts is False


def test_clear_previous_inference_refuses_repo_root():
    """安全闸：产物根目录 = 仓库根 ⇒ 拒绝（防手滑 `--out .`）。"""
    with pytest.raises(ValueError, match="拒绝清理"):
        inf.clear_previous_inference(str(_ROOT))


# ---------------- B. 每次推理前清空上次产物（只清 infer_*） ----------------

def test_clear_previous_inference_removes_only_infer_entries(tmp_path):
    """删掉**所有** `infer_*`（含上一次别的权重），其它历史内容原封不动。"""
    out = tmp_path / "result"
    (out / "infer_best_ep7" / "boardA").mkdir(parents=True)
    (out / "infer_best_ep7" / "boardA" / "boardA.png").write_text("old", encoding="utf-8")
    (out / "infer_ckpt_ep50").mkdir()
    (out / "infer_stale_file.txt").write_text("old", encoding="utf-8")
    (out / "compare").mkdir()                                   # ← 上一代历史产物：不许动
    (out / "compare" / "keep.png").write_text("keep", encoding="utf-8")
    (out / "kicad_board_scan.json").write_text("keep", encoding="utf-8")

    removed = inf.clear_previous_inference(str(out))

    assert removed == ["infer_best_ep7", "infer_ckpt_ep50", "infer_stale_file.txt"]
    assert not (out / "infer_best_ep7").exists() and not (out / "infer_ckpt_ep50").exists()
    assert not (out / "infer_stale_file.txt").exists()
    assert (out / "compare" / "keep.png").is_file(), "历史产物必须原封不动"
    assert (out / "kicad_board_scan.json").is_file()

    assert inf.clear_previous_inference(str(tmp_path / "fresh")) == []   # 不存在 ⇒ 新建 + 空列表


def test_infer_clears_previous_run_before_writing(tmp_path):
    """端到端：上一次推理的目录在本次推理开始时被清掉，且不影响非 infer 内容。"""
    out = tmp_path / "result"
    (out / "infer_old_ep1").mkdir(parents=True)
    (out / "infer_old_ep1" / "x.png").write_text("old", encoding="utf-8")
    (out / "keepme.txt").write_text("keep", encoding="utf-8")

    s = inf.infer(ckpt=_ckpt(tmp_path), out=str(out), boards=[_cheapest_board()], viz=False)
    assert not (out / "infer_old_ep1").exists()
    assert (out / "keepme.txt").is_file()
    assert s["cleared_previous"] == ["infer_old_ep1"]
    assert os.path.isdir(os.path.join(out, "infer_best_ep7"))


# ---------------- C. 产物齐（日志 / 汇总 / 图 / 单板 json） ----------------

def test_infer_smoke_writes_log_summary_and_figure(tmp_path):
    b = _cheapest_board()
    out = tmp_path / "result"
    s = inf.infer(ckpt=_ckpt(tmp_path), out=str(out), boards=[b], viz=True, dpi=60)

    root = out / "infer_best_ep7"
    log = (root / "inference.log").read_text(encoding="utf-8")
    assert "[ckpt]" in log and "mask = L only" in log and "纯贪心" in log
    assert "[clean]" in log
    # 两张表（评判表 + 原始值表）与 4 指标列（§10.2）
    assert "【评判表】" in log and "【原始值表】" in log
    assert "HPWL降低%" in log and "σ提升%" in log and "reward提升%" in log
    assert "专家HPWL" in log and "专家σ" in log and b[:22] in log
    assert "[mean·评判]" in log and "[mean]" in log and "[最好 3" in log and "[产物]" in log

    # 汇总
    assert (root / "summary.json").is_file()
    sm = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert sm["kind"] == "inference_summary" and sm["n_boards"] == 1
    assert sm["mask_mode"] == "L_only" and sm["lookahead"] is False
    assert sm["policy"] == "pure_greedy(masked_argmax)"
    assert sm["ckpt_episode"] == 7 and sm["ckpt_metrics"]["completion"] == 1.0
    row = sm["boards"][0]
    assert row["board"] == b and 0 <= row["n_placed"] <= row["n_total"]
    assert 0.0 <= row["completion"] <= 1.0
    assert row["hpwl_model"] > 0.0 and row["hpwl_expert"] > 0.0
    assert row["hpwl_ratio"] == pytest.approx(row["hpwl_model"] / row["hpwl_expert"])
    # σ：模型 / 专家（完整布局）/ 同件集专家 + 归一化（σ÷D）
    assert row["sigma_model"] >= 0.0 and row["sigma_expert"] > 0.0
    assert row["sigma_over_d"] == pytest.approx(row["sigma_model"] / _d_board(b))
    if row["n_placed"] == row["n_total"]:
        assert row["sigma_expert_on_placed"] == pytest.approx(row["sigma_expert"])
    assert row["sigma_ratio"] == pytest.approx(row["sigma_model"] / max(row["sigma_expert_on_placed"], 1e-9))
    for k in ("sigma_model", "sigma_expert", "sigma_ratio", "sigma_over_d"):
        assert set(sm[k]) == {"mean", "median", "min", "max"}, f"汇总缺 {k} 的分位数"
    # ★ 评判标准（§10.2）：4 指标进 summary 与逐板行；`metrics/` 里三张对比图
    assert set(sm["metrics"]) == {"completion", "hpwl_drop_pct", "sigma_gain_pct", "reward_gain_pct"}
    for k in ("hpwl_drop_pct", "sigma_gain_pct", "reward_gain_pct"):
        assert row[k] is not None
        assert sm["metrics"][k]["mean"] == pytest.approx(row[k])       # 单板 ⇒ 均值就是它自己
    assert row["reward_expert"] is not None and row["reward_delta"] == pytest.approx(
        row["reward_model"] - row["reward_expert"])
    assert sm["charts_on"] is True and len(sm["charts"]) == 3
    for v in sm["charts"].values():
        assert os.path.isfile(v["path"]) and os.path.getsize(v["path"]) > 5_000
    assert (root / "metrics" / "metric_reward_gain.png").is_file()
    assert row["stopped_reason"] in ("all_placed", "L_empty")
    assert row["at"] in (None, "L", "mask")
    assert set(row["poses"]) == set(row["order"][:len(row["poses"])])
    for v in row["poses"].values():
        assert len(v) == 3 and v[2] in (0, 1, 2, 3), "第三分量是**旋转序号** k（0/1/2/3），不是角度"
    for k in ("completion", "hpwl_model", "hpwl_expert", "hpwl_ratio"):
        assert set(sm[k]) == {"mean", "median", "min", "max"}

    # 图 + 单板 json（按板名分文件夹）
    png = root / b / f"{b}.png"
    assert png.is_file() and png.stat().st_size > 5_000, "2×2 对比图应当有实际内容"
    assert row["png"] == f"{b}/{b}.png"
    one = json.loads((root / b / f"{b}.json").read_text(encoding="utf-8"))
    assert one["board"] == b and one["poses"] == row["poses"]
    # HPWL 包围框**默认不画**（Owner 2026-09-23）；"打开时画出来"的守门见下一条测试
    assert one["net_boxes"] == {"expert": 0, "model": 0}
    assert sm["net_boxes"] is False

    print(f"[infer 实测] {b}: n={row['n_placed']}/{row['n_total']} 完成度={row['completion']:.4f} "
          f"真实HPWL={row['hpwl_model']:.2f} 专家HPWL={row['hpwl_expert']:.2f} "
          f"比值={row['hpwl_ratio']:.4f} 终止={row['stopped_reason']} 用时={row['seconds']}s "
          f"图={png.stat().st_size // 1024}KB")


def test_no_viz_writes_no_figure(tmp_path):
    """`--no-viz` ⇒ 只有日志与汇总，不落图与单板 json。"""
    b = _cheapest_board()
    out = tmp_path / "result"
    inf.infer(ckpt=_ckpt(tmp_path), out=str(out), boards=[b], viz=False)
    root = out / "infer_best_ep7"
    assert not (root / b).exists()
    assert json.loads((root / "summary.json").read_text(encoding="utf-8"))["dpi"] == inf.DEFAULTS["dpi"]


# ---------------- D. 口径：只用 L / 纯贪心 / 可复现 / CLI ----------------

def test_mask_is_L_only_and_policy_is_pure_greedy():
    """`use_lookahead=False` ⇒ 推理期 `mask == L`；动作 = `mask` 内 max Q，且**不消耗 rng**。"""
    b = _cheapest_board()
    env = PcbEnv(boards=[b], use_lookahead=False)
    assert env.use_lookahead is False
    obs, _info = env.reset(options={"board": b})
    mask, legal = env.action_masks(), env.frame.legal
    assert np.array_equal(mask, legal), "推理档 mask 必须就是 L（不做 lookahead 收缩）"

    agent = _tiny_agent()
    before = copy.deepcopy(agent.rng.bit_generator.state)
    k, i, j = inf.greedy_action(agent, obs, mask)

    q = agent.q_values(obs)[0].detach().cpu().numpy()
    assert bool(mask[k, i, j]), "动作必须落在 mask 内"
    assert q[k, i, j] == pytest.approx(float(np.where(mask, q, -np.inf).max())), "必须是 mask 内 max Q"
    assert agent.rng.bit_generator.state == before, "纯贪心不该消耗 rng（不经 ε 分支）"


def test_same_ckpt_two_runs_are_bit_identical(tmp_path):
    """同一权重两跑 ⇒ 逐位相同（纯贪心、无随机性）。"""
    b = _cheapest_board()
    ck = _ckpt(tmp_path)
    r1 = inf.infer(ckpt=ck, out=str(tmp_path / "r1"), boards=[b], viz=False)
    r2 = inf.infer(ckpt=ck, out=str(tmp_path / "r2"), boards=[b], viz=False)
    for key in ("n_placed", "completion", "hpwl_model", "hpwl_ratio", "stopped_reason", "order", "poses"):
        assert r1["boards"][0][key] == r2["boards"][0][key], f"两跑不一致：{key}"


def test_cli_main_runs_end_to_end(tmp_path, capsys):
    b = _cheapest_board()
    out = str(tmp_path / "result")
    code = inf.main(["--ckpt", _ckpt(tmp_path), "--out", out, "--boards", b, "--no-viz"])
    assert code == 0
    got = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert got["n_boards"] == 1 and 0.0 <= got["completion_mean"] <= 1.0
    assert got["hpwl_ratio_mean"] > 0.0 and got["deadlocks"] in (0, 1)
    assert os.path.isfile(os.path.join(out, "infer_best_ep7", "inference.log"))


# ---------------- E. HPWL 包围框（Owner 2026-09-23：可视化里画出来）----------------

def test_net_boxes_are_exactly_the_hpwl_boxes():
    """框必须**就是** HPWL 的框：`Σ[(Δx+Δy)] == model.hpwl()`；排除电源网；只取给定位姿的件。"""
    b = _cheapest_board()
    m = mc.load_model(b, allow_build=False)
    poses = vz.expert_poses_of(m)
    boxes = vz.net_boxes_of(m, poses)
    total = sum((x1 - x0) + (y1 - y0) for x0, y0, x1, y1 in boxes.values())
    assert total == pytest.approx(m.hpwl(poses=poses)), "框的半周长之和必须等于 HPWL"

    n_expect = sum(1 for net in m.nets.values()
                   if not net.is_power and len(net.connection_points(None, poses)) >= 2)
    assert len(boxes) == n_expect > 0
    assert not any(n.net_id in boxes for n in m.nets.values() if n.is_power), "电源网不该有框"

    half = dict(list(poses.items())[: max(1, len(poses) // 2)])          # 只给一半件
    hb = vz.net_boxes_of(m, half)
    assert set(hb) <= set(boxes), "未给位姿的件不得混进框里"
    assert sum((x1 - x0) + (y1 - y0) for x0, y0, x1, y1 in hb.values()) \
        == pytest.approx(m.hpwl(refs=set(half), poses=half))

    print(f"[net 框实测] {b[:34]}: 专家位姿下 {len(boxes)} 个框｜Σ 半周长 {total:.2f} mm "
          f"== HPWL {m.hpwl(poses=poses):.2f} mm（给一半件 ⇒ {len(hb)} 个框）")


def test_net_boxes_are_optional_and_default_off(tmp_path):
    """HPWL 包围框是**可选项、默认不画**；显式 `show_net_boxes=True` 才画（4 格都画、每格 = 该排 net 数）。"""
    b = _cheapest_board()
    m = mc.load_model(b, allow_build=False)
    poses = vz.expert_poses_of(m)
    n = len(vz.net_boxes_of(m, poses))
    assert n > 0

    off = str(tmp_path / "off.png")
    st0 = vz.draw_compare(m, model_poses=poses, out_path=off, dpi=60, board=b, label_refs=False)
    assert st0["show_net_boxes"] is False and st0["n_net_boxes"] == {"expert": 0, "model": 0}
    assert all(v["net_box"] == 0 for v in st0["panels"].values()), "默认不该画任何框"
    assert os.path.getsize(off) > 5_000

    on = str(tmp_path / "on.png")
    st1 = vz.draw_compare(m, model_poses=poses, out_path=on, dpi=60, board=b, label_refs=False,
                          show_net_boxes=True)
    assert st1["show_net_boxes"] is True and st1["n_net_boxes"] == {"expert": n, "model": n}
    assert {k: v["net_box"] for k, v in st1["panels"].items()} == {k: n for k in st1["panels"]}


# ---------------- F. σ（Owner 2026-09-23：日志里也要与专家组比较）----------------

def test_sigma_recorded_and_matches_independent_recompute(tmp_path):
    """σ 必须落进 log 与 summary，且能从**落盘的位姿**独立重算出来（另一条路径）。"""
    b = _cheapest_board()
    out = tmp_path / "result"
    s = inf.infer(ckpt=_ckpt(tmp_path), out=str(out), boards=[b], viz=False)
    row = s["boards"][0]
    m = mc.load_model(b, allow_build=False)

    poses = {r: (float(v[0]), float(v[1]), int(v[2])) for r, v in row["poses"].items()}
    assert row["sigma_model"] == pytest.approx(
        float(gr.dispersion_of(m, bs.derive_placed(m, poses))), abs=1e-9)
    expert_placed = bs.derive_placed(m, vz.expert_poses_of(m))
    assert row["sigma_expert"] == pytest.approx(float(gr.dispersion_of(m, expert_placed)), abs=1e-9)
    if row["n_placed"] == row["n_total"]:
        assert row["sigma_expert_on_placed"] == pytest.approx(row["sigma_expert"], abs=1e-9)
    assert row["sigma_ratio"] == pytest.approx(row["sigma_model"] / max(row["sigma_expert_on_placed"], 1e-9))

    log = (out / "infer_best_ep7" / "inference.log").read_text(encoding="utf-8")
    assert "【原始值表】" in log and "模型σ" in log and "专家σ" in log
    assert "σ提升%" in log, "评判表里必须出现 σ 提升% 这一列"

    print(f"[σ 实测] {b[:34]}: 模型 σ {row['sigma_model']:.3f} mm｜专家 σ {row['sigma_expert']:.3f} mm"
          f"（同件集 {row['sigma_expert_on_placed']:.3f}）｜σ÷专家σ {row['sigma_ratio']:.4f}"
          f"｜σ/D {row['sigma_over_d']:.4f}｜完成度 {row['completion']:.4f}")
