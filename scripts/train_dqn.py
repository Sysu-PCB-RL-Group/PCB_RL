# -*- coding: utf-8 -*-
"""Double DQN 训练入口 —— 台账 §9.2（算法 Owner 2026-09-23 拍：Double DQN）。

默认配置（**Owner 2026-09-23 拍定**，改动即需同步台账）
```text
--episodes 1000        训练局数（本地 CPU 口径；≈ 41,750 env 步）
--update-every 4       每 4 个 env 步做一次梯度更新（≈ 10,438 次更新）
--batch 32             采样批量
--eval-every-episodes 50   每 50 局跑一次评估（16 板；贪心、推理档 mask = L）
--log-every 10         每 10 局打一行进度
--ckpt-every 100       每 100 局落一次断点
--capacity 10000       打包回放容量（≈ 3.80 GB @200×200；Owner 暂定）
--target-every-episodes 20   目标网硬同步周期（Owner 2026-09-23：先 100，后因"只跑 100 局"改为 **20**）
--seed 114514          **固定种子**（Owner 2026-09-23）：连网络初始化的随机权重也由它决定 ⇒ 同一命令可复现
其余（`env/dqn_agent.py` 内）：目标网每 100 局硬同步｜ε = 0.1 + 0.8·exp(−ep/2000)｜Adam lr 1e-4｜γ = 1.0
```
本机 CPU 实测单位成本 ⇒ 上述默认约 **4.0 小时**（见 §9.2 的时间估算表）。

用法
```text
& python scripts/train_dqn.py                                  # 正式跑（1000 局 → output/）
& python scripts/train_dqn.py --episodes 5 --eval-every-episodes 5 --log-every 1   # 冒烟
& python scripts/train_dqn.py --boards <板名>                    # 只在一两块板上跑（调试）
```
产物（`--out`，默认 **`output/`**；**写入前清零** —— Owner 2026-09-23 拍："结果跟之前一样写入 output/ 即可，写入前清零"）
```text
train.log       **人类可读**的完整运行日志（与终端同内容：开跑头信息 + 每 log_every 局的进度 + 每次评估 +
                收尾 `[summary]` 汇总块：配置 / 用时 / best / 每次评估的指标）—— 事后直接打开就能读全貌
log.jsonl       第 1 行 kind=config（配置）｜每局 1 行 kind=episode｜每 log_every 局 1 行 kind=state
                （**训练状态**：ε / lr / steps / updates / episodes / 目标网同步次数 / 回放条数与 GB / 吞吐）
                ｜每 eval_every 局 1 行 kind=eval（16 板逐板 + 均值 + is_best）
ckpt_ep{N}.pt   每 ckpt_every 局一个断点（实测 0.68 MB：在线网 + 目标网 + Adam + 计数器）
best.pt         每次评估若"更好"就（覆盖）保存 —— 判据：**完成度更高**，打平则 `HPWL÷H_ref` 更低（§0.6 m 主指标）
```
⚠️ "清零" = 删掉 `--out` 目录内的全部内容；带一条**安全闸**：目标目录若看起来像仓库目录（含 `.git`/`data`/`env`/
`tests`/`scripts`/`docs`/`utils` 或就是仓库根）⇒ **拒绝**并报错（防手滑清错目录）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.dqn_agent import DoubleDQNAgent                       # noqa: E402
from env.pcb_env import PcbEnv                                 # noqa: E402
from utils.kicad_utils import model_cache as mc                # noqa: E402

# ---- 默认值 = Owner 2026-09-23 拍定（测试里有守门，防漂） ----
DEFAULTS: Dict[str, Any] = {
    "episodes": 1000,
    "update_every": 4,
    "batch": 32,
    "eval_every_episodes": 50,
    "log_every": 10,
    "ckpt_every": 100,
    "capacity": 10_000,
    "target_every_episodes": 20,   # 目标网硬同步：Owner 2026-09-23 由 100 改为 20（因只跑 100 局）
    "seed": 114514,          # 固定种子（Owner 2026-09-23）—— 连网络初始化也由它决定，见 seed_all()
    "out": "output",
    "device": "cpu",
}

# 清零安全闸：目标目录里出现这些名字 ⇒ 判定为"仓库目录"，拒绝清零（防手滑）
_REPO_MARKERS = (".git", "data", "env", "tests", "scripts", "docs", "utils")


def all_boards(cache_root: Optional[str] = None) -> List[str]:
    """缓存索引里的全部板（= `data/model_cache/index.json`，当前 16 块）。"""
    return [e["board"] for e in mc.read_index(root=cache_root)["boards"]]


# ===================== 评估（贪心 · 推理档） =====================

def evaluate(agent: DoubleDQNAgent, boards: Sequence[str], *, cache_root: Optional[str] = None
             ) -> Dict[str, Any]:
    """逐板贪心跑完（**推理档：`mask = L`**，§6.3）→ 完成度 / `HPWL÷H_ref` / 是否提前终止。

    `HPWL÷H_ref` 是"离专家布局多远"的尺子：**专家 = 1.0**（§7.1.1），越小越好。
    """
    env = PcbEnv(boards=list(boards), cache_root=cache_root, use_lookahead=False, verify=True)
    rows: List[Dict[str, Any]] = []
    for b in boards:
        obs, info = env.reset(options={"board": b})
        while True:
            mask = env.action_masks()
            if not mask.any():                       # 物理无位置（推理档终止判据，§6.3/§5.4）
                break
            obs, _r, term, trunc, info = env.step(agent.act(obs, mask, explore=False))
            if term or trunc:
                break
        st = env.state
        hpwl = float(env.model.hpwl(refs=set(st.placed_refs), poses=st.poses))
        rows.append({
            "board": b, "n_total": int(info["n_total"]), "n_placed": int(info["n_placed"]),
            "completion": float(info["completion"]),
            "hpwl": hpwl, "h_ratio": hpwl / max(float(env.h_ref), 1e-9),
            "deadlock_kind": info["deadlock_kind"], "at": info["at"],
        })
    return {
        "board_rows": rows,
        "completion": float(np.mean([r["completion"] for r in rows])),
        "h_ratio": float(np.mean([r["h_ratio"] for r in rows])),
        "deadlocks": int(sum(r["deadlock_kind"] is not None for r in rows)),
        "n_boards": len(rows),
    }


# ===================== 可复现：固定种子 =====================

def seed_all(seed: Optional[int]) -> None:
    """把**全部随机源**钉在同一颗种子上（Owner 2026-09-23：固定种子 **114514**）。

    覆盖四处，缺一处都不可复现：
    ```text
    torch.manual_seed(seed)   网络初始化（`QNet()` 的卷积权重）+ 后续任何 torch 随机
    np.random.seed(seed)      numpy 全局随机（第三方/自写代码里若有 np.random.*）
    agent(seed=seed)          智能体自己的 `np.random.Generator`（ε-贪心采样）
    env.reset(seed=seed + ep) 每局的板抽样（及 env 内部任何随机）
    ```
    """
    if seed is None:
        return
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))


# ===================== 产物处置（清零 / best / 训练状态） =====================

def clear_output(out: str) -> None:
    """**写入前清零**（Owner 2026-09-23："结果跟之前一样写入 `output\\` 即可，写入前清零"）。

    删掉 `out` 目录内的全部内容（目录本身保留；不存在则新建）。**安全闸**：目标目录若"看起来像仓库目录"
    （含 `.git`/`data`/`env`/`tests`/`scripts`/`docs`/`utils` 之一，或就是本仓库根）⇒ 抛 `ValueError` 且**不清**。
    """
    p = os.path.abspath(str(out))
    if os.path.isdir(p):
        if p == os.path.abspath(_ROOT) or any(os.path.exists(os.path.join(p, m)) for m in _REPO_MARKERS):
            raise ValueError(f"拒绝清零 {p}：它看起来是仓库目录（命中 {_REPO_MARKERS} 之一）")
        for name in os.listdir(p):
            q = os.path.join(p, name)
            if os.path.isdir(q) and not os.path.islink(q):
                shutil.rmtree(q)
            else:
                os.remove(q)
    else:
        os.makedirs(p, exist_ok=True)


def better(ev: Mapping[str, Any], best: Optional[Mapping[str, Any]]) -> bool:
    """`ev` 是否优于 `best`（`best=None` ⇒ True）—— `best.pt` 的判据。

    **完成度是主指标**（§0.6 m 明说 `best_model` 选择以它为准）：完成度更高者胜；
    完成度打平（差 ≤ 1e-9）时，`HPWL÷H_ref` 更小者胜。
    """
    if best is None:
        return True
    if float(ev["completion"]) > float(best["completion"]) + 1e-9:
        return True
    if (abs(float(ev["completion"]) - float(best["completion"])) <= 1e-9
            and float(ev["h_ratio"]) < float(best["h_ratio"]) - 1e-9):
        return True
    return False


def state_record(agent: DoubleDQNAgent, *, episode: int, wall_s: float,
                 recent_ret: float, recent_loss: Optional[float]) -> Dict[str, Any]:
    """**训练状态**一条（`log.jsonl` 的 `kind="state"`）：进度 + 超参 + 回放占用 + 吞吐。"""
    rp = agent.replay
    return {
        "kind": "state", "episode": int(episode),
        "steps": agent.steps, "episodes_agent": agent.episodes, "updates": agent.updates,
        "eps": float(agent.epsilon()), "lr": float(agent.opt.param_groups[0]["lr"]),
        "gamma": float(agent.gamma),
        "target_every_episodes": agent.target_every_episodes,
        "target_syncs": agent.episodes // max(1, agent.target_every_episodes),
        "eps_top": agent.eps_top, "eps_bottom": agent.eps_bottom, "eps_tau_episodes": agent.eps_tau,
        "buffer_len": len(rp), "buffer_capacity": rp.capacity,
        "buffer_gb": round(rp.nbytes / 1e9, 3), "per_transition_bytes": rp.per_transition_bytes,
        "recent_ret": float(recent_ret),
        "recent_loss": None if recent_loss is None else float(recent_loss),
        "steps_per_s": round(agent.steps / max(wall_s, 1e-9), 2),
        "wall_s": round(float(wall_s), 1),
    }


# ===================== 训练主循环 =====================

def train(*, episodes: int = DEFAULTS["episodes"], update_every: int = DEFAULTS["update_every"],
          batch: int = DEFAULTS["batch"], eval_every_episodes: int = DEFAULTS["eval_every_episodes"],
          log_every: int = DEFAULTS["log_every"], ckpt_every: int = DEFAULTS["ckpt_every"],
          capacity: int = DEFAULTS["capacity"],
          target_every_episodes: int = DEFAULTS["target_every_episodes"],
          seed: int = DEFAULTS["seed"],
          out: str = DEFAULTS["out"], device: str = DEFAULTS["device"],
          boards: Optional[Sequence[str]] = None, cache_root: Optional[str] = None,
          agent: Optional[DoubleDQNAgent] = None, env: Optional[PcbEnv] = None,
          verbose: bool = True) -> Dict[str, Any]:
    """跑 `episodes` 局（每局从板池里抽一块）并落盘日志/断点；返回汇总字典。

    `agent` / `env` 可注入（测试用小配置直接调本函数，不经 CLI）。
    """
    clear_output(out)                                            # ⚠️ 写入前清零（Owner 拍定）
    seed_all(seed)                                               # ⚠️ 必须在建网络之前（固定种子 114514）
    log_path = os.path.join(out, "log.jsonl")
    board_list = list(boards) if boards else all_boards(cache_root)
    env = env if env is not None else PcbEnv(boards=board_list, cache_root=cache_root,
                                             use_lookahead=True, verify=True)
    agent = agent if agent is not None else DoubleDQNAgent(
        capacity=capacity, batch_size=batch, device=device, seed=seed,
        target_every_episodes=target_every_episodes)

    cfg = dict(episodes=episodes, update_every=update_every, batch=batch,
               eval_every_episodes=eval_every_episodes, capacity=capacity, seed=seed,
               target_every_episodes=target_every_episodes, boards=board_list, device=device)
    t0 = time.perf_counter()
    hist: List[Dict[str, Any]] = []
    evals: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None                        # best.pt 的当前最优评估
    recent_rets: List[float] = []
    last_loss_seen: Optional[float] = None

    tlog = open(os.path.join(out, "train.log"), "w", encoding="utf-8")   # 人类可读版日志（与终端同内容）

    def say(msg: str) -> None:
        """同时写 `train.log` 与终端（**日志永远落盘**，与 `--verbose` 无关 ⇒ 事后可读全貌）。"""
        tlog.write(msg + "\n")
        tlog.flush()
        if verbose:
            print(msg, flush=True)

    say(f"[out] {os.path.abspath(out)}（已清零）｜板 {len(board_list)} 块｜{episodes} 局｜"
        f"每 {update_every} 步更新｜batch {batch}｜每 {eval_every_episodes} 局评估｜"
        f"目标网每 {target_every_episodes} 局同步｜seed {seed}｜device {device}")

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "config", **cfg}, ensure_ascii=False) + "\n")
        for ep in range(1, int(episodes) + 1):
            obs, info = env.reset(seed=(None if seed is None else int(seed) + ep))
            ep_ret, ep_steps, last_loss = 0.0, 0, None
            while True:
                mask = env.action_masks()
                a = agent.act(obs, mask)                                  # ε-贪心（mask 内）
                prev = obs
                obs, r, term, trunc, info = env.step(a)
                # ⚠️ 存进回放的"下一状态 mask"必须是**落子后那一帧**的 action_masks（Double DQN 的候选集）
                agent.observe(prev, a, r, obs, env.action_masks(), bool(term or trunc))
                ep_ret += r
                ep_steps += 1
                if update_every and agent.steps % int(update_every) == 0:
                    last_loss = agent.update()
                    if last_loss is not None:
                        last_loss_seen = last_loss
                if term or trunc:
                    break
            recent_rets.append(ep_ret)
            recent_rets = recent_rets[-50:]
            rec = {"kind": "episode", "episode": ep, "steps": ep_steps, "ret": ep_ret,
                   "eps": agent.epsilon(), "loss": last_loss,
                   "completion": float(info["completion"]),
                   "deadlock_kind": info["deadlock_kind"], "at": info["at"],
                   "episodes_agent": agent.episodes, "updates": agent.updates,
                   "wall_s": round(time.perf_counter() - t0, 1)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hist.append(rec)

            if log_every and ep % int(log_every) == 0:                 # ★ 训练状态一条（Owner 要求）
                fh.write(json.dumps(state_record(agent, episode=ep, wall_s=time.perf_counter() - t0,
                                                 recent_ret=float(np.mean(recent_rets)),
                                                 recent_loss=last_loss_seen),
                                    ensure_ascii=False) + "\n")

            if eval_every_episodes and ep % int(eval_every_episodes) == 0:
                ev = evaluate(agent, board_list, cache_root=cache_root)
                ev.update(kind="eval", episode=ep, wall_s=round(time.perf_counter() - t0, 1))
                ev["is_best"] = better(ev, best)
                if ev["is_best"]:                                      # ★ best.pt（覆盖保存）
                    best = {k: ev[k] for k in ("episode", "completion", "h_ratio", "deadlocks")}
                    torch_save(os.path.join(out, "best.pt"),
                               {"agent": agent.state_dict(), "episode": ep, "eval": best, "config": cfg})
                evals.append(ev)
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                say(f"[eval] ep{ep:>5} 完成度 {ev['completion']:.3f}｜HPWL÷H_ref {ev['h_ratio']:.3f}"
                    f"｜早终止 {ev['deadlocks']}/{ev['n_boards']}"
                    f"{'｜★best.pt' if ev['is_best'] else ''}｜{ev['wall_s']}s")
            elif log_every and ep % int(log_every) == 0:
                say(f"[train] ep{ep:>5} ret {ep_ret:+.4f}｜完成度 {info['completion']:.3f}"
                    f"｜ε {agent.epsilon():.3f}｜loss {'—' if last_loss is None else f'{last_loss:.4f}'}"
                    f"｜steps {agent.steps}｜updates {agent.updates}｜{rec['wall_s']}s")

            if ckpt_every and ep % int(ckpt_every) == 0:
                torch_save(os.path.join(out, f"ckpt_ep{ep}.pt"),
                           {"agent": agent.state_dict(), "episode": ep, "config": cfg})

            fh.flush()          # 每局把日志落盘 ⇒ 可以边跑边 `tail` / 打开 log.jsonl 看

    elapsed = time.perf_counter() - t0
    summary = {"episodes": episodes, "env_steps": agent.steps, "updates": agent.updates,
               "elapsed_s": round(elapsed, 1), "steps_per_s": round(agent.steps / max(elapsed, 1e-9), 2),
               "log": log_path, "out": out, "best_pt": os.path.join(out, "best.pt") if best else None,
               "best": ({k: best[k] for k in ("episode", "completion", "h_ratio", "deadlocks")}
                        if best else None),
               "last_evals": evals[-1] if evals else None}
    say(f"[done] {episodes} 局｜{agent.steps} env 步｜{agent.updates} 次更新｜"
        f"{elapsed / 60:.1f} min（{summary['steps_per_s']} 步/s）"
        + (f"｜best: ep{best['episode']} 完成度 {best['completion']:.3f} "
           f"HPWL÷H_ref {best['h_ratio']:.3f}" if best else ""))

    # ---- 收尾汇总（写进 train.log，事后一眼可读） ----
    say("=" * 78)
    say(f"[summary] 配置：episodes={episodes}｜update_every={update_every}｜batch={batch}｜"
        f"eval_every={eval_every_episodes}｜target_every={target_every_episodes}｜capacity={capacity}｜"
        f"seed={seed}｜device={device}｜boards={len(board_list)}")
    say(f"[summary] 用时 {elapsed / 60:.1f} min｜env 步 {agent.steps}｜更新 {agent.updates} 次｜"
        f"{summary['steps_per_s']} 步/s｜ε {agent.epsilon():.4f}")
    if best:
        say(f"[summary] **best.pt**：ep{best['episode']}｜完成度 {best['completion']:.4f}｜"
            f"HPWL÷H_ref {best['h_ratio']:.4f}｜早终止 {best['deadlocks']}（专家 = 1.0，越小越好）")
    for ev in evals:
        say(f"[summary] eval@{ev['episode']:>4}：完成度 {ev['completion']:.4f}｜"
            f"HPWL÷H_ref {ev['h_ratio']:.4f}｜早终止 {ev['deadlocks']}/{ev['n_boards']}"
            f"{'｜★best' if ev.get('is_best') else ''}")
    say(f"[summary] 产物：{os.path.abspath(out)}（log.jsonl / train.log / best.pt / ckpt_ep*.pt）")
    say("=" * 78)
    tlog.close()
    return summary


def torch_save(path: str, payload: Dict[str, Any]) -> None:
    import torch                                    # 局部导入：测试无 torch 时也不影响 import 本模块
    torch.save(payload, path)


# ===================== CLI =====================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Double DQN 训练入口（台账 §9.2）")
    p.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    p.add_argument("--update-every", type=int, default=DEFAULTS["update_every"],
                   help="每多少个 env 步做一次梯度更新（0 = 不更新，只收集）")
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--eval-every-episodes", type=int, default=DEFAULTS["eval_every_episodes"],
                   help="每多少局跑一次评估（0 = 不评估）")
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--ckpt-every", type=int, default=DEFAULTS["ckpt_every"])
    p.add_argument("--capacity", type=int, default=DEFAULTS["capacity"])
    p.add_argument("--target-every-episodes", type=int, default=DEFAULTS["target_every_episodes"],
                   help="每多少个 episode 硬同步一次目标网（Owner 2026-09-23：由 100 改为 20）")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--out", default=DEFAULTS["out"])
    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--boards", default="", help="逗号分隔的板名（缺省 = 缓存索引里的全部板）")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--threads", type=int, default=0, help="torch CPU 线程数（0 = 用默认）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        import torch
        torch.set_num_threads(int(args.threads))
    boards = [b.strip() for b in args.boards.split(",") if b.strip()] or None
    summary = train(episodes=args.episodes, update_every=args.update_every, batch=args.batch,
                    eval_every_episodes=args.eval_every_episodes, log_every=args.log_every,
                    ckpt_every=args.ckpt_every, capacity=args.capacity,
                    target_every_episodes=args.target_every_episodes, seed=args.seed,
                    out=args.out, device=args.device, boards=boards, cache_root=args.cache_root)
    print(json.dumps({k: v for k, v in summary.items() if k != "last_evals"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
