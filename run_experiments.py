# -*- coding: utf-8 -*-
"""跨板优化对照实验调度器。

复用 Runner / run_compare.py 的既有能力，通过 CLI 参数选择要跑的实验组。
每次运行前动态生成一个临时 argument.txt 副本（改 EXPERT_APPLY_PENALTIES /
EXPERT_TRANSITIONS_DIR 等），保证「唯一变量原则」，不修改原 config/argument.txt。

支持三种实验（可组合）：
  A. expert 有无      : with_expert vs baseline（use_expert 开关）
  D. reward 尺度错配  : EXPERT_APPLY_PENALTIES=0 vs 1
  E. 跨板 vs 单板     : board=None(全部板) vs board=指定板

用法示例：
  # 实验 D：验证 reward 错配假说（开关小板，各 3000 轮）
  python run_experiments.py --experiment D --board 开关小板 --episodes 3000

  # 实验 A：专家注入有效性（开关小板，各 5000 轮）
  python run_experiments.py --experiment A --board 开关小板 --episodes 5000

  # 实验 E：跨板 vs 单板迁移（开关小板，各 5000 轮）
  python run_experiments.py --experiment E --board 开关小板 --episodes 5000

  # A+D+E 全部顺序执行（每组 10000 轮，全部 5 板）
  python run_experiments.py --experiment A,D,E --episodes 10000

产物：result/experiments/<exp>/<group>/ 下，与 run_compare.py 结构一致。
说明：实验 E 中跨板训练只跑一次，结果对各目标板复用；--board 为空时对全部板逐个做单板对比。
"""

import argparse
import logging
import os
import shutil
import sys

import torch

from run import Runner
from utils.pcb_scan import find_pcb_files
from utils.visualization import plot_loss_compare, plot_reward_compare

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _default_data_dir() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data", "kicad-native-batch"))


def _default_config_path() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "config", "argument.txt"))


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨板优化对照实验调度器")
    parser.add_argument(
        "--experiment", type=str, required=True,
        help="实验类型(可逗号组合, 顺序执行): A=专家有无 / D=reward尺度错配 / E=跨板vs单板, 例如 A,D,E",
    )
    parser.add_argument(
        "--board", type=str, default=None,
        help="板子名子串（实验 E 中留空=跨板训练）",
    )
    parser.add_argument(
        "--episodes", type=int, default=3000,
        help="每组训练总 episode 数（默认 3000，快速验证）",
    )
    parser.add_argument("--seed", type=int, default=114514, help="随机种子")
    parser.add_argument("--config_path", type=str, default=_default_config_path())
    parser.add_argument("--data_dir", type=str, default=_default_data_dir())
    parser.add_argument(
        "--max_episode_step", type=int, default=None,
        help="每个 episode 最大步数（None 用 argument.txt 的 MAX_STEPS）",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _scan_boards(data_dir: str, board_filter: str) -> list:
    """扫描并过滤训练板子，board_filter 为空则返回全部（跨板）。"""
    all_files = find_pcb_files(data_dir)
    if board_filter:
        filtered = [f for f in all_files if board_filter in os.path.basename(f)]
        if filtered:
            return [(os.path.basename(f), f) for f in filtered]
        logger.warning(
            f"Board filter '{board_filter}' matched no files, using all {len(all_files)}")
    return [(os.path.basename(f), f) for f in all_files]


def _patch_config(config_path: str, overrides: dict, out_path: str) -> str:
    """复制 argument.txt 并注入 overrides 键值，返回新文件路径。

    Args:
        config_path: 原始 config 路径
        overrides: 要覆盖的 {KEY: 值} 字典（写成 int）
        out_path: 输出临时 config 路径
    """
    lines = []
    applied = set()
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            key = line.split("=")[0].strip() if "=" in line else None
            if key and key in overrides:
                lines.append(f"{key}        = {int(overrides[key])}")
                applied.add(key)
            else:
                lines.append(line)
    # 若原 config 中缺失 override 键，则追加（保证注入一定生效）
    for k, v in overrides.items():
        if k not in applied:
            lines.append(f"{k}        = {int(v)}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"临时 config: {out_path} (overrides={overrides})")
    return out_path


def _build_runner_args(args, use_expert, save_dir, config_path, board,
                       expert_transitions_dir=""):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return argparse.Namespace(
        config_path=config_path,
        max_episode_step=args.max_episode_step,
        total_episodes=args.episodes,
        seed=args.seed,
        device=device,
        data_dir=args.data_dir,
        save_dir=save_dir,
        board=board,
        use_expert=use_expert,
        explicit_total_episodes=False,
        expert_transitions_dir=expert_transitions_dir,
    )


def _run_one(label, args, use_expert, config_path, board, save_dir,
             expert_transitions_dir=""):
    """运行一组训练，返回 Runner 或 None。"""
    logger.info("=" * 70)
    logger.info(f"==== 开始 {label}（use_expert={use_expert}, board={board or '跨板'}）====")
    logger.info("=" * 70)
    rargs = _build_runner_args(args, use_expert, save_dir, config_path, board,
                               expert_transitions_dir=expert_transitions_dir)
    runner = Runner(rargs)
    try:
        runner.run()
        return runner
    except Exception:
        logger.exception(f"{label} 训练失败")
        return None


def _print_pair(pcb_name, reward_g1, reward_g2, name1, name2):
    """打印两组 reward 汇总对比（前20%/后20%/最佳）。"""
    def _stat(history):
        vals = [r for _, r in history]
        if not vals:
            return None
        n = max(1, len(vals) // 5)
        return {"first": sum(vals[:n]) / n, "last": sum(vals[-n:]) / n,
                "best": max(vals)}

    s1, s2 = _stat(reward_g1), _stat(reward_g2)
    logger.info("=" * 70)
    logger.info(f"[{pcb_name}] 对比汇总: {name1} vs {name2}")
    if s1:
        logger.info(f"  {name1:18s}: 前20% R={s1['first']:8.2f}  "
                    f"后20% R={s1['last']:8.2f}  最佳 R={s1['best']:8.2f}")
    if s2:
        logger.info(f"  {name2:18s}: 前20% R={s2['first']:8.2f}  "
                    f"后20% R={s2['last']:8.2f}  最佳 R={s2['best']:8.2f}")
    if s1 and s2:
        logger.info(f"  后20% 差异: {name1}-{name2} = {s1['last']-s2['last']:+.2f}")
        logger.info(f"  最佳差异  : {name1}-{name2} = {s1['best']-s2['best']:+.2f}")
    logger.info("=" * 70)


def _run_experiment(args, exp, exp_root):
    """运行单个实验，返回该实验的产物根目录。"""
    exp_dir = os.path.join(exp_root, exp)

    if exp == "A":
        # 实验 A：with_expert vs baseline（单板或跨板，用 use_expert 开关）
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            cfg = _patch_config(args.config_path, {}, os.path.join(out_dir, "config.txt"))
            r1 = _run_one("with_expert", args, True, cfg, pcb_name,
                          os.path.join(out_dir, "with_expert"))
            r2 = _run_one("baseline", args, False, cfg, pcb_name,
                          os.path.join(out_dir, "baseline"))
            if r1 and r2:
                ra = r1.per_board_reward_history.get(pcb_name, [])
                rb = r2.per_board_reward_history.get(pcb_name, [])
                plot_reward_compare(ra, rb, ["with_expert", "baseline"],
                                    os.path.join(out_dir, "reward_compare.png"),
                                    pcb_name=pcb_short)
                _print_pair(pcb_name, ra, rb, "with_expert", "baseline")

    elif exp == "D":
        # 实验 D：reward 尺度错配（EXPERT_APPLY_PENALTIES=0 vs 1）
        # EXPERT_APPLY_PENALTIES 是 int 可经 config 注入；专家缓存目录则用
        # args.expert_transitions_dir 覆盖（字符串路径无法经 argument.txt 注入）。
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            # 组1：无惩罚专家（原行为，对照）
            expert1 = os.path.join(out_dir, "expert_nopen")
            cfg1 = _patch_config(
                args.config_path,
                {"EXPERT_APPLY_PENALTIES": 0},
                os.path.join(out_dir, "config_nopen.txt"))
            # 组2：带惩罚专家（假说修复）
            expert2 = os.path.join(out_dir, "expert_pen")
            cfg2 = _patch_config(
                args.config_path,
                {"EXPERT_APPLY_PENALTIES": 1},
                os.path.join(out_dir, "config_pen.txt"))
            r1 = _run_one("expert_nopen(原)", args, True, cfg1, pcb_name,
                          os.path.join(out_dir, "expert_nopen"),
                          expert_transitions_dir=expert1)
            r2 = _run_one("expert_pen(修复)", args, True, cfg2, pcb_name,
                          os.path.join(out_dir, "expert_pen"),
                          expert_transitions_dir=expert2)
            if r1 and r2:
                ra = r1.per_board_reward_history.get(pcb_name, [])
                rb = r2.per_board_reward_history.get(pcb_name, [])
                plot_reward_compare(ra, rb, ["nopen", "pen"],
                                    os.path.join(out_dir, "reward_compare.png"),
                                    pcb_name=pcb_short)
                _print_pair(pcb_name, ra, rb, "expert_nopen", "expert_pen")

    elif exp == "E":
        # 实验 E：跨板 vs 单板迁移
        # 目标板 = --board 指定的板子；若为空则对全部板逐个做跨板 vs 单板。
        target_boards = _scan_boards(args.data_dir, args.board or "")
        n_boards = len(target_boards)
        if n_boards == 0:
            logger.error("实验 E 未找到任何目标板，跳过")
            return exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        cfg = _patch_config(args.config_path, {}, os.path.join(exp_dir, "config.txt"))
        # 跨板训练只跑一次（board=None → 全部板），其结果对所有目标板复用。
        # 关键修复：跨板训练随机选板，每板平均仅分配到 total_episodes/N 轮；
        # 为与单板训练的 args.episodes 轮对等，跨板总轮数放大 N 倍（每板仍约 args.episodes 轮）。
        cross_dir = os.path.join(exp_dir, "cross")
        logger.info(
            f"实验 E：跨板训练总轮数放大 {n_boards} 倍 → {args.episodes * n_boards} "
            f"（每板平均约 {args.episodes} 轮，与单板对等）")
        orig_episodes = args.episodes
        args.episodes = args.episodes * n_boards
        try:
            r_cross = _run_one("跨板训练", args, True, cfg, None, cross_dir)
        finally:
            args.episodes = orig_episodes
        for pcb_name, _f in target_boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            # 单板训练
            r2 = _run_one("单板训练", args, True, cfg, pcb_name,
                          os.path.join(out_dir, "solo"))
            if r_cross and r2:
                ra = r_cross.per_board_reward_history.get(pcb_name, [])
                rb = r2.per_board_reward_history.get(pcb_name, [])
                plot_reward_compare(ra, rb, ["跨板", "单板"],
                                    os.path.join(out_dir, "reward_compare.png"),
                                    pcb_name=pcb_short)
                _print_pair(pcb_name, ra, rb, "跨板", "单板")

    logger.info(f"实验 {exp} 完成，产物目录: {exp_dir}")
    return exp_dir


def run():
    args = get_config()
    exp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result", "experiments")
    exps = [e.strip().upper() for e in args.experiment.split(",") if e.strip()]
    invalid = [e for e in exps if e not in ("A", "D", "E")]
    if invalid:
        logger.error(f"未知实验类型: {invalid}，合法值为 A/D/E（可逗号组合）")
        sys.exit(1)
    logger.info(f"待执行实验（顺序）: {exps}")
    for exp in exps:
        _run_experiment(args, exp, exp_root)
    logger.info(f"全部实验完成，产物根目录: {exp_root}")


if __name__ == "__main__":
    run()
