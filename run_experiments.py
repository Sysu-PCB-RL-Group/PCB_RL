# -*- coding: utf-8 -*-
"""跨板优化对照实验调度器。

复用 Runner / run_compare.py 的既有能力，通过 CLI 参数选择要跑的实验组。
每次运行前动态生成一个临时 argument.txt 副本（改 EXPERT_APPLY_PENALTIES /
EXPERT_TRANSITIONS_DIR 等），保证「唯一变量原则」，不修改原 config/argument.txt。

支持三种实验（可组合）：
  A. expert 有无      : with_expert vs baseline（use_expert 开关）
  D. reward 尺度错配  : EXPERT_APPLY_PENALTIES=0 vs 1
  E. 跨板 vs 单板     : board=None(全部板) vs board=指定板

用法示例（--board 缺省 = utils.pcb_scan.DEFAULT_BOARD，即 4.IMU小板）：
  # 实验 D：验证 reward 错配假说（默认单板，各 3000 轮）
  python run_experiments.py --experiment D --episodes 3000

  # 实验 A：专家注入有效性（默认单板，各 5000 轮）
  python run_experiments.py --experiment A --episodes 5000

  # 实验 E：跨板 vs 单板迁移（默认单板，各 5000 轮）
  python run_experiments.py --experiment E --episodes 5000

  # A+D+E 全部顺序执行（每组 10000 轮）
  python run_experiments.py --experiment A,D,E --episodes 10000

产物：result/experiments/<exp>/<group>/ 下，与 run_compare.py 结构一致。
说明：实验 E 中跨板训练只跑一次，结果对各目标板复用；--board 为空时对全部板逐个做单板对比。
注意：当前仅默认单板（4.IMU小板）兼容固定维度框架，其余 19 块超限 → 跨板臂（board=None）暂不可用。
"""

import argparse
import logging
import os
import shutil
import sys

import torch

from run import Runner
from utils.pcb_scan import DEFAULT_BOARD, find_pcb_files
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
        help="实验类型(可逗号组合, 顺序执行): A=专家有无 / D=reward尺度错配 / E=跨板vs单板 "
             "/ F=出界惩罚强度 / G=Curriculum升温速度+多板泛化 / H=Early stopping, 例如 A,D,E",
    )
    parser.add_argument(
        "--board", type=str, default=DEFAULT_BOARD,
        help=f"板子名子串（默认 '{DEFAULT_BOARD}' = 唯一兼容当前框架的板；"
             f"留空=跨板训练，但当前其余板均超限，跨板臂不可用）",
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
    parser.add_argument(
        "--early_stop_patience", type=int, default=0,
        help="实验 H 用：全局 best 停滞超过 N 个 episode 即提前终止训练（0=不早停，跑满）",
    )
    parser.add_argument(
        "--mrv_order", action="store_true",
        help="路线3：开启 MRV 顺序约束（step0 只放行剩余合法落子最少的元件，需重训做对照）",
    )
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
        overrides: 要覆盖的 {KEY: 值} 字典（支持 int/float）
        out_path: 输出临时 config 路径
    """
    lines = []
    applied = set()
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            key = line.split("=")[0].strip() if "=" in line else None
            if key and key in overrides:
                v = overrides[key]
                val_str = str(int(v)) if float(v).is_integer() else str(v)
                lines.append(f"{key}        = {val_str}")
                applied.add(key)
            else:
                lines.append(line)
    # 若原 config 中缺失 override 键，则追加（保证注入一定生效）
    for k, v in overrides.items():
        if k not in applied:
            val_str = str(int(v)) if float(v).is_integer() else str(v)
            lines.append(f"{k}        = {val_str}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"临时 config: {out_path} (overrides={overrides})")
    return out_path


def _build_runner_args(args, use_expert, save_dir, config_path, board,
                       expert_transitions_dir="", early_stop_patience=None):
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
        early_stop_patience=early_stop_patience if early_stop_patience is not None
        else getattr(args, "early_stop_patience", 0),
        mrv_order=getattr(args, "mrv_order", False),
    )


def _run_one(label, args, use_expert, config_path, board, save_dir,
             expert_transitions_dir="", early_stop_patience=None):
    """运行一组训练，返回 Runner 或 None。"""
    logger.info("=" * 70)
    logger.info(f"==== 开始 {label}（use_expert={use_expert}, board={board or '跨板'}）====")
    logger.info("=" * 70)
    rargs = _build_runner_args(args, use_expert, save_dir, config_path, board,
                               expert_transitions_dir=expert_transitions_dir,
                               early_stop_patience=early_stop_patience)
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

    elif exp == "F":
        # 实验 F：出界惩罚强度消融（W_OUT_OF_BOUNDS_FINAL 高 vs 低）
        # 假说：后期出界惩罚 20.0 >> 单步完成奖励(100/N_movable)，导致 Q 网络
        # "只学规避惩罚、不学完成布局"。降低 W_OUT_OF_BOUNDS_FINAL 看是否改善学习。
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            # 组1：高惩罚（原值 20，对照）
            cfg1 = _patch_config(
                args.config_path,
                {"W_OUT_OF_BOUNDS_FINAL": 20},
                os.path.join(out_dir, "config_oob20.txt"))
            # 组2：低惩罚（假说修复，降至完成奖励同量级以下）
            cfg2 = _patch_config(
                args.config_path,
                {"W_OUT_OF_BOUNDS_FINAL": 6},
                os.path.join(out_dir, "config_oob6.txt"))
            r1 = _run_one("oob20(原)", args, True, cfg1, pcb_name,
                          os.path.join(out_dir, "oob20"))
            r2 = _run_one("oob6(消融)", args, True, cfg2, pcb_name,
                          os.path.join(out_dir, "oob6"))
            if r1 and r2:
                ra = r1.per_board_reward_history.get(pcb_name, [])
                rb = r2.per_board_reward_history.get(pcb_name, [])
                plot_reward_compare(ra, rb, ["oob20(原)", "oob6(消融)"],
                                    os.path.join(out_dir, "reward_compare.png"),
                                    pcb_name=pcb_short)
                _print_pair(pcb_name, ra, rb, "oob20(原)", "oob6(消融)")

    elif exp == "G":
        # 实验 G：Curriculum 升温速度消融 + 多板泛化
        # 假说：CURRICULUM_END_EP 决定出界惩罚从 2→20 的升温快慢。
        #   快升温(150ep)：前期太早收紧，探索期大量出界惩罚压制学习。
        #   慢升温(1500ep)：惩罚长时间宽松，后期可能来不及收紧出界行为。
        #   默认(500ep)：原配置，作对照。
        # 对每个扫描到的板都跑三档，同时验证跨板泛化是否一致。
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            variants = [
                ("fast_150", "快升温(150)", 150),
                ("default_500", "默认(500)", 500),
                ("slow_1500", "慢升温(1500)", 1500),
            ]
            results = {}  # name -> (runner, label)
            for folder, label, end_ep in variants:
                cfg = _patch_config(
                    args.config_path,
                    {"CURRICULUM_END_EP": end_ep},
                    os.path.join(out_dir, f"config_{folder}.txt"))
                r = _run_one(f"curriculum_{folder}({label})", args, True, cfg,
                             pcb_name, os.path.join(out_dir, folder))
                results[folder] = (r, label)
            # 绘图与对比（两两对比 + 汇总表）
            histories = {folder: (r.per_board_reward_history.get(pcb_name, [])
                                  if r else []) for folder, (r, _) in results.items()}
            labels = [label for _, label in results.values()]
            # 三线叠加图
            plot_reward_compare(histories["fast_150"], histories["default_500"],
                                [labels[0], labels[1]],
                                os.path.join(out_dir, "reward_compare_fast_vs_default.png"),
                                pcb_name=pcb_short)
            plot_reward_compare(histories["slow_1500"], histories["default_500"],
                                [labels[2], labels[1]],
                                os.path.join(out_dir, "reward_compare_slow_vs_default.png"),
                                pcb_name=pcb_short)
            # 汇总表
            logger.info("=" * 70)
            logger.info(f"[{pcb_name}] Curriculum 升温速度对比汇总")
            for folder in ("fast_150", "default_500", "slow_1500"):
                h = histories[folder]
                vals = [r for _, r in h]
                if not vals:
                    logger.info(f"  {results[folder][1]:14s}: 无数据")
                    continue
                n = max(1, len(vals) // 5)
                logger.info(f"  {results[folder][1]:14s}: "
                            f"episodes={len(vals)} 前20% R={sum(vals[:n])/n:8.2f}  "
                            f"后20% R={sum(vals[-n:])/n:8.2f}  最佳 R={max(vals):8.2f}")
            logger.info("=" * 70)

    elif exp == "H":
        # 实验 H：Early stopping（方向A - 后期策略漂移）
        # 假说：训练拖满 total_episodes，后期无条件更新把早期学好的 best 策略冲掉。
        #   通过「全局 best 停滞 N ep 就早停」验证：若早停组后期 R 明显更高/更稳，
        #   说明后期退化确实源于「训练过久导致策略漂移」。
        # 四组：无早停(对照) / 停滞300停 / 停滞100停 / 停滞30停
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            variants = [
                ("no_es", "无早停(对照)", 0),
                ("es_300", "早停300ep", 300),
                ("es_100", "早停100ep", 100),
                ("es_30", "早停30ep", 30),
            ]
            results = {}
            for folder, label, patience in variants:
                cfg = _patch_config(args.config_path, {},
                                    os.path.join(out_dir, f"config_{folder}.txt"))
                sub = os.path.join(out_dir, folder)
                r = _run_one(f"earlystop_{folder}({label})", args, True, cfg,
                             pcb_name, sub, early_stop_patience=patience)
                results[folder] = (r, label)
            # 汇总表
            logger.info("=" * 70)
            logger.info(f"[{pcb_name}] Early stopping 对比汇总")
            for folder, label in [(f, l) for f, (_, l) in results.items()]:
                r = results[folder][0]
                h = r.per_board_reward_history.get(pcb_name, []) if r else []
                vals = [x for _, x in h]
                if not vals:
                    logger.info(f"  {label:14s}: 无数据")
                    continue
                n = max(1, len(vals) // 5)
                logger.info(f"  {label:14s}: 实跑ep={len(vals):4d}  "
                            f"前20% R={sum(vals[:n])/n:8.2f}  "
                            f"后20% R={sum(vals[-n:])/n:8.2f}  最佳 R={max(vals):8.2f}")
            logger.info("=" * 70)

    elif exp == "I":
        # 实验 I（治本）：诊断发现两个病根——ε 衰减太慢(ep198 仍 0.39) + Q 值发散(loss 60~200)
        # fix_eps     : 仅加快 ε 衰减 (EPSILON_DECAY_EPISODES=300, ε 提前降到底)
        # fix_eps_reg : ε 快衰减 + 加强 BC 值正则 (BC_REG_VALUE_LAMBDA 10→50) 压 Q 值
        # 对照组：es_200 原配置（复现诊断结果）
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            variants = [
                ("ctrl_es200", "对照(原配置)", {}),
                ("fix_eps", "仅ε快衰减", {"EPSILON_DECAY_EPISODES": 300}),
                ("fix_eps_reg", "ε快衰减+强值正则", {"EPSILON_DECAY_EPISODES": 300,
                                                     "BC_REG_VALUE_LAMBDA": 50}),
            ]
            results = {}
            for folder, label, ov in variants:
                cfg = _patch_config(args.config_path, ov,
                                    os.path.join(out_dir, f"config_{folder}.txt"))
                sub = os.path.join(out_dir, folder)
                r = _run_one(f"fix_{folder}({label})", args, True, cfg,
                             pcb_name, sub, early_stop_patience=200)
                results[folder] = (r, label)
            # 汇总表
            logger.info("=" * 70)
            logger.info(f"[{pcb_name}] 治本实验(I) 对比汇总 (early_stop_patience=200)")
            for folder, label in [(f, l) for f, (_, l) in results.items()]:
                r = results[folder][0]
                h = r.per_board_reward_history.get(pcb_name, []) if r else []
                vals = [x for _, x in h]
                if not vals:
                    logger.info(f"  {label:16s}: 无数据")
                    continue
                n = max(1, len(vals) // 5)
                logger.info(f"  {label:16s}: 实跑ep={len(vals):4d}  "
                            f"前20% R={sum(vals[:n])/n:8.2f}  "
                            f"后20% R={sum(vals[-n:])/n:8.2f}  最佳 R={max(vals):8.2f}")
            logger.info("=" * 70)

    elif exp == "J":
        # 实验 J（治本·Q值尺度）：诊断发现 BC 交叉熵 logits 尺度无约束(±30~100)，
        # 与 RL 归一化 reward(±1~3) 错配 → loss 60+ 永不收敛、reward 震荡±160。
        # 修复：BC 后缩放 Q 输出层(×factor) 适配 reward 尺度。
        # 组1 ctrl        : 不缩放(复现)     组2 fix_q01  : 缩放×0.1
        # 组3 fix_q01_eps : 缩放×0.1 + ε快衰减(EPSILON_DECAY_EPISODES=300)
        boards = _scan_boards(args.data_dir, args.board or "")
        for pcb_name, _f in boards:
            pcb_short = pcb_name.replace(".kicad_pcb", "")
            out_dir = os.path.join(exp_dir, pcb_short)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            variants = [
                ("ctrl", "对照(不缩放)", {"Q_SCALE_FACTOR": 0}),
                ("fix_q01", "Q×0.1", {"Q_SCALE_FACTOR": 0.1}),
                ("fix_q01_eps", "Q×0.1+ε快衰减", {"Q_SCALE_FACTOR": 0.1,
                                                   "EPSILON_DECAY_EPISODES": 300}),
            ]
            results = {}
            for folder, label, ov in variants:
                cfg = _patch_config(args.config_path, ov,
                                    os.path.join(out_dir, f"config_{folder}.txt"))
                sub = os.path.join(out_dir, folder)
                r = _run_one(f"J_{folder}({label})", args, True, cfg,
                             pcb_name, sub, early_stop_patience=200)
                results[folder] = (r, label)
            # 汇总表
            logger.info("=" * 70)
            logger.info(f"[{pcb_name}] 治本实验(J·Q值尺度) 对比汇总 (early_stop_patience=200)")
            for folder, label in [(f, l) for f, (_, l) in results.items()]:
                r = results[folder][0]
                h = r.per_board_reward_history.get(pcb_name, []) if r else []
                vals = [x for _, x in h]
                if not vals:
                    logger.info(f"  {label:16s}: 无数据")
                    continue
                n = max(1, len(vals) // 5)
                logger.info(f"  {label:16s}: 实跑ep={len(vals):4d}  "
                            f"前20% R={sum(vals[:n])/n:8.2f}  "
                            f"后20% R={sum(vals[-n:])/n:8.2f}  最佳 R={max(vals):8.2f}")
            logger.info("=" * 70)

    logger.info(f"实验 {exp} 完成，产物目录: {exp_dir}")
    return exp_dir


def run():
    args = get_config()
    exp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "result", "experiments")
    exps = [e.strip().upper() for e in args.experiment.split(",") if e.strip()]
    invalid = [e for e in exps if e not in ("A", "D", "E", "F", "G", "H", "I", "J")]
    if invalid:
        logger.error(f"未知实验类型: {invalid}，合法值为 A/D/E/F/G/H/I/J（可逗号组合）")
        sys.exit(1)
    logger.info(f"待执行实验（顺序）: {exps}")
    for exp in exps:
        _run_experiment(args, exp, exp_root)
    logger.info(f"全部实验完成，产物根目录: {exp_root}")


if __name__ == "__main__":
    run()
