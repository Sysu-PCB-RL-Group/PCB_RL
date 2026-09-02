"""伪造轨迹有效性对比实验入口。

顺序跑两组训练，唯一变量为「是否使用伪造轨迹（专家轨迹注入）」：
- A 组（with_expert）：Phase 1 BC 预训练 + Phase 2 RL（ExpertReplayBuffer + PER + 20% 专家常驻）
- B 组（baseline）  ：纯在线 RL，标准 ReplayBuffer，彻底无专家注入（跳过 BC 预训练）

两组除 use_expert 外，episode 数 / seed / 惩罚权重 / epsilon 衰减 /
target 同步频率 / 500ep 预热期全部一致，保证唯一变量原则。

产物（result/compare/<pcb_short>/ 下）：
- with_expert/   A 组完整产物（training_log.txt、best_model.pt、bc_pretrained.pt、<pcb_short>/ 子目录）
- baseline/      B 组完整产物（training_log.txt、best_model.pt、<pcb_short>/ 子目录）
- loss_compare.png    对比版 loss 曲线（A/B 同一坐标系）
- reward_compare.png  对比版 reward 曲线（A/B 同一坐标系）

不 import 也不修改 main.py / run.py 现有行为，直接复用 Runner 的 use_expert 开关。
"""

import argparse
import json
import logging
import os
import shutil
import sys

import torch

from run import Runner
from dqn import DQNAgent
from env.pcb_layout_env import Config
from inference_eval import run_inference_eval
from utils.pcb_scan import find_pcb_files
from utils.visualization import plot_loss_compare, plot_reward_compare, plot_inference_eval

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 默认参数（用户澄清：默认开关小板，跑 10000 epi）
DEFAULT_BOARD = "开关小板"
DEFAULT_TOTAL_EPISODES = 10000
DEFAULT_SEED = 114514


def _default_data_dir() -> str:
    """返回默认训练数据目录（与 main.py 保持一致）。"""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data", "kicad-native-batch",
    ))


def _default_config_path() -> str:
    """返回默认超参数配置文件路径（与 main.py 保持一致）。"""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "config", "argument.txt",
    ))


def get_config() -> argparse.Namespace:
    """解析对比实验 CLI 参数。

    Returns:
        argparse.Namespace 参数对象
    """
    parser = argparse.ArgumentParser(
        description="伪造轨迹有效性对比实验（A=with_expert / B=baseline）"
    )
    parser.add_argument(
        "--board", type=str, default=DEFAULT_BOARD,
        help="训练板子名（子串匹配），默认开关小板",
    )
    parser.add_argument(
        "--total_episodes", type=int, default=DEFAULT_TOTAL_EPISODES,
        help="每组训练的总 episode 数（默认 10000）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--config_path", type=str, default=_default_config_path())
    parser.add_argument("--data_dir", type=str, default=_default_data_dir())
    parser.add_argument(
        "--max_episode_step", type=int, default=None,
        help="每个 episode 最大步数（None 则用 argument.txt 的 MAX_STEPS）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备 (cpu / cuda)，None 则自动选择",
    )
    return parser.parse_args()


def _scan_boards(data_dir: str, board_filter: str) -> list:
    """扫描并过滤训练板子，返回 [(pcb_name, filepath), ...]。

    与 Runner._get_pcb_files 的过滤逻辑一致（子串包含匹配 basename）。

    Args:
        data_dir: PCB 数据根目录
        board_filter: 板子名过滤子串（None/空则返回全部）

    Returns:
        (pcb_name, filepath) 列表，pcb_name = os.path.basename(filepath)
    """
    all_files = find_pcb_files(data_dir)
    if board_filter:
        filtered = [f for f in all_files if board_filter in os.path.basename(f)]
        if filtered:
            return [(os.path.basename(f), f) for f in filtered]
        logger.warning(
            f"Board filter '{board_filter}' matched no files, "
            f"using all {len(all_files)} files"
        )
    return [(os.path.basename(f), f) for f in all_files]


def _build_runner_args(
    args: argparse.Namespace,
    use_expert: bool,
    save_dir: str,
    board: str = None,
) -> argparse.Namespace:
    """构造传给 Runner 的完整参数对象（补充 main.py 缺失的 use_expert 字段）。

    Args:
        args: 顶层 CLI 参数
        use_expert: 是否启用专家轨迹注入（A=True / B=False）
        save_dir: 本组结果保存目录
        board: 精确的板子名（basename），None 则回退用 args.board 子串过滤

    Returns:
        Runner 可用的 argparse.Namespace 参数对象
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return argparse.Namespace(
        config_path=args.config_path,
        max_episode_step=args.max_episode_step,
        total_episodes=args.total_episodes,
        seed=args.seed,
        device=device,
        data_dir=args.data_dir,
        save_dir=save_dir,
        # 精确指定当前板子名，避免 --board 匹配多板时每组重复训练全部板子
        board=board or args.board,
        use_expert=use_expert,
        # 对比实验无需保存 final_ep*.pt 中间 checkpoint，保持产物干净
        explicit_total_episodes=False,
    )


def _print_summary(
    pcb_short: str,
    reward_a: list,
    reward_b: list,
):
    """打印两组前后 20% 平均 reward 与最佳 reward 汇总。

    最佳 reward 由 `max(reward_history)` 计算，与 per_board_best_reward 等价，
    无需额外传参。

    Args:
        pcb_short: 板子短名（无 .kicad_pcb 后缀）
        reward_a: A 组 [(ep, reward), ...]（仅正式训练阶段）
        reward_b: B 组 [(ep, reward), ...]
    """
    def _stat(history):
        vals = [r for _, r in history]
        if not vals:
            return None
        n = max(1, len(vals) // 5)
        return {
            "first": sum(vals[:n]) / n,
            "last": sum(vals[-n:]) / n,
            "best": max(vals),
        }

    sa, sb = _stat(reward_a), _stat(reward_b)

    logger.info("=" * 70)
    logger.info(f"[{pcb_short}] 对比汇总")
    if sa:
        logger.info(
            f"  A 组 (with_expert): 前20% R={sa['first']:8.2f}  "
            f"后20% R={sa['last']:8.2f}  最佳 R={sa['best']:8.2f}"
        )
    if sb:
        logger.info(
            f"  B 组 (baseline)    : 前20% R={sb['first']:8.2f}  "
            f"后20% R={sb['last']:8.2f}  最佳 R={sb['best']:8.2f}"
        )
    if sa and sb:
        delta_last = sa["last"] - sb["last"]
        delta_best = sa["best"] - sb["best"]
        logger.info(f"  后20% 差异: A-B = {delta_last:+.2f}")
        logger.info(f"  最佳差异  : A-B = {delta_best:+.2f}")
        if delta_last > 0:
            logger.info("  初步结论: 伪造轨迹有效（A 组后20%平均 R 更高）")
        else:
            logger.info("  初步结论: 伪造轨迹无效 / 无明显提升（A 组未优于 B 组）")
    logger.info("=" * 70)


def _load_agent_for_eval(model_path: str, config_path: str, device: str = "cpu") -> "DQNAgent":
    """加载一个已训模型用于 100 次推理评测（与 run_inference.py::main 同范式）。

    Args:
        model_path: best_model.pt 路径
        config_path: 训练用 config（保证网络维度与 reward 权重口径一致）
        device: 推理设备（默认 cpu）

    Returns:
        已加载权重的 DQNAgent
    """
    Config.load_from_file(config_path)
    agent = DQNAgent(
        n_movable=Config.MAX_MOVABLE,
        grid_w=Config.GRID_W,
        grid_h=Config.GRID_H,
        device=device,
        lr=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
    )
    agent.load(model_path)
    logger.info(f"[inference_eval] Agent loaded weights from: {model_path}")
    return agent


def run():
    """主入口：顺序跑 A/B 两组训练 → 画对比图 → 打印汇总。"""
    args = get_config()

    # 扫描板子（与 Runner._get_pcb_files 逻辑一致）
    boards = _scan_boards(args.data_dir, args.board)
    if not boards:
        logger.error(f"No PCB files found in {args.data_dir}")
        sys.exit(1)

    # 记录训练失败的组，最终以非零退出码上报，避免"假成功"
    failed_groups: list = []

    compare_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "result", "compare",
    )

    logger.info("=" * 70)
    logger.info("伪造轨迹有效性对比实验")
    logger.info(f"板子: {[n for n, _ in boards]}")
    logger.info(f"每组 episodes: {args.total_episodes}   seed: {args.seed}")
    logger.info(f"产物目录: {compare_root}")
    logger.info("=" * 70)

    # 逐板对比（默认单板：开关小板）
    for pcb_name, _filepath in boards:
        pcb_short = pcb_name.replace(".kicad_pcb", "")
        board_out_dir = os.path.join(compare_root, pcb_short)
        # 重跑前清空本板旧产物，避免上次对比结果（图/模型/日志）残留误导
        if os.path.isdir(board_out_dir):
            shutil.rmtree(board_out_dir)
            logger.info(f"Cleaned old compare dir: {board_out_dir}")
        save_dir_a = os.path.join(board_out_dir, "with_expert")
        save_dir_b = os.path.join(board_out_dir, "baseline")

        # ---- A 组：with_expert（board=pcb_name 精确过滤，避免多板重复训练） ----
        logger.info("=" * 70)
        logger.info(f"==== 开始 A 组训练（with expert）: {pcb_short} ====")
        logger.info("=" * 70)
        args_a = _build_runner_args(args, use_expert=True, save_dir=save_dir_a,
                                    board=pcb_name)
        runner_a = Runner(args_a)
        try:
            runner_a.run()
        except Exception:
            logger.exception(f"A 组训练失败（{pcb_short}），跳过本板对比")
            failed_groups.append(f"{pcb_short}/with_expert")
            continue

        # ---- B 组：baseline ----
        logger.info("=" * 70)
        logger.info(f"==== 开始 B 组训练（baseline）: {pcb_short} ====")
        logger.info("=" * 70)
        args_b = _build_runner_args(args, use_expert=False, save_dir=save_dir_b,
                                    board=pcb_name)
        runner_b = Runner(args_b)
        try:
            runner_b.run()
        except Exception:
            logger.exception(f"B 组训练失败（{pcb_short}），跳过本板对比")
            failed_groups.append(f"{pcb_short}/baseline")
            continue

        # ---- 读内存数据画对比图（仅正式训练阶段，两组天然对齐） ----
        loss_a = runner_a.per_board_loss_history.get(pcb_name, [])
        loss_b = runner_b.per_board_loss_history.get(pcb_name, [])
        reward_a = runner_a.per_board_reward_history.get(pcb_name, [])
        reward_b = runner_b.per_board_reward_history.get(pcb_name, [])

        labels = ["with_expert", "baseline"]
        plot_loss_compare(
            loss_a, loss_b, labels,
            os.path.join(board_out_dir, "loss_compare.png"),
            pcb_name=pcb_short,
        )
        plot_reward_compare(
            reward_a, reward_b, labels,
            os.path.join(board_out_dir, "reward_compare.png"),
            pcb_name=pcb_short,
        )

        _print_summary(pcb_short, reward_a, reward_b)

        # ---- 100 次推理评测（对比实验新评测标准）----
        # 对 A/B 各自 best_model.pt 各跑 100 次独立贪心推理，记录每条最终结果
        # （完成度 / 尝试次数 / reward），存原始 JSON 并画 A/B 三合一分布对比图。
        try:
            eval_root = os.path.join(board_out_dir, "inference_eval")
            os.makedirs(eval_root, exist_ok=True)
            eval_records: dict = {}

            for group, save_dir in (("with_expert", save_dir_a), ("baseline", save_dir_b)):
                model_path = os.path.join(save_dir, "best_model.pt")
                if not os.path.isfile(model_path):
                    logger.warning(
                        f"[inference_eval] 跳过 {group}：找不到模型 {model_path}"
                    )
                    continue
                agent = _load_agent_for_eval(model_path, args.config_path, device="cpu")
                logger.info("=" * 70)
                logger.info(
                    f"==== 100 次推理评测：{group} ({pcb_short}) ===="
                )
                logger.info("=" * 70)
                rec = run_inference_eval(
                    _filepath, agent, n=100, inference_epsilon=0.01
                )
                eval_records[group] = rec

                grp_dir = os.path.join(eval_root, group)
                os.makedirs(grp_dir, exist_ok=True)
                with open(
                    os.path.join(grp_dir, "inference_eval.json"),
                    "w", encoding="utf-8",
                ) as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)

                s = rec["stats"]
                logger.info(
                    f"[inference_eval] {group}: "
                    f"完整放置率={s['full_place_rate']:.2%}  "
                    f"平均完成度={s['completion_ratio']['mean']:.3f}  "
                    f"平均Reward={s['rewards']['mean']:.2f}  "
                    f"平均尝试={s['attempts']['mean']:.2f}"
                )

            # A/B 都成功评测才画对比图 + 打印结论
            if "with_expert" in eval_records and "baseline" in eval_records:
                plot_inference_eval(
                    eval_records["with_expert"],
                    eval_records["baseline"],
                    pcb_short,
                    os.path.join(eval_root, "inference_eval_compare.png"),
                )
                sa = eval_records["with_expert"]["stats"]
                sb = eval_records["baseline"]["stats"]
                logger.info("=" * 70)
                logger.info(f"[{pcb_short}] 100 次推理评测对比")
                logger.info(
                    f"  完整放置率: A={sa['full_place_rate']:.2%}  "
                    f"B={sb['full_place_rate']:.2%}  "
                    f"Δ={sa['full_place_rate'] - sb['full_place_rate']:+.2%}"
                )
                logger.info(
                    f"  平均完成度: A={sa['completion_ratio']['mean']:.3f}  "
                    f"B={sb['completion_ratio']['mean']:.3f}"
                )
                logger.info(
                    f"  平均 Reward: A={sa['rewards']['mean']:.2f}  "
                    f"B={sb['rewards']['mean']:.2f}  "
                    f"Δ={sa['rewards']['mean'] - sb['rewards']['mean']:+.2f}"
                )
                # 对比口径：reward 用 100 次推理的整体平均 reward
                da = sa["rewards"]["mean"]
                db = sb["rewards"]["mean"]
                if da - db > 0:
                    logger.info("  初步结论: 伪造轨迹有效（A 组平均Reward更高）")
                else:
                    logger.info("  初步结论: 伪造轨迹无效 / 无明显提升（A 组未优于B组）")
                logger.info("=" * 70)
        except Exception:
            logger.exception(
                "[inference_eval] 评测失败（不影响对比实验训练主结论），跳过"
            )

    if failed_groups:
        logger.error(f"以下组训练失败，对比不完整: {failed_groups}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"对比实验完成，产物目录: {compare_root}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run()
