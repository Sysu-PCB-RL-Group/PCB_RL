"""PCB_RL CLI 入口。

基于 SPEC v2.1。
"""
import argparse
import os
import torch

from run import Runner


def get_config():
    """解析 CLI 参数。

    包括核心路径、训练参数、设备选择和保存路径。

    Returns:
        argparse.Namespace 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="PCB Layout RL Training (AABB-based, Gymnasium)"
    )

    # === 核心路径 ===
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "kicad-native-batch")),
        help="PCB 文件目录",
    )
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "argument.txt")),
        help="超参数配置文件路径",
    )

    # === 训练参数 ===
    parser.add_argument("--total_episodes", type=int, default=10000, help="总训练 episode 数")
    parser.add_argument("--max_episode_step", type=int, default=100, help="每个 episode 最大步数")

    # === 环境 & 设备 ===
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备 (cpu / cuda)",
    )
    parser.add_argument("--seed", type=int, default=114514, help="随机种子")

    # === 保存 ===
    parser.add_argument("--save_dir", type=str, default="./result", help="模型保存目录")
    parser.add_argument("--task_name", type=str, default="pcb_layout", help="任务名称")

    args = parser.parse_args()
    return args


def run():
    """主入口：解析参数 → 创建 Runner → 启动训练。"""
    args = get_config()
    os.makedirs(args.save_dir, exist_ok=True)

    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    run()
