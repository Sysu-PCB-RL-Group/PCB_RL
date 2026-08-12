"""PCB_RL CLI 入口。

使用真实 KiCad PCB 数据训练：python main.py（默认跨 5 个板子随机训练）

基于 SPEC v2.1。
"""
import argparse
import os
import sys
import torch


def get_config():
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="PCB Layout RL Training (AABB-based, Gymnasium)"
    )

    # === 核心路径 ===
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "data", "kicad-native-batch",
        )),
        help="PCB 批次根目录（内含多个板子子目录，递归扫描 .kicad_pcb）",
    )
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "argument.txt")),
        help="超参数配置文件路径",
    )

    # === Debug 模式 ===
    parser.add_argument(
        "--debug", action="store_true",
        help="每 episode 打印日志 + reward 分解详情",
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

    # 标记 --total_episodes 是否为用户显式指定
    args.explicit_total_episodes = "--total_episodes" in sys.argv

    return args


def run():
    """主入口：解析参数 → 创建 Runner → 启动训练。"""
    args = get_config()
    os.makedirs(args.save_dir, exist_ok=True)

    # Debug 模式：打印每个 step 的 reward 分解和训练详情
    if getattr(args, "debug", False):
        import logging
        logging.getLogger("dqn").setLevel(logging.DEBUG)
        from env.pcb_layout_env import Config
        Config.LOG_INTERVAL = 1  # 每 episode 都打日志

    from run import Runner
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    run()
