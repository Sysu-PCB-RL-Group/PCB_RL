"""PCB_RL CLI 入口。

支持两种模式：
  - 真实 KiCad 模式：python main.py（默认使用开关小板）
  - Mock 模式（无需 KiCad）：python main.py --mock --data_dir data/模拟

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
            os.path.dirname(__file__), "kicad-native-batch",
            "ran-pang-airlink-86231b34--开关小板-883285b3",
        )),
        help="PCB 文件目录（.kicad_pcb 或 .json）",
    )
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "argument.txt")),
        help="超参数配置文件路径",
    )

    # === Mock 模式 ===
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock 模式（无需 KiCad，使用 data/模拟/*.json 数据）",
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

    # --mock 模式下自动切换默认 data_dir
    if args.mock and args.data_dir == os.path.abspath(
        os.path.join(os.path.dirname(__file__), "kicad-native-batch")):
        args.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "模拟"))

    return args


def run():
    """主入口：解析参数 → Mock 注入（如需）→ 创建 Runner → 启动训练。"""
    args = get_config()
    os.makedirs(args.save_dir, exist_ok=True)

    # Debug 模式：打印每个 step 的 reward 分解和训练详情
    if getattr(args, "debug", False):
        import logging
        logging.getLogger("dqn").setLevel(logging.DEBUG)
        from env.pcb_layout_env import Config
        Config.LOG_INTERVAL = 1  # 每 episode 都打日志

    # Mock 模式：必须在 import Runner 之前注入，阻止 utils.kicad_utils 加载 DLL
    if args.mock:
        from unittest import mock
        _m = mock.MagicMock()
        sys.modules["utils.kicad_utils"] = _m

        from env.mock_pcb import MockKiCadUtils, MockComponent, MockNet
        import utils as _utils
        _utils.kicad_utils = sys.modules["utils.kicad_utils"]
        _utils.kicad_utils.KiCadUtils = MockKiCadUtils
        _utils.kicad_utils.Component = MockComponent
        _utils.kicad_utils.Net = MockNet

    from run import Runner
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    run()
