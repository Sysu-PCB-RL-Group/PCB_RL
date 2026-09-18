"""可复用实验启动器：仅跑 A 组（with_expert）+ 当前 argument.txt 配置，按 --tag 区分 save_dir。

用法：
  $env:KICAD_PATH="E:\\Kicad"
  E:\\miniconda\\envs\\pcb_rl_311\\python.exe run_exp.py --tag bcreg_l10 --total_episodes 10000
"""
import os
import argparse

from run import Runner
from utils.pcb_scan import DEFAULT_BOARD

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run A-group (with_expert) experiment with current config")
    parser.add_argument("--tag", required=True, help=f"experiment tag → result/exp_<tag>/{DEFAULT_BOARD}/with_expert")
    parser.add_argument("--total_episodes", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=114514)
    args = parser.parse_args()

    save_dir = os.path.join(ROOT, "result", f"exp_{args.tag}", DEFAULT_BOARD, "with_expert")
    ns = argparse.Namespace(
        config_path=os.path.join(ROOT, "config", "argument.txt"),
        max_episode_step=None,
        total_episodes=args.total_episodes,
        seed=args.seed,
        device=args.device,
        data_dir=os.path.join(ROOT, "data", "kicad-native-batch"),
        save_dir=save_dir,
        board=DEFAULT_BOARD,
        use_expert=True,
        explicit_total_episodes=False,
    )
    Runner(ns).run()


if __name__ == "__main__":
    main()
