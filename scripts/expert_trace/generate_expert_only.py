"""仅生成专家轨迹（不训练、不对比）。

用法（在用户终端）:
    python generate_expert_only.py --board IMU    # 缺省即默认单板（4.IMU小板）

流程:
    1. 加载指定 PCB（真实 KiCad，需 KICAD_PATH）
    2. 调用 generate_and_save_expert_transitions() 直接生成 0.2mm 专家轨迹
       （复用 run.py 相同生成逻辑，但不进入训练/BC/RL）
"""
import argparse
import glob
import logging
import os

from env.expert_trajectory import generate_and_save_expert_transitions
from env.pcb_layout_env import PCBLayoutEnv, Config
from utils.pcb_scan import DEFAULT_BOARD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "kicad-native-batch"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "expert_transitions"))


def _find_pcb(board_filter: str) -> str:
    """按名称子串匹配一个 .kicad_pcb 文件路径。"""
    files = glob.glob(os.path.join(DATA_DIR, "*", "*.kicad_pcb"))
    if board_filter:
        matches = [f for f in files if board_filter in os.path.basename(f)]
        if matches:
            return matches[0]
    if len(files) == 1:
        return files[0]
    raise SystemExit(
        f"需唯一匹配一个 PCB：filter='{board_filter}' 匹配 {len(files)} 个，"
        f"请用 --board 指定精确子串"
    )


def main():
    parser = argparse.ArgumentParser(description="仅生成专家轨迹（不训练）")
    parser.add_argument("--board", type=str, default=DEFAULT_BOARD,
                        help=f"PCB 名称子串（默认 '{DEFAULT_BOARD}' = 默认单板）")
    args = parser.parse_args()

    pcb_path = _find_pcb(args.board)
    pcb_name = os.path.basename(pcb_path)
    logger.info(f"加载 PCB: {pcb_path}")

    env = PCBLayoutEnv(os.path.abspath(pcb_path))
    logger.info(
        f"env: grid=({env.grid_width},{env.grid_height}), CELL_SIZE={Config.CELL_SIZE}, "
        f"GRID_W/H=({Config.GRID_W},{Config.GRID_H})"
    )

    transitions, filepath = generate_and_save_expert_transitions(
        env=env,
        pcb_name=pcb_name,
        output_dir=OUTPUT_DIR,
        total_transitions=Config.EXPERT_TOTAL_TRANSITIONS,
        max_trajectories=Config.EXPERT_MAX_TRAJECTORIES,
        epsilon_decompose=Config.EXPERT_EPSILON_DECOMPOSE,
        pos_noise_prob=Config.EXPERT_POS_NOISE_PROB,
        rot_noise_prob=Config.EXPERT_ROT_NOISE_PROB,
    )
    logger.info(f"生成完成：{len(transitions)} 条 → {filepath}")
    env.close()


if __name__ == "__main__":
    main()
