"""全量重新生成所有板子的专家轨迹。

背景：修复了 bbox y 顺序 bug、层误判 bug，并让专家 reward 去掉惩罚项后，
旧的专家 JSON（含负 reward 惩罚项）必须全部重新生成，否则 run.py 会
直接加载磁盘上的旧缓存继续用坏数据。

用法（KiCad 自带 Python，因为需要 pcbnew）：
    & "D:\\KiCad\\bin\\python.exe" _regen_all_expert.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.pcb_layout_env import PCBLayoutEnv, Config
from env.expert_trajectory import generate_and_save_expert_transitions
from utils.pcb_scan import find_pcb_files

Config.load_from_file("config/argument.txt")

expert_dir = "data/expert_transitions"
os.makedirs(expert_dir, exist_ok=True)

traj_per_board = Config.EXPERT_TRAJECTORIES_PER_BOARD
eps_decomp = Config.EXPERT_EPSILON_DECOMPOSE
pos_noise = Config.EXPERT_POS_NOISE_PROB
rot_noise = Config.EXPERT_ROT_NOISE_PROB
# 修复：必须把 EXPERT_APPLY_PENALTIES 传给生成器，
# 否则 apply_penalties 用默认 False，专家 reward 与 RL 环境口径不一致（reward 尺度错配）。
apply_pen = bool(getattr(Config, "EXPERT_APPLY_PENALTIES", False))

pcb_files = find_pcb_files("data")
print(f"Found {len(pcb_files)} PCB files")

for f in pcb_files:
    pcb_name = os.path.basename(f)
    try:
        env = PCBLayoutEnv(f, config_path="config/argument.txt")
    except Exception as e:
        print(f"[SKIP] {pcb_name}: {e}")
        continue
    if env.N_movable < 1 or len(env.kicad.nets) < 1:
        print(f"[SKIP] {pcb_name}: 非电路板")
        continue

    print(f"===== {pcb_name} =====")
    transitions, filepath = generate_and_save_expert_transitions(
        env=env,
        pcb_name=pcb_name,
        output_dir=expert_dir,
        trajectories_per_board=traj_per_board,
        epsilon_decompose=eps_decomp,
        pos_noise_prob=pos_noise,
        rot_noise_prob=rot_noise,
        apply_penalties=apply_pen,
    )
    rewards = [t[2] for t in transitions]
    neg = sum(1 for r in rewards if r < 0)
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"  {len(transitions)} transitions | reward: "
        f"min={min(rewards):.2f} max={max(rewards):.2f} mean={mean:.2f} | 负={neg}"
    )
    print(f"  [saved] {filepath}")

print("DONE")
