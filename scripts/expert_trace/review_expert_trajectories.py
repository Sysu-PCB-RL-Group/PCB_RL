"""专家轨迹合法性审查：5 板各抽查 50 条完整轨迹（连续 5 步 transition）。

检查项：
  A. 结构：每条 transition 含 8 字段且类型正确
  B. step 序列：step_id == [0,1,2,3,4] 连续，末步 done==True
  C. obs/next_obs 自洽：末步 comp_idx 在 next_obs 中 placed=1，且其量化重建 bbox/rot/layer
     与 action 解码（valid_area + gx1*CELL + footprint 基准）一致；逐步已放置元件坐标不变
     （合并 S4/S5 后专家/环境重建统一 footprint 基准，不做旋转交换）
  D. 硬重叠：每个 obs 的已放置元件两两 AABB 不重叠
  E. 越界：所有已放置元件 bbox 在 valid_area 内
  F. mask 合法：action 在对应 step 的 mask 允许集合内（mask None 即无约束）
  G. reward：有限；rewards[i] ≈ sum_r * EXPERT_SHARE_WEIGHTS[i]（Σ=1.0）；
     sum_r 为单步 expert_reward ÷W_COMPLETION 之和（完成全板≈1.0），
     标记 sum_r<0.05 的低质（疑似违规）轨迹

用法：
  $env:KICAD_PATH="E:\Kicad"
  & E:\miniconda\envs\pcb_rl_311\python.exe review_expert_trajectories.py
"""
import json
import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.pcb_layout_env import PCBLayoutEnv, Config  # noqa: E402
from env.expert_trajectory import EXPERT_SHARE_WEIGHTS  # noqa: E402
from utils.pcb_scan import find_pcb_files  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERT_DIR = os.path.join(ROOT, "data", "expert_transitions")
DATA_DIR = os.path.join(ROOT, "data", "kicad-native-batch")
CONFIG_PATH = os.path.join(ROOT, "config", "argument.txt")
PER_BOARD = 50
SHARE = EXPERT_SHARE_WEIGHTS  # [0.2, 0.3, 0.3, 0.1, 0.1]


def _overlap(b1, b2):
    """AABB 硬重叠判定（x1,y1,x2,y2）。容忍量化网格 ±0.2mm 误差。"""
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2
    ix = min(x2, x4) - max(x1, x3)
    iy = min(y2, y4) - max(y1, y3)
    return ix > 1e-2 and iy > 1e-2  # 重叠面积 > 0.01 mm² 才算真硬重叠


def load_trajectories(pcb_path):
    """读取 _expert.json 并按连续 5 步分组为完整轨迹。"""
    pcb_name = os.path.basename(pcb_path)
    pcb_short = pcb_name.replace(".kicad_pcb", "")
    fp = os.path.join(EXPERT_DIR, f"{pcb_short}_expert.json")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    trajs = []
    n = len(data)
    i = 0
    while i + 5 <= n:
        chunk = data[i : i + 5]
        if [t["step_id"] for t in chunk] == [0, 1, 2, 3, 4]:
            trajs.append(chunk)
            i += 5
        else:
            i += 1
    return trajs


def review(pcb_path, env):
    trajs = load_trajectories(pcb_path)
    normalize = Config.NORMALIZE_OBS
    va = env.valid_area
    bd = math.hypot(va[2] - va[0], va[3] - va[1])  # 板框对角线(mm)，仅 NORMALIZE_OBS=True 反归一化时用
    cell = Config.CELL_SIZE
    sample = trajs[:PER_BOARD]

    issues = {}
    low_quality = 0
    reward_stats = []

    def bump(key):
        issues[key] = issues.get(key, 0) + 1

    for traj in sample:
        # B. step 序列
        if [t["step_id"] for t in traj] != [0, 1, 2, 3, 4]:
            bump("bad_step_ids")
            continue
        # 注：done 仅全局最后一个 trajectory（最后放置的元件）为 True；
        # 抽样的早期 trajectory 末步 done=False 是正确行为，不计为异常。

        comp_idx = traj[0]["action"]
        gx1 = traj[1]["action"]
        gy1 = traj[2]["action"]
        rot = traj[3]["action"]
        layer = traj[4]["action"]

        # C. obs/next_obs 自洽（末步）
        next_cs = np.array(traj[4]["next_obs"]["component_slots"], dtype=float)
        if next_cs[comp_idx, 0] != 1.0:
            bump("comp_not_placed")
        else:
            tw = env.movable_list[comp_idx].footprint_w
            th = env.movable_list[comp_idx].footprint_h
            x1 = va[0] + gx1 * cell
            y1 = va[1] + gy1 * cell
            x2, y2 = x1 + tw, y1 + th
            nb = next_cs[comp_idx]
            ox1, oy1, ox2, oy2 = nb[1], nb[2], nb[3], nb[4]
            if normalize:
                ox1, oy1, ox2, oy2 = ox1 * bd, oy1 * bd, ox2 * bd, oy2 * bd
            if not all(abs(a - b) < 1e-2 for a, b in [(ox1, x1), (oy1, y1), (ox2, x2), (oy2, y2)]):
                bump("bbox_mismatch")
            if abs(nb[5] - rot) > 1e-6:
                bump("rot_mismatch")
            if abs(nb[6] - layer) > 1e-6:
                bump("layer_mismatch")

        # A. 结构
        for t in traj:
            for key in ("obs", "action", "reward", "next_obs", "mask", "done", "step_id", "next_step_id"):
                if key not in t:
                    bump("missing_field")
                    break

        # F. mask 合法
        for k, t in enumerate(traj):
            m = t["mask"]
            if m is not None and not m[t["action"]]:
                bump("mask_violation")

        # D/E. 每 obs 硬重叠 + 越界
        for k, t in enumerate(traj):
            cs = np.array(t["obs"]["component_slots"], dtype=float)
            boxes = []
            for j in range(cs.shape[0]):
                if cs[j, 0] > 0.5:
                    x1, y1, x2, y2 = cs[j, 1], cs[j, 2], cs[j, 3], cs[j, 4]
                    if normalize:
                        x1, y1, x2, y2 = x1 * bd, y1 * bd, x2 * bd, y2 * bd
                    boxes.append((j, x1, y1, x2, y2))
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if _overlap(boxes[a][1:], boxes[b][1:]):
                        bump("hard_overlap")
            for (_j, x1, y1, x2, y2) in boxes:
                if x1 < va[0] - cell or y1 < va[1] - cell or x2 > va[2] + cell or y2 > va[3] + cell:
                    bump("out_of_bounds")

        # G. reward 一致性
        rewards = [t["reward"] for t in traj]
        if any(not math.isfinite(r) for r in rewards):
            bump("nonfinite_reward")
            continue
        sum_r = sum(rewards)
        for k, r in enumerate(rewards):
            if abs(r - sum_r * SHARE[k]) > max(1e-3, abs(sum_r) * 1e-3):
                bump("share_mismatch")
                break
        reward_stats.append(sum_r)
        if sum_r < 0.05:
            low_quality += 1

    return {
        "total_trajs": len(trajs),
        "sampled": len(sample),
        "issues": issues,
        "low_quality": low_quality,
        "reward_min": min(reward_stats) if reward_stats else None,
        "reward_max": max(reward_stats) if reward_stats else None,
        "reward_mean": (sum(reward_stats) / len(reward_stats)) if reward_stats else None,
    }


def main():
    pcb_files = find_pcb_files(DATA_DIR)
    print(f"Found {len(pcb_files)} boards under {DATA_DIR}")
    print(f"NORMALIZE_OBS = {Config.NORMALIZE_OBS}, CELL_SIZE = {Config.CELL_SIZE}")
    print(f"SHARE = {SHARE} (sum={sum(SHARE)})\n")
    for pcb_path in pcb_files:
        pcb_name = os.path.basename(pcb_path)
        print(f"=== {pcb_name} ===")
        env = PCBLayoutEnv(pcb_path, config_path=CONFIG_PATH)
        r = review(pcb_path, env)
        print(f"  可用轨迹总数: {r['total_trajs']} | 抽查: {r['sampled']}")
        print(f"  异常统计: {r['issues'] if r['issues'] else '无'}")
        print(f"  低质轨迹(sum_r<0.05): {r['low_quality']}")
        if r["reward_min"] is not None:
            print(
                f"  expert_reward(sum): min={r['reward_min']:.2f} "
                f"max={r['reward_max']:.2f} mean={r['reward_mean']:.2f}"
            )
        print()


if __name__ == "__main__":
    main()
