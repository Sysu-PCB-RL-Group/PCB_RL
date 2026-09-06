"""诊断：灯板2 / Button 专家轨迹的几何合法性与 reward 拆解（权威复检）。

复用 env._detect_collisions / _calculate_reward，与生成 / 训练 100% 同口径
（带 spacing_threshold 的硬重叠 + 软间距 + 出界 + flat 失败成本），并以此
对比上一轮 review_expert_trajectories.py 的“无阈值 AABB”判定，给出假阳性率。

覆盖：
  - 权威几何复检：每 chunk 重建 env 状态，对“刚放置元件”调 env._detect_collisions
    得真实 hard/soft/oob；同时统计整局 pairwise 硬重叠对数（与上一轮 AABB 对比）。
  - reward 拆解：用 env._calculate_reward(quality_scale=1.0) 重算每 chunk 的
    expert_reward，验证与缓存 sum(rewards) 一致，并拆硬/软/出界/flat 罚分定位主导项。
  - 源布局交叉核对：读 _expert_raw.json 的 target bbox，统计元件最小间隙与边缘距板框，
    判断源布局固有 sub-grid 间隙 / 边缘元件（量化后必合并 / 出界）。

用法（用户终端，需 KiCad DLL）：
  $env:KICAD_PATH="E:\Kicad"
  & E:\miniconda\envs\pcb_rl_311\python.exe scripts/diagnose_expert_boards.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.pcb_layout_env import PCBLayoutEnv, Config  # noqa: E402
from env.expert_trajectory import EXPERT_SHARE_WEIGHTS  # noqa: E402
from env.pcb_utils import PCBUtils  # noqa: E402
from utils.pcb_scan import find_pcb_files  # noqa: E402

EXPERT_DIR = os.path.join(ROOT, "data", "expert_transitions")
DATA_DIR = os.path.join(ROOT, "data", "kicad-native-batch")
CONFIG_PATH = os.path.join(ROOT, "config", "argument.txt")

PER_BOARD = 50
# 目标板 + 健康基线（开关小板）
TARGET_SUBSTR = ["灯板2", "Button", "开关小板"]


def load_chunks(pcb_short):
    fp = os.path.join(EXPERT_DIR, f"{pcb_short}_expert.json")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    i = 0
    n = len(data)
    while i + 5 <= n:
        c = data[i : i + 5]
        if [t["step_id"] for t in c] == [0, 1, 2, 3, 4]:
            chunks.append(c)
            i += 5
        else:
            i += 1
    return chunks


def reconstruct_env(env, cs):
    """按 obs component_slots (placed,x1,y1,x2,y2,rot,layer,net) 重建 env 已放置状态。

    约定 slot j == env.movable_list[j]（与 review_expert_trajectories 一致）。
    NORMALIZE_OBS=False → 原始 mm，直接写入。
    """
    placed_ids = []
    for j in range(cs.shape[0]):
        if cs[j, 0] > 0.5:
            x1, y1, x2, y2 = cs[j, 1:5]
            comp = env.movable_list[j]
            comp.bbox = (float(x1), float(y1), float(x2), float(y2))
            comp.rotation = int(cs[j, 5])
            comp.layer = int(cs[j, 6])
            comp.is_placed = True
            placed_ids.append(comp.id)
    env.placed_list = placed_ids
    return placed_ids


def pairwise_hard_pairs(env):
    """统计 env 当前已放置 movable 元件中，同层且几何硬重叠（hard>0）的配对对数。

    用与 env._detect_collisions 同一套 bbox_overlap_penalty + spacing_threshold。
    """
    placed = [env.kicad.components[rid] for rid in env.placed_list]
    pairs = 0
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            ca, cb = placed[a], placed[b]
            if ca.layer != cb.layer:
                continue
            thr = Config.spacing_threshold_for(ca.package_type, cb.package_type)
            hard, _ = PCBUtils.bbox_overlap_penalty(ca.bbox, cb.bbox, thr)
            if hard > 0.0:
                pairs += 1
    return pairs


def diagnose_board(env, chunks, pcb_short):
    sample = chunks[:PER_BOARD]
    n = len(sample)

    n_chunk_hard_pair = 0      # 整局含 >=1 硬重叠配对的 chunk 数（对比上一轮 AABB 的 50）
    n_chunk_oob = 0            # 整局含 >=1 出界元件的 chunk 数（对比上一轮 AABB 的 95）
    n_place_hard = 0           # 刚放置元件 hard>0 的 chunk 数
    n_place_soft = 0
    n_place_oob = 0
    n_viol = 0                 # has_violations=True 的 chunk 数

    hard_area = soft_area = oob_area = 0.0
    p_hard = p_soft = p_oob = p_flat = 0.0
    cache_mismatch = 0
    rewards = []
    extreme = None  # (reward, comp_idx, (hard,soft,oob,keepout), per_term)

    for chunk in sample:
        comp_idx = int(chunk[0]["action"])
        cs_next = np.array(chunk[4]["next_obs"]["component_slots"], dtype=float)
        reconstruct_env(env, cs_next)

        # 整局 pairwise 硬重叠（权威，带阈值）
        hp = pairwise_hard_pairs(env)
        if hp > 0:
            n_chunk_hard_pair += 1

        # 整局出界：逐元件检测
        chunk_oob = False
        for rid in env.placed_list:
            comp = env.kicad.components[rid]
            if PCBUtils.out_of_bounds_area(comp.bbox, env.valid_area) > 0.0:
                chunk_oob = True
                break
        if chunk_oob:
            n_chunk_oob += 1

        # 刚放置元件的权威碰撞 + reward：用 chunk 内存储的 expert_actions
        # (gx1,gy1,rot,layer) 逐字节复刻 _compute_expert_reward 的输入重建，
        # 与生成 100% 同口径（避免读 obs bbox 的重建歧义）。
        # 生成时 comp.is_placed=False 且未入 placed_list（net_aggregation 靠
        #  _calculate_reward L747 手动 append），故此处保持同口径。
        gx1 = int(chunk[1]["action"])
        gy1 = int(chunk[2]["action"])
        rot = int(chunk[3]["action"])
        layer = int(chunk[4]["action"])
        comp = env.movable_list[comp_idx]
        comp.is_placed = False
        if comp.id in env.placed_list:
            env.placed_list.remove(comp.id)
        cell = Config.CELL_SIZE
        tw, th = comp.footprint_w, comp.footprint_h  # 与合并后 expert/env 同口径（footprint 基准）
        x1 = env.valid_area[0] + gx1 * cell
        y1 = env.valid_area[1] + gy1 * cell
        comp.bbox = (x1, y1, x1 + tw, y1 + th)
        comp.rotation = rot
        comp.layer = layer
        hard, soft, oob, keepout = env._detect_collisions(comp)
        # S3=off：软间距不构成违规（与 env.step / expert _compute_expert_reward 同口径）
        has_viol = hard > 0.0 or oob > 0.0 or keepout

        # 权威 reward（与生成 quality_scale=1.0 一致）
        expert_reward = env._calculate_reward(
            comp, hard, soft, oob, has_viol, quality_scale=1.0
        )
        cached_sum = sum(float(t["reward"]) for t in chunk)
        if abs(expert_reward - cached_sum) > max(1e-2, abs(cached_sum) * 1e-2):
            cache_mismatch += 1

        # 拆解罚分
        th = Config.W_OVERLAP_HARD * hard
        ts = Config.W_OVERLAP_SOFT * soft
        to = env.W_OUT_OF_BOUNDS_CURRENT * oob
        tf = 0.0  # S3=off：方案B 的 flat 已废除

        rewards.append(expert_reward)
        hard_area += hard
        soft_area += soft
        oob_area += oob
        p_hard += th
        p_soft += ts
        p_oob += to
        p_flat += tf

        if hard > 0.0:
            n_place_hard += 1
        if soft > 0.0:
            n_place_soft += 1
        if oob > 0.0:
            n_chunk_oob  # noqa (already counted at chunk level)
        if oob > 0.0:
            n_place_oob += 1
        if has_viol:
            n_viol += 1

        if extreme is None or expert_reward < extreme[0]:
            extreme = (expert_reward, comp_idx, (hard, soft, oob, keepout), (th, ts, to, tf))

    rewards_arr = np.array(rewards)
    return {
        "pcb_short": pcb_short,
        "n_sampled": n,
        "chunk_hard_pair": n_chunk_hard_pair,   # 对比上一轮 AABB 硬重叠=50
        "chunk_oob": n_chunk_oob,               # 对比上一轮 AABB 越界=95
        "place_hard": n_place_hard,
        "place_soft": n_place_soft,
        "place_oob": n_place_oob,
        "viol": n_viol,
        "hard_area": hard_area,
        "soft_area": soft_area,
        "oob_area": oob_area,
        "p_hard": p_hard,
        "p_soft": p_soft,
        "p_oob": p_oob,
        "p_flat": p_flat,
        "cache_mismatch": cache_mismatch,
        "reward_min": float(rewards_arr.min()) if n else None,
        "reward_max": float(rewards_arr.max()) if n else None,
        "reward_mean": float(rewards_arr.mean()) if n else None,
        "reward_median": float(np.median(rewards_arr)) if n else None,
        "extreme": extreme,
    }


def crosscheck_source(env, pcb_short):
    """读 _expert_raw.json 的 target bbox，统计源布局固有 sub-grid 间隙 / 边缘出界。"""
    fp = os.path.join(EXPERT_DIR, f"{pcb_short}_expert_raw.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        raw = json.load(f)
    va = env.valid_area
    n_traj = len(raw[:PER_BOARD])
    traj_with_subgrid = 0     # 存在同层元件最小间隙 < CELL_SIZE 的轨迹数
    traj_with_edge = 0        # 存在 target bbox 出界（量化后必出界）的轨迹数
    gap_min_global = float("inf")
    edge_min_global = float("inf")

    for traj in raw[:PER_BOARD]:
        steps = traj.get("steps", [])
        boxes = []
        for s in steps:
            t = s["target"]
            boxes.append((t["bbox"], t["layer"], t.get("rotation", 0)))
        # 同层最小间隙
        gmin = float("inf")
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ba, la, _ = boxes[a]
                bb, lb, _ = boxes[b]
                if la != lb:
                    continue
                dx = min(ba[2], bb[2]) - max(ba[0], bb[0])
                dy = min(ba[3], bb[3]) - max(ba[1], bb[1])
                if dx > 0 and dy > 0:
                    gap = 0.0  # 已硬重叠
                elif dx > 0 >= dy:
                    gap = -dy
                elif dy > 0 >= dx:
                    gap = -dx
                else:
                    gap = float("inf")  # 角对角
                gmin = min(gmin, gap)
        # 边缘距板框（target 出界 = 量化后出界的前兆）
        emin = float("inf")
        for (bb, _, _) in boxes:
            ox = max(0.0, bb[0] - va[0]) + max(0.0, va[2] - bb[2])
            oy = max(0.0, bb[1] - va[1]) + max(0.0, va[3] - bb[3])
            emin = min(emin, min(ox, oy))
        gap_min_global = min(gap_min_global, gmin)
        edge_min_global = min(edge_min_global, emin)
        if gmin < Config.CELL_SIZE:
            traj_with_subgrid += 1
        if emin < 0.0:
            traj_with_edge += 1

    return {
        "n_traj": n_traj,
        "traj_with_subgrid_gap": traj_with_subgrid,
        "traj_with_edge_oob": traj_with_edge,
        "gap_min_global": (gap_min_global if gap_min_global != float("inf") else None),
        "edge_min_global": (edge_min_global if edge_min_global != float("inf") else None),
        "cell": Config.CELL_SIZE,
    }


def main():
    pcb_files = find_pcb_files(DATA_DIR)
    print(f"Found {len(pcb_files)} boards under {DATA_DIR}")
    print(f"NORMALIZE_OBS={Config.NORMALIZE_OBS} CELL_SIZE={Config.CELL_SIZE} "
          f"W_OVERLAP_HARD={Config.W_OVERLAP_HARD} W_OVERLAP_SOFT={Config.W_OVERLAP_SOFT} "
          f"W_OUT_OF_BOUNDS_INITIAL={Config.W_OUT_OF_BOUNDS_INITIAL}\n")

    for pcb_path in pcb_files:
        pcb_name = os.path.basename(pcb_path)
        if not any(sub in pcb_name for sub in TARGET_SUBSTR):
            continue
        pcb_short = pcb_name.replace(".kicad_pcb", "")
        env = PCBLayoutEnv(pcb_path, config_path=CONFIG_PATH)
        env.reset()  # 填充 decoupling_map / area_ranks / board_diagonal 等（_calculate_reward 需要）
        chunks = load_chunks(pcb_short)
        print(f"=== {pcb_name} ===  (cached chunks={len(chunks)}, sampled={min(PER_BOARD, len(chunks))})")

        d = diagnose_board(env, chunks, pcb_short)
        print(f"  [权威几何/上一轮AABB对比]")
        print(f"    chunk含硬重叠配对: {d['chunk_hard_pair']}/{d['n_sampled']}  "
              f"(上一轮AABB硬重叠=50 -> 假阳性率~(1-18/50) 见正文)")
        print(f"    chunk含出界元件:   {d['chunk_oob']}/{d['n_sampled']}  (上一轮AABB越界=95)")
        print(f"    刚放置元件 hard>0: {d['place_hard']}  soft>0: {d['place_soft']}  oob>0: {d['place_oob']}")
        print(f"    has_violations chunk 数: {d['viol']}/{d['n_sampled']}")
        print(f"  [reward 拆解]")
        print(f"    缓存一致(mismatch): {d['cache_mismatch']}/{d['n_sampled']}")
        print(f"    expert_reward min={d['reward_min']:.2f} median={d['reward_median']:.2f} "
              f"mean={d['reward_mean']:.2f} max={d['reward_max']:.2f}")
        print(f"    罚分累计: hard={d['p_hard']:.1f}(面积{d['hard_area']:.2f}mm^2) "
              f"soft={d['p_soft']:.1f}({d['soft_area']:.2f}mm^2) "
              f"oob={d['p_oob']:.1f}({d['oob_area']:.2f}mm^2) flat={d['p_flat']:.1f}")
        if d["extreme"] is not None:
            r, ci, (h, s, o, k), (th, ts, to, tf) = d["extreme"]
            print(f"    极端最低 reward chunk: comp_idx={ci} reward={r:.2f}")
            print(f"      hard={h:.3f}->罚{th:.2f}  soft={s:.3f}->罚{ts:.2f}  "
                  f"oob={o:.3f}->罚{to:.2f}  flat={tf:.2f}  keepout={k}")
        x = crosscheck_source(env, pcb_short)
        if x is not None:
            print(f"  [源布局交叉核对]")
            print(f"    轨迹含 sub-grid(<{x['cell']}mm)间隙: {x['traj_with_subgrid_gap']}/{x['n_traj']}")
            print(f"    轨迹含 target 出界: {x['traj_with_edge_oob']}/{x['n_traj']}")
            print(f"    全局最小同层间隙={x['gap_min_global']}mm  全局最小边缘余量={x['edge_min_global']}mm")
        print()


if __name__ == "__main__":
    main()
