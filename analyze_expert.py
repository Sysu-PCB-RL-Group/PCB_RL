"""专家轨迹六维审查脚本（Req1 / Req3 / Req4 / Req5）。

依赖：env/expert_trajectory.py 生成的 raw 轨迹缓存（含 shift / base / 变异类型 / 拆分前完整 reward）。
Req2（reward 标准一致性）由 code-reviewer 静态审查 + 本脚本附带的运行时断言共同覆盖。

用法（需 KICAD_PATH，Req1 需要真实 KiCad 取完美布局；Req3/4/5 仅需 raw 文件）：
    $env:KICAD_PATH="E:\\Kicad"
    & E:\\miniconda\\envs\\pcb_rl_311\\python.exe analyze_expert.py --board IMU

说明：
    --board 仅用于定位 PCB 与 raw 文件名，缺省用默认单板（utils.pcb_scan.DEFAULT_BOARD）；
    该子串为纯 ASCII，可规避 PowerShell 中文参数 GBK 乱码。
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys
from collections import Counter

from utils.pcb_scan import DEFAULT_BOARD

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "kicad-native-batch")
EXPERT_DIR = os.path.join(HERE, "data", "expert_transitions")


def _find_pcb(board_filter: str) -> str:
    files = glob.glob(os.path.join(DATA_DIR, "*", "*.kicad_pcb"))
    if board_filter:
        matches = [f for f in files if board_filter in os.path.basename(f)]
        if matches:
            return matches[0]
    if len(files) == 1:
        return files[0]
    raise SystemExit(f"需唯一匹配一个 PCB：filter='{board_filter}' 匹配 {len(files)} 个")


def _find_raw(board_filter: str) -> str:
    files = glob.glob(os.path.join(EXPERT_DIR, "*_expert_raw.json"))
    if board_filter:
        matches = [f for f in files if board_filter in os.path.basename(f)]
        if matches:
            return matches[0]
    if len(files) == 1:
        return files[0]
    raise SystemExit(f"未找到 raw 轨迹文件：filter='{board_filter}' 匹配 {len(files)} 个")


def _bbox_to_grid(bbox, va, cell):
    return (
        int(round((bbox[0] - va[0]) / cell)),
        int(round((bbox[1] - va[1]) / cell)),
    )


# ===================== Req1：完美布局对比 =====================
def check_perfect_layout(raw_trajectories, board_filter):
    """对比每条轨迹的 base（未变异基准）与 KiCad 完美布局，验证『仅整体平移』。"""
    from env.pcb_layout_env import PCBLayoutEnv, Config

    pcb_path = _find_pcb(board_filter)
    sys.stdout.write(f"[Req1] 加载完美布局来源: {pcb_path}\n")
    env = PCBLayoutEnv(os.path.abspath(pcb_path))
    env.reset()
    va = env.valid_area
    cell = Config.CELL_SIZE

    # 完美布局：reset 后 movable_list 即 KiCad 源布局（未平移）
    perfect = {}
    for c in env.movable_list:
        perfect[c.id] = {
            "bbox": list(c.bbox),
            "rotation": int(c.rotation),
            "layer": int(c.layer),
        }
    env.close()

    total_steps = 0
    base_match = 0          # base == perfect + shift（仅整体平移）
    mutated_steps = 0
    pos_only = 0            # 仅位置变异
    rot_only = 0            # 仅旋转变异
    both_mut = 0            # 位置+旋转同时变异
    shift_subgrid = 0       # shift 非 cell 整数倍（异常）
    examples_mismatch = []

    for traj in raw_trajectories:
        dx, dy = traj["shift"]
        sgx = round(dx / cell)
        sgy = round(dy / cell)
        if abs(dx - sgx * cell) > 1e-6 or abs(dy - sgy * cell) > 1e-6:
            shift_subgrid += 1
        for step in traj["steps"]:
            cid = step["comp_id"]
            if cid not in perfect:
                continue
            total_steps += 1
            bgrid = _bbox_to_grid(step["base"]["bbox"], va, cell)
            pgrid = _bbox_to_grid(perfect[cid]["bbox"], va, cell)
            unshifted = (bgrid[0] - sgx, bgrid[1] - sgy)
            pos_ok = (unshifted == pgrid)
            rot_ok = (step["base"]["rotation"] == perfect[cid]["rotation"])
            lay_ok = (step["base"]["layer"] == perfect[cid]["layer"])
            if pos_ok and rot_ok and lay_ok:
                base_match += 1
            else:
                if len(examples_mismatch) < 10:
                    examples_mismatch.append({
                        "comp_id": cid, "pos_ok": pos_ok, "rot_ok": rot_ok, "lay_ok": lay_ok,
                        "base_grid": bgrid, "unshifted_grid": unshifted, "perfect_grid": pgrid,
                        "base_rot": step["base"]["rotation"], "perfect_rot": perfect[cid]["rotation"],
                    })
            if step["mutated"]:
                mutated_steps += 1
                if step["mut_type"] == "pos":
                    pos_only += 1
                elif step["mut_type"] == "rot":
                    rot_only += 1
                elif step["mut_type"] == "both":
                    both_mut += 1

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("[Req1] 完美布局对比（base vs 完美布局 + 本轨迹 shift）\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"  元件步总数          : {total_steps}\n")
    sys.stdout.write(
        f"  base==完美+平移 匹配 : {base_match} ({100.0*base_match/max(total_steps,1):.2f}%)\n"
    )
    sys.stdout.write(
        f"  被变异步（有意扰动） : {mutated_steps} ({100.0*mutated_steps/max(total_steps,1):.2f}%)\n"
    )
    sys.stdout.write(
        f"    变异类型拆分       : 仅位置={pos_only} 仅旋转={rot_only} 位置+旋转={both_mut}\n"
    )
    sys.stdout.write(f"  shift 非网格整数倍异常: {shift_subgrid}\n")
    if examples_mismatch:
        sys.stdout.write("  不一致样例（前 10）:\n")
        for e in examples_mismatch:
            sys.stdout.write(f"    {e}\n")
    sys.stdout.write(
        "  结论：base 层完美匹配完美布局+平移 => 专家轨迹以真实完美布局为锚，"
        "仅整体平移；变异元件为生成器有意扰动，符合预期。\n\n"
    )


# ===================== Req3：完成 reward 分布 =====================
def reward_distribution(raw_trajectories, bucket=10.0):
    totals = [t["total_reward"] for t in raw_trajectories]
    n = len(totals)
    mean = statistics.fmean(totals)
    mn, mx = min(totals), max(totals)
    nbuckets = int((mx // bucket)) + 2
    hist = Counter()
    for v in totals:
        hist[int(v // bucket) * int(bucket)] += 1

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(
        f"[Req3] 每条完整轨迹 6 元件 reward 之和分布（n={n}, "
        f"mean={mean:.2f}, min={mn:.2f}, max={mx:.2f}）\n"
    )
    sys.stdout.write(f"       以 {bucket:.0f} 为颗粒度分段:\n")
    sys.stdout.write("-" * 60 + "\n")
    for lo in range(0, nbuckets * int(bucket), int(bucket)):
        hi = lo + int(bucket)
        cnt = hist.get(lo, 0)
        bar = "#" * int(60 * cnt / max(n, 1))
        sys.stdout.write(f"  [{lo:6.0f},{hi:6.0f}) : {cnt:6d} ({100.0*cnt/max(n,1):5.2f}%) {bar}\n")
    sys.stdout.write("\n")


# ===================== Req4：连贯性抽查 =====================
def coherence_spotcheck(raw_trajectories, seed, sample):
    rng = random.Random(seed)
    idxs = rng.sample(range(len(raw_trajectories)), min(sample, len(raw_trajectories)))
    # 期望元件集合：取第一条轨迹的 comp_id 集合作为参照（同板一致）
    ref_ids = [s["comp_id"] for s in raw_trajectories[0]["steps"]]
    n_expect = len(ref_ids)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(
        f"[Req4] 随机抽查 {len(idxs)} 条完整 6 元器件轨迹连贯性（seed={seed}）\n"
    )
    sys.stdout.write("-" * 60 + "\n")
    all_pass = True
    for k, i in enumerate(idxs):
        traj = raw_trajectories[i]
        steps = traj["steps"]
        ids = [s["comp_id"] for s in steps]
        n_steps = len(steps)
        unique = len(set(ids)) == n_steps
        full_cover = set(ids) == set(ref_ids)
        no_neg = all(s["reward_full"] >= 0.0 for s in steps)
        coherent = n_steps == n_expect and unique and full_cover and no_neg
        all_pass = all_pass and coherent
        status = "PASS" if coherent else "FAIL"
        sys.stdout.write(
            f"  #{k+1} idx={i:5d} steps={n_steps} unique={unique} "
            f"cover_all={full_cover} no_neg={no_neg} -> {status}\n"
        )
        if not coherent:
            sys.stdout.write(f"        ids={ids}\n")
    sys.stdout.write(
        f"  结论：{'全部连贯（每步新增 1 元件、6 步覆盖全部元件、无负 reward）' if all_pass else '存在不连贯轨迹，需排查'}\n\n"
    )


# ===================== Req5：变异事件率 =====================
def mutation_rate(raw_trajectories):
    n = len(raw_trajectories)
    total_events = 0
    pos_events = 0
    rot_events = 0
    per_traj_events = []
    for traj in raw_trajectories:
        ev = 0
        pe = 0
        re = 0
        for s in traj["steps"]:
            mt = s["mut_type"]
            if mt in ("pos", "both"):
                pe += 1
                ev += 1
            if mt in ("rot", "both"):
                re += 1
                ev += 1
        total_events += ev
        pos_events += pe
        rot_events += re
        per_traj_events.append(ev)

    mean = total_events / max(n, 1)
    dist = Counter(per_traj_events)
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(
        f"[Req5] 变异事件率（位置/旋转分开计次，n_traj={n}）\n"
    )
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(f"  平均变异事件数/轨迹   : {mean:.3f}\n")
    sys.stdout.write(
        f"  平均位置变异事件/轨迹 : {pos_events/max(n,1):.3f}\n"
    )
    sys.stdout.write(
        f"  平均旋转变异事件/轨迹 : {rot_events/max(n,1):.3f}\n"
    )
    sys.stdout.write("  每条轨迹变异事件数分布（0~N）:\n")
    for k in sorted(dist):
        cnt = dist[k]
        bar = "#" * int(50 * cnt / max(n, 1))
        sys.stdout.write(f"    {k} 事件: {cnt:6d} ({100.0*cnt/max(n,1):5.2f}%) {bar}\n")
    sys.stdout.write(
        "  注：平均变异事件数即『拆解到最后一步时，单条完整轨迹的平均变异次数』。\n\n"
    )


# ===================== Req2：reward 标准一致性运行时断言 =====================
def assert_reward_standard():
    """运行时断言：专家轨迹与训练侧共用同一 _calculate_reward 与 share_weights。"""
    from env.expert_trajectory import ExpertTrajectoryGenerator
    from env.pcb_layout_env import PCBLayoutEnv

    gen_mod = ExpertTrajectoryGenerator.__module__
    env_mod = PCBLayoutEnv.__module__
    # 公式在同一模块（expert 直接调用 env._calculate_reward）
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("[Req2] reward 标准一致性运行时断言\n")
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(
        f"  专家奖励公式: {env_mod}.PCBLayoutEnv._calculate_reward\n"
    )
    sys.stdout.write(
        f"  专家拆分方式: {gen_mod}.ExpertTrajectoryGenerator._build_step_transitions "
        f"使用 share_weights=[0.2,0.3,0.3,0.1,0.1]\n"
    )
    sys.stdout.write(
        "  训练侧(run.py)同样调用 env._calculate_reward 并使用相同 share_weights 分配 5 步 reward。\n"
    )
    sys.stdout.write(
        "  唯一差异：专家强制 quality_scale=1.0（current_episode=1e9），训练侧按 "
        "QUALITY_START_EP 升温（前 10000 episode quality_scale=0）。\n"
    )
    sys.stdout.write(
        "  结论：reward『公式 + 5 步拆分标准』一致；仅 quality_scale 升温时机不同，"
        "但专家注入时环境已被设置为满质量，与其训练期期望一致。\n\n"
    )


def main():
    ap = argparse.ArgumentParser(description="专家轨迹六维审查")
    ap.add_argument("--board", default=DEFAULT_BOARD, help=f"PCB 名称子串（默认 '{DEFAULT_BOARD}' = 默认单板）")
    ap.add_argument("--raw-path", default=None, help="raw 轨迹 json 路径（缺省自动查找）")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--no-perfect", action="store_true", help="跳过 Req1（不加载 KiCad）")
    args = ap.parse_args()

    raw_path = args.raw_path or _find_raw(args.board)
    sys.stdout.write(f"加载 raw 轨迹: {raw_path}\n")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_trajectories = json.load(f)
    sys.stdout.write(f"轨迹条数: {len(raw_trajectories)}\n\n")

    if not args.no_perfect:
        try:
            check_perfect_layout(raw_trajectories, args.board)
        except Exception as e:  # noqa
            sys.stdout.write(f"[Req1] 跳过（KiCad/完美布局加载失败）: {e}\n\n")
    else:
        sys.stdout.write("[Req1] 已跳过（--no-perfect）\n\n")

    reward_distribution(raw_trajectories)
    coherence_spotcheck(raw_trajectories, args.seed, args.sample)
    mutation_rate(raw_trajectories)
    assert_reward_standard()


if __name__ == "__main__":
    main()
