# -*- coding: utf-8 -*-
"""方法 A：统计专家轨迹的总 reward（每条完整轨迹的 8 步 reward 求和），
与 RL best_reward（~400+）对比，判断专家轨迹本身 reward 是否偏低。
"""
import json
import glob
import os
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "expert_transitions")
PCBS = ["Button", "Connector", "灯板2", "开关小板", "按键侧板"]


def load_for_pcb(keyword):
    pat = os.path.join(BASE, f"*{keyword}*expert.json")
    return glob.glob(pat)


def split_trajectories(transitions):
    """按 step_id 从 0 重新开始切分轨迹。"""
    # 按文件顺序遍历，step_id==0 或比前一个小则视为新轨迹起点
    trajs = []
    cur = []
    prev_sid = None
    for t in transitions:
        sid = t.get("step_id", None)
        if sid is None:
            continue
        if sid == 0 or (prev_sid is not None and sid <= prev_sid):
            if cur:
                trajs.append(cur)
            cur = [t]
        else:
            cur.append(t)
        prev_sid = sid
    if cur:
        trajs.append(cur)
    return trajs


def report(pcb_key, files):
    if not files:
        print(f"[{pcb_key}] 未找到专家轨迹文件")
        return
    fp = files[0]
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    trajs = split_trajectories(data)
    print("=" * 60)
    print(f"[{pcb_key}] 文件: {os.path.basename(fp)}")
    print(f"  transitions 总数: {len(data)}，切分轨迹数: {len(trajs)}")
    lens = [len(tr) for tr in trajs]
    print(f"  轨迹长度分布: min={min(lens)} max={max(lens)} "
          f"最常出现={max(set(lens), key=lens.count)}")
    # 每条轨迹 reward 求和（分两种口径：raw 求和 / 单条 reward 直接是局部）
    totals = []
    for tr in trajs:
        s = sum(t.get("reward", 0.0) for t in tr)
        totals.append(s)
    totals.sort()
    n = len(totals)
    if n == 0:
        return
    print(f"  轨迹总分: min={totals[0]:.2f} p10={totals[max(0,int(n*0.1))]:.2f} "
          f"p25={totals[max(0,int(n*0.25))]:.2f} median={totals[n//2]:.2f} "
          f"p75={totals[min(n-1,int(n*0.75))]:.2f} max={totals[-1]:.2f}  mean={sum(totals)/n:.2f}")
    # 对比基准
    print(f"  >>> 与 RL best（~400+）对比：专家轨迹平均总分 {sum(totals)/n:.2f} "
          f"≈ RL best 的 {sum(totals)/n/max(totals[-1],1)*100:.0f}% 量级")
    print()


for pcb in PCBS:
    report(pcb, load_for_pcb(pcb))
