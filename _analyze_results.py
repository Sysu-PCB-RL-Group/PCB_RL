# -*- coding: utf-8 -*-
"""对照实验结果分析：读取 result/experiments/{A,D,E} 下所有 reward_history.json，
按 实验/板/组 归类计算指标，生成对比图 + 汇总表。"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'result/experiments'
OUT = os.path.join(ROOT, 'analysis')
os.makedirs(OUT, exist_ok=True)

BOARD_ALIASES = ['Connector', 'Button', '灯板2', '按键侧板', '开关小板']

def board_short(dirname):
    for k in BOARD_ALIASES:
        if k in dirname:
            return k
    return dirname

def load_reward(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return np.array([r for _, r in data], dtype=float)

def metrics(rew):
    n = len(rew)
    k = max(1, n // 5)
    return {
        'best': float(rew.max()),
        'last': float(rew[-1]),
        'mean_last20': float(rew[-k:].mean()),
        'mean_first20': float(rew[:k].mean()),
    }

def find_reward(exp, board, group):
    """在实验 exp 中找某板 group 的 reward_history.json。
    兼容实验 E cross（E/cross/<板>/reward_history.json）与 solo（E/<板>/solo/...）"""
    if exp == 'E' and group == 'cross':
        for rh in glob.glob(os.path.join(ROOT, 'E', 'cross', '**', 'reward_history.json'), recursive=True):
            if board in rh:
                return rh
        return None
    for rh in glob.glob(os.path.join(ROOT, exp, '**', group, '**', 'reward_history.json'), recursive=True):
        if board in rh:
            return rh
    return None

def collect(exp, groups):
    results = {}
    base = os.path.join(ROOT, exp)
    if not os.path.isdir(base):
        return results
    for board_dir in glob.glob(os.path.join(base, '*')):
        if not os.path.isdir(board_dir):
            continue
        board = board_short(os.path.basename(board_dir))
        for g in groups:
            rh = find_reward(exp, board, g)
            if rh:
                results.setdefault(board, {})[g] = metrics(load_reward(rh))
    return results

def plot_pair(exp, board, results, groups, colors, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for g in groups:
        if g not in results.get(board, {}):
            continue
        # 重新取原始曲线
        rh = find_reward(exp, board, g)
        if not rh:
            continue
        rew = load_reward(rh)
        win = min(50, max(2, len(rew)//10))
        if win > 1 and len(rew) > win:
            sm = np.convolve(rew, np.ones(win)/win, mode='valid')
            ax.plot(sm, label=g, color=colors.get(g), alpha=0.9)
        else:
            ax.plot(rew, label=g, color=colors.get(g), alpha=0.9)
    ax.set_title('实验%s - %s 组间对比' % (exp, board))
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)

def print_table(exp, groups, results):
    print('\n========== 实验 %s ==========' % exp)
    print('| 板子 | ' + ' | '.join('%s:Best/末20' % g for g in groups) + ' |')
    print('|' + '---|' * (len(groups) + 1))
    for board, grps in sorted(results.items()):
        cells = []
        for g in groups:
            if g in grps:
                m = grps[g]
                cells.append('%.1f/%.1f' % (m['best'], m['mean_last20']))
            else:
                cells.append('-')
        print('| %s | %s |' % (board, ' | '.join(cells)))

# ============ 实验 A: with_expert vs baseline ============
expA = collect('A', ['with_expert', 'baseline'])
print_table('A', ['with_expert', 'baseline'], expA)
for b in sorted(expA):
    plot_pair('A', b, expA, ['with_expert', 'baseline'],
              {'with_expert': '#d62728', 'baseline': '#1f77b4'}, 'A_%s.png' % b)

# ============ 实验 D: expert_nopen vs expert_pen ============
expD = collect('D', ['expert_nopen', 'expert_pen'])
print_table('D', ['expert_nopen', 'expert_pen'], expD)
for b in sorted(expD):
    plot_pair('D', b, expD, ['expert_nopen', 'expert_pen'],
              {'expert_nopen': '#2ca02c', 'expert_pen': '#ff7f0e'}, 'D_%s.png' % b)

# ============ 实验 E: cross vs solo ============
expE = collect('E', ['cross', 'solo'])
print_table('E', ['cross', 'solo'], expE)
for b in sorted(expE):
    plot_pair('E', b, expE, ['cross', 'solo'],
              {'cross': '#9467bd', 'solo': '#8c564b'}, 'E_%s.png' % b)

print('\n图表输出目录:', os.path.abspath(OUT))
