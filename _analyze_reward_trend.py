# -*- coding: utf-8 -*-
"""分析 reward 走势：分段平均、best、负值占比，判断是否已收敛。"""
import json
import glob

p = glob.glob('result/*/reward_history.json')[0]
data = json.load(open(p, encoding='utf-8'))
rewards = [r for _, r in data]
print('总轮数:', len(rewards))
print('全量 mean=%.2f  best=%.2f' % (sum(rewards) / len(rewards), max(rewards)))
print()
for s in range(0, 10000, 1000):
    seg = rewards[s:s + 1000]
    print('ep %5d-%5d: mean=%8.2f  best=%8.2f' % (
        s + 1, s + 1000, sum(seg) / len(seg), max(seg)))
print()
last = rewards[-1000:]
neg = sum(1 for r in last if r < 0)
print('后1000轮: 负值占比 %.1f%%  均值 %.2f' % (neg / 10.0, sum(last) / len(last)))
print('最近500轮均值: %.2f' % (sum(rewards[-500:]) / 500))
