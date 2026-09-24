# 工作总结：PCB 元件布局 RL 训练改进与探索

<!-- STALE-BANNER -->
> ⚠️ **上一代文档（已作废，仅作历史记录/对照）**：本文描述的是重构前（≤2026-09-15）的架构 ——
> 如 `MAX_MOVABLE` / `GRID_W` / `GRID_H` / `CELL_SIZE=0.2`、7 元动作 `(comp_id, x1,y1,x2,y2, rot, layer)`、
> 违规罚分式 reward、`SPEC v2.x` 等。
> **现行口径一律以 [`../讨论结果/导入信息重构.md`](../讨论结果/导入信息重构.md) 为准**：动作 =（旋转 k, cell i, j）、L/G = 4×200×200、
> 违规被 mask **禁止**（不罚分）、固定顺序不选件、开槽随件（未放置不在板上）。

> 日期：2026-08-16
> 范围：DQN 训练稳定性改进、两次 10000ep 完整训练对比、后续优化方向分析

---

## 一、任务背景

PCB 元件自动布局 RL 任务：DQN agent 按「选元件 → 定坐标 → 定旋转」的 5 步子任务，自回归地逐件放置元件，目标是让所有元件在板框内无重叠、品质良好地完成布局。

改进前训练存在两大问题：

1. **Q 值过估计**：标准 DQN 对动作的选择与求值都在同一 target 网络，导致 Q 值偏高，难板训练后期出现灾难性崩盘。
2. **极端惩罚值主导训练**：单步重叠惩罚可低至 -240000，极端值主导 TD 目标，使 loss 爆炸。

---

## 二、已做的改进（算法层）

### C. Double DQN（`dqn/agent.py`）

- TD 目标从「target 网络同时选动作 + 求值」（过估计）改为：**online 网络选动作 → target 网络求值**。
- 新增共用掩码方法 `_apply_next_state_mask()`：online/target 两侧对称屏蔽非法动作（已放置元件、越界坐标）。
- 增加「全被屏蔽行重置为 0」的兜底逻辑，避免 `-inf` 在计算中传播。

### B. Reward Clip（`env/pcb_layout_env.py`）

- 新增 `REWARD_CLIP_MIN = -30` / `REWARD_CLIP_MAX = 60`，在 `_calculate_reward` 返回前截断。
- 设计思路：下限比上限紧——截断极端惩罚，同时保留「轻微违规 vs 严重违规」的区分度。

---

## 三、实验验证（两次 10000ep 完整训练）

- 改进前：`result1/`（8/15，改动前代码）
- 改进后：`result/`（8/16，Double DQN + Reward Clip）

### 3.1 客观指标 placed（放置数）对比 —— 衡量改进是否有效的核心指标

| 板子 | 改进前 last_placed | 改进后 last_placed | 变化 |
|---|---|---|---|
| Button (2件) | 0.50 | **0.94** | +88% |
| Connector (8件) | 2.57 | **3.50** | +36% |
| 灯板2 (6件) | 4.65 | **5.36** | +15% |
| 开关小板 (6件) | 5.88 | **5.97** | + |
| 按键侧板 (2件) | 2.00 | 2.00 | 持平 |

**结论：改进有效。** 所有板子放置数提升或持平，Button 板提升最明显。

### 3.2 训练稳定性对比（8000-10000ep 段）

| 指标 | 改进前 (8/15) | 改进后 (8/16) |
|---|---|---|
| 后期 loss | 300 ~ 739 | 40 ~ 90 |
| Button 后期 R | -480 ~ -960（灾难性崩盘） | -160（稳定在早停线） |
| Connector 后期 R | -201 ~ -305（崩盘） | -155 ~ -172（稳定） |
| 灯板2 后期 placed | 3 ~ 6（乱跳） | 2 ~ 6（仍波动） |
| 开关小板后期 | R=-110~+70 乱跳 | placed=6 稳定，R 收敛到 +8 |

**结论：灾难性崩盘已被消灭，但后期仍有「温和退化」**——灯板2 的 placed 从 5.66 降到 5.36、Connector 从 3.76 降到 3.50。

---

## 四、核心发现与探索

### 4.1 方法论：如何判断训练结果好坏（reward 与客观指标的双轨制）

**reward 的绝对数值本身没有意义**，它只是「教练的哨声」——agent 学的是「怎么选动作让累计 reward 最大」，而不是「reward 等于多少」。因此：

- reward 不能「想正就正想负就负」，它是在编码**任务目标的偏好排序**：放置元件(+) > 什么都不做(0) > 重叠/出界(-)。定义反了，agent 学到的东西就反了。
- **看训练好坏的正确姿势**：
  1. **reward 曲线**（训练信号）→ 看**趋势**：是否上升、是否收敛，不看绝对值；
  2. **客观指标**（真实目标）→ 放置数 `placed`、完成率、重叠次数、出界次数——这些**不受 reward 设计影响，跨版本可比**，是「考试成绩」。

> 实践印证：改进后 reward 数值变好有一部分是 clip 造成的「数值假象」，但 `placed` 的提升是实打实的策略改善。

### 4.2 result 目录结构说明

每次训练自动生成一套产物（非手动创建）：

```
result/
├── training_log.txt            训练日志（每 ep 的 reward/placed/ε/loss）
├── training_summary.json       汇总：各板 first/last_avg_R、best_R、placed 等
├── best_model.pt               全局最优模型权重（训练中自动保存）
├── bc_pretrained.pt            Phase 1 BC 预训练权重（Phase 2 起点）
├── bc_loss_history.json        BC 预训练 loss 数据
├── backup/                     历次训练的历史存档（各保留 5 份，带时间戳）
└── <板名>/                     每块板独立统计
    ├── loss_curve.png / reward_curve.png
    ├── loss_history.json / reward_history.json
    └── best_layout/best.json   该板最优一次的布局快照
```

### 4.3 辅助调试脚本

- `_compare_seeds.py`：固定 seed 的严格对比实验工具（验证改进是否可复现）。
- `_debug_early_stop.py`：早停逻辑调试工具。

---

## 五、当前不足（按严重程度排序）

### ① ε 探索率衰减参数错配（最核心问题）

`_compute_epsilon` 按 **per-board 计数**（`self.per_board_episode`），但 `EPSILON_DECAY_EPISODES=10000` 是按全局 ep 设计的。10000 ep 分给 5 块板，**每块板实际只有 ~2000 ep**。

```
t = self.per_board_episode.get(pcb_name, 0)
return eps_min + (eps_start - eps_min) * math.exp(-4.0 * t / Config.EPSILON_DECAY_EPISODES)
```

训练结束时 `t≈2000`，ε ≈ 0.1 + 0.5·e^(-4×2000/10000) ≈ **0.325**——即 10000 ep 跑完仍有 32.5% 概率随机乱走，**未收敛到 `EPSILON_MIN=0.1`**，这是后期波动的直接原因。

### ② Button 板（2 元件）始终学不会

best_R = -151.72（出现在 ep224，后期反而一路 -160 早停），而同为 2 件的按键侧板稳定满分。需对照专家轨迹排查任务可解性（板小元件大，网格可能放不下第 2 件）。

### ③ 每板训练量不足

5 板分 10000 ep，平均每板仅 2000 ep，对 8 件难板（Connector）不够，只能放 3.5/8。

### ④ target 网络更新太频繁

`TARGET_UPDATE_FREQ=20`，Double DQN 下 target 漂移快，Q 目标不稳。

---

## 六、下一步计划

### P0：参数三连（零代码风险，优先执行）

| 改动 | 当前 → 目标 | 预期效果 |
|---|---|---|
| `EPSILON_DECAY_EPISODES` | 10000 → **2500** | 训练结束时 ε≈0.12，策略真正收敛，消除后期波动 |
| `TOTAL_EPISODES` | 10000 → **20000** | 每板 4000 ep，难板有充分训练量 |
| `TARGET_UPDATE_FREQ` | 20 → **100** | 目标网络更稳，缓解后期退化 |

跑一次完整训练，验证「后期温和退化」是否根治。

### P1：Button 板专项排查

- 对照专家轨迹验证任务可解性（`data/expert_transitions/` 下有该板的专家数据）。
- 用 `_compare_seeds.py` 固定 seed 严格复现。

### P2（可选）：MSE → Huber loss

对 clip 后边缘样本更鲁棒（当前 MSE 对异常 TD error 敏感）。

---

## 七、附：奖励设计要点备忘

- `REWARD_CLIP_MIN = -30`，`REWARD_CLIP_MAX = 60`。
- 奖励是自定信号，跨版本不可直接比较绝对值；**跨版本比较一律用 placed / 完成率等客观指标**。
- `best_reward` 会被 `EARLY_STOP_THRESHOLD` 和 clip 共同钳制（难板不会超过 -150），看它没意义，要看平均和 placed。
