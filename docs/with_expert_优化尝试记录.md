# with_expert 训练优化尝试记录（专家轨迹 600→20000 起）

> 生成日期：2026-08-24
> 记录范围：从「专家轨迹 600→20000」扩充起，到当前定稿配置为止的全部优化尝试。
> 结构：每节统一「技术路线 / 实际效果 / 证据支持」三要素。
> 证据来源：记忆 `.codebuddy/memory/2026-08-17~23.md`；本会话（2026-08-23~24）
> 实测 T1–T5 贪心与 config 写回。

## 0. 背景与目标

**任务设定**：用强化学习（DQN 自回归）训练 AI 把 6 个电子元件摆到 PCB 板
（开关小板，FP=6 / Net=3 / grid 36×36）上。对比实验 A 组 `with_expert`
（注入专家轨迹）+ B 组 `baseline`（纯 RL）。

**起点崩坏（A 组，100k 训练，2026-08-22 分析）**：

- mean R = **-89.06**，median = -152.76，**75.8% 负 episode**
- placed mean = 4.92/6（59.5% 未放满）
- loss mean = 102.6（起始 ~25），**max = 193.89** → 明显发散
- NEW GLOBAL BEST 仅 31 次，峰值 R=161.39 @ ep56154（10 万次里唯一幸运 epi）；
  最终权重 ≠ best
- 完美布局 reward 上限 ≈ 158（baseline 收敛 157.62；纯 RL 探索到 163.77）

**总目标**：① 训练稳定（loss 有界、不发散）；② 贪心（确定性）能高效放满 6 个；
③ reward 逼近完美 ~158。

## 1. P0 伪专家诊断（2026-08-17）

**技术路线**：先跑通公平对比（修 random 种子、基线组短路、假成功吞异常、清旧产物），
再做 50000 epi A/B 对比，并直接查专家轨迹 JSON 找根因。

**实际效果**：

- baseline best=163.77（后20%均132.17）> with_expert best=162.75
  （后20%均83.31）→ baseline 反超。
- 查专家轨迹 JSON：**仅 600 transition / 20 轨迹**；reward 全正、违规率仅 0.8%，
  但整条轨迹总 reward mean=150.87、**max=155.74**（品质项 55.74）。

**证据支持**：

- 关键数据：KiCad 人工布局在自定义 reward 下**天花板 155.74 < 纯 RL 163.77**。
- 结论：「伪专家」问题——专家注入 = 把策略锚定在**次优**布局，A 组被拖累。
  这正是后续「扩充专家轨迹 + 质量修复」的动机。
- 顺带修复（非本次主线）：`run_inference.py` 原不加载 `argument.txt`
  致 W_COMPLETION 口径（200 vs 100）翻倍不可比；已加 `Config.load_from_file` 对齐。

## 2. P1 专家轨迹扩充 600→20000（2026-08-18）

**技术路线**：以 KiCad 完美布局为**唯一数据源**（否决 reward 导向生成）；
通过「拆解顺序随机化 ε=0.5 + 变异扩幅（带冲突回退）+ 几何平移（不越界）」
扩充专家轨迹，用**总数阈值**驱动生成直到达到 20000（buffer=5×20000=100000
对齐 baseline）。用户补充：变异须查硬重叠 + 板框越界。

**实际改动**：

- `config/argument.txt` / `env/pcb_layout_env.py`：删
  `EXPERT_TRAJECTORIES_PER_BOARD`，新增 `EXPERT_TOTAL_TRANSITIONS=20000`、
  `EXPERT_MAX_TRAJECTORIES=5000`，`EXPERT_EPSILON_DECOMPOSE` 0.2→0.5。
- `env/expert_trajectory.py`：`_mutate_component_state`（位置 ±1~2 格、旋转全 4 种、非法回退）、
  `_is_legal_placement`（越界 + AABB 硬重叠）、`_sample_legal_shift`、
  `decompose_backward` 加 `shift` 参数。

**附带修复（审查/排查发现）**：

1. `_sample_legal_shift` floor/ceil 反转 bug（原写法非整数坐标越界，mock 测不出）：

```python
# 修正后（env/expert_trajectory.py L692-697）
lo_x = int(np.ceil((va_xmin - min_x) / cell))   # 下界向上取整
hi_x = int(np.floor((va_xmax - max_x) / cell))  # 上界向下取整
lo_y = int(np.ceil((va_ymin - min_y) / cell))
hi_y = int(np.floor((va_ymax - max_y) / cell))
```

2. **y 坐标反向系统级 bug**（`utils/kicad_utils/kicad_utils.py` L250-259）：
   KiCad Y 轴向下致 `GetBottom()` y > `GetTop()`，bbox `y1>y2`
   → 碰撞/软间距检测全失效。修复为 `y1=GetTop()（小值）、y2=GetBottom()（大值）`，
   并同步 `get_keepout_zones`、可视化 `invert_yaxis`。

**证据支持**：

- 生成命令（用户终端）：

```powershell
$env:KICAD_PATH="E:\Kicad"; cd E:\PCB_AI\PCB_RL
python -c "from env.pcb_layout_env import PCBLayoutEnv; from env.expert_trajectory import generate_and_save_expert_transitions; import glob,os; f=glob.glob('data/kicad-native-batch/*开关小板*/*.kicad_pcb')[0]; env=PCBLayoutEnv(os.path.abspath(f)); tr,_=generate_and_save_expert_transitions(env, os.path.basename(f), output_dir='data/expert_transitions', total_transitions=20000, max_trajectories=5000, epsilon_decompose=0.5, pos_noise_prob=0.2, rot_noise_prob=0.1); print('total transitions:', len(tr)); env.close()"
```

- 真实生成 **20010 条**（667 轨迹×30），buffer = 5×20010 = **100050**。
- 变异代码片段（`_mutate_component_state` L178-208）：位置 20% 概率 ±1~2 格、
  旋转 10% 概率换 3 种之一，越界/硬重叠则回退原状态：

```python
# env/expert_trajectory.py L178-208（节选）
if random.random() < self.pos_noise_prob:
    direction = random.choice(["x", "y"])
    offset = random.choice([-2, -1, 1, 2]) * cell_size
    if direction == "x":
        bbox[0] += offset; bbox[2] += offset
    else:
        bbox[1] += offset; bbox[3] += offset
if random.random() < self.rot_noise_prob:
    rotation = random.choice([r for r in range(4) if r != rotation])
# 越界或硬重叠则回退到原始状态
if not self._is_legal_placement(mutated_bbox, comp_id, snapshot, env.valid_area, env=env):
    return {"bbox": tuple(comp_state["bbox"]), ...}  # 回退
```

- ⚠ 抽检 reward 结构变化（单次 mean 8.55 vs 旧 25.15）：变异产生大量软间距
  次优样本 → 引出 P2。

## 3. P2 数据健康化连环修复（2026-08-19 ~ 08-21）

**技术路线**：网格细化到 0.2mm → 修坐标边界聚集（shift 污染）→ 量化一致性
+ 全局合法性，目标把专家轨迹负样本率压到 0。

**3.1 网格细化（2026-08-19）**：`CELL_SIZE=0.2`（动作空间 57600），
删旧 1.0mm 专家 JSON 重生成 20010。

**3.2 shift 污染 bug（2026-08-20）**

- 现象：`data/expert_transitions/..._开关小板_expert.json`（20010 条）gx 分布异常：
  **gx=0 占 39.5%、gx=35 占 32.8%**（真实布局 gx∈1~28）；
  reward mean=1.808、负样本 5.3%、min=-12.5。
- 根因：`trajectory_to_transitions` 末尾把 env bbox 留在 shift 后状态不恢复；
  `generate_all_expert_transitions` 在 `decompose_backward`（内部 reset）**之前**
  调 `_sample_legal_shift`，读到**上一轮残留 shift 后 bbox**
  → 采样基准与应用基准不一致 → 越界 → 正反馈放大
  （小批量正常、大批量 32.8% 聚集，量变到质变）。
- 修复（`generate_all_expert_transitions` L728-739）：循环内 `_sample_legal_shift`
  前 `env.reset()`，保证基准始终是原始布局。
- 验证（`_verify_fix.py` / `generate_expert_only.py` 重生成）：gx=35 32.8%→**0%**；
  reward min -12.5→**+0.133（全正）**；负样本 5.3%→**0%**；
  `pytest tests/test_expert_trajectory.py` 9 passed。

**3.3 量化一致性 + 全局合法性（2026-08-21）**

- 根因：变异产生**连续坐标** target_bbox，env 状态用连续坐标、reward/next_obs
  用量化坐标 → ±0.1mm 在软间距阈值 0.2 边界翻转违规判定
  → 2.8%（555 条）坏样本（reward<20，min 1.15）→ 经专家区均匀采样注入矛盾信号
  → with_expert 崩坏（reward -51→-142、placed 5.6→3.6、loss 90+）。
- 深层根因：原 `_is_legal_placement` 只检查 `is_placed=True` 元件
  → 先放者变异侵入「稍后才放」元件空间时漏检
  （负 reward 挂在后放者身上，表面矛盾）。
- 修复：
  - `_quantize_bbox_to_grid`（L574-608）：env 状态 / actual_snapshot / reward /
    next_obs **四处统一量化坐标**，消除 ±0.1mm 边界翻转。
  - `_is_legal_placement`（L211-268）改为**全局检查**：遍历 snapshot 所有元件
    （含未放置者），仅跳过自身；**只拦硬重叠**（软间距仅轻微惩罚、不构
    `has_violations`），新增 keepout 检查；违规回退到 base 量化位置
    （完美布局碰撞自由，回退必通过）。
- 验证：`test_expert_trajectory.py` **15 passed**（含
  `test_generated_transitions_no_negative_reward`）；
  `generate_expert_only.py --board airlink` 重生成 20010 条，
  `_check_expert2.py` **负样本占比 0.0%**（reward mean=5.11/trans、min=1.908，全部为正）。
- 结论：专家轨迹数据地基健康，**with_expert 不再因坏样本崩坏**（数据层根因清除）。

```python
# _quantize_bbox_to_grid（env/expert_trajectory.py L596-608，统一量化）
gx1 = int(round((bbox[0] - env.valid_area[0]) / cell_size))
gy1 = int(round((bbox[1] - env.valid_area[1]) / cell_size))
gx1 = max(0, min(gx1, env.grid_width - 1))
gy1 = max(0, min(gy1, env.grid_height - 1))
x1 = env.valid_area[0] + gx1 * cell_size
y1 = env.valid_area[1] + gy1 * cell_size
w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
return (gx1, gy1, (x1, y1, x1 + w, y1 + h))
```

```python
# _is_legal_placement 全局检查（env/expert_trajectory.py L257-268，节选）
for other_id, other_state in snapshot.items():
    if other_id == comp_id:
        continue
    other_bbox = other_state["bbox"]
    threshold = Config.spacing_threshold_for(self_pkg, other_pkg)
    hard, soft = PCBUtils.bbox_overlap_penalty(bbox, other_bbox, threshold)
    if hard > 0.0:          # 只拦截硬重叠，软间距不在此拦
        return False
return True
```

## 4. P3 扩容轨迹占比 0.2→0.7（2026-08-22）

**技术路线**：假设「提高每 batch 专家采样占比」可修复崩坏。`EXPERT_BUFFER_RATIO`
0.2→0.7（语义 = **per-batch 采样占比**，非 buffer 切分；专家区恒常驻 20010、
普通区固定 80040=4×20010，改 ratio 只改采样混合比、无需重生成专家数据）。

**实际改动**：`config/argument.txt` L55 `0.2→0.7`；同步
`env/pcb_layout_env.py` L82 默认值、`README.md` L204。

**冒烟验证**（通过）：

```powershell
pytest tests/test_expert_ratio.py   # 3/3：Config 加载 0.7、sample(64) 拆 44 专家/20 自探索、buffer 容量 20010/80040
```

**实际效果（⚠ 无效，作为独立杠杆）**：20k/10k sanity（A 组 ratio=0.7）趋势**恶化**，
与 0.2 失败**同构**：

- mean R **-102.9** / %neg **76.9%** / placed 4.23 / loss mean 106.1
  **max 222.7**（0.2: 102.6/193.9）
- 前半 5000-ep meanR -89.0 → 后半 **-118.6**（退化非收敛）
- 仅 NEW GLOBAL BEST 更早到 155.38@ep5001，但只是「更快找到幸运布局」，非策略变好。
- 已 kill 止损。

**证据支持 / 结论**：提高专家占比**不是正确杠杆**——平均训练仍发散。
根因不在采样混合比，而在 RL 阶段**持续矛盾梯度**（80% 自探索 OOD 与 20% 专家梯度冲突；
gradient clipping 已在 `dqn/agent.py` L237/L307 且 norm=1.0，限制步长不限制方向，
挡不住持续矛盾）。但 ratio=0.7 成为后续稳定专家注入基底（沿用至今）。

## 5. P4 BC 值正则化 λv=10（止住 Q 发散，2026-08-22）

**技术路线**：根因重判 = RL 阶段 **Q 值发散**（正反馈：Q 漂→target 漂→Q 更漂；
PER/DDQN、专家占比、政策 KL 单独都压不住）。解法：加 **BC 值正则 λv=10**，
用 `MSE(Q_当前, Q_BC)` 把 Q 值幅度锚定到冻结 BC 网络量级。

**代码落地**（`dqn/agent.py`）：

```python
# L249：总损失 = TD 损失 + 策略 KL 锚定 + 值 L2 锚定
loss = loss + self.bc_reg_lambda * kl_reg + self.bc_reg_value_lambda * val_reg
```

```python
# L282-343 _bc_regularization（冻结 bc_net 由 init_bc_anchor 初始化，L272-280）
# 值 L2：MSE(Q_当前, Q_BC)，非法动作对齐为 0（差值=0，无 NaN）
cur_qv[~legal] = 0.0; bc_qv[~legal] = 0.0
vm = ((cur_qv - bc_qv) ** 2).mean()          # val_reg
# 策略 KL：KL(π_θ ‖ π_BC)，π=softmax(Q/τ)，τ=BC_REG_TEMP=1.0
cur_p = softmax(cur_qm / tau, dim=-1); bc_p = softmax(bc_qm / tau, dim=-1)
kl = (cur_p * (cur_p.clamp_min(1e-8).log() - bc_p.clamp_min(1e-8).log())).sum(dim=-1)  # kl_reg
```

**实际效果（✅ 有效，止血）**：

- val 全程稳定 **~3.7**（Q 被钉在 BC 量级，零增长），eff_val≈37 持续回拉。
- ep3500-3600（ε~0.22）：loss 稳定 **64-77**（OLD 同区曾炸 150-222）；
  Ep3500 R=-24.5/placed=6（全程最佳）。
- 10k 对比：loss **222→71**（有界）。本质≈稳定 BC 推理。
- 当时配置：`BC_REG_LAMBDA=100`（P5 后续下调）、`BC_REG_VALUE_LAMBDA=10`。

**证据支持**：窗口化滚动统计 + 10k 对比 training_log 解析
（`analyze_with_expert_log.py`）。遗留：强锚定≈冻结 BC，RL 难在「不发散」前提下
超越 BC（引出 P5 对 λkl 的再平衡）。

## 6. P5 诊断 T1 + λkl 调参循环（2026-08-23）

**诊断 T1（λkl=100，OLD FIXED best_model，ε=0.15 贪心）**：

- 证据：`[Ep 1] R=-17659.59  placed=6/6  steps=957`
- 机制：模型能放满 6 个（能力在），但**确定性贪心陷入重叠死循环**
  ——反复选同一冲突位置，obs 不变。
- 死循环合法性代码证据：

```python
# run_inference.py L209-212：确定性死锁注释
# 未放置元件 slot 全为零 → argmax 总选索引最小元件 → 若与已放置冲突 → obs 不变 → 无限循环
# run_inference.py L370-371：推理故意关闭早停
saved_early_stop = Config.EARLY_STOP_THRESHOLD
Config.EARLY_STOP_THRESHOLD = float("-inf")   # 让死循环跑到尽头以暴露问题
# env/pcb_layout_env.py L694-712 _calculate_reward：失败步惩罚项始终生效
reward = 0.0
reward -= Config.W_OVERLAP_HARD * total_hard_overlap   # W=20，始终扣
reward -= Config.W_OVERLAP_SOFT * total_soft_spacing   # W=10，始终扣
```

- 根因：`λkl=100` 把 BC 自身不完美（贪心也选重叠）**锁死**，RL 无法纠正
  → 死循环每步扣 ~-18，957 步 ≈ -17000 + 6 次放置 +100 ≈ -17659
  （真实且自洽，非程序 bug）。

**λkl 调参循环（λv=10 恒钉 Q）**：

| 实验 | λkl | 训练 best_R | 贪心(ε=0.15) | 结论 |
|---|---|---|---|---|
| T1 评 OLD FIXED | 100 | 157.63 | 6/6，**957 步**，R=-17659 | ❌ 无效→锁定根因 |
| **T3** | **10** | 152.77 | 6/6，**15 步**，R=**+77.8** | ✅ **贪心甜点** |
| T4 | 2 | 156.21 | 6/6，52 步，R=-486 | ❌ 贪心退化 |
| T5 | 5 | 156.68 | 6/6，28 步，R=-308 | ❌ 弱于 T3 |

**结论**：**λkl=10 是贪心部署策略最优配置**——loss 不发散 + 贪心高效放满 6 个
（15 步、R 转正）。更低 λkl(2/5) 训练峰值更高(156)但靠探索辅助撞出，
确定性贪心反而更差（steps 28-52、R 巨负）。

**证据支持**：`run_inference.py --inference-epsilon 0.15` 公平贪心评估；
临时 `tmp_analyze_exp.py` 分窗拉指标。

## 7. P6 定稿最佳配置 + 遗留差距

**定稿配置**（当前 `config/argument.txt`，2026-08-24 写回 `BC_REG_LAMBDA` 5.0→10.0）：

| 参数 | 值 | 为什么（每项依据） |
|---|---|---|
| `EXPERT_TOTAL_TRANSITIONS` | 20000 | P1：扩充专家轨迹，给足监督信号（非 600 伪专家） |
| `EXPERT_BUFFER_RATIO` | 0.7 | P3：稳定专家注入占比（单独不足，但为基底）；per-batch 采样占比 |
| `BC_REG_VALUE_LAMBDA` (λv) | 10.0 | P4：MSE(Q,Q_BC) 锚定 Q 幅度，止住 Q 发散（loss 222→71） |
| `BC_REG_LAMBDA` (λkl) | 10.0 | P5：策略锚定甜点，修好贪心死循环（15 步/+77.8），不锁死 BC 毛病 |
| `EPSILON_START/MIN/DECAY` | 0.6 / 0.1 / 10000 | 不变：ε 衰减到下限 0.1（10k≈0.11、5k≈0.17），保留探索余量 |
| `CELL_SIZE` | 0.2 | P2：细网格表达真实间距 |

**验证链小结**：P1/P2 → 数据健康（负样本 0.0%）；P3 → 专家占比 0.7 基底；
P4 → Q 不发散；P5 → 贪心策略正确。四者叠加 = 当前定稿。

**遗留差距**：

- T3 贪心 R=**+77.8** vs 完美 **157.63**：差距在**质量**（间距/聚合），不在「能否放满 6 个」。
- 候选杠杆（非 λkl）：
  1. **奖励塑形**增强 spacing/aggregation（W_NET_AGGREGATION=10 / W_CONGESTION=5
     过度奖励紧凑，human 布局考虑了布线/散热等未建模因素）；
  2. **重叠感知动作 mask**（env 屏蔽重叠位置，fail-closed），从源头杜绝死循环；
  3. **更长训练 / 更优 BC 初始化**。

**诚实标注（不确定点）**：

- ⚠ **λkl=10 vs 5 非铁证**：贪心对比每次仅 1 个 episode（ε=0.15），
  steps 差距（15 vs 28）较稳但 R 单样本噪声大；
  建议多 seed 各跑 5–10 次取均值再定稿。
- P3 的 20k sanity 已跑且显示**无效**（非「待归档」），本文已据实记录。
- 各尝试证据来源：P0–P4 来自记忆 `2026-08-17~22.md`；P5 来自本会话（2026-08-23）
  实测 T1–T5 与 `2026-08-23.md`；P6 config 写回来自我 2026-08-24 操作。
