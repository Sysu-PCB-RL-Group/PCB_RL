# PCB 布局核心约束列表（最终版）

> **用途**：RL 环境建模直接使用  
> **筛选标准**：普遍性 + 几何可判 + 阈值明确 + 直接影响合法性/可行性  
> **日期**：2026-07-23（整合版）  
> **配套文档**：`PCB_constraints_survey_final.md`（80 条全部约束）

---

## 约束分级说明

| 级别 | 标识 | 含义 | 数量 |
|------|------|------|:--:|
| **一级核心** | C0-C6 | 硬合法性约束 — 违反则 PCB 无法制造，必须实现 | 7 |
| **二级核心** | S1-S3 | 强烈建议的软约束 — 不实现不影响合法性，但显著影响布局质量和训练效率 | 3 |

---

## 约束总览

| ID | 级别 | 类别 | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 对 Doc C 对应 |
|----|:--:|------|---------|---------|:--:|---------|:--:|
| C0 | 一级 | 环境机制 | Reset 初始位置为板框外 | 所有元件初始坐标在 board_bounds 外部 | hard | `x < x_min or x > x_max or y < y_min or y > y_max` | — |
| C1 | 一级 | 环境机制 | 锁定已放置元件 | 已放置元件不可再次被选为 action 目标 | hard | `component_id ∉ placed_list` | 3.2 动作掩码 |
| C2 | 一级 | 制造 | 元件不重叠 | 任意两元件边缘间距 ≥ 0.25mm | hard | AABB 包围盒交集面积 = 0（含 0.25mm 外扩） | 1.1 重叠约束 |
| C3 | 一级 | 制造 | 元件在板框内 | 元件 AABB 四角均在板框 1mm 内缩区内 | hard | `max(x±w/2) ≤ x_max-1mm, min(x±w/2) ≥ x_min+1mm` | 1.3 出界约束 |
| C4 | — | — | （已合并至 C3） | C3 的内缩区检查同时覆盖出界和板边距 | — | — | 1.2 间距约束 |
| C5 | 一级 | 制造 | 固定元件不可移动 | 预标记的固定元件（连接器、安装孔等）不可被 agent 操作 | hard | `component_id ∉ movable_list` + KiCad `IsLocked()` | 1.4 禁止区域 |
| C6 | 一级 | 电气 | 去耦电容距离 | 电容中心到目标 IC 的 VCC 引脚 ≤ 5mm | hard | `sqrt(dx² + dy²) ≤ 5mm`（AABB 包围盒 → 引脚距） | 1.5 热/功耗域 |
| S1 | 二级 | 布线代理 | 拥塞密度控制 | 2mm×2mm 网格容量 = 8 根走线，超过按平方惩罚 | soft | MST + L-shape 估计走线 → 网格累计 → 超额² | 2.1 拥塞密度 |
| S2 | 二级 | 优化目标 | HPWL 线长最小化 | `Σ_net[(x_max-x_min)+(y_max-y_min)]`，替换紧凑度 | soft | 所有网络 bounding box 半周长之和 | 2.3 HPWL |
| S3 | 二级 | 流程优化 | 放置优先级排序 | 按连接度/面积降序，固定元件优先 | hard | `sorted(refs, key=conn_score, reverse=True)` | 3.1 放置顺序 |

---

## 一级核心约束（硬合法性）

### C0：Reset 初始位置为板框外

**含义**：RL 环境 reset 时，所有可移动元件的初始坐标应设置在板框外部，模拟"空板开始"的逐步布局过程。

**建模动机**：现有代码在 reset 时随机放置元件于板内，违反布局的顺序逻辑——正确的 RL 建模应该是 agent 从空板出发，每次 action 将一颗元件移入板内并放置。

**判别条件**：在 reset 时，对每个 movable 元件，设置坐标 `(x, y)` 使得 `x < board_x_min - 50mm`（或其他明确在板外的固定位置）。

**来源**：环境建模约束（非文献约束），基于 RL 顺序决策问题的标准范式。

---

### C1：锁定已放置元件

**含义**：agent 每步 action 只能移动一颗尚未放置的元件。已放置元件不允许再次选中。

**建模动机**：PCB 布局本质上是 sequential assignment 问题——每步将一个元件分配到一个最终位置。允许重复移动已放置元件会导致搜索空间爆炸。

**判别条件**：维护 `placed_list`，step 时检查 `action.component_id` 是否已在该列表中。

**来源**：环境建模约束。OmniLayout (2026): "a predicted placement assigns each component vi ∈ V a unique position (xi, yi)" 隐含了"每个元件最终只有一个位置"。

---

### C2：元件不重叠（增强版）

**含义**：任意两个元件的封装轮廓不能重叠。这是最根本的物理约束——两个实体不能同时占据同一空间。

**量化阈值**：IPC-7351 Density Level B (Nominal)：元件 courtyard 间距 ≥ 0.25mm（约 10 mils）。JLCPCB 推荐 0.25mm。

**增强要点（源自 Doc C 1.1）**：

原始实现使用中心点距离 + 0.2mm 硬阈值，存在以下问题：
1. 中心点距离不反映包围盒交叉——大芯片角可能已压住小电容
2. 二值判断无梯度——agent 收不到"你在变好"的信号
3. 间距 0.19mm 和 0.21mm 在实践中无差别，但 reward 剧烈波动

**改进方案：AABB 包围盒 + 软硬分层连续惩罚**

```
判别逻辑升级为：

对于任意两个元件 i, j (i < j)：
  获取 AABB 包围盒 R_i, R_j（含 0.25mm 外扩）
  计算两包围盒的交叉面积
  ┌─ 交叉面积 > 0 → 硬重叠：惩罚 = 交叉面积 × 权重 1000
  ├─ 无交叉但间隙 < 2mm → 软间距：惩罚 = (2.0-gap)² × edge_length × 权重 50
  └─ 间隙 ≥ 2mm（角对角）→ 无惩罚
```

**参考实现**：pyplacer `bbox_overlap_penalty()`（placer.py:59-83），约 30 行可移植代码。详见配套全部约束文档附录 B.1.1。

**来源**：
- IPC-7351B, Section on Courtyard Density Levels
- OmniLayout (2026): "Cp2p(x,y) penalizes pad-to-pad clearance violations"
- pyplacer 开源实现

---

### C3：元件在板框内 + 板边距（增强版，合并原 C3+C4）

**含义**：元件必须完全位于板框有效区域内（含制造安全裕度）。原 C3（不出界）和 C4（板边距）统一为一条以 AABB 包围盒为基���的约束。

**量化阈值**：元件 AABB 四角均需在板框 1mm 内缩区内。1mm 是基于 IPC-2221A（0.5mm 本体到板边）与多家制造商推荐（1mm）之间的折中选择。

**增强要点（源自 Doc C 1.3）**：

原始实现使用中心点二值判断，问题：
1. 中心点在板内但元件角已出界
2. 出界 0.1mm 和 5mm 惩罚相同
3. 矩形近似无法处理异形板框

**改进方案：连续出界面积 + 训练/推理分层处理**

```
判别逻辑升级为：

设元件 AABB 为 [(x-w/2, y-h/2), (x+w/2, y+h/2)]
设板框内缩 1mm 后的有效区域 valid_area

出界面积 = AABB总面积 - AABB与valid_area交叉面积

训练时：
  若出界面积 > 0 → 惩罚 = 出界面积 × 权重 200
  若元件中心完全在板外 → 巨额惩罚 + 终止 episode

推理时：
  若坐标超出 valid_area → clamp 至最近合法位置（安全兜底）
```

**参考实现**：
- pyplacer `out_of_bounds_area()`（placer.py:86-95）：连续面积惩罚
- RL_PCB：出界 = 剩余步数×15 惩罚 + done=True
- kicad-mcp：推理阶段 clamp 兜底

详见配套全部约束文档附录 B.1.3。

**来源**：
- OmniLayout (2026): "out-of-boundary ratio, measuring whether the placement is physically valid within board boundary"
- IPC-2221A §5.2: Component-to-Board Edge Clearance

---

### C5：固定元件不可移动

**含义**：一部分元件的位置由机械结构（外壳、安装孔）或接口标准（USB 位置、天线位置）锁定，不可被 agent 改变。

**建模实现**：
- KiCad 中通过 `fp.IsLocked()` 标记固定元件
- 读取 PCB 文件时收集固定元件列表
- Agent 的 action space 中排除这些元件
- 在 `check_constraints()` 中检查固定元件是否被意外移动

**来源**：
- 工业设计实践：连接器、安装孔、天线、传感器等必须在特定位置
- 现有 `utils/kicad_utils/kicad_utils.py:105` 已识别 `fp.IsLocked()` 但未在环境建模中正确使用

---

### C6：去耦电容距离

**含义**：去耦电容必须放置在目标 IC 的电源引脚附近，距离过远会导致寄生电感过大，电容失去高频去耦效果。

**量化阈值**：第一版采用 **≤ 5mm**（最宽松的硬约束，覆盖所有数字电路场景）。

| 应用场景 | 最大距离 |
|---------|---------|
| 低速 MCU (≤50MHz) | ≤ 5mm |
| 通用数字逻辑 | 3-5mm |
| 高速数字 (>100MHz) | 1-3mm |
| FPGA / BGA | 0-2mm |

**物理原理**：PCB 走线每 1mm 引入约 1nH 寄生电感。在 100MHz 时，5nH 电感产生 3.14Ω 阻抗，远超去耦电容在谐振点附近的 0.5Ω 低阻抗。

**判别条件**：
```
对于每个 IC 的每个 VCC 引脚 group：
  找到与其对应的去耦电容列表
  检查每个电容中心到该 group 最近引脚的距离 ≤ 5mm
  使用 AABB 包围盒计算，而非元件中心距离
```

**改进建议**（源自 Doc C 1.5）：在 reward 中额外加入距离优化项 `reward += max(0, 1 - dist/5.0) × bonus_weight`，鼓励距离越近越好。

**来源**：
- IPC-2221 design guidelines for decoupling capacitor placement
- minintelpcb (2024): "Layout Distance Guidelines for Decoupling Capacitors"
- SpecAp (2026): "Bypass and Decoupling Capacitor Placement Rules"

---

## 二级核心约束（强烈建议）

> **说明**：以下约束不属于硬合法性要求（不违反也能制造 PCB），但它们是 Doc C 调研中识别出的"对布局质量和训练效率影响最大"的缺失项。建议在实现 C0-C6 后优先实施。

---

### S1：拥塞密度控制

**入选理由**：Doc C 评为「所有缺失约束中最重要的一项」——直接影响布线可行性。即使 HPWL 很长、引脚方���不对，拥塞控制得好布线器仍可能成功；拥塞控制不好，其他指标再漂亮布线器也走不通。

**量化阈值**：
- 网格大小：2mm × 2mm
- 网格容量：8 根走线（4 层板标准值）
- 惩罚方式：超过容量部分按 `excess²` 计算（平方放大优先解决最拥挤区域）

**判别条件**：
```
1. 板面划分为 2mm×2mm 均匀网格
2. 对每个网络，用 Prim MST 连接所有焊盘
3. 每条 MST 边用 L-shape 路径近似走线，累计经过的网格
4. 每个网格密度 > 8 的部分按 (density-8)² 累加为拥塞成本
5. 必须使用增量更新——移动一个元件只更新关联网络对应的网格
```

**实现要点**：pyplacer 的 CongestionMap 类仅 153 行，可直接移植。容量参数可按板层调整：2 层板 4-5，6 层板 12-15。

**来源**：pyplacer congestion.py 开源实现

---

### S2：HPWL 线长最小化

**入选理由**：当前代码使用"紧凑度"（所有元件到板中心平均距离）作为线长代理，该指标完全不考虑元件连接关系——会把电源、MCU、无线模块全部推向中心堆在一起。HPWL 利用已有 netlist 信息，是学术界和工业界最通用的布局质量度量。

**量化公式**：
```
HPWL = Σ_{net∈N} [(x_max - x_min) + (y_max - y_min)]
```

**判别条件**：对每个网络，取所有焊盘 x 坐标极差 + y 坐标极差，所有网络求和。计算复杂度 O(N) 对每个网络。

**实现要点**：
1. 用 HPWL 替换紧凑度，改动量约 20 行
2. 建议采用相对改进率 `(初始-当前)/(初始-最优)` 归一化 reward 到 [-1,1]
3. **必须同时实现网络优先级**——否则 GND 的 HPWL 将占总量的 60-80%，误导优化方向
4. 电源网络（GND/VCC/VDD）HPWL 权重设为 0.2，普通信号网络权重 1.0

**来源**：pyplacer placer.py:50；RL_PCB agent.py 相对改进率公式

---

### S3：放置优先级排序

**入选理由**：大元件先放有更多选择空间，小元件后放见缝插针。随机顺序会让大芯片被迫放在次优位置。改动量极小（step() 前排序即可）但训练效果提升显著。

**量化阈值**：固定元件（连接器、安装孔）最先放，之后按连接度或面积降序。

**实现方法**：
```python
# kicad-mcp 风格：连接度降序
sorted_refs = sorted(refs,
    key=lambda r: (conn_score[r], area[r]), reverse=True)

# circuit_training 风格：面积降序
sorted_refs = sorted(refs, key=lambda r: area[r], reverse=True)
```

**来源**：circuit_training environment.py:234；kicad-mcp

---

## 实施优先级矩阵

| 优先级 | 约束 | 改动量 | 影响范围 | 前置依赖 |
|:--:|------|:--:|------|------|
| P0 | C2 重叠（升级为 AABB 连续惩罚） | 中（~30 行） | 所有元件对检测 | 无 |
| P0 | C3 出界+板边距（升级为面积比例） | 中（~30 行） | 所有元件边界检测 | AABB 包围盒 |
| P0 | C6 去耦电容（完善实现） | 小（~15 行） | IC 电源引脚检测 | AABB 包围盒 |
| P1 | S2 HPWL（替换紧凑度） | 小（~20 行） | reward 函数核心 | netlist 信息 |
| P1 | 网络优先级（HPWL 配套） | 极小（~5 行） | HPWL 权重分配 | HPWL |
| P1 | S3 放置排序（step() 前排序） | 极小（~5 行） | 训练效率 | 无 |
| P2 | S1 拥塞密度（CongestionMap） | 大（~153 行） | reward 函数 + 增量更新 | HPWL |
| P3 | C0/C1/C5（Reset + Lock + 固定元件） | 小 | 环境机制 | KiCad API |

> **建议实施顺序**：P0 → P1 → P2 → P3。P0 修复基础检测，P1 更换优化目标并配套权重，P2 加入布线可行性代理，P3 完善环境建模机制。

---

## 环境改造参考伪代码

### reset() 改造

```python
def reset(self):
    # C0: 可移动元件初始置于板外
    for comp in self.movable_components:
        comp.x = board_x_min - 100  # 板外位置
        comp.y = board_y_max + 100

    # C5: 固定元件已在 KiCad 中预置，无需移动
    for comp in self.fixed_components:
        comp.x = comp.fixed_x
        comp.y = comp.fixed_y

    # S3: 按连接度降序排列放置顺序
    self.placement_order = sorted(
        self.movable_components,
        key=lambda r: (self.conn_score[r], self.area[r]),
        reverse=True
    )
    self.placed_list = []
```

### check_constraints() 改造

```python
def check_constraints(self):
    violations = []

    # C2: 元件不重叠检查（AABB + 连续惩罚）
    for i, j in combinations(self.placed_components, 2):
        hard_area, soft_penalty = self._bbox_overlap_penalty(
            self._get_aabb(comp_i, margin=0.25),
            self._get_aabb(comp_j, margin=0.25)
        )
        if hard_area > 0:
            violations.append(("overlap_hard", comp_i.ref, comp_j.ref, hard_area))
        elif soft_penalty > 0:
            violations.append(("overlap_soft", comp_i.ref, comp_j.ref, soft_penalty))

    # C3: 元件在板框内（合�� C4 板边距）
    for comp in self.placed_components:
        oob_area = self._out_of_bounds_area(
            self._get_aabb(comp),
            self._get_valid_area(margin=1.0)  # 1mm 内缩
        )
        if oob_area > 0:
            violations.append(("out_of_bounds", comp.ref, oob_area))

    # C6: 去耦电容距离
    for ic, cap_group in self.decoupling_map.items():
        for cap in cap_group:
            min_dist = self._min_pin_to_cap_distance(ic, cap)  # AABB → 引脚
            if min_dist > 5.0:
                violations.append(("decoupling_dist", cap.ref, ic.ref, min_dist))

    return violations
```

### reward 改造（核心变化对比）

```python
# === 旧版 reward ===
compactness = utils.calculate_layout_compactness(positions)  # 推往中心
reward += compactness * weight_compactness

# === 新版 reward ===
# S2: HPWL 替换紧凑度
hpwl = self._calculate_hpwl(netlist, positions)
reward += (initial_hpwl - hpwl) / (initial_hpwl - optimal_hpwl) * weight_hpwl

# C2: 软硬重叠分层惩罚
reward -= W_OVERLAP_HARD * total_hard_overlap  # 权重 1000
reward -= W_OVERLAP_SOFT * total_soft_spacing   # 权重 50

# C3: 连续出界惩罚
reward -= W_OUT_OF_BOUNDS * total_oob_area      # 权重 200

# S1: 拥塞密度
congestion_cost = congestion_map.cost(capacity=8.0)
reward -= W_CONGESTION * congestion_cost        # 权重 40

# C6: 去耦电容距离奖励（越近越好）
for cap in placed_caps:
    dist = min_pin_to_cap_distance(cap.target_ic, cap)
    reward += max(0, 1 - dist / 5.0) * weight_decoupling
```

---

## 参考文献

1. **IPC-2221A** — Generic Standard on Printed Board Design
2. **IPC-7351B** — Generic Requirements for Surface Mount Design and Land Pattern Standard
3. **OmniLayout** (2026) — A Schematic-Coupled Multimodal Benchmark for Constraint-Aware Geometric Reasoning in PCB Layout, arXiv:2607.03261
4. **HiePlace** (2026) — Efficient Hierarchical PCB Placement, IEEE TCAD
5. **pyplacer** — Open-source Python PCB placer with overlap penalty, congestion map, HPWL, and pin exit direction
6. **RL_PCB** — Reinforcement learning for PCB placement with multi-agent LoS observation
7. **circuit_training** — Google's RL-based macro placement framework with blockage and mask support
8. **kicad-mcp** — KiCad MCP server with automatic keepout detection
9. minintelpcb (2024) — Layout Distance Guidelines for Decoupling Capacitors
10. SpecAp (2026) — PCB Layout Capacitor Placement Best Practices
11. **PCB布局约束体系_最终完整版.md** — 开源项目约束参考与代码改进分析（本项目整合源之一）
