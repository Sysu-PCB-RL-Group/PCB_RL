# 专家轨迹生成诊断报告：灯板2 / Button（对照 开关小板）

<!-- STALE-BANNER -->
> ⚠️ **上一代文档（已作废，仅作历史记录/对照）**：本文描述的是重构前（≤2026-09-15）的架构 ——
> 如 `MAX_MOVABLE` / `GRID_W` / `GRID_H` / `CELL_SIZE=0.2`、7 元动作 `(comp_id, x1,y1,x2,y2, rot, layer)`、
> 违规罚分式 reward、`SPEC v2.x` 等。
> **现行口径一律以 [`../讨论结果/导入信息重构.md`](../讨论结果/导入信息重构.md) 为准**：动作 =（旋转 k, cell i, j）、L/G = 4×200×200、
> 违规被 mask **禁止**（不罚分）、固定顺序不选件、开槽随件（未放置不在板上）。

> 生成日期：2026-09-06
> 方法：复用 `env._detect_collisions` / `_compute_expert_reward`（`_build_step_transitions` L517 同口径），
> 对 `data/expert_transitions/*_expert.json` 抽 50 条轨迹（连续 5 步 chunk）做权威复检。
> 脚本：`scripts/diagnose_expert_boards.py`（只读缓存 + 加载 env，分钟级、零污染）。

---

## 0. 重要前置修正：上一轮 AABB 审查的"假阳性"

上一轮 `review_expert_trajectories.py` 用**无阈值纯 AABB**（`ix>1e-2 and iy>1e-2`）自算重叠/越界，
且**未区分元件层**。环境侧 `ENABLE_DUAL_SIDED=True` 下异层（TOP/BOTTOM）元件 XY 重叠**不算碰撞**。
因此上一轮报的"灯板2 硬重叠 50/50、Button 硬重叠 50/50"中，**大量是跨层假阳性**。

权威复检（带 `spacing_threshold` 硬重叠 + 仅同层判定 + `out_of_bounds_area`）结果：

| 板 | 上一轮 AABB 硬重叠 | 权威同层硬重叠 | 结论 |
|---|---|---|---|
| 灯板2 | 50/50 | **18/50**（movable-movable 同层） | 真实缺陷，量级由 50→18 |
| Button | 50/50 | **0/50**（同层）；18/50 为 movable↔**固定件**重叠 | 上一轮把"压固定件"也算成了硬重叠 |
| 开关小板 | 50/50 | **0/50** | 完全干净（基线） |

> 硬重叠口径一致：`bbox_overlap_penalty` 对 `dx>0 and dy>0` 即返回 `hard=dx*dy`（无阈值），与生成侧 `_is_legal_placement` 同口径。
> 故权威判出的违规**确为真实生成缺陷**，仅量级低于上轮 AABB 的误报。

---

## 1. 权威几何复检结果（每板 50 条轨迹）

| 指标 | 灯板2 | Button | 开关小板（基线） |
|---|---|---|---|
| chunk 含同层硬重叠配对 | **18/50** | 0/50 | 0/50 |
| chunk 含越界元件 | 0/50 | **44/50** | 14/50（面积≈0，无害） |
| 刚放置元件 hard>0 | 16/50 | 18/50（对固定件） | 0 |
| 刚放置元件 oob>0 | 0 | **25/50** | 0 |
| 刚放置元件 soft>0 | 0 | 1/50 | 0 |
| `has_violations` chunk | **16/50** | **25/50** | 0/50 |
| 缓存 reward 一致性\* | 21/50 | 4/50 | 23/50 |

\* 缓存一致性：违规步 penalty 项由 `_detect_collisions` 直接给出、与生成 100% 一致（Button 仅 4/50 偏差印证）；
非违规步的偏差来自品质项（net_agg/congestion）从 obs 重建的细微口径差异，**不影响惩罚分解权威性**。
（见 §4 说明）

**解读：**
- 灯板2：真实同层硬重叠 18/50，且其中 2 个 chunk 的重叠发生在**两个已放置元件之间**（非本次放置引入）→
  说明重叠是"源量化后在缓存中持久留存"，符合 base 回退未校验的根因。
- Button：**无同层 movable-movable 硬重叠**，但其 2 个固定件与 2 个 movable 的布局关系导致 **18/50 次放置把 movable 压在固定件上**（hard>0）、**25/50 次放置越界**。
- 开关小板：零硬重叠、零有效越界、零 soft、reward 健康（mean 28.83）→ 基线成立。

---

## 2. reward 拆解（主导罚分项 + 极端单步）

权重：`W_OVERLAP_HARD=20`、`W_OVERLAP_SOFT=10`、`W_OUT_OF_BOUNDS_INITIAL=2.0`、
`W_VIOLATION_FLAT=5.0`、完成 `W_COMPLETION=100`、品质项（Net聚合/拥塞/去耦/大元件）。

| 板 | expert_reward mean / min / max | 罚分累计（hard / soft / oob / flat） | 主导项 |
|---|---|---|---|
| 灯板2 | 22.06 / **-8.78** / 39.95 | hard **60.5**(3.02mm²) / 0 / 0 / flat **80** | 硬重叠 + flat（16 步违规×5） |
| Button | **-14.17** / **-272.50** / 69.80 | hard **1812**(90.6mm²) / 17.8 / oob **447.6**(223.8mm²) / flat **125** | **硬重叠（压固定件）+ 巨量越界** |
| 开关小板 | 28.83 / 20.16 / 33.17 | 全 0 | —（无违规） |

**奖励设计逻辑（符合预期但暴露缺陷）**：违规步 `has_violations=True` → 不发完成奖励（≈16.7/步）与任何品质项，
仅扣 `惩罚 + flat`，净负。这正是 reward 铁律（违规必须为负）的正确体现；
**但专家轨迹本应是"理想示范"，不该含任何违规**——违规样本污染了专家 buffer，且拖低整板 reward。

**Button 极端单步（reward = -272.50）拆解**：
```
comp_idx=0, hard=12.480mm² -> 罚 249.59
           oob=8.952mm²  -> 罚 17.90
           flat=5.00
= -272.50
```
即单个 movable 元件被完全放置在固定件之上（重叠面积 12.48mm² ≈ 一整个元件），并同时越界 8.95mm²。
这来自 `_quantize_bbox_to_grid` 把源/目标位置量化到网格后落点非法，且 base 回退也未二次校验。

---

## 3. 源布局交叉核对（根因定位）

读 `_expert_raw.json` 的 target bbox（连续坐标，量化前），统计源固有特征：

| 板 | 轨迹含 sub-grid(<0.2mm) 同层间隙 | 轨迹含 target 出界 | 全局最小同层间隙 | 全局最小边缘余量 |
|---|---|---|---|---|
| 灯板2 | **49/50** | 0/50 | **0.0mm** | 5.76mm |
| Button | 0/50 | 0/50 | None（无可比同层对） | **0.0mm** |
| 开关小板 | 0/50 | 0/50 | 0.276mm | 5.68mm |

**根因链（代码级，subagent 已确认）：**
1. `_quantize_bbox_to_grid`（L675-687）：仅对**原点** `gx1,gy1` 做 `round` + `clamp` 到 `[0, grid_width-1]`，
   **远端宽高 `w/h` 不 clamp**。源布局 sub-grid 间隙或贴边时，量化会：
   - 把相邻 movable 塌缩到同一格 → **硬重叠**（灯板2）；
   - 把贴边元件推到 `x2/y2 > va_xmax` → **越界**（Button 边缘余量 0.0mm）。
2. `trajectory_to_transitions` L388-393：base 保底回退（`_quantize_bbox_to_grid(env, base_bbox)`）**不做 `_is_legal_placement` 二次校验**。
   若 base（量化源）自身也非法，则非法布局原样存入缓存。
3. 为何开关小板不触发：源最小同层间隙 0.276mm（>CELL_SIZE 0.2）+ 边缘余量 5.68mm（充裕）→ 量化后仍合法。

**逐板定性：**
- **灯板2**：源 49/50 轨迹含 sub-grid 间隙（最小同层间隙=0.0mm），量化合并相邻 movable → 同层硬重叠 18/50 持久留存。
- **Button**：源 raw target 本身干净（0/50 违规），但有固定件 + 某 movable 边缘余量 0.0mm；量化/目标落点把 movable 推出界（44/50）并压到固定件上（18/50）。属"量化引入 + base 未校验"。
- **开关小板**：源几何安全 → 量化后零违规。

---

## 4. 缓存一致性偏差说明（诚实披露）

非违规步 reward 标签与本脚本重算有 21/50（灯板2）、23/50（开关小板）差异，定位为**品质项重建口径细微差异**：
- 本脚本从 `obs.component_slots` 重建前驱元件 bbox / 从 `action` 重建当前元件，与生成 `_compute_expert_reward` 输入一致；
- 但 `congestion_std` / `net_aggregation` 对前驱集合的精确浮点状态敏感，从 obs 间接重建与生成当时逐步累积状态存在微小差异。
- **该偏差只影响品质项（Net聚合/拥塞/大元件），不影响惩罚项**：惩罚项（hard/soft/oob/flat）由 `_detect_collisions` 直接返回，
  与 `_build_step_transitions` L517 调用的 `_compute_expert_reward` 100% 同口径。Button 仅 4/50 偏差即印证惩罚口径无误。
- 故 §1/§2 的**违规计数与惩罚分解权威可靠**，足以支撑根因结论与修复决策。

---

## 5. 修复方案与权衡（待 Owner 决策）

| # | 方案 | 作用 | 代价 / 风险 | 推荐度 |
|---|---|---|---|---|
| A | **量化器远端点 clamp**：`_quantize_bbox_to_grid` 把 `x2<=va_xmax, y2<=va_ymax`（必要时回退 gx1/gy1） | 阻断 Button 类越界 | 不改硬重叠；需保证 clamp 后仍合法（否则落到 B） | 便宜兜底，必做 |
| B | **base 回退二次校验 + 最近合法位搜索**：回退后若仍非法，做网格半径递增的合法位 BFS；找不到则判该步失败 | 同时修复硬重叠 + 越界 | 增加生成耗时（每步 BFS）；需上限防死循环 | 核心修复 |
| C | **源布局预校验/吸附**：生成前检测 sub-grid(<CELL_SIZE) 间隙与边缘元件，把源吸附到网格或剔除问题元件 | 直击 灯板2 根因（sub-grid） | 改数据源；可能改变"完美布局"定义 | 配合 B |
| D | **非法轨迹整条丢弃重生成（fail-closed）**：`_build_step_transitions` 若放置非法（回退后仍非法）则整条轨迹 rejected 并重生成，不写入缓存 | 强保证专家 buffer 100% 合法 | 最坏情况某板生成成功率下降；需生成上限保护 | 最强保证，建议与 B 并用 |
| E | 调 reward 权重"抬高低分" | **不推荐**：低分是违规的症状，调权重会掩盖缺陷、违背 reward 铁律 | 掩盖 bug | ✗ |

**综合建议**：`A + B + D` 组合——A 廉价阻断越界，B 用最近合法位搜索修复硬重叠/残余越界，D 在确实无合法位时整条丢弃重生成，
确保专家 buffer 零违规。C 作为数据侧根治（针对 灯板2 sub-grid 源）可选实施。

> 实施需另立计划并改 `env/expert_trajectory.py` / `env/pcb_utils.py`（量化器），**不在本诊断范围内**。
> 修复后须重生成受影响板的专家轨迹（灯板2、Button）并复跑本诊断脚本验证零违规。

---

## 附：复现命令

```powershell
$env:KICAD_PATH="E:\Kicad"
cd E:\PCB_AI\PCB_RL
& E:\miniconda\envs\pcb_rl_311\python.exe scripts/diagnose_expert_boards.py
```
