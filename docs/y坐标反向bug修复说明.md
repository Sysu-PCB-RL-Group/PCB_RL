# KiCad 数据加载 y 坐标反向 Bug 修复说明

<!-- STALE-BANNER -->
> ⚠️ **上一代文档（已作废，仅作历史记录/对照）**：本文描述的是重构前（≤2026-09-15）的架构 ——
> 如 `MAX_MOVABLE` / `GRID_W` / `GRID_H` / `CELL_SIZE=0.2`、7 元动作 `(comp_id, x1,y1,x2,y2, rot, layer)`、
> 违规罚分式 reward、`SPEC v2.x` 等。
> **现行口径一律以 [`讨论结果/导入信息重构.md`](讨论结果/导入信息重构.md) 为准**：动作 =（旋转 k, cell i, j）、L/G = 4×200×200、
> 违规被 mask **禁止**（不罚分）、固定顺序不选件、开槽随件（未放置不在板上）。

> 日期：2026-08-18
> 影响模块：`utils/kicad_utils/kicad_utils.py`（核心）、`utils/visualization.py`
> 状态：已修复 + 已验证

---

## 一、问题概述

KiCad 数据加载层构造元件 bbox 时，y 坐标存反（`y1 > y2`），导致下游所有 y 方向的碰撞、软间距、出界、keepout 检测**完全失效**。表现为「完美布局软间距全为 0」的假象，掩盖了 reward 函数与人类布局间距标准不匹配的深层问题。

---

## 二、根因分析

### 2.1 坐标系差异

KiCad 使用 **Y 轴向下** 的屏幕坐标系，`GetBoundingBox().GetBottom()` 返回的 y 值 **大于** `GetTop()`：

```
GetTop()    = 元件上边缘（小 y 值）
GetBottom() = 元件下边缘（大 y 值）
```

而下游所有检测函数（`bbox_overlap_penalty`、`out_of_bounds_area`、`check_keepout`）均假设标准笛卡尔坐标 `y1 < y2`。

### 2.2 核心错误：交换 ≠ 取反

原代码把 `GetBottom()`（大值）塞进 `y1` 槽、`GetTop()`（小值）塞进 `y2` 槽，**只交换位置、没有取负号**：

```python
# 修复前（bug）：交换了 GetBottom/GetTop，但没乘 -1
bbox_mm = (
    pcbnew.ToMM(bbox.GetLeft()),
    pcbnew.ToMM(bbox.GetBottom()),   # 大值 → y1
    pcbnew.ToMM(bbox.GetRight()),
    pcbnew.ToMM(bbox.GetTop()),      # 小值 → y2
)
# 结果：y1 > y2，h = y2 - y1 为负
```

「取反」（镜像翻转）应该是 `(y1, y2) → (-y2, -y1)`，保持 `y1 < y2`、相对距离不变；而「交换」（swap）是 `(y1, y2) → (y2, y1)`，导致 `y1 > y2`、破坏约定。**这是少了一个负号的 bug**。

### 2.3 软间距为什么被算错

下游核心一行：

```python
dy = min(a.y2, b.y2) - max(a.y1, b.y1)   # 本意：相邻端间隙
gap = -dy
```

以开关小板两排焊盘为例（真实物理间隙 0.37mm）：

| 元件 | 物理 top | 物理 bottom |
|---|---|---|
| Pad_e15 | 107.47 | 108.99 |
| Pad_e16 | 109.36 | 110.89 |

**修复前（y1>y2）：**

```python
dy = min(107.47, 109.36) - max(108.99, 110.89) = -3.42
gap = 3.42   # ❌ 变成「两端总跨度」，不是间隙
```

`min(y2)` 取到最靠上的 top、`max(y1)` 取到最靠下的 bottom，`min - max` 从「相邻端间隙」变成了「两端总跨度」，软间距因此被漏检。

**修复后（y1<y2）：**

```python
dy = min(108.99, 110.89) - max(107.47, 109.36) = -0.37
gap = 0.37   # ✅ 真实间隙
```

---

## 三、影响范围

| 下游 | 影响 | 说明 |
|---|---|---|
| `_calc_board_bounds` / `valid_area` | 无 | 用 min/max 计算，本就 y_min<y_max |
| `bbox_overlap_penalty`（碰撞/软间距） | 修复 | 假设 y1<y2，核心收益 |
| `out_of_bounds_area` / `clamp_to_board` | 修复 | 同假设 y1<y2 |
| `check_keepout` | 修复 | 同假设 y1<y2，keepout 区也需同步修 |
| `min_pin_to_cap_distance` | 无 | `abs_y = bbox[1] + rel_y`，`rel_y = pad_y - bbox[1]` 抵消 |
| `congestion_std` | 顺带修复 | 此前 y 反向导致 row 覆盖为空 |
| `_get_obs` 坐标编码 | 自洽 | 直接读 bbox[1]/[3]，全元件同一约定 |
| grid 坐标转换 + reward 重建 | 修复 | gy1 语义从「底部」纠正为「顶部」 |
| `draw_layout` 可视化 | 顺带修复 | 此前 h<0 元件不绘制；需加 invert_yaxis |

---

## 四、修复内容

### 4.1 `utils/kicad_utils/kicad_utils.py` — `_cache_components`（L250-259）

```python
# AABB: (x1=left, y1=top, x2=right, y2=bottom)
# 注意：KiCad 坐标系 Y 轴向下，GetTop() 的 y 值 < GetBottom()，
# 为满足下游 bbox 统一约定 y1 < y2，这里 y1 取 GetTop（小值）、
# y2 取 GetBottom（大值）。
bbox_mm = (
    pcbnew.ToMM(bbox.GetLeft()),
    pcbnew.ToMM(bbox.GetTop()),
    pcbnew.ToMM(bbox.GetRight()),
    pcbnew.ToMM(bbox.GetBottom()),
)
```

### 4.2 `utils/kicad_utils/kicad_utils.py` — `get_keepout_zones`（L454-462）

keepout zone 的 AABB 存在**相同 bug**，一并修复：

```python
keepouts.append((
    pcbnew.ToMM(bbox.GetLeft()),
    pcbnew.ToMM(bbox.GetTop()),
    pcbnew.ToMM(bbox.GetRight()),
    pcbnew.ToMM(bbox.GetBottom()),
))
```

### 4.3 `utils/visualization.py` — `draw_layout`（L203 后）

加 `ax.invert_yaxis()`，使可视化方向与 KiCad 实际视图（Y 向下、top 在上方）一致：

```python
ax.set_ylim(min_y - margin, max_y + margin)
ax.invert_yaxis()
ax.set_aspect("equal")
```

---

## 五、验证结果

1. `py_compile` + `read_lints` 无错误。
2. mock 测试 `tests/test_expert_trajectory.py` 9 个用例全部通过。
3. 真实 KiCad 加载后，所有 movable 元件 `bbox` 满足 `y1 < y2`（每元件 `h = 1.52` 为正）。
4. 完美布局真实 y 间距 0.37mm 被 `bbox_overlap_penalty` 正确识别为软间距（`soft > 0`）。
5. 用户确认可视化方向与 KiCad 视图一致。

---

## 六、遗留问题（待处理）

修复 y 反向后，暴露出 **reward 间距阈值与人类布局不匹配** 的问题：

- 完美布局（人类布局）焊盘实际间距 **0.37mm**；
- reward 的 `spacing_threshold_for` 对焊盘类（pkg=4）阈值 **2.0mm**（约 5.4 倍）；
- 导致「人类认可的完美布局」在 reward 函数下被判定为软间距、reward 为负（实测 min = -72.38）。

这是 reward 软间距惩罚部分**需要重塑**的方向，涉及 `env/pcb_layout_env.py` 的 `spacing_threshold_for` 阈值表校准，待后续决策。

---

## 七、副作用

1. **旧 checkpoint 失效**：obs 的 y 坐标语义变化，已有 `best_model.pt` 需重新训练。
2. **pins 的 `rel_y` 基准变化**：从「相对 bottom」变「相对 top」，但下游绝对值不变（抵消）。
3. **专家轨迹数据过期**：y 修复前生成的 `data/expert_transitions/*.json` 需重新生成。
