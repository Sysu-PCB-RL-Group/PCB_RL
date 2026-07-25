# SPEC：PCB 布局 RL 环境技术标准

> **地位**：项目唯一技术标准，所有代码实现以此为准  
> **范围**：PCB 元件布局阶段（placement），不含布线（routing）  
> **框架**：Gymnasium（`gym.Env` 接口）  
> **日期**：2026-07-25  
> **基于**：`环境构建约束.md`（已整合用户反馈）

---

## 1. 环境架构

### 1.1 模块组成

```
main.py ──▶ run.py ──▶ env/pcb_joint_env.py (PCBJointEnv)
                          ├── env/pcb_layout_env.py  (PCBLayoutEnv)
                          ├── env/pcb_routing_env.py (PCBRoutingEnv)
                          └── env/pcb_utils.py       (PCBUtils)
                                  └── kicad_utils/kicad_utils.py (KiCadUtils)
```

### 1.2 依赖方向

- `kicad_utils/`——KiCad `pcbnew` API 封装，提供 `Component`、`Net`、`board_bounds`
- `env/pcb_utils.py`——调用 `KiCadUtils`，提供 `check_layout_constraints`、`calculate_hpwl` 等
- `env/pcb_layout_env.py`——Gymnasium Env，布局环境（本期重点）
- `env/pcb_routing_env.py`——布线环境（暂缓）
- `env/pcb_joint_env.py`——联合环境，包装 Layout + Routing
- `run.py`——训练循环 Runner，调用 `PCBJointEnv`
- `main.py`——CLI 入口，argparse 配置

### 1.3 架构修复清单（A-1 ~ A-15）

| 编号 | 问题 | 要求 |
|------|------|------|
| A-1 | `PCBUtils.__init__()` 不接受参数，但调用方传参 | 统一接口：`__init__` 接受 `config_path` 或调用方移除传参 |
| A-2 | `get_component_list()` 不存在 | 改用已有的 `get_all_component_states()` |
| A-3 | `get_netlist_state()` 不存在 | 改用已有的 `get_all_net_states()`，布局阶段不使用"未布线网络数" |
| A-4 | `self.config` 不存在 | 移除 JSON 依赖，板框用 `kicad.get_board_bounds()`，reward 权重定义在代码常量中 |
| A-5 | 板框来源不明 | 统一使用 `kicad_utils.get_board_bounds()` |
| A-6 | `component_constraints` 不存在 | 使用 `Component.is_fixed` 属性替代 JSON 配置 |
| A-7 | 动作空间缺少 rotation / layer | 5 元动作（见 §3.1） |
| A-8 | 观察空间缺元数据 | 扩展 observation（见 §2.1） |
| A-9 | 无动作掩码 | 提供 mask 向量，非法动作 logits → `-inf` |
| A-10 | `get_all_files()` 不存在 | 在 `PCBJointEnv` 中实现或修改 Runner |
| A-11 | `max_episode_step` 未定义 | `main.py` 添加 `--max_episode_step` 参数 |
| A-12 | `forbidden_areas` 为空 | 从 KiCad Zone 读取 keepout |
| A-13 | compactness 终止条件 | 改为 `placed_all && no_violations` |
| A-14 | "未布线网络数"观察无效 | 移除，替换为 HPWL |
| A-15 | UTF-8 乱码 | 重新保存为 UTF-8 |

---

## 2. 状态空间（Observation Space）

### 2.1 Observation 字典

```python
observation = {
    # --- 当前待放置元件 ---
    "current_component": {
        "id": int,              # 元件索引 (0 ~ N-1)
        "type": int,            # 封装类型编码 (0: Chip, 1: SOIC/SOT, 2: QFP, 3: BGA, 4: IC通用)
        "width": float,         # 封装宽度 (mm)
        "height": float,        # 封装高度 (mm)
        "pin_count": int,       # 引脚数
        "net_degree": int,      # 当前元件连接的网络数
    },

    # --- 已放置元件列表 ---
    "placed_components": [
        {
            "id": int,
            "x": float, "y": float,     # 中心坐标 (mm)
            "rotation": int,            # 0/90/180/270
            "layer": int,               # 0=TOP, 1=BOTTOM
            "width": float, "height": float,
        },
        ...
    ],

    # --- 板框几何 ---
    "board": {
        "x_min": float, "y_min": float,
        "x_max": float, "y_max": float,
        "valid_area": {                 # 板框内缩 1mm 后
            "x_min": float, "y_min": float,
            "x_max": float, "y_max": float,
        },
    },

    # --- 全局指标 ---
    "global": {
        "hpwl": float,                  # 当前总 HPWL (mm)
        "initial_hpwl": float,          # reset 时的初始 HPWL (mm)
        "placed_count": int,            # 已放置元件数
        "total_movable": int,           # 可移动元件总数
        "steps_done": int,              # 当前 step 数
    },

    # --- 动作掩码 ---
    "action_mask": np.ndarray,          # shape=(N, grid_x, grid_y, 4, 2), 0/1
}
```

### 2.2 封装类型编码

| 编码 | 封装类型 | 最小间距 |
|------|----------|---------|
| 0 | Chip (0402/0603/0805) | 0.2mm |
| 1 | SOIC / SOT | 1.25mm |
| 2 | QFP | 2.5mm |
| 3 | BGA | 5mm |
| 4 | IC 通用 | 2mm |

---

## 3. 动作空间（Action Space）

### 3.1 动作结构

```python
action = (component_id, grid_x, grid_y, rotation, layer)
#         Discrete(N)   Discrete(GX)  Discrete(GY)  Discrete(4)  Discrete(2)
```

- `component_id`：在 `movable_list` 中选取（已放置元件不可选）
- `grid_x`, `grid_y`：离散化坐标，映射到 `valid_area` 内的网格点
- `rotation`：`0`=0°, `1`=90°, `2`=180°, `3`=270°
- `layer`：`0`=TOP, `1`=BOTTOM

### 3.2 网格离散化

网格单元大小由 `cell_size` 配置（默认 1.0mm），网格维度由 `valid_area` 范围决定：

```python
grid_width  = int((valid_area.x_max - valid_area.x_min) / cell_size)
grid_height = int((valid_area.y_max - valid_area.y_min) / cell_size)
```

坐标还原公式：`x = valid_area.x_min + grid_x * cell_size`，`y` 同理。

### 3.3 动作掩码

- 已放置元件对应 slot 的 mask = 0
- 非法坐标（超出 valid_area）的 mask = 0
- 策略网络 softmax 前：`logits[mask == 0] = -inf`

### 3.4 推理阶段 clamp

```python
def clamp_to_board(x, y, w, h, valid_area, spacing=0.0):
    half_w, half_h = w / 2, h / 2
    x = max(valid_area["x_min"] + half_w + spacing,
            min(x, valid_area["x_max"] - half_w - spacing))
    y = max(valid_area["y_min"] + half_h + spacing,
            min(y, valid_area["y_max"] - half_h - spacing))
    return x, y
```

---

## 4. 生命周期

### 4.1 reset()

```
reset():
    1. 通过 kicad.get_board_bounds() 获取板框
    2. valid_area = board 内缩 1mm（板边安全裕度）
    3. 读取所有元件，过滤 Component.is_fixed：
       - fixed_list   = [comp | comp.is_fixed == True]
       - movable_list = [comp | comp.is_fixed == False]
    4. 固定元件：直接设置 KiCad 预置位置
    5. 可移动元件：统一放到板框外
       - 位置：x = board.x_min - 100mm, y = board.y_max + 100mm
       - 判定标准：元件中心点在板外 && AABB 完全不在板框内
       - 验证 -100/+100 偏移是否足够覆盖所有元件尺寸
    6. 按面积降序排列 movable_list → placement_order（§7.2）
    7. placed_list = []
    8. 计算 initial_hpwl（所有固定元件的焊盘 HPWL）
    9. 返回 observation
```

### 4.2 step(action)

```
step(action):
    1. 解析 action (component_id, grid_x, grid_y, rotation, layer)
    2. 检查 component_id ∉ placed_list（C1-2），已放置则拒绝
    3. 将元件放置到指定位置
    4. 计算约束违规量（§5）→ violations
    5. 计算 reward（§6）
    6. 如果无违规：placed_list.append(component_id)
    7. 判定 done（§4.3）
    8. 返回 (obs, reward, done, truncated, info)
```

### 4.3 终止条件

```python
done = (len(placed_list) == len(movable_list)) and (total_violations == 0)
```

### 4.4 Cosine Epsilon 出界衰减

训练阶段使用余弦衰减控制出界允许率，替代一刀切终止：

```python
epsilon = cos((steps_done / max_steps) * (pi / 2))
```

- `epsilon` 接近 1（前期）：容忍出界，施加惩罚但不终止
- `epsilon` 接近 0（后期）：严格禁止，累积惩罚导致终止

---

## 5. 约束体系

### 5.1 C1：元件锁定

| 子项 | 要求 |
|------|------|
| C1-1 | reset() 中初始化 `placed_list = []` |
| C1-2 | step() 中：`component_id ∉ placed_list` 才允许操作 |
| C1-3 | 终止条件：`len(placed_list) == len(movable_list) && violations == 0` |

`placed_list` 与 `movable_list`（C5-2）统一管理，三处引用同一列表。

### 5.2 C2：元件不重叠

#### 5.2.1 碰撞检测方法

使用 AABB 包围盒检测，**不引入多边形轮廓**。获取包围盒：

```python
bbox = fp.GetBoundingBox()  # → (x_min, y_min, x_max, y_max)
```

#### 5.2.2 bbox_overlap_penalty

```python
KEEPOUT_SOFT = 2.0  # 软间距阈值 (mm)

def bbox_overlap_penalty(a, b):
    """
    返回 (hard_overlap, soft_spacing)
    - hard_overlap: 交叉面积 (mm²)
    - soft_spacing: 连续软间距惩罚值
    """
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])

    if dx > 0 and dy > 0:
        return (dx * dy, 0.0)        # 硬重叠

    if dx > 0 >= dy:
        gap, edge = -dy, dx
    elif dy > 0 >= dx:
        gap, edge = -dx, dy
    else:
        return (0.0, 0.0)            # 角对角：零惩罚

    if gap >= KEEPOUT_SOFT:
        return (0.0, 0.0)            # 间距 ≥ 2mm，不惩罚
    return (0.0, (KEEPOUT_SOFT - gap) ** 2 * edge)
```

**角对角规则**：对角距离大于阈值时零惩罚（阈值待定）。

#### 5.2.3 分层惩罚权重

| 类型 | 条件 | 权重 |
|------|------|------|
| 硬重叠 | `hard_overlap > 0` | `W_OVERLAP_HARD = 1000.0` |
| 软间距 | `hard_overlap == 0 && soft_spacing > 0` | `W_OVERLAP_SOFT = 50.0` |
| 不惩罚 | `soft_spacing == 0` | 0 |

#### 5.2.4 分类间距阈值

阈值配置为数组，按元器件封装类型索引：

```python
SPACING_THRESHOLDS = {
    0: 0.2,     # Chip (0402/0603/0805)
    1: 1.25,    # SOIC/SOT
    2: 2.5,     # QFP
    3: 5.0,     # BGA
    4: 2.0,     # IC 通用
}
```

默认阈值 0.25mm（IPC-7351B Level B Nominal）。

### 5.3 C3：板框约束

#### 5.3.1 检测方法

计算元件 AABB 四角 `(x ± w/2, y ± h/2)` 是否均在 `valid_area` 内。

#### 5.3.2 out_of_bounds_area

```python
W_OUT_OF_BOUNDS = 1000.0  # 与 W_OVERLAP_HARD 对齐

def out_of_bounds_area(bbox, valid_area):
    bxmin, bymin, bxmax, bymax = bbox
    oxmin, oymin, oxmax, oymax = valid_area

    bbox_area = max(0.0, bxmax - bxmin) * max(0.0, bymax - bymin)
    ix = max(0.0, min(bxmax, oxmax) - max(bxmin, oxmin))
    iy = max(0.0, min(bymax, oymax) - max(bymin, oymin))

    return bbox_area - ix * iy  # 出界面积 (mm²)
```

#### 5.3.3 板边安全裕度

板框内缩 **1mm** 后为 `valid_area`（IPC-2221A 0.5mm 与制造商推荐 1mm 之间折中）。

#### 5.3.4 出界控制

- 训练阶段：cosine epsilon 衰减（§4.4）
- 推理阶段：clamp_to_board（§3.4）

### 5.4 C5：固定元件

#### 5.4.1 固定元件识别

使用 KiCad `fp.IsLocked()` → `Component.is_fixed`，不再依赖 JSON：

```python
self.movable_list = [c for c in self.kicad.components.values() if not c.is_fixed]
self.fixed_list   = [c for c in self.kicad.components.values() if c.is_fixed]
```

#### 5.4.2 联动

`movable_list` 同时用于 C1-2（已放置检查）、C1-3（终止判定）、S3（放置排序）。

### 5.5 C6：去耦电容 + Net 聚合度

#### 5.5.1 去耦电容识别

遍历 netlist → IC 含 VCC/VDD 引脚 → 同一网络上查找 MLCC 电容（value/footprint 含 "C"）→ 建立 `decoupling_map = {ic_ref: [cap_ref, ...]}`。

#### 5.5.2 去耦电容距离检查

对每个 `(ic, cap)` 配对，电容到 IC 电源引脚组最小距离 > 5mm → 违规。

#### 5.5.3 去耦电容距离奖励

```python
for cap in placed_caps:
    dist = min_pin_to_cap_distance(cap.target_ic, cap)
    reward += max(0, 1 - dist / 5.0) * W_DECOUPLING
```

#### 5.5.4 Net 聚合度

对同一 net 的所有元器件（中心点），定义聚合度为各元件到 net 中心的平均距离：

```python
def net_aggregation(comp_positions):
    if len(comp_positions) < 2:
        return 0.0
    xs = [p[0] for p in comp_positions]
    ys = [p[1] for p in comp_positions]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    dists = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in comp_positions]
    return sum(dists) / len(dists)
```

聚合度越小 → 元件越紧密 → 奖励越大。

---

## 6. Reward 函数

### 6.1 完整公式

```python
reward = 0.0

# --- 惩罚项 ---
reward -= W_OVERLAP_HARD   * total_hard_overlap        # 硬重叠面积 × 1000
reward -= W_OVERLAP_SOFT   * total_soft_spacing         # 软间距惩罚 × 50
reward -= W_OUT_OF_BOUNDS  * total_out_of_bounds_area   # 出界面积 × 1000

# --- 奖励项 ---
# HPWL 线长改进
hpwl = calculate_hpwl(netlist, positions)
reward += (initial_hpwl - hpwl) / (initial_hpwl - optimal_hpwl) * W_HPWL

# 去耦电容距离
for cap in placed_caps:
    dist = min_pin_to_cap_distance(cap.target_ic, cap)
    reward += max(0, 1 - dist / 5.0) * W_DECOUPLING

# Net 聚合度
for net_id, comps in net_groups.items():
    agg = net_aggregation(comps)
    reward -= agg * W_NET_AGGREGATION  # 聚合度越小越好（减惩罚）
```

### 6.2 HPWL 计算

```python
def hpwl_for_net(pad_positions):
    if len(pad_positions) < 2:
        return 0.0
    xs = [p[0] for p in pad_positions]
    ys = [p[1] for p in pad_positions]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

def calculate_hpwl(netlist, positions):
    total = 0.0
    for net in netlist:
        pad_positions = [positions[ref] for ref in net.connections]
        hpwl = hpwl_for_net(pad_positions)
        # 网络优先级：电源网络 0.2，普通信号 1.0
        weight = 0.2 if net.is_power else 1.0
        total += hpwl * weight
    return total
```

### 6.3 网络优先级

电源网络 `(GND/VCC/VDD/VSS)` HPWL 权重 = **0.2**，普通信号网络 = **1.0**。

---

## 7. 辅助系统

### 7.1 S2：HPWL 线长

- 替换 `calculate_layout_compactness()` 为 `calculate_hpwl()`（§6.2）
- 直接使用 `kicad.nets` 缓存

### 7.2 S3：放置优先级排序

```python
self.placement_order = sorted(
    self.movable_list,
    key=lambda r: self.area[r],   # 仅按面积降序
    reverse=True
)
```

固定元件不参与排序，reset 时直接设 KiCad 预置位置。

### 7.3 S1：拥塞密度控制

> ⚠️ **待确认**：询问学长 layout 阶段是否需要拥塞密度计算。

S1-2 / S1-3 / S1-4 暂不纳入实现。

---

## 8. 常量表

### 8.1 Reward 权重

| 常量 | 值 | 说明 |
|------|-----|------|
| `W_OVERLAP_HARD` | 1000.0 | 硬重叠惩罚 |
| `W_OVERLAP_SOFT` | 50.0 | 软间距惩罚 |
| `W_OUT_OF_BOUNDS` | 1000.0 | 出界惩罚（与硬重叠对齐） |
| `W_HPWL` | 待定 | HPWL 改进奖励 |
| `W_DECOUPLING` | 待定 | 去耦电容距离奖励 |
| `W_NET_AGGREGATION` | 待定 | Net 聚合度惩罚 |

### 8.2 几何阈值

| 常量 | 值 | 说明 |
|------|-----|------|
| `KEEPOUT_SOFT` | 2.0 mm | 软间距阈值 |
| `BOARD_MARGIN` | 1.0 mm | 板边安全裕度 |
| `DECOUPLING_MAX_DIST` | 5.0 mm | 去耦电容最大允许距离 |
| `CELL_SIZE` | 1.0 mm | 网格离散化单元大小 |
| `DEFAULT_SPACING` | 0.25 mm | 默认元件间距（IPC-7351B Level B） |

### 8.3 分类间距阈值

| 封装编码 | 封装类型 | 间距 (mm) |
|----------|----------|-----------|
| 0 | Chip | 0.2 |
| 1 | SOIC/SOT | 1.25 |
| 2 | QFP | 2.5 |
| 3 | BGA | 5.0 |
| 4 | IC 通用 | 2.0 |

### 8.4 网络优先级

| 网络类型 | HPWL 权重 |
|----------|-----------|
| 电源 (GND/VCC/VDD/VSS) | 0.2 |
| 普通信号 | 1.0 |

### 8.5 动作空间维度

| 维度 | 类型 | 范围 |
|------|------|------|
| `component_id` | Discrete | `0 ~ len(movable_list)-1` |
| `grid_x` | Discrete | `0 ~ grid_width-1` |
| `grid_y` | Discrete | `0 ~ grid_height-1` |
| `rotation` | Discrete | `{0, 1, 2, 3}` → `{0°, 90°, 180°, 270°}` |
| `layer` | Discrete | `{0, 1}` → `{TOP, BOTTOM}` |

---

## 附录：变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-07-25 | 1.0 | 基于 `环境构建约束.md`（用户反馈整合版）初始发布 |
