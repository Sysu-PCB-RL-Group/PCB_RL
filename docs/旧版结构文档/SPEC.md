# SPEC：PCB 布局 RL 环境技术标准

<!-- STALE-BANNER -->
> ⚠️ **上一代文档（已作废，仅作历史记录/对照）**：本文描述的是重构前（≤2026-09-15）的架构 ——
> 如 `MAX_MOVABLE` / `GRID_W` / `GRID_H` / `CELL_SIZE=0.2`、7 元动作 `(comp_id, x1,y1,x2,y2, rot, layer)`、
> 违规罚分式 reward、`SPEC v2.x` 等。
> **现行口径一律以 [`../讨论结果/导入信息重构.md`](../讨论结果/导入信息重构.md) 为准**：动作 =（旋转 k, cell i, j）、L/G = 4×200×200、
> 违规被 mask **禁止**（不罚分）、固定顺序不选件、开槽随件（未放置不在板上）。

> **地位**：项目唯一技术标准，所有代码实现以此为准  
> **范围**：PCB 元件布局阶段（placement），不含布线（routing）  
> **框架**：Gymnasium（`gym.Env` 接口）  
> **日期**：2026-07-27  
> **基于**：`环境构建约束.md` + 多轮设计讨论确认

---

## 1. 环境架构

### 1.1 模块组成

```
main.py ──▶ run.py ──▶ env/pcb_layout_env.py  (PCBLayoutEnv)
                          └── env/pcb_utils.py       (PCBUtils)
                                └── utils/kicad_utils/kicad_utils.py (KiCadUtils)
```

### 1.2 依赖方向

- `utils/kicad_utils/`——KiCad `pcbnew` API 封装，提供 `Component`、`Net`、`board_bounds`
- `env/pcb_utils.py`——调用 `KiCadUtils`，提供约束检测、碰撞检测、reward 各分项计算
- `env/pcb_layout_env.py`——Gymnasium Env，布局环境核心
- `run.py`——训练循环 Runner，epsilon-greedy + curriculum 机制
- `main.py`——CLI 入口，argparse 配置

### 1.3 架构修复清单（A-1 ~ A-15）

| 编号 | 问题 | 状态 |
|------|------|------|
| A-1 | `PCBUtils.__init__()` 接口不匹配 | 统一：移除参数，内部从 `argument.txt` 加载配置 |
| A-2 | `get_component_list()` 不存在 | 改用已有的 `get_all_component_states()` |
| A-3 | `get_netlist_state()` 不存在 | 改用已有的 `get_all_net_states()` |
| A-4 | `self.config` 不存在 | 移除 JSON 依赖，板框用 `kicad.get_board_bounds()`，参数统一在 `argument.txt` |
| A-5 | 板框来源不明 | 统一使用 `KiCadUtils.get_board_bounds()` |
| A-6 | `component_constraints` 不存在 | 使用 `Component.is_fixed` 属性替代 JSON 配置 |
| A-7 | 动作空间缺少 rotation / layer | 7 元动作：`(component_id, x1, y1, x2, y2, rotation, layer)` |
| A-8 | 观察空间缺元数据 | Fixed-slot 编码（见 §3.2） |
| A-9 | 无动作掩码 | 提供 mask 向量，非法动作 logits → `-inf` |
| A-10 | `get_all_files()` 不存在 | 在 `Runner` 中实现，直接扫描 `data/` 目录 |
| A-11 | `max_episode_step` 未定义 | `main.py` 添加 `--max_episode_step` 参数 |
| A-12 | `forbidden_areas` 为空 | 从 KiCad Zone 读取 keepout |
| A-13 | compactness 终止条件 | 改为 `placed_all && no_violations` |
| A-14 | "未布线网络数"观察无效 | 已移除（layout 阶段此值恒为总数，无信息量） |
| A-15 | UTF-8 乱码 | 所有文件重新保存为 UTF-8 |

---

## 2. 核心数据结构

所有数据结构统一在 `utils/kicad_utils/kicad_utils.py` 中定义，通过 `KiCadUtils` 的 `_cache_components()` 和 `_cache_nets()` 从 KiCad pcbnew API 填充。

### 2.1 Component（元器件）

```python
@dataclass
class Component:
    id: str                              # 参考编号（如 R1、U2）
    footprint: str                       # 封装名称
    package_type: int                    # 封装类型编码 0-4（见 §2.1.1）
    bbox: Tuple[float, float, float, float]  # 当前位置 AABB (x1, y1, x2, y2) mm
    rotation: int                        # 0=0°, 1=90°, 2=180°, 3=270°
    layer: int                           # 0=TOP, 1=BOTTOM
    is_fixed: bool                       # 是否固定（不可移动）
    is_placed: bool                      # 是否已放置
    nets: List[str]                      # 该元件连接的网络 ID 列表
    pins: List[Tuple[str, float, float]] # [(pin_number, rel_x, rel_y), ...] 相对【源姿态】bbox 左上角的引脚坐标
    footprint_w: float                   # 源姿态下的 AABB 宽度 mm（非 0° 尺寸）
    footprint_h: float                   # 源姿态下的 AABB 高度 mm
    source_rotation: int                 # 源文件中的旋转（0-3），footprint_w/h 的测量姿态
    source_bbox: Tuple[float, float, float, float]  # 源姿态 AABB（导入时，不可变）
    pin_offsets_by_rotation: Dict[int, List[Tuple[float, float]]]  # r∈0..3 各 pad 相对该旋转 bbox 中心的偏移

    @property
    def footprint_area(self) -> float:
        """封装面积 mm²（源姿态下的面积，旋转不改变面积）"""
        return self.footprint_w * self.footprint_h

    @property
    def center(self) -> Tuple[float, float]:
        """从 AABB 推导中心点（mm）"""
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    def dims_at_rotation(self, target_rotation: int) -> Tuple[float, float]:
        """目标旋转下的 AABB 宽高：相对 source_rotation 的 delta 为奇数时交换。"""
        delta = (target_rotation - self.source_rotation) % 4
        if delta % 2 == 0:
            return self.footprint_w, self.footprint_h
        return self.footprint_h, self.footprint_w

    def pin_abs_positions(self, rotation: int = None) -> List[Tuple[str, float, float]]:
        """指定（或当前）旋转下各引脚的绝对坐标：bbox 中心 + 预计算偏移。"""
        if rotation is None:
            rotation = self.rotation
        cx, cy = self.center
        offsets = self.pin_offsets_by_rotation.get(rotation, [])
        return [(num, cx + ox, cy + oy)
                for (num, _rel_x, _rel_y), (ox, oy) in zip(self.pins, offsets)]
```

#### 2.1.1 封装类型编码

| 编码 | 封装类型 | 来源 KiCad footprint 名称 |
|------|----------|--------------------------|
| 0 | Chip (0402/0603/0805 等) | 含 `0402`/`0603`/`0805` |
| 1 | SOIC / SOT | 含 `SOIC`/`SOT` |
| 2 | QFP | 含 `QFP`/`TQFP`/`LQFP` |
| 3 | BGA | 含 `BGA` |
| 4 | IC 通用 | 其他 IC 封装（DIP/SIP 等）及无法归类的元件 |

编码规则：遍历 `footprint` 字符串匹配关键字，未命中默认 = 4。

### 2.2 Net（网络）

```python
@dataclass
class Net:
    net_id: str                          # 网络名称（如 VCC、GND）
    connections: List[Tuple[str, str]]   # [(comp_id, pin_number), ...]
    is_power: bool                       # 是否电源网络
```

电源网络判定：`net_id` 含 `VCC`/`VDD`/`VSS`/`GND`/`PWR`（大小写不敏感）→ `is_power = True`。

### 2.3 Board（板框）

板框简化为两个矩形元组，不定义独立 dataclass。原 `BoardBorder` 及其 Line/Circle/Arc/Polygon 细节仅 KiCad 渲染时需要，RL 环境只需要边界矩形：

```python
# KiCadUtils 属性
board_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
# ((min_x, min_y), (max_x, max_y)) mm

# PCBLayoutEnv 属性（reset 时由 board_bounds 内缩生成）
valid_area: Tuple[float, float, float, float]
# (x_min, y_min, x_max, y_max) mm，board_bounds 内缩 BOARD_MARGIN(1mm) 后
```

**简化代价**：非矩形 PCB（L 形、圆形、有挖空）出界检测可能误判。当前阶段接受此简化，后续可升级为多边形精确检测。

### 2.4 全局变量（env 层，不进入 observation）

以下变量在 `PCBLayoutEnv` 级别维护，用于 reward 计算和约束检测，Agent 不可见：

| 变量 | 类型 | 来源 | 用途 |
|------|------|------|------|
| `movable_list` | `List[Component]` | reset 中从 is_fixed 过滤 | C1/C5 联动 |
| `fixed_list` | `List[Component]` | reset 中从 is_fixed 过滤 | C2 碰撞检测范围 |
| `placed_list` | `List[str]` | step 中增量维护 | C1 追踪 |
| `nets` | `Dict[str, Net]` | KiCadUtils.nets | net_aggregation、去耦电容 |
| `decoupling_map` | `Dict[str, List[str]]` | reset 中构建 | C6 去耦电容奖励 |
| `area_ranks` | `Dict[str, int]` | reset 中按 footprint_area 降序排列 | 大元件排名奖励 |
| `board_bounds` | 见 §2.3 | KiCadUtils | 板框几何 |
| `valid_area` | 见 §2.3 | board_bounds 内缩 | C3 出界检测 |

### 2.5 KiCadUtils 接口

> **注意**：以下仅定义 RL 层依赖的**对外接口签名**。`KiCadUtils` 内部的 `load_board()`、`_cache_components()`、`_cache_nets()` 等 KiCad 导入逻辑由其他人负责实现，此处不规定细节——只要最终输出符合 `Component` / `Net` 数据结构即可。

```python
class KiCadUtils:
    """KiCad PCB 数据层封装。导入逻辑由他人实现，RL 层仅依赖以下接口。"""

    # ========== 属性（导入后可用） ==========
    components: Dict[str, Component]  # ref -> Component
    nets: Dict[str, Net]              # net_id -> Net
    board_bounds: Tuple[Tuple[float, float], Tuple[float, float]]

    # ========== 对外 getter ==========
    def load_board(self, pcb_path: str) -> bool: ...
    """加载 PCB 文件，失败返回 False。调用后 components/nets/board_bounds 可用。"""

    def get_board_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]: ...
    """返回 ((min_x, min_y), (max_x, max_y)) mm。"""

    def get_keepout_zones(self) -> List[Tuple[float, float, float, float]]: ...
    """从 KiCad Zone / Rule Area 读取禁止区域。
       返回 List[(x1, y1, x2, y2)] AABB 列表。
       遍历 fp.Zones() → zone.GetIsRuleArea() → 提取包围盒。
    """
```

### 2.6 PCBUtils 接口

`PCBUtils` 是 env 层纯计算工具，**不持有 KiCad 依赖**，所有数据由 `PCBLayoutEnv` 传入。负责约束检测和 reward 各分项计算。

```python
class PCBUtils:
    """PCB RL 环境专用工具类（纯计算，无 KiCad 依赖）。"""

    # ========== C2：AABB 碰撞检测 ==========
    @staticmethod
    def bbox_overlap_penalty(
        a: Tuple[float, float, float, float],   # (x1,y1,x2,y2)
        b: Tuple[float, float, float, float],
        spacing_threshold: float                # 分类间距阈值 (mm)，由调用方查表传入
    ) -> Tuple[float, float]:
        """
        返回 (hard_overlap_area_mm2, soft_spacing_penalty)。
        hard_overlap > 0: 物理交叉面积
        soft_spacing > 0: gap < spacing_threshold 的连续惩罚值
        见 §6.3.2 详细算法。
        """
        ...

    # ========== C3：板框约束 ==========
    @staticmethod
    def out_of_bounds_area(
        bbox: Tuple[float, float, float, float],
        valid_area: Tuple[float, float, float, float]
    ) -> float:
        """
        返回 bbox 超出 valid_area 的面积 (mm²)。
        见 §6.4.1 详细算法。
        """
        ...

    @staticmethod
    def clamp_to_board(
        x1: float, y1: float, x2: float, y2: float,
        valid_area: Tuple[float, float, float, float],
        margin: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        推理阶段：将 AABB clamp 至 valid_area 内（保留 margin 裕度）。
        保持宽高不变，仅平移。
        见 §4.4 详细算法。
        """
        ...

    # ========== C6：Net 聚合度 ==========
    @staticmethod
    def net_aggregation(
        comp_bboxes: List[Tuple[float, float, float, float]]
    ) -> float:
        """
        comp_bboxes: 同一 net 内所有已放置元件的 AABB 列表。
        返回各元件中心到 net 中心的平均距离 (mm)，越小越好。
        内部从 AABB 推导中心点：cx=(x1+x2)/2, cy=(y1+y2)/2。
        见 §6.6.4 详细算法。
        """
        ...

    # ========== C6：去耦电容 ==========
    @staticmethod
    def identify_decoupling_pairs(
        nets: Dict[str, Net],
        components: Dict[str, Component]
    ) -> Dict[str, List[str]]:
        """
        识别去耦电容配对关系。
        遍历 nets → IC 含 VCC/VDD 引脚 → 同网络查找 MLCC 电容 → 建立映射。
        返回 {ic_id: [cap_id, ...]}。
        见 §6.6.1 识别规则。
        """
        ...

    @staticmethod
    def min_pin_to_cap_distance(
        ic: Component,
        cap: Component
    ) -> float:
        """
        计算电容到 IC 电源引脚组的最小欧氏距离 (mm)。
        从 ic.pins (电容相对坐标) 和 cap.center 计算。
        """
        ...

    # ========== S1：拥塞密度 ==========
    @staticmethod
    def congestion_std(
        grid_cells: List[Tuple[float, float, float, float]],  # 每个格子的 (x1,y1,x2,y2)
        placed_components: List[Component],
        nets: Dict[str, Net],
        cell_size: float
    ) -> float:
        """
        将板面划分为 cell_size × cell_size 网格，
        统计每个格子中涉及的 net 数量（通过 AABB 与格子交集判断），
        返回所有格子的 net 计数标准差。越小越均匀。
        见 §8.1。
        """
        ...

    # ========== S3：大元件排名 ==========
    @staticmethod
    def build_area_ranks(
        movable_list: List[Component]
    ) -> Dict[str, int]:
        """
        按 footprint_area 降序排列 movable 元件。
        返回 {comp_id: rank}，rank 从 1 开始（最大元件 rank=1）。
        见 §7.2。
        """
        ...

    # ========== 禁止区域 ==========
    @staticmethod
    def check_keepout(
        bbox: Tuple[float, float, float, float],
        keepout_zones: List[Tuple[float, float, float, float]]
    ) -> bool:
        """
        检查 bbox 是否与任何 keepout zone 有交集。
        返回 True = 违规（进入禁止区域）。
        """
        ...
```

### 2.4 全局变量（env 层，不进入 observation）

以下变量在 `PCBLayoutEnv` 级别维护，用于 reward 计算和约束检测，Agent 不可见：

| 变量 | 类型 | 来源 | 用途 |
|------|------|------|------|
| `movable_list` | `List[Component]` | reset 中从 is_fixed 过滤 | C1/C5 联动 |
| `fixed_list` | `List[Component]` | reset 中从 is_fixed 过滤 | C2 碰撞检测范围 |
| `placed_list` | `List[str]` | step 中增量维护 | C1 追踪 |
| `nets` | `Dict[str, Net]` | KiCadUtils.nets | net_aggregation、去耦电容 |
| `decoupling_map` | `Dict[str, List[str]]` | reset 中构建 | C6 去耦电容奖励 |
| `area_ranks` | `Dict[str, int]` | reset 中按 footprint_area 降序排列 | 大元件排名奖励 |
| `board_bounds` | 见 §2.3 | KiCadUtils | 板框几何 |
| `valid_area` | 见 §2.3 | board_bounds 内缩 | C3 出界检测 |

---

## 3. 状态空间（Observation Space）

### 3.1 设计原则

- **观测 = 当前布局快照**：仅包含已放置元件信息，板框几何、未放置元件列表、net 拓扑作为全局变量不进入 observation
- **单板训练**：每个 PCB 独立训练一个模型，元件 ID 固定，不需要通用类型编码
- **固定维度**：observation 维度在 `reset()` 时确定，在整个 episode 中不变

### 3.2 Fixed-Slot 观测编码

每个可移动元件在观测中占用一个固定 slot（8 维），未放置时填零：

```python
# 每个 slot: [placed_flag, x1, y1, x2, y2, rotation, layer, net_count]
# 总维度 = N_movable × 8

# 示例：N_movable = 5，U1 U3 已放置
observation = [
    1, 10.0, 20.0, 15.0, 25.0, 0, 0, 3,   # U1: placed, (10,20)-(15,25), 0°, TOP, 3 nets
    0, 0, 0, 0, 0, 0, 0, 0,              # U2: 未放置 → 全零
    1, 50.0, 30.0, 65.0, 45.0, 1, 1, 5,   # U3: placed, (50,30)-(65,45), 90°, BOTTOM, 5 nets
    0, 0, 0, 0, 0, 0, 0, 0,              # U4: 未放置 → 全零
    0, 0, 0, 0, 0, 0, 0, 0,              # U5: 未放置 → 全零
]
```

**各字段含义**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `placed_flag` | int | 1=已放置, 0=未放置 |
| `x1, y1` | float | AABB 左下角坐标 (mm) |
| `x2, y2` | float | AABB 右上角坐标 (mm) |
| `rotation` | int | 0/1/2/3 → 0°/90°/180°/270° |
| `layer` | int | 0=TOP, 1=BOTTOM |
| `net_count` | int | 该元件连接的网络数 |

### 3.3 Pending 观测扩展

自回归 5 步决策期间（见 §4），观测向量需额外携带当前已选中的中间状态信息。在原始 fixed-slot 编码（N_movable × 8）后追加固定长度的 pending 槽位：

```python
# pending 槽: [pending_comp_onehot(N_movable), pending_gx1, pending_gy1, pending_rot, pending_lay]
# 未成交的维度填 0，总计追加 N_movable + 4 维
# 各步观测维度统一 = N_movable * 8 + N_movable + 4
```

子步骤对应的 pending 状态：

| step_id | 含义 | pending 中有效字段 |
|:---|:---|:---|
| 1 | 选元件 | 全部清零 |
| 2 | 选 x1 | comp_onehot 已填充 |
| 3 | 选 y1 | comp_onehot + gx1 已填充 |
| 4 | 选旋转 | comp_onehot + gx1 + gy1 已填充 |
| 5 | 选层 | comp_onehot + gx1 + gy1 + rot 已填充 |

### 3.4 动作掩码

动作掩码随子步骤变化，改为每步独立的 **1D bool 向量**：

```python
# Step 1: 元件掩码 shape=(N_movable,)
comp_mask[i] = not movable_list[i].is_placed and not movable_list[i].is_fixed

# Step 2: x1 掩码 shape=(grid_width,)
x1_mask[gx1] = (gx1 + pending_comp.grid_w < grid_width)

# Step 3: y1 掩码 shape=(grid_height,)
y1_mask[gy1] = (gy1 + pending_comp.grid_h < grid_height)

# Step 4-5: 旋转和层无掩码
rot_mask = [True, True, True, True]
lay_mask = [True, True]
```

策略网络 softmax 前对每步对应 Head 输出做：`logits[mask == 0] = -inf`。

---

## 4. 动作空间（Action Space）

### 4.1 设计动机

原 7 元动作 `(comp_id, x1, y1, x2, y2, rot, lay)` 若展平为单一 Discrete，总动作数 = N_movable × GX × GY × GX × GY × 4 × 2。以 100×80mm PCB（CELL_SIZE=1mm, 50 元件）为例：总动作数 ≈ **256 亿**，DQN 输出层内存爆炸且 99.99% 动作为非法。

采用**自回归分解（Autoregressive Decomposition）**：将 1 次 7 元决策拆为 5 步顺序决策。

### 4.2 5 步自回归流程

x2、y2 由元件 footprint 尺寸自动推导（`x2 = gx1 + fp_w/cell_size`），无需 Agent 选择。5 步决策流：

```
Step 1: comp_id = argmax Q_comp(obs)      ← Discrete(N_movable)
Step 2: gx1     = argmax Q_x1(obs)        ← Discrete(grid_width)
Step 3: gy1     = argmax Q_y1(obs)        ← Discrete(grid_height)
Step 4: rot     = argmax Q_rot(obs)       ← Discrete(4)
Step 5: lay     = argmax Q_lay(obs)       ← Discrete(2)

最终动作组装（目标旋转下真实尺寸，网格占用向上取整 ceil）：
    w, h = comp.dims_at_rotation(rot)                 # 宽高随旋转交换
    gx2 = gx1 + ceil(w / CELL_SIZE)
    gy2 = gy1 + ceil(h / CELL_SIZE)
    action = (comp_id, gx1, gy1, gx2, gy2, rot, lay)  → env.step()
```

每步为一个独立的小 Discrete 空间，DQN 原生兼容。

### 4.3 每步动作空间

| step_id | 含义 | 动作空间 | 典型维度 | 掩码 |
|:---|:---|:---|:---:|:---|
| 1 | 选元件 | `Discrete(N_movable)` | ~50 | 已放置/固定 → mask=False |
| 2 | 选 x1 网格 | `Discrete(grid_width)` | ~100 | gx1+fp_w ≥ grid_width → mask=False |
| 3 | 选 y1 网格 | `Discrete(grid_height)` | ~80 | gy1+fp_h ≥ grid_height → mask=False |
| 4 | 选旋转 | `Discrete(4)` | 4 | 无掩码 |
| 5 | 选层 | `Discrete(2)` | 2 | 无掩码 |

每步掩码为 1D bool 向量（详见 §3.4），内存 < 1KB。

### 4.4 DQN 网络结构

```python
class DQN(nn.Module):
    """5-Head 自回归 DQN。Shared Body + 5 个独立输出 Head。"""

    def __init__(self, obs_dim, n_movable, grid_w, grid_h):
        # 共享 Body：3 层 MLP
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # 5 个独立 Head（每步一个）
        self.head_comp = nn.Linear(128, n_movable)   # Step 1
        self.head_x1   = nn.Linear(128, grid_w)       # Step 2
        self.head_y1   = nn.Linear(128, grid_h)       # Step 3
        self.head_rot  = nn.Linear(128, 4)            # Step 4
        self.head_lay  = nn.Linear(128, 2)            # Step 5

    def forward(self, obs, step_id: int):
        """step_id: 0=comp, 1=x1, 2=y1, 3=rot, 4=lay"""
        x = self.shared(obs)
        return self._heads[step_id](x)
```

每次 `forward()` 只走一个 Head，其他 4 个 Head 不在计算图上。反向传播时只更新 shared Body + 当前 Head，其余 Head 梯度为 None。

### 4.5 Reward 分配

5 步共享同一个最终 reward（delayed reward）：

```python
# 采样阶段：暂存 transition，不回填 reward
transitions_5 = []
for step_id in ['comp', 'x1', 'y1', 'rot', 'lay']:
    s = build_obs_with_pending(step_id)
    mask = build_mask(step_id)
    a = epsilon_greedy(dqn, s, step_id, mask)
    transitions_5.append((s, a, step_id, mask))
    update_pending(a, step_id)

# 5 步结束，执行最终放置
final_action = assemble_action()
obs_next, reward, terminated, truncated, _ = env.step(final_action)

# 5 条 transition：仅 step 4（最终步）获得环境 reward，中间步 reward=0
# 原因：若中间步也获得 reward，Bellman 方程中 γ 会倍乘导致训练不稳定
for i, (s, a, step_id, mask) in enumerate(transitions_5):
    step_reward = reward if i == 4 else 0.0
    replay_buffer.add(s, a, step_reward, mask, ..., step_id, next_step_id)
```

- Step 1~4 的 `done=False`（下个状态是下一个子步骤）
- Step 5 的 `done` = 环境返回的 `terminated || truncated`

### 4.6 网格离散化

网格单元大小由 `CELL_SIZE` 配置（默认 1.0mm），网格维度由 `valid_area` 范围决定：

```python
grid_width  = int((valid_area.x_max - valid_area.x_min) / CELL_SIZE)
grid_height = int((valid_area.y_max - valid_area.y_min) / CELL_SIZE)
```

坐标还原公式：
```python
x = valid_area.x_min + grid_index * CELL_SIZE
```

### 4.7 推理阶段 clamp

```python
def clamp_to_board(x1, y1, x2, y2, valid_area, margin=0.0):
    """
    将 AABB clamp 至 valid_area 内（进一步保留 margin 裕度）。
    推理阶段兜底，防止模型输出出界坐标。
    """
    w, h = x2 - x1, y2 - y1
    x1 = max(valid_area["x_min"] + margin,
             min(x1, valid_area["x_max"] - w - margin))
    y1 = max(valid_area["y_min"] + margin,
             min(y1, valid_area["y_max"] - h - margin))
    return x1, y1, x1 + w, y1 + h
```

### 4.8 训练频率

DQN off-policy 在线学习：每个 env step 后，若 replay buffer 样本数 ≥ BATCH_SIZE，
采样一个 mini-batch 进行一次 BP。不再等待 episode 结束：

```
Episode:
  ├── 放元件1: 采样 5 条 → buffer → buffer 够 batch? → learn()
  ├── 放元件2: 采样 5 条 → buffer → buffer 够 batch? → learn()
  ├── ...
  └── 结束
```

### 4.9 与 7 元 Tuple 方案对比

| 对比项 | 原 7 元 Tuple | 自回归 5 步 |
|:---|:---|:---|
| 总动作数（展平） | ~256 亿 | 每步 ≤100 |
| 动作空间类型 | `Tuple[7×Discrete]` | 每步 `Discrete(K)` |
| DQN 兼容 | ❌ | ✅ |
| 掩码内存 | ~25.6 GB | < 1 KB |
| 输出 Head 数 | 1 个 256 亿维 | 5 个 ≤100 维 |
| 每颗元件产生 transition | 1 条 | 5 条 |
| 共享 Reward | 不需要 | 5 步共享同 1 个 reward |

---

## 5. 生命周期

### 5.1 reset()

```
reset():
    1. 通过 kicad.get_board_bounds() 获取板框
    2. valid_area = board 内缩 BOARD_MARGIN(1mm)
    3. 读取所有元件，过滤 Component.is_fixed：
       - fixed_list   = [comp | comp.is_fixed == True]
       - movable_list = [comp | comp.is_fixed == False]
    4. 固定元件：直接使用 KiCad 预置位置
    5. 可移动元件：不设置初始位置（未放置 = observation slot 全零）
    6. placed_list = []
    7. 构建 network topology 缓存（去耦电容 map、net 成员关系等）
    8. 返回 observation（全零 slots + action_mask）
```

### 5.2 step(action)

```
step(action):
    1. 解析 action (component_id, x1, y1, x2, y2, rotation, layer)
       （x2/y2 由 5 步自回归流程 §4.3 自动推导并组装）
    2. C1-2 检查：component_id ∉ placed_list，已放置则拒绝操作
    3. AABB 尺寸校验：根据元件 footprint+rotation 验证 (x2-x1, y2-y1) 是否匹配
    4. 将元件放置到指定位置（更新内部状态）
    5. 计算约束违规量（§6）
    6. 计算 reward（§7）
    7. 如果 violation_count == 0：placed_list.append(component_id)
    8. 判定 done（§5.3）
    9. 构建 action_mask（§3.4）
    10. 返回 (obs, reward, done, truncated, info)
```

### 5.3 终止条件

```python
# 正常完成：所有可移动元件已放置且无违规
done_success = (len(placed_list) == len(movable_list)) and (total_violations == 0)

# 步数上限
done_max_steps = (current_step >= max_steps)

# 早停：累计 reward 低于阈值
done_early_stop = (cumulative_reward < EARLY_STOP_THRESHOLD)  # 默认 -500

done = done_success or done_max_steps or done_early_stop
```

---

## 6. 约束体系

### 6.1 C0：初始状态

| 子项 | 要求 |
|------|------|
| C0-1 | reset() 中可移动元件不设置初始位置，observation slots 全零 |
| C0-2 | fixed_list 元件直接使用 KiCad 预置位置，进入 observation 对应 slot |

### 6.2 C1：元件锁定

| 子项 | 要求 |
|------|------|
| C1-1 | reset() 中初始化 `placed_list = []` |
| C1-2 | step() 中：`component_id ∉ placed_list` 才允许操作 |
| C1-3 | 终止条件：`len(placed_list) == len(movable_list) && violations == 0` |

`placed_list` 与 `movable_list`（C5-2）统一管理，三处引用同一列表。

### 6.3 C2：元件不重叠

#### 6.3.1 碰撞检测方法

使用 AABB 包围盒检测，**不引入多边形轮廓**。获取包围盒：

```python
bbox = fp.GetBoundingBox()  # → (x_min, y_min, x_max, y_max)
```

#### 6.3.2 bbox_overlap_penalty

```python
def bbox_overlap_penalty(a, b, spacing_threshold):
    """
    参数:
        a, b: (x1, y1, x2, y2) AABB 包围盒
        spacing_threshold: 分类间距阈值 (mm)，见 §6.3.4
    返回:
        (hard_overlap, soft_spacing)
        - hard_overlap: 硬重叠面积 (mm²)
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

    if gap >= spacing_threshold:
        return (0.0, 0.0)            # 间距 ≥ 分类阈值，不惩罚
    return (0.0, (spacing_threshold - gap) ** 2 * edge)
```

**注意**：`spacing_threshold` 由两个元件的封装类型共同决定（取较大值），不再是全局常量 `KEEPOUT_SOFT`。

#### 6.3.3 分层惩罚权重

| 类型 | 条件 | 权重 |
|------|------|------|
| 硬重叠 | `hard_overlap > 0` | `W_OVERLAP_HARD = 1000.0` |
| 软间距 | `hard_overlap == 0 && soft_spacing > 0` | `W_OVERLAP_SOFT = 50.0` |
| 不惩罚 | `soft_spacing == 0` | 0 |

#### 6.3.4 分类间距阈值

两元件间距阈值取两者封装类型对应阈值的**最大值**：

```python
SPACING_THRESHOLDS = [0.2, 1.25, 2.5, 5.0, 2.0]
threshold = max(SPACING_THRESHOLDS[type_a], SPACING_THRESHOLDS[type_b])
```

| 封装编码 | 封装类型 | 间距 (mm) |
|----------|----------|-----------|
| 0 | Chip (0402/0603/0805) | 0.2 |
| 1 | SOIC / SOT | 1.25 |
| 2 | QFP | 2.5 |
| 3 | BGA | 5.0 |
| 4 | IC 通用 | 2.0 |

默认阈值 0.25mm（IPC-7351B Level B Nominal）。

### 6.4 C3：板框约束

#### 6.4.1 out_of_bounds_area

```python
def out_of_bounds_area(bbox, valid_area):
    """
    参数:
        bbox: (x1, y1, x2, y2)
        valid_area: (x_min, y_min, x_max, y_max)
    返回:
        出界面积 (mm²)
    """
    bxmin, bymin, bxmax, bymax = bbox
    oxmin, oymin, oxmax, oymax = valid_area

    bbox_area = max(0.0, bxmax - bxmin) * max(0.0, bymax - bymin)
    ix = max(0.0, min(bxmax, oxmax) - max(bxmin, oxmin))
    iy = max(0.0, min(bymax, oymax) - max(bymin, oymin))

    return bbox_area - ix * iy
```

#### 6.4.2 板边安全裕度

板框内缩 **BOARD_MARGIN = 1.0mm** 后为 `valid_area`（IPC-2221A 0.5mm 与制造商推荐 1mm 之间折中）。

#### 6.4.3 出界控制

- 训练阶段：跨 episode curriculum 衰减出界惩罚权重（§10.2）
- 推理阶段：clamp_to_board（§4.4）

### 6.5 C5：固定元件

#### 6.5.1 固定元件识别

使用 KiCad `fp.IsLocked()` → `Component.is_fixed`，不再依赖 JSON：

```python
self.movable_list = [c for c in self.kicad.components.values() if not c.is_fixed]
self.fixed_list   = [c for c in self.kicad.components.values() if c.is_fixed]
```

#### 6.5.2 联动

`movable_list` 同时用于 C1-2（已放置检查）和 C1-3（终止判定），确保引用同一列表。

### 6.6 C6：去耦电容 + Net 聚合度

#### 6.6.1 去耦电容识别

遍历 netlist → IC 含 VCC/VDD 引脚 → 同一网络上查找 MLCC 电容（value/footprint 含 "C"）→ 建立 `decoupling_map = {ic_ref: [cap_ref, ...]}`。

#### 6.6.2 去耦电容距离检查

对每个 `(ic, cap)` 配对，电容到 IC 电源引脚组最小距离 > `DECOUPLING_MAX_DIST = 5.0mm` → 违规。

#### 6.6.3 去耦电容距离奖励

```python
for cap in placed_caps:
    dist = min_pin_to_cap_distance(cap.target_ic, cap)
    reward += max(0, 1 - dist / DECOUPLING_MAX_DIST) * W_DECOUPLING
```

#### 6.6.4 Net 聚合度

对同一 net 的所有已放置元器件（AABB），取中心点 `cx=(x1+x2)/2, cy=(y1+y2)/2`，定义聚合度为各元件中心到 net 中心的平均距离：

```python
def net_aggregation(comp_bboxes):
    """comp_bboxes: list of (x1, y1, x2, y2)"""
    if len(comp_bboxes) < 2:
        return 0.0
    # 从 AABB 推导中心点
    cxs = [(b[0] + b[2]) / 2 for b in comp_bboxes]
    cys = [(b[1] + b[3]) / 2 for b in comp_bboxes]
    cx, cy = sum(cxs) / len(cxs), sum(cys) / len(cys)
    dists = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in zip(cxs, cys)]
    return sum(dists) / len(dists)
```

聚合度越小 → 元件越紧密 → 奖励越大。

---

## 7. Reward 函数

### 7.1 完整公式

```python
reward = 0.0

# ========== 惩罚项 ==========

# 硬重叠（包含与固定元件的碰撞检测）
reward -= W_OVERLAP_HARD   * total_hard_overlap        # 权重 1000.0

# 软间距（分类阈值，见 §6.3.4）
reward -= W_OVERLAP_SOFT   * total_soft_spacing         # 权重 50.0

# 出界面积（curriculum 权重，见 §10.2）
reward -= W_OUT_OF_BOUNDS_CURRENT  * total_out_of_bounds_area

# ========== 奖励项 ==========

# Net 聚合度（各 net 内元件到中心平均距离，越小越好）
for net_id, comps in net_groups.items():
    agg = net_aggregation(comps)
    reward += max(0, 1 - agg / BOARD_DIAGONAL) * W_NET_AGGREGATION

# 拥塞均衡度（网格中 net 数分布标准差，越均匀越好）
congestion_std = self.congestion_map.standard_deviation()
reward += max(0, 1 - congestion_std / MAX_CONGESTION_STD) * W_CONGESTION

# 去耦电容距离（见 §6.6.3）
for cap in placed_caps:
    dist = min_pin_to_cap_distance(cap.target_ic, cap)
    reward += max(0, 1 - dist / DECOUPLING_MAX_DIST) * W_DECOUPLING

# 大元件优先放置（按面积排名，先放大元件奖励更高）
rank = area_rank[component_id]  # 1 = 最大, N = 最小
reward += max(0, 1 - rank / len(movable_list)) * W_LARGE_COMP

# 完成奖励
if done_success:
    reward += W_COMPLETION
```

### 7.2 大元件放置排名奖励

```python
# reset() 时按面积降序排列
self.area_ranks = {
    comp_id: rank
    for rank, comp_id in enumerate(
        sorted(movable_list, key=lambda c: c.footprint_area, reverse=True),
        start=1
    )
}
# 第一名（最大元件）= 1，最后一名 = N
# 奖励 = (1 - rank / N) * W_LARGE_COMP
# 最大元件得 ~W_LARGE_COMP，最小元件得 ~0
```

---

## 8. 辅助系统

### 8.1 S1：拥塞密度控制

> ⚠️ **纳入 Reward**：拥塞均衡度已纳入 §7.1 的 reward 公式。通过
> `PCBUtils.congestion_std(placed_components, nets, cell_size)` 静态方法实现，
> 每次放置后从已放置元件集合重新计算网格 net 计数标准差。

- `congestion_std()`：静态方法，O(components × cells) 全量计算
- 未采用增量式 `CongestionMap` 类，简化实现
- 归一化上界使用 `sqrt(N_movable)`

---

## 9. 常量表

### 9.1 Reward 权重

| 常量 | 值 | 说明 |
|------|-----|------|
| `W_OVERLAP_HARD` | 1000.0 | 硬重叠面积惩罚 |
| `W_OVERLAP_SOFT` | 50.0 | 软间距连续惩罚 |
| `W_OUT_OF_BOUNDS_INITIAL` | 100.0 | 出界惩罚初始值（curriculum 起点） |
| `W_OUT_OF_BOUNDS_FINAL` | 1000.0 | 出界惩罚最终值（curriculum 终点） |
| `W_NET_AGGREGATION` | 1.0 | Net 聚合度奖励 |
| `W_CONGESTION` | 1.0 | 拥塞均衡度奖励 |
| `W_DECOUPLING` | 1.0 | 去耦电容距离奖励 |
| `W_LARGE_COMP` | 1.0 | 大元件优先放置奖励 |
| `W_COMPLETION` | 500.0 | 完成全部放置奖励 |

### 9.2 几何阈值

| 常量 | 值 | 说明 |
|------|-----|------|
| `BOARD_MARGIN` | 1.0 mm | 板边安全裕度 |
| `DECOUPLING_MAX_DIST` | 5.0 mm | 去耦电容最大允许距离 |
| `CELL_SIZE` | 1.0 mm | 动作空间网格离散化单元大小 |
| `CONGESTION_CELL_SIZE` | 2.0 mm | 拥塞密度网格单元大小 |
| `DEFAULT_SPACING` | 0.25 mm | 默认元件间距（IPC-7351B Level B） |
| `BOARD_DIAGONAL` | — | 板框对角线长度（由实际 PCB 决定，用于归一化 net 聚合度） |

### 9.3 分类间距阈值

| 封装编码 | 封装类型 | 间距 (mm) |
|----------|----------|-----------|
| 0 | Chip (0402/0603/0805) | 0.2 |
| 1 | SOIC / SOT | 1.25 |
| 2 | QFP | 2.5 |
| 3 | BGA | 5.0 |
| 4 | IC 通用 | 2.0 |

### 9.4 训练参数

| 常量 | 值 | 说明 |
|------|-----|------|
| `GAMMA` | 0.99 | 折扣因子（标准 RL γ） |
| `EPSILON_START` | 1.0 | ε-greedy 初始探索率 |
| `EPSILON_END` | 0.01 | ε-greedy 最终探索率 |
| `EPSILON_DECAY_EPISODES` | 10000 | ε 衰减总 episode 数 |
| `MAX_STEPS` | 100 | 单个 episode 最大步数（指环境 step，非子步骤） |
| `EARLY_STOP_THRESHOLD` | -500.0 | 累计 reward 早停阈值 |
| `CURRICULUM_START_EP` | 0 | curriculum 开始 episode |
| `CURRICULUM_END_EP` | 500 | curriculum 结束 episode |
| `BATCH_SIZE` | 64 | DQN 训练 mini-batch 大小 |
| `REPLAY_BUFFER_SIZE` | 10000 | 经验回放缓冲区大小 |
| `LEARNING_RATE` | 1e-4 | Adam 学习率 |
| `TARGET_UPDATE_FREQ` | 100 | target 网络每 N 步同步一次 |

### 9.5 动作空间维度（5 步自回归，见 §4）

| step_id | 含义 | 类型 | 范围 |
|:---|:---|:---|:---|
| `comp` | 选元件 | `Discrete` | `0 ~ N_movable-1` |
| `x1` | 选 x1 网格 | `Discrete` | `0 ~ grid_width-1` |
| `y1` | 选 y1 网格 | `Discrete` | `0 ~ grid_height-1` |
| `rot` | 选旋转 | `Discrete` | `0~3` → `{0°, 90°, 180°, 270°}` |
| `lay` | 选层 | `Discrete` | `0~1` → `{TOP, BOTTOM}` |

x2、y2 自动推导：先取目标旋转下的真实尺寸 `w, h = dims_at_rotation(rot)`（宽高随旋转交换），再按网格占用向上取整：`gx2 = gx1 + ceil(w/CELL_SIZE)`，`gy2 = gy1 + ceil(h/CELL_SIZE)`。碰撞/越界/reward 用**精确 mm 尺寸**（`x2 = x1 + w`），网格占用（action mask / NN 输入）用 ceil，二者分离、不互相污染。

---

## 10. 训练机制

### 10.1 epsilon-greedy 探索衰减

```python
# 跨 episode 线性衰减
decay_rate = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
epsilon = max(EPSILON_END, EPSILON_START - episode_num * decay_rate)

# 动作选择
if random() < epsilon:
    action = random_valid_action()  # 从 action_mask 中随机选合法动作
else:
    action = policy.select_best_action(obs)  # 策略网络最优
```

### 10.2 出界惩罚 Curriculum Learning

```python
# 跨 episode 线性递增
progress = min(1.0, (episode_num - CURRICULUM_START_EP) /
                     (CURRICULUM_END_EP - CURRICULUM_START_EP))
W_OUT_OF_BOUNDS_CURRENT = W_OUT_OF_BOUNDS_INITIAL +
    progress * (W_OUT_OF_BOUNDS_FINAL - W_OUT_OF_BOUNDS_INITIAL)
```

- 前期（episode 少）：出界惩罚轻（~100），允许 Agent 探索放置位置
- 后期（episode 多）：出界惩罚重（~1000），强制 Agent 放置到位

### 10.3 早停机制

```python
if cumulative_reward < EARLY_STOP_THRESHOLD:  # 默认 -500
    done = True
    truncated = True
```

目的：避免模型在 hopeless episode 中浪费训练时间。

---

## 11. 参数管理

### 11.1 argument.txt

所有超参数统一由 `config/argument.txt` 管理，格式如下：

```
# ========== Reward 权重 ==========
W_OVERLAP_HARD = 1000.0
W_OVERLAP_SOFT = 50.0
W_OUT_OF_BOUNDS_INITIAL = 100.0
W_OUT_OF_BOUNDS_FINAL = 1000.0
W_NET_AGGREGATION = 1.0
W_CONGESTION = 1.0
W_DECOUPLING = 1.0
W_LARGE_COMP = 1.0
W_COMPLETION = 500.0

# ========== 训练参数 ==========
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 10000
MAX_STEPS = 100
EARLY_STOP_THRESHOLD = -500.0
CURRICULUM_START_EP = 0
CURRICULUM_END_EP = 500

# ========== 几何阈值 ==========
BOARD_MARGIN = 1.0
DECOUPLING_MAX_DIST = 5.0
CELL_SIZE = 1.0
CONGESTION_CELL_SIZE = 2.0
DEFAULT_SPACING = 0.25

# ========== 分类间距阈值 ==========
SPACING_CHIP = 0.2
SPACING_SOIC = 1.25
SPACING_QFP = 2.5
SPACING_BGA = 5.0
SPACING_IC = 2.0
```

CLI 参数（`main.py`）可覆盖 `argument.txt` 中的值。

---

## 12. 代码注释规范

统一采用 **Google-style Docstring**，不要求 SPEC 约束编号追溯。

### 12.1 模块 docstring

```python
"""模块用途简述。

基于 SPEC v2.0。
"""
```

### 12.2 类 docstring

```python
class PCBLayoutEnv(gym.Env):
    """PCB 布局 Gymnasium 环境（AABB-based，纯布局阶段）。"""
```

- 一句话说明职责，放在 `class` 下一行。
- 不需要列 Attributes（属性在 `__init__` 中通过行内注释说明，见 §12.5）。

### 12.3 方法 docstring

```python
def step(self, action: Tuple[int, ...]):
    """执行一步放置操作。

    解析 7 元动作 → 校验合法性 → 约束检测 → 计算 reward → 更新状态。
    （7 元动作由 §4.2 的 5 步自回归流程组装。）

    Args:
        action: 7 元动作 (comp_idx, gx1, gy1, gx2, gy2, rot_idx, lay_idx)

    Returns:
        (obs, reward, terminated, truncated, info) 五元组

    Raises:
        无显式异常，非法动作通过 _invalid_action 内部消化。
    """
```

规则：
- 首行：一句话概括功能。
- 第二段（可选）：多步骤流程简述。
- `Args` / `Returns` / `Raises` 按需出现，不为空写 "None"。
- 类型注解优先放在函数签名中（如 `action: Tuple[int, ...]`），docstring 中可省略类型。
- 无参数或简单函数可仅写一行 docstring：`"""获取电路板边界。"""`

### 12.4 内联注释

```python
# AABB: (x1=left, y1=bottom, x2=right, y2=top)
bbox_mm = (left, bottom, right, top)

# 已放置/固定元件 → 整个 slot 屏蔽
if comp.is_placed or comp.is_fixed:
    mask[slot_idx, ...] = False
```

规则：
- 解释 **为什么** 而非 **做什么**（代码本身已说明做什么）。
- 放在被注释代码的上一行。
- 使用中文，保持简洁。

### 12.5 属性行内注释

```python
self._pcb_path = pcb_path          # 保存路径，供 reset() 重新加载
self.valid_area = (...)            # (x_min, y_min, x_max, y_max) mm，board_bounds 内缩后
self.N_movable = len(...)          # 可移动元件数（C1-2 / 终止判定用）
```

规则：
- `#` 后空一格，用中文说明用途和约束。
- 不强制对齐 `#`。

### 12.6 区块分隔

```python
# ==================== Gymnasium 接口 ====================

# ==================== 内部方法 ====================

# -------------------- 辅助 --------------------
```

- 一级分隔用 `# ====` + 双等号宽分隔 + `====`。
- 二级分隔用 `# ----`。
- 分隔符内文字居中。

### 12.7 TODO / FIXME

```python
# TODO: 初始化 RL Agent（策略网络 + 经验回放）
# FIXME: 拥塞密度上界 MAX_CONGESTION_STD 待校准
```

- 大写标签 + 英文冒号 + 空格 + 中文描述。
- 格式：`# TODO: 具体事项` / `# FIXME: 问题描述`。

---

## 附录：变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-07-25 | 1.0 | 基于 `环境构建约束.md` 初始发布 |
| 2026-07-27 | 2.0 | 全面重构：（1）动作空间扩展为 7 元 AABB；（2）状态空间改为 fixed-slot 编码；（3）删除 S2(HPWL)、S3(放置优先级)；（4）Reward 新增拥塞均衡度+大元件排名+完成奖励；（5）引入 ε-greedy + curriculum 训练机制；（6）新增参数管理体系（argument.txt）；（7）修复 A-1~A-15 所有架构问题 |
| 2026-07-27 | 2.1 | 新增 §12 代码注释规范：统一 Google-style Docstring，含模块/类/方法 docstring、内联注释、属性行内注释、区块分隔、TODO/FIXME 格式 |
| 2026-07-27 | 2.2 | 动作空间重构为自回归 5 步分解（§4）：消除 7D 稠密掩码内存爆炸、动作展平为每步 Discrete、引入 pending 观测扩展（§3.3）、定义 DQN 5-Head 网络结构（§4.4）、5 步共享 reward（§4.5）、修正 GAMMA 0.1→0.99、新增 DQN 训练超参数（BATCH_SIZE/REPLAY_BUFFER_SIZE/LEARNING_RATE/TARGET_UPDATE_FREQ）、训练频率确定为每 episode 结束后**一次 BP** |
