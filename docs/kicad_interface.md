# KiCad 导入导出接口对齐文档

> **文档目的**：定义 KiCad 导入导出模块的双向数据契约，用于 RL 团队与 KiCad 负责同学之间的接口对齐。
> **模块位置**：`utils/kicad_utils/` 是 RL 流程的**唯一数据入口**，所有 PCB 信息的读取和写回均通过该模块完成。
> **维护方**：KiCad 导入导出同学独立负责该模块的实现和维护。
> **日期**：2026-07-27

---

## 1. 概述

### 1.1 模块定位

```
main.py ──▶ run.py ──▶ env/pcb_layout_env.py  (PCBLayoutEnv)
                          └── env/pcb_utils.py       (PCBUtils, 纯计算, 无 KiCad 依赖)
                                └── utils/kicad_utils/kicad_utils.py (KiCadUtils, 唯一 KiCad 依赖点)
```

`utils/kicad_utils/` 是项目与 KiCad 交互的**唯一桥梁**，上层的 `PCBLayoutEnv` 和 `PCBUtils` 不直接调用 `pcbnew` API。

### 1.2 依赖关系

```
utils/kicad_utils/kicad_utils.py
  ├── pcbnew          (KiCad 9.0 Python API)
  ├── numpy           (数值计算)
  └── Python stdlib   (dataclasses, typing, os, sys)
```

### 1.3 双向职责

| 方向 | 接口 | 输入 | 输出 |
|------|------|------|------|
| **导入** | `load_board(pcb_path)` | `.kicad_pcb` 文件路径 | `Component` / `Net` / board_bounds |
| **导出** | `export_layout(output_path, ...)` | RL 优化后的 `Component` 字典 | `.kicad_pcb` 文件 |

---

## 2. 数据结构定义

所有数据结构均为 `@dataclass`，定义在 `utils/kicad_utils/kicad_utils.py` 中。

### 2.1 Component（元器件）

```python
@dataclass
class Component:
    """元器件核心数据结构（AABB-based）"""
    id: str                                          # 参考编号（如 R1、U2）
    footprint: str                                   # 封装名称
    package_type: int                                # 封装类型编码 0-4（见 §2.3）
    bbox: Tuple[float, float, float, float]          # 当前位置 AABB (x1, y1, x2, y2) mm
    rotation: int                                    # 0=0°, 1=90°, 2=180°, 3=270°
    layer: int                                       # 0=TOP("F.Cu"), 1=BOTTOM("B.Cu"), 2=内层
    is_fixed: bool                                   # 是否固定（不可移动）
    is_placed: bool                                  # 是否已放置（初始 = is_fixed）
    nets: List[str]                                  # 连接的 net_id 列表
    pins: List[Tuple[str, float, float]]             # [(pin_number, rel_x, rel_y), ...]
    footprint_w: float                               # 源姿态下的 AABB 宽度 mm（非 0° 尺寸）
    footprint_h: float                               # 源姿态下的 AABB 高度 mm
    source_rotation: int                             # 源文件中的旋转（0-3），footprint_w/h 的测量姿态
    source_bbox: Tuple[float, float, float, float]   # 源姿态 AABB（导入时），与 footprint_w/h 一致（不可变）
    pin_offsets_by_rotation: Dict[int, List[Tuple[float, float]]]  # r∈0..3 各 pad 相对该旋转 bbox 中心的偏移

    @property
    def footprint_area(self) -> float:
        """封装面积 (mm²)（源姿态下的面积，旋转不改变面积）"""
        return self.footprint_w * self.footprint_h

    @property
    def center(self) -> Tuple[float, float]:
        """从 AABB 推导中心点 (mm)"""
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)

    def dims_at_rotation(self, target_rotation: int) -> Tuple[float, float]:
        """目标旋转下的 AABB 宽高：相对 source_rotation 的 delta 为奇数时交换宽高。"""
        delta = (target_rotation - self.source_rotation) % 4
        if delta % 2 == 0:
            return self.footprint_w, self.footprint_h
        return self.footprint_h, self.footprint_w

    def pin_abs_positions(self, rotation: Optional[int] = None) -> List[Tuple[str, float, float]]:
        """指定（或当前）旋转下各引脚的绝对坐标：bbox 中心 + 该旋转预计算偏移。"""
        if rotation is None:
            rotation = self.rotation
        cx, cy = self.center
        offsets = self.pin_offsets_by_rotation.get(rotation, [])
        return [(num, cx + ox, cy + oy)
                for (num, _rel_x, _rel_y), (ox, oy) in zip(self.pins, offsets)]
```

#### 字段导入来源映射

| 字段 | 类型 | 来源 KiCad API | 说明 |
|------|------|---------------|------|
| `id` | `str` | `fp.GetReference()` | 参考编号 |
| `footprint` | `str` | `fp.GetFPID().GetLibItemName()` | 封装库项名称 |
| `package_type` | `int` | `_classify_package_type(footprint)` | 通过关键字匹配（见 §2.3） |
| `bbox` | `(x1,y1,x2,y2)` | `fp.GetBoundingHull().BBox()` + `pcbnew.ToMM()` | 源姿态 AABB 包围盒 mm（含旋转，与导出层一致） |
| `rotation` | `int` | `fp.GetOrientationDegrees() // 90 % 4` | 0/1/2/3 编码 |
| `layer` | `int` | `pcbnew.LayerName(fp.GetLayer())` | "F.Cu"→0, "B.Cu"→1, 其他→2 |
| `is_fixed` | `bool` | `fp.IsLocked()` | 固定元件不可移动 |
| `is_placed` | `bool` | 初始值 = `is_fixed` | 固定元件视为"已放置" |
| `nets` | `List[str]` | 遍历 pad netcode 回填 | 由 `_cache_nets()` 填充 |
| `pins` | `[(pin, rx, ry)]` | `fp.Pads()` → `pad.GetPosition()` | **相对源姿态 AABB 左上角**的偏移 mm |
| `footprint_w` | `float` | `pcbnew.ToMM(hull.GetWidth())` | 源姿态 AABB 宽度（非 0° 尺寸） |
| `footprint_h` | `float` | `pcbnew.ToMM(hull.GetHeight())` | 源姿态 AABB 高度 |
| `source_rotation` | `int` | `fp.GetOrientationDegrees() // 90 % 4` | footprint_w/h 的测量姿态 |
| `pin_offsets_by_rotation` | `dict` | 旋转 footprint 后读 `pad.GetPosition() - hull 中心` | 预计算 4 旋转 pad 偏移 |

**关键约束**：
- `pins` 坐标是**相对源姿态 bbox 左上角**的偏移（源姿态 = 源文件当前旋转），旋转后必须用 `pin_abs_positions(rotation)` 取绝对坐标，**不要用 `bbox[0] + rel_x`**——元件旋转后 bbox 尺寸/位置已变，旧公式会失真。
- `footprint_w` / `footprint_h` 是**源姿态**下的 AABB 尺寸（`GetBoundingHull().BBox()`，含源文件旋转），不是 0° 原始尺寸。目标旋转下的宽高须用 `dims_at_rotation(target_rotation)`：相对 `source_rotation` 的 delta 为奇数（90°/270°）时宽高交换。
- `bbox` 是源姿态 AABB，是**不可变快照**；RL 放置后应更新 `bbox` 为当前坐标，但 `footprint_w/h` / `source_bbox` 保持源姿态不变。
- `nets` 字段在 `_cache_components()` 中初始为空列表，由 `_cache_nets()` 回填。

### 2.2 Net（网络）

```python
@dataclass
class Net:
    """网络数据结构（精简版，无 routing 字段）"""
    net_id: str                                      # 网络名称（如 VCC、GND）
    connections: List[Tuple[str, str]]               # [(comp_id, pin_number), ...]
    is_power: bool                                   # 是否电源网络
```

#### 字段导入来源映射

| 字段 | 类型 | 来源 KiCad API | 说明 |
|------|------|---------------|------|
| `net_id` | `str` | `net.GetNetname()` | 网络名称 |
| `connections` | `[(comp_id, pin)]` | pad netcode 匹配 | 遍历 footprints 匹配 netcode |
| `is_power` | `bool` | `Net.is_power_net(net_name)` | 白名单匹配（见下方） |

电源网络判定（大小写不敏感）：精确匹配 14 项 + 前缀匹配 14 类。

```python
# 精确匹配
_POWER_NET_EXACT = {"VCC", "VDD", "VSS", "GND", "PWR", "VEE", "VPP", "VREF",
                    "AGND", "DGND", "AVCC", "AVDD", "DVCC", "DVDD"}
# 前缀匹配
_POWER_NET_PREFIXES = ("VCC", "VDD", "VSS", "VEE",
                       "+5V", "+3.3V", "+3V3", "+12V", "+1.8V",
                       "VIN", "VBUS", "VBAT", "VDDC", "VDDA",
                       "VCCINT", "VCCIO", "VCCPLL")
```

### 2.3 封装类型编码规则

通过 `_classify_package_type(footprint_name: str) -> int` 根据封装名称关键字匹配：

| 编码 | 封装类型 | 匹配关键字 | 示例 |
|------|---------|-----------|------|
| 0 | Chip | `0402`/`0603`/`0805`/`1206`/`1210`/`2512` | "R_0402_1005Metric" |
| 1 | SOIC/SOT/QFN | `SOIC`/`SOT`/`SSOP`/`TSSOP`/`DFN`/`QFN`/`WSON`/`MSOP` | "SOIC-8_3.9x4.9mm" |
| 2 | QFP | `QFP`/`TQFP`/`LQFP`/`PLCC` | "LQFP-48_7x7mm" |
| 3 | BGA/阵列 | `BGA`/`LGA`/`CSP`/`WLP` | "BGA-256_17x17mm" |
| 4 | IC 通用 | 未匹配任何上述关键字 | "DIP-16" 或其他非标封装 |

**注意**：匹配时 footprint 名称转换为大写后进行关键字查找，未命中默认返回 4。

### 2.4 坐标系统约定

| 属性 | 约定 |
|------|------|
| 坐标系 | 笛卡尔坐标系：x 轴向右，y 轴向上 |
| 单位 | RL 内部统一使用 **mm**（毫米） |
| KiCad 内部单位 | 纳米整数（1 mm = 1,000,000 nm） |
| 转换函数 | `pcbnew.ToMM(nm_value)` / `pcbnew.FromMM(mm_value)` |
| 元件定位 | RL 使用 **AABB 包围盒** `(x1, y1, x2, y2)` |
| KiCad 元件定位 | KiCad 使用**元件中心点** `(cx, cy)` |

**AABB 与中心点互转**：

```python
# AABB → center
cx = (x1 + x2) / 2.0
cy = (y1 + y2) / 2.0

# center → AABB (目标旋转下的宽高，须经 dims_at_rotation 交换)
w, h = comp.dims_at_rotation(comp.rotation)
x1 = cx - w / 2.0
y1 = cy - h / 2.0
x2 = cx + w / 2.0
y2 = cy + h / 2.0
```

**rotation 编码**：

| RL 值 | 角度 | KiCad 度数 |
|-------|------|-----------|
| 0 | 0° | 0 |
| 1 | 90° | 90 |
| 2 | 180° | 180 |
| 3 | 270° | 270 |

**layer 编码**：

| RL 值 | KiCad 层名 | 说明 |
|-------|-----------|------|
| 0 | `F.Cu` | TOP（顶层） |
| 1 | `B.Cu` | BOTTOM（底层） |
| 2 | `In*.Cu` 等 | 内层 / 未知层（当前 RL 不支持多层元件放置，此类元件会被标记但不会参与决策） |

### 2.5 板框边界

```python
# 类型
board_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
# 示例：((0.0, 0.0), (100.0, 80.0)) 表示板框从 (0,0) 到 (100,80) mm
```

### 2.6 禁止区域

```python
# 类型
keepout_zones: List[Tuple[float, float, float, float]]
# 每个元素为 AABB：(x1, y1, x2, y2) mm
```

---

## 3. 导入接口规范

### 3.1 `load_board(pcb_path) -> bool`

```python
def load_board(self, pcb_path: str) -> bool:
    """加载 PCB 文件并缓存所有核心数据。

    依次执行 footprint 缓存、元器件信息提取、网络连接解析和板框边界计算。

    Args:
        pcb_path: PCB 文件路径（.kicad_pcb 格式）

    Returns:
        True 加载成功，False 加载失败
    """
```

**调用时机**：`PCBLayoutEnv.__init__()` 和 `reset()` 中调用。

**内部执行顺序**：

1. `pcbnew.LoadBoard(pcb_path)` 加载板级对象
2. `_cache_footprints()` —— 缓存所有 footprint 对象
3. `_cache_components()` —— 提取 Component 数据结构（nets 字段此时为空）
4. `_cache_nets()` —— 提取 Net 数据结构，并回填 Component.nets
5. `_calc_board_bounds()` —— 计算板框边界

**调用后效果**：`self.components`、`self.nets`、`self.board_bounds` 三个属性立即可用。

### 3.2 `get_board_bounds() -> ((min_x, min_y), (max_x, max_y))`

```python
def get_board_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """获取电路板边界。

    Returns:
        ((min_x, min_y), (max_x, max_y)) 单位 mm
    """
```

**边界计算优先级**：

1. **Edge.Cuts 层**（优先）：遍历该层所有图形（Line / Circle / Arc / Polygon），收集所有顶点坐标求包围盒
2. **元器件边界**（回退）：无 Edge.Cuts 层时，使用所有元器件 AABB 的最小包围盒
3. **默认值**（兜底）：两者都为空时返回 `((0, 0), (100, 80))` mm

**Edge.Cuts 层图形处理**：

| 图形类型 | 提取方式 |
|---------|---------|
| Line | `GetStart()` + `GetEnd()` |
| Circle | `GetCenter()` ± `GetRadius()` |
| Arc | `GetStart()` + `GetEnd()` |
| Polygon | `GetPolyShape()` 遍历所有 `Outline(oi).CPoint(i)` |

### 3.3 `get_keepout_zones() -> List[(x1, y1, x2, y2)]`

```python
def get_keepout_zones(self) -> List[Tuple[float, float, float, float]]:
    """从 KiCad Zone / Rule Area 读取禁止区域。

    遍历 board.Zones()，筛选规则区域（GetIsRuleArea），提取 AABB 包围盒。

    Returns:
        List[(x1, y1, x2, y2)] 禁止区域的 AABB 列表，单位 mm
    """
```

**筛选规则**：`zone.GetIsRuleArea() == True`。

### 3.4 导入流程图

```
load_board(pcb_path)
  │
  ├─ pcbnew.LoadBoard(pcb_path) → self.board
  │
  ├─ _cache_footprints()
  │     └─ board.Footprints() → self.footprints[ref]
  │
  ├─ _cache_components()
  │     ├─ fp.GetReference()              → Component.id
  │     ├─ fp.GetFPID().GetLibItemName()  → Component.footprint
  │     ├─ _classify_package_type()       → Component.package_type
  │     ├─ fp.GetBoundingHull().BBox() + ToMM() → Component.bbox / footprint_w/h
  │     ├─ fp.GetOrientationDegrees()     → Component.rotation / source_rotation
  │     ├─ pcbnew.LayerName(fp.GetLayer()) → Component.layer
  │     ├─ fp.IsLocked()                  → Component.is_fixed
  │     ├─ fp.Pads()                      → Component.pins（相对源姿态 bbox 左上角）
  │     └─ 4 旋转 mutate+restore 预计算    → Component.pin_offsets_by_rotation
  │
  ├─ _cache_nets()
  │     ├─ board.GetNetsByName().values() → 遍历所有网络
  │     ├─ net.GetNetname()               → Net.net_id
  │     ├─ pad.GetNetCode() 匹配          → Net.connections
  │     ├─ Net.is_power_net()             → Net.is_power
  │     └─ 回填 Component.nets
  │
  └─ _calc_board_bounds()
        ├─ Edge.Cuts 层图形              → 优先
        ├─ 元器件包围盒                    → 回退
        └─ 默认值                          → 兜底
```

---

## 4. 导出接口规范（新增，方案 A）

### 4.1 `export_layout(output_path, components, template_path) -> bool`

```python
def export_layout(
    self,
    output_path: str,
    components: Dict[str, Component],
    template_path: Optional[str] = None,
) -> bool:
    """将 RL 优化后的布局导出为 KiCad PCB 文件。

    遍历 components 字典，将每个元件的 AABB 位置、旋转角度、所在层
    通过 pcbnew API 写回 board 对象，最后 Save 到 output_path。

    Args:
        output_path: 输出 PCB 文件路径（.kicad_pcb 格式）
        components: RL 优化后的元件数据，key 为参考编号（如 "R1"）
        template_path: 模板 PCB 文件路径。为 None 时使用 self.board（当前加载的板）。
                       首次调用建议传入模板路径以确保干净的起点。

    Returns:
        True 导出成功，False 导出失败

    Raises:
        不抛出异常，所有错误内部捕获并返回 False。
    """
```

**调用时机**：RL 训练完成或推理完成后，由 `main.py` 或 `Runner` 调用。

### 4.2 导出内部流程

```
export_layout(output_path, components, template_path)
  │
  ├─ 1. 加载模板板
  │     └─ 若 template_path 非空：pcbnew.LoadBoard(template_path)
  │        否则：使用 self.board（需确保已 load_board）
  │
  ├─ 2. 遍历 components 中的每个元件
  │     ├─ 跳过固定元件（is_fixed == True）→ 保持原位置不变
  │     └─ 对可移动元件执行写入：
  │         ├─ AABB → center → pcbnew.VECTOR2I(FromMM(cx), FromMM(cy))
  │         │   → fp.SetPosition(center_nm)
  │         ├─ rotation × 90 → degrees
  │         │   → fp.SetOrientationDegrees(degrees)
  │         └─ layer (0→F.Cu, 1→B.Cu)
  │             → board.GetLayerID(layer_name) → fp.SetLayer(layer_id)
  │
  ├─ 3. board.Save(output_path)
  │
  └─ 4. 返回 True / False
```

### 4.3 字段写入映射表

| RL 字段 | RL 值 | 转换 | KiCad API | KiCad 值 |
|---------|-------|------|-----------|---------|
| `bbox` | `(x1,y1,x2,y2)` mm | `cx=(x1+x2)/2, cy=(y1+y2)/2` | `fp.SetPosition(VECTOR2I(FromMM(cx), FromMM(cy)))` | 中心点 (nm) |
| `rotation` | `0/1/2/3` | `deg = rotation * 90` | `fp.SetOrientationDegrees(deg)` | 度数 (0/90/180/270) |
| `layer` | `0` (TOP) | `"F.Cu"` | `fp.SetLayer(board.GetLayerID("F.Cu"))` | F.Cu 层 ID |
| `layer` | `1` (BOTTOM) | `"B.Cu"` | `fp.SetLayer(board.GetLayerID("B.Cu"))` | B.Cu 层 ID |

### 4.4 footprint 匹配

导出时需要根据 `Component.id`（参考编号）找到对应的 `pcbnew.FOOTPRINT` 对象：

```python
# 方法 1：通过 _cache_footprints 中已有的映射
fp = self.footprints.get(component.id)

# 方法 2：遍历 board.Footprints() 按 reference 匹配
for fp in board.Footprints():
    if fp.GetReference() == component.id:
        # 找到目标 footprint
        break
```

### 4.5 导出流程图

```
┌─────────────┐
│  RL 训练完成  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  KiCadUtils.export_layout(          │
│    output_path,                     │
│    components,   # 修改后的元件      │
│    template_path # 原始 .kicad_pcb  │
│  )                                  │
└──────┬──────────────────────────────┘
       │
       ▼
  加载模板板
       │
       ▼
  for each comp in components:
    ├─ is_fixed? ──── yes ──▶ 跳过
    │
    └─ no:
        ├─ bbox → center → SetPosition()
        ├─ rotation → SetOrientationDegrees()
        └─ layer → SetLayer()
       │
       ▼
  board.Save(output_path)
       │
       ▼
  输出 .kicad_pcb 文件
```

---

## 5. KiCad API 快速参考

### 5.1 常用 pcbnew API

```python
import pcbnew

# ---- 板级操作 ----
board = pcbnew.LoadBoard("path/to/board.kicad_pcb")   # 加载 PCB
board.Save("path/to/output.kicad_pcb")                  # 保存 PCB
board.Footprints()                                       # 获取所有 footprint
board.GetNetsByName()                                    # 获取所有网络 (name → NETINFO_ITEM)
board.GetDrawings()                                      # 获取所有图形元素
board.Zones()                                            # 获取所有 Zone（含 keepout）
board.GetLayerID("F.Cu")                                 # 获取层 ID

# ---- Footprint 操作 ----
fp = board.Footprints()[0]                               # 获取第一个 footprint
fp.GetReference()                                        # 参考编号 "R1"
fp.GetFPID().GetLibItemName()                            # 封装名称
fp.GetBoundingBox()                                      # 返回 pcbnew.BOX2I
fp.GetPosition()                                         # 返回 pcbnew.VECTOR2I (nm)
fp.GetOrientationDegrees()                               # 返回旋转角度 (度)
fp.GetLayer()                                            # 返回层 ID
fp.IsLocked()                                            # 是否锁定
fp.Pads()                                                # 获取所有焊盘

# Footprint 写入
fp.SetPosition(pcbnew.VECTOR2I(x_nm, y_nm))              # 设置位置
fp.SetOrientationDegrees(degrees)                        # 设置旋转
fp.SetLayer(layer_id)                                    # 设置所在层

# ---- Pad 操作 ----
pad = fp.Pads()[0]                                       # 获取第一个焊盘
pad.GetNumber()                                          # 焊盘编号 "1"
pad.GetPosition()                                        # 焊盘位置 (nm)
pad.GetNetCode()                                         # 网络代码

# ---- 图形元素 ----
drawing = board.GetDrawings()[0]
drawing.GetLayer()                                       # 图形所在层 ID
drawing.GetShapeStr()                                    # 图形类型 "Line"/"Circle"/"Arc"/"Polygon"

# ---- 单位转换 ----
pcbnew.ToMM(nm_value)                                    # nm → mm
pcbnew.FromMM(mm_value)                                  # mm → nm

# ---- 层名 ----
pcbnew.LayerName(layer_id)                               # 层 ID → 层名
```

### 5.2 BOX2I 常用方法

```python
bbox = fp.GetBoundingBox()
bbox.GetLeft()     # 左边界 (nm)
bbox.GetRight()    # 右边界 (nm)
bbox.GetTop()      # 上边界 (nm)
bbox.GetBottom()   # 下边界 (nm)
bbox.GetWidth()    # 宽度 (nm)
bbox.GetHeight()   # 高度 (nm)
```

---

## 6. 实现约束与注意事项

### 6.1 固定元件处理

- **导入**：`is_fixed = fp.IsLocked()`，固定元件初始 `is_placed = True`。
- **导出**：`is_fixed == True` 的元件**不修改位置、旋转、层**，直接从模板板保留原值。
- **RL 层依赖**：`PCBLayoutEnv` 通过 `is_fixed` 过滤出 `movable_list`，固定元件参与 C2 碰撞检测但不参与放置决策。

### 6.2 AABB 与中心点转换

导出时将 RL 的 AABB 坐标转为 KiCad 中心点：

```python
# RL AABB → KiCad center
cx_mm = (bbox[0] + bbox[2]) / 2.0
cy_mm = (bbox[1] + bbox[3]) / 2.0
center_nm = pcbnew.VECTOR2I(
    pcbnew.FromMM(cx_mm),
    pcbnew.FromMM(cy_mm),
)
fp.SetPosition(center_nm)
```

### 6.3 错误处理约定

- **导入**：`load_board()` 返回 `bool`，异常时打印错误信息并返回 `False`。
- **导出**：`export_layout()` 返回 `bool`，所有错误内部 catch 并返回 `False`，不抛出异常到调用方。
- **日志**：使用 `print()` 输出关键操作状态，格式统一为 `✅ 成功信息` / `❌ 失败信息`。

### 6.4 性能建议

- `_cache_nets()` 中遍历所有 footprints 的所有 pads 来匹配 netcode，是 O(N_footprints × N_pads) 的复杂度。对于大型 PCB 可能需要优化索引结构。
- `_cache_components()` 中 `_classify_package_type()` 是简单的字符串匹配，不会成为瓶颈。
- 导出时 `board.Save()` 直接调用 pcbnew 的 C++ 层，性能取决于 KiCad 自身实现。

### 6.5 KiCad 版本依赖

- KiCad 路径通过环境变量 `KICAD_PATH` 配置，**不设默认值**（不同开发者路径不同）。
- 若 KiCad 不可用，模块自动降级为空操作并打印警告（Mock 测试仍可正常使用）。
- `pcbnew` Python API 在不同 KiCad 版本间可能存在 API 差异，升级版本时需回归测试导入导出功能。

### 6.6 状态管理约束

- `load_board()` 被 `PCBLayoutEnv.reset()` 调用（每 episode 一次），因此需要确保每次加载时 `self.components` 和 `self.nets` 被完整清空后重建。
- `self.board` 对象在 `load_board()` 后保持有效，导出时可复用。

### 6.7 引脚坐标的相对性

- `Component.pins` 中的 `(rel_x, rel_y)` 是**相对 AABB 左下角**的偏移。
- RL 层 `PCBUtils.min_pin_to_cap_distance()` 通过 `ic.bbox[0] + rel_x, ic.bbox[1] + rel_y` 还原为绝对坐标。
- 当元件 rotation 改变时，引脚相对位置理论上也会旋转，但当前 AABB-based 方案忽略此变化（RL 层通过 AABB 而非多边形做碰撞检测）。

---

## 附录 A：模块文件清单

| 文件 | 职责 | 维护方 |
|------|------|--------|
| `utils/kicad_utils/__init__.py` | 模块导出 `Component`, `Net`, `KiCadUtils` | KiCad 同学 |
| `utils/kicad_utils/kicad_utils.py` | 核心实现：数据加载、缓存、导出 | KiCad 同学 |
| `env/pcb_utils.py` | RL 工具类（纯计算，**不依赖** utils/kicad_utils） | RL 同学 |
| `env/pcb_layout_env.py` | Gymnasium 环境（**仅通过** KiCadUtils 获取数据） | RL 同学 |
| `docs/kicad_interface.md` | 本文档，接口对齐约定 | 双方共同维护 |

## 附录 B：导出方法签名桩

```python
# ==== 在 KiCadUtils 类中新增 ====

def export_layout(
    self,
    output_path: str,
    components: Dict[str, Component],
    template_path: Optional[str] = None,
) -> bool:
    """将 RL 优化后的布局导出为 KiCad PCB 文件。

    遍历 components 字典，将每个非固定元件的 AABB 位置、旋转角度、
    所在层通过 pcbnew API 写回 board 对象，最后 Save 到 output_path。

    固定元件（is_fixed=True）保持模板中的原始位置不变。

    Args:
        output_path: 输出 PCB 文件路径（.kicad_pcb 格式）
        components: RL 优化后的元件数据，key 为参考编号
        template_path: 模板 PCB 文件路径，None 则使用当前加载的 board

    Returns:
        True 导出成功，False 导出失败
    """
    # TODO: 由 KiCad 同学实现
    raise NotImplementedError("export_layout 待实现")
```

## 附录 C：RL 层依赖的 KiCadUtils 完整接口

RL 层 (`PCBLayoutEnv` / `PCBUtils`) **仅通过以下接口**访问 KiCad 数据：

```python
# 属性（load_board 后可用）
kicad.components: Dict[str, Component]
kicad.nets: Dict[str, Net]
kicad.board_bounds: Tuple[Tuple, Tuple]

# 方法
kicad.load_board(pcb_path) -> bool
kicad.get_board_bounds() -> Tuple[Tuple, Tuple]
kicad.get_keepout_zones() -> List[Tuple]
kicad.export_layout(output_path, components, template_path) -> bool  # 新增
```
