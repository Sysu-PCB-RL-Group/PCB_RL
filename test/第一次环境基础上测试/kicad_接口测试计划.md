# KiCad 接口测试计划（第一次环境基础测试）

> **测试目的**：验证 `kicad_utils/` 模块的导入导出功能是否符合 `kicad_interface_完整版.md` 定义的接口契约。
> **测试数据**：`90i0-quan-cai-shi-zhong-ac67228b` 数据集
> **验收标准**：所有测试通过后，生成的 `.kicad_pcb` 能被 KiCad 10.0 正常打开，数据无丢失。

---

## 执行顺序（按优先级分4轮）

### 第一轮：格式基础（必须过——否则 KiCad 打不开）

| ID | 测试 | 测试内容 | 断言 |
|----|------|---------|------|
| **T1** | 无 graphics 包装 | 搜索导出文件中 `(graphics` token | 匹配数 = 0 |
| **T2** | 无 nets 包装 | 搜索导出文件中 `(nets ` token（注意空格，`(net ` 是合法的） | 匹配数 = 0 |
| **T3** | 括号平衡 | 统计 `(` 和 `)` 数量 | 两者相等 |
| **T4** | 文件开头 | 文件是否以 `(kicad_pcb` 开头 | True |
| **T5** | 文件结尾 | 文件是否以 `)` 结尾（最后一个非空字符） | True |
| **T6** | KiCad 能打开 | `pcbnew.LoadBoard()` 加载导出的 PCB 不报异常 | True |

---

### 第二轮：导入功能

| ID | 测试 | 测试内容 | 断言 |
|----|------|---------|------|
| **T7** | load_board 返回值 | 加载原版 `.kicad_pcb` | 返回 True |
| **T8** | 元件数量 | 提取的 Component 数量 | == 47 |
| **T9** | 元件字段非空 | 每个 Component 的 id, footprint, bbox, rotation 不为空 | 全部 True |
| **T10** | 网络数量 | 提取的 Net 数量（唯一网络名） | >= 5 |
| **T11** | 网络连接非空 | 每个 Net 的 connections 列表 | 不为空 |
| **T12** | 电源网络标记 | GND / VCC / 3V3 等网络的 is_power | True |
| **T13** | 板框边界 | board_bounds 的值 | 符合原板范围 |
| **T14** | 封装类型识别 | R0603(0), LQFP-48(2), SOP-14(1), BGA(3), 其他(4) | 分类正确 |
| **T15** | load_schematic 返回值 | 加载原理图 `.kicad_sch` | 返回 True |
| **T16** | 原理图元件数量 | 原理图中 Component 数量 | == 48 |
| **T17** | 原理图元件无位置 | 原理图提取的 Component 的 bbox 为默认值 | 无位置信息 |
| **T18** | 原理图网络数量 | 原理图中 Net 数量 | >= 5 |

---

### 第三轮：导出功能

| ID | 测试 | 测试内容 | 断言 |
|----|------|---------|------|
| **T19** | export_layout 返回值 | 导出布局文件 | 返回 True |
| **T20** | 导出的元件完整 | 导出的 `.kicad_pcb` 包含正确的 footprint 数量 | == 47 |
| **T21** | 导出的元件位置正确 | 随机抽3个元件，导出的 at 坐标与传入的一致 | 坐标匹配 |
| **T22** | export_board_from_scratch 返回值 | 从零生成 PCB | 返回 True |
| **T23** | 从零生成包含板框 | 生成的 PCB 包含 `(gr_line ... Edge.Cuts)` | 至少1条 |
| **T24** | 从零生成包含网络定义 | 生成的 PCB 包含 `(net ...)` 定义 | 数量=传入的 nets 数量 |
| **T25** | 从零生成包含元件 | 生成的 PCB 包含 `(footprint ...)` | 数量=传入的 components 数量 |
| **T26** | 从零生成包含走线 | 生成的 PCB 包含 `(segment ...)` | 数量=传入的 traces 段数 |
| **T27** | 从零生成包含过孔 | 生成的 PCB 包含 `(via ...)` | 数量=传入的 vias 数量 |

---

### 第四轮：数据完整性（Round-trip）

| ID | 测试 | 测试内容 | 断言 |
|----|------|---------|------|
| **T28** | 导入→导出→再导入，元件数量不变 | load_board → export_layout → load_board | 元件数一致 |
| **T29** | 导入→导出→再导入，网络数量不变 | load_board → export_board_from_scratch → load_board | 网络数一致 |
| **T30** | 从零生成→导入，前后一致 | 手动构建 PCBState → export → load_board | 数据结构正确 |

---

## 测试环境准备

### 所需文件

```
测试数据：
  D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b/
  ├── 90i0-quan-cai-shi-zhong-ac67228b.kicad_pcb      ← 答案PCB
  ├── 90i0-quan-cai-shi-zhong-ac67228b.kicad_sch       ← 原理图
  ├── 90i0-quan-cai-shi-zhong-ac67228b.kicad_pro       ← 项目文件
  └── 90i0-quan--easyedapro.kicad_sym                  ← 符号库

测试代码：
  D:/pcb大模型/AI_PCB_main_origin/test/第一次环境基础上测试/
  └── test_kicad_interface.py                          ← 测试代码

输出（自动生成）：
  D:/pcb大模型/kicad-native-batch/90i0-quan-cai-shi-zhong-ac67228b/
  ├── 测试_export_test.kicad_pcb                       ← 导出测试输出
  └── 测试_from_scratch_test.kicad_pcb                 ← 从零生成测试输出
```

### 依赖

```python
import pcbnew          # KiCad 10.0 Python API
import os, sys, json, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
```

---

## 测试方法说明

### 格式检查（T1-T5）

直接对生成的 `.kicad_pcb` 文件做纯文本检查：

```python
def check_no_graphics_wrapper(filepath: str) -> bool:
    """检查文件不包含 '(graphics' 这个非法 token"""
    with open(filepath, 'r') as f:
        content = f.read()
    return '(graphics' not in content

def check_no_nets_wrapper(filepath: str) -> bool:
    """检查文件不包含 '(nets ' 这个非法 token（注意空格区别 '(net'）"""
    with open(filepath, 'r') as f:
        content = f.read()
    return '(nets ' not in content

def check_paren_balance(filepath: str) -> bool:
    with open(filepath, 'r') as f:
        content = f.read()
    return content.count('(') == content.count(')')
```

### KiCad 打开测试（T6）

```python
def check_kicad_opens(filepath: str) -> bool:
    """真正的验收测试：KiCad 能加载这个文件"""
    try:
        board = pcbnew.LoadBoard(filepath)
        return True
    except Exception as e:
        print(f"KiCad 加载失败: {e}")
        return False
```

### 导入测试（T7-T18）

```python
# 示例：T7+T8
kicad = KiCadUtils()
assert kicad.load_board(PCB_PATH) == True
assert len(kicad.components) == 47
assert len(kicad.nets) >= 5

# T9：抽检一个元件
r1 = kicad.components.get("R1")
assert r1 is not None
assert r1.id == "R1"
assert r1.footprint != ""
assert len(r1.bbox) == 4
assert r1.rotation in [0, 1, 2, 3]
```

### 导出测试（T19-T27）

```python
# T19+T20：模板导出
kicad.export_layout("output.kicad_pcb", kicad.components)
board2 = pcbnew.LoadBoard("output.kicad_pcb")
fp_count = len(list(board2.Footprints()))
assert fp_count == 47

# T22+T23：从零生成
result = kicad.export_board_from_scratch(
    output_path="output2.kicad_pcb",
    components=components,
    nets=nets,
    board_outline=outline,
    traces=traces,
    vias=vias,
)
assert result == True
assert check_kicad_opens("output2.kicad_pcb")
```

---

## 测试结果记录

| 轮次 | ID | 测试名 | 结果 | 备注 |
|------|----|--------|------|------|
| 1 | T1 | 无 graphics 包装 | ⬜ | |
| 1 | T2 | 无 nets 包装 | ⬜ | |
| 1 | T3 | 括号平衡 | ⬜ | |
| 1 | T4 | 文件开头 | ⬜ | |
| 1 | T5 | 文件结尾 | ⬜ | |
| 1 | T6 | KiCad 能打开 | ⬜ | ✅ 最终验收 |
| 2 | T7 | load_board 返回值 | ⬜ | |
| 2 | T8 | 元件数量 | ⬜ | 期望 47 |
| 2 | T9 | 元件字段非空 | ⬜ | |
| 2 | T10 | 网络数量 | ⬜ | |
| 2 | T11 | 网络连接非空 | ⬜ | |
| 2 | T12 | 电源网络标记 | ⬜ | |
| 2 | T13 | 板框边界 | ⬜ | |
| 2 | T14 | 封装类型识别 | ⬜ | |
| 2 | T15 | load_schematic 返回值 | ⬜ | |
| 2 | T16 | 原理图元件数量 | ⬜ | |
| 2 | T17 | 原理图元件无位置 | ⬜ | |
| 2 | T18 | 原理图网络数量 | ⬜ | |
| 3 | T19 | export_layout 返回值 | ⬜ | |
| 3 | T20 | 导出的元件完整 | ⬜ | |
| 3 | T21 | 导出的元件位置正确 | ⬜ | |
| 3 | T22 | from_scratch 返回值 | ⬜ | |
| 3 | T23 | 包含板框 | ⬜ | |
| 3 | T24 | 包含网络定义 | ⬜ | |
| 3 | T25 | 包含元件 | ⬜ | |
| 3 | T26 | 包含走线 | ⬜ | |
| 3 | T27 | 包含过孔 | ⬜ | |
| 4 | T28 | Round-trip 元件数 | ⬜ | |
| 4 | T29 | Round-trip 网络数 | ⬜ | |
| 4 | T30 | 手动构建一致性 | ⬜ | |

---

## 最终验收条件

> **全部 30 项测试通过**，且 **T6（KiCad 能打开）必须为 ✅**，视为本次接口测试通过。
