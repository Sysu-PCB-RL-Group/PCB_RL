# PCB_RL

中山大学 PCBxRL 小组仓库 —— 基于强化学习的 PCB 自动布局与布线。

## 目录结构

```
PCB_RL/
├── main.py              # 入口
├── run.py               # 训练循环
├── requirements.txt
├── data/                # PCB 设计文件
├── docs/                # 文档
├── env/                 # RL 环境
└── kicad_utils/         # KiCad 解析工具
```

## 文档

`docs/` 目录包含项目约束体系与工程规格，阅读顺序：

| 文件 | 用途 |
|------|------|
| `PCB全部约束清单.md` | 背景调研：80+ 条 PCB 布局约束（学术+IPC标准+开源参考） |
| `PCB核心约束清单.md` | 筛选后的核心约束 C0-C6 及伪代码（可直接建模） |
| `环境构建约束.md` | 施工版最终方案（整合评审反馈，含完整 Reward 公式） |
| `SPEC.md` | **项目唯一技术标准**，所有代码实现以该文件为准 |

## 依赖

- Python 3.10+
- KiCad 9.0（提供 `pcbnew` Python API）
- 见 `requirements.txt`

## 快速开始

```bash
pip install -r requirements.txt
python main.py --task_name my_task
```
