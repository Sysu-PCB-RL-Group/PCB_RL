# PCB_RL

基于强化学习的 PCB 自动布局系统。

## 目录结构

```
PCB_RL/
├── main.py                      # CLI 入口（训练入口）
├── run.py                       # 训练循环（自回归 5 步 DQN）
├── requirements.txt
├── config/
│   └── argument.txt             # 超参数配置
├── kicad-native-batch/          # 438 个立创 EDA KiCad 项目（实际数据集）
│   └── <project_name>/          # 每个项目一个文件夹
│       ├── *.kicad_pcb          # PCB 布局文件（RL 读取的数据源）
│       ├── *.kicad_sch          # 原理图
│       └── *.epro               # 立创 EDA 原生项目
├── data/
│   ├── 模拟/                    # 合成 PCB JSON 数据（旧版，用于无 KiCad 调试）
│   └── README.txt
├── docs/                        # 文档 & SPEC
├── dqn/                         # DQN 模块
│   ├── network.py               # 5-Head 自回归网络
│   ├── replay_buffer.py         # 经验回放缓冲区
│   └── agent.py                 # DQN Agent
├── env/                         # RL 环境
│   ├── pcb_layout_env.py        # Gymnasium Env（自回归 5 步）
│   ├── pcb_utils.py             # 约束检测 & Reward 计算
│   └── mock_pcb.py              # Mock 合成数据（无需 KiCad）
├── scripts/
│   ├── run_inference.py         # 推理脚本（支持 --pcb 和 --data 两种输入）
│   ├── generate_mock_pcbs.py    # 合成 PCB JSON 生成器（旧版）
│   └── fix_board_bounds.py      # 板框修复工具（旧版）
├── utils/                       # 工具模块
│   ├── kicad_utils/             # KiCad 解析 + 导出工具
│   └── visualization.py         # matplotlib 布局可视化
├── tests/                        # 测试套件（86 用例）
│   ├── conftest.py               # Mock 测试 fixtures
│   ├── test_dqn_pipeline.py      # DQN 全管线集成测试（20 ep）
│   ├── test_pcb_utils.py         # 约束检测单测（26 用例）
│   ├── test_env.py               # PCBLayoutEnv 单测（14 用例）
│   ├── test_mock_pcb.py          # MockPCB 单测（13 用例）
│   ├── test_network.py           # DQN 网络单测（7 用例）
│   ├── test_replay_buffer.py     # ReplayBuffer 单测（8 用例）
│   └── kicad/                    # 真实 KiCad 集成测试（17 用例，需 KICAD_PATH）
│       ├── test_load_board.py    # load_board 加载验证
│       ├── test_export_layout.py # export_layout 导出回吐验证
│       ├── test_restore_reset.py # restore_initial_state 缓存恢复
│       └── test_real_env.py      # PCBLayoutEnv 真实数据集成
└── output/                      # 推理输出
    └── <pcb_name>/              # 每块 PCB 一个文件夹
        └── best_layout.kicad_pcb  # 布局优化后的 PCB 文件
```

## 依赖

- **KiCad 10.0+**（必装，用于解析 .kicad_pcb）
- Python 3.11+（需与 KiCad 内置 Python 版本匹配）
- PyTorch + Gymnasium + NumPy + Matplotlib
- 设置环境变量 `KICAD_PATH` 指向 KiCad 安装目录
- 见 `requirements.txt`

## 数据

`kicad-native-batch/` 目录下每个项目含：
- 元件列表（参考编号、封装尺寸、引脚、位置、旋转、层）
- 网络连接（网络名、连接关系）
- 板框（Edge.Cuts 层）
- 已有走线 + 敷铜（导出时全部保留）

## 推理

**前置条件**：设置 `KICAD_PATH` 环境变量指向 KiCad 安装目录（mock 模式不需要）。

**Windows (PowerShell)：**

```powershell
$env:KICAD_PATH = "E:\Kicad"

# 从真实 .kicad_pcb 加载（推荐）
python scripts/run_inference.py --pcb kicad-native-batch/<项目>/xxx.kicad_pcb

# 加载训练好的模型权重
python scripts/run_inference.py --pcb kicad-native-batch/<项目>/xxx.kicad_pcb --model result/xxx_ep5000.pt

# 加 --debug 输出步骤图 + 布局图 + JSON
python scripts/run_inference.py --pcb kicad-native-batch/<项目>/xxx.kicad_pcb --debug

# 限制搜索轮数（默认 500 轮）
python scripts/run_inference.py --pcb kicad-native-batch/<项目>/xxx.kicad_pcb --max-attempts 50

# 从 JSON mock 数据加载（无需 KiCad）
python scripts/run_inference.py --data data/模拟/pcb_001.json
```

**macOS / Linux (bash)：**

```bash
export KICAD_PATH="/path/to/kicad"

# 从真实 .kicad_pcb 加载（推荐）
python scripts/run_inference.py --pcb path/to/your.kicad_pcb

# 加载训练好的模型权重
python scripts/run_inference.py --pcb path/to/your.kicad_pcb --model result/xxx_ep5000.pt

# 加 --debug 输出步骤图 + 布局图 + JSON
python scripts/run_inference.py --pcb path/to/your.kicad_pcb --debug

# 限制搜索轮数
python scripts/run_inference.py --pcb path/to/your.kicad_pcb --max-attempts 50

# 从 JSON mock 数据加载（无需 KiCad）
python scripts/run_inference.py --data data/模拟/pcb_001.json
```

输出到 `output/<pcb_name>/`：

| 文件 | 说明 | 默认生成 |
|:---|:---|:---:|
| `best_layout.kicad_pcb` | 优化后的 PCB 文件（走线/敷铜全部保留） | ✅ |
| `step_01~N.png` | 每步放置过程图 | `--debug` 时 |
| `best_layout.png` | 最终布局可视化 | `--debug` 时 |
| `best_layout_state.json` | 布局状态快照 | `--debug` 时 |

## 训练

```bash
pip install -r requirements.txt
```

**Windows (PowerShell)：**

```powershell
$env:KICAD_PATH = "E:\Kicad"

# 训练 kicad-native-batch/ 下所有 PCB
python main.py --task_name my_task

# 指定训练某块板
python main.py --data_dir kicad-native-batch/<project> --total_episodes 5000 --save_dir ./result
```

**macOS / Linux (bash)：**

```bash
export KICAD_PATH="/path/to/kicad"

# 训练 kicad-native-batch/ 下所有 PCB
python main.py --task_name my_task

# 指定训练某块板
python main.py --data_dir kicad-native-batch/<project> --total_episodes 5000 --save_dir ./result
```

输出 `./result/<pcb_name>_ep<编号>.pt` 模型权重，推理时通过 `--model` 加载。

## 推理参数

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `--pcb` | 真实 `.kicad_pcb` 文件路径 | - |
| `--data` | Mock JSON 数据路径（与 `--pcb` 二选一） | - |
| `--model` | 训练好的 `.pt` 模型权重（不指定则用随机权重） | `None` |
| `--debug` | 输出步骤图 + 布局图 + JSON | `False` |
| `--max-attempts` | 最大搜索 episode 数 | `500` |
| `--export` | 导出路径（默认自动生成） | `output/<name>/best_layout.kicad_pcb` |

## 训练流程说明

- 每块 PCB 独立训练一个 DQN 模型（因为每块板元件数不同）
- ε-greedy 探索从 1.0 衰减到 0.01
- Replay buffer 64 batch / 100000 容量
- 每步训练后自动学习，不等待 episode 结束

## 导出机制

训练/推理后通过 `export_layout()` 导出：
1. 从缓存的全量 board 中找到对应 footprint
2. 只改位置/旋转/层（原有走线、敷铜、丝印、3D 模型全部保留）
3. 输出新的 .kicad_pcb 文件

## 测试

```powershell
# Mock 测试（无需 KiCad，随时可跑，69 用例）
pytest tests/ --ignore=tests/kicad

# KiCad 集成测试（需设置 KICAD_PATH，17 用例）
pytest tests/kicad/

# 全部 86 用例（需设置 KICAD_PATH）
pytest tests/
```
