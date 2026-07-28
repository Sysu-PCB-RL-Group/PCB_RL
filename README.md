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
├── tests/                        # 测试套件（69 用例）
│   ├── conftest.py               # 共享 fixture
│   ├── test_dqn_pipeline.py      # DQN 全管线集成测试（20 ep）
│   ├── test_pcb_utils.py         # 约束检测单测（26 用例）
│   ├── test_env.py               # PCBLayoutEnv 单测（14 用例）
│   ├── test_mock_pcb.py          # MockPCB 单测（13 用例）
│   ├── test_network.py           # DQN 网络单测（7 用例）
│   └── test_replay_buffer.py     # ReplayBuffer 单测（8 用例）
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

```bash
# 设置 KiCad 路径（或已配置 KICAD_PATH 环境变量则跳过）
export KICAD_PATH="/path/to/kicad"

# 从真实 .kicad_pcb 加载（推荐）
python scripts/run_inference.py --pcb path/to/your.kicad_pcb

# 输出调试文件（步骤图 + 布局图 + JSON）：加 --debug
python scripts/run_inference.py --pcb path/to/your.kicad_pcb --debug

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
export KICAD_PATH="/path/to/kicad"

# 训练 kicad-native-batch/ 下所有 PCB
python main.py --task_name my_task

# 指定训练某块板
python main.py \
  --data_dir kicad-native-batch/<project> \
  --total_episodes 5000 \
  --save_dir ./result
```

输出 `./result/<pcb_name>_ep<编号>.pt` 模型权重。

## 训练流程说明

- 每块 PCB 独立训练一个 DQN 模型（因为每块板元件数不同）
- ε-greedy 探索从 1.0 衰减到 0.01
- Replay buffer 64 batch / 10000 容量
- 每步训练后自动学习，不等待 episode 结束

## 导出机制

训练/推理后通过 `export_layout()` 导出：
1. 从缓存的全量 board 中找到对应 footprint
2. 只改位置/旋转/层（原有走线、敷铜、丝印、3D 模型全部保留）
3. 输出新的 .kicad_pcb 文件
