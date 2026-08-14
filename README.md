# PCB_RL

基于强化学习的 PCB 自动布局系统。

## 目录结构

```
PCB_RL/
├── main.py                       # CLI 入口（训练入口）
├── run.py                        # 两阶段训练循环（BC 预训练 + RL + 专家注入）
├── run_inference.py              # 推理脚本（支持单板 / 批量）
├── requirements.txt
├── config/
│   └── argument.txt              # 超参数配置（含专家注入参数）
├── data/
│   ├── kicad-native-batch/       # 立创 EDA KiCad 项目（5个极简训练板）
│   │   ├── *.kicad_pcb           # PCB 布局文件（RL 读取的数据源）
│   │   ├── *.kicad_sch           # 原理图
│   │   └── *.kicad_pro           # KiCad 项目文件
│   ├── expert_transitions/       # 专家 transition 缓存（自动生成，可复用）
│   └── README.txt
├── docs/                         # 文档 & SPEC
├── dqn/                          # DQN 模块
│   ├── __init__.py               # 导出 DQN / ReplayBuffer / ExpertReplayBuffer / DQNAgent
│   ├── network.py                # 5-Head 自回归网络（共享 slot_encoder）
│   ├── replay_buffer.py          # ReplayBuffer + ExpertReplayBuffer（专家常驻 + PER）
│   └── agent.py                  # DQNAgent（epsilon-greedy + learn + learn_bc）
├── env/                          # RL 环境
│   ├── pcb_layout_env.py         # PCBLayoutEnv（自回归 5 步 Gymnasium Env）
│   ├── pcb_utils.py              # 约束检测 & Reward 计算
│   ├── mock_pcb.py               # Mock 合成数据（无需 KiCad，用于测试）
│   └── expert_trajectory.py      # 专家轨迹生成（反向拆解 + 微量变异 → BC 预训练数据）
├── utils/                        # 工具模块
│   ├── kicad_utils/              # KiCad 解析 + 导出工具
│   ├── pcb_scan.py               # PCB 文件扫描
│   └── visualization.py          # matplotlib 布局可视化
├── tests/                        # 测试套件（含 expert_trajectory 测试）
│   ├── conftest.py               # Mock 测试 fixtures
│   ├── test_dqn_pipeline.py      # DQN 全管线集成测试（20 ep）
│   ├── test_expert_trajectory.py # 专家轨迹生成测试（8 用例）
│   ├── test_pcb_utils.py         # 约束检测单测（26 用例）
│   ├── test_env.py               # PCBLayoutEnv 单测（14 用例）
│   ├── test_mock_pcb.py          # MockPCB 单测（13 用例）
│   ├── test_network.py           # DQN 网络单测（7 用例）
│   ├── test_replay_buffer.py     # ReplayBuffer 单测（8 用例）
│   ├── test_agent.py             # DQNAgent 单测（8 用例）
│   ├── test_visualization.py     # 可视化单测（6 用例）
│   └── kicad/                    # 真实 KiCad 集成测试（17 用例，需 KICAD_PATH）
└── output/                       # 推理输出
    └── <pcb_name>/                # 每块 PCB 一个文件夹
        ├── best_layout.kicad_pcb  # 布局优化后的 PCB 文件
        ├── final_output.kicad_pcb # 最后一次尝试的布局
        ├── step_01~N.png          # 每步放置过程图（--debug）
        └── best_layout.png        # 最终布局可视化（--debug）
```

## 依赖

- **KiCad 10.0+**（必装，用于解析 .kicad_pcb）
- Python 3.11+（需与 KiCad 内置 Python 版本匹配）
- PyTorch + Gymnasium + NumPy + Matplotlib
- 设置环境变量 `KICAD_PATH` 指向 KiCad 安装目录
- 见 `requirements.txt`

## 数据

当前训练数据集：`data/kicad-native-batch/` 下 **5 个极简板子**，全部适配 21×21 网格：

| 板子 | 元件数 | 网络数 | 网格尺寸 |
|:---|:---|:---|:---|
| 开关小板 | 6 | 3 | 6×3 |
| Connector | 8 | 3 | 13×13 |
| 灯板2 | 6 | 3 | 12×10 |
| Button | 4 | 3 | 16×10 |
| 按键侧板V0.1 | 2 | 3 | 2×16 |

每个 PCB 包含：元件列表、网络连接、板框（Edge.Cuts 层）、走线 + 敷铜（导出时全部保留）。

训练采用跨板随机选板机制：每 episode 从 5 个板子中随机选一个，Agent 和 Buffer 全局共享。

## 训练

训练分为两个阶段：

### Phase 1：BC 预训练（Behavior Cloning）

利用 KiCad 源数据自带的「完美布局」生成专家轨迹：
1. **反向拆解**：从完整布局逐件拆除（先拆小、后拆大），反转得到正向完美轨迹
2. **微量变异**：20% 概率位置 ±1 格 / 10% 概率旋转 ±90°，增加轨迹多样性
3. **纯监督学习**：用专家 transition 训练网络 50 epoch（交叉熵损失）

### Phase 2：RL 训练 + 专家注入

1. 加载 BC 预训练权重作为初始化
2. 正常 DQN 训练，ε=0.6→0.1 指数衰减（10000 ep）
3. Buffer 中 20% 专家 transition 常驻 + 80% 探索经验
4. 优先经验回放（PER），按 TD-error 优先级采样

```powershell
# Windows（PowerShell）
$env:KICAD_PATH = "E:\Kicad"

# 默认训练（两阶段自动执行）
python main.py

# 指定训练轮数
python main.py --total_episodes 10000

# 禁用专家注入（纯 DQN 训练）
python main.py --use_expert False
```

```bash
# macOS / Linux
export KICAD_PATH="/path/to/kicad"

python main.py
python main.py --total_episodes 10000
```

**专家 transition 生成**：首次训练自动从 KiCad 源数据生成专家轨迹，缓存到 `data/expert_transitions/`。后续训练自动从缓存加载，无需重新生成。

**输出**：`result/best_model.pt`（全局最佳权重）、`result/bc_pretrained.pt`（BC 预训练权重）、每板 loss 曲线和布局导出。

## 推理

```powershell
# Windows（PowerShell）
$env:KICAD_PATH = "E:\Kicad"

# 默认批量推理所有训练板（使用随机权重）
python run_inference.py

# 加载训练好的模型权重
python run_inference.py --model result/best_model.pt

# 加 --debug 输出步骤图 + 布局图 + JSON
python run_inference.py --model result/best_model.pt --debug

# 单板推理
python run_inference.py --pcb data/kicad-native-batch/开关小板/开关小板.kicad_pcb --model result/best_model.pt

# 限制搜索轮数（默认 500）
python run_inference.py --max-attempts 100
```

```bash
# macOS / Linux
export KICAD_PATH="/path/to/kicad"

python run_inference.py --model result/best_model.pt --debug
```

输出到 `output/<pcb_name>/`：

| 文件 | 说明 | 默认生成 |
|:---|:---|:---:|
| `best_layout.kicad_pcb` | 搜索到的最佳布局（走线/敷铜/丝印/3D 全保留） | ✅ |
| `final_output.kicad_pcb` | 最后一次尝试的布局 | ✅ |
| `step_01~N.png` | 每步放置过程图 | `--debug` |
| `best_layout.png` | 最终布局可视化 | `--debug` |
| `best_layout_state.json` | 布局状态快照 | `--debug` |
| `inference_log.txt` | 推理日志 | ✅ |

## 训练流程说明

- **Phase 1**（BC 预训练）：仅用专家 transition，50 epoch 纯监督学习
- **Phase 2**（RL）：ε 从 0.6 指数衰减到 0.1（10000 ep，BC 预训练后无需从 1.0 起步）
- Episode 0~2999 仅精简版 reward（惩罚+完成奖励），3000+ 品质项线性升温，10000+ 完整版 reward
- Replay buffer：ExpertReplayBuffer（专家容量 = 专家 transition 总数，普通容量 = 4× 专家数）
- PER α=0.6, β=0.4→1.0，每条普通经验写入时初始优先级设为当前最大

## 导出机制

训练/推理后通过 `export_layout()` 导出：
1. 从缓存的全量 board 中找到对应 footprint
2. 只改位置/旋转/层（原有走线、敷铜、丝印、3D 模型全部保留）
3. 输出新的 .kicad_pcb 文件

## 测试

```powershell
# Mock 测试（无需 KiCad，随时可跑）
pytest tests/ --ignore=tests/kicad

# KiCad 集成测试（需设置 KICAD_PATH）
pytest tests/kicad/

# 全部测试（需设置 KICAD_PATH）
pytest tests/
```

## 超参数配置

关键超参数见 `config/argument.txt`，部分重要参数：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| EPSILON_START | 0.6 | ε-greedy 初值 |
| EPSILON_MIN | 0.1 | ε-greedy 下界 |
| EXPERT_TRAJECTORIES_PER_BOARD | 20 | 每板生成专家轨迹数 |
| EXPERT_EPSILON_DECOMPOSE | 0.5 | 拆解随机化概率 |
| EXPERT_BC_EPOCHS | 50 | BC 预训练 epoch 数 |
| EXPERT_BUFFER_RATIO | 0.2 | Phase 2 buffer 专家占比 |
| PER_ALPHA | 0.6 | 优先级指数 |
