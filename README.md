# PCB_RL

中山大学 PCBxRL 小组仓库 —— 基于强化学习的 PCB 自动布局与布线。

## 目录结构

```
PCB_RL/
├── main.py                      # CLI 入口
├── run.py                       # 训练循环（自回归 5 步 DQN）
├── requirements.txt
├── config/
│   └── argument.txt             # 超参数配置
├── data/
│   ├── 模拟/                    # 合成 PCB JSON 数据（pcb_001~010.json）
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
├── scripts/                     # 工具脚本
│   ├── run_inference.py         # 纯推理 Demo → 步骤图 + 布局图
│   ├── generate_mock_pcbs.py    # 合成 PCB JSON 生成器
│   └── fix_board_bounds.py      # 板框修复工具
├── utils/                       # 工具模块
│   ├── kicad_utils/             # KiCad 解析工具（由独立同学维护）
│   └── visualization.py         # matplotlib 布局可视化
├── tests/                        # 测试套件（69 用例，pytest）
│   ├── conftest.py               # 共享 fixture（Mock 注入 + env 工厂）
│   ├── test_dqn_pipeline.py      # DQN 全管线集成测试（20 ep）
│   ├── test_pcb_utils.py         # 约束检测单测（26 用例）
│   ├── test_env.py               # PCBLayoutEnv 单测（14 用例）
│   ├── test_mock_pcb.py          # MockPCB 单测（13 用例）
│   ├── test_network.py           # DQN 网络单测（7 用例）
│   └── test_replay_buffer.py     # ReplayBuffer 单测（8 用例）
└── output/                      # 推理输出（自动生成）
    └── <pcb_name>/              # 每块 PCB 一个文件夹
        ├── step_01.png          # 中间步骤图
        ├── best_layout.png      # 最佳布局图
        ├── best_layout_state.json
        └── inference_log.txt
```

## 依赖

- Python 3.10+
- PyTorch + Gymnasium + NumPy + Matplotlib
- pytest（测试框架）
- KiCad 9.0+（训练需 PCB 文件，推理验证和测试不需要）
- 见 `requirements.txt`

## 文档

`docs/` 目录包含项目约束体系与工程规格，阅读顺序：

| 文件 | 用途 |
|------|------|
| `PCB全部约束清单.md` | 背景调研：80+ 条 PCB 布局约束 |
| `PCB核心约束清单.md` | 核心约束 C0-C6 及伪代码 |
| `环境构建约束.md` | 施工版最终方案 |
| `SPEC.md` | **项目唯一技术标准**，所有代码实现以此为据 |
| `kicad_interface.md` | KiCad 模块接口契约 |
| `核心重构.md` | DQN 自回归架构重构记录 |

## 快速验证（无需 KiCad）

以下命令不依赖 KiCad，可直接运行。

### 运行全部测试（一键）

```bash
python -m pytest tests/ -v
```

69 个用例，覆盖 env 约束检测、DQN 组件、Mock 数据、全管线集成。

### 按模块测试

```bash
python -m pytest tests/test_pcb_utils.py -v    # 约束检测（26 用例）
python -m pytest tests/test_env.py -v           # 环境行为（14 用例）
python -m pytest tests/test_mock_pcb.py -v      # Mock 数据（13 用例）
python -m pytest tests/test_network.py -v       # DQN 网络（7 用例）
python -m pytest tests/test_replay_buffer.py -v # 回放缓冲（8 用例）
```

### DQN 模块快速验证

```bash
# DQN 网络 forward（5 个 Head 输出 shape 检查）
python -c "
from dqn.network import DQN
import torch
net = DQN(obs_dim=504, n_movable=50, grid_w=100, grid_h=80)
x = torch.randn(3, 504)
for sid in range(5): print(f'head {sid}: {list(net(x, sid).shape)}')
"
# 期望: head 0: [3, 50] | head 1: [3, 100] | head 2: [3, 80] | head 3: [3, 4] | head 4: [3, 2]

# ReplayBuffer 读写
python -c "
from dqn.replay_buffer import ReplayBuffer
import numpy as np
buf = ReplayBuffer(capacity=100)
obs = np.zeros(504, dtype=np.float32)
for i in range(20):
    sid = i % 5
    buf.add(obs, i, 1.0 if sid==4 else 0.0, obs, np.ones(50,dtype=bool), False, sid, sid+1 if sid<4 else 0)
b = buf.sample(8)
print(f'obs={b[0].shape}, act={b[1].shape}, next_sid={b[7].shape}')
"
# 期望: obs=(8, 504), act=(8,), next_sid=(8,)

# Agent learn 完整管线
python -c "
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
import numpy as np, torch
agent = DQNAgent(obs_dim=504, n_movable=50, grid_w=100, grid_h=80, device=torch.device('cpu'))
buf = ReplayBuffer(capacity=100)
obs = np.random.randn(504).astype(np.float32)
for sid in range(5):
    buf.add(obs, sid%2, 10.0 if sid==4 else 0.0, obs, np.ones(50,dtype=bool), False, sid, sid+1 if sid<4 else 0)
buf.add(obs, 0, -5.0, obs, np.ones(50,dtype=bool), True, 4, -1)
loss = agent.learn(buf.sample(6))
agent.sync_target()
print(f'loss={loss:.4f}, train_steps={agent.train_steps}')
"
# 期望: loss 非 NaN, train_steps=1
```

### 纯推理 Demo（输出步骤图 + 布局图）

```bash
# 加载 JSON 数据文件，输出到 output/<pcb_name>/
python scripts/run_inference.py --data data/模拟/pcb_001.json
```

无训练（epsilon=0，随机 DQN 权重），反复尝试放置全部元件。
输出到 `output/<pcb_name>/`：

- `step_01.png` ~ `step_N.png` — 每放置一个元件的中间截图
- `best_layout.png` — 最佳 episode 的最终布局图
- `best_layout_state.json` — 完整元件状态（供下游消费）
- `inference_log.txt` — 运行日志

> **注意**：随机权重下通常只能放置部分元件（因为未训练，模型不会选好位置）。
> 训练收敛后再跑推理才能看到完整布局。

## 训练

```bash
pip install -r requirements.txt

# 基本训练
python main.py --task_name my_task

# 自定义参数
python main.py \
  --total_episodes 5000 \
  --max_episode_step 100 \
  --device cuda \
  --save_dir ./result
```

超参数通过 `config/argument.txt` 管理，CLI 参数可覆盖文件中的值。
