# PCB_RL

基于强化学习的 PCB 自动布局（Double DQN）。

> ⚠️ 本项目 **2026-09-22 起整体重构**：旧的 5 步动作栈（`env/pcb_layout_env.py` +
> `dqn/` + `config/argument.txt` + `main.py/run.py/run_inference.py` 那套）**已废弃并删除**。
> 本文以下内容全部是现行架构。

## 先读这三份

- **技术总览（新人从这里开始，约 300 行）**：[`docs/项目技术文档.md`](docs/项目技术文档.md)
- **当前参数一览（现在到底配的是什么）**：[`PARAMS.md`](PARAMS.md)
- **设计台账（唯一事实来源，2400+ 行，含每个数字的由来）**：[`docs/导入信息重构.md`](docs/导入信息重构.md)

（重构前的旧文档在 `docs/旧版结构文档/`，**已作废**，仅作历史对照。）

## 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.11** | 项目既有环境即 3.11（conda 环境名 `pcb_rl_311`）；更低版本未验证 |
| 第三方依赖 | 只有 5 个 | `numpy` / `matplotlib` / `gymnasium` / `torch` / `pytest`，见 `requirements.txt`（实测全仓库只 import 这 5 个） |
| KiCad | **训练/推理都不需要** | 只有「重建缓存」和「读真实板的测试」需要 KiCad 10.0+，见下文 |
| 图形显示 | **不需要** | 3 个绘图模块都强制了 `matplotlib.use("Agg")`，没有 `DISPLAY` 也能出图 |
| 网络 | 只有装依赖时需要 | 能访问 PyPI（或内网镜像）即可；依赖装完后训练/推理全程离线 |
| 磁盘 | 约 62 MB | 其中 `data/` 约 62 MB（板源文件 + 离线缓存） |

## 安装（新机器 / Linux 服务器）

```bash
git clone https://github.com/Sysu-PCB-RL-Group/PCB_RL.git
cd PCB_RL

# ① conda（推荐，与本机环境同名同版本）
conda create -n pcb_rl_311 python=3.11 -y
conda activate pcb_rl_311
# ② 或者 venv
#   python3 -m venv .venv && source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# 自检：这行必须打印出版本号，否则说明解释器或依赖没装对
python -c "import torch, numpy, gymnasium, matplotlib; print(torch.__version__, numpy.__version__)"
```

想用 GPU：把 torch 那一行换成

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

训练时再加 `--device cuda`（**默认是 `cpu`**）。

## 最快跑起来

```bash
python main.py train        # 训练：默认 1000 局 → output/
python main.py infer        # 推理：读 output/best.pt → result/
python main.py help         # 用法
python main.py train --help # 训练的全部参数

# 冒烟（几分钟）：装完环境先跑这个，确认跑得通再上正式训练
python main.py train --episodes 5 --eval-every-episodes 5 --log-every 1 --out output_smoke

python -m pytest tests -q   # 测试（需要 KiCad 的用例会自动 skip）
```

`main.py train|infer` 的参数**原样透传**给 `scripts/train_dqn.py` /
`scripts/infer_dqn.py`，所以用 `--help` 看全部选项。

Windows 本机等价写法（本机 conda 环境的绝对路径）：

```powershell
cd E:\PCB_AI\PCB_RL
& E:\miniconda\envs\pcb_rl_311\python.exe main.py train
```

## 常用参数

默认值来自 `scripts/train_dqn.py`：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--episodes` | `1000` | 训练局数（冒烟用 `5`） |
| `--device` | `cpu` | `cpu` / `cuda` |
| `--threads` | `0` | torch CPU 线程数；`0` = 用默认（吃满核，多人共用机器时建议显式限制） |
| `--out` | `output` | 产物目录（**训练开始时会先清空该目录**，见下） |
| `--boards` | 全部 | 逗号分隔的板名，只跑其中几块 |
| `--seed` | `114514` | 随机种子，固定（连网络初始化都由它决定） |
| `--eval-every-episodes` | `50` | 每多少局评估一次（评估更优就写 `best.pt`） |

## 输入数据与产物

- **输入**：`data/model_cache/` —— 16 块板的离线几何缓存（`index.json` + 16 个
  `.pkl`，共 17 个文件、约 1.1 MB，**随仓库一起 clone 下来**）。这是训练/推理
  **唯一**的几何来源：**不要删、也不要重新生成**（重新生成需要 KiCad）。
- **产物**：`output/`（`best.pt` / `train_curves.png` / `train.log` / `log.jsonl`）、
  `result/`。这两类目录都在 `.gitignore` 里，不会污染仓库。
- ⚠️ **`--out` 指向的目录会在训练开始时被清空**。代码里有安全闸：目标目录若含
  `.git` / `data` / `env` / `tests` / `scripts` / `docs` / `utils` 之一（或就是仓库
  根目录）会直接报错拒绝执行 —— 但别去试。

## 关于 KiCad

训练与推理**不需要 KiCad**：`env/` 里 0 处调用 `pcbnew`（只在一条注释里提到），
几何全部从 `data/model_cache/` 读。只有下面两件事需要本机装 KiCad **10.0+**
并设置 `KICAD_PATH`：

1. 重建模型缓存：`python scripts/dump_models.py`
2. 读真实板的测试：`pytest tests/kicad`

没装 KiCad 时，`pytest tests` 里需要 KiCad 的用例会**自动跳过**，其余照常运行。

下载：<https://www.kicad.org/download/>

## 训练要多久 / 怎么盯进度

默认 1000 局（16 块板），**本机 CPU 实测参考约 4.4 小时**。训练过程中每 10 局往
`output/train.log` 打一行进度、每 50 局打一行评估；日志每局都落盘，可以随时

```bash
tail -f output/train.log
```

看进度。

## 排障

| 症状 | 原因 / 处理 |
|---|---|
| `main.py` 提示"这个解释器没有 torch" | 用错解释器了：必须用装了依赖的那个（见「安装」） |
| `CacheError: 缓存索引不存在：…` | `data/model_cache/index.json` 缺失 —— 从仓库重新 clone，别删缓存 |
| `CacheVersionError` | 缓存版本与代码不匹配：代码改过 `PcbModel` 字段，需要重新 dump（要有 KiCad） |
| 训练一开始就报"拒绝清零 …" | `--out` 指到了仓库目录，换成 `output` 之类的独立目录 |
| 出图里的中文显示成方框 | 运行环境缺中文字体。仓库里有 `setup_chinese_font` 做回退（`utils/viz_layout.py`），但**未在没装字体的 Linux 上验证过**；只影响图面文字，不影响数值与训练结果 |
