# 当前参数一览

> **本文件是生成物，请勿手改** —— 由 `python scripts/dump_params.py --write` 生成。
> 真源是代码：每条参数的「代码位置」列就是它被定义的地方；清单与代码不一致时，
> `pytest tests/kicad/test_params_single_source.py` 会直接失败。

「改了要重训 / 重标金表」的口径常数在前几节；运行参数（每次跑可覆盖）在最后几节。

## 网格与几何（改了要重跑 L/G、重标金表）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `GRID_M` | `200` | 画布边长（M×M 网格；cell 尺寸由板框与 M 决定） | §5（Owner 2026-09-16 拍案） | `env.grid_render:GRID_M` |
| `SOFT_GAP_MM` | `0.2` | 软间距阈值（= 硬∪软的合并阈值） ⚠️ 与 `env/violations.py` 那份是**两份独立定义**（值相同、无守门） | §3.4（判据本体） | `env.grid_render:SOFT_GAP_MM` |
| `_EPS` | `1e-09` | 边界数值容差（禁区外扩 / 板框内收；只为消浮点末位差，方向保守） | §5.7 | `env.grid_render:_EPS` |
| `MODE_FILL` | `fill` | 板框 → 网格的映射口径（`fill` = 按轴铺满 M×M） | §5（Owner 拍案口径） | `env.grid_render:MODE_FILL` |

## G：增益矩阵（改了要重标 G 金表）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `G_W_HPWL` | `0.7` | G 的 HPWL 分量权重（λ 的前一项） | §5.4（Owner 2026-09-17 拍案） | `env.grid_render:G_W_HPWL` |
| `G_W_DISP` | `0.3` | G 的离散度分量权重（λ 的后一项） | §5.4（Owner 2026-09-17 拍案） | `env.grid_render:G_W_DISP` |
| `G_P_LO` | `5.0` | 稳健 min-max 的下分位（4 张表共用一把标尺） | §5.4 | `env.grid_render:G_P_LO` |
| `G_P_HI` | `95.0` | 稳健 min-max 的上分位 | §5.4 | `env.grid_render:G_P_HI` |

## reward：回报（改了要重训）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `ALPHA` | `5.6` | σ 项系数（`+α·Δσ/D`）；语义 = 每步判别力 HPWL:σ 按 λ 配平 | §7.3（probe202 + Owner 2026-09-19 拍） | `env.reward:ALPHA` |
| `BETA_MARGIN` | `1.2` | β = margin × β_crit(该板)（余量覆盖口径近似） | §7.3（Owner 2026-09-22 拍） | `env.reward:BETA_MARGIN` |
| `GAMMA` | `1.0` | 折扣（episode 有限且会终止 ⇒ γ = 1） | §7.3（Owner 2026-09-22 拍） | `env.reward:GAMMA` |

## obs：观测编码（改了要重训）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `N_ROT` | `4` | 旋转数（动作第一维；转换角 {0,1,2,3}） ⚠️ `env/policy_net.py` 另有一份同值 `N_ROT` | §5.5 / §6.1 | `env.obs_spec:N_ROT` |
| `SPATIAL_CHANNELS` | `8` | spatial 通道数 = N_ROT × {L, G} ⚠️ `env/policy_net.py` 的 `CHANNELS` 是同值另一份 | §5.5 | `env.obs_spec:SPATIAL_CHANNELS` |
| `PAD_MAX` | `80` | piece 归一化上界：单件最大 pad 数（超界不裁剪） | §5.5（`scripts/dataset_report.py` 实测） | `env.obs_spec:PAD_MAX` |
| `NET_MAX` | `28` | piece 归一化上界：单件最大**非电源** net 数 | §5.5（同上实测） | `env.obs_spec:NET_MAX` |
| `GAIN_FILL` | `0.0` | 送网络的 G 里**非法格**的填值（0 = 无效/非法） | §5.5（Owner 2026-09-22 拍） | `env.obs_spec:GAIN_FILL` |
| `PIECE_KEYS` | `('pad', 'np_net', 'progress')` | piece (3,) 的语义顺序（写死） | §5.5 | `env.obs_spec:PIECE_KEYS` |

## 放置顺序（改了会改整条轨迹）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `DEFAULT_W_AREA` | `0.7` | 固定顺序的面积权重（连通性权重 = 1 − 该值） ⚠️ 与 G 的 0.7 **无关系**（同名不同物） | §6.2（Owner 2026-09-17 拍案） | `env.order_plan:DEFAULT_W_AREA` |

## 违规判定（与 L 必须同口径）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `SOFT_GAP_MM` | `0.2` | 软间距阈值（`Judge` 用的那一份；严格 0.2，不做豁免） ⚠️ 与 `env/grid_render.py` 那份**重复定义**（两份都 = 0.2） | §3.4 | `env.violations:SOFT_GAP_MM` |
| `_PAD_CACHE_MAX` | `8192` | pad AABB 缓存上限（位姿连续 ⇒ 必须封顶） | 源码注释（防止训练中无界增长） | `env.violations:_PAD_CACHE_MAX` |

## 网络结构

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `CHANNELS` | `8` | 网络输入通道数（= N_ROT × {L, G}） | §9.2 | `env.policy_net:CHANNELS` |
| `N_ROT` | `4` | 网络输出通道数 / 动作第一维 ⚠️ 与 `env/obs_spec.py` 那份**重复定义** | §9.2 | `env.policy_net:N_ROT` |

## agent 超参（训练旋钮）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `DEFAULT_CAPACITY` | `10000` | 打包回放容量（条数；≈ 3.80 GB @200×200） | §9.2（Owner 2026-09-23 暂定） | `env.dqn_agent:DEFAULT_CAPACITY` |
| `DEFAULT_SHAPE` | `(8, 200, 200)` | 回放打包的 obs 形状（spatial） | §9.2 | `env.dqn_agent:DEFAULT_SHAPE` |
| `lr` | `0.0001` | Adam 学习率 | §9.2（Owner 2026-09-23 拍） | `env.dqn_agent:DoubleDQNAgent.__init__:lr` |
| `batch_size` | `64` | 采样批量（**训练脚本用 `--batch 32` 覆盖**） | §9.2（我方自定） | `env.dqn_agent:DoubleDQNAgent.__init__:batch_size` |
| `target_every_episodes` | `20` | 目标网硬同步周期（按 episode 记数） | §9.2（Owner 2026-09-23：先 100、后因只跑 100 局改 20） | `env.dqn_agent:DoubleDQNAgent.__init__:target_every_episodes` |
| `eps_top` | `0.9` | ε 曲线上界（ep = 0 处的探索率） | §9.2（Owner 2026-09-23 拍） | `env.dqn_agent:DoubleDQNAgent.__init__:eps_top` |
| `eps_bottom` | `0.1` | ε 曲线下界（ep → ∞ 时的渐近探索率） | §9.2（Owner 2026-09-23 拍） | `env.dqn_agent:DoubleDQNAgent.__init__:eps_bottom` |
| `eps_tau_episodes` | `2000.0` | ε 时间常数 τ（`ε = bottom + (top−bottom)·exp(−ep/τ)`） ⚠️ 1,000 局跑完时 ε ≈ 0.585（仍近六成随机探索） | §9.2（**我方自定** = 2,000） | `env.dqn_agent:DoubleDQNAgent.__init__:eps_tau_episodes` |
| `gamma` | `1.0` | 折扣（直接引用 `reward.GAMMA`，不另设常数） | §7.3 / §9.2 | `env.dqn_agent:DoubleDQNAgent.__init__:gamma` |

## 训练默认值（CLI 可覆盖）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `episodes` | `1000` | 训练局数 | §9.2（Owner 2026-09-23 拍定） | `scripts.train_dqn:DEFAULTS:episodes` |
| `update_every` | `4` | 每多少个 env 步做一次梯度更新 | §9.2（同上） | `scripts.train_dqn:DEFAULTS:update_every` |
| `batch` | `32` | 采样批量（覆盖 agent 的 64） | §9.2（同上） | `scripts.train_dqn:DEFAULTS:batch` |
| `eval_every_episodes` | `50` | 每多少局跑一次 16 板评估 | §9.2（同上） | `scripts.train_dqn:DEFAULTS:eval_every_episodes` |
| `log_every` | `10` | 每多少局打一行进度 | §9.2（同上） | `scripts.train_dqn:DEFAULTS:log_every` |
| `ckpt_every` | `0` | 每多少局落一次断点 | §9.2（同上） | `scripts.train_dqn:DEFAULTS:ckpt_every` |
| `capacity` | `10000` | 回放容量（传给 agent） | §9.2（同上） | `scripts.train_dqn:DEFAULTS:capacity` |
| `target_every_episodes` | `20` | 目标网硬同步周期（传给 agent） | §9.2（同上） | `scripts.train_dqn:DEFAULTS:target_every_episodes` |
| `seed` | `114514` | 固定种子（网络初始化也由它决定 ⇒ 同命令可复现） | §9.2（Owner 2026-09-23） | `scripts.train_dqn:DEFAULTS:seed` |
| `out` | `output` | 产物目录（**写入前清零**） | §9.2 | `scripts.train_dqn:DEFAULTS:out` |
| `device` | `cpu` | 计算设备 | §9.2 | `scripts.train_dqn:DEFAULTS:device` |

## 推理默认值（CLI 可覆盖）

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `ckpt` | `output/best.pt` | 权重文件（默认训练产物 `output/best.pt`） | §9.2 / §10.1 | `scripts.infer_dqn:DEFAULTS:ckpt` |
| `out` | `result` | 产物根目录（只清 `infer_*`） | §9.2 | `scripts.infer_dqn:DEFAULTS:out` |
| `boards` | `""` | 逗号分隔的板名（空 = 全部板） | §9.2 | `scripts.infer_dqn:DEFAULTS:boards` |
| `dpi` | `110` | 对比图分辨率 | §10.1 | `scripts.infer_dqn:DEFAULTS:dpi` |
| `viz` | `True` | 是否出 2×2 对比图（`--no-viz` 关） | §10.1 | `scripts.infer_dqn:DEFAULTS:viz` |
| `net_boxes` | `False` | 是否画 HPWL 包围框（`--net-boxes` 开） ⚠️ 默认**关** | §10.1（Owner 2026-09-23） | `scripts.infer_dqn:DEFAULTS:net_boxes` |
| `charts` | `True` | 是否出 3 张指标对比图（`--no-charts` 关） | §10.1.1 | `scripts.infer_dqn:DEFAULTS:charts` |
| `device` | `cpu` | 计算设备 | §9.2 | `scripts.infer_dqn:DEFAULTS:device` |
| `cache_root` | `None` | 模型缓存目录覆盖（None = `data/model_cache`） | §4 | `scripts.infer_dqn:DEFAULTS:cache_root` |

## env 构造默认值

| 参数 | 当前值 | 含义 | 依据 | 代码位置 |
|---|---|---|---|---|
| `use_lookahead` | `True` | True = 训练档（`mask = L & lookahead`）｜False = 推理档（`mask = L`） | §6.3（Owner 2026-09-18） | `env.pcb_env:PcbEnv.__init__:use_lookahead` |
| `verify` | `True` | 落子前是否用 `Judge` 精确复核（不一致 ⇒ 抛 `EnvContractError`） | §5.7 | `env.pcb_env:PcbEnv.__init__:verify` |
| `with_contrib` | `True` | 完成度 ③（贡献占比）是否建表（只影响 `info`） | §0.6 m | `env.pcb_env:PcbEnv.__init__:with_contrib` |
| `resample_dead_start` | `True` | 开局 `L` 就空时是否重采一块板 | §9.1 | `env.pcb_env:PcbEnv.__init__:resample_dead_start` |
