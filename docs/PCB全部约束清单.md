# PCB 布局约束完整调研报告（最终版）

> **范围**：仅 PCB 布局 (component placement)，不含布线 (routing)  
> **来源**：学术论文 + IPC 行业标准 + 设计手册 + 开源项目参考实现  
> **日期**：2026-07-23（整合版）

---

## 目录

0. [环境建模基础约束](#0-环境建模基础约束)
1. [制造约束 (DFM)](#1-制造约束-dfm)
2. [电气约束](#2-电气约束)
3. [热管理约束](#3-热管理约束)
4. [装配约束 (DFA)](#4-装配约束-dfa)
5. [高速 / EMC 约束](#5-高速--emc-约束)
6. [布线质量代理指标](#6-布线质量代理指标-rqp) ← **新增**
7. [RL 流程约束](#7-rl-流程约束-rlp) ← **新增**
附录A：[按算法化等级分类的约束矩阵](#附录a按算法化等级分类的约束矩阵)
附录B：[详细改进实施方案](#附录b详细改进实施方案) ← **新增**

---

## 约束字段说明

| 字段 | 说明 |
|------|------|
| ID | 模块前缀 + 序号 |
| 约束名称 | 简洁中文描述 |
| 量化阈值 | 具体数值或公式 |
| hard/soft | hard=必须满足（violation）；soft=优化目标（reward） |
| 判别条件 | 几何/电气判定逻辑 |
| 算法化 | A=纯几何计算；B=需仿真；C=启发式近似；D=需工艺数据 |
| 改进建议 | 源自开源项目参考的方法论改进方向（如有） |

---

## 0. 环境建模基础约束

> **说明**：以下约束定义 RL 环境的搜索空间、动作空间和核心优化目标。它们不来自 PCB 设计手册，而是来自 PCB placement 学术论文对问题形式化定义的一致共识，以及 RL 顺序决策问题的建模范式。

### 0.1 搜索空间定义

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ENV-01 | 元件旋转角度离散 | `θ ∈ {0°, 90°, 180°, 270°}` | hard | 检查 rotation 是否在允许集合中 | A |
| ENV-02 | 元件层分配 | `layer ∈ {TOP, BOTTOM}` | hard | 每个元件明确指定所在面 | A |
| ENV-03 | 顺序放置（单步单元件） | 每步 action 仅移动 1 个元件 | hard | `action = (component_id, x, y, θ, layer)` | A |
| ENV-04 | 已放置元件锁定 | agent 不可重新选中已放置元件 | hard | `component_id ∉ placed_set` | A |

> **改进建议**（ENV-03/ENV-04）：引入**放置优先级排序**——按元件连接度/面积降序排列，大芯片先放、小电容后放，模拟人工设计"先大后小"的顺序逻辑。引入**动作空间掩码**——在 observation 中提供 0/1 mask 滤除已被占据或非法的网格位置，agent 只从合法位置中选择。（详见附录B-RLP）

**来源**：OmniLayout (2026): "θi ∈ {0°, 90°, 180°, 270°}, li ∈ {top, back}, ∀vi ∈ V, and (xi, yi) ∈ B"

### 0.2 核心优化目标（奖励函数驱动源）

> **关键遗漏说明**：原报告 60 条约束全部为 "violation 检查"（不能做什么），缺少 "优化方向"（应该优化什么）。以下 4 条定义了 RL 奖励函数的正反馈来源，是 agent 学习布局策略的根本驱动力。

| ID | 约束名称 | 量化公式 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| OBJ-01 | 总走线长度最小化 (HPWL) | `HPWL = Σ_{net∈N} (max_x - min_x + max_y - min_y)` | soft | 所有网络 bounding box 的半周长之和 | A |
| OBJ-02 | 同网络元件空间聚集 | 共享 net 的元件 centroid 分散度最小化 | soft | `Σ_{net} std_dev(component_positions_on_net)` | A |
| OBJ-03 | 电气拓扑保持 | 预测布局中连接元件的相对位移向量应接近参考布局 | soft | OmniLayout 的 `F(x,y,x*,y*)` 函数 | A |
| OBJ-04 | 元件对齐 | 相同尺寸元件应尽量对齐排列 | soft | 统计同尺寸元件行列对齐程度 | A |

> **注**：OBJ-01 (HPWL) 的详细实现方法论和与"紧凑度"指标的对比分析，已迁移至[模块6 RQP-03](#rqp-03-hpwl-线长优化)。建议用 HPWL 替换现有的紧凑度指标。

**来源**：OmniLayout (2026): `min W(x,y,θ) + λ1·Cp2p + λ2·D + λ3·F`; HiePlace (2026): alignment constraint; MARS-Place (2026): "force-based alignment mechanism"

---

## 1. 制造约束 (DFM)

### 1.1 元件间最小间距

**来源**：IPC-7351B, IPC-2221A, TopfastPCB (2024), AllPCB (2024)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-01 | Chip 类元件间距 (0402/0603/0805) | ≥ 0.2mm (8mil) | hard | 检查两元件 rotated bounding box 交集 | A |
| MFG-02 | SOIC/SOT 类元件间距 | ≥ 1.25mm (50mil) | hard | bounding box 间距检查 | A |
| MFG-03 | QFP 类元件间距 | ≥ 2.5mm (100mil) | hard | bounding box 间距检查 | A |
| MFG-04 | BGA 类元件间距 | ≥ 5mm (200mil) | hard | bounding box 间距检查 | A |
| MFG-05 | IC 类通用间距 | ≥ 2mm (80mil) | hard | bounding box 间距检查 | A |
| MFG-06 | BTC (Bottom Termination Component) 间距 | ≥ 0.5mm | hard | IPC-7093 规范 | A |

> **改进建议**（MFG-01~06）：将当前的中心点距离检测替换为 **AABB 包围盒碰撞检测**。引入 **硬重叠/软间距分层惩罚**：硬重叠（物理交叉）按交叉面积 × 权重 1000 惩罚，软间距（靠近但未交叉）按 `(KEEPOUT - gap)² × edge_length × 权重 50` 惩罚，实现从"完全重叠"到"安全距离"的连续梯度信号。pyplacer 的 `bbox_overlap_penalty()` 函数可直接移植（约 30 行代码）。（详见附录B-1.1）

### 1.2 焊盘间距与阻焊桥

**来源**：AllPCB (2024), TopfastPCB (2024), IPC-7351B

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-07 | SMD 焊盘最小间距 | ≥ 0.15mm (6mil) | hard | 检查相邻焊盘边缘距离 | A |
| MFG-08 | 阻焊桥保留条件 | 焊盘间距 ≥ 0.1mm (4mil) | hard | 若间距 < 0.1mm 则阻焊桥消失 | A |

> **注**：layout only，MFG-07 和 MFG-08 主要影响元件封装的焊盘设计而非放置位置，在纯布局阶段可作为参考约束。

### 1.3 元件-板边距离

**来源**：IPC-2221A, Andwin Circuits (2024), Magellan Circuits (2024)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-09 | 元件-铣边距离 (routed edge) | ≥ 2.54mm | hard | 元件 bounding box 到板边的垂直距离 | A |
| MFG-10 | 元件-V-cut 距离 | ≥ 1.27mm | hard | 元件 bounding box 到 V-cut 中心线距离 | A |
| MFG-11 | 通用元件-板边距离（推荐） | ≥ 1mm | soft | 用于奖励函数（越近惩罚越大） | A |
| MFG-12 | 高元件-板边距离 | ≥ 5mm | hard | 高度 > 5mm 的元件距离板边更远 | A |

> **改进建议**（MFG-09~12）：将中心点二值判断替换为 **出界面积比例** 的连续度量。使用 AABB 包围盒代替中心点：计算元件包围盒总面积与板框交叉面积之差，完全在板内 = 0，一半在外面 = 总面积的一半。出界面积 × 权重 200 作为惩罚。训练时出界施加巨额惩罚并终止 episode，推理时用 clamp 强制拉回板内作为安全兜底。（详见附录B-1.3）

### 1.4 铜平衡

**来源**：IWDFS Solutions (2025), MorePCB (2024)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-13 | 铜覆盖率均衡 | 各层铜覆盖率差异 < 20% | soft | 可简化为元件密度均匀分布 | C |

### 1.5 高矮元件布局

**来源**：TopfastPCB (2024), 凡亿教育 (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-14 | 高矮元件间距 | 高元件（>5mm）与矮元件间距 ≥ 元件高度差 | hard | 防止回流焊热遮蔽 | A |
| MFG-15 | 小元件不被高元件包围 | 小元件（≤0603）周围 3mm 内无高元件 | soft | 可维修性检查 | A |

### 1.6 BGA 逃逸布线空间

**来源**：HiePlace (2026), TopfastPCB (2024)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-16 | BGA 逃逸布线空间 | BGA 同面周边 ≥ 2mm 保留为走线通道 | hard | BGA bounding box 外扩 2mm 检查元件遮挡 | A |

### 1.7 测试与标记

**来源**：EMS Tech (2025), CeramicPCB (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| MFG-17 | 测试点探针可访问性 | 测试点周围 ≥ 1mm 禁布元件 | hard | 以测试点为中心的圆形禁布区 | A |
| MFG-18 | 标记点 (Fiducial) 禁布 | Fiducial mark 周围 ≥ 2mm 禁布 | hard | 板角 fiducial 的圆形禁布区 | A |

> **改进建议**（MFG-17~18）：从手动硬编码禁止区域列表改为 **从 KiCad 自动读取 Zone 和 Rule Area**。使用 pcbnew API 中的 `Zone.GetIsRuleArea()` 方法自动检测设计文件中已定义的禁止区域。支持圆形、多边形等非矩形禁止区域，支持 circuit_training 中的 rate 参数（0~1 表示禁止强度）实现分级禁止。（详见附录B-1.4）

---

## 2. 电气约束

### 2.1 去耦电容距离

**来源**：IPC-2221 设计指南, SpecAp (2026), minintelpcb (2024), HiElectronic (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ELEC-01 | 去耦电容-IC 距离（低速 ≤50MHz） | ≤ 5mm | hard | `sqrt(dx² + dy²) ≤ 5` | A |
| ELEC-02 | 去耦电容-IC 距离（中速 50-100MHz） | ≤ 3mm | hard | `sqrt(dx² + dy²) ≤ 3` | A |
| ELEC-03 | 去耦电容-IC 距离（高速 >100MHz） | ≤ 1.5mm | hard | `sqrt(dx² + dy²) ≤ 1.5` | A |
| ELEC-04 | 去耦电容距离优化（靠近奖励） | 距离越小奖励越高 | soft | `reward += max(0, 1 - dist/5)` | A |

> **改进建议**：第一版采用 ELEC-01 的 5mm 最宽松阈值覆盖所有场景。实现时采用 AABB 包围盒计算电容到 IC 电源引脚组的最小距离（而非元件中心距离）。后续可按 IC 速度等级动态调整阈值。（详见附录B-1.5）

### 2.2 高压间距

**来源**：IPC-2221 §6.3, Andwin Circuits (2024), Magellan Circuits (2024)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ELEC-05 | 低压元件间距 (≤15V) | ≥ 0.2mm | hard | 爬电距离检查 | A |
| ELEC-06 | 中压元件间距 (31-50V) | ≥ 0.6mm | hard | 爬电距离检查 | A |
| ELEC-07 | 高压元件间距 (220V) | ≥ 2.5mm | hard | 爬电距离检查 | A |

> **改进建议**（ELEC-05~07）：与重叠/间距约束的统一 AABB 检测框架保持一致。高压网络自动触发间距放大——在检测到网络电位差时动态增大间距要求。

### 2.3 数字/模拟分区

**来源**：捷配 PCB 学堂 (2025), 凡亿 EDA (2025), eefocus (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ELEC-08 | 数字-模拟区最小间距 | ≥ 2mm | soft | 数字元件 density centroid 与模拟区 centroid 的距离 | A |
| ELEC-09 | 大电流-小信号隔离 | ≥ 3mm | hard | 大电流路径元件（>1A）与小信号元件间距 | A |

### 2.4 晶振布局

**来源**：fany-EDA (2025), 捷配 (2025), dzsc (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ELEC-10 | 晶振-IC 距离 | ≤ 10mm | hard | `sqrt(dx² + dy²) ≤ 10` | A |
| ELEC-11 | 晶振-板边距离 | ≥ 25.4mm (1inch) | hard | 防止辐射耦合 | A |
| ELEC-12 | 晶振-大功率/热源距离 | ≥ 3mm | hard | 防止频率漂移 | A |

### 2.5 电源模块布局

**来源**：eefocus (2025), dzsc (2025), 捷配 (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| ELEC-13 | DC-DC 紧凑布局 | 功率电感-开关引脚 ≤ 5mm | hard | DC-DC 周边元件间距检查 | A |
| ELEC-14 | 电源远离敏感器件 | DC-DC/开关电源-ADC/运放 ≥ 10mm | soft | 距离越远奖励越高 | A |
| ELEC-15 | LDO 散热空间预留 | LDO 周围 ≥ 2mm 空区 | soft | 散热路径检查 | A |

---

## 3. 热管理约束

### 3.1 功率器件分散

**来源**：Suwa & Hadim (2006), 阎德劲等 (2008), 吴沐羿等 (2022)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| THERM-01 | 大功率元件分散 | 两高功率元件（>0.5W）间距 ≥ 10mm | soft | 功率密度 hotspot 检测 | A |
| THERM-02 | 大功率靠板边 | 高功率元件距板中心距离最大化 | soft | `reward ∝ distance_to_center` | A |
| THERM-03 | 小功率围绕大功率 | — | soft | 热分布均匀性检查 | C |

### 3.2 热敏感元件保护

**来源**：eefocus (2025), TecircuitPCBA (2024), dzsc (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| THERM-04 | 电解电容-热源距离 | ≥ 5mm | hard | 电解电容封装到功率器件间距 | A |
| THERM-05 | 晶振-热源距离 | ≥ 3mm | hard | 晶振到功率器件间距 | A |
| THERM-06 | ADC/运放-热源距离 | ≥ 3mm | hard | 模拟器件到功率器件间距 | A |

### 3.3 散热路径

**来源**：Structure-aware (2025), eefocus (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| THERM-07 | 功率器件散热空间 | 散热面方向无遮挡 | soft | 功率器件到散热器路径上无高元件 | A |

> **改进建议**（THERM-01~07）：在布局阶段采用简化方案——给热源和热敏元件打标签，在 reward 中加间距项。定义热源列表（DC-DC、MOSFET 等）和热敏列表（传感器、晶振、NTC 等），热源到热敏元件距离 < 5mm 时产生连续惩罚。定义"推荐靠近"列表（如 DC-DC 及其输入电容），距离 > 3mm 时产生惩罚。不需要完整的 thermal simulation。（详见附录B-1.5）

---

## 4. 装配约束 (DFA)

### 4.1 元件方向

**来源**：EMS Tech (2025), 凡亿教育 (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| DFA-01 | 极性元件方向一致性 | 所有极性元件朝向种类 ≤ 2 | soft | 统计元件 rotation 值的种类数 | A |
| DFA-02 | Chip 元件方向统一 | 电阻/电容 0° 或 90° 方向 | soft | 奖励非 45° 等斜角度 | A |

### 4.2 禁布区

**来源**：凡亿教育 (2025), eepw (2025), EMS Tech (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| DFA-03 | 连接器插拔禁布区 | 连接器插拔方向 3mm 内禁布 | hard | 检查连接器前方 3mm 区域内是否有元件 | A |
| DFA-04 | 安装孔禁布区 | 安装孔周围 3.175mm (125mil) 禁布 | hard | 检查安装孔周边区域 | A |
| DFA-05 | 板边工艺边禁布区 | 板边 5mm 内无贴片元件 | hard | 若无贴片可省略工艺边 | A |
| DFA-06 | 压接连接器禁布区 | 压接面 3mm 内无高元件，1.5mm 内无焊接件 | hard | 逐元件检查 | A |
| DFA-07 | BGA 周边禁布区 | BGA 周边 3-5mm 禁布同面元件 | hard | BGA bounding box 外扩 5mm | A |
| DFA-08 | BGA 背面投影禁布区 | 禁止正反面 BGA 投影重叠 | hard | 检查正反面 BGA 的投影是否重叠 | A |
| DFA-09 | PCB 开窗边距 | 元件距开窗边缘 ≥ 2mm | hard | 开窗区域外扩检查 | A |

> **改进建议**（DFA-03~09）：禁止区域应从 KiCad 自动读取而非手动硬编码。使用 pcbnew API 检测封装自带的 Rule Area（如 ESP32 封装中定义的天线禁区）和板级 Zone 定义。引入 circuit_training 的 rate 参数实现分级禁止（rate=1 完全禁止，rate=0.5 最多覆盖 50% 面积）。（详见附录B-1.4）

### 4.3 限高

**来源**：eepw (2025), 凡亿教育 (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| DFA-10 | 区域限高 | 元件高度 < 该区域的限高值 | hard | 逐区域检查 | A |
| DFA-11 | 双面布局高度分层 | TOP 放高元件/IC，BOTTOM 放 0402/0603 | soft | 按高度排序放置层 | A |

### 4.4 连接器布局

**来源**：eepw (2025), CeramicPCB (2025), EMS Tech (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| DFA-12 | 连接器靠板边 | 连接器边缘与板边距离 ≤ 2mm | soft | 连接器到最近板边的距离 | A |
| DFA-13 | 连接器板边距 | 连接器到板边缘 ≥ 2.54mm (100mil) | hard | 防止连接器损坏 | A |
| DFA-14 | 连接器方向朝外 | 连接器插拔方向指向板外 | soft | 检查连接器朝向与最近板边的关系 | A |

---

## 5. 高速 / EMC 约束

### 5.1 时钟元件隔离

**来源**：PCBWay (2025), 捷配 (2025), eecraftsman (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| HS-01 | 时钟源-连接器距离 | ≥ 25.4mm (1inch) | hard | 时钟芯片到最近连接器的距离 | A |
| HS-02 | 时钟源-板边距离 | ≥ 12.7mm (0.5inch) | hard | 防止边缘辐射 | A |
| HS-03 | 高速器件居中 | 高速芯片靠近板中心放置 | soft | `reward ∝ -distance_to_center` | A |

### 5.2 差分对元件布局

**来源**：PCBWay (2025), HiElectronic (2025), pcbcool (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| HS-04 | 差分对起止元件紧邻 | 差分对关联元件间距 ≤ 5mm | soft | 两元件中心距离 | A |
| HS-05 | 差分对元件朝向一致 | 差分对元件的 rotation 相同 | soft | 比较两元件的角度 | A |

### 5.3 天线净空

**来源**：PCBWay (2025), eecraftsman (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| HS-06 | 天线净空区 | 天线周围铜皮/元件禁布半径 ≥ 5mm | hard | 天线元件为中心画圆检查 | A |
| HS-07 | 天线靠板边 | 天线距板边 ≤ 5mm | soft | 天线到最近板边距离 | A |

> **改进建议**（HS-06）：天线净空区应从封装自带 Rule Area 自动检测。kicad-mcp 的做法：遍历封装 Zone 对象，标记为 Rule Area 的区域即为禁区，判断禁区朝哪个方向延伸最远，将周围元件向反方向推。（详见附录B-1.4）

### 5.4 EMI 滤波布局

**来源**：eecraftsman (2025), 捷配 (2025), fany-EDA (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| HS-08 | EMI 滤波器靠近连接器 | 滤波器件-连接器间距 ≤ 5mm | soft | 滤波元件到连接器距离 | A |
| HS-09 | TVS/ESD 靠近连接器 | TVS 管-连接器间距 ≤ 3mm | hard | ESD 保护路径最短 | A |
| HS-10 | 功能模块分区 | 高速/数字/电源/模拟区不交叉 | soft | 模块 centroid 的空间隔离度 | A |

### 5.5 高速信号元件分组

**来源**：pcbcool (2025), 捷配 (2025)

| ID | 约束名称 | 量化阈值 | hard/soft | 判别条件 | 算法化 |
|----|---------|---------|-----------|---------|--------|
| HS-11 | DDR 总线元件组靠近 MCU | DDR 芯片距 MCU ≤ 15mm | soft | 最小化总线长度 | A |
| HS-12 | USB/PCIe 元件靠近连接器 | 高速接口元件-连接器 ≤ 10mm | soft | 减少信号路径 | A |

---

## 6. 布线质量代理指标 (RQP)

> **说明**：以下约束是布线可行性的"代理指标"（proxy metrics）——在布局阶段无法确切知道布线是否成功，但这些指标与实际布线成功率高度相关。它们属于优化层（权重低于空间约束），但其中拥塞密度是预测布线可行性的最重要指标。

### RQP-01：拥塞密度

| 字段 | 内容 |
|------|------|
| **量化阈值** | 2mm×2mm 网格容量 = 8 根走线（4 层板），超过容量部分按平方惩罚 |
| **hard/soft** | soft |
| **判别条件** | 板面划分为 2mm 网格 → 每个网络用 MST+L-shape 估计走线路径 → 各网格累计经过的走线数 → 超过容量部分按 `excess²` 计算成本 |
| **算法化** | A |

**实现方法**：
1. 将板面划分为 2mm × 2mm 的均匀网格
2. 对每个网络，用 Prim 算法求所有焊盘的 MST（最小生成树）
3. 每条树边用 L 形路径（先水平再垂直）近似走线，累加经过的网格
4. 网格容量设为 8（4 层板实用值），超过容量部分按 `excess²` 惩罚
5. **必须使用增量更新**——移动一个元件时只更新与该元件相连的网络对应的网格

**改进建议**：这是当前缺失约束中**最重要的一项**。pyplacer 的 CongestionMap 类仅 153 行，可直接移植。容量参数可按板层数调整：2 层板约 4-5，4 层板约 8-10，6 层板约 12-15。（详见附录B-2.1）

**来源**：pyplacer congestion.py（开源实现）

---

### RQP-02：引脚退出方向

| 字段 | 内容 |
|------|------|
| **量化阈值** | 焊盘退出方向与网络质心方向的点积 < 0 时触发惩罚 |
| **hard/soft** | soft |
| **判别条件** | 计算每个焊盘的自然退出方向（基于焊盘在封装中的本地坐标）→ 计算网络所有焊盘的质心 → 若退出方向指向质心反方向（点积<0），说明走线需绕过元件本体 |
| **算法化** | A |

**实现方法**：
1. 根据焊盘本地坐标判断退出方向（|x|≥|y| → 垂直退出，否则水平退出），乘以封装旋转角度得到世界坐标方向
2. 对每个网络计算焊盘质心，从焊盘指向质心为"期望方向"
3. 退出方向与期望方向点积 < 0 → 惩罚 = -(点积)

**改进建议**：高级约束，权重从 1-5 开始。在重叠、出界、拥塞、HPWL 稳定后再加入。（详见附录B-2.2）

**来源**：pyplacer placer.py:98（开源实现）

---

### RQP-03：HPWL 线长优化

| 字段 | 内容 |
|------|------|
| **量化阈值** | `HPWL = Σ_{net} [(x_max - x_min) + (y_max - y_min)]`，越小越好 |
| **hard/soft** | soft |
| **判别条件** | 对每个网络，所有焊盘 x 坐标极差 + y 坐标极差，所有网络求和 |
| **算法化** | A |

**关键说明**：当前代码使用"紧凑度"（所有元件到板中心的平均距离）作为线长代理，该指标完全不考虑元件连接关系。HPWL 利用已有 netlist 信息，是与实际布线长度高度相关的标准度量。计算复杂度 O(N) 对每个网络。

**改进建议**：用 HPWL 替换紧凑度，改动量约 20 行。建议采用 RL_PCB 的相对改进率公式 `(初始-当前)/(初始-最优)` 将 reward 归一化到 [-1,1]，使同一组超参数可跨不同 PCB 尺寸通用。同时实现网络优先级（RQP-04）以避免 GND 等大网络主导 HPWL。（详见附录B-2.3）

**来源**：pyplacer placer.py:50；RL_PCB agent.py

---

### RQP-04：网络优先级

| 字段 | 内容 |
|------|------|
| **量化阈值** | 电源网络（GND/VCC/VDD）HPWL 权重 = 0.2，普通信号网络权重 = 1.0 |
| **hard/soft** | soft |
| **判别条件** | 在 HPWL 计算中，根据网络名称应用不同的权重系数 |
| **算法化** | A |

**改进建议**：定义 `POWER_NET_NAMES = {"GND", "+3V3", "+5V", "VCC", "VDD", "VSS"}`，从网络表中自动识别（而非手动硬编码）。电源网络通过内层平面供电，走线长度不影响电气性能，但其线长仍占用布线通道。权重 0.2 而非 0，完全忽略可能导致拥塞密度偏差。（详见附录B-2.4）

**来源**：pyplacer placer.py:46；RL_PCB agent.py

---

## 7. RL 流程约束 (RLP)

> **说明**：以下约束不衡量布局好坏，而决定训练如何进行——如何让 agent 学得更快、更高效。它们不影响最终布局的质量上限，但影响达到该上限所需的训练步数。

### RLP-01：放置顺序策略

| 字段 | 内容 |
|------|------|
| **量化阈值** | 按面积或连接度降序放置；固定元件最先放 |
| **hard/soft** | hard（环境机制） |
| **判别条件** | 在 reset() 中对元件列表排序：固定元件 → 最大元件 → 最小元件 |
| **算法化** | A |

**实现方法**：
- circuit_training：按面积降序排列，RL 处理宏单元决策，DREAMPlace 处理标准单元
- kicad-mcp：按连接度降序，连接最多的元件放中央

**改进建议**：改动很小但效果显著——在 step() 前对元件列表排序，固定元件（连接器、安装孔）最先放，之后按面积或连接度降序。当项目扩展到 50+ 元件时，影响尤其明显。（详见附录B-3.1）

**来源**：circuit_training environment.py:234；kicad-mcp

---

### RLP-02：动作空间掩码

| 字段 | 内容 |
|------|------|
| **量化阈值** | mask 向量：0 = 非法位置（已占据/出界），1 = 合法位置 |
| **hard/soft** | hard（动作空间限制） |
| **判别条件** | 在 observation 中提供 mask 向量，策略网络将 mask=0 位置的 logits 设为 -inf |
| **算法化** | A |

**实现方法**：
- 离散动作空间：在策略网络 softmax 前将 mask=0 位置的 logits 替换为 -inf
- 连续动作空间：动作后检查 + 回退到最近合法位置

**改进建议**：采用动作掩码可在相同训练步数下减少约 2-5 倍的探索时间——agent 永远不需要浪费时间尝试"明显错误"的动作（如把新元件放在已有元件的位置上）。（详见附录B-3.2）

**来源**：circuit_training environment.py:381

---

### RLP-03：元件功能聚类

| 字段 | 内容 |
|------|------|
| **量化阈值** | 连接紧密的元件自动识别为功能组，同组元件就近放置 |
| **hard/soft** | soft |
| **判别条件** | DFS 遍历连接图 → 连通分量 = 功能簇 → 识别连接度 > 5 的 Hub 节点放簇中心 |
| **算法化** | A |

**实现方法**：
- circuit_training：使用 hMetis 超图划分工具，分三层聚类（宏单元引脚 → I/O 端口 → 逻辑层级）
- KICAD-autowire：DFS 遍历连接图，连通分量即为功能簇（实现简单，约 20 行代码）

**改进建议**：建议从简单 DFS 聚类开始，将 100 个元件的直接决策降维为"先定组位置，再定组内元件位置"的分层决策。如果效果不理想再升级到 hMetis。（详见附录B-3.3）

**来源**：circuit_training grouping.py:118；KICAD-autowire

---

## 附录A：按算法化等级分类的约束矩阵

### 算法化 A（纯几何计算，可直接实现）

| 类别 | 数量 | ID 列表 |
|------|------|---------|
| 环境建模 | 8 | ENV-01~04, OBJ-01~04 |
| 制造 | 17 | MFG-01~12（除 MFG-13）, MFG-14~18 |
| 电气 | 15 | ELEC-01~15 |
| 热 | 6 | THERM-01~02, 04~07 |
| 装配 | 14 | DFA-01~14 |
| 高速/EMC | 12 | HS-01~12 |
| 布线质量代理 | 4 | RQP-01~04 ← 新增 |
| RL流程约束 | 3 | RLP-01~03 ← 新增 |
| **合计** | **79** | |

### 算法化 B/C（需仿真或启发式）

| 类别 | 数量 | ID 列表 |
|------|------|---------|
| 制造 | 1 | MFG-13 (铜平衡, C) |
| 热 | 1 | THERM-03 (热分布均匀, C) |
| **合计** | **2** | |

> 结论：**81 条布局约束中，79 条为算法化 A 级**（纯几何计算），可直接在 RL 环境的 `action_space`、`check_constraints()` 和 `_calculate_reward()` 中实现。仅 2 条需启发式近似。

---

## 附录B：详细改进实施方案

> **说明**：以下内容源自 `PCB布局约束体系_最终完整版.md` 的代码级分析，包含文件路径、行号引用和参考实现的完整代码片段。正文中各约束的"改进建议"字段为本附录的方法论摘要。

---

### B.1 空间约束改进

#### B.1.1 重叠约束：从中心点二值到 AABB 连续惩罚

**当前代码问题定位**：
- `pcb_utils.py` 第 107-113 行：遍历元件对，调用 `check_component_spacing()`
- `kicad_utils.py` 第 262-269 行：`check_component_spacing()` 使用中心点距离 + 0.2mm 硬阈值

**核心问题**：
1. 中心点距离不能反映包围盒交叉——大芯片角可能已压住小电容
2. 二值判断没有梯度——从"完全重叠"到"稍微重叠" reward 不变
3. 没有区分软重叠（间距不足）和硬重叠（物理交叉）

**参考实现（pyplacer placer.py:59-83）**：

```python
KEEPOUT_SOFT = 2.0
W_OVERLAP_HARD = 1000.0
W_OVERLAP_SOFT = 50.0

def bbox_overlap_penalty(a, b):
    """输入: (x_min, y_min, x_max, y_max)，返回 (hard_area, soft_penalty)"""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    
    if dx > 0 and dy > 0:
        return (dx * dy, 0.0)              # 硬重叠：面积惩罚
    if dx > 0 >= dy:
        gap = -dy; edge = dx
    elif dy > 0 >= dx:
        gap = -dx; edge = dy
    else:
        return (0.0, 0.0)                  # 角对角：无惩罚
    
    if gap >= KEEPOUT_SOFT:
        return (0.0, 0.0)
    deficit = KEEPOUT_SOFT - gap
    return (0.0, deficit * deficit * edge)  # 二次惩罚
```

**移植步骤**：
1. 使用 KiCad pcbnew 的 `fp.GetBoundingBox()` 获取包围盒
2. 在 reward 中累加 `W_OVERLAP_HARD * total_hard + W_OVERLAP_SOFT * total_soft`
3. 可选：参考 RL_PCB 的 8 方向视线检测为 agent 提供方向性信息

---

#### B.1.3 出界约束：从中心点二值到面积比例

**当前代码问题定位**：
- `pcb_utils.py` 第 115-119 行：检查中心点是否在矩形板框内

**核心问题**：
1. 中心点在板内但元件角已出界——检测不到
2. 二值判断，出界 0.1mm 和 5mm 惩罚相同
3. 矩形近似无法处理 L 形、圆形等异形板框

**参考实现（pyplacer placer.py:86-95）**：

```python
W_OUT_OF_BOUNDS = 200.0

def out_of_bounds_area(bbox, outline):
    """计算包围盒超出板框的面积 (mm²)。完全在板内返回 0"""
    bxmin, bymin, bxmax, bymax = bbox
    oxmin, oymin, oxmax, oymax = outline
    bbox_area = max(0.0, (bxmax-bxmin)) * max(0.0, (bymax-bymin))
    ix = max(0.0, min(bxmax, oxmax) - max(bxmin, oxmin))
    iy = max(0.0, min(bymax, oymax) - max(bymin, oymin))
    return bbox_area - ix * iy

oob = sum(out_of_bounds_area(bb, board.outline) for bb in bboxes)
total += W_OUT_OF_BOUNDS * oob
```

**互补方案（RL_PCB agent.py:198-206）**：
```python
# 训练时：出界 = 巨额惩罚 + 终止 episode
if (obs["position"][0] > 1 or obs["position"][0] < 0 or
    obs["position"][1] > 1 or obs["position"][1] < 0):
    reward -= (self.max_steps - self.steps_done) * 15
    done = True
```

**互补方案（kicad-mcp clamp）**：
```python
def clamp_to_board(x, y, w, h, outline, spacing):
    """推理阶段安全兜底：强制拉回板内"""
    half_w, half_h = w/2, h/2
    x = max(outline["x_min"]+half_w+spacing,
            min(x, outline["x_max"]-half_w-spacing))
    y = max(outline["y_min"]+half_h+spacing,
            min(y, outline["y_max"]-half_h-spacing))
    return x, y
```

**移植步骤**：
1. 用 `fp.GetBoundingBox()` 获取包围盒，替换中心点判断
2. 采用出界面积比例作为连续惩罚
3. 训练时出界给巨量惩罚 + 终止，推理时用 clamp 兜底

---

#### B.1.4 禁止区域/Keepout：从硬编码到自动读取

**当前代码问题定位**：
- `pcb_utils.py` 禁止区域检测段：从 `self.forbidden_areas` 硬编码列表读取矩形区域

**参考实现（kicad-mcp KeepoutAutoDetect）**：
```python
if hasattr(fp, "Zones"):
    for zone in fp.Zones():
        if zone.GetIsRuleArea():
            extents = {"left": abs(zx), "right": abs(zr),
                       "top": abs(zy), "bottom": abs(zb)}
            keepout_side = max(extents, key=extents.get)
            # 将元件向 keepout 反方向推
```

**参考实现（circuit_training placement_util.py:263-268）**：
```python
blockages = get_blockages_from_comments([netlist_file, init_placement])
for blockage in blockages:
    plc.create_blockage(*blockage)  # [x1,y1,x2,y2,rate]
# rate=1 完全禁止，rate=0.5 最多覆盖 50% 面积
```

**移植步骤**：
1. 使用 pcbnew API 遍历 Zone 对象，调用 `GetIsRuleArea()` 自动读取
2. 遍历封装中的 Rule Area 检测封装自带 keepout
3. 引入 rate 参数支持分级禁止
4. 用 AABB 碰撞检测代替中心点检测

---

#### B.1.5 热/功耗域约束

**当前状态**：完全缺失。

**简化实现方案**（约 30 行代码）：
```python
HEAT_SOURCES = {"U1_DCDC", "U2_MOSFET", "R1_PWR"}
HEAT_SENSITIVE = {"U3_SENSOR", "Y1_CRYSTAL", "R2_NTC"}
RECOMMEND_CLOSE = {("U1_DCDC", "C1_IN"), ("U1_DCDC", "C2_OUT")}

# reward 惩罚
for src in HEAT_SOURCES:
    for sens in HEAT_SENSITIVE:
        if distance(src, sens) < 5.0:
            reward -= (5.0 - distance(src, sens)) * weight

# reward 奖励
for (a, b) in RECOMMEND_CLOSE:
    if distance(a, b) > 3.0:
        reward -= (distance(a, b) - 3.0) * weight
```

---

### B.2 布线质量代理改进

#### B.2.1 拥塞密度（CongestionMap 移植指南）

**参考实现（pyplacer congestion.py，共 153 行）**：

```python
class CongestionMap:
    def __init__(self, outline, cell_size=2.0):
        self.xmin, self.ymin, xmax, ymax = outline
        self.cell_size = cell_size
        self.nx = max(1, int((xmax-self.xmin)//cell_size))
        self.ny = max(1, int((ymax-self.ymin)//cell_size))
        self.grid = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.net_contribs = {}  # 增量更新用

    def _l_shape_cells(self, x1, y1, x2, y2):
        """L 形路径经过的所有网格"""
        cx1, cy1 = int((x1-self.xmin)//self.cell_size), int((y1-self.ymin)//self.cell_size)
        cx2, cy2 = int((x2-self.xmin)//self.cell_size), int((y2-self.ymin)//self.cell_size)
        cells = []
        for cx in range(min(cx1,cx2), max(cx1,cx2)+1):
            cells.append((cy1, cx))
        for cy in range(min(cy1,cy2), max(cy1,cy2)+1):
            if cy != cy1: cells.append((cy, cx2))
        return cells

    def add_net(self, net_id, pin_positions):
        """添加一个网络到拥塞图（Prim MST + L-shape）"""
        if len(pin_positions) < 2: return
        # Prim 求 MST
        edges = self._prim_mst(pin_positions)
        contrib = {}
        for i, j in edges:
            for cell in self._l_shape_cells(*pin_positions[i], *pin_positions[j]):
                self.grid[cell] += 1.0
                contrib[cell] = contrib.get(cell, 0) + 1
        self.net_contribs[net_id] = contrib

    def remove_net(self, net_id):
        """增量更新：移除网络"""
        if net_id in self.net_contribs:
            for cell, val in self.net_contribs[net_id].items():
                self.grid[cell] -= val
            del self.net_contribs[net_id]

    def cost(self, capacity=8.0):
        """二次超额惩罚"""
        excess = np.maximum(self.grid - capacity, 0.0)
        return float(np.sum(excess * excess))

W_CONGESTION = 40.0
```

**移植要点**：
- 增量更新是关键——每次移动一个元件只更新与该元件相连的网络
- 容量参数：2 层板约 4-5，4 层板约 8-10，6 层板约 12-15

---

#### B.2.2 引脚退出方向

**参考实现（pyplacer placer.py:98）**：

```python
def pad_exit_dir(fp, pad):
    """计算焊盘退出方向（世界坐标系）"""
    lx, ly = pad.x_local, pad.y_local
    if abs(lx) >= abs(ly):
        local = (1.0 if lx>=0 else -1.0, 0.0)
    else:
        local = (0.0, 1.0 if ly>=0 else -1.0)
    return fp.rotate_point(local[0], local[1])

W_PAD_EXIT = 5.0
# 对每个网络，若焊盘退出方向与质心方向点积 < 0：
# cost += -(点积)
```

**移植要点**：高级约束，在重叠/出界/拥塞/HPWL 稳定后再添加，权重从 1-5 开始。

---

#### B.2.3 HPWL 替换紧凑度

**当前代码问题定位**：
- `pcb_layout_env.py` 第 78-80 行：使用紧凑度（所有元件到板中心平均距离）
- `pcb_utils.py` 第 129-146 行：`calculate_layout_compactness()` 不利用 netlist 信息

**参考实现（pyplacer placer.py:50）**：
```python
def hpwl_for_net(pad_positions):
    if len(pad_positions) < 2: return 0.0
    xs = [p[0] for p in pad_positions]
    ys = [p[1] for p in pad_positions]
    return (max(xs)-min(xs)) + (max(ys)-min(ys))

hpwl = sum(weight * hpwl_for_net(pad_positions) for net in nets)
```

**参考实现（RL_PCB 相对改进率）**：
```python
# 归一化到 [-1, 1]，同一组超参数可跨 PCB 通用
x = np.clip((Wi - W) / (Wi - We), -1, 1)
y = np.clip((HPWLi - HPWL) / (HPWLi - HPWLe), -1, 1)
```

---

#### B.2.4 网络优先级

**当前状态**：缺失。实现 HPWL 后 GND 将占 60-80% 权重。

**参考实现（pyplacer placer.py:46）**：
```python
POWER_NET_NAMES = {"GND", "+3V3", "+5V"}
W_POWER_NET = 0.2
weight = W_POWER_NET if net_name in POWER_NET_NAMES else 1.0
```

**移植要点**：改动用不到 5 行。建议从网络表中自动识别所有名字包含 GND/VCC/VDD/VSS 的网络。

---

### B.3 流程约束改进

#### B.3.1 放置顺序

**参考实现（circuit_training environment.py:234 + kicad-mcp）**：

```python
# circuit_training：面积降序
self._sorted_node_indices = placement_util.get_ordered_node_indices(
    mode="descending_size_macro_first")

# kicad-mcp：连接度降序
sorted_refs = sorted(refs,
    key=lambda r: (conn_score[r], area[r]), reverse=True)
```

**移植要点**：在 step() 前排序。固定元件（连接器、安装孔）最先放。

---

#### B.3.2 动作掩码

**参考实现（circuit_training environment.py:381）**：

```python
def _get_mask(self):
    if self._done: return np.zeros(max_grid_size**2, dtype=np.int32)
    mask = self._plc.get_node_mask(node_index)
    # ... reshape / pad ...
    return mask.astype(np.int32)
```

**移植要点**：离散空间在 softmax 前将 mask=0 的 logits 设为 -inf。连续空间改为动作后检查 + 回退。

---

#### B.3.3 元件聚类

**参考实现（KICAD-autowire DFS 聚类，约 20 行）**：

```python
def find_clusters(self):
    visited = set(); clusters = []
    def dfs(s, c):
        if s in visited: return
        visited.add(s); c.add(s)
        for conn in self.connections:
            if conn.pin_a.symbol == s: dfs(conn.pin_b.symbol, c)
            elif conn.pin_b.symbol == s: dfs(conn.pin_a.symbol, c)
    for s in self.nodes.values():
        if s not in visited:
            c = set(); dfs(s, c)
            if c: clusters.append(c)
    return clusters
```

**移植要点**：建议从简单 DFS 聚类开始，如需更精确再考虑 hMetis。

---

## 参考文献

### 学术论文

1. OmniLayout: A Schematic-Coupled Multimodal Benchmark for Constraint-Aware Geometric Reasoning in PCB Layout, arXiv:2607.03261, 2026
2. HiePlace: Efficient Hierarchical PCB Placement, IEEE TCAD, 2026
3. MARS-Place: Multi-stage alignment-refined strategy for PCB placement and routing optimization, Integration, 2026
4. Analytical approach and fine-tuning strategy for PCB placement optimization, Integration, 2025
5. Suwa T, Hadim H. Multidisciplinary Placement Optimization of Heat Generating Electronic Components on PCBs, Journal of Electronic Packaging, 2006
6. 阎德劲, 周德俭, 黄春跃. 基于遗传算法的表面组装电子元件热布局优化, 2008
7. 吴沐羿, 李先允, 王书征. 基于改进模拟退火算法的PCB电子元器件热优化布置研究, 电子设计工程, 2022

### 行业标准

8. IPC-2221A — Generic Standard on Printed Board Design
9. IPC-7351B — Generic Requirements for Surface Mount Design and Land Pattern Standard
10. IPC-7093 — Design and Assembly Process Implementation for Bottom Termination Components

### 开源项目参考

11. pyplacer — Python-based PCB placement optimization with congestion map and overlap penalty
12. RL_PCB — Reinforcement learning for PCB placement with multi-agent observation
13. circuit_training — Google's RL-based macro placement with blockage and mask support
14. kicad-mcp — KiCad MCP server with automatic keepout detection and clamping

### 设计手册与在线资源

15-33. 同原报告参考文献 11-31
