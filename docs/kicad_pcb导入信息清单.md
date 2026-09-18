# KiCad PCB 导入信息清单（pcbnew 原始信息面）

> 日期：2026-09-11
> 环境：KiCad 10.0.4（`E:\Kicad`）+ conda `pcb_rl_311`（Python 3.11）
> 实测对象：`data/kicad-native-batch/jackt-…--4.IMU小板v1_1-09db0370/*.kicad_pcb`（默认板 IMU）
> 目的：说明 `pcbnew.LoadBoard()` 一次导入后内存里**有哪些原始信息**、
> 当前 RL 管线**取了哪些**、以及**还剩哪些没取**
> 关联文档：`docs/kicad_interface.md`（导入导出双向契约，接口以该文档为准）

---

## 一、导入动作与对象模型

一次导入就是一个调用：

```python
board = pcbnew.LoadBoard(path)   # -> pcbnew.BOARD
```

`BOARD` 是全部原始信息的根容器。pcbnew 存根（`E:\Kicad\bin\Lib\site-packages\pcbnew.py`）
共定义 **177 个类**，`BOARD` 自身 **190 个方法**（其中只读查询约 90 个）。信息分布在以下几类对象上：

| 对象 | 类（方法数） | 获取入口 |
|---|---|---|
| 电路板 | `BOARD`(190) | `LoadBoard()` |
| 元件 | `FOOTPRINT`(197) | `board.GetFootprints()` |
| 焊盘 | `PAD`(236) | `footprint.Pads()` |
| 走线 | `PCB_TRACK`(36) / `PCB_ARC`(16) | `board.GetTracks()` |
| 过孔 | `PCB_VIA`(132) | `board.GetTracks()` 中按类型筛 |
| 敷铜 / 规则区 | `ZONE`(132) | `board.Zones()` |
| 板级图形 | `PCB_SHAPE`(22) / `PCB_TEXT`(17) / `PCB_TEXTBOX`(27) / `PCB_TARGET`(13) | `board.GetDrawings()` |
| 分组 / 标记 | `PCB_GROUP`(16) / `PCB_MARKER`(12) | `board.Groups()` / `board.Markers()` |
| 网络 | `NETINFO_ITEM`(19) | `board.GetNetsByName()` |
| 设计规则 | `BOARD_DESIGN_SETTINGS`(65) | `board.GetDesignSettings()` |
| 网络类 | `NETCLASS`(104) | `board.GetNetClasses()` |

---

## 二、板级信息（BOARD）

下表右列为 IMU 板实测值（KiCad 10.0.4）。

| 信息 | API | IMU 板实测 |
|---|---|---|
| 文件格式版本 / 生成器 | `GetFileFormatVersionAtLoad()` / `GetGenerator()` | `20260206` / `pcbnew` |
| 铜层数 / 启用层总数 | `GetCopperLayerCount()` / `BoardLayerCount()` / `GetEnabledLayers()` | 2 层铜 / 27 层 |
| 板厚 | `GetDesignSettings().GetBoardThickness()` | 1.6 mm |
| 辅助原点 / 网格原点 | `GetDesignSettings().GetAuxOrigin()` / `GetGridOrigin()` | (140, 110) / (0, 0) |
| 元件 | `GetFootprints()` | 10 |
| 网络总数 / 名单 | `GetNetCount()` / `GetNetsByName()` | 12（名单见第九节） |
| 自定义网络类 | `GetNetClasses()` | 0 个 |
| 走线 + 过孔 | `GetTracks()` | 219（182 段走线 + 37 过孔） |
| 敷铜 / 规则区 | `Zones()` / `GetAreaCount()` / `GetZoneList()` | 74 |
| 板级图形 | `GetDrawings()` | 9 |
| 板框 | `GetBoardPolygonOutlines()` / `BoardOutline()` / `GetBoardEdgesBoundingBox()` | 可读多边形 |
| 标题栏 | `GetTitleBlock()` | 本板全空 |
| 层栈 | `GetStackupOrDefault()` / `BoardLayerSet()` | 介质铜厚叠层 |
| 变体（KiCad 10 新增） | `GetVariantNames()` / `GetCurrentVariant()` | 无 |
| 分组 / DRC 标记 | `Groups()` / `Markers()` | 0 / 0 |
| 嵌入资源 | `GetEmbeddedFiles()` / `GetEmbeddedFonts()` / `GetFonts()` | 可读 |
| 工程与连通性 | `GetProject()` / `GetConnectivity()` | 可读（ratsnest 未布线信息） |
| 页面 / 绘图设置 | `GetPageSettings()` / `GetPlotOptions()` | 可读 |

---

## 三、元件（FOOTPRINT，197 方法）

实测 C1：`pos (141.2, 107.6) mm`、`rot -90.0°`、`layer Top Layer (F.Cu)`、`attr 0`、
`pads 2`、`DNP False`、`boardOnly False`、`exclBOM False`、`localClearance None`。

| 分组 | 接口 |
|---|---|
| 标识 | `GetReference()` / `GetValue()` / `GetField()` / `GetFields()` / `GetFPID()` / `GetLIB_ID()` / `GetName()` / `GetLibNickname()` |
| 描述 | `GetDesc()` / `GetKeywords()` / `GetLibDescription()` / `GetSearchTerms()` / `GetFilters()` |
| 位姿 | `GetPosition()` / `GetOrientationDegrees()` / `GetOrientation()` / `GetLayer()` / `GetSide()` / `IsFlipped()` / `GetFPRelativePosition()` |
| 几何 | `GetBoundingHull()` / `GetLayerBoundingBox()` / `GetFpPadsLocalBbox()` |
| 焊盘 | `GetPads()` / `GetPadCount()` / `GetUniquePadCount()` / `GetUniquePadNumbers()` / `FindPadByNumber()` / `GetNetTiePads()` / `IsNetTie()` |
| 属性 | `GetAttributes()` / `IsDNP()` / `IsBoardOnly()` / `IsExcludedFromBOM()` / `IsExcludedFromPosFiles()` / `GetLikelyAttribute()` / `IsPlaced()` |
| 规则覆盖 | `GetLocalClearance()` / `GetLocalSolderMaskMargin()` / `GetLocalSolderPasteMargin()` / `GetLocalZoneConnection()` / `GetClearanceOverrides()` |
| 结构与库 | `GetPath()` / `GetSheetname()` / `GetSheetfile()` / `GetUnitInfo()` / `GetPrivateLayers()` / `GetStackupMode()` |
| 变体 | `GetVariant()` / `GetVariants()` / `GetDNPForVariant()` / `GetFieldValueForVariant()` |

此外，元件自身的丝印/装配图形与文字（`fp_line` / `fp_arc` / `fp_circle` / `fp_rect` /
`fp_poly` / `fp_text`）不通过上述接口暴露，需遍历 footprint 的图形子项；本项目
IMU 板的元件内部共含 `fp_line 118` / `fp_poly 66` / `fp_arc 20` / `fp_rect 10` / `fp_circle 6`。

---

## 四、焊盘（PAD，236 方法）

实测 1 号焊盘：`net GND`、`shape 1`、`size 0.8 × 0.9 mm`、`drill 0.0`、
`attr 1`（SMD）、`layers [F.Cu, F.Mask, F.Paste]`、`localClearance None`。

| 分组 | 接口 |
|---|---|
| 标识 | `GetNumber()` / `GetPadName()` / `GetPinFunction()` / `GetPinType()`（原理图引脚语义） |
| 几何 | `GetShape()` / `GetSize()` / `GetSizeX/Y()` / `GetDelta()` / `GetOffset()` / `GetRoundRectCornerRadius()` / `GetChamferPositions()` / `GetPrimitives()` / `GetEffectivePolygon()` |
| 钻孔 | `GetDrillSize()` / `GetDrillShape()` / `GetPrimaryDrillSize()` / `GetSecondaryDrill*()` / `GetTertiaryDrill*()` / `GetPrimaryDrillStartLayer()` / `GetPrimaryDrillEndLayer()` |
| 属性 | `GetAttribute()`（SMD/PTH/NPTH/CONN）/ `GetProperty()` / `IsNoConnectPad()` / `IsFreePad()` / `IsAperturePad()` / `IsNPTHWithNoCopper()` |
| 层 | `GetLayerSet()` / `GetPrincipalLayer()` / `GetZoneLayerOverride()` |
| 网络 | `GetNetCode()` / `GetNetname()` |
| 规则 | `GetLocalClearance()` / `GetOwnClearance()` / `GetZoneConnection()` / `GetLocalSolderMaskMargin()` / `GetSolderMaskExpansion()` / `GetThermalSpoke*()` |
| 工艺 | `GetBackdrillMode()` / `GetFrontPostMachining*()` / `GetBackPostMachining*()` / `GetPrimaryDrillFilled()` / `GetPrimaryDrillCapped()` |

---

## 五、走线（PCB_TRACK / PCB_ARC）—— 本板 182 条

实测一段：`(142.2105, 95.5685) → (142.2105, 96.2555)` mm、`width 0.254 mm`、
`layer Top Layer (F.Cu)`、`net GND`、`length 0.687 mm`。

| 接口 | 说明 |
|---|---|
| `GetStart()` / `GetEnd()` | 端点坐标 |
| `GetWidth()` | 线宽 |
| `GetLength()` | 段长 |
| `GetLayer()` / `GetNetCode()` / `GetNetname()` | 所在层与网络 |
| `GetDelay()` | 信号延迟（叠层介电常数推导） |
| `GetWidthConstraint()` | 有效线宽约束 |
| `HasSolderMask()` / `GetLocalSolderMaskMargin()` / `GetSolderMaskExpansion()` | 阻焊相关 |
| `GetEffectiveShape()` | 有效几何 |

圆弧走线用 `PCB_ARC`，额外提供 `GetMid()` / `GetRadius()` / `GetAngle()` /
`GetArcAngleStart()` / `GetArcAngleEnd()` / `IsCCW()`。IMU 板无圆弧走线。

---

## 六、过孔（PCB_VIA，132 方法）—— 本板 37 个

| 分组 | 接口 |
|---|---|
| 类型 | `GetViaType()` / `IsMicroVia()` / `IsBlindVia()` / `IsBuriedVia()` / `HasValidLayerPair()` |
| 尺寸 | `GetFrontWidth()` / `GetWidth(layer)` / `GetDrillValue()` / `GetMinAnnulus()` / `GetPrimaryDrillSize()` |
| 连接 | `GetOutermostConnectedLayers()` / `GetRemoveUnconnected()` / `GetKeepStartEnd()` / `GetIsFree()` |
| 图层 | `GetLayerSet()` / `GetZoneLayerOverride()` |
| 工艺 | `GetFrontTentingMode()` / `GetBackTentingMode()` / `GetFrontPluggingMode()` / `GetCappingMode()` / `GetFillingMode()` / 背钻相关 |

---

## 七、敷铜与规则区（ZONE，132 方法）—— 本板 74 个

实测第一个 zone：`name ''`、`net $1N533`、`layer F.Cu`、`ruleArea False`、
`priority 600`、`filled True`、`padConn 1`、`corners 16`、`localClearance 0`（未设置）、
`minThickness 0`。

| 分组 | 接口 |
|---|---|
| 标识 | `GetZoneName()` / `GetNetname()` / `GetNetCode()` |
| 层 | `GetLayerSet()` / `GetFirstLayer()` |
| 填充 | `GetFillMode()` / `IsFilled()` / `HasFilledPolysForLayer()` / `GetFilledPolysList()` |
| 面积 | `GetOutlineArea()` / `GetFilledArea()` / `GetNumCorners()` / `GetCornerPosition()` |
| 规则 | `GetLocalClearance()` / `GetMinThickness()` / `GetPadConnection()` / `GetThermalReliefGap()` / `GetThermalReliefSpokeWidth()` / `GetAssignedPriority()` |
| 规则区 | `GetIsRuleArea()` + `GetDoNotAllowVias()` / `GetDoNotAllowTracks()` / `GetDoNotAllowPads()` / `GetDoNotAllowFootprints()` / `GetDoNotAllowZoneFills()` / `GetPlacementAreaEnabled()` |
| 孤岛 | `GetIslandRemovalMode()` / `GetMinIslandArea()` / `IsIsland()` |

---

## 八、板级图形与其它板级对象 —— 本板 9 个

`board.GetDrawings()` 返回板级图形：

| 类 | 主要接口 |
|---|---|
| `PCB_SHAPE` | `GetShapeStr()` / `GetCorners()` / `GetConnectionPoints()` / `GetArcAngleStart()` / `GetStroke()` |
| `PCB_TEXT` | `GetShownText()` / `GetTextTypeDescription()` |
| `PCB_TEXTBOX` | `GetShownText()` / `GetMarginLeft/Top/Right/Bottom()` / `IsBorderEnabled()` |
| `PCB_TARGET` | `GetShape()` / `GetSize()` / `GetWidth()` |
| `PCB_GROUP` | `GetItems()` / `GetBoardItems()` / `GetItemsDeque()` |
| `PCB_MARKER` | `GetShapes()` / `GetPath()` / `GetColorLayer()`（DRC 违规标记） |

存在但 `BOARD` 上**没有集合入口**的类，须遍历 `GetItemSet()` 按类型筛：
尺寸标注 `PCB_DIM_ALIGNED` / `PCB_DIM_ORTHOGONAL` / `PCB_DIM_RADIAL` / `PCB_DIM_LEADER` /
`PCB_DIM_CENTER`（共同基类 `PCB_DIMENSION_BASE`）、表格 `PCB_TABLE`、
参考图 `PCB_REFERENCE_IMAGE`。

---

## 九、网络与连通性（NETINFO_ITEM，19 方法）—— 本板 12 个网络

实测名单：`''`（未连接）、`$1N533`、`3V3`、`GND`、`ICM_CS`、`ICM_INT`、
`IMU_MISO`、`IMU_MOSI`、`IMU_SCK`、`LIS3_CS`、`LIS3_DRDY`、`MCO_32K`。

| 接口 | 说明 |
|---|---|
| `GetNetCode()` / `GetNetname()` | 网络编号与名称 |
| `GetShortNetname()` / `GetDisplayNetname()` | 短名 / 显示名（去掉层级前缀） |
| `GetNetClass()` / `GetNetClassName()` | 所属网络类 |
| `HasAutoGeneratedNetname()` / `IsCurrent()` | 名称是否自动生成 / 是否为当前网络 |

`BOARD` 侧辅助接口：`FindNet()` / `GetNetInfo()` / `GetNetcodeFromNetname()` /
`GetConnectivity()`（未布线 ratsnest）/ `GetTrackLength()`（按网络统计走线总长）。

---

## 十、设计规则与网络类

`BOARD_DESIGN_SETTINGS`（65 方法）实测值：

| 参数 | API | IMU 板实测 |
|---|---|---|
| 最小间距 | `GetSmallestClearanceValue()` | 0.2 mm |
| 最大间距 | `GetBiggestClearanceValue()` | 1.0 mm |
| DRC 容差 | `GetDRCEpsilon()` | 0.0005 mm |
| 默认线宽 | `GetCurrentTrackWidth()` | 0.2 mm |
| 默认过孔 / 钻孔 | `GetCurrentViaSize()` / `GetCurrentViaDrill()` | 0.6 / 0.3 mm |
| 板厚 | `GetBoardThickness()` | 1.6 mm |
| 辅助 / 网格原点 | `GetAuxOrigin()` / `GetGridOrigin()` | (140, 110) / (0, 0) |

另有 `GetStackupDescriptor()`（叠层）、`GetHolePlatingThickness()`、`GetDRCEpsilon()`、
`GetTextSize()` / `GetTextThickness()`、`GetDiffPair*()`（差分对）、
`GetDefaultZoneSettings()`、`GetTeadropParamsList()`、`GetComponentClassManager()`。

实际用到的全部线宽 / 过孔尺寸集合走 `BOARD.GetTrackWidthList()` 与
`BOARD.GetViasDimensionsList()`（注意不在 `BOARD_DESIGN_SETTINGS` 上）。

`NETCLASS`（104 方法）：`GetClearance()` / `GetTrackWidth()` / `GetViaDiameter()` /
`GetViaDrill()` / `GetuViaDiameter()` / `GetuViaDrill()` / `GetDiffPairWidth()` /
`GetDiffPairGap()` / `GetPriority()` / `GetPcbColor()` / `GetLineStyle()` /
`GetTuningProfile()`。**IMU 板 `GetNetClasses()` 返回空**，即该板未定义自定义网络类，
不能假定能取到元素。

---

## 十一、当前管线已用 / 未用对照

`utils/kicad_utils/kicad_utils.py` 当前调用的 pcbnew API 约 25 个：
`LoadBoard`、`Footprints`、`GetReference`、`GetFPID().GetLibItemName`、`GetPosition`、
`GetOrientationDegrees`、`GetLayer`、`GetBoundingHull().BBox`、`IsLocked`、`Pads`、
`GetNumber`、`GetNetname`、`GetNetCode`、`GetLayerSet().Seq`、`GetAttribute`、
`GetNetsByName`、`GetDrawings`（仅 Edge.Cuts）、`GetLayerID`、`Zones().GetIsRuleArea`、
`Save`、`SetPosition`、`SetOrientationDegrees`、`SetLayer`、`ToMM`、`FromMM`。

由此得到的 RL 侧信息只有：`Component`（位号/封装/类型/AABB/旋转/层/固定标记/网络名/
引脚相对坐标/焊盘层事实）、`Net`（名称/连接表/是否电源）、板框 AABB、规则区 AABB。

**未取用**的信息及其在本数据集 20 块板上的规模：

| 未取用信息 | IMU 板 | 20 块板范围 | 对应入口 |
|---|---|---|---|
| 走线 | 182 条 | 2 ~ 1024 | `GetTracks()` |
| 过孔 | 37 个 | 0 ~ 1024 | `GetTracks()` |
| 敷铜 / 规则区 | 74 个 | 1 ~ 520 | `Zones()` |
| 板级图形（丝印/文字） | 9 个 | 0 ~ 49 | `GetDrawings()` |
| 层栈与内层 | 27 层、2 铜层 | 4 块板含 `In*.Cu` | `BoardLayerSet()` / `GetStackupOrDefault()` |
| 设计规则全套 | 见第十节 | — | `GetDesignSettings()` |
| 网络类 | 0 个 | — | `GetNetClasses()` |
| 焊盘几何（形状/尺寸/钻孔） | 52 个 pad | — | `PAD` 各 Get 接口 |
| 元件描述/属性/DNP/丝印图形/3D/变体 | — | — | `FOOTPRINT` 各 Get 接口 |
| 标题栏 / 页面设置 / 变体 / 嵌入资源 | — | — | `BOARD` 各 Get 接口 |

补充一点链路性质：`export_layout()` 只写回元件的 position / rotation / layer，
走线与敷铜原封不动保留在文件里，因此走线信息在整条链路中是**被搬运**而非**被参与决策**。

---

## 十二、实测坑位

1. **`PCB_VIA::GetWidth()` 无参调用会在 KiCad 10 触发 C++ 层 assert 并中止进程**
   （实测复现，报错来自 `pcb_track.cpp(387)`）。必须改用 `GetWidth(layer)` 或
   `GetFrontWidth()`。`utils/visualize_expert_layouts.py` 已为此加了规避逻辑。
2. **`pcbnew.ToMM()` 收到 `None` 会抛 `TypeError`**。`GetLocalClearance()`、
   `GetLocalSolderMaskMargin()` 等在"未设置覆盖"时返回 `None`，取值前必须判空。
3. **`GetBoundingBox()` 是缓存，`SetOrientationDegrees()` 后不刷新**，必须用
   `GetBoundingHull().BBox()` 才能拿到旋转后的真实几何（`kicad_utils` 已按此实现）。
4. **`BOARD.GetNetClasses()` 可能返回空**（IMU 板即如此），不能直接取首个元素。
5. **`BOARD_DESIGN_SETTINGS` 没有 `GetTrackWidthList()`**，线宽/过孔尺寸集合要走
   `BOARD.GetTrackWidthList()` 与 `BOARD.GetViasDimensionsList()`。
6. **`board.GetNetsByName()` 含空名网络 `''`**（未连接网络），
   `kicad_utils._cache_nets()` 已跳过空名。
7. `Component.pins` 的坐标基准是**源姿态 AABB 左上角**（代码 `kicad_utils.py` L348-349
   与同文件 `docs/kicad_interface.md` §2.1 一致）；`kicad_interface.md` §6.7 写的
   "左下角"是过期描述，以 §2.1 与代码为准。

---

## 十三、复现方式

全部结论可用下面三条命令复现（在用户自己的终端执行，需要 KiCad DLL）：

```powershell
$env:KICAD_PATH="E:\Kicad"
cd E:\PCB_AI\PCB_RL
& E:\miniconda\envs\pcb_rl_311\python.exe -c "import os,sys,glob;os.add_dll_directory(r'E:\Kicad\bin');sys.path.insert(0,r'E:\Kicad\bin\Lib\site-packages');import pcbnew as P;f=[x for x in glob.glob('data/kicad-native-batch/*/*.kicad_pcb') if '09db0370' in x][0];b=P.LoadBoard(f);ds=b.GetDesignSettings();tr=list(b.GetTracks());print('ver',b.GetFileFormatVersionAtLoad(),'gen',b.GetGenerator());print('cuLayers',b.GetCopperLayerCount(),'enabled',len(list(b.GetEnabledLayers().Seq())));print('thickness_mm',P.ToMM(ds.GetBoardThickness()));print('fp',len(list(b.GetFootprints())),'nets',b.GetNetCount(),'netclasses',len(list(b.GetNetClasses().values())));print('tracks',len(tr),'vias',len([x for x in tr if isinstance(x,P.PCB_VIA)]),'zones',b.GetAreaCount(),'drawings',len(list(b.GetDrawings())));print('minClearance_mm',P.ToMM(ds.GetSmallestClearanceValue()),'maxClearance_mm',P.ToMM(ds.GetBiggestClearanceValue()),'eps_mm',P.ToMM(ds.GetDRCEpsilon()))"
```

枚举 pcbnew 全部类与只读接口（不需要 KiCad DLL）：

```powershell
python -c "import re;t=open(r'E:\Kicad\bin\Lib\site-packages\pcbnew.py',encoding='utf-8',errors='replace').read();idx=[(m.start(),m.group(1)) for m in re.finditer(r'^class (\w+)\(',t,re.M)];d={n:t[s:(idx[i+1][0] if i+1<len(idx) else len(t))] for i,(s,n) in enumerate(idx)};print(len(idx),'classes');print([m.group(1) for m in re.finditer(r'^    def (\w+)\(',d['BOARD'],re.M)][:40])"
```

统计 `.kicad_pcb` 文本里的对象规模（纯文本，不需要 KiCad DLL）：

```powershell
python -c "import glob,re,collections;fs=[f for f in glob.glob('data/kicad-native-batch/*/*.kicad_pcb') if '09db0370' in f];t=open(fs[0],encoding='utf-8',errors='replace').read();print(collections.Counter(re.findall(r'^\t\((\w+)',t,re.M)).most_common(12))"
```
