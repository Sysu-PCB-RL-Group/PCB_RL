"""PCB 转换数据模型（训练/建模使用的唯一事实来源）。

标准依据：`docs/讨论结果/导入信息重构.md`
  - §0.1 对象集合：footprint 857 ｜ pad 3780（**不裁剪**，含 619 个 netcode=0）｜ via 3422（含无 net 24）｜ 禁布区 46
  - §1.2 旋转：保留"原始角度"，另给"转换角 `{0,1,2,3}`"（等距点按数值降级）
  - §1.4 层：**只讨论 F.Cu / B.Cu**；pad 独立记录所属层
  - §1.5 板框：`GetBoardPolygonOutlines` 结果 + 外接矩形；45 个开槽 → 禁布区（**必须带归属**）
  - §2   对象定义：footprint / pad / via / 禁布区（+ net / pin / 板框）
  - §4.1 转换数据字段与三条硬要求

写死的约定（勿改）：
  * 单位 **mm**；KiCad 板坐标，**Y 轴向下**（`y1 < y2`）
  * AABB = `(x1, y1, x2, y2)`
  * 层：内部只用 `LAYER_F = 0`（F.Cu）、`LAYER_B = 1`（B.Cu）；**无铜层的对象（开槽 / 马达 hull）视作两层通用**
  * 角度：`rotation_deg_raw` = 文件真实值（含 45°）；`rotation_index ∈ {0,1,2,3}` = 转换角
  * 旋转公式：`板坐标 = 锚点 + R(−θ)·局部`；逆变换 `局部 = R(+θ)·(板坐标 − 锚点)`
  * **板坐标几何（AABB / pad 中心 / pad 绝对角）一律在「转换角」下测量** —— 与成品/导出一致（§1.2）；
    原始 45°/135° 姿态的几何属**原始数据**，不在转换数据里
  * **局部几何**（θ=0、相对锚点：`local_aabb` / `local_hull_aabb` / `rel_offset`）是 AABB 与中心
    **反解的唯一依据**；`Footprint.hull_aabb_at() / pad_aabb_at() / pad_center_at()` 就是反解入口

本模块**只读** pcbnew 对象，不修改板；原始数据（raw）仍留在 `KiCadUtils`。
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Set, Tuple

KICAD_PATH = os.environ.get("KICAD_PATH", "")
KICAD_AVAILABLE = False

if KICAD_PATH:
    try:
        os.add_dll_directory(os.path.join(KICAD_PATH, "bin"))
        sys.path.insert(0, os.path.join(KICAD_PATH, "bin", "Lib", "site-packages"))
        sys.path.insert(0, os.path.join(KICAD_PATH, "bin"))
        import pcbnew  # noqa: F401

        KICAD_AVAILABLE = True
    except (OSError, ImportError) as e:  # pragma: no cover
        print(f"[WARN] pcb_model: KiCad 加载失败 ({KICAD_PATH}): {e}")

# 抑制 KiCad 的"字体缺失"模态弹窗（例：Font 'Rockwell' not found; substituting 'DejaVu Sans Book'）。
# 必须在任何 LoadBoard / 字体度量之前设置，否则会弹出对话框卡住进程（无人值守训练/脚本会挂死）。
# 详见《导入信息重构》§5.8（D9 字体弹窗）。
if KICAD_AVAILABLE:
    try:
        import wx

        wx.Log.SetActiveTarget(wx.LogStderr())   # 日志改走 stderr，不弹窗
        wx.Log.SetLogLevel(wx.LOG_Error)         # 只保留 error 级；字体替换属 info/warning，被吞掉
    except Exception as e:  # pragma: no cover
        print(f"[WARN] pcb_model: wx 日志抑制失败（可能出现字体弹窗）: {e}")

# ===================== 常量与基础工具 =====================

AABB = Tuple[float, float, float, float]

LAYER_F = 0
LAYER_B = 1
BOTH_LAYERS: Tuple[int, ...] = (LAYER_F, LAYER_B)

# 开槽与其"所属 footprint 内嵌 Edge.Cuts 图形"的 bbox 匹配容差（2µm）
SLOT_OWNER_TOL_NM = 2000

# 电源网判定（与 kicad_utils.Net.is_power_net 同规则；重构完成后以此为准）
#
# 2026-09-17 修补（Owner 拍案；依据 probe139/probe140）：
#   旧判据只有"精确白名单 + 前缀"，**漏掉 10 条真实供电轨**（3V3 / 3.3V / P12V / P60V /
#   1V1 / 24V / USB_VBUS / MOTOR-VCC / A_3V3 / PGND，共 264 pad = 非电源 pad 的 12.4%）。
#   其中 `3V3` 一条就挂 86 个元件、168 pad —— 把它算进 HPWL 会让目标被这条总线主导。
#   布线形态实测（probe140）：**信号网**平均走线 17.7mm / 敷铜 5.2mm²；
#   **已判电源** 117.2mm / 7556.9mm²；**被漏判的轨** 144.5mm / 337.3mm²
#   ⇒ 这些轨与电源同族、与信号差 1~3 个数量级（形态证据，独立于名字）。
#   修补后 HPWL 变化：逐板 0% ~ −42.6%，全集 −7.5%。
_POWER_NET_EXACT = frozenset({
    "VCC", "VDD", "VSS", "GND", "PWR", "VEE", "VPP", "VREF",
    "AGND", "DGND", "AVCC", "AVDD", "DVCC", "DVDD",
})
_POWER_NET_PREFIXES = (
    "VCC", "VDD", "VSS", "VEE", "+5V", "+3.3V", "+3V3", "+12V", "+1.8V",
    "VIN", "VBUS", "VBAT", "VDDC", "VDDA", "VCCINT", "VCCIO", "VCCPLL",
)
# 名字里含这些"轨"特征也算（覆盖 MOTOR-VCC / PGND / GNDA / USB_VBUS 等）
_POWER_NET_SUBSTR = ("VBUS", "VBAT", "VCC", "VDD", "VSS", "VEE", "GND")
# 以"电压名"结尾（覆盖 3V3 / 3.3V / 1V1 / 24V / P60V / A_3V3 等）
_POWER_VOLTAGE_TAIL = re.compile(r"[+-]?\d+(?:[.,]\d+)?V\d*$")
# 使能 / 参考 / 检测类**信号**：即使名字里带电压或轨字样，也不判为电源
# （覆盖 3V3_EN / ADC_VREF / DCDC_PWR_EN / JDY_PWRC；实测数据集内没有"既被旧判据命中、
#   又匹配本提示"的 net，故不改变任何既有判定）
_POWER_SIGNAL_HINT = re.compile(
    r"_EN|_DET|_SENSE|_SEL|_FLAG|_PG|_INT|_CS|_FAULT|_STAT|_ALERT|_OK|_GOOD|_REF|PWRC"
)


def is_power_net(net_name: str) -> bool:
    """电源网 / 地网判定（大小写不敏感）。

    四级：① 精确白名单 ② **排除使能/参考/检测类信号** ③ 前缀匹配
          ④ 以电压名结尾，或名字含 VBUS/VBAT/VCC/VDD/VSS/VEE/GND。
    """
    upper = net_name.upper()
    if upper in _POWER_NET_EXACT:
        return True
    if _POWER_SIGNAL_HINT.search(upper):
        return False
    if any(upper.startswith(p) for p in _POWER_NET_PREFIXES):
        return True
    if _POWER_VOLTAGE_TAIL.search(upper):
        return True
    return any(s in upper for s in _POWER_NET_SUBSTR)


def convert_rotation_index(deg: float) -> int:
    """原始角度 → 转换角索引 `{0,1,2,3}`（§1.2 决策）。

    规则：**等距点（|θ| mod 90 == 45）一律"按数值降级"**（取数值更小的一档）：
        45 → 0、135 → 90、−45 → −90(≡270)、−135 → −180(≡180)
    非等距按"最接近"取整（本数据集唯一例：lyg0927 X1 = −0.62° → 0°）。
    """
    a = ((deg + 180.0) % 360.0) - 180.0          # 归一化到 (−180, 180]
    r = abs(a) % 90.0
    if abs(r - 45.0) < 1e-9:                     # 等距点：数值降级
        return int(math.floor(a / 90.0)) % 4
    return int(math.floor(a / 90.0 + 0.5)) % 4   # 否则取最接近


def world_offset_to_local(dx: float, dy: float, deg: float) -> Tuple[float, float]:
    """板坐标偏移 → 局部偏移（以锚点为原点、未施加 θ）。

    逆变换 `局部 = R(+θ)·(板坐标 − 锚点)`：
        lx =  cosθ·dx − sinθ·dy
        ly =  sinθ·dx + cosθ·dy
    """
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return (dx * c - dy * s, dx * s + dy * c)


def _mm(nm_value: int) -> float:
    """纳米 → mm（KiCad 内部单位）。"""
    return pcbnew.ToMM(nm_value)


def rotate_aabb(local_aabb: AABB, deg: float, anchor: Tuple[float, float]) -> AABB:
    """局部 AABB → 板坐标 AABB：`板坐标 = 锚点 + R(−θ)·局部`（对四个角点变换后取 min/max）。

    **精确成立当且仅当 `deg` 是 90° 的倍数**（＝本模型的转换角范围 {0,90,180,270}）；
    对其它角度得到的是"局部外接矩形旋转后再取外接矩形"，可能略大于真实形状的 AABB
    （原始 45° 姿态仅供排查，不参与建模，见 §1.2）。
    """
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    x1, y1, x2, y2 = local_aabb
    xs, ys = [], []
    for lx, ly in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        xs.append(anchor[0] + lx * c + ly * s)
        ys.append(anchor[1] - lx * s + ly * c)
    return (min(xs), min(ys), max(xs), max(ys))


# ===================== 位姿更新（env 真实落子后必须用它） =====================

def board_to_local_aabb(aabb: AABB, anchor: Tuple[float, float], rotation_index: int) -> AABB:
    """板坐标 AABB → **局部（θ=0、相对锚点）** AABB（`板坐标 = 锚点 + R(−θ)·局部` 的逆）。"""
    th = math.radians(rotation_index * 90.0)
    c, s = math.cos(th), math.sin(th)
    xs, ys = [], []
    for bx, by in ((aabb[0], aabb[1]), (aabb[2], aabb[1]), (aabb[2], aabb[3]), (aabb[0], aabb[3])):
        dx, dy = bx - anchor[0], by - anchor[1]
        xs.append(dx * c - dy * s)
        ys.append(dx * s + dy * c)
    return (min(xs), min(ys), max(xs), max(ys))


def place_at(fp: Footprint, anchor: Tuple[float, float], rotation_index: int) -> Footprint:
    """把 `fp` 摆到 `(anchor, rotation_index)` 后的**克隆**：pad 的 `aabb`/`center` 实时反解。

    ⚠️ **env 里"真实落子"后必须用返回值替换 `placed` 集合里的对象**：否则 `grid_render` 会拿
    导入时的 `pad.aabb`（= **源位姿**快照，那是专家布局的位置）当障碍 ⇒ 过期几何
    （台账 §0.6 o；probe165 实测：板内挪 2mm 即可"放过 4 个违规格 + 多封 247 格"）。
    依据：台账 §4.2 硬要求 5 —— "pad 记录 + footprint 位置 ⇒ 唯一反解每个 pad 板坐标"。
    """
    th = math.radians(rotation_index * 90.0)
    c, s = math.cos(th), math.sin(th)
    pads = []
    for p in fp.pads:
        aabb = rotate_aabb(p.local_aabb, rotation_index * 90.0, anchor) if p.local_aabb else p.aabb
        lx, ly = p.rel_offset
        center = (anchor[0] + lx * c + ly * s, anchor[1] - lx * s + ly * c)
        pads.append(replace(p, aabb=aabb, center=center))
    hull = (rotate_aabb(fp.local_hull_aabb, rotation_index * 90.0, anchor)
            if fp.local_hull_aabb else fp.hull_aabb)
    return replace(fp, anchor=(anchor[0], anchor[1]), rotation_index=int(rotation_index),
                   rotation_deg_raw=float(int(rotation_index) * 90),
                   hull_aabb=hull, pads=tuple(pads))


def keepout_at(kp: KeepoutRegion, owner: Footprint, anchor: Tuple[float, float],
               rotation_index: int) -> KeepoutRegion:
    """把归属 `owner` 的禁布区项（**开槽**）挪到 owner 的新位姿（开槽随件旋转）。"""
    local = board_to_local_aabb(kp.aabb, owner.anchor, owner.rotation_index)
    return replace(kp, aabb=rotate_aabb(local, rotation_index * 90.0, anchor))


def state_keepouts(model: PcbModel, placed: Sequence[Footprint]) -> Tuple[KeepoutRegion, ...]:
    """**当前局面**的禁布区集合 —— 全项目**唯一**的 keepout 入口（口径 (A)，2026-09-18 Owner 定案）。

    规则：
    * owner **不是可动 footprint**（如 `TT马达` 的 hull：其 owner 是零 pad 封装、不在动作空间）
      ⇒ **恒在场**（板上真实存在的实体件）；
    * owner 是**已放件** ⇒ 其槽按该件的**当前位姿**位姿化（`placed` 必须是 `place_at` 的克隆）；
    * owner 是**未放置件** ⇒ 其槽**不在板上**（该件还没放上去）。

    ⚠️ 口径依据（probe169，2026-09-18）：用 `pcbnew` 平移带槽 footprint (+3,+2)，
    `GetBoardPolygonOutlines()` 报告的**板轮廓内孔正好跟着移 (+3,+2)**、旋转也跟着转
    ⇒ 这些开槽是**封装的一部分**（源数据里就是封装内嵌 Edge.Cuts），不是板自带特征。
    详见台账 §0.6 p、§2.4。

    ⚠️ **不要再引入"保留源位姿"的同义函数**（`pose_updated_keepouts` / `state_keepouts_by_pose`
    已于 2026-09-18 删除）：那样的语义是"已放件挪到位姿、**其余保持原样**" ⇒ **未放置件的槽会留在
    专家位**（"幽灵槽"），实测让待放件 `U1` 的 `L` 少 **1485 格**（bliysfyt，probe177）；它只在
    "poses 恰好覆盖全部槽主"时才等价于口径 A，而函数名看不出这个前提。若某处确实需要"保留源位姿"
    的诊断行为，请**显式构造** keepout 列表传 `keepouts=`，不要复活这个函数。
    """
    poses = {f.ref: (f.anchor[0], f.anchor[1], int(f.rotation_index)) for f in placed}
    byref = {f.ref: f for f in model.footprints}
    out: List[KeepoutRegion] = []
    for kp in model.keepouts:
        owner = byref.get(kp.owner_ref) if kp.owner_ref else None
        if owner is None:
            out.append(kp)                                   # 不可动实体 ⇒ 恒在场
            continue
        p = poses.get(kp.owner_ref)
        if p is not None:                                    # 已放件 ⇒ 槽随件位姿
            out.append(keepout_at(kp, owner, (float(p[0]), float(p[1])), int(p[2])))
        # else: 该件未放置 ⇒ 它的槽不在板上
    return tuple(out)


# ===================== 数据结构（转换数据） =====================


@dataclass(frozen=True)
class Pad:
    """焊盘（判定对象本体；§2.2 —— **不裁剪**，含 netcode=0 的 pad）。"""

    ref: str                              # 所属 footprint 参考号
    number: str                           # 焊盘号
    netcode: int                          # 0 = 未赋 net
    net_id: Optional[str]                 # netcode=0 时为 None
    layers: Tuple[int, ...]               # (0,) / (1,) / (0,1)
    aabb: AABB                            # 板坐标 AABB（判定用；不含阻焊/clearance 扩展）
    center: Tuple[float, float]           # 焊盘中心（板坐标）＝ pin 连接点
    rel_offset: Tuple[float, float]       # 相对锚点、未施加 θ 的局部偏移（供反解）
    size: Tuple[float, float]             # (sizeX, sizeY) mm
    shape: int                            # pcbnew.PAD_SHAPE_*
    angle_deg: float                      # 板坐标下的绝对朝向（不是相对 footprint）
    drill: Tuple[float, float]            # (0.0, 0.0) = 无钻孔
    attribute: int                        # PTH / SMD / NPTH / CONN
    prop: int = 0                         # PAD_PROP_*（本数据集全 NONE）
    local_aabb: AABB = ()                 # θ=0 姿态、相对锚点的 AABB（**pad AABB 反解的唯一依据**）

    @property
    def has_net(self) -> bool:
        """是否有 net（判定过滤条件之一：`netcode != 0`）。"""
        return self.netcode != 0

    @property
    def is_through(self) -> bool:
        """是否贯穿孔（PTH/NPTH）——两层都参与判定。"""
        return self.attribute in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH)

    @property
    def annular_ok(self) -> bool:
        """环宽 > 0（False = 「孔比盘大」的伪焊盘，全数据集 28 个）。"""
        if self.drill[0] <= 0.0 and self.drill[1] <= 0.0:
            return True
        return min(self.size[0] - self.drill[0], self.size[1] - self.drill[1]) > 0.0


@dataclass(frozen=True)
class Footprint:
    """封装（§2.1 —— **至少含一个 pad**；零 pad 的封装不进此集合，只产禁布区）。"""

    ref: str
    fpid: str                             # 库名
    anchor: Tuple[float, float]           # 锚点 = 旋转中心（文件 `(at X Y)`）
    rotation_deg_raw: float               # 文件真实角度（含 45°，仅排查用）
    rotation_index: int                   # 转换角 {0,1,2,3}（建模与导出用）
    layer: int                            # 元件所在面 0/1（实测无内层元件）
    flipped: bool
    hull_aabb: AABB                       # **转换角下**的 hull AABB —— 仅展示/占位，不参与判定
    pads: Tuple[Pad, ...]
    local_hull_aabb: AABB = ()            # θ=0 姿态、相对锚点的 hull AABB（**hull AABB 反解的唯一依据**）
    ref_is_synthetic: bool = False        # ref 是否由我们合成（源文件参考号为空时；见下）

    def hull_aabb_at(self, rotation_index: Optional[int] = None) -> AABB:
        """按转换角计算板坐标 hull AABB（θ 为 90° 倍数时精确）；默认当前 `rotation_index`。"""
        idx = self.rotation_index if rotation_index is None else rotation_index
        return rotate_aabb(self.local_hull_aabb, idx * 90.0, self.anchor)

    @property
    def local_hull_area(self) -> float:
        """**件面积的单点定义**：θ=0 局部 hull AABB 的面积（mm²）。

        用 **θ=0 局部**（不是"当前位姿的 AABB"）是刻意的：该量只依赖件本身
        ⇒ 与位姿、与"当前局面"**全都无关**（口径唯一）。
        （注：4 个旋转只有 90° 倍数 ⇒ hull AABB 只是宽高互换 ⇒ 两者的**面积数值本就相等**，
        差别在于"θ=0 局部"是单一事实来源。）
        消费方：`env/order_plan` 的面积权重/tie-break、`env/completion` 的面积占比（§0.6 m）。
        """
        x0, y0, x1, y1 = self.local_hull_aabb
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def pad_aabb_at(self, pad: Pad, rotation_index: Optional[int] = None) -> AABB:
        """按转换角计算某个 pad 的板坐标 AABB（θ 为 90° 倍数时精确）。"""
        idx = self.rotation_index if rotation_index is None else rotation_index
        return rotate_aabb(pad.local_aabb, idx * 90.0, self.anchor)

    def pad_center_at(self, pad: Pad, rotation_index: Optional[int] = None) -> Tuple[float, float]:
        """按转换角计算某个 pad 的中心板坐标（= HPWL 连接点；任意角均精确）。"""
        idx = self.rotation_index if rotation_index is None else rotation_index
        th = math.radians(idx * 90.0)
        c, s = math.cos(th), math.sin(th)
        lx, ly = pad.rel_offset
        return (self.anchor[0] + lx * c + ly * s, self.anchor[1] - lx * s + ly * c)


@dataclass(frozen=True)
class Via:
    """过孔（§2.3 —— 全部纳入，含无 net 的 24 个）。"""

    netcode: int
    net_id: Optional[str]
    layers: Tuple[int, ...]               # {F}/{B}/{F,B}；**空集 = 不在 F/B（埋孔/盲孔）⇒ 不参与 F/B 判定**
    center: Tuple[float, float]
    diameter: float                       # 外径（判定用外径，不用钻孔）
    drill: float
    aabb: AABB                            # 正方形外接框（判定走 AABB↔AABB）

    @property
    def has_net(self) -> bool:
        return self.netcode != 0


@dataclass(frozen=True)
class Net:
    """网络 = "有 net 的 pad" 的集合视图（**修订台账 §2.5 / §12 D3：不再单独存 pin 表**）。

    连接点就是 pad 本身（`Pad.center`），故这里只存 pad 引用（对象共享、不复制），
    并提供 HPWL 需要的取点接口。
    """

    net_id: str
    pads: Tuple[Pad, ...]
    is_power: bool

    def connection_points(self, refs: Optional[Set[str]] = None,
                          poses: Optional[Mapping[str, Tuple[float, float, int]]] = None
                          ) -> Tuple[Tuple[str, str, float, float], ...]:
        """HPWL 用连接点 = pad 中心：`[(ref, number, x, y), ...]`。

        Args:
            refs: 只取这些 footprint 的连接点（例："只统计已放置元件"）；None = 全部。
            poses: `{ref: (x, y, rot)}` —— 给出时**按当前位姿实时反解中心**；
                缺省用 `p.center`，那是**导入时的源位姿快照**（= 专家布局位置），
                episode 中已放件被挪动后必须传 `poses`（台账 §0.6 o）。
        """
        out = []
        for p in self.pads:
            if refs is not None and p.ref not in refs:
                continue
            pose = poses.get(p.ref) if poses else None
            if pose is not None:
                x, y = float(pose[0]), float(pose[1])
                th = math.radians(int(pose[2]) * 90.0)
                c, s = math.cos(th), math.sin(th)
                lx, ly = p.rel_offset
                out.append((p.ref, p.number, x + lx * c + ly * s, y - lx * s + ly * c))
            else:
                out.append((p.ref, p.number, p.center[0], p.center[1]))
        return tuple(out)


@dataclass(frozen=True)
class KeepoutRegion:
    """禁布区（§2.4）—— 本项目新增的障碍层概念，**不是** KiCad rule area。"""

    kind: str                             # 'motor_hull' | 'board_slot'
    aabb: AABB
    owner_ref: Optional[str]              # 归属 footprint：判定时排除它自己的 pad
    layers: Tuple[int, ...] = BOTH_LAYERS  # 无铜层对象 ⇒ 两层通用


@dataclass(frozen=True)
class BoardOutline:
    """板框【转换数据】（§1.5）—— **只记一个 AABB、不做尺度内缩**。

    - `bbox` = `GetBoardPolygonOutlines()` 结果的外接矩形，**原样**记录；
      "内缩"属 env 层（`BOARD_MARGIN`），用 `valid_area(margin)` 叠加，模型自身不内缩。
    - 板框的**原始几何**（Edge.Cuts 图形清单、多边形外轮廓 + 孔顶点、edges_bbox）留在 raw：
      `KiCadUtils.get_raw_board_outline()`（训练侧不读）。
    - 45 个内孔不在此处重复存：它们的 AABB 已作为 `KeepoutRegion(kind='board_slot')`
      进 `PcbModel.keepouts`（同一批对象）。
    - `outline_count` / `inferred` 是**导入质量诊断字段**，不参与任何判定。
    """

    bbox: AABB                            # 外接矩形（出界判定口径；**不内缩**）
    outline_count: int                    # 外轮廓数（本数据集全为 1）
    slots: Tuple[KeepoutRegion, ...]      # 板轮廓内孔；与 PcbModel.keepouts 里的 board_slot 是**同一批对象**
                                          # ⇒ 判定只遍历 keepouts，**勿再遍历 slots**（会重复计数）
    inferred: bool = False                # 是否用了 GetBoardPolygonOutlines(..., True)

    def valid_area(self, margin: float = 0.0) -> AABB:
        """出界判定用的有效区 = `bbox` 各边内缩 `margin`（默认 **0 = 不内缩**）。"""
        x1, y1, x2, y2 = self.bbox
        return (x1 + margin, y1 + margin, x2 - margin, y2 - margin)


@dataclass(frozen=True)
class PcbModel:
    """转换数据总包（训练/建模的唯一事实来源）。"""

    path: str = ""
    footprints: Tuple[Footprint, ...] = ()
    vias: Tuple[Via, ...] = ()
    nets: Mapping[str, Net] = field(default_factory=dict)
    board_outline: Optional[BoardOutline] = None
    keepouts: Tuple[KeepoutRegion, ...] = ()
    zero_pad_footprints: Tuple[str, ...] = ()   # 只产禁布区、不进 footprint 集合
    standalone_pads: int = 0                    # 不属于任何 footprint 的 pad（数据集 0 例）
    pads: Tuple[Pad, ...] = ()                  # 全部焊盘（展平缓存，由 __post_init__ 填充）

    def __post_init__(self):
        """一次性展平 + 冻结：让"唯一事实来源"名副其实，并免去热路径上的重复计算。"""
        object.__setattr__(self, "pads", tuple(p for fp in self.footprints for p in fp.pads))
        object.__setattr__(self, "nets", MappingProxyType(dict(self.nets or {})))

    @property
    def pads_by_key(self) -> Dict[Tuple[str, str], Tuple[Pad, ...]]:
        """`(ref, pad_number) → 该号下的全部 pad`（**不丢件**）。

        §1.1：同封装同号多 pad 真实存在（`m3 铜柱` 一个封装 7 个 `pad "1"`、半孔排针每号 2~3 片）
        ⇒ 不能只保留第一片，否则命中统计/回溯会系统性少算。
        """
        out: Dict[Tuple[str, str], List[Pad]] = {}
        for pad in self.pads:
            out.setdefault((pad.ref, pad.number), []).append(pad)
        return {k: tuple(v) for k, v in out.items()}

    def net_pads(self, net_id: str, refs: Optional[Set[str]] = None) -> Tuple[Pad, ...]:
        """某个 net 上的 pad（`refs` 给定时只取这些 footprint）——HPWL 的唯一取点入口。"""
        net = self.nets.get(net_id)
        if net is None:
            return ()
        if refs is None:
            return net.pads
        return tuple(p for p in net.pads if p.ref in refs)

    def hpwl(self, refs: Optional[Set[str]] = None, exclude_power: bool = True,
             poses: Optional[Mapping[str, Tuple[float, float, int]]] = None) -> float:
        """半周长线长（HPWL）：对每个 net 求 `(max_x−min_x)+(max_y−min_y)` 后求和。

        §7.1 + §2.5：**排除电源/地网**（`exclude_power=True`）、**不分层**、**不看过孔**；
        `refs` 用于"只统计已放置元件的连接点"（episode 内递增语义）。

        ⚠️ `poses={ref: (x, y, rot)}`：episode 中已放件被挪动后**必须传**，否则连接点取的是
        **导入时的源位姿快照**（= 专家布局位置），HPWL 结果属于"专家布局"而非当前局面
        （台账 §0.6 o）。
        """
        total = 0.0
        for net in self.nets.values():
            if exclude_power and net.is_power:
                continue
            pts = net.connection_points(refs, poses)
            if len(pts) < 2:
                continue
            xs = [p[2] for p in pts]
            ys = [p[3] for p in pts]
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
        return total

    def stats(self) -> Dict[str, int]:
        """验收计数（应满足：footprint 857 / pad 3780 / via 3422 / 禁布区 46 / slot 45）。"""
        pads = self.pads
        return {
            "footprint": len(self.footprints),
            "zero_pad_footprint": len(self.zero_pad_footprints),
            "footprint_synthetic_ref": sum(1 for f in self.footprints if f.ref_is_synthetic),
            "pad": len(pads),
            "pad_net0": sum(1 for p in pads if not p.has_net),
            "pad_negative_annular": sum(1 for p in pads if not p.annular_ok),
            "via": len(self.vias),
            "via_net0": sum(1 for v in self.vias if not v.has_net),
            "net": len(self.nets),
            "net_power": sum(1 for n in self.nets.values() if n.is_power),
            "pad_with_net": sum(1 for p in pads if p.has_net),
            "pad_layers_empty": sum(1 for p in pads if not p.layers),      # 违反 §1.4（层 ∈ {F}/{B}/{F,B}）即 >0
            "footprint_ref_duplicate": len(self.footprints) - len({f.ref for f in self.footprints}),
            "keepout": len(self.keepouts),
            "keepout_slot": sum(1 for k in self.keepouts if k.kind == "board_slot"),
            "keepout_hull": sum(1 for k in self.keepouts if k.kind == "motor_hull"),
        }


# ===================== 构建器（pcbnew → 转换数据） =====================


class PcbModelBuilder:
    """把 `pcbnew.BOARD` 转成 `PcbModel`（只读，不修改板）。"""

    def build(self, board, path: str = "", validate: bool = True) -> PcbModel:
        footprints: List[Footprint] = []
        zero_pad: List[str] = []
        keepouts: List[KeepoutRegion] = []
        net_pads: Dict[str, List[Pad]] = {}

        for fp_index, fp in enumerate(board.GetFootprints()):
            raw_ref = fp.GetReference()
            ref = self._footprint_ref(fp, fp_index)
            pads = list(fp.Pads())

            if not pads:
                # 零 pad 封装：不进 footprint 集合；其实体占位 → 禁布区（§2.4 成员一）
                # 几何与 pad/footprint **同一口径**：在转换角下实测
                #（不用 `GetBoundingBox()` —— 它是缓存，旋转后不刷新，见 §1.5/§5.6）
                zero_pad.append(ref)
                z_idx = convert_rotation_index(float(fp.GetOrientationDegrees()))
                hull_conv, _boxes, _centers, _angles = self._boxes_at(fp, [], z_idx * 90.0)
                keepouts.append(KeepoutRegion(
                    kind="motor_hull",
                    aabb=hull_conv,
                    owner_ref=ref,
                ))
                continue

            anchor = (_mm(fp.GetPosition().x), _mm(fp.GetPosition().y))
            deg_raw = float(fp.GetOrientationDegrees())
            rot_idx = convert_rotation_index(deg_raw)

            # 局部几何（θ=0）：AABB / 中心反解的**唯一依据**
            hull0, boxes0, centers0, _angles0 = self._boxes_at(fp, pads, 0.0)
            # 板坐标几何：一律在**转换角**下测量（模型与成品/导出一致，§1.2）
            hull_b, boxes_b, centers_b, angles_b = self._boxes_at(fp, pads, rot_idx * 90.0)

            pad_objs = tuple(
                self._build_pad(
                    ref, pad,
                    self._sub(boxes0[i], anchor),        # local_aabb（θ=0，相对锚点）
                    boxes_b[i],                          # aabb（转换角，板坐标）
                    centers_b[i],                        # center（转换角，板坐标）
                    self._sub_pt(centers0[i], anchor),   # rel_offset（θ=0，相对锚点）
                    angles_b[i],                         # pad 绝对角（转换角）
                )
                for i, pad in enumerate(pads)
            )
            footprints.append(Footprint(
                ref=ref,
                fpid=str(fp.GetFPID().GetLibItemName()),
                anchor=anchor,
                rotation_deg_raw=deg_raw,
                rotation_index=rot_idx,
                layer=self._layer(fp.GetLayer()),
                flipped=bool(fp.IsFlipped()),
                hull_aabb=hull_b,
                pads=pad_objs,
                local_hull_aabb=self._sub(hull0, anchor),
                ref_is_synthetic=(raw_ref == ""),
            ))

            for pad in pad_objs:
                if pad.net_id is None:
                    continue
                net_pads.setdefault(pad.net_id, []).append(pad)

        # 板框 + 开槽（开槽归属 = 封装内嵌 Edge.Cuts 图形，§2.4 成员二）
        outline = self._build_outline(board)
        keepouts.extend(outline.slots)

        nets = {
            name: Net(
                net_id=name,
                pads=tuple(sorted(net_pad_list, key=lambda p: (p.ref, p.number))),
                is_power=is_power_net(name),
            )
            for name, net_pad_list in net_pads.items()
        }

        model = PcbModel(
            path=path,
            footprints=tuple(footprints),
            vias=tuple(self._build_via(board)),
            nets=nets,
            board_outline=outline,
            keepouts=tuple(keepouts),
            zero_pad_footprints=tuple(zero_pad),
            standalone_pads=self._count_standalone_pads(board),
        )
        if validate:
            # 导入期自检（§0.6 q-⑥）：这些字段一旦缺失，后果全部是**静默错**（不报错、但结果错）
            problems = validate_model(model)
            if problems:
                raise ModelValidationError(
                    f"{path or '<board>'}: 导入期自检失败（{len(problems)} 项）\n  - "
                    + "\n  - ".join(problems[:12])
                    + (f"\n  ...（共 {len(problems)} 项）" if len(problems) > 12 else ""))
        return model

    # ---------------- 内部：单个对象 ----------------

    @staticmethod
    def _aabb(box) -> AABB:
        """BOX2I → mm AABB（Y 向下：y1 = Top < y2 = Bottom）。"""
        return (_mm(box.GetLeft()), _mm(box.GetTop()), _mm(box.GetRight()), _mm(box.GetBottom()))

    @staticmethod
    def _footprint_ref(fp, fp_index: int) -> str:
        """footprint 的稳定 id（**保证板内唯一**）。

        EasyEDA 导出的"独立焊盘"是 **1 个 pad、参考号为空** 的匿名封装（全数据集 25 个，
        分布在 7 块板，库名形如 `Pad_e377`，pad 全部带 net）。空 ref 时合成
        `@<库名>#<序号>` —— 序号是必需的：同板若出现同名库的匿名封装，仅靠库名会撞，
        而 ref 一旦重复：① 按 ref 建键会丢件；② 硬重叠 A 的"不同 footprint"判据会把
        两个不同封装误认为同一个（漏判）。
        """
        raw_ref = fp.GetReference()
        if raw_ref:
            return raw_ref
        lib = str(fp.GetFPID().GetLibItemName()) or "fp"
        return f"@{lib}#{fp_index}"

    @staticmethod
    def _layer(layer_id: int) -> int:
        if layer_id == pcbnew.F_Cu:
            return LAYER_F
        if layer_id == pcbnew.B_Cu:
            return LAYER_B
        raise ValueError(
            f"footprint 所在层不是 F.Cu/B.Cu：{pcbnew.LayerName(layer_id)}"
            f"（§1.4 假设元件只在两面；若数据确实如此，需先扩展模型/判定语义）"
        )

    @staticmethod
    def _pad_layers(pad) -> Tuple[int, ...]:
        out = []
        if pad.IsOnLayer(pcbnew.F_Cu):
            out.append(LAYER_F)
        if pad.IsOnLayer(pcbnew.B_Cu):
            out.append(LAYER_B)
        return tuple(out)

    @classmethod
    def _boxes_at(cls, fp, pads, deg: float):
        """在指定角度下测 hull / pad 的板坐标 AABB、pad 中心与 pad 绝对角。

        临时改姿态，`finally` **原样恢复**（含 45° 等非正交原始角）。
        `deg=0` 的结果减去锚点即"局部几何"（`SetOrientationDegrees(0)` 不改镜像 ⇒ 翻面件同样正确）。
        """
        saved = fp.GetOrientationDegrees()
        try:
            fp.SetOrientationDegrees(deg)
            hull = cls._aabb(fp.GetBoundingHull().BBox())
            boxes = [cls._aabb(p.GetBoundingBox()) for p in pads]
            centers = [(_mm(p.GetPosition().x), _mm(p.GetPosition().y)) for p in pads]
            angles = [float(p.GetOrientationDegrees()) for p in pads]
        finally:
            fp.SetOrientationDegrees(saved)
        return hull, boxes, centers, angles

    @staticmethod
    def _sub(aabb: AABB, anchor: Tuple[float, float]) -> AABB:
        return (aabb[0] - anchor[0], aabb[1] - anchor[1], aabb[2] - anchor[0], aabb[3] - anchor[1])

    @staticmethod
    def _sub_pt(pt: Tuple[float, float], anchor: Tuple[float, float]) -> Tuple[float, float]:
        return (pt[0] - anchor[0], pt[1] - anchor[1])

    def _build_pad(self, ref: str, pad, local_aabb: AABB, board_aabb: AABB,
                   board_center: Tuple[float, float], rel_offset: Tuple[float, float],
                   angle_deg: float) -> Pad:
        try:
            prop = int(pad.GetProperty())
        except Exception:
            prop = 0
        return Pad(
            ref=ref,
            number=pad.GetNumber(),
            netcode=int(pad.GetNetCode()),
            net_id=(pad.GetNetname() or None) if pad.GetNetCode() != 0 else None,
            layers=self._pad_layers(pad),
            aabb=board_aabb,
            center=board_center,
            rel_offset=rel_offset,
            size=(_mm(pad.GetSizeX()), _mm(pad.GetSizeY())),
            shape=int(pad.GetShape()),
            angle_deg=angle_deg,
            drill=(_mm(pad.GetDrillSizeX()), _mm(pad.GetDrillSizeY())),
            attribute=int(pad.GetAttribute()),
            prop=prop,
            local_aabb=local_aabb,
        )

    def _build_via(self, board) -> List[Via]:
        vias: List[Via] = []
        for tr in board.GetTracks():
            if tr.Type() != pcbnew.PCB_VIA_T:
                continue
            pos = tr.GetPosition()
            cx, cy = _mm(pos.x), _mm(pos.y)
            dia = _mm(tr.GetFrontWidth())
            layers = tuple(L for L, lid in ((LAYER_F, pcbnew.F_Cu), (LAYER_B, pcbnew.B_Cu)) if tr.IsOnLayer(lid))
            half = dia / 2.0
            netcode = int(tr.GetNetCode())
            vias.append(Via(
                netcode=netcode,
                net_id=(tr.GetNetname() or None) if netcode != 0 else None,
                layers=layers,          # 不捏造层归属：空集（埋孔/盲孔）就是不参与 F/B 判定
                center=(cx, cy),
                diameter=dia,
                drill=_mm(tr.GetDrillValue()),
                aabb=(cx - half, cy - half, cx + half, cy + half),
            ))
        return vias

    def _build_outline(self, board) -> BoardOutline:
        poly = pcbnew.SHAPE_POLY_SET()
        inferred = False
        ok = board.GetBoardPolygonOutlines(poly, False)
        if not ok or poly.OutlineCount() == 0:
            poly = pcbnew.SHAPE_POLY_SET()
            board.GetBoardPolygonOutlines(poly, True)
            inferred = True

        # 归属映射：封装内嵌 Edge.Cuts 图形的 bbox
        owner_boxes: List[Tuple[AABB, str]] = []
        for fp_index, fp in enumerate(board.GetFootprints()):
            ref = self._footprint_ref(fp, fp_index)   # 与 build() 同一套 id 规则（否则"排除自家槽"失效）
            for item in fp.GraphicalItems():
                if item.GetLayer() == pcbnew.Edge_Cuts:
                    owner_boxes.append((self._aabb(item.GetBoundingBox()), ref))

        slots: List[KeepoutRegion] = []
        for i in range(poly.OutlineCount()):
            for h in range(poly.HoleCount(i)):
                hole_box = poly.Hole(i, h).BBox()
                aabb = self._aabb(hole_box)
                owner, best = None, 10 ** 9
                for cand_aabb, ref in owner_boxes:
                    d = max(abs(cand_aabb[k] - aabb[k]) for k in range(4))
                    if d < best:
                        best, owner = d, ref
                slots.append(KeepoutRegion(
                    kind="board_slot",
                    aabb=aabb,
                    owner_ref=owner if best * 1e6 <= SLOT_OWNER_TOL_NM else None,
                ))
        return BoardOutline(
            bbox=self._aabb(poly.BBox()),
            outline_count=poly.OutlineCount(),
            slots=tuple(slots),
            inferred=inferred,
        )

    @staticmethod
    def _count_standalone_pads(board) -> int:
        """独立焊盘（不属于任何 footprint）计数 —— 判据必须用 uuid 比对。

        注意：`isinstance(pad.GetParent(), pcbnew.FOOTPRINT)` 在本 SWIG 下恒为 False，
        不可用（见台账 §5.1）。
        """
        uuids = {fp.m_Uuid.AsString() for fp in board.GetFootprints()}
        n = 0
        for pad in board.GetPads():
            try:
                if pad.GetParent().m_Uuid.AsString() not in uuids:
                    n += 1
            except Exception:
                n += 1
        return n


class ModelValidationError(ValueError):
    """**导入期自检失败**（§0.6 q-⑥）：模型身份或几何字段不完整。

    这些缺失的后果**全部是静默错**（不报错、结果错）⇒ 必须在导入期就拦住：
      * `ref` 重复            ⇒ 按 ref 建键会丢件；硬重叠的"不同 footprint"判据会漏判
      * 缺 `local_aabb`       ⇒ `place_at` **保留源位姿 aabb**（移动后几何过期）/ `legal_masks` 对候选件**跳过** /
                                 `Judge._pads_at` **崩** / `_obstacles` 对已放件**照算** ⇒ 四处行为不一致
      * 缺 `local_hull_aabb`  ⇒ `Judge` 粗筛与掩膜的前置几何失效
      * 空 `pads` 却在 footprints 里 ⇒ 语义矛盾（零 pad 封装应只作禁布区）
      * `rel_offset` 非法     ⇒ `place_at` 的 pad 中心反解带着错值走
    """


def validate_model(model: PcbModel) -> List[str]:
    """导入期自检：**身份唯一 + 几何字段齐全**（全数据集 857 件 / 3780 pad 实测 0 项问题）。

    Returns:
        问题描述列表（空 = 通过）。每条的静默后果见 `ModelValidationError`。
    """
    from collections import Counter

    problems: List[str] = []
    cnt = Counter(f.ref for f in model.footprints)
    dup = sorted(r for r, n in cnt.items() if n > 1)
    if dup:
        problems.append(f"footprint ref 重复 {len(dup)} 个：{dup[:5]}（按 ref 建键会丢件、硬重叠会漏判）")
    for f in model.footprints:
        if not f.pads:
            problems.append(f"{f.ref}: 无 pad 却进了 footprints（零 pad 封装应只作禁布区）")
        if not f.local_hull_aabb:
            problems.append(f"{f.ref}: 缺 local_hull_aabb（Judge 粗筛 / 掩膜前置几何失效）")
        for p in f.pads:
            if not p.local_aabb:
                problems.append(f"{f.ref}/{p.number}: 缺 local_aabb（place_at 保留源位姿 aabb、L 跳过、Judge 崩）")
            if len(p.rel_offset) != 2:
                problems.append(f"{f.ref}/{p.number}: rel_offset 非法 {p.rel_offset!r}")
    return problems


def build_model(board, path: str = "", validate: bool = True) -> PcbModel:
    """便捷入口：`board` 为 `pcbnew.BOARD`。"""
    return PcbModelBuilder().build(board, path, validate=validate)
