# -*- coding: utf-8 -*-
"""参数一览（**只读**）：`PARAMS.md` 的数据源 + 生成/校验实现（Owner 2026-09-24 拍）。

```text
需求（Owner）："能有个口子看到现在配置的参数"——放在**仓库根 `PARAMS.md`**，不进台账。
设计（关键）：**真源仍是各模块里的常数**，本模块只做两件事 ——
  ① `SPEC` = 登记表："有哪些参数 / 什么含义 / 依据在哪（台账节号）/ 代码位置在哪"；
  ② `collect()` 运行时**到真源处读值**（模块属性 ｜ 函数签名默认值 ｜ 脚本 `DEFAULTS`）。
⇒ 因此**不存在第二份真源**：清单不可能与代码不同步（`--check` 与测试兜底）。
⚠️ 本模块**不修改任何口径**：它只"读"，不写回；也不参与训练/推理的运行路径。
```

用法

```text
& python -m env.params              # 打印到终端
& python -m env.params --write      # 覆盖写 `PARAMS.md`（改过参数后跑一次）
& python -m env.params --check      # 与 `PARAMS.md` 比对（不一致 ⇒ 退出码 1）
入口（推荐）：`scripts/dump_params.py`（本模块的薄封装）
守门：`tests/kicad/test_params_single_source.py`
```

⚠️ 刻意**不写生成时间/机器名**进清单 —— 否则每跑一次都产生无意义的 git diff。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 清单文件（仓库根；Owner 2026-09-24 选定）
MANIFEST = os.path.join(_ROOT, "PARAMS.md")

__all__ = ["MANIFEST", "Param", "SPEC", "collect", "read_value", "render_markdown", "main"]


# ===================== 登记表 =====================
#
# `where` 语法（三种，`read_value` 解析）：
#   ① `模块:属性`                 —— 模块级常数（多数情况）
#   ② `模块:函数:参数名`           —— 函数/方法的签名默认值（`__init__` 的旋钮）
#   ③ `模块:常量:键`               —— 字典常量里的键（脚本的 `DEFAULTS`）
# `source` = 依据（台账节号 / Owner 拍案日期）；`note` = 附加说明（含"重复定义"警告）。


@dataclass(frozen=True)
class Param:
    """一条参数的**登记信息**（值不在这里，运行时去 `where` 处读）。"""

    section: str      # 分组标题（清单里的小节）
    name: str         # 展示名
    where: str        # 取值地址（见上方三种语法）
    meaning: str      # 一句话含义
    source: str       # 依据（台账节号 / 拍案日期）
    note: str = ""    # 附加说明


SEC_MESH = "网格与几何（改了要重跑 L/G、重标金表）"
SEC_G = "G：增益矩阵（改了要重标 G 金表）"
SEC_REWARD = "reward：回报（改了要重训）"
SEC_OBS = "obs：观测编码（改了要重训）"
SEC_ORDER = "放置顺序（改了会改整条轨迹）"
SEC_JUDGE = "违规判定（与 L 必须同口径）"
SEC_NET = "网络结构"
SEC_AGENT = "agent 超参（训练旋钮）"
SEC_TRAIN = "训练默认值（CLI 可覆盖）"
SEC_INFER = "推理默认值（CLI 可覆盖）"
SEC_ENV = "env 构造默认值"

SPEC: Sequence[Param] = (
    # ---- A. 网格与几何 ----
    Param(SEC_MESH, "GRID_M", "env.grid_render:GRID_M",
          "画布边长（M×M 网格；cell 尺寸由板框与 M 决定）",
          "§5（Owner 2026-09-16 拍案）"),
    Param(SEC_MESH, "SOFT_GAP_MM", "env.grid_render:SOFT_GAP_MM",
          "软间距阈值（= 硬∪软的合并阈值）",
          "§3.4（判据本体）",
          note="⚠️ 与 `env/violations.py` 那份是**两份独立定义**（值相同、无守门）"),
    Param(SEC_MESH, "_EPS", "env.grid_render:_EPS",
          "边界数值容差（禁区外扩 / 板框内收；只为消浮点末位差，方向保守）",
          "§5.7"),
    Param(SEC_MESH, "MODE_FILL", "env.grid_render:MODE_FILL",
          "板框 → 网格的映射口径（`fill` = 按轴铺满 M×M）",
          "§5（Owner 拍案口径）"),
    # ---- B. G ----
    Param(SEC_G, "G_W_HPWL", "env.grid_render:G_W_HPWL",
          "G 的 HPWL 分量权重（λ 的前一项）",
          "§5.4（Owner 2026-09-17 拍案）"),
    Param(SEC_G, "G_W_DISP", "env.grid_render:G_W_DISP",
          "G 的离散度分量权重（λ 的后一项）",
          "§5.4（Owner 2026-09-17 拍案）"),
    Param(SEC_G, "G_P_LO", "env.grid_render:G_P_LO",
          "稳健 min-max 的下分位（4 张表共用一把标尺）",
          "§5.4"),
    Param(SEC_G, "G_P_HI", "env.grid_render:G_P_HI",
          "稳健 min-max 的上分位",
          "§5.4"),
    # ---- C. reward ----
    Param(SEC_REWARD, "ALPHA", "env.reward:ALPHA",
          "σ 项系数（`+α·Δσ/D`）；语义 = 每步判别力 HPWL:σ 按 λ 配平",
          "§7.3（probe202 + Owner 2026-09-19 拍）"),
    Param(SEC_REWARD, "BETA_MARGIN", "env.reward:BETA_MARGIN",
          "β = margin × β_crit(该板)（余量覆盖口径近似）",
          "§7.3（Owner 2026-09-22 拍）"),
    Param(SEC_REWARD, "GAMMA", "env.reward:GAMMA",
          "折扣（episode 有限且会终止 ⇒ γ = 1）",
          "§7.3（Owner 2026-09-22 拍）"),
    # ---- D. obs ----
    Param(SEC_OBS, "N_ROT", "env.obs_spec:N_ROT",
          "旋转数（动作第一维；转换角 {0,1,2,3}）",
          "§5.5 / §6.1",
          note="⚠️ `env/policy_net.py` 另有一份同值 `N_ROT`"),
    Param(SEC_OBS, "SPATIAL_CHANNELS", "env.obs_spec:SPATIAL_CHANNELS",
          "spatial 通道数 = N_ROT × {L, G}",
          "§5.5",
          note="⚠️ `env/policy_net.py` 的 `CHANNELS` 是同值另一份"),
    Param(SEC_OBS, "PAD_MAX", "env.obs_spec:PAD_MAX",
          "piece 归一化上界：单件最大 pad 数（超界不裁剪）",
          "§5.5（`scripts/dataset_report.py` 实测）"),
    Param(SEC_OBS, "NET_MAX", "env.obs_spec:NET_MAX",
          "piece 归一化上界：单件最大**非电源** net 数",
          "§5.5（同上实测）"),
    Param(SEC_OBS, "GAIN_FILL", "env.obs_spec:GAIN_FILL",
          "送网络的 G 里**非法格**的填值（0 = 无效/非法）",
          "§5.5（Owner 2026-09-22 拍）"),
    Param(SEC_OBS, "PIECE_KEYS", "env.obs_spec:PIECE_KEYS",
          "piece (3,) 的语义顺序（写死）",
          "§5.5"),
    # ---- E. 放置顺序 ----
    Param(SEC_ORDER, "DEFAULT_W_AREA", "env.order_plan:DEFAULT_W_AREA",
          "固定顺序的面积权重（连通性权重 = 1 − 该值）",
          "§6.2（Owner 2026-09-17 拍案）",
          note="⚠️ 与 G 的 0.7 **无关系**（同名不同物）"),
    # ---- F. 违规判定 ----
    Param(SEC_JUDGE, "SOFT_GAP_MM", "env.violations:SOFT_GAP_MM",
          "软间距阈值（`Judge` 用的那一份；严格 0.2，不做豁免）",
          "§3.4",
          note="⚠️ 与 `env/grid_render.py` 那份**重复定义**（两份都 = 0.2）"),
    Param(SEC_JUDGE, "_PAD_CACHE_MAX", "env.violations:_PAD_CACHE_MAX",
          "pad AABB 缓存上限（位姿连续 ⇒ 必须封顶）",
          "源码注释（防止训练中无界增长）"),
    # ---- G. 网络 ----
    Param(SEC_NET, "CHANNELS", "env.policy_net:CHANNELS",
          "网络输入通道数（= N_ROT × {L, G}）",
          "§9.2"),
    Param(SEC_NET, "N_ROT", "env.policy_net:N_ROT",
          "网络输出通道数 / 动作第一维",
          "§9.2",
          note="⚠️ 与 `env/obs_spec.py` 那份**重复定义**"),
    # ---- H. agent 超参 ----
    Param(SEC_AGENT, "DEFAULT_CAPACITY", "env.dqn_agent:DEFAULT_CAPACITY",
          "打包回放容量（条数；≈ 3.80 GB @200×200）",
          "§9.2（Owner 2026-09-23 暂定）"),
    Param(SEC_AGENT, "DEFAULT_SHAPE", "env.dqn_agent:DEFAULT_SHAPE",
          "回放打包的 obs 形状（spatial）",
          "§9.2"),
    Param(SEC_AGENT, "lr", "env.dqn_agent:DoubleDQNAgent.__init__:lr",
          "Adam 学习率",
          "§9.2（Owner 2026-09-23 拍）"),
    Param(SEC_AGENT, "batch_size", "env.dqn_agent:DoubleDQNAgent.__init__:batch_size",
          "采样批量（**训练脚本用 `--batch 32` 覆盖**）",
          "§9.2（我方自定）"),
    Param(SEC_AGENT, "target_every_episodes", "env.dqn_agent:DoubleDQNAgent.__init__:target_every_episodes",
          "目标网硬同步周期（按 episode 记数）",
          "§9.2（Owner 2026-09-23：先 100、后因只跑 100 局改 20）"),
    Param(SEC_AGENT, "eps_top", "env.dqn_agent:DoubleDQNAgent.__init__:eps_top",
          "ε 曲线上界（ep = 0 处的探索率）",
          "§9.2（Owner 2026-09-23 拍）"),
    Param(SEC_AGENT, "eps_bottom", "env.dqn_agent:DoubleDQNAgent.__init__:eps_bottom",
          "ε 曲线下界（ep → ∞ 时的渐近探索率）",
          "§9.2（Owner 2026-09-23 拍）"),
    Param(SEC_AGENT, "eps_tau_episodes", "env.dqn_agent:DoubleDQNAgent.__init__:eps_tau_episodes",
          "ε 时间常数 τ（`ε = bottom + (top−bottom)·exp(−ep/τ)`）",
          "§9.2（**我方自定** = 2,000）",
          note="⚠️ 1,000 局跑完时 ε ≈ 0.585（仍近六成随机探索）"),
    Param(SEC_AGENT, "gamma", "env.dqn_agent:DoubleDQNAgent.__init__:gamma",
          "折扣（直接引用 `reward.GAMMA`，不另设常数）",
          "§7.3 / §9.2"),
    # ---- I. 训练默认 ----
    Param(SEC_TRAIN, "episodes", "scripts.train_dqn:DEFAULTS:episodes",
          "训练局数", "§9.2（Owner 2026-09-23 拍定）"),
    Param(SEC_TRAIN, "update_every", "scripts.train_dqn:DEFAULTS:update_every",
          "每多少个 env 步做一次梯度更新", "§9.2（同上）"),
    Param(SEC_TRAIN, "batch", "scripts.train_dqn:DEFAULTS:batch",
          "采样批量（覆盖 agent 的 64）", "§9.2（同上）"),
    Param(SEC_TRAIN, "eval_every_episodes", "scripts.train_dqn:DEFAULTS:eval_every_episodes",
          "每多少局跑一次 16 板评估", "§9.2（同上）"),
    Param(SEC_TRAIN, "log_every", "scripts.train_dqn:DEFAULTS:log_every",
          "每多少局打一行进度", "§9.2（同上）"),
    Param(SEC_TRAIN, "ckpt_every", "scripts.train_dqn:DEFAULTS:ckpt_every",
          "每多少局落一次断点", "§9.2（同上）"),
    Param(SEC_TRAIN, "capacity", "scripts.train_dqn:DEFAULTS:capacity",
          "回放容量（传给 agent）", "§9.2（同上）"),
    Param(SEC_TRAIN, "target_every_episodes", "scripts.train_dqn:DEFAULTS:target_every_episodes",
          "目标网硬同步周期（传给 agent）", "§9.2（同上）"),
    Param(SEC_TRAIN, "seed", "scripts.train_dqn:DEFAULTS:seed",
          "固定种子（网络初始化也由它决定 ⇒ 同命令可复现）", "§9.2（Owner 2026-09-23）"),
    Param(SEC_TRAIN, "out", "scripts.train_dqn:DEFAULTS:out",
          "产物目录（**写入前清零**）", "§9.2"),
    Param(SEC_TRAIN, "device", "scripts.train_dqn:DEFAULTS:device",
          "计算设备", "§9.2"),
    # ---- J. 推理默认 ----
    Param(SEC_INFER, "ckpt", "scripts.infer_dqn:DEFAULTS:ckpt",
          "权重文件（默认训练产物 `output/best.pt`）", "§9.2 / §10.1"),
    Param(SEC_INFER, "out", "scripts.infer_dqn:DEFAULTS:out",
          "产物根目录（只清 `infer_*`）", "§9.2"),
    Param(SEC_INFER, "boards", "scripts.infer_dqn:DEFAULTS:boards",
          "逗号分隔的板名（空 = 全部板）", "§9.2"),
    Param(SEC_INFER, "dpi", "scripts.infer_dqn:DEFAULTS:dpi",
          "对比图分辨率", "§10.1"),
    Param(SEC_INFER, "viz", "scripts.infer_dqn:DEFAULTS:viz",
          "是否出 2×2 对比图（`--no-viz` 关）", "§10.1"),
    Param(SEC_INFER, "net_boxes", "scripts.infer_dqn:DEFAULTS:net_boxes",
          "是否画 HPWL 包围框（`--net-boxes` 开）", "§10.1（Owner 2026-09-23）",
          note="⚠️ 默认**关**"),
    Param(SEC_INFER, "charts", "scripts.infer_dqn:DEFAULTS:charts",
          "是否出 3 张指标对比图（`--no-charts` 关）", "§10.1.1"),
    Param(SEC_INFER, "device", "scripts.infer_dqn:DEFAULTS:device",
          "计算设备", "§9.2"),
    Param(SEC_INFER, "cache_root", "scripts.infer_dqn:DEFAULTS:cache_root",
          "模型缓存目录覆盖（None = `data/model_cache`）", "§4"),
    # ---- K. env 构造默认 ----
    Param(SEC_ENV, "use_lookahead", "env.pcb_env:PcbEnv.__init__:use_lookahead",
          "True = 训练档（`mask = L & lookahead`）｜False = 推理档（`mask = L`）",
          "§6.3（Owner 2026-09-18）"),
    Param(SEC_ENV, "verify", "env.pcb_env:PcbEnv.__init__:verify",
          "落子前是否用 `Judge` 精确复核（不一致 ⇒ 抛 `EnvContractError`）",
          "§5.7"),
    Param(SEC_ENV, "with_contrib", "env.pcb_env:PcbEnv.__init__:with_contrib",
          "完成度 ③（贡献占比）是否建表（只影响 `info`）",
          "§0.6 m"),
    Param(SEC_ENV, "resample_dead_start", "env.pcb_env:PcbEnv.__init__:resample_dead_start",
          "开局 `L` 就空时是否重采一块板",
          "§9.1"),
)


# ===================== 取值（到真源处读） =====================

_MODULE_CACHE: Dict[str, Any] = {}


def _module(path: str) -> Any:
    """取模块对象：`scripts.*` 用文件路径加载（`scripts/` 不是包），其余走 `importlib`。"""
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    if path.startswith("scripts."):
        name = path.split(".", 1)[1]
        fp = os.path.join(_ROOT, "scripts", name + ".py")
        spec = importlib.util.spec_from_file_location("_params_script_" + name, fp)
        if spec is None or spec.loader is None:      # pragma: no cover
            raise ImportError(f"无法加载脚本模块：{fp}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path)
    _MODULE_CACHE[path] = mod
    return mod


def read_value(where: str) -> Any:
    """按 `where` 语法（`模块:属性` ｜ `模块:函数:参数` ｜ `模块:常量:键`）读**当前值**。"""
    parts = where.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"where 语法应为 `模块:属性` 或 `模块:函数:参数`，收到 {where!r}")
    obj: Any = _module(parts[0])
    if len(parts) == 2:
        return getattr(obj, parts[1])
    for attr in parts[1].split("."):
        obj = getattr(obj, attr)
    key = parts[2]
    if callable(obj):
        params = inspect.signature(obj).parameters
        if key not in params:
            raise KeyError(f"{where}：签名里没有参数 {key!r}")
        return params[key].default
    return obj[key]


def collect() -> List[Dict[str, Any]]:
    """按 `SPEC` 顺序逐条取值 ⇒ `[{section, name, value, meaning, source, where, note}, ...]`。"""
    out: List[Dict[str, Any]] = []
    for p in SPEC:
        out.append({"section": p.section, "name": p.name, "value": read_value(p.where),
                    "meaning": p.meaning, "source": p.source, "where": p.where, "note": p.note})
    return out


# ===================== 渲染（PARAMS.md） =====================

def _fmt(v: Any) -> str:
    """值的展示形式（表格里用反引号包住 ⇒ 只做 `|` 转义；路径统一正斜杠；空串显示 `""`）。"""
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    if isinstance(v, str):
        return (v.replace("\\", "/").replace("|", "\\|")) if v else '""'
    return str(v).replace("|", "\\|")


def render_markdown() -> str:
    """生成 `PARAMS.md` 的完整内容（**不含时间戳** ⇒ 稳定可 diff）。"""
    rows = collect()
    lines: List[str] = [
        "# 当前参数一览",
        "",
        "> **本文件是生成物，请勿手改** —— 由 `python scripts/dump_params.py --write` 生成。",
        "> 真源是代码：每条参数的「代码位置」列就是它被定义的地方；清单与代码不一致时，",
        "> `pytest tests/kicad/test_params_single_source.py` 会直接失败。",
        "",
        "「改了要重训 / 重标金表」的口径常数在前几节；运行参数（每次跑可覆盖）在最后几节。",
        "",
    ]
    order: List[str] = []
    for r in rows:
        if r["section"] not in order:
            order.append(r["section"])
    for sec in order:
        lines += [f"## {sec}", "",
                  "| 参数 | 当前值 | 含义 | 依据 | 代码位置 |", "|---|---|---|---|---|"]
        for r in rows:
            if r["section"] != sec:
                continue
            note = f" {r['note']}" if r["note"] else ""
            lines.append(f"| `{r['name']}` | `{_fmt(r['value'])}` | {r['meaning']}{note} "
                         f"| {r['source']} | `{r['where']}` |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ===================== CLI =====================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="参数一览（PARAMS.md）的打印 / 生成 / 校验")
    ap.add_argument("--write", action="store_true", help="覆盖写 PARAMS.md")
    ap.add_argument("--check", action="store_true", help="与 PARAMS.md 比对（不一致 ⇒ 退出码 1）")
    args = ap.parse_args(list(argv) if argv is not None else None)

    text = render_markdown()
    if args.write and args.check:
        print("[FAIL] --write 与 --check 互斥")
        return 2
    if args.write:
        with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"[ok] 已写 {MANIFEST}（{len(SPEC)} 条参数 / {len(text.splitlines())} 行）")
        return 0
    if args.check:
        cur = ""
        if os.path.isfile(MANIFEST):
            with open(MANIFEST, encoding="utf-8") as f:
                cur = f.read()
        if cur == text:
            print(f"[ok] PARAMS.md 与代码一致（{len(SPEC)} 条参数）")
            return 0
        print("[FAIL] PARAMS.md 与代码不一致 ⇒ 跑 `python scripts/dump_params.py --write`")
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
