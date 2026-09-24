# -*- coding: utf-8 -*-
"""PCB RL 环境模块。

⚠️ 本包同时存在**两代**内容（2026-09-18 审计说明）：

```text
上一代（仍在，留作对照）：PCBUtils / PCBLayoutEnv
    旧动作空间 = 7 元 Tuple（component_id, x1,y1,x2,y2, rotation, layer）+ 100×100 网格 + 违规罚分；
    新台账 `docs/讨论结果/导入信息重构.md` 已定案：动作 =（旋转 k, cell i, j）、不选件、违规被 mask 禁止。

新一代（组件已落地、**尚未组装成 env**）：
    grid_render  L 合法掩码 / G 增益矩阵        violations  判定器 Judge
    order_plan   固定放置顺序（plan_order）      lookahead   单步预测掩膜
    board_state  局面状态 BoardState（唯一状态载体）
    completion   放置完成度（件数/面积/贡献 三量）  obs_spec  obs 口径
    pcb_env      **环境层 `PcbEnv`（reset / step / action_masks）** —— 台账 §9.1 的落地
```

刻意**不在 `__init__` 里 re-export 新一代模块** —— 它们各自用完整路径导入
（`from env import grid_render as gr`），避免"上一代符号"与"新一代符号"在同一命名空间混淆。

**上一代符号改惰性导出（PEP 562，2026-09-22）**：`PCBUtils` / `PCBLayoutEnv` 延迟到**真正访问**时才
import 对应模块 —— 否则 `from env import grid_render as gr`（训练代码的标准写法）会连带
`gymnasium` + 上一代 `pcb_utils` + `utils.kicad_utils.KiCadUtils` 整条旧依赖链进训练进程。
`__all__` 与旧用法**完全不变**（`from env import PCBUtils` 照旧），拼错的名字仍抛 `AttributeError`。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["PCBUtils", "PCBLayoutEnv"]

# 上一代符号 → (模块, 类名)；只在被访问时才 import（见 __getattr__）
_LEGACY_EXPORTS = {
    "PCBUtils": ("env.pcb_utils", "PCBUtils"),
    "PCBLayoutEnv": ("env.pcb_layout_env", "PCBLayoutEnv"),
}

if TYPE_CHECKING:                       # 仅供 IDE / 静态检查（运行期不执行）
    from env.pcb_layout_env import PCBLayoutEnv
    from env.pcb_utils import PCBUtils


def __getattr__(name: str) -> Any:
    """PEP 562 惰性导出：访问上一代符号时才 import（拼错的名字仍抛 `AttributeError`）。"""
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value             # 缓存：后续访问不再走 __getattr__
    return value
