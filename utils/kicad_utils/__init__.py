# -*- coding: utf-8 -*-
"""KiCad 工具包模块。

导出上一代 `Component` / `Net` / `KiCadUtils`（**惰性** —— 见下）。

⚠️ **上一代**（基于 SPEC v2.1）：本包的 `Component`/`Net` 是旧数据结构，但 **`KiCadUtils.export_layout()`
导出路径仍在使用**；**新的转换数据模型见同目录 `pcb_model.py`**（`Pad/Footprint/Via/Net/PcbModel`），
建模口径一律以 `docs/讨论结果/导入信息重构.md` 为准。

为什么"惰性"（P1：训练机不依赖 KiCad）
------------------------------------
Python 在 import **任何子模块**（`utils.kicad_utils.pcb_model`）之前，必先执行本 `__init__`。
若这里静态 `from .kicad_utils import ...`，**训练机（无 KiCad）**会在 import 期就被拖进上一代模块的
模块级 bootstrap ⇒ 白打两行 `[INFO]` 噪音（`kicad_utils.py:24-27`：`KICAD_PATH 未设置…`），
并把 `wx`/`pcbnew` 的加载分支一起拉进训练进程。

⇒ 用 PEP 562 的模块级 `__getattr__` 把这次导入**推迟到真正访问旧符号时**：
新一代链路（`model_cache` / `pcb_model` / `env.*`）零噪音、零耦合；
`from utils.kicad_utils import KiCadUtils` 的用法**完全不变**（旧链路无感，只是晚一点 import）。
守门测试：`tests/kicad/test_no_kicad_runtime.py`。
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                    # 仅供静态检查/IDE 补全；运行期不执行
    from .kicad_utils import Component, KiCadUtils, Net

__all__ = ["Component", "Net", "KiCadUtils"]

_LEGACY_NAMES = frozenset(__all__)
_LEGACY_MODULE = f"{__name__}.kicad_utils"           # 上一代模块（模块级 bootstrap 就在它里面）


def __getattr__(name: str):
    """惰性导出上一代符号（PEP 562）；访问一次后缓存进 `globals()`。"""
    if name in _LEGACY_NAMES:
        mod = importlib.import_module(_LEGACY_MODULE)
        value = getattr(mod, name)
        globals()[name] = value                      # 缓存 ⇒ 后续访问不再走本函数
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_NAMES))
