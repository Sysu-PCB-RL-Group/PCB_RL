"""KiCad 工具包模块。

导出 Component、Net 数据结构和 KiCadUtils 工具类。

⚠️ **上一代**（基于 SPEC v2.1）：本包的 `Component`/`Net` 是旧数据结构，但 **`KiCadUtils.export_layout()`
导出路径仍在使用**；**新的转换数据模型见同目录 `pcb_model.py`**（`Pad/Footprint/Via/Net/PcbModel`），
建模口径一律以 `docs/讨论结果/导入信息重构.md` 为准。
"""

from .kicad_utils import Component, Net, KiCadUtils

__all__ = ['Component', 'Net', 'KiCadUtils']