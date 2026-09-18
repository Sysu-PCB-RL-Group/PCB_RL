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
    env 层本身待落地（台账 §9 留空；动作/观测/终止口径已定案见 §5/§6）
```

刻意**不在 `__init__` 里 re-export 新一代模块** —— 它们各自用完整路径导入
（`from env import grid_render as gr`），避免"上一代符号"与"新一代符号"在同一命名空间混淆。
"""
from .pcb_utils import PCBUtils
from .pcb_layout_env import PCBLayoutEnv

__all__ = ["PCBUtils", "PCBLayoutEnv"]        # 上一代（新模块请用完整路径导入）
