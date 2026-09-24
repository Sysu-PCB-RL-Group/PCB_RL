# -*- coding: utf-8 -*-
"""PCB RL 环境模块（**新一代替**）。

各子模块一律用**完整路径**导入（`from env import grid_render as gr`）—— 本包刻意不做 re-export：
    board_state（BoardState 唯一状态载体）／ completion（完成度三量）／ grid_render（L 掩码 + G 增益）
    lookahead（单步预测）／ obs_spec（obs 口径）／ order_plan（plan_order）／ pcb_env（PcbEnv 环境层）
    violations（Judge）／ reward（奖励公式）／ policy_net + dqn_agent（Double DQN，含 PackedReplay）

上一代（`pcb_layout_env` / `pcb_utils` / `mock_pcb`）已于 2026-09-24 删除（历史见 `590fa92`），
随之移除的还有为它们写的 PEP 562 惰性导出（`__all__` / `_LEGACY_EXPORTS` / `__getattr__`）。
"""
