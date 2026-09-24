# -*- coding: utf-8 -*-
"""环境层：`PcbEnv` —— 台账 `docs/讨论结果/导入信息重构.md` §9.1 的落地。

本文件只做**接线**，口径一律引用既有定稿：动作 §6.1 ｜ L/G §5.3/§5.4 ｜ obs §5.5 ｜
mask（训练 `L & lookahead`、推理 `L`）§6.3 ｜ 终止 §5.4 ｜ 精度分工（`L` 快校 + `Judge` 复核）§5.7 ｜
reward §7.3 ｜ 完成度三量 §0.6 m ｜ 状态载体 `BoardState`（唯一写入口）§0.6 o。

reset（§9.1.3）
    `model_cache.load_model(board, allow_build=False)`（训练机无 KiCad；缓存缺失 ⇒ 报错、不回落）
    → `plan_order` / `grid_spec` → **A 层常驻**（`CompletionTable` / `β` / `H_ref` / `D` / 件特征）
    → `BoardState.start` → 算当前帧（`L` / `G` / `mask`）→ obs / info

step（§9.1.3，逐步对应注释里的 ①~⑧）
    ① `comp = state.comp`（固定顺序 ⇒ 不选件）；`L` ② `mask` ③ 合法性（**被 mask 阻挡**）④ `Judge` 复核
    ⑤ `state.place` ⑥ reward ⑦ 终止判定 ⑧ 返回 obs / reward / terminated / truncated / info

Owner 2026-09-22 的三条拍案（§9.1.5）
    ① 文件名/类名 = `env/pcb_env.py` / `PcbEnv`（Owner："自己定，不需要问"）
    ⑦ 落在 `mask` 之外的动作 **被 mask 阻挡** ⇒ 本文件抛 `EnvActionError`（fail-fast）：
      采样侧必须只在 `mask` 内采（训练：argmax over mask ／ ε 时 uniform over mask；推理：argmax over mask）
    ⑨ 末件（`next_comp is None`）⇒ `mask = L`（与 `lookahead` 既有边界"下一件本来就无位置 ⇒ 原样 pass"同源）

⚠️ 本文件里两处**我方自定**（台账未写、Owner 未拍，已如实登记待确认）
    * `deadlock_kind` 的判据：`at == "L"` 且 `index == 0` ⇒ `structural`｜`at == "L"` 且 `index > 0` ⇒ `policy`
      ｜`at == "mask"`（`L` 非空但被预测掩膜清空）⇒ `order_locked`。台账只列了三个名字、未给定义。
    * `info` 多两个键：`terminated`（reset 后即终止也能看出来）与 `skipped_dead_start`（见下）。
    * 开局的 `L` 就空（"结构性死板"，如台账点名的 `trumpx/U1`）默认**重采一块板**（0 步 episode 无学习信号）；
      要保留该板则 `resample_dead_start=False`（此时 `action_masks()` 全 False + `info["deadlock_kind"]="structural"`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import board_state as bs
from env import completion as cp
from env import grid_render as gr
from env import lookahead as la
from env import obs_spec as osp
from env import order_plan as op
from env import reward as rw
from env import violations as vio
from utils.kicad_utils import model_cache as mc

__all__ = [
    "PcbEnv",
    "Frame",
    "EnvContractError",
    "EnvActionError",
    "EnvDoneError",
    "DEADLOCK_KINDS",
    "OBS_SPATIAL",
    "OBS_PIECE",
]

# `info["deadlock_kind"]` 的取值域（台账 §9.1.4 / §12）
DEADLOCK_KINDS = ("structural", "order_locked", "policy")

# obs 的两个键（§5.5：`spatial (8,200,200)` + `piece (3,)`）
OBS_SPATIAL = "spatial"
OBS_PIECE = "piece"


class EnvContractError(RuntimeError):
    """口径被破坏（典型：`L` 说合法而 `Judge` 判违规 —— §6.3 的"两层校验不一致即报错"）。"""


class EnvActionError(ValueError):
    """动作落在 `mask` 之外（Owner 2026-09-22 拍：**被 mask 阻挡**）。"""


class EnvDoneError(RuntimeError):
    """episode 已终止却继续 `step`。"""


@dataclass(frozen=True)
class Frame:
    """**同一帧**的 `L` / `G` / `mask`（同 `placed` + 同 `keepouts`，§4.2-6 纪律）。

    `legal` = `L`（硬重叠 ∪ 软间距 ∪ 出界，§5.3）；`gain` = `G`（非法格 `−inf`，§5.4）；
    `mask` = 动作掩膜（训练 `L & lookahead`，推理/末件 = `L` 的副本）。
    """

    legal: np.ndarray                     # bool (4, m, m)
    gain: np.ndarray                      # float32 (4, m, m)，非法格 −inf
    mask: np.ndarray                      # bool (4, m, m)
    degenerate: bool                      # G 两分量同时退化（只作元数据，§5.4）
    lookahead_applied: bool               # False = 末件或推理档（mask == legal）
    lookahead_info: Dict[str, Any] = field(default_factory=dict)


class PcbEnv(gym.Env):
    """PCB 布局环境（新一代；上一代 `env/pcb_layout_env.py` 仍在原位、互不引用）。

    Args:
        board: 固定板名（缺省 ⇒ 每次 `reset` 从缓存索引里按 `np_random` 抽一块）。
        boards: 候选板名集合（缺省 ⇒ `model_cache.read_index()` 的全部板）。
        cache_root: 缓存目录覆盖（缺省 `data/model_cache`）。
        use_lookahead: True = 训练档（`mask = L & lookahead`）；False = 推理档（`mask = L`，§6.3）。
        verify: True = 落子前用 `Judge` 精确复核（连续 mm、1nm，§5.7）；不一致 ⇒ `EnvContractError`。
        with_contrib: 完成度 ③（贡献占比）是否建表（要跑 n 次 `hpwl`，只影响 `info`）。
        resample_dead_start: 开局 `L` 就空时是否重采一块板（见模块说明）。
        seed: 初始种子（与 `reset(seed=...)` 等价）。
    """

    metadata = {"render_modes": [], "name": "pcb_layout_v1"}

    def __init__(self, board: Optional[str] = None, *,
                 boards: Optional[Sequence[str]] = None,
                 cache_root: Optional[str] = None,
                 use_lookahead: bool = True,
                 verify: bool = True,
                 with_contrib: bool = True,
                 resample_dead_start: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.requested_board = board
        self.candidate_boards = tuple(boards) if boards else None
        self.cache_root = cache_root
        self.use_lookahead = bool(use_lookahead)
        self.verify = bool(verify)
        self.with_contrib = bool(with_contrib)
        self.resample_dead_start = bool(resample_dead_start)

        m = gr.GRID_M
        self.action_space = spaces.MultiDiscrete([osp.N_ROT, m, m])
        self.observation_space = spaces.Dict({
            OBS_SPATIAL: spaces.Box(0.0, 1.0, shape=(osp.SPATIAL_CHANNELS, m, m), dtype=np.float32),
            # `piece` 的第三个量是 t/n ∈ [0,1]，前两个量**不裁剪**（超 MAX 保留信息，§5.5）⇒ 上界取 inf
            OBS_PIECE: spaces.Box(0.0, np.inf, shape=(len(osp.PIECE_KEYS),), dtype=np.float32),
        })

        # ---- 每 episode 重算的成员（类型声明便于阅读；reset 前为 None） ----
        self.board: Optional[str] = None
        self.model = None                  # PcbModel
        self.order: Tuple[str, ...] = ()
        self.grid: Optional[gr.GridSpec] = None
        self.table: Optional[cp.CompletionTable] = None
        self.beta: float = 0.0
        self.h_ref: float = 0.0
        self.d_board: float = 0.0
        self.features: Dict[str, Tuple[int, int]] = {}
        self.state: Optional[bs.BoardState] = None

        self._frame: Optional[Frame] = None
        self._h: float = 0.0              # 当前局面的 HPWL（当前位姿；§7.3）
        self._s: float = 0.0              # 当前局面的 σ
        self._terminated: bool = True
        self._deadlock_kind: Optional[str] = None
        self._deadlock_at: Optional[str] = None
        self._skipped_dead_start: Tuple[str, ...] = ()

        if seed is not None:
            self.reset(seed=seed)          # 便于 `PcbEnv(board, seed=0)` 直接用

    # ===================== 只读访问 =====================

    @property
    def frame(self) -> Optional[Frame]:
        """当前帧（`L`/`G`/`mask`）；终局（放完）时为 `None`。"""
        return self._frame

    def action_masks(self) -> np.ndarray:
        """当前 `action_mask`（`(4, m, m)` bool；终止后全 `False`）。

        **采样侧必须只用这张表**（Owner 2026-09-22：mask 外的动作被阻挡）——
        与 `sb3-contrib` 的 `MaskablePPO` 约定同名，可直接被它识别。
        """
        m = gr.GRID_M
        if self._terminated or self._frame is None:
            return np.zeros((osp.N_ROT, m, m), dtype=bool)
        return self._frame.mask

    # ===================== reset =====================

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        super().reset(seed=seed)
        opt = dict(options or {})
        board = opt.pop("board", None) or self.requested_board
        if board is None:
            board = str(self.np_random.choice(self._board_names()))
        if opt:
            raise ValueError(f"未知 options：{sorted(opt)}（当前支持 'board'）")

        skipped: list = []
        while True:
            self._load(str(board))
            self._enter(self.state)
            if (self._terminated and self._deadlock_kind == "structural"
                    and self.resample_dead_start):
                skipped.append(str(board))
                pool = [b for b in self._board_names() if b not in skipped]
                if not pool:
                    break                      # 全是结构死板 ⇒ 认了，把当前这块报出去
                board = str(self.np_random.choice(pool))
                continue
            break

        self._skipped_dead_start = tuple(skipped)
        return self._obs(), self._info()

    def _board_names(self) -> Tuple[str, ...]:
        if self.candidate_boards:
            return self.candidate_boards
        return tuple(e["board"] for e in mc.read_index(root=self.cache_root)["boards"])

    def _load(self, board: str) -> None:
        """取板 + 建 A 层（§9.1.3 的 reset 前半段）。"""
        self.board = board
        self.model = mc.load_model(board, root=self.cache_root, allow_build=False)   # 训练机口径
        self.order = tuple(op.plan_order(self.model))
        self.grid = gr.grid_spec(self.model, m=gr.GRID_M)
        self.table = cp.CompletionTable.build(self.model, self.order, with_contrib=self.with_contrib)
        self.beta = float(rw.beta_of(self.model, self.order))
        self.h_ref = float(rw.h_ref_of(self.model))
        self.d_board = float(rw.d_board_of(self.model))
        self.features = op.piece_features(self.model)
        self.state = bs.BoardState.start(self.model, self.order)

    # ===================== 帧（L / G / mask） =====================

    def _make_frame(self, state: bs.BoardState) -> Frame:
        """算**同一帧**的 `L` / `G` / `mask`（§4.2-6；全部走 `state` 的 `placed` / `keepouts`）。"""
        comp = state.comp
        legal = gr.legal_masks(self.model, state.placed, comp, self.grid, keepouts=state.keepouts)
        gain, degenerate = gr.gain_maps(self.model, state.placed, comp, self.grid,
                                        legal=legal, keepouts=state.keepouts)
        nxt = state.next_comp
        if self.use_lookahead and nxt is not None:
            mask, la_info = la.lookahead_mask(self.model, state.placed, comp, nxt, self.grid,
                                              legal=legal, keepouts=state.keepouts)
            return Frame(legal=legal, gain=gain, mask=mask, degenerate=bool(degenerate),
                         lookahead_applied=True, lookahead_info=dict(la_info))
        # 末件（next_comp is None，Owner 2026-09-22 ⑨）或推理档（§6.3）⇒ mask = L
        return Frame(legal=legal, gain=gain, mask=legal.copy(), degenerate=bool(degenerate),
                     lookahead_applied=False, lookahead_info={})

    def _enter(self, state: bs.BoardState) -> bool:
        """进入 `state`（算 h/σ、帧、终止标记）；返回 `terminated`。"""
        self.state = state
        self._h = self._hpwl(state)
        self._s = float(gr.dispersion_of(self.model, state.placed))
        if state.done:
            self._frame = None
            self._terminated, self._deadlock_kind, self._deadlock_at = True, None, None
            return True
        frame = self._make_frame(state)
        self._frame = frame
        if not frame.mask.any():
            self._terminated = True
            self._deadlock_at = "L" if not frame.legal.any() else "mask"
            self._deadlock_kind = self._classify_deadlock(state, self._deadlock_at)
            return True
        self._terminated, self._deadlock_kind, self._deadlock_at = False, None, None
        return False

    @staticmethod
    def _classify_deadlock(state: bs.BoardState, at: str) -> str:
        """⚠️ 本判据是**我方自定**（台账只列了三个名字，未给定义，见模块说明）。

        * `at == "L"` 且 `index == 0` ⇒ `structural`（**开局**就无解 = 板本身，与任何选择无关）。
        * `at == "L"` 且 `index > 0` ⇒ `policy`（**路径依赖**：不是开局就无解）。
          ⚠️ 诚实标注：这条只说"不是开局无解"，**不排除"固定顺序本身把大件排太后"的锅**；
          要严格区分"策略自陷" vs "顺序锁死"需要离线 oracle（对每个前缀做可行性判定），本期没做。
        * `at == "mask"`（`L` 非空、被单步预测掩膜清空）⇒ `order_locked`。
        """
        if at == "L":
            return "structural" if state.index == 0 else "policy"
        return "order_locked"

    def _hpwl(self, state: bs.BoardState) -> float:
        """当前位姿下的 HPWL（⚠️ 必须传 `poses`，否则取到导入快照，§7.1）。"""
        return float(self.model.hpwl(refs=set(state.placed_refs), poses=state.poses))

    # ===================== step =====================

    def step(self, action: Union[int, Sequence[int]]):
        if self._terminated or self._frame is None:
            raise EnvDoneError(
                f"episode 已终止（board={self.board}，deadlock_kind={self._deadlock_kind}，"
                f"at={self._deadlock_at}）⇒ 请先 reset()")

        k, i, j = self._decode(action)
        frame = self._frame
        if not bool(frame.mask[k, i, j]):                      # ③ 被 mask 阻挡（Owner 2026-09-22）
            why = ("在 L 内但被单步预测掩膜砍掉" if bool(frame.legal[k, i, j]) else "落在 L 之外")
            raise EnvActionError(
                f"动作 {k}@({i},{j}) {why} ⇒ 被 mask 阻挡（采样侧必须只在 action_masks() 内采）")
        if self.verify:                                        # ④ 执行复核（连续 mm、1nm，§5.7）
            self._verify(self.state, k, i, j)

        h_prev, s_prev = self._h, self._s                       # ⑤ 落子（唯一写入口）
        state2 = self.state.place(k, i, j, spec=self.grid, legal=frame.mask)
        terminated = self._enter(state2)
        r = float(rw.step_reward(d_hpwl=self._h - h_prev, d_sigma=self._s - s_prev,   # ⑥ §7.3
                                 h_ref=self.h_ref, d_board=self.d_board, beta=self.beta))
        return self._obs(), r, terminated, False, self._info()  # ⑦⑧ 截断恒 False（§9.1.5 ⑧）

    def _decode(self, action: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
        """动作 = `[rot, X, Y]`（§6.1）；也接受一维化整数 `rot·m² + X·m + Y`。"""
        m = gr.GRID_M
        if isinstance(action, (int, np.integer)):
            a = int(action)
            if not (0 <= a < osp.N_ROT * m * m):
                raise ValueError(f"一维动作越界：{a} ∉ [0, {osp.N_ROT * m * m})")
            k, rem = divmod(a, m * m)
            i, j = divmod(rem, m)
            return k, i, j
        seq = tuple(action)
        if len(seq) != 3:
            raise ValueError(f"动作必须是 (rot, X, Y) 或一维整数，收到 {action!r}")
        k, i, j = (int(v) for v in seq)
        if not (0 <= k < osp.N_ROT and 0 <= i < m and 0 <= j < m):
            raise ValueError(f"动作越界：{(k, i, j)} ∉ [0,{osp.N_ROT})×[0,{m})²")
        return k, i, j

    def _verify(self, state: bs.BoardState, k: int, i: int, j: int) -> None:
        """§6.3 两层校验：格级 `L` 说合法 ⇒ 连续 mm 的 `Judge` 必须也认可，否则报错（不静默放行）。"""
        x, y = self.grid.mm_of(i, j)
        judge = vio.Judge(self.model, keepouts=state.keepouts, placed=state.placed)
        res = judge.judge(state.comp.ref, (float(x), float(y), int(k)), state.others)
        if res.is_violation:
            raise EnvContractError(
                f"L 与 Judge 不一致（§6.3）：{state.comp.ref} @ k={k} 格({i},{j}) → "
                f"mm({x:.4f}, {y:.4f}) 被判违规 ⇒ {res.describe()}")

    # ===================== obs / info =====================

    def _obs(self) -> Dict[str, np.ndarray]:
        m = gr.GRID_M
        if self._frame is None:                                 # 终局（放完）：占位 obs（不做 bootstrap）
            return {OBS_SPATIAL: np.zeros((osp.SPATIAL_CHANNELS, m, m), dtype=np.float32),
                    OBS_PIECE: np.asarray((0.0, 0.0, 1.0), dtype=np.float32)}
        frame = self._frame
        gain_sane = osp.sanitize_gain(frame.gain, frame.legal)  # 非法格 −inf → 0（§5.5 / A3）
        spatial = np.zeros((osp.SPATIAL_CHANNELS, m, m), dtype=np.float32)
        for k in range(osp.N_ROT):
            spatial[osp.channel_of(k, "L")] = frame.legal[k]
            spatial[osp.channel_of(k, "G")] = gain_sane[k]
        pad_count, np_net_count = self.features[self.state.comp.ref]
        piece = np.asarray(osp.piece_vector(pad_count, np_net_count, self.state.index,
                                            self.state.n_total), dtype=np.float32)
        return {OBS_SPATIAL: spatial, OBS_PIECE: piece}

    def _info(self) -> Dict[str, Any]:
        """`info`（§9.1.6 草案键 + 本文件自加的两个键）—— 不进网络、不进 reward。"""
        state = self.state
        info: Dict[str, Any] = self.table.of(state).as_info()
        info.update(
            board=self.board,
            H_ref=self.h_ref,
            degenerate=(bool(self._frame.degenerate) if self._frame is not None else False),
            deadlock_kind=self._deadlock_kind,
            at=self._deadlock_at,
            audit_failures=int(bs.audit_failures()),
            terminated=bool(self._terminated),                 # ⚠️ 我方自加
            skipped_dead_start=list(self._skipped_dead_start),  # ⚠️ 我方自加
        )
        return info
