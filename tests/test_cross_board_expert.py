"""跨板专家轨迹生成数量对齐单元测试（纯 Python，无需 KiCad/训练）。

验证：
1. divmod 整除分配使 sum(per_board) == 总目标，且每块≈总量/N；
2. N=1 时 per_board == 总目标（单板兼容，result/compare 基线可比）；
3. 缓存充足分支随机采样后每板长度 == per_board、总量守恒、固定 seed 可复现；
4. 集成：构造 fake Runner 跑 _generate_or_load_expert_transitions，
   - 缓存充足分支：all_transitions 长度 == 总目标，且每块板贡献 == per_board（N=1/3/5）；
   - 无缓存分支：生成调用 total_transitions == per_board，总量守恒。
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run  # noqa: E402
from run import Runner, EXPERT_SAMPLE_SEED  # noqa: E402

TOTAL = 70000  # EXPERT_TOTAL_TRANSITIONS 默认派生值
CACHE = 70020  # 旧缓存每板条数


def _alloc(total, n):
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


# ---------- 纯逻辑 ----------

def test_divmod_conservation():
    for n in (1, 2, 3, 5, 7, 13, 50):
        alloc = _alloc(TOTAL, n)
        assert sum(alloc) == TOTAL
        floor = TOTAL // n
        assert all(a == floor or a == floor + 1 for a in alloc)


def test_single_board_compat():
    assert _alloc(TOTAL, 1) == [TOTAL]


def test_random_sample_reproducible():
    n_cached, per_board = CACHE, TOTAL // 5
    i1 = np.random.default_rng(EXPERT_SAMPLE_SEED).choice(n_cached, per_board, replace=False)
    i2 = np.random.default_rng(EXPERT_SAMPLE_SEED).choice(n_cached, per_board, replace=False)
    assert len(i1) == per_board
    assert list(i1) == list(i2)


# ---------- 集成（fake Runner，monkeypatch 生成/加载） ----------

class _FakeEnv:
    pass


def _make_runner(n_boards):
    runner = Runner.__new__(Runner)  # 不调用 __init__，避免加载 KiCad/配置
    runner.use_expert = True
    pool = [(f"b{k}.kicad_pcb", None) for k in range(n_boards)]
    runner.pcb_pool = pool
    runner.envs = {name: _FakeEnv() for name, _ in pool}
    runner.expert_transitions_dir = "data/expert_transitions"  # 仅占位，isfile 由 monkeypatch 控制
    return runner


def _fake_load(filepath):
    # 文件名形如 b{k}_expert.json，返回带板号标记的假缓存（长度=CACHE）
    k = int(os.path.basename(filepath).split("b")[1].split("_")[0])
    return [(k,)] * CACHE


def _fake_gen(**kwargs):
    k = int(kwargs["pcb_name"].split("b")[1].split(".")[0])
    return [(k,)] * kwargs["total_transitions"], kwargs.get("output_dir")


def test_cache_branch_per_board_split(tmp_path, monkeypatch):
    monkeypatch.setattr("env.expert_trajectory.ExpertTrajectoryGenerator.load_transitions", _fake_load)
    monkeypatch.setattr(run, "generate_and_save_expert_transitions", _fake_gen)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)  # 一律视为命中缓存

    for n in (1, 3, 5):
        runner = _make_runner(n)
        result = runner._generate_or_load_expert_transitions()
        assert len(result) == TOTAL
        counts = Counter(t[0] for t in result)
        for k in range(n):
            assert counts[k] == _alloc(TOTAL, n)[k], f"板 {k} 贡献 {counts[k]} != per_board"


def test_generate_branch_per_board_split(tmp_path, monkeypatch):
    monkeypatch.setattr("env.expert_trajectory.ExpertTrajectoryGenerator.load_transitions", _fake_load)
    monkeypatch.setattr(run, "generate_and_save_expert_transitions", _fake_gen)
    monkeypatch.setattr(os.path, "isfile", lambda p: False)  # 一律视为无缓存，走生成分支

    for n in (1, 5):
        runner = _make_runner(n)
        result = runner._generate_or_load_expert_transitions()
        assert len(result) == TOTAL
        counts = Counter(t[0] for t in result)
        for k in range(n):
            assert counts[k] == _alloc(TOTAL, n)[k], f"板 {k} 生成 {counts[k]} != per_board"


def test_empty_pool_returns_empty():
    runner = Runner.__new__(Runner)
    runner.use_expert = True
    runner.pcb_pool = []
    runner.envs = {}
    runner.expert_transitions_dir = "data/expert_transitions"
    assert runner._generate_or_load_expert_transitions() == []
