# -*- coding: utf-8 -*-
"""违规判定**正向测试**（`env/violations.py`）。

依据：《导入信息重构》§0.2（三条硬约束）、§3.1~§3.5（判定式）、§3.6（标准设计原则）、
§1.4（层，元件不跨层移动）、§4.1（转换数据）、§0.6 k（违规升级为硬约束）。

两类断言
  1. **合成微型模型**（几何不依赖真实板，**也不依赖 pcbnew**）：逐条覆盖 7 个违规方向 + 全部边界
     （相切 / 阈值端点 / 层隔离 / net 过滤 / 归属排除 / others 含自身 / 缓存串味）+ **构造参数**
     （`margin` / `soft_gap`）与**便捷入口**（`judge_model`）——
     与 `test_violations_mutation.py` 共用同一套检查（见 `violation_checks.py`）。
     注：能否在**未设 `KICAD_PATH`** 的环境运行，取决于导入链
     （`env/__init__` → `pcb_layout_env` → `utils.kicad_utils`）能否被导入；
     该链已由 `kicad_utils.py` 顶部的 `from __future__ import annotations` 解耦（原先会 `NameError: pcbnew`）。
  2. **真实 KiCad 数据**：20 板源布局的 §0.4 数字（硬重叠 4/36/4、软间距 5/133/0、出界 160）、
     单件精确计数（probe83/85 取证的实例）、以及基于真实几何**构造**的违规位姿
     （S3 在源数据里 0 例 ⇒ 只能构造）。

运行（需要 KiCad 环境）：
    $env:KICAD_PATH="E:\\Kicad"; python -m pytest tests/kicad/test_violations.py -v
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import env.violations as violations  # noqa: E402
from violation_checks import (  # noqa: E402
    REAL_CHECKS_FAST,
    REAL_CHECKS_SLOW,
    SYNTHETIC_CHECKS,
    real_data_all,
    real_data_fast,
)


def _require_kicad() -> None:
    if not os.environ.get("KICAD_PATH") or not os.path.isdir(os.environ["KICAD_PATH"]):
        pytest.skip("KICAD_PATH not set or does not exist")
    pytest.importorskip("pcbnew", reason="需要 KiCad 环境（KICAD_PATH）")


@pytest.fixture(scope="session")
def real_fast():
    """5 块真实板（覆盖 A/B/C/S1/S2/出界 + 整板零违规 + 干净件）。"""
    _require_kicad()
    return real_data_fast()


@pytest.fixture(scope="session")
def real_all():
    """全部 20 块真实板（§0.4 全量回归）。"""
    _require_kicad()
    return real_data_all()


# ===================== 一、合成微型模型：逐条规则与边界 =====================


@pytest.mark.parametrize("name,fn", SYNTHETIC_CHECKS, ids=[n for n, _ in SYNTHETIC_CHECKS])
def test_rule_synthetic(name, fn):
    """合成模型逐条规则检查（不依赖 pcbnew 几何）。"""
    fn(violations)


# ===================== 二、真实 KiCad 数据 =====================


@pytest.mark.require_kicad
@pytest.mark.parametrize("name,fn", REAL_CHECKS_FAST, ids=[n for n, _ in REAL_CHECKS_FAST])
def test_rule_real_fast(name, fn, real_fast):
    """真实板快速检查（每板计数 / 精确实例 / 构造位姿）。"""
    fn(violations, real_fast)


@pytest.mark.require_kicad
@pytest.mark.parametrize("name,fn", REAL_CHECKS_SLOW, ids=[n for n, _ in REAL_CHECKS_SLOW])
def test_rule_real_all_boards(name, fn, real_all):
    """§0.4 全量回归（20 板）。"""
    fn(violations, real_all)


# ===================== 三、接口不变式（与台账对齐） =====================


def test_reason_codes_match_doc():
    """7 个违规方向必须与台账 §0.2/§3 的判定式一一对应（顺序 = 汇总顺序）。"""
    assert violations.ALL_REASONS == (
        "hard.pad_pad", "hard.pad_via", "hard.pad_keepout",
        "soft.pad_pad", "soft.pad_via", "soft.pad_keepout",
        "out_of_bounds",
    )
    assert violations.SOFT_GAP_MM == 0.2, "软间距阈值 = 原生 netclass clearance（§3.4）"


def test_pose_has_no_layer():
    """拍案（§1.4 新规定）：动作空间**不含层动作** ⇒ 位姿里不能出现层。"""
    import inspect

    assert violations.Pose._fields == ("x", "y", "rot"), f"Pose 字段异常：{violations.Pose._fields}"
    params = list(inspect.signature(violations.Judge.judge).parameters)
    assert "layer" not in params, f"judge 不应有层参数：{params}"
    assert params[:3] == ["self", "ref", "pose"], f"judge 参数异常：{params}"


def test_result_semantics():
    """`ViolationResult` 的合并语义：三层任一命中即总违规；空结果 == 无违规。"""
    ok = violations.ViolationResult()
    assert ok.is_violation is False and bool(ok) is False and ok.describe() == "ok"
    only_soft = violations.ViolationResult(reasons=(violations.SOFT_PAD_VIA,),
                                           counts={violations.SOFT_PAD_VIA: 2})
    assert only_soft.is_violation is True and only_soft.soft is True
    assert only_soft.hard is False and only_soft.out_of_bounds is False
    assert "soft.pad_via" in only_soft.describe()
