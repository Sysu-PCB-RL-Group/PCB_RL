"""层感知碰撞修复测试。

覆盖：
- PCBUtils.layers_conflict 纯函数语义（全部层组合 + 开关）。
- 训练主路径 _detect_collisions 接入同层过滤（含 layer==2 隔离、开关回退）。
- 终局主指标 evaluate_final_layout 对异层重叠不罚、同层罚。

通过 sys.modules 注入 MockKiCadUtils，避免本环境无 pcbnew 时导入失败；
fixture 为 data/模拟/pcb_dual_layer.json（A=TOP/B=BOTTOM/C,D=内层，均同 XY 重叠）。
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest import mock

# ---- Mock 注入（必须在导入 env 相关模块前完成）----
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m
# run_inference 顶层 import torch。本环境(pcb_rl_311)已有真实 torch，无需 mock；
# 仅当真实 torch 不可用(如纯净 core 环境)时才 mock，避免污染同 pytest 会话内
# 其它真实 torch 测试（全局 sys.modules["torch"]=MagicMock 会泄漏到后续测试模块）。
try:
    import torch  # noqa: F401  真实 torch 可导入则不 mock
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
if not _HAS_TORCH:
    sys.modules["torch"] = mock.MagicMock()
    for _sub in ("torch.nn", "torch.nn.functional", "torch.optim", "torch.utils"):
        sys.modules[_sub] = mock.MagicMock()

from env.mock_pcb import create_env_from_json, MockKiCadUtils, MockComponent, MockNet  # noqa: E402

import utils  # noqa: E402

utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_utils import PCBUtils  # noqa: E402
from env.pcb_layout_env import Config  # noqa: E402

_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "模拟", "pcb_dual_layer.json")


def _make_dual_env():
    env = create_env_from_json(_JSON_PATH)
    env.reset()
    return env


def _place(env, comp_id, bbox, layer):
    c = env.kicad.components[comp_id]
    c.bbox = bbox
    c.layer = layer
    c.is_placed = True
    if comp_id not in env.placed_list:
        env.placed_list.append(comp_id)
    return c


# ==================== 纯函数：layers_conflict ====================

def test_layers_conflict_top_vs_bottom_false():
    assert PCBUtils.layers_conflict(0, 1, True) is False
    assert PCBUtils.layers_conflict(1, 0, True) is False


def test_layers_conflict_same_layer_true():
    assert PCBUtils.layers_conflict(0, 0, True) is True
    assert PCBUtils.layers_conflict(1, 1, True) is True


def test_layers_conflict_layer2_vs_surface_false():
    # 内层/埋入件与表层元件隔离
    assert PCBUtils.layers_conflict(2, 0, True) is False
    assert PCBUtils.layers_conflict(2, 1, True) is False
    assert PCBUtils.layers_conflict(0, 2, True) is False


def test_layers_conflict_layer2_vs_layer2_true():
    # 两个内层件按同层保守处理
    assert PCBUtils.layers_conflict(2, 2, True) is True


def test_layers_conflict_dual_sided_off_ignores_layer():
    # 开关关闭：回退旧行为（忽略层，全部按同层）
    assert PCBUtils.layers_conflict(0, 1, False) is True
    assert PCBUtils.layers_conflict(2, 0, False) is True


# ==================== 集成：_detect_collisions 同层过滤 ====================

def test_detect_collisions_cross_layer_no_penalty():
    env = _make_dual_env()
    _place(env, "A", (10, 10, 20, 20), 0)
    B = _place(env, "B", (10, 10, 20, 20), 1)  # BOTTOM，与 TOP 的 A 正对重叠
    hard, soft, oob, keep = env._detect_collisions(B)
    assert hard == 0.0, f"异层重叠应不计硬重叠, got {hard}"
    assert soft == 0.0, f"异层重叠应不计软间距, got {soft}"


def test_detect_collisions_same_layer_penalized():
    env = _make_dual_env()
    _place(env, "A", (10, 10, 20, 20), 0)
    B = _place(env, "B", (10, 10, 20, 20), 0)  # 同为 TOP，正对重叠
    hard, soft, oob, keep = env._detect_collisions(B)
    assert hard > 0.0, "同层重叠必须判硬重叠"


def test_detect_collisions_layer2_isolated():
    env = _make_dual_env()
    _place(env, "A", (10, 10, 20, 20), 0)
    C = _place(env, "C", (10, 10, 20, 20), 2)  # 内层件 vs 表层件
    hard, soft, oob, keep = env._detect_collisions(C)
    assert hard == 0.0, "内层件与表层件应隔离"


def test_detect_collisions_layer2_vs_layer2_conflict():
    env = _make_dual_env()
    _place(env, "C", (10, 10, 20, 20), 2)
    D = _place(env, "D", (10, 10, 20, 20), 2)  # 两个内层件正对重叠
    hard, soft, oob, keep = env._detect_collisions(D)
    assert hard > 0.0, "两个内层件应保守判冲突"


def test_detect_collisions_dual_sided_off_legacy_behavior():
    env = _make_dual_env()
    _place(env, "A", (10, 10, 20, 20), 0)
    B = _place(env, "B", (10, 10, 20, 20), 1)
    prev = Config.ENABLE_DUAL_SIDED
    Config.ENABLE_DUAL_SIDED = False
    try:
        hard, soft, oob, keep = env._detect_collisions(B)
    finally:
        Config.ENABLE_DUAL_SIDED = prev
    assert hard > 0.0, "开关关闭应回退旧行为(忽略层误判冲突)"


# ==================== 集成：终局主指标 evaluate_final_layout ====================

def test_evaluate_final_layout_cross_layer_clean():
    from run_inference import evaluate_final_layout

    env = _make_dual_env()
    _place(env, "A", (10, 10, 20, 20), 0)
    _place(env, "B", (10, 10, 20, 20), 1)
    cross = evaluate_final_layout(env)

    env2 = _make_dual_env()
    _place(env2, "A", (10, 10, 20, 20), 0)
    _place(env2, "B", (10, 10, 20, 20), 0)  # 同层
    same = evaluate_final_layout(env2)

    # 异层不罚、同层罚硬重叠 → 异层终局分应更高
    assert cross > same, f"异层终局分应高于同层, cross={cross} same={same}"


# ==================== 修复回归：Config bool 类型守卫 + 缺层 fail-loud ====================

def test_config_bool_type_guard_keeps_numeric():
    """数值型配置误写为 true/false 不应被静默解析为 bool（保留原 int），
    同时真正的 bool 开关 ENABLE_DUAL_SIDED 仍正确解析。"""
    import tempfile

    prev = (Config.REPLAY_BUFFER_SIZE, Config.ENABLE_DUAL_SIDED)
    tf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tf.write("REPLAY_BUFFER_SIZE = true\nENABLE_DUAL_SIDED = False\n")
    tf.close()
    try:
        Config.load_from_file(tf.name)
        assert Config.REPLAY_BUFFER_SIZE == prev[0], "数值型 true 不应静默污染 REPLAY_BUFFER_SIZE"
        assert Config.ENABLE_DUAL_SIDED is False, "ENABLE_DUAL_SIDED=False 应正确解析为 bool"
    finally:
        Config.REPLAY_BUFFER_SIZE, Config.ENABLE_DUAL_SIDED = prev
        os.unlink(tf.name)


def test_is_legal_placement_missing_layer_fails_loud():
    """快照缺失 layer 应 fail-loud(KeyError)，而非被静默当作 TOP 误判碰撞。"""
    from env.expert_trajectory import ExpertTrajectoryGenerator

    env = _make_dual_env()
    valid_area = env.valid_area
    bad_snapshot = {
        "A": {"bbox": (10, 10, 20, 20), "package_type": 0},  # 故意缺 layer
        "B": {"bbox": (10, 10, 20, 20), "package_type": 0, "layer": 0},
    }
    gen = ExpertTrajectoryGenerator()
    with pytest.raises(KeyError):
        gen._is_legal_placement((10, 10, 20, 20), "A", bad_snapshot, valid_area, env)
