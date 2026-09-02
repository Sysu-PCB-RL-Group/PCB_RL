"""可视化模块单元测试。

验证 draw_layout 对不同板框尺寸、元件数不崩溃，且网格/刻度对齐 0.2mm 颗粒度。
"""
import sys
import os
from unittest import mock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import pytest

# matplotlib 在 test 环境中必须是 Agg backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.pcb_layout_env import Config


def _make_comp(cid, x1, y1, x2, y2, is_fixed=False, nets=None):
    """构造 draw_layout 兼容的元件对象。"""
    class VizComp:
        pass
    c = VizComp()
    c.id = cid
    c.bbox = (x1, y1, x2, y2)
    c.is_fixed = is_fixed
    c.is_placed = True
    c.nets = nets or []
    return c


def test_draw_small_board():
    """小板框 + 少量元件 → 不崩溃。"""
    from utils.visualization import draw_layout
    comps = [
        _make_comp("U1", 5, 5, 15, 15),
        _make_comp("C1", 20, 20, 22, 21, nets=["VCC"]),
    ]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout(comps, ((0, 0), (50, 30)), tmp, title="Test Small Board")
        assert os.path.getsize(tmp) > 100, "PNG file should not be empty"
    finally:
        os.unlink(tmp)


def test_draw_empty_components():
    """空元件列表 → 不崩溃。"""
    from utils.visualization import draw_layout
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout([], ((0, 0), (100, 80)), tmp)
        assert os.path.getsize(tmp) > 100
    finally:
        os.unlink(tmp)


def test_draw_with_fixed_and_movable():
    """固定 + 可移动混合 → 不同颜色、不崩溃。"""
    from utils.visualization import draw_layout
    comps = [
        _make_comp("J1", 0, 20, 10, 30, is_fixed=True),
        _make_comp("J2", 90, 20, 100, 30, is_fixed=True),
        _make_comp("U1", 20, 10, 30, 20),
        _make_comp("C1", 40, 15, 42, 16, nets=["VCC", "GND"]),
    ]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout(comps, ((0, 0), (100, 60)), tmp, title="Test Mixed")
        assert os.path.getsize(tmp) > 100
    finally:
        os.unlink(tmp)


def test_draw_wide_board():
    """极宽板（长宽比 >> 1）→ 不崩溃。"""
    from utils.visualization import draw_layout
    comps = [_make_comp("R1", 10, 5, 13, 6.5)]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout(comps, ((0, 0), (200, 20)), tmp)
        assert os.path.getsize(tmp) > 100
    finally:
        os.unlink(tmp)


def test_draw_tall_board():
    """极高板（长宽比 << 1）→ 不崩溃。"""
    from utils.visualization import draw_layout
    comps = [_make_comp("R1", 5, 10, 6.5, 13)]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout(comps, ((0, 0), (20, 200)), tmp)
        assert os.path.getsize(tmp) > 100
    finally:
        os.unlink(tmp)


def test_draw_with_zero_size_component():
    """含零尺寸 bbox 的元件应被跳过不崩溃。"""
    from utils.visualization import draw_layout
    comps = [
        _make_comp("U1", 10, 10, 20, 20),
        _make_comp("BAD", 5, 5, 5, 5),  # zero-width
    ]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        draw_layout(comps, ((0, 0), (50, 50)), tmp)
        assert os.path.getsize(tmp) > 100
    finally:
        os.unlink(tmp)


def test_plot_reward_curve_normal():
    """正常 reward 数据（含正负值）→ 生成 PNG 文件。"""
    from utils.visualization import plot_reward_curve
    # 300 条数据，覆盖正负 reward，模拟训练中 reward 逐步上升
    history = [(i, (i - 150) + 10 * (i % 3)) for i in range(300)]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        plot_reward_curve(history, tmp, pcb_name="test_board", window=100)
        assert os.path.getsize(tmp) > 100, "PNG file should not be empty"
    finally:
        os.unlink(tmp)


def test_plot_reward_curve_insufficient_data():
    """数据不足 window → 不生成文件。"""
    from utils.visualization import plot_reward_curve
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    # 先删除临时文件：函数应因数据不足直接 return，不重建文件
    os.unlink(tmp)
    history = [(i, float(i)) for i in range(50)]  # 50 < window=100
    plot_reward_curve(history, tmp, window=100)
    assert not os.path.exists(tmp), "数据不足 window 时不应生成文件"


def test_plot_reward_curve_exact_window_and_plus_one():
    """边界：恰好 window 条与 window+1 条 → 均应正常生成 PNG。"""
    from utils.visualization import plot_reward_curve
    for n in (100, 101):
        history = [(i, float(i) * (1 if i % 2 == 0 else -1)) for i in range(n)]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            plot_reward_curve(history, tmp, window=100)
            assert os.path.getsize(tmp) > 100, f"{n} 条应生成非空 PNG"
        finally:
            os.unlink(tmp)


def _capture_ax(components, board, **kwargs):
    """调用 draw_layout 并捕获内部创建的 axes，便于断言刻度/网格。"""
    from utils.visualization import draw_layout

    captured = {}
    real_subplots = plt.subplots

    def _spy_subplots(*args, **kwargs2):
        fig, ax = real_subplots(*args, **kwargs2)
        captured["ax"] = ax
        return fig, ax

    with mock.patch("matplotlib.pyplot.subplots", _spy_subplots):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            draw_layout(components, board, tmp, **kwargs)
        finally:
            os.unlink(tmp)
    return captured["ax"]


def _grid_line_positions(ax):
    """从 ax.lines 提取网格线坐标（比 get_ticklocs 更稳健，直接反映作图精度）。

    按线宽区分主次网格（次=0.15 / 主=0.4），返回
    (minor_xs, minor_ys, major_xs, major_ys) 四个排序集合。
    """
    minor_xs, minor_ys, major_xs, major_ys = [], [], [], []
    for ln in ax.lines:
        xd = ln.get_xdata()
        yd = ln.get_ydata()
        lw = round(float(ln.get_linewidth()), 3)
        is_major = abs(lw - 0.4) < 1e-6
        if len(xd) >= 2 and abs(float(xd[0]) - float(xd[-1])) < 1e-12:
            (major_xs if is_major else minor_xs).append(round(float(xd[0]), 6))
        if len(yd) >= 2 and abs(float(yd[0]) - float(yd[-1])) < 1e-12:
            (major_ys if is_major else minor_ys).append(round(float(yd[0]), 6))
    return (sorted(set(minor_xs)), sorted(set(minor_ys)),
            sorted(set(major_xs)), sorted(set(major_ys)))


def _assert_minor_precision(minor_positions, cell):
    """次网格最小线距须 == CELL_SIZE（0.2mm 作图精度）。"""
    diffs = [minor_positions[i + 1] - minor_positions[i]
             for i in range(len(minor_positions) - 1)]
    min_diff = min(diffs)
    assert abs(min_diff - cell) < 1e-6, (
        f"次网格最小线距应为 CELL_SIZE={cell}, 实际 {min_diff}"
    )


def _assert_major_subset_of_minor(minor_positions, major_positions):
    """主网格(1mm)线须全部落在次网格(0.2mm)线上（刻度-网格对齐）。"""
    minor_set = set(minor_positions)
    for m in major_positions:
        assert any(abs(m - p) < 1e-6 for p in minor_set), (
            f"主网格线 {m} 应落在次网格线上"
        )


@pytest.mark.parametrize("board", [
    ((0.0, 0.0), (100.0, 80.0)),   # 大板
    ((0.0, 0.0), (10.0, 10.0)),    # 标准板
    ((0.0, 0.0), (0.5, 0.5)),      # 极小板（防 1mm 兜底导致刻度稀疏）
])
def test_draw_layout_grid_and_ticks_aligned_to_cell(board):
    """网格精度须对齐 CELL_SIZE（X/Y 双轴），且网格范围覆盖 margin 区。

    - 次网格最小线距 == CELL_SIZE（作图精度 0.2mm）
    - 主网格(1mm)线全部落在次网格(0.2mm)线上（对齐）
    - 网格外扩到 margin，使坐标轴刻度留白处也有网格线（消除「有刻度无线」）
    """
    cell = Config.CELL_SIZE
    ax = _capture_ax([], board)
    minor_xs, minor_ys, major_xs, major_ys = _grid_line_positions(ax)
    assert minor_xs and minor_ys, "应存在纵横次网格线"

    # 次网格精度 == CELL_SIZE（X/Y 双轴）
    _assert_minor_precision(minor_xs, cell)
    _assert_minor_precision(minor_ys, cell)
    # 主网格(1mm)须落在次网格(0.2mm)上（X/Y 双轴）
    _assert_major_subset_of_minor(minor_xs, major_xs)
    _assert_major_subset_of_minor(minor_ys, major_ys)

    # 网格范围须覆盖 margin 区
    W = board[1][0] - board[0][0]
    H = board[1][1] - board[0][1]
    margin = max(W, H) * 0.10
    assert minor_xs[0] <= board[0][0] - margin + 1e-6, "网格左界应外扩到 margin"
    assert minor_xs[-1] >= board[1][0] + margin - 1e-6, "网格右界应外扩到 margin"
    assert minor_ys[0] <= board[0][1] - margin + 1e-6, "网格下界应外扩到 margin"
    assert minor_ys[-1] >= board[1][1] + margin - 1e-6, "网格上界应外扩到 margin"


def test_draw_layout_show_grid_false_no_lines():
    """show_grid=False 时不应绘制任何网格线。"""
    ax = _capture_ax([], ((0, 0), (100, 80)), show_grid=False)
    assert len(ax.lines) == 0, "show_grid=False 时不应有网格线"


def test_draw_layout_cell_size_override(monkeypatch):
    """CELL_SIZE 改变时，次网格线距须跟随（确保 cell 是单一来源）。"""
    new_cell = 0.5
    monkeypatch.setattr(Config, "CELL_SIZE", new_cell)
    ax = _capture_ax([], ((0.0, 0.0), (10.0, 10.0)))
    minor_xs, minor_ys, _, _ = _grid_line_positions(ax)
    assert minor_xs and minor_ys, "应存在纵横次网格线"
    _assert_minor_precision(minor_xs, new_cell)
    _assert_minor_precision(minor_ys, new_cell)
