"""可视化模块单元测试。

验证 draw_layout 对不同板框尺寸、元件数不崩溃。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import pytest

# matplotlib 在 test 环境中必须是 Agg backend
import matplotlib
matplotlib.use("Agg")


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
