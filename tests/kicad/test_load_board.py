"""真实 KiCad load_board 集成测试。

验证 pcbnew API 加载流程、元件/网络计数、板框边界、None 检查等。
"""

import pytest
from utils.kicad_utils.kicad_utils import KiCadUtils


@pytest.mark.require_kicad
class TestLoadBoard:
    """load_board() 核心加载路径。"""

    def test_load_real_pcb(self, kicad_pcb_path):
        """加载真实 .kicad_pcb 并验证核心数据完整性。"""
        ki = KiCadUtils()
        ok = ki.load_board(kicad_pcb_path)
        assert ok, f"Failed to load: {kicad_pcb_path}"

        # 核心数据不能为空
        assert len(ki.components) > 0, "No components loaded"
        assert len(ki.nets) > 0, "No nets loaded"
        assert ki.board is not None, "Board object should not be None"

        # 板框边界为有效值
        (min_x, min_y), (max_x, max_y) = ki.board_bounds
        assert max_x > min_x, f"Invalid board bounds X: {min_x} ~ {max_x}"
        assert max_y > min_y, f"Invalid board bounds Y: {min_y} ~ {max_y}"

    def test_load_board_none_on_missing_file(self):
        """不存在的文件应返回 False，不应崩溃。"""
        ki = KiCadUtils()
        ok = ki.load_board("nonexistent_file.kicad_pcb")
        assert not ok, "Should return False for nonexistent file"

    def test_fixed_vs_movable_components(self, kicad_pcb_path):
        """固定元件和可移动元件应正确区分。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        fixed = [c for c in ki.components.values() if c.is_fixed]
        movable = [c for c in ki.components.values() if not c.is_fixed]
        assert len(fixed) + len(movable) == len(ki.components)
        # 真实板子通常有至少 1 个可移动元件
        assert len(movable) > 0, "Expected at least one movable component"

    def test_component_has_valid_bbox(self, kicad_pcb_path):
        """每个元件应有有效的 bbox [x1, y1, x2, y2]，均为数值。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        for ref, comp in ki.components.items():
            assert len(comp.bbox) == 4, f"{ref}: bbox should have 4 values, got {len(comp.bbox)}"
            assert all(isinstance(v, (int, float)) for v in comp.bbox), (
                f"{ref}: all bbox values should be numeric"
            )
            x1, y1, x2, y2 = comp.bbox
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            assert width > 0, f"{ref}: bbox width should be > 0, got {width}"
            assert height > 0, f"{ref}: bbox height should be > 0, got {height}"

    def test_reload_is_idempotent(self, kicad_pcb_path):
        """连续两次 load_board 结果应一致。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        count1 = len(ki.components)
        net_count1 = len(ki.nets)
        bounds1 = ki.board_bounds

        ki.load_board(kicad_pcb_path)
        count2 = len(ki.components)
        net_count2 = len(ki.nets)
        bounds2 = ki.board_bounds

        assert count1 == count2, f"Component count changed: {count1} -> {count2}"
        assert net_count1 == net_count2, f"Net count changed: {net_count1} -> {net_count2}"
        assert bounds1 == bounds2, f"Board bounds changed: {bounds1} -> {bounds2}"
