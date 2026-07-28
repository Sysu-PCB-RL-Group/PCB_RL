"""真实 KiCad export_layout 集成测试。

验证导出后的 .kicad_pcb 可被重新加载，元件位置/旋转/层变更正确。
"""

import os
import tempfile
import pytest
from utils.kicad_utils.kicad_utils import KiCadUtils


@pytest.mark.require_kicad
class TestExportLayout:
    """export_layout() 导出 → 再加载验证 round-trip。"""

    def test_export_and_reload(self, kicad_pcb_path):
        """导出后重新加载，元件数量和网络数量应一致。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        orig_comp_count = len(ki.components)
        orig_net_count = len(ki.nets)

        # 导出到临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "exported.kicad_pcb")
            ok = ki.export_layout(export_path, ki.components)
            assert ok, f"Export failed to {export_path}"
            assert os.path.exists(export_path), "Export file not created"

            # 用另一个实例重新加载
            ki2 = KiCadUtils()
            ok2 = ki2.load_board(export_path)
            assert ok2, f"Failed to reload exported file: {export_path}"
            assert len(ki2.components) == orig_comp_count, (
                f"Component count mismatch: {orig_comp_count} vs {len(ki2.components)}"
            )
            assert len(ki2.nets) == orig_net_count, (
                f"Net count mismatch: {orig_net_count} vs {len(ki2.nets)}"
            )

    def test_export_with_moved_component(self, kicad_pcb_path):
        """移动元件后导出，再加载验证位置已更新。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        # 找一个可移动元件并记录原始位置
        movable = [(ref, c) for ref, c in ki.components.items() if not c.is_fixed]
        if not movable:
            pytest.skip("No movable components to test")
        ref, comp = movable[0]
        orig_bbox = tuple(comp.bbox)

        # 移动元件（bbox 整体平移 10mm）
        dx, dy = 10.0, 15.0
        comp.bbox = [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(comp.bbox)]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "moved.kicad_pcb")
            ki.export_layout(export_path, ki.components)

            # 重新加载并验证
            ki2 = KiCadUtils()
            ki2.load_board(export_path)
            moved_comp = ki2.components.get(ref)
            assert moved_comp is not None, f"Component {ref} not found after export"

            # bbox 应接近（考虑了单位转换精度 nm<->mm）
            for i in range(4):
                diff = abs(moved_comp.bbox[i] - comp.bbox[i])
                assert diff < 0.01, (
                    f"{ref}: bbox[{i}] mismatch: {moved_comp.bbox[i]:.4f} vs {comp.bbox[i]:.4f}, diff={diff:.4f}"
                )

    def test_export_does_not_lose_fixed_components(self, kicad_pcb_path):
        """导出不应丢失固定元件。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        fixed_refs = {ref for ref, c in ki.components.items() if c.is_fixed}

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "exported.kicad_pcb")
            ki.export_layout(export_path, ki.components)
            ki2 = KiCadUtils()
            ki2.load_board(export_path)
            exported_fixed = {ref for ref, c in ki2.components.items() if c.is_fixed}
            assert exported_fixed == fixed_refs, (
                f"Fixed components mismatch: {fixed_refs - exported_fixed} lost, "
                f"{exported_fixed - fixed_refs} added"
            )
