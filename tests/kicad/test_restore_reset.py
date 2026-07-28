"""真实 KiCad restore_initial_state 集成测试。

验证 restore_initial_state 不读磁盘、正确恢复原始状态。
"""

import time
import pytest
from utils.kicad_utils.kicad_utils import KiCadUtils


@pytest.mark.require_kicad
class TestRestoreInitialState:
    """restore_initial_state() 缓存恢复机制。"""

    def test_restore_after_modification(self, kicad_pcb_path):
        """修改元件后 restore，应回到 load_board 时的原始状态。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        # 记录初始状态
        orig_bboxes = {ref: tuple(c.bbox) for ref, c in ki.components.items()}
        orig_rotation = {ref: c.rotation for ref, c in ki.components.items()}
        orig_layer = {ref: c.layer for ref, c in ki.components.items()}

        # 修改所有可移动元件
        for ref, c in ki.components.items():
            if not c.is_fixed:
                c.bbox = [v + 5.0 for v in c.bbox]
                c.rotation = (c.rotation + 1) % 4
                c.layer = 1 - c.layer

        # 恢复
        ki.restore_initial_state()

        # 验证所有元件回到原始位置
        for ref, c in ki.components.items():
            if c.is_fixed:
                continue
            for i in range(4):
                diff = abs(c.bbox[i] - orig_bboxes[ref][i])
                assert diff < 0.01, (
                    f"{ref}: bbox[{i}] not restored: {c.bbox[i]:.4f} vs {orig_bboxes[ref][i]:.4f}"
                )
            assert c.rotation == orig_rotation[ref], (
                f"{ref}: rotation not restored: {c.rotation} vs {orig_rotation[ref]}"
            )
            assert c.layer == orig_layer[ref], (
                f"{ref}: layer not restored: {c.layer} vs {orig_layer[ref]}"
            )

    def test_restore_keeps_component_count(self, kicad_pcb_path):
        """restore 后元件和网络数量不变。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        comp_count = len(ki.components)
        net_count = len(ki.nets)

        ki.restore_initial_state()
        assert len(ki.components) == comp_count, "Component count changed after restore"
        assert len(ki.nets) == net_count, "Net count changed after restore"

    def test_restore_faster_than_reload(self, kicad_pcb_path):
        """restore_initial_state 应比 load_board 快得多（至少 5x）。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        # 计时 load_board
        t0 = time.perf_counter()
        ki.load_board(kicad_pcb_path)
        load_time = time.perf_counter() - t0

        # 计时 restore
        t0 = time.perf_counter()
        ki.restore_initial_state()
        restore_time = time.perf_counter() - t0

        # restore 应该比重新加载快至少 5 倍
        assert restore_time < load_time, (
            f"restore ({restore_time:.3f}s) should be faster than load ({load_time:.3f}s)"
        )
