"""真实 KiCad 引脚旋转同步集成测试。

验证几何修复第 2 项：引脚坐标随旋转更新。
核心不变量：
- _cache_components 预计算的 pin_offsets_by_rotation[r] 与 KiCad 真实几何一致
  （footprint 旋转到 r 后 pad 世界坐标 - hull bbox 中心）。
- dims_at_rotation 的宽高交换与真实 AABB 一致。
- pin_abs_positions(r) 绝对坐标与 KiCad 世界坐标一致（footprint 置于 comp 中心后）。
"""

import pytest
from utils.kicad_utils.kicad_utils import KiCadUtils


@pytest.mark.require_kicad
class TestPinRotation:
    """引脚偏移预计算 / 绝对坐标 / 旋转交换一致性。"""

    def _verify_offsets_match_kicad(self, ki, comp):
        """逐一旋转 footprint，比对预计算偏移与 KiCad 真实 pad 相对 hull 中心偏移。"""
        import pcbnew  # noqa: E402 — 由 conftest 保证可用

        fp = ki.footprints[comp.id]
        _saved_rot = fp.GetOrientationDegrees()
        _saved_pos = fp.GetPosition()
        try:
            for r in range(4):
                fp.SetOrientationDegrees(float(r * 90))
                h = fp.GetBoundingHull().BBox()
                hcx = (h.GetLeft() + h.GetRight()) / 2.0
                hcy = (h.GetTop() + h.GetBottom()) / 2.0
                expected = [
                    (pcbnew.ToMM(p.GetPosition().x - hcx),
                     pcbnew.ToMM(p.GetPosition().y - hcy))
                    for p in fp.Pads()
                ]
                actual = comp.pin_offsets_by_rotation.get(r, [])
                assert len(actual) == len(expected), f"rot={r}: 偏移数 != 引脚数"
                for (a_ox, a_oy), (e_ox, e_oy) in zip(actual, expected):
                    assert abs(a_ox - e_ox) < 1e-6, f"rot={r}: 偏移 x 不一致"
                    assert abs(a_oy - e_oy) < 1e-6, f"rot={r}: 偏移 y 不一致"
        finally:
            fp.SetOrientationDegrees(_saved_rot)
            fp.SetPosition(_saved_pos)

    def test_pin_offsets_by_rotation_match_kicad(self, kicad_pcb_path):
        """全部有引脚元件的 4 旋转偏移与 KiCad 真实几何一致。"""
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        tested = 0
        for ref, comp in ki.components.items():
            if not comp.pins or not comp.pin_offsets_by_rotation:
                continue
            self._verify_offsets_match_kicad(ki, comp)
            tested += 1
        assert tested >= 1, "没有找到带引脚的元件进行校验"

    def test_pin_abs_positions_matches_kicad_world(self, kicad_pcb_path):
        """footprint 置于 comp 中心后，KiCad pad 世界坐标 == pin_abs_positions(r)。"""
        import pcbnew  # noqa: E402 — 由 conftest 保证可用

        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        comp = next((c for c in ki.components.values()
                     if c.pins and not c.is_fixed), None)
        if comp is None:
            pytest.skip("No movable component with pins")
        fp = ki.footprints[comp.id]
        _saved_rot = fp.GetOrientationDegrees()
        _saved_pos = fp.GetPosition()
        try:
            for r in range(4):
                fp.SetOrientationDegrees(float(r * 90))
                h = fp.GetBoundingHull().BBox()
                hcx = (h.GetLeft() + h.GetRight()) / 2.0
                hcy = (h.GetTop() + h.GetBottom()) / 2.0
                pos = fp.GetPosition()
                # 平移 footprint，使 hull 中心对齐 comp.center（与 export 同法反推参考点）
                fp.SetPosition(pcbnew.VECTOR2I(
                    pcbnew.FromMM(comp.center[0]) - round(hcx - pos.x),
                    pcbnew.FromMM(comp.center[1]) - round(hcy - pos.y),
                ))
                env_pins = comp.pin_abs_positions(r)
                assert len(env_pins) == len(comp.pins), f"rot={r}: 引脚数不一致"
                for (num, ax, ay), pad in zip(env_pins, fp.Pads()):
                    p = pad.GetPosition()
                    assert abs(ax - pcbnew.ToMM(p.x)) < 1e-6, \
                        f"rot={r}: 引脚 {num} x 不一致"
                    assert abs(ay - pcbnew.ToMM(p.y)) < 1e-6, \
                        f"rot={r}: 引脚 {num} y 不一致"
        finally:
            fp.SetOrientationDegrees(_saved_rot)
            fp.SetPosition(_saved_pos)

    def test_dims_at_rotation_swap_matches_kicad(self, kicad_pcb_path):
        """dims_at_rotation 的宽高交换与真实 KiCad AABB 交换一致。"""
        import pcbnew  # noqa: E402 — 由 conftest 保证可用

        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        comp = next((c for c in ki.components.values() if c.pins), None)
        if comp is None:
            pytest.skip("No components with pins")
        fp = ki.footprints[comp.id]
        _saved_rot = fp.GetOrientationDegrees()
        _saved_pos = fp.GetPosition()
        try:
            for r in range(4):
                fp.SetOrientationDegrees(float(r * 90))
                h = fp.GetBoundingHull().BBox()
                kicad_w = pcbnew.ToMM(h.GetWidth())
                kicad_h = pcbnew.ToMM(h.GetHeight())
                env_w, env_h = comp.dims_at_rotation(r)
                assert abs(env_w - kicad_w) < 0.01, f"rot={r}: 宽度不一致"
                assert abs(env_h - kicad_h) < 0.01, f"rot={r}: 高度不一致"
        finally:
            fp.SetOrientationDegrees(_saved_rot)
            fp.SetPosition(_saved_pos)
