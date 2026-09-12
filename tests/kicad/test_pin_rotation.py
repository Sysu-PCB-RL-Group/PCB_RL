"""真实 KiCad 引脚姿态（2 层 x 4 角 = 8 个）集成测试。

核心不变量：
- _cache_components 预计算的 pin_offsets_by_pose[(layer, rotation)] 与 KiCad 真实
  几何一致（footprint 切到该面、转到该角后 pad 世界坐标 - hull bbox 中心）。
- pin_abs_positions(rotation, layer) 的绝对坐标与 KiCad 世界坐标一致。
- dims_at_rotation 的宽高交换与真实 AABB 一致。
- env/mock_pcb 的解析式 8 姿态表与真实 KiCad 一致（合成非对称 3 pad 板上比对）。

为什么必须带 layer：面镜像不是"同角度下左右镜像"，而是在【局部未旋转坐标系】
里镜像后再旋转。合成非对称 3 pad 实测：TOP rot=0 (-2.00,-1.50) -> BOT rot=0
(-2.00,+1.50)（Y 变号），TOP rot=90 (-1.50,+2.00) -> BOT rot=90 (+1.50,+2.00)
（X 变号）。旧的 4 姿态表只测源面，元件移到对面后底面引脚全错（本样本错 3mm）。

切面一律用 Flip，绝不用 SetLayer —— 后者只改父层，不动 pad 子层与几何，
量出来的偏移与正面完全一样（等于没切面）。
"""

import pytest

from utils.kicad_utils.kicad_utils import KiCadUtils

pcbnew = pytest.importorskip("pcbnew")

TOL_EXACT = 1e-6      # 同一条测量路径的前后比对
TOL_MM = 0.01         # 与解析式/环境的比对
POSES = [(lay, r) for lay in (0, 1) for r in range(4)]

# 非对称 3 pad：L 形，任何镜像/旋转都能区分（2 脚电阻镜像前后一样，不能用）
PAD_LOCAL = [("1", 0.0, 0.0), ("2", 4.0, 0.0), ("3", 0.0, 3.0)]


def _set_pose(fp, layer, r):
    """把 footprint 摆到 (layer, r)：先切面（Flip），再写【绝对】角度。

    Flip 会改写角度，所以顺序不能反，且必须写绝对值而不是累加。
    """
    if bool(fp.IsFlipped()) != (layer == 1):
        fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
    fp.SetOrientationDegrees(float(r * 90))


def _hull_center_nm(fp):
    h = fp.GetBoundingHull().BBox()
    return ((h.GetLeft() + h.GetRight()) / 2.0,
            (h.GetTop() + h.GetBottom()) / 2.0)


def _measure(fp):
    """当前姿态下 pad 相对 hull bbox 中心的偏移 (mm)，顺序同 fp.Pads()。"""
    hcx, hcy = _hull_center_nm(fp)
    return [(pcbnew.ToMM(p.GetPosition().x - hcx),
             pcbnew.ToMM(p.GetPosition().y - hcy)) for p in fp.Pads()]


def build_asym_board(path):
    """合成一块只含一个非对称 3 pad 封装的板并落盘，返回路径。"""
    board = pcbnew.BOARD()
    fp = pcbnew.FOOTPRINT(board)
    fp.SetFPID(pcbnew.LIB_ID("Test", "Asym3"))
    fp.SetReference("U1")
    fp.SetValue("Asym3")
    base = pcbnew.VECTOR2I(pcbnew.FromMM(50.0), pcbnew.FromMM(50.0))
    fp.SetPosition(base)
    for num, dx, dy in PAD_LOCAL:
        pad = pcbnew.PAD(fp)
        pad.SetNumber(num)
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(1.0), pcbnew.FromMM(1.0)))
        off = pcbnew.VECTOR2I(pcbnew.FromMM(dx), pcbnew.FromMM(dy))
        pad.SetFPRelativePosition(off)          # 相对 footprint 的局部坐标
        pad.SetPosition(base + off)             # 世界坐标
        pad.SetLayerSet(pad.SMDMask())
        fp.Add(pad)
    board.Add(fp)
    pcbnew.SaveBoard(str(path), board)
    return str(path)


@pytest.fixture
def asym_comp(tmp_path):
    """合成板 + 真实 KiCadUtils 解析结果 + 8 姿态实测值。"""
    path = build_asym_board(tmp_path / "Asym3.kicad_pcb")
    ki = KiCadUtils()
    assert ki.load_board(path) is True
    comp = next(iter(ki.components.values()))
    board = pcbnew.LoadBoard(path)
    fp = next(iter(board.Footprints()))
    measured = {}
    for lay, r in POSES:
        _set_pose(fp, lay, r)
        measured[(lay, r)] = _measure(fp)
    return {"path": path, "ki": ki, "comp": comp, "fp": fp, "measured": measured}


@pytest.mark.require_kicad
class TestPinPose:

    # ---------------------------------------------------------- 合成非对称板

    def test_all_8_poses_measured_and_distinct(self, asym_comp):
        """8 个姿态全部测到，且几何各不相同（样本确实非对称）。"""
        measured = asym_comp["measured"]
        assert len(measured) == 8
        assert all(len(v) == 3 for v in measured.values())
        assert len({tuple(v) for v in measured.values()}) == 8

    def test_front_and_back_differ_at_every_angle(self, asym_comp):
        """同角度下正面与背面必须不同 —— 旧 4 姿态表就是在这里错的。"""
        measured = asym_comp["measured"]
        for r in range(4):
            front, back = measured[(0, r)], measured[(1, r)]
            assert front != back, f"rot={r * 90}°：正/背面几何相同，样本选错了"

    def test_offsets_by_pose_match_kicad(self, asym_comp):
        """预计算的 8 姿态偏移与真实 KiCad 实测一致。"""
        comp = asym_comp["comp"]
        assert len(comp.pin_offsets_by_pose) == 8, \
            f"表里只有 {len(comp.pin_offsets_by_pose)} 个姿态"
        for pose, expected in asym_comp["measured"].items():
            actual = comp.pin_offsets_by_pose.get(pose, [])
            assert len(actual) == len(expected), f"pose {pose}: 偏移数 != 引脚数"
            for (a_ox, a_oy), (e_ox, e_oy) in zip(actual, expected):
                assert abs(a_ox - e_ox) < TOL_EXACT, f"pose {pose}: 偏移 x 不一致"
                assert abs(a_oy - e_oy) < TOL_EXACT, f"pose {pose}: 偏移 y 不一致"

    def test_precompute_restores_source_pose(self, tmp_path):
        """预计算跑完必须完整恢复源姿态：面、角度、pad 子层都不能被污染。

        回归点：恢复若用 SetLayer 而不是 Flip，pad 子层会残留在被翻的一面
        （SetLayer 只改父层）。这里直接查子层，不查 IsFlipped —— 后者只看父层。
        """
        path = build_asym_board(tmp_path / "src.kicad_pcb")
        ki = KiCadUtils()
        assert ki.load_board(path) is True
        comp = next(iter(ki.components.values()))
        fp = ki.footprints[comp.id]

        assert bool(fp.IsFlipped()) is False, "源在 F.Cu，预计算后应仍在正面"
        assert fp.GetOrientationDegrees() % 360 == 0.0
        names = {pcbnew.LayerName(l) for p in fp.Pads()
                 for l in p.GetLayerSet().Seq()}
        assert not any(n.startswith("B.") for n in names), \
            f"pad 子层被翻面污染: {sorted(names)}"

    def test_pin_abs_positions_matches_kicad_world(self, asym_comp):
        """footprint 置于 comp 中心后，KiCad pad 世界坐标 == pin_abs_positions(r, layer)。"""
        comp, fp = asym_comp["comp"], asym_comp["fp"]
        measured = asym_comp["measured"]
        for lay, r in POSES:
            _set_pose(fp, lay, r)
            hcx, hcy = _hull_center_nm(fp)
            pos = fp.GetPosition()
            # 平移 footprint，使 hull 中心对齐 comp.center（与 export 同法反推参考点）
            fp.SetPosition(pcbnew.VECTOR2I(
                pcbnew.FromMM(comp.center[0]) - round(hcx - pos.x),
                pcbnew.FromMM(comp.center[1]) - round(hcy - pos.y),
            ))
            env_pins = comp.pin_abs_positions(r, layer=lay)
            assert len(env_pins) == len(comp.pins), f"pose ({lay},{r}): 引脚数不一致"
            for (num, ax, ay), pad in zip(env_pins, fp.Pads()):
                p = pad.GetPosition()
                assert abs(ax - pcbnew.ToMM(p.x)) < TOL_MM, \
                    f"pose ({lay},{r}): 引脚 {num} x 不一致"
                assert abs(ay - pcbnew.ToMM(p.y)) < TOL_MM, \
                    f"pose ({lay},{r}): 引脚 {num} y 不一致"

    def test_mock_analytic_table_matches_kicad(self, asym_comp):
        """env/mock_pcb 的解析式 8 姿态表必须与真实 KiCad 一致。

        mock 是训练侧（无 KiCad）的唯一几何来源，它若与真实不一致，
        mock 环境下训出来的策略迁到真机就是错的。
        """
        from env.mock_pcb import MockComponent

        comp = asym_comp["comp"]
        src_bbox = comp.source_bbox
        mock = MockComponent(
            id=comp.id, footprint=comp.footprint, package_type=comp.package_type,
            bbox=src_bbox, rotation=comp.rotation, layer=comp.layer,
            pins=list(comp.pins),                     # 相对源 bbox 左上角
            footprint_w=comp.footprint_w, footprint_h=comp.footprint_h,
            source_rotation=comp.source_rotation, source_bbox=src_bbox,
        )
        table = mock._build_pin_offsets_by_pose()
        assert len(table) == 8
        for pose, expected in asym_comp["measured"].items():
            actual = table.get(pose, [])
            assert len(actual) == len(expected), f"pose {pose}: 偏移数 != 引脚数"
            for (a_ox, a_oy), (e_ox, e_oy) in zip(actual, expected):
                assert abs(a_ox - e_ox) < TOL_MM, f"pose {pose}: mock 偏移 x 不符"
                assert abs(a_oy - e_oy) < TOL_MM, f"pose {pose}: mock 偏移 y 不符"

    # ---------------------------------------------------------- 真实数据板

    def test_real_board_all_8_poses_match_kicad(self, kicad_pcb_path):
        """真实板上每个带引脚元件的 8 姿态偏移都必须与 KiCad 一致。"""
        ki = KiCadUtils()
        assert ki.load_board(kicad_pcb_path) is True
        tested = 0
        for ref, comp in ki.components.items():
            if not comp.pins or not comp.pin_offsets_by_pose:
                continue
            fp = ki.footprints[ref]
            # 预计算循环会就地 Flip 真实 footprint，这里先钉住它确实被还原了：
            # 加载后的姿态必须就是元件记录的源姿态（否则后面测的是被污染过的表）
            assert bool(fp.IsFlipped()) == (comp.layer == 1), \
                f"{ref}: 预计算后 footprint 所在面与 Component.layer 不一致"
            assert int(round(fp.GetOrientationDegrees() / 90.0)) % 4 == comp.rotation, \
                f"{ref}: 预计算后角度与 Component.rotation 不一致"
            _saved_flipped = bool(fp.IsFlipped())
            _saved_rot = fp.GetOrientationDegrees()
            _saved_pos = fp.GetPosition()
            try:
                for lay, r in POSES:
                    _set_pose(fp, lay, r)
                    expected = _measure(fp)
                    actual = comp.pin_offsets_by_pose.get((lay, r), [])
                    assert len(actual) == len(expected), \
                        f"{ref} pose ({lay},{r}): 偏移数 != 引脚数"
                    for (a_ox, a_oy), (e_ox, e_oy) in zip(actual, expected):
                        assert abs(a_ox - e_ox) < TOL_EXACT, \
                            f"{ref} pose ({lay},{r}): 偏移 x 不一致"
                        assert abs(a_oy - e_oy) < TOL_EXACT, \
                            f"{ref} pose ({lay},{r}): 偏移 y 不一致"
            finally:
                if bool(fp.IsFlipped()) != _saved_flipped:
                    fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_TOP_BOTTOM)
                fp.SetOrientationDegrees(_saved_rot)
                fp.SetPosition(_saved_pos)
            tested += 1
        assert tested >= 1, "没有找到带引脚的元件进行校验"

    def test_dims_at_rotation_swap_matches_kicad(self, kicad_pcb_path):
        """dims_at_rotation 的宽高交换与真实 KiCad AABB 交换一致。"""
        ki = KiCadUtils()
        assert ki.load_board(kicad_pcb_path) is True
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
                assert abs(env_w - kicad_w) < TOL_MM, f"rot={r}: 宽度不一致"
                assert abs(env_h - kicad_h) < TOL_MM, f"rot={r}: 高度不一致"
        finally:
            fp.SetOrientationDegrees(_saved_rot)
            fp.SetPosition(_saved_pos)
