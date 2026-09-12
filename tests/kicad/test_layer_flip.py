"""跨面导出（Top <-> Bottom）子层一致性测试。

回归的 P0 问题
--------------
旧 export_layout 用 FOOTPRINT.SetLayer() 换面，只改 footprint 父对象层：
    footprint layer = B.Cu
    pad layers      = F.Cu, F.Paste, F.Mask     ← 仍在顶面
    graphics        = F.SilkS, F.Fab, F.CrtYd   ← 仍在顶面
且 IsFlipped() 会返回 True（它只看父层），所以只查父层/IsFlipped 查不出来。

正确做法是 FOOTPRINT.Flip()：它同时搬走父层、pad 子层、丝印/Fab/CrtYd，
并完成内部几何与焊盘坐标的镜像。

断言策略
--------
必须检查【子层】而不是只查 GetLayer()/IsFlipped()（见
test_setlayer_only_leaves_sublayers_on_front 的负向对照）。
用层 ID 断言而非层名：KiCad 10 中 board.GetLayerName() 对文件加载的板返回
用户可见名（"Top Layer"/"Bottom Silkscreen Layer"），不是规范名 "F.Cu"。

注意：非对称封装（pad 数 >= 3）是必需的，2 脚电阻镜像前后看起来一样。
"""

import os
import tempfile

import pytest

from utils.kicad_utils.kicad_utils import KiCadUtils

pcbnew = pytest.importorskip("pcbnew")


# 已知会让 pcbnew.LoadBoard 稳定挂死的板（KiCad 10.0.4 实测 >90s 无返回，
# 其它板 0.01~0.10s）。与换面逻辑无关，属独立问题；此处跳过以免测试卡死。
HANGING_BOARD_KEYWORDS = ("Button-cbd1bfb1",)

TOL_MM = 0.01


def _movable_asymmetric(ki):
    """返回 (ref, comp)：可动、有网络、pad 数 >= 3（非对称）、且【源在 F.Cu】。

    要求源在顶面有两个原因：
      1. 测试要把元件「移到」底面，源在顶面才是真实场景；
      2. 负向对照 test_setlayer_only_leaves_sublayers_on_front 里的
         assert not fp.IsFlipped() 需要样本确实在顶面，否则是靠样本碰巧成立。
    """
    best = None
    for ref, comp in ki.components.items():
        if comp.is_fixed or not comp.nets or len(comp.pins) < 3:
            continue
        if comp.layer != 0:          # 源必须已在 F.Cu
            continue
        if best is None or len(comp.pins) > len(best[1].pins):
            best = (ref, comp)
    return best


@pytest.fixture(scope="module")
def asym_setup():
    """找一块「能加载且含非对称可动元件」的板。

    返回 (ki, ref, comp, orig)；orig 固化源姿态（bbox/rotation/layer），
    因为本 fixture 是 module 级共享的，comp 会被前序用例就地修改。
    按文件大小升序遍历，避开已知挂死的大板。
    """
    batch_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                             "data", "kicad-native-batch")
    cands = []
    for root, _, files in os.walk(batch_dir):
        for f in files:
            if f.endswith(".kicad_pcb"):
                cands.append(os.path.join(root, f))
    cands.sort(key=os.path.getsize)

    for path in cands:
        if any(k in path for k in HANGING_BOARD_KEYWORDS):
            continue
        ki = KiCadUtils()
        if not ki.load_board(path):
            continue
        picked = _movable_asymmetric(ki)
        if picked:
            ref, comp = picked
            orig = {"bbox": tuple(comp.bbox),
                    "rotation": comp.rotation,
                    "layer": comp.layer}
            comp.is_placed = True
            return ki, ref, comp, orig
    pytest.skip("无「源在 F.Cu 的非对称可动元件」的可加载测试板")


def _fp_of(board, ref):
    return next((f for f in board.Footprints() if f.GetReference() == ref), None)


def _export_and_find(ki, ref, tmpdir, name, template=None):
    out = os.path.join(tmpdir, name)
    assert ki.export_layout(out, ki.components, template_path=template), \
        f"export_layout 失败: {name}"
    board = pcbnew.LoadBoard(out)
    assert board is not None, f"导出文件无法加载: {out}"
    fp = _fp_of(board, ref)
    assert fp is not None, f"导出文件中找不到元件 {ref}"
    return board, fp


def _hull_center(fp):
    b = fp.GetBoundingHull().BBox()
    return (pcbnew.ToMM((b.GetLeft() + b.GetRight()) / 2.0),
            pcbnew.ToMM((b.GetTop() + b.GetBottom()) / 2.0))


def _snapshot(fp):
    """记录全部待比对量：父层 / pad 子层 / 图形层 / 角度 / hull 中心 / pad 世界坐标。"""
    return {
        "layer": fp.GetLayer(),
        "flipped": bool(fp.IsFlipped()),
        "angle": fp.GetOrientationDegrees() % 360,
        "hull": _hull_center(fp),
        "pad_layers": {p.GetNumber(): frozenset(p.GetLayerSet().Seq())
                       for p in fp.Pads()},
        "pads": {p.GetNumber(): (pcbnew.ToMM(p.GetPosition().x),
                                 pcbnew.ToMM(p.GetPosition().y))
                 for p in fp.Pads()},
        "graphic_layers": frozenset(it.GetLayer() for it in fp.GraphicalItems()),
    }


def _assert_pose(fp, comp):
    """断言 footprint 处于目标层/角度，且所有子层与父层一致。"""
    want_bottom = (comp.layer == 1)
    want_cu = pcbnew.B_Cu if want_bottom else pcbnew.F_Cu
    want_mask = pcbnew.B_Mask if want_bottom else pcbnew.F_Mask
    want_paste = pcbnew.B_Paste if want_bottom else pcbnew.F_Paste
    opp_cu = pcbnew.F_Cu if want_bottom else pcbnew.B_Cu

    assert fp.GetLayer() == want_cu
    assert bool(fp.IsFlipped()) == want_bottom

    for pad in fp.Pads():
        layers = set(pad.GetLayerSet().Seq())
        num = pad.GetNumber()
        assert want_cu in layers, f"pad {num} 不在目标铜层，实际 {sorted(layers)}"
        assert want_paste in layers, f"pad {num} 不在目标钢网层"
        assert want_mask in layers, f"pad {num} 不在目标阻焊层"
        assert opp_cu not in layers, f"pad {num} 残留在对面铜层"

    for item in fp.GraphicalItems():
        name = pcbnew.LayerName(item.GetLayer())
        side = "F." if want_bottom else "B."
        assert not name.startswith(side), f"图形残留在对面: {name}"

    got = fp.GetOrientationDegrees() % 360
    assert abs(got - (comp.rotation * 90) % 360) < 0.01


def test_to_bottom_moves_all_sublayers(asym_setup):
    """F.Cu → B.Cu：pad 子层、图形子层必须一起搬到背面。"""
    ki, ref, comp, orig = asym_setup
    comp.layer = 1
    comp.rotation = orig["rotation"]
    comp.bbox = list(orig["bbox"])

    with tempfile.TemporaryDirectory() as tmpdir:
        _, fp = _export_and_find(ki, ref, tmpdir, "to_bottom.kicad_pcb")
        _assert_pose(fp, comp)


def test_eight_poses_consistency(asym_setup):
    """4 角度 × 2 层 = 8 种姿态，逐一断言子层、角度与真实 hull 中心。"""
    ki, ref, comp, orig = asym_setup

    with tempfile.TemporaryDirectory() as tmpdir:
        for layer in (0, 1):
            for rot in range(4):
                comp.layer = layer
                comp.rotation = rot
                comp.bbox = list(orig["bbox"])
                _, fp = _export_and_find(
                    ki, ref, tmpdir, f"pose_{layer}_{rot}.kicad_pcb")
                _assert_pose(fp, comp)

                hx, hy = _hull_center(fp)
                tx = (comp.bbox[0] + comp.bbox[2]) / 2.0
                ty = (comp.bbox[1] + comp.bbox[3]) / 2.0
                assert abs(hx - tx) < TOL_MM and abs(hy - ty) < TOL_MM, (
                    f"layer={layer} rot={rot*90}°: hull 中心 ({hx:.4f},{hy:.4f}) "
                    f"偏离目标 bbox 中心 ({tx:.4f},{ty:.4f})"
                )


def test_flip_twice_restores_full_pose(asym_setup):
    """真两段导出 F→B→F，逐项比对是否精确复原。

    与「只导出一次并检查仍在顶面」不同：第二段导出的**模板是第一段导出的
    文件本身**，因此真正走了一遍 Flip 之后再 Flip 的路径，能验证
    Flip 的幂等性与 finally 里的面恢复逻辑。
    """
    ki, ref, comp, orig = asym_setup
    before = _snapshot(_fp_of(pcbnew.LoadBoard(ki._source_path), ref))

    with tempfile.TemporaryDirectory() as tmpdir:
        # ---- 第一段：F → B ----
        comp.layer = 1
        comp.rotation = orig["rotation"]
        comp.bbox = list(orig["bbox"])
        stage1 = os.path.join(tmpdir, "stage1_bottom.kicad_pcb")
        b1, fp1 = _export_and_find(ki, ref, tmpdir, "stage1_bottom.kicad_pcb")
        assert fp1.GetLayer() == pcbnew.B_Cu, "第一段未到 B.Cu"
        for pad in fp1.Pads():
            layers = set(pad.GetLayerSet().Seq())
            assert pcbnew.B_Cu in layers and pcbnew.F_Cu not in layers, (
                f"第一段 pad {pad.GetNumber()} 子层未到背面: {sorted(layers)}"
            )

        # ---- 第二段：以第一段的结果为模板 → F ----
        comp.layer = 0
        _, fp2 = _export_and_find(ki, ref, tmpdir, "stage2_top.kicad_pcb",
                                  template=stage1)
        after = _snapshot(fp2)

    assert after["layer"] == before["layer"], \
        f"父层未复原: {after['layer']} vs {before['layer']}"
    assert after["flipped"] == before["flipped"]
    assert abs(after["angle"] - before["angle"]) < 0.01, \
        f"角度未复原: {after['angle']} vs {before['angle']}"
    assert after["pad_layers"] == before["pad_layers"], "pad 子层未复原"
    assert after["graphic_layers"] == before["graphic_layers"], \
        f"图形层未复原: {sorted(after['graphic_layers'])} vs " \
        f"{sorted(before['graphic_layers'])}"
    assert set(after["pads"]) == set(before["pads"])

    for num, (x, y) in before["pads"].items():
        gx, gy = after["pads"][num]
        assert abs(gx - x) < TOL_MM and abs(gy - y) < TOL_MM, (
            f"pad {num} 坐标未复原: ({gx:.4f},{gy:.4f}) vs ({x:.4f},{y:.4f})"
        )

    assert abs(after["hull"][0] - before["hull"][0]) < TOL_MM
    assert abs(after["hull"][1] - before["hull"][1]) < TOL_MM


def test_setlayer_only_leaves_sublayers_on_front(asym_setup):
    """负向对照：钉住 SetLayer 的语义，说明为什么必须用 Flip。

    这是回归测试的「反向证据」——如果将来有人把 export 改回 SetLayer，
    上面三个用例会失败，而本用例会继续通过（SetLayer 语义未变）。
    """
    ki, ref, comp, orig = asym_setup
    board = pcbnew.LoadBoard(ki._source_path)
    fp = _fp_of(board, ref)
    assert fp is not None

    # fixture 已保证源在 F.Cu，这里不依赖样本碰巧
    assert fp.GetLayer() == pcbnew.F_Cu
    assert not fp.IsFlipped()

    fp.SetLayer(pcbnew.B_Cu)

    # 父层与 IsFlipped 看起来都对了 …
    assert fp.GetLayer() == pcbnew.B_Cu
    assert fp.IsFlipped()
    # … 但子层完全没动，这正是旧实现的症状
    for pad in fp.Pads():
        layers = set(pad.GetLayerSet().Seq())
        assert pcbnew.F_Cu in layers, (
            "SetLayer 竟然搬走了 pad 铜层；KiCad 语义变了，"
            "需重新评估 export_layout 是否还需要 Flip"
        )
        assert pcbnew.B_Cu not in layers


def _footprints_snapshot(ki):
    """缓存中每个 footprint 的完整几何指纹（层/面/角/位置/pad 坐标与子层）。"""
    out = {}
    for r, fp in ki.footprints.items():
        out[r] = (
            fp.GetLayer(),
            bool(fp.IsFlipped()),
            round(fp.GetOrientationDegrees() % 360, 6),
            (fp.GetPosition().x, fp.GetPosition().y),
            tuple(tuple(sorted(p.GetLayerSet().Seq())) for p in fp.Pads()),
            tuple((p.GetNumber(), p.GetPosition().x, p.GetPosition().y)
                  for p in fp.Pads()),
        )
    return out


def test_cache_components_restores_footprint_state(asym_setup):
    """_cache_components 的临时姿态操作必须完全复原（含【面】）。

    预计算引脚偏移时会临时改写 footprint 的旋转（第二步会再加上 Flip），
    finally 必须精确复原，否则 self.board / self.footprints 缓存被静默污染，
    restore_initial_state() 与后续导出会一起出错。

    恢复【不能】用 SetLayer：它只改父层，撤销不了 Flip 对 pad 子层与几何的
    镜像。本用例在循环内尚无 Flip 时也是有效的（钉住「缓存不得被改写」），
    循环内出现 Flip 后它会直接验证 Flip 复原分支。
    """
    ki, ref, comp, orig = asym_setup
    before = _footprints_snapshot(ki)

    ki.restore_initial_state()          # 公开的重置路径，内部调用 _cache_components
    after = _footprints_snapshot(ki)

    assert set(before) == set(after)

    for r in before:
        b, a = before[r], after[r]
        assert a[0] == b[0], f"{r}: 层被污染 {a[0]} vs {b[0]}"
        assert a[1] == b[1], f"{r}: 面(flipped)被污染 {a[1]} vs {b[1]}"
        assert a[2] == b[2], f"{r}: 角度被污染 {a[2]} vs {b[2]}"
        assert a[3] == b[3], f"{r}: 位置被污染 {a[3]} vs {b[3]}"
        assert a[4] == b[4], f"{r}: pad 子层被污染"
        assert a[5] == b[5], f"{r}: pad 坐标被污染"

    # hull 中心用独立路径再确认一次（GetBoundingHull 实算，与缓存无关）
    src = pcbnew.LoadBoard(ki._source_path)
    for r, fp in ki.footprints.items():
        ref_fp = _fp_of(src, r)
        if ref_fp is None:
            continue
        hx, hy = _hull_center(fp)
        rx, ry = _hull_center(ref_fp)
        assert abs(hx - rx) < TOL_MM and abs(hy - ry) < TOL_MM, (
            f"{r}: hull 中心被污染 ({hx:.4f},{hy:.4f}) vs 源 ({rx:.4f},{ry:.4f})"
        )
