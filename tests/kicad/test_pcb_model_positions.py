"""真实数据回归测试：转换数据的"位置信息（AABB）可唯一计算 + 可反解"。

依据《导入信息重构》§1.1b（旋转公式）、§1.2（转换角）、§4.1B（硬要求 5：pad 坐标可反解）。

三项测试（对应 footprint / pad / via）：
  1. `TestFootprintPosition`：`local_hull_aabb` + 锚点 + 转换角 ⇒ 板坐标 AABB，
     与"**真实旋转该 footprint 后再用 pcbnew 实测**"逐角对比；
     同时校验源姿态实测值 `hull_aabb` 与反解值一致（二者不一致即数据有问题）。
  2. `TestPadPosition`：pad 中心（由 `rel_offset` 反解）与 pad AABB（由 `local_aabb` 旋转）
     逐角对比实测。
  3. `TestViaPosition`：via 的 AABB 必须能由 `中心 ± 直径/2` **唯一算出**（且与 pcbnew 实测一致），
     并检查记录唯一性（同一中心不重复）与几何自洽（diameter > 0、drill ≤ diameter）。

设计原则：**能失败**。任何一处不一致都会带上"板名 / 位号 / pad 号 / 角度 / 实测 vs 反解"断言失败。
数据来源：`data/kicad-native-batch/` **全部真实板**（不伪造）；可用 `PCB_RL_TEST_MAX_BOARDS` 限量调试。
"""
import math
import os

import pytest

from utils.kicad_utils import pcb_model  # noqa: F401  触发 bootstrap（DLL 路径 + wx 日志抑制）
from utils.kicad_utils.pcb_model import (
    LAYER_B,
    LAYER_F,
    build_model,
    world_offset_to_local,
)

pcbnew = pytest.importorskip("pcbnew", reason="需要 KiCad 环境（KICAD_PATH）")

# 转换角是 90° 的倍数 ⇒ 由"局部 AABB 旋转"推出的板坐标 AABB 应当**精确**（KiCad 内部为 nm 整数）
TOL_EXACT_MM = 1e-6
# 对照 pcbnew 实测值时留 1µm 容差（浮点/单位换算裕度）
TOL_MEASURED_MM = 1e-3

ANGLE_INDICES = (0, 1, 2, 3)


def _mm(nm_value: int) -> float:
    """纳米 → mm。"""
    return pcbnew.ToMM(nm_value)


def _mm_box(box):
    """BOX2I → mm AABB（y1 = Top < y2 = Bottom）。"""
    return (_mm(box.GetLeft()), _mm(box.GetTop()), _mm(box.GetRight()), _mm(box.GetBottom()))


def _max_abs_diff(a, b) -> float:
    return max(abs(a[i] - b[i]) for i in range(4))


# ===================== §1.2 转换角规则：测试内【独立实现 + 金表】 =====================
# 注意：**故意不调用被测的 `pcb_model.convert_rotation_index()`** —— 否则该函数被改错时
# 期望值会跟着一起错（mutation testing 实测：这样会漏检）。
GOLDEN_ROTATIONS = (
    # 正交角（4 值）
    (0.0, 0), (90.0, 1), (180.0, 2), (270.0, 3), (-90.0, 3), (-180.0, 2),
    # 等距点（|θ| mod 90 == 45）：一律"按数值降级"（取数值更小的一档）
    (45.0, 0), (135.0, 1), (-45.0, 3), (-135.0, 2), (225.0, 2), (-225.0, 1),
    # 非等距：取最接近
    (44.0, 0), (46.0, 1), (-44.0, 0), (-46.0, 3), (-0.62, 0), (89.0, 1), (91.0, 1),
)


def reference_rotation_index(deg: float) -> int:
    """§1.2 规则的**独立实现**（写法与产品代码不同，避免同源错误）。"""
    a = ((deg + 180.0) % 360.0) - 180.0
    r = abs(a) % 90.0
    if abs(r - 45.0) < 1e-9:                    # 等距点：数值降级
        return int(math.floor(a / 90.0)) % 4
    return int(math.ceil(a / 90.0 - 0.5)) % 4   # 非等距：最接近（ceil(x−0.5) 形式）


@pytest.mark.require_kicad
class TestRotationRule:
    """转换角规则本身（§1.2）——用金表锁定，防止"函数与期望一起改错"。"""

    def test_golden_rotation_table(self):
        from utils.kicad_utils.pcb_model import convert_rotation_index

        bad = []
        for deg, want in GOLDEN_ROTATIONS:
            assert reference_rotation_index(deg) == want, (
                f"测试内参考实现与金表不符：{deg}° → {reference_rotation_index(deg)}，金表 {want}"
            )
            got = convert_rotation_index(deg)
            if got != want:
                bad.append(f"{deg}° → 实现 {got}，金表 {want}")
        assert not bad, "转换角规则与 §1.2 金表不符：\n   " + "\n   ".join(bad)
        print(f"\n[转换角金表] {len(GOLDEN_ROTATIONS)} 个角度全部符合 §1.2")


# 注：`boards_with_models` fixture 由 tests/kicad/conftest.py 统一提供（多个文件共享同一份）。


@pytest.mark.require_kicad
class TestRotateAabb:
    """`rotate_aabb()` 的几何正确性（含**非 90° 倍数**，那里必须用到全部 4 个角点）。"""

    def test_all_four_corners_at_arbitrary_angle(self):
        from utils.kicad_utils.pcb_model import rotate_aabb

        # 细长矩形 10×2 绕原点转 45°：极值由"另一条对角线"的两个角点决定，
        # 只用 (x1,y1)/(x2,y2) 两点会得到偏小的框（区分力来自这条）
        local = (0.0, 0.0, 10.0, 2.0)
        got = rotate_aabb(local, 45.0, (0.0, 0.0))

        th = math.radians(45.0)
        c, s = math.cos(th), math.sin(th)

        def _rot(pts):
            xs = [px * c + py * s for px, py in pts]
            ys = [-px * s + py * c for px, py in pts]
            return (min(xs), min(ys), max(xs), max(ys))

        want = _rot(((0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)))
        d = _max_abs_diff(got, want)
        assert d <= 1e-9, f"rotate_aabb(45°) 与独立计算不符 got={got} want={want} 误差={d:.9f}"

        # 反向自检：本用例必须能区分"只变换 2 个对角点"的错误实现
        wrong = _rot(((0.0, 0.0), (10.0, 2.0)))
        assert _max_abs_diff(got, wrong) > 1e-6, (
            f"该用例无法区分 2 角点实现（测试无区分力）：got={got} two_corner={wrong}"
        )
        print(f"\n[rotate_aabb 45°] got={tuple(round(v, 3) for v in got)}"
              f"（2 角点错误实现会是 {tuple(round(v, 3) for v in wrong)}）")

    def test_anchor_translation(self):
        from utils.kicad_utils.pcb_model import rotate_aabb

        local = (0.0, 0.0, 4.0, 2.0)
        base = rotate_aabb(local, 90.0, (0.0, 0.0))
        moved = rotate_aabb(local, 90.0, (100.0, 50.0))
        want = (base[0] + 100.0, base[1] + 50.0, base[2] + 100.0, base[3] + 50.0)
        d = _max_abs_diff(moved, want)
        assert d <= 1e-9, f"rotate_aabb 锚点平移不可加：got={moved} want={want} 误差={d:.9f}"


@pytest.mark.require_kicad
class TestModelGuards:
    """模型层的 fail-fast 守卫（这些错误在当前数据集上不可观测，必须单测）。"""

    def test_layer_guard_raises_on_inner_layer(self):
        from utils.kicad_utils.pcb_model import PcbModelBuilder

        with pytest.raises(ValueError):
            PcbModelBuilder._layer(pcbnew.In1_Cu)
        # 正常两面仍必须放行
        assert PcbModelBuilder._layer(pcbnew.F_Cu) == LAYER_F
        assert PcbModelBuilder._layer(pcbnew.B_Cu) == LAYER_B

    def test_footprint_ref_unique_per_board(self, boards_with_models):
        """合成 ref 必须构造上唯一（否则硬重叠 A 的"不同 footprint"判据会误并两个封装）。"""
        for path, _board, model, _raw in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            refs = [f.ref for f in model.footprints]
            assert len(refs) == len(set(refs)), (
                f"{tag}：footprint ref 不唯一（{len(refs) - len(set(refs))} 个重复），"
                f"匿名件的合成 ref 必须带序号"
            )
            for f in model.footprints:
                if f.ref_is_synthetic:
                    assert f.ref.startswith("@") and "#" in f.ref, (
                        f"{tag}：合成 ref 格式异常 {f.ref}（应为 `@<库名>#<序号>`）"
                    )


@pytest.mark.require_kicad
class TestFootprintPosition:
    """footprint：位置（AABB）可唯一计算 / 可反解。"""

    def test_footprint_aabb_roundtrip(self, boards_with_models):
        n_fp = 0
        n_check = 0
        worst = 0.0
        non_orth_seen = 0
        for path, board, model, raw_angles in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            fps = [f for f in board.GetFootprints() if list(f.Pads())]
            assert len(fps) == len(model.footprints), (
                f"{tag}：footprint 数不一致 实测 {len(fps)} vs 模型 {len(model.footprints)}"
            )
            for i, (mfp, fp) in enumerate(zip(model.footprints, fps)):
                n_fp += 1

                assert mfp.local_hull_aabb, f"{mfp.ref}@{tag}：local_hull_aabb 缺失"

                # (a0) 副作用检查：build_model **不得改动板的姿态**（必须原样还原）
                assert abs(fp.GetOrientationDegrees() - raw_angles[i]) < 1e-9, (
                    f"{mfp.ref}@{tag}：build_model 改了板的姿态（未还原）"
                    f"当前={fp.GetOrientationDegrees()} 原始={raw_angles[i]}"
                )
                # (a1) 锚点语义：必须等于文件 `(at X Y)` = GetPosition()（不是 hull 中心/pad 中心）
                assert (abs(mfp.anchor[0] - _mm(fp.GetPosition().x)) < 1e-9
                        and abs(mfp.anchor[1] - _mm(fp.GetPosition().y)) < 1e-9), (
                    f"{mfp.ref}@{tag}：anchor ≠ GetPosition() "
                    f"模型={mfp.anchor} 实际=({_mm(fp.GetPosition().x)}, {_mm(fp.GetPosition().y)})"
                )

                # (a) 角度语义：原始角必须原样保留，转换角必须由 §1.2 规则从原始角推出
                assert abs(mfp.rotation_deg_raw - raw_angles[i]) < 1e-9, (
                    f"{mfp.ref}@{tag}：rotation_deg_raw 被改写（原始角丢了）"
                    f"文件={raw_angles[i]} 模型={mfp.rotation_deg_raw}"
                )
                expect_idx = reference_rotation_index(raw_angles[i])   # 独立实现，不用被测函数
                assert mfp.rotation_index == expect_idx, (
                    f"{mfp.ref}@{tag}：转换角不符合 §1.2 规则（等距点应数值降级）"
                    f"raw={raw_angles[i]:.2f}° → 模型 {mfp.rotation_index}，期望 {expect_idx}"
                )
                if abs(raw_angles[i] % 90.0) > 1e-6:
                    non_orth_seen += 1

                saved = fp.GetOrientationDegrees()
                try:
                    # (b) 存储的 hull AABB 必须等于"在转换角下真旋转后实测"的 AABB
                    #     —— 这一条锁死"几何姿态口径"（原始姿态几何会在此暴露）
                    fp.SetOrientationDegrees(mfp.rotation_index * 90)
                    measured_conv = _mm_box(fp.GetBoundingHull().BBox())
                    d = _max_abs_diff(measured_conv, mfp.hull_aabb)
                    worst = max(worst, d)
                    n_check += 1
                    assert d <= TOL_MEASURED_MM, (
                        f"{mfp.ref}@{tag}：存储的 hull_aabb 不是转换角（{mfp.rotation_index * 90}°）下的几何 "
                        f"实测={measured_conv} 存储={mfp.hull_aabb} 误差={d:.6f}mm"
                    )

                    # (c) 反解一致性：局部 AABB + 锚点 + 转换角 ⇒ 存储值
                    rev_source = mfp.hull_aabb_at()
                    d = _max_abs_diff(rev_source, mfp.hull_aabb)
                    worst = max(worst, d)
                    assert d <= TOL_MEASURED_MM, (
                        f"{mfp.ref}@{tag}：hull AABB 反解不一致 "
                        f"存储={mfp.hull_aabb} 反解={rev_source} 误差={d:.6f}mm"
                    )

                    # (d) 四个转换角逐角：真旋转实测 vs 反解值
                    for idx in ANGLE_INDICES:
                        fp.SetOrientationDegrees(idx * 90)
                        measured = _mm_box(fp.GetBoundingHull().BBox())
                        computed = mfp.hull_aabb_at(idx)
                        d = _max_abs_diff(measured, computed)
                        worst = max(worst, d)
                        n_check += 1
                        assert d <= TOL_MEASURED_MM, (
                            f"{mfp.ref}@{tag} 角{idx * 90}°：hull AABB 反解不一致 "
                            f"实测={measured} 反解={computed} 误差={d:.6f}mm"
                        )
                finally:
                    fp.SetOrientationDegrees(saved)

        assert n_fp == 857, f"footprint 数异常：{n_fp}（期望 857）"
        # 覆盖性：必须真的走过"非正交 → 转换角"这条分支（否则 tie 规则永远测不到）
        assert non_orth_seen == 16, (
            f"非正交原件数异常：{non_orth_seen}（期望 16）⇒ 转换角 tie 分支未被覆盖"
        )
        print(
            f"\n[footprint] 校验 {n_fp} 个 × 4 角 = {n_check} 次；最大误差 {worst:.6f} mm；"
            f"非正交 {non_orth_seen} 个"
        )


@pytest.mark.require_kicad
class TestPadPosition:
    """pad：中心（连接点）与 AABB 可唯一计算 / 可反解。"""

    def test_pad_center_and_aabb_roundtrip(self, boards_with_models):
        n_pad = 0
        n_check = 0
        worst_center = 0.0
        worst_aabb = 0.0
        for path, board, model, _raw_angles in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            fps = [f for f in board.GetFootprints() if list(f.Pads())]
            assert len(fps) == len(model.footprints), (
                f"{tag}：footprint 数不一致 实测 {len(fps)} vs 模型 {len(model.footprints)}"
            )
            for mfp, fp in zip(model.footprints, fps):
                pads = list(fp.Pads())
                assert len(pads) == len(mfp.pads), (
                    f"{mfp.ref}@{tag}：pad 数不一致 实测 {len(pads)} vs 模型 {len(mfp.pads)}"
                )

                for mpad, pad in zip(mfp.pads, pads):
                    n_pad += 1
                    assert mpad.local_aabb, f"{mfp.ref}#{mpad.number}@{tag}：local_aabb 缺失"
                    assert mpad.rel_offset, f"{mfp.ref}#{mpad.number}@{tag}：rel_offset 缺失"

                    # (a) 源姿态：中心与 AABB 的实测值必须等于反解值
                    c = mfp.pad_center_at(mpad)
                    meas_c = (mpad.center[0], mpad.center[1])
                    dc = max(abs(c[0] - meas_c[0]), abs(c[1] - meas_c[1]))
                    a = mfp.pad_aabb_at(mpad)
                    da = _max_abs_diff(a, mpad.aabb)
                    worst_center = max(worst_center, dc)
                    worst_aabb = max(worst_aabb, da)
                    assert dc <= TOL_MEASURED_MM, (
                        f"{mfp.ref}#{mpad.number}@{tag}：pad 中心反解不一致 "
                        f"实测={meas_c} 反解={c} 误差={dc:.6f}mm"
                    )
                    assert da <= TOL_MEASURED_MM, (
                        f"{mfp.ref}#{mpad.number}@{tag}：源姿态 pad AABB 反解不一致 "
                        f"实测={mpad.aabb} 反解={a} 误差={da:.6f}mm"
                    )

                    # (a2) 存储的 AABB / 中心必须等于"在**转换角**下真旋转后实测"的值
                    #      —— 锁死几何姿态口径（存原始姿态几何会在此暴露）
                    saved_c = fp.GetOrientationDegrees()
                    try:
                        fp.SetOrientationDegrees(mfp.rotation_index * 90)
                        pos2 = pad.GetPosition()
                        meas_c2 = (_mm(pos2.x), _mm(pos2.y))
                        meas_a2 = _mm_box(pad.GetBoundingBox())
                        meas_ang2 = float(pad.GetOrientationDegrees())
                    finally:
                        fp.SetOrientationDegrees(saved_c)
                    dc2 = max(abs(meas_c2[0] - mpad.center[0]), abs(meas_c2[1] - mpad.center[1]))
                    da2 = _max_abs_diff(meas_a2, mpad.aabb)
                    d_ang = abs(meas_ang2 - mpad.angle_deg)
                    assert d_ang <= TOL_MEASURED_MM, (
                        f"{mfp.ref}#{mpad.number}@{tag}：pad 绝对角不是在转换角"
                        f"（{mfp.rotation_index * 90}°）下测的 "
                        f"实测={meas_ang2:.3f}° 存储={mpad.angle_deg:.3f}° 误差={d_ang:.6f}°"
                    )
                    worst_center = max(worst_center, dc2)
                    worst_aabb = max(worst_aabb, da2)
                    assert dc2 <= TOL_MEASURED_MM and da2 <= TOL_MEASURED_MM, (
                        f"{mfp.ref}#{mpad.number}@{tag}：存储值不是在转换角"
                        f"（{mfp.rotation_index * 90}°）下测的 "
                        f"中心 实测={meas_c2} 存储={mpad.center} 误差={dc2:.6f}mm；"
                        f"AABB 实测={meas_a2} 存储={mpad.aabb} 误差={da2:.6f}mm"
                    )

                    # (b) 四个转换角：真旋转后 pcbnew 实测
                    saved = fp.GetOrientationDegrees()
                    try:
                        for idx in ANGLE_INDICES:
                            fp.SetOrientationDegrees(idx * 90)
                            pos = pad.GetPosition()
                            meas_c = (_mm(pos.x), _mm(pos.y))
                            meas_a = _mm_box(pad.GetBoundingBox())
                            c = mfp.pad_center_at(mpad, idx)
                            a = mfp.pad_aabb_at(mpad, idx)
                            dc = max(abs(c[0] - meas_c[0]), abs(c[1] - meas_c[1]))
                            da = _max_abs_diff(a, meas_a)
                            worst_center = max(worst_center, dc)
                            worst_aabb = max(worst_aabb, da)
                            n_check += 1
                            assert dc <= TOL_MEASURED_MM, (
                                f"{mfp.ref}#{mpad.number}@{tag} 角{idx * 90}°：pad 中心反解不一致 "
                                f"实测={meas_c} 反解={c} 误差={dc:.6f}mm"
                            )
                            assert da <= TOL_MEASURED_MM, (
                                f"{mfp.ref}#{mpad.number}@{tag} 角{idx * 90}°：pad AABB 反解不一致 "
                                f"实测={meas_a} 反解={a} 误差={da:.6f}mm"
                            )
                    finally:
                        fp.SetOrientationDegrees(saved)

        assert n_pad == 3780, f"pad 数异常：{n_pad}（期望 3780）"
        print(
            f"\n[pad] 校验 {n_pad} 个 × 4 角（{n_check} 组中心+AABB）；"
            f"中心最大误差 {worst_center:.6f} mm，AABB 最大误差 {worst_aabb:.6f} mm"
        )

    def test_reversal_formula_selfconsistent(self, boards_with_models):
        """反解公式自洽：`anchor + R(−θ)·rel_offset` 与模型中所有 pad 的中心一致。

        覆盖全部 pad × 4 角（纯计算，不依赖 pcbnew 旋转），确保"中心"只由
        `锚点 + rel_offset + 转换角` 决定，不存在第二套口径。
        """
        n = 0
        worst = 0.0
        worst_inv = 0.0
        for _path, _board, model, _raw_angles in boards_with_models:
            for mfp in model.footprints:
                for mpad in mfp.pads:
                    for idx in ANGLE_INDICES:
                        th = math.radians(idx * 90.0)
                        c, s = math.cos(th), math.sin(th)
                        lx, ly = mpad.rel_offset
                        by_matrix = (mfp.anchor[0] + lx * c + ly * s, mfp.anchor[1] - lx * s + ly * c)
                        direct = mfp.pad_center_at(mpad, idx)
                        d = max(abs(by_matrix[0] - direct[0]), abs(by_matrix[1] - direct[1]))
                        worst = max(worst, d)
                        n += 1
                        assert d <= TOL_EXACT_MM, (
                            f"{mfp.ref}#{mpad.number} 角{idx * 90}°：中心反解公式不自洽 误差={d:.9f}mm"
                        )

                    # 逆变换自洽：R(+θ)·(当前角下的 center − 锚点) 必须等于存储的 rel_offset
                    dx = mpad.center[0] - mfp.anchor[0]
                    dy = mpad.center[1] - mfp.anchor[1]
                    back = world_offset_to_local(dx, dy, mfp.rotation_index * 90.0)
                    di = max(abs(back[0] - mpad.rel_offset[0]), abs(back[1] - mpad.rel_offset[1]))
                    worst_inv = max(worst_inv, di)
                    assert di <= TOL_EXACT_MM, (
                        f"{mfp.ref}#{mpad.number}：局部偏移逆变换不自洽 误差={di:.9f}mm"
                    )
        print(f"\n[pad 反解公式] {n} 次正/逆变换一致；最大误差 正 {worst:.9f} mm / 逆 {worst_inv:.9f} mm")


@pytest.mark.require_kicad
class TestViaPosition:
    """via：位置（AABB）可唯一计算，且记录自洽、不重复。"""

    def test_via_aabb_consistency(self, boards_with_models):
        n = 0
        worst = 0.0
        dup_groups = 0
        for path, board, model, _raw_angles in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))

            # 与 pcbnew 原始 via 逐条对齐（同一 board 对象、同一遍历顺序）
            raw_vias = [t for t in board.GetTracks() if t.Type() == pcbnew.PCB_VIA_T]
            assert len(raw_vias) == len(model.vias), (
                f"{tag}：via 数不一致 实测 {len(raw_vias)} vs 模型 {len(model.vias)}"
            )

            for mv, rv in zip(model.vias, raw_vias):
                n += 1
                cx, cy = mv.center
                half = mv.diameter / 2.0
                computed = (cx - half, cy - half, cx + half, cy + half)
                d = _max_abs_diff(computed, mv.aabb)
                worst = max(worst, d)
                assert d <= TOL_EXACT_MM, (
                    f"{tag} via@({cx:.3f},{cy:.3f})：AABB 不可由 中心±直径/2 唯一算出 "
                    f"记录={mv.aabb} 计算={computed} 误差={d:.9f}mm"
                )

                # 与 pcbnew 实测中心/直径一致
                pos = rv.GetPosition()
                mcx, mcy = _mm(pos.x), _mm(pos.y)
                mdia = _mm(rv.GetFrontWidth())
                assert abs(mcx - cx) <= TOL_MEASURED_MM and abs(mcy - cy) <= TOL_MEASURED_MM, (
                    f"{tag}：via 中心与 pcbnew 不一致 记录=({cx:.6f},{cy:.6f}) 实测=({mcx:.6f},{mcy:.6f})"
                )
                assert abs(mdia - mv.diameter) <= TOL_MEASURED_MM, (
                    f"{tag}：via 直径与 pcbnew 不一致 记录={mv.diameter} 实测={mdia}"
                )

                # 几何自洽
                assert mv.diameter > 0, f"{tag}：via 直径非正 {mv.diameter}"
                assert mv.drill <= mv.diameter, (
                    f"{tag}：via 钻孔 {mv.drill} > 外径 {mv.diameter}（数据异常）"
                )
                assert mv.layers and set(mv.layers) <= {LAYER_F, LAYER_B}, (
                    f"{tag}：via 层集合异常 {mv.layers}"
                )
                # 层集合必须与 pcbnew 实测一致（防止"只留 F.Cu"之类的漏层）
                raw_layers = tuple(
                    L for L, lid in ((LAYER_F, pcbnew.F_Cu), (LAYER_B, pcbnew.B_Cu))
                    if rv.IsOnLayer(lid)
                )
                assert mv.layers == raw_layers, (
                    f"{tag}：via 层集合与 pcbnew 不一致 模型={mv.layers} 实测={raw_layers}"
                )

            # 记录唯一性诊断：源文件允许"叠放的同参数过孔"（全数据集实测仅 1 组），
            # 但重复项的几何/网络必须完全一致，否则就是数据歧义
            seen = {}
            for mv in model.vias:
                key = (round(mv.center[0], 6), round(mv.center[1], 6), round(mv.diameter, 6))
                prev = seen.get(key)
                if prev is not None:
                    assert (prev.drill, prev.layers, prev.netcode) == (mv.drill, mv.layers, mv.netcode), (
                        f"{tag}：via 记录歧义 —— 同一 (中心, 直径)={key} 出现两条不同记录：{prev} vs {mv}"
                    )
                    dup_groups += 1
                else:
                    seen[key] = mv

        assert n == 3422, f"via 数异常：{n}（期望 3422）"
        print(
            f"\n[via] 校验 {n} 个：AABB 全部可由 中心±直径/2 唯一算出；最大误差 {worst:.9f} mm；"
            f"同参数叠放记录 {dup_groups} 组（合法）"
        )
