"""真实 KiCad export_layout 集成测试。

验证导出后的 .kicad_pcb 可被重新加载，元件位置/旋转/层变更正确。

⚠️ **不要用"字节相同"判定"两次导出等价"**（probe210，2026-09-22）：
`export_layout` 的产物有**两处合法的、每次都不同的东西** ——
① `(uuid …)` **每次导出都重新生成**；② footprint **块的顺序不稳定**（疑为对 SWIG 对象集合迭代）。
⇒ 字节比对**恒失败**，且失败**不代表**"缓存被污染"。要判定等价，请用本文件下方的
`_footprint_records()`（件级 `ref → (layer, at)`，与块序/uuid 无关）+ `_lines_without_uuid()`（行级多重集）。
"""

import os
import re
import tempfile
import pytest
from utils.kicad_utils.kicad_utils import KiCadUtils


# ---- 规范化解析：判决"两次导出内容等价"（与块顺序、uuid 无关） ----

_FP_START = re.compile(r"^\t\(footprint\b")
_FP_LAYER = re.compile(r'^\t\t\(layer "([^"]+)"\)\s*$')
_FP_AT = re.compile(r"^\t\t\(at ([-\d.]+) ([-\d.]+)(?: ([-\d.]+))?\)\s*$")
_FP_REF = re.compile(r'\(property "Reference" "([^"]+)"')


def _footprint_records(pcb_path):
    """把 `.kicad_pcb` 解析成 `{ref: (layer, x, y, rot)}`（**丢掉 uuid**、与块顺序无关）。"""
    out, cur = {}, None
    with open(pcb_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            ln = raw.rstrip("\r\n")
            if _FP_START.match(ln):
                if cur and cur["ref"]:
                    out[cur["ref"]] = (cur["layer"], cur["at"])
                cur = {"layer": None, "at": None, "ref": None}
                continue
            if cur is None:
                continue
            if cur["ref"] is None:
                m = _FP_REF.search(ln)
                if m:
                    cur["ref"] = m.group(1)
                    continue
            if cur["layer"] is None:
                m = _FP_LAYER.match(ln)
                if m:
                    cur["layer"] = m.group(1)
                    continue
            if cur["at"] is None:
                m = _FP_AT.match(ln)
                if m:
                    cur["at"] = (float(m.group(1)), float(m.group(2)), float(m.group(3) or 0.0))
    if cur and cur["ref"]:
        out[cur["ref"]] = (cur["layer"], cur["at"])
    return out


def _lines_without_uuid(pcb_path):
    """去掉 `(uuid …)` 后的**排序行多重集**（块序无关；用于抓"除 uuid/块序外"的任何内容差异）。"""
    with open(pcb_path, "r", encoding="utf-8") as fh:
        return sorted(ln.rstrip("\r\n") for ln in fh if "(uuid" not in ln)


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
        """移动元件后导出，再加载验证位置已更新。

        注意：export_layout 只写 is_placed=True 的元件（固定 + 未放置保持源坐标），
        测试必须先置 is_placed=True，否则元件不会被导出。
        """
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        # 找一个可移动元件并记录原始位置
        movable = [(ref, c) for ref, c in ki.components.items() if not c.is_fixed]
        if not movable:
            pytest.skip("No movable components to test")
        ref, comp = movable[0]
        orig_bbox = tuple(comp.bbox)
        comp.is_placed = True          # 否则 export_layout 会跳过未放置元件
        comp.layer = 0                 # 保证层守卫通过（F.Cu）

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

    def test_export_does_not_mutate_cache_board(self, kicad_pcb_path):
        """export_layout 每次重载独立模板，不修改缓存的 self.board / self.footprints。

        回归：旧实现会原地修改缓存中的 footprint 对象，导致 final/best 多次导出互相污染。
        """
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)

        movable = [(ref, c) for ref, c in ki.components.items() if not c.is_fixed]
        if not movable:
            pytest.skip("No movable components to test")
        ref, comp = movable[0]
        orig_bbox = tuple(comp.bbox)
        comp.is_placed = True
        comp.layer = 0
        # 修改 RL 状态（bbox 平移 10mm）——只应影响导出，不应动缓存 board
        comp.bbox = (orig_bbox[0] + 10, orig_bbox[1], orig_bbox[2] + 10, orig_bbox[3])

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "a.kicad_pcb")
            p2 = os.path.join(tmpdir, "b.kicad_pcb")
            assert ki.export_layout(p1, ki.components)
            assert ki.export_layout(p2, ki.components)

            # 缓存中的 footprint 仍应在原始位置（hull 中心 == 原始 bbox 中心）
            import pcbnew  # noqa: E402 — 由 conftest 保证可用
            fp = ki.footprints.get(ref)
            assert fp is not None
            h = fp.GetBoundingHull().BBox()
            hcx = pcbnew.ToMM((h.GetLeft() + h.GetRight()) / 2.0)
            hcy = pcbnew.ToMM((h.GetTop() + h.GetBottom()) / 2.0)
            orig_cx = (orig_bbox[0] + orig_bbox[2]) / 2.0
            orig_cy = (orig_bbox[1] + orig_bbox[3]) / 2.0
            assert abs(hcx - orig_cx) < 0.05, (
                f"缓存 board 被污染: hull 中心 x {hcx:.4f} != 原始 {orig_cx:.4f}"
            )
            assert abs(hcy - orig_cy) < 0.05

            # 两次导出**内容等价**（不是"字节相同"：uuid 每次重生成、footprint 块顺序不稳定，
            # 见文件头说明与 probe210；用规范化比较才是在守"互不污染"这个真意图）
            r1, r2 = _footprint_records(p1), _footprint_records(p2)
            assert len(r1) == len(ki.footprints) == len(r2), (
                f"解析出的件数应等于板上件数：{len(r1)} / {len(ki.footprints)} / {len(r2)}")
            bad = {k: (r1.get(k), r2.get(k)) for k in set(r1) | set(r2) if r1.get(k) != r2.get(k)}
            assert not bad, f"两次导出的件级内容不一致（真污染）：{bad}"
            s1, s2 = _lines_without_uuid(p1), _lines_without_uuid(p2)
            assert s1 == s2, ("去掉 uuid 后仍有行级差异（除块顺序外不该有）："
                              f"{[(a, b) for a, b in zip(s1, s2) if a != b][:5]}")

    def test_canonical_compare_has_discriminating_power(self, kicad_pcb_path):
        """**判别力守门**（纪律：断言不能是"永不失败的"）：规范化比较必须能抓到**真差异**。

        做法：同一块板导出两次，第二次把同一个件的 bbox 再挪 (+5, 0) ⇒ 两次的件级记录里
        **只有该件不同、其余件全部相同**（若 `_footprint_records` 解析失效或比对退回"恒真"，本条会红）。
        """
        ki = KiCadUtils()
        ki.load_board(kicad_pcb_path)
        movable = [(ref, c) for ref, c in ki.components.items() if not c.is_fixed]
        if not movable:
            pytest.skip("No movable components to test")
        ref, comp = movable[0]
        orig = tuple(comp.bbox)
        comp.is_placed = True
        comp.layer = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "x.kicad_pcb")
            p2 = os.path.join(tmpdir, "y.kicad_pcb")
            comp.bbox = (orig[0] + 10, orig[1], orig[2] + 10, orig[3])
            assert ki.export_layout(p1, ki.components)
            comp.bbox = (orig[0] + 15, orig[1], orig[2] + 15, orig[3])
            assert ki.export_layout(p2, ki.components)

            r1, r2 = _footprint_records(p1), _footprint_records(p2)   # 必须在临时目录仍存在时读
            assert r1 and set(r1) == set(r2)
            diff = {k for k in r1 if r1[k] != r2[k]}
            assert diff == {ref}, f"应只有 {ref} 变（实得 {sorted(diff)}）⇒ 解析器或比对失效"
            # 其余件（含固定件与有旋转的件）必须逐位相同 ⇒ 解析器覆盖到了 rot 与 layer
            assert all(r1[k][0] is not None and r1[k][1] is not None for k in r1), "有件没解析出 layer/at"

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
