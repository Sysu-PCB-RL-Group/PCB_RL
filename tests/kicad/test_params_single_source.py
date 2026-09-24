# -*- coding: utf-8 -*-
"""参数一览（`PARAMS.md`）的守门（**不需要 KiCad**）。

测四件事：
  ① 清单与代码**逐字符一致**（改了参数忘了重新生成 ⇒ 这里红）
  ② 每条登记项都能从真源读到值（且**不是 Mock** —— 防 conftest 的 `utils.kicad_utils` 注入污染）
  ③ 已知的"重复定义"两处**值相等**、且清单里**有警告标注**（避免"两份悄悄跑偏"）
  ④ 清单符合 markdown 规矩（非表格行 ≤ 80 字符；不含时间戳 ⇒ 可稳定 diff）
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import params as P  # noqa: E402

_MOCK_HINTS = ("MagicMock", "Mock object", "mock.")


def test_manifest_matches_code():
    """`PARAMS.md` 必须与 `render_markdown()` 完全一致（否则跑 `--write` 重新生成）。"""
    assert os.path.isfile(P.MANIFEST), \
        f"缺 {P.MANIFEST} ⇒ 跑 `python scripts/dump_params.py --write`"
    with open(P.MANIFEST, encoding="utf-8") as f:
        got = f.read()
    want = P.render_markdown()
    assert got == want, ("PARAMS.md 与代码不一致 ⇒ 跑 `python scripts/dump_params.py --write`"
                         "（改了参数就要重新生成）")


def test_every_spec_value_is_readable_and_not_mocked():
    """每条都能读到真值（不是 Mock、不是异常）。"""
    rows = P.collect()
    assert len(rows) == len(P.SPEC) >= 40, "登记项数异常（是否漏搬了参数？）"
    bad = [r["name"] for r in rows if any(h in str(r["value"]) for h in _MOCK_HINTS)]
    assert not bad, f"这些参数读到了 Mock（被 conftest 污染）：{bad}"
    assert all(r["meaning"] and r["source"] and r["where"] for r in rows), \
        "每条都必须有含义 / 依据 / 代码位置"


def test_known_duplicate_definitions_agree_and_are_flagged():
    """已知重复定义：`SOFT_GAP_MM`（grid_render ↔ violations）、`N_ROT`（obs_spec ↔ policy_net）。"""
    from env import grid_render as gr
    from env import obs_spec as osp
    from env import policy_net as pn
    from env import violations as vio

    assert gr.SOFT_GAP_MM == vio.SOFT_GAP_MM == 0.2, "两份软间距阈值悄悄跑偏了"
    assert osp.N_ROT == pn.N_ROT == 4, "两份旋转数悄悄跑偏了"
    assert osp.SPATIAL_CHANNELS == pn.CHANNELS == 8, "两份通道数悄悄跑偏了"

    text = P.render_markdown()
    for name in ("SOFT_GAP_MM", "N_ROT", "CHANNELS"):
        assert f"`{name}`" in text, f"{name} 未进清单"
    assert text.count("⚠️ 另一份") + text.count("⚠️ 与 `env") >= 3, "重复定义必须带警告标注"


def test_manifest_follows_markdown_rules():
    """非表格行 ≤ 80 字符；标题前后空行；不含时间戳（否则每次跑都产生 diff）。"""
    text = P.render_markdown()
    assert "生成时间" not in text and "2026-" not in text.split("##")[0], "清单里不许写时间戳"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|"):          # 表格行不换行（项目 markdown 规矩）
            continue
        assert len(line) <= 80, f"第 {i + 1} 行超过 80 字符：{line!r}"
        if line.startswith("#"):
            if i > 0:
                assert lines[i - 1] == "", f"第 {i + 1} 行标题前缺空行"
            assert i + 1 < len(lines) and lines[i + 1] == "", f"第 {i + 1} 行标题后缺空行"


def test_sections_are_complete():
    """11 个分组都要出现在清单里（防渲染时丢节）。"""
    text = P.render_markdown()
    sections = []
    for p in P.SPEC:
        if p.section not in sections:
            sections.append(p.section)
    assert len(sections) == 11, f"分组数变了：{len(sections)}"
    for sec in sections:
        assert f"## {sec}" in text, f"清单缺分组：{sec}"


if __name__ == "__main__":     # 便于单独跑：python tests/kicad/test_params_single_source.py
    raise SystemExit(pytest.main([__file__, "-q"]))
