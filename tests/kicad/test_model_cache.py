# -*- coding: utf-8 -*-
"""离线模型缓存（P1，`utils/kicad_utils/model_cache.py`）的回归测试。

覆盖（P1 的目标 = "训练机不依赖 KiCad"）：
  A. **dump → load 逐字段一致**（全部 16 板：footprint / pad / via / keepout / outline / net 计数与逐项字段）
  B. **几何与派生量逐位一致**（`legal_masks` / `gain_maps`（含 −inf）/ `hpwl` / `Judge` 计数）
  C. **共享引用保住了**（`net.pads[i] is footprint.pads[j]`，pickle 的坑；退化成副本会让 hpwl/Judge 静默错）
  D. **无 KiCad 也能加载**（核心目标）：屏蔽 `build_model` 后仍能拿到模型；且"缓存存在时绝不走 build"
  E. **版本 / 过期守门**：`cache_version` 不符 ⇒ `CacheVersionError`；源 SHA1 不符 ⇒ `CacheStaleError`
     （`allow_stale=True` 才放行）
  F. `index.json` 与 `.pkl` 文件一致（板顺序 = sorted，作为"这 16 块板"的唯一事实来源）
"""
import os
import pickle

import numpy as np
import pytest

from env import grid_render as gr
from env import violations as vio
from utils.kicad_utils import model_cache as mc
from utils.kicad_utils import pcb_model as pm

pytestmark = pytest.mark.require_kicad


# ---------------- fixtures ----------------

@pytest.fixture(scope="session")
def cache_root(tmp_path_factory, boards_with_models):
    """把全部板 dump 进一个 session 级临时目录（所有缓存测试复用）。"""
    root = str(tmp_path_factory.mktemp("model_cache"))
    entries = [mc.dump_model(model, path, root=root) for path, _b, model, _r in boards_with_models]
    mc.write_index(entries, root=root)
    return root


def _fp_tuple(f):
    return (f.ref, f.fpid, f.anchor, f.rotation_deg_raw, f.rotation_index, f.layer, f.flipped,
            f.hull_aabb, f.local_hull_aabb, f.ref_is_synthetic, len(f.pads))


def _pad_tuple(p):
    return (p.ref, p.number, p.netcode, p.net_id, p.layers, p.aabb, p.center, p.size, p.shape,
            p.angle_deg, p.drill, p.attribute, p.local_aabb, p.rel_offset)


def _via_tuple(v):
    return (v.netcode, v.net_id, v.layers, v.center, v.diameter, v.drill, v.aabb)


def _ko_tuple(k):
    return (k.kind, k.aabb, k.owner_ref, k.layers)


# ---------------- A. 逐字段一致 ----------------

def test_roundtrip_fields_for_all_boards(cache_root, boards_with_models):
    """全部 16 板：缓存加载后的模型与"现读"模型逐字段相等。"""
    n_boards = 0
    for path, _b, model, _r in boards_with_models:
        board = mc.board_key(path)
        m2 = mc.load_model(board, source_path=None, root=cache_root, allow_build=False)

        assert len(m2.footprints) == len(model.footprints)
        assert len(m2.pads) == len(model.pads)
        assert len(m2.vias) == len(model.vias)
        assert len(m2.keepouts) == len(model.keepouts)
        assert set(m2.nets) == set(model.nets)
        assert m2.zero_pad_footprints == model.zero_pad_footprints
        assert m2.standalone_pads == model.standalone_pads

        assert [_fp_tuple(f) for f in m2.footprints] == [_fp_tuple(f) for f in model.footprints], board
        assert [_pad_tuple(p) for p in m2.pads] == [_pad_tuple(p) for p in model.pads], board
        assert [_via_tuple(v) for v in m2.vias] == [_via_tuple(v) for v in model.vias], board
        assert [_ko_tuple(k) for k in m2.keepouts] == [_ko_tuple(k) for k in model.keepouts], board

        o1, o2 = m2.board_outline, model.board_outline
        assert (o1.bbox, o1.outline_count, o1.inferred, len(o1.slots)) == \
               (o2.bbox, o2.outline_count, o2.inferred, len(o2.slots)), board

        for k, net in model.nets.items():
            assert m2.nets[k].is_power == net.is_power
            assert len(m2.nets[k].pads) == len(net.pads)
        n_boards += 1
    assert n_boards == len(boards_with_models) and n_boards > 1


# ---------------- B. 几何/派生量逐位一致 ----------------

def test_bitwise_geometry_identical(cache_root, boards_with_models):
    """`L` / `G`（含 −inf）/ `hpwl` / `Judge` 计数：两份模型逐位一致。"""
    checked = 0
    for path, _b, model, _r in boards_with_models[:3]:          # 控时：3 块板
        board = mc.board_key(path)
        m2 = mc.load_model(board, source_path=None, root=cache_root, allow_build=False)
        spec = gr.grid_spec(model, m=gr.GRID_M)
        comp = model.footprints[0]

        L1 = gr.legal_masks(model, [], comp, spec)
        L2 = gr.legal_masks(m2, [], comp, spec)
        assert L1.shape == L2.shape and np.array_equal(L1, L2), f"{board}: L 不一致"

        G1, d1 = gr.gain_maps(model, [], comp, spec)
        G2, d2 = gr.gain_maps(m2, [], comp, spec)
        assert np.array_equal(G1, G2) and d1 == d2, f"{board}: G/degenerate 不一致"
        assert np.isneginf(G1).any() or np.isfinite(G1).all()   # 记录：G 里可能有 −inf（非法格）

        refs = {f.ref for f in model.footprints}
        assert model.hpwl(refs=refs) == m2.hpwl(refs=refs), f"{board}: HPWL 不一致"

        pl = vio.Judge(model).source_placements()
        assert dict(vio.Judge(model).judge_all(pl).counts) == \
               dict(vio.Judge(m2).judge_all(pl).counts), f"{board}: Judge 计数不一致"
        checked += 1
    assert checked == 3


# ---------------- C. 共享引用 ----------------

def test_shared_pad_references_preserved(cache_root, boards_with_models):
    """`Net.pads` 与 `Footprint.pads`、展平 `pads` 必须指向**同一批对象**（pickle 会保住共享）。"""
    path, _b, _model, _r = boards_with_models[0]
    m2 = mc.load_model(mc.board_key(path), source_path=None, root=cache_root, allow_build=False)

    net = next(n for n in m2.nets.values() if len(n.pads) >= 2)
    pad = net.pads[0]
    assert any(pad is q for f in m2.footprints for q in f.pads), \
        "net.pads 里的 pad 与 footprint.pads 不是同一对象（退化成了副本）"
    assert any(pad is q for q in m2.pads), "net.pads 里的 pad 不在展平的 model.pads 里"
    # 同一个 (ref, number) 下的多片 pad 也必须是对象本身
    key = (pad.ref, pad.number)
    assert all(any(q is p for f in m2.footprints for q in f.pads) for p in m2.pads_by_key[key])


# ---------------- D. 无 KiCad 也能加载（P1 核心） ----------------

def test_loads_without_kicad_and_never_builds_when_cached(cache_root, boards_with_models, monkeypatch):
    """① 缓存存在时**绝不**走 `build_model`；② 即使 KiCad 不可用也能加载。"""
    path, _b, model, _r = boards_with_models[0]
    board = mc.board_key(path)

    def _boom(*a, **k):
        raise AssertionError("缓存命中时不应调用 build_model")

    monkeypatch.setattr(pm, "build_model", _boom)
    m2 = mc.load_model(board, source_path=None, root=cache_root, allow_build=True)
    assert len(m2.footprints) == len(model.footprints)

    monkeypatch.setattr(pm, "KICAD_AVAILABLE", False, raising=False)
    m3 = mc.load_model(board, source_path=None, root=cache_root, allow_build=False)
    assert len(m3.pads) == len(model.pads)


def test_missing_cache_without_kicad_raises(tmp_path, boards_with_models, monkeypatch):
    """无缓存 + 不许 build + KiCad 不可用 ⇒ 必须**明确报错**（而不是静默返回坏数据）。"""
    monkeypatch.setattr(pm, "KICAD_AVAILABLE", False, raising=False)
    with pytest.raises(mc.CacheError):
        mc.load_model("不存在的板", source_path=None, root=str(tmp_path), allow_build=False)


# ---------------- E. 版本 / 过期守门 ----------------

def test_version_guard(cache_root, boards_with_models, tmp_path):
    """`cache_version` 不符 ⇒ `CacheVersionError`（fail-fast，绝不静默用旧缓存）。"""
    board = mc.board_key(boards_with_models[0][0])
    with open(os.path.join(cache_root, f"{board}.pkl"), "rb") as fh:
        blob = pickle.load(fh)
    blob["header"]["cache_version"] = mc.CACHE_VERSION + 1
    root = str(tmp_path)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{board}.pkl"), "wb") as fh:
        pickle.dump(blob, fh)

    with pytest.raises(mc.CacheVersionError):
        mc.load_model(board, source_path=None, root=root, allow_build=False)


def test_stale_source_guard(cache_root, boards_with_models, tmp_path):
    """源板文件变了（SHA1 不符）⇒ `CacheStaleError`；显式 `allow_stale=True` 才放行。"""
    path, _b, model, _r = boards_with_models[0]
    board = mc.board_key(path)
    fake = tmp_path / "fake.kicad_pcb"
    fake.write_bytes("(kicad_pcb (version 20240108)) ; 与缓存里的源不同".encode("utf-8"))

    with pytest.raises(mc.CacheStaleError):
        mc.load_model(board, source_path=str(fake), root=cache_root, allow_build=False)

    m2 = mc.load_model(board, source_path=str(fake), root=cache_root,
                       allow_build=False, allow_stale=True)
    assert len(m2.footprints) == len(model.footprints)


# ---------------- F. index.json ----------------

def test_index_matches_pkl_files(cache_root, boards_with_models):
    """索引：板数、顺序（sorted）、与磁盘上的 `.pkl` 一一对应。"""
    idx = mc.read_index(root=cache_root)
    boards = [e["board"] for e in idx["boards"]]
    assert idx["cache_version"] == mc.CACHE_VERSION
    assert idx["n_boards"] == len(boards) == len(boards_with_models)
    assert boards == sorted(boards), "板序必须是 sorted（确定性）"
    for board in boards:
        assert os.path.isfile(os.path.join(cache_root, f"{board}.pkl"))
        assert idx_has_counts(idx, board)


def idx_has_counts(idx, board):
    e = next(x for x in idx["boards"] if x["board"] == board)
    return e["counts"]["footprints"] > 0 and e["counts"]["pads"] > 0
