# -*- coding: utf-8 -*-
"""**离线模型缓存**（Owner 2026-09-22 拍案 P1）—— 让训练/推理**不再依赖 KiCad**。

为什么需要它
------------

全项目的 KiCad（`pcbnew`）依赖**只有一处**：把 `.kicad_pcb` 读成 `PcbModel` 的那一次
（`pcbnew.LoadBoard` + `pcb_model.build_model`）。此后 `L` / `G` / `Judge` / 掩膜 / `hpwl` /
完成度 / `BoardState` **全是纯 numpy 几何**（`env/` 里 0 处 `pcbnew` 调用）。

⇒ 在有 KiCad 的机器上把 16 板**预处理一次**，训练机（例如 AutoDL 的 Linux 容器，装不了 KiCad）
   只读缓存即可 ⇒ 镜像极小、启动快、且几何被【一次性冻结】在本机 KiCad 10.0.4 的口径下。

缓存长什么样
------------

```text
data/model_cache/
  index.json          ← 板清单 + 每板的源文件路径/SHA1/缓存版本/对象计数/生成时间（**顺序 = sorted，唯一事实来源**）
  <board>.pkl         ← {header: {...}, payload: {PcbModel 的 8 个构造字段}}
```

* `payload` **只存构造字段**（`path/footprints/vias/nets/board_outline/keepouts/zero_pad_footprints/
  standalone_pads`），**不存派生的 `pads` 与 `MappingProxyType`**
  ⇒ 加载时走 `PcbModel(**payload)`，由 `__post_init__` 重新派生 `pads` 与冻结 `nets`
  （顺带绕开 `MappingProxyType` **不可 pickle** 的问题，并保证派生字段永远自洽）。
* **共享引用必须保住**：`Net.pads` 存的是 pad **对象引用**（§2.5），而同一批 pad 也在
  `Footprint.pads` 里；单次 pickle 会保留这层共享（`net.pads[i] is footprint.pads[j]`）
  ⇒ 测试里有一条断言守着它（一旦退化成副本，`hpwl`/`Judge` 会**静默**错）。
* **版本与哈希守门（fail-fast，绝不静默用旧缓存）**：
  `cache_version != CACHE_VERSION` ⇒ 抛 `CacheVersionError`；
  源文件存在且 `sha1` 对不上 ⇒ 抛 `CacheStaleError`（除非显式 `allow_stale=True`）。
  源文件**不存在**（训练机上没搬板文件）⇒ 只校验版本并放行。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

from utils.kicad_utils.pcb_model import PcbModel

__all__ = [
    "CACHE_VERSION",
    "CacheError",
    "CacheVersionError",
    "CacheStaleError",
    "cache_dir",
    "sha1_file",
    "board_key",
    "source_of",
    "dump_model",
    "write_index",
    "read_index",
    "load_model",
    "is_cached",
    "cache_status",
]

CACHE_VERSION = 1                       # ⚠️ 任何会改 `PcbModel` 字段/语义的改动都要 +1
INDEX_NAME = "index.json"
CACHE_ENV = "PCB_RL_MODEL_CACHE"        # 覆盖缓存目录（默认 <repo>/data/model_cache）
_PAYLOAD_FIELDS = (                     # = `PcbModel` 的**构造字段**（不含派生的 pads / MappingProxyType）
    "path", "footprints", "vias", "nets", "board_outline",
    "keepouts", "zero_pad_footprints", "standalone_pads",
)


class CacheError(RuntimeError):
    """缓存相关的错误基类。"""


class CacheVersionError(CacheError):
    """缓存版本与代码不匹配（必须重新 dump）。"""


class CacheStaleError(CacheError):
    """源板文件已变（SHA1 不符）⇒ 缓存过期。"""


# ===================== 路径与哈希 =====================

def cache_dir(root: Optional[str] = None) -> str:
    """缓存目录（默认 `<repo>/data/model_cache`；可用 `PCB_RL_MODEL_CACHE` 覆盖）。"""
    if root:
        return os.path.abspath(root)
    env = os.environ.get(CACHE_ENV, "").strip()
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))          # utils/kicad_utils → 仓库根
    return os.path.join(repo, "data", "model_cache")


def sha1_file(path: str) -> str:
    """文件 SHA1（分块读，避免一次性吃内存）。"""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def board_key(pcb_path: str) -> str:
    """板标识 = `.kicad_pcb` **所在目录名**（全项目的板名口径，例如 `jackt-…-4.IMU小板v1_1-09db0370`）。"""
    return os.path.basename(os.path.dirname(os.path.abspath(pcb_path)))


def source_of(board: str, data_dir: str) -> Optional[str]:
    """在 `data_dir` 下按板目录名找 `.kicad_pcb`（找不到返回 None）。"""
    d = os.path.join(data_dir, board)
    if not os.path.isdir(d):
        return None
    for name in sorted(os.listdir(d)):
        if name.endswith(".kicad_pcb") and not name.endswith("best_layout.kicad_pcb"):
            return os.path.join(d, name)
    return None


# ===================== dump / load =====================

def _payload_of(model: PcbModel) -> Dict[str, Any]:
    """只取构造字段（**不含派生字段**）⇒ 加载时由 `__post_init__` 重建。

    ⚠️ `model.nets` 在 `__post_init__` 里被换成 `MappingProxyType`（**不可 pickle**）
    ⇒ 这里必须还原成普通 `dict`（加载时 `__post_init__` 会再冻结一次）。
    """
    out = {k: getattr(model, k) for k in _PAYLOAD_FIELDS}
    out["nets"] = dict(model.nets)
    return out


def _counts_of(model: PcbModel) -> Dict[str, int]:
    return {
        "footprints": len(model.footprints),
        "pads": len(model.pads),
        "vias": len(model.vias),
        "keepouts": len(model.keepouts),
        "nets": len(model.nets),
    }


def dump_model(model: PcbModel, source_path: Optional[str], *, root: Optional[str] = None) -> Dict[str, Any]:
    """把 `model` 写成 `<root>/<board>.pkl`，返回 header（供 `index.json`）。"""
    src = os.path.abspath(source_path) if source_path else (model.path or None)
    board = board_key(src) if src else os.path.basename(str(model.path))
    out_dir = cache_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    header = {
        "cache_version": CACHE_VERSION,
        "board": board,
        "source": src,
        "source_sha1": sha1_file(src) if (src and os.path.isfile(src)) else None,
        "dumped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": _counts_of(model),
    }
    blob = {"header": header, "payload": _payload_of(model)}
    with open(os.path.join(out_dir, f"{board}.pkl"), "wb") as fh:
        pickle.dump(blob, fh, protocol=4)        # 4 = 兼容性较好的现代协议
    return header


def write_index(entries: List[Dict[str, Any]], *, root: Optional[str] = None) -> str:
    """写 `index.json`（**板顺序 = 按板名 sorted**，作为"这 16 块板"的唯一事实来源）。"""
    out_dir = cache_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    entries = sorted(entries, key=lambda e: e["board"])
    doc = {
        "cache_version": CACHE_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_boards": len(entries),
        "boards": entries,
    }
    path = os.path.join(out_dir, INDEX_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return path


def read_index(*, root: Optional[str] = None) -> Dict[str, Any]:
    """读 `index.json`（不存在 ⇒ 抛 `CacheError`）。"""
    path = os.path.join(cache_dir(root), INDEX_NAME)
    if not os.path.isfile(path):
        raise CacheError(f"缓存索引不存在：{path}（先跑 `scripts/dump_models.py`）")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_cached(board: str, *, root: Optional[str] = None) -> bool:
    return os.path.isfile(os.path.join(cache_dir(root), f"{board}.pkl"))


def load_model(board: str, *, source_path: Optional[str] = None, root: Optional[str] = None,
               allow_build: bool = True, allow_stale: bool = False) -> PcbModel:
    """**缓存优先**地拿到 `PcbModel`；缓存缺失时（若有 KiCad）回落到 `pcbnew` 现读。

    Args:
        board: 板标识（= 板目录名）。
        source_path: 源 `.kicad_pcb`（给出且存在 ⇒ 校验 SHA1；为空 ⇒ 只校验版本）。
        allow_build: 缓存缺失时是否允许用 pcbnew 现读（训练机上应为 False）。
        allow_stale: SHA1 不符时是否放行（默认 **不放行** —— 宁可报错也不静默用旧几何）。
    """
    pkl = os.path.join(cache_dir(root), f"{board}.pkl")
    if os.path.isfile(pkl):
        with open(pkl, "rb") as fh:
            blob = pickle.load(fh)
        header = blob.get("header", {})
        ver = header.get("cache_version")
        if ver != CACHE_VERSION:
            raise CacheVersionError(
                f"{board}: 缓存版本 {ver} ≠ 代码要求 {CACHE_VERSION} ⇒ 重新跑 scripts/dump_models.py")
        src = source_path or header.get("source")
        if src and os.path.isfile(src):
            want, got = header.get("source_sha1"), sha1_file(src)
            if want and got != want:
                msg = f"{board}: 源板已变（sha1 {got[:8]} ≠ 缓存 {str(want)[:8]}）⇒ 缓存过期"
                if not allow_stale:
                    raise CacheStaleError(msg)
                print(f"[WARN] model_cache: {msg}（allow_stale=True，继续）")
        return PcbModel(**blob["payload"])

    if not allow_build:
        raise CacheError(f"{board}: 没有缓存 {pkl}，且 allow_build=False（训练机上请先 dump 缓存）")

    from utils.kicad_utils import pcb_model as pm          # 局部导入：避免模块级依赖
    if not getattr(pm, "KICAD_AVAILABLE", False):
        raise CacheError(
            f"{board}: 无缓存且 KiCad 不可用（KICAD_AVAILABLE=False）⇒ 先在有 KiCad 的机器上 dump 缓存")
    if not source_path or not os.path.isfile(source_path):
        raise CacheError(f"{board}: 无缓存且没给可用的 source_path={source_path!r}")
    board_obj = pm.pcbnew.LoadBoard(source_path)
    return pm.build_model(board_obj, source_path)


def cache_status(*, root: Optional[str] = None, data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """逐板体检（供 `scripts/dump_models.py --check` 与诊断）：版本 / 缓存是否存在 / 源是否变了。"""
    out: List[Dict[str, Any]] = []
    try:
        idx = read_index(root=root)
    except CacheError:
        idx = {"boards": []}
    for entry in idx.get("boards", []):
        board, src = entry["board"], entry.get("source")
        row = dict(board=board, cached=is_cached(board, root=root),
                   cache_version=entry.get("cache_version"), source=src,
                   counts=entry.get("counts"))
        if src and os.path.isfile(src):
            row["source_sha1_now"] = sha1_file(src)
            row["stale"] = row["source_sha1_now"] != entry.get("source_sha1")
        else:
            row["stale"] = None                    # 源不在本机 ⇒ 只校验版本
        out.append(row)
    return out
