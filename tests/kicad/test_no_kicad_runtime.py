# -*- coding: utf-8 -*-
"""**训练机路径的"无 KiCad"端到端冒烟**（P1 收尾；Owner 2026-09-22 拍）。

它证明什么
----------
训练机（AutoDL 等装不了 KiCad 的容器）上**整条新一代链路**能跑通：

```text
model_cache.load_model(allow_build=False) → plan_order → CompletionTable
  → BoardState.start → legal_masks / gain_maps / dispersion_delta / lookahead_mask
  → place → hpwl / CompletionReport / PlacedAudit → judge_model
```

且**一次都不需要** `pcbnew` / `wx`。

三条硬纪律（比"能跑"更重要）
--------------------------
1. **子进程 + `sys.meta_path` 拦截**：`pcbnew` / `wx` 一旦被 import 就抛 `ImportError`
   ⇒ 谁把 KiCad 拖回热路径，本测试立刻红（这是"不依赖 KiCad"的**强证明**，不是"没报错"）。
2. **`KICAD_PATH` 置空**：模拟训练机（连 KiCad 路径都没有）。
3. **stdout/stderr 必须干净**（无 `[INFO]` / `[WARN]`）⇒ 守住两个 `__init__.py` 的**惰性导出**：
   静态 `from .kicad_utils import ...` 会让训练机在 import 期白打两行 `[INFO]`
   （`utils/kicad_utils/kicad_utils.py:24-27`），并把上一代模块整条拉进训练进程。

为什么必须用子进程
----------------
本仓库其它测试会在**进程内**注入 mock、设 `KICAD_PATH`、删模块缓存
（`tests/kicad/conftest.py:16-31`）⇒ 只有独立子进程能保证"干净环境"这一前提成立。
"""
import json
import os
import subprocess
import sys

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================== 子进程里跑的冒烟脚本 =====================

_SMOKE = r'''
import json
import os
import sys

os.environ.pop("KICAD_PATH", None)                  # 训练机：连 KiCad 路径都没有


class _NoKiCad:
    """训练路径上 `pcbnew` / `wx` 必须一次都不被 import。"""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("pcbnew", "wx"):
            raise ImportError("[BLOCKED] 训练路径不应 import " + fullname)
        return None


sys.meta_path.insert(0, _NoKiCad())

import numpy as np                                  # noqa: E402

from utils.kicad_utils import model_cache as mc     # noqa: E402
from env import board_state as bs                   # noqa: E402
from env import completion as cp                    # noqa: E402
from env import grid_render as gr                   # noqa: E402
from env import lookahead as la                     # noqa: E402
from env import order_plan as op                    # noqa: E402
from env import violations as vio                   # noqa: E402

# ---- 惰性导出守门：上一代模块一个都不许被连带导入 ----
_LEGACY = ("utils.kicad_utils.kicad_utils", "env.pcb_layout_env", "env.pcb_utils")
legacy = sorted(m for m in sys.modules if m in _LEGACY)
assert not legacy, "上一代模块被连带导入了（惰性导出被破坏）：" + repr(legacy)

BOARDS = [e["board"] for e in mc.read_index()["boards"]]
assert BOARDS, "缓存索引里没有板"

# ---- 选一块"开局有合法位置"的板（结构性死锁板，例如 trumpx/U1，会被自然跳过）----
picked = None
for board in BOARDS[:5]:
    model = mc.load_model(board, allow_build=False)          # ★ allow_build=False = 训练机口径
    order = tuple(op.plan_order(model))
    if len(order) < 2:
        continue
    state0 = bs.BoardState.start(model, order)
    spec = gr.grid_spec(model, m=gr.GRID_M)
    L0 = gr.legal_masks(model, state0.placed, state0.comp, spec)
    if L0.any():
        picked = (board, model, order, state0, spec, L0)
        break
assert picked is not None, f"前 5 块板在 t=0 都没有合法位置：{BOARDS[:5]}"

board, model, order, state0, spec, L0 = picked
n = len(order)

# ① 完成度（§0.6 m）：起步 0 → 放一件后 1/n
table = cp.CompletionTable.build(model, order)
assert table.n == n
r0 = table.of(state0)
assert r0.index == 0 and r0.count_ratio == 0.0 and r0.primary == r0.count_ratio

# ② L / G / σ（grid_render，纯 numpy）
G0, _degen = gr.gain_maps(model, state0.placed, state0.comp, spec, legal=L0)
Dd0 = gr.dispersion_delta(model, state0.placed, state0.comp, spec, L0)
assert L0.shape == G0.shape == (4, spec.m, spec.m)
assert np.isfinite(Dd0).all()

# ③ 前瞻掩膜（lookahead）
L1 = gr.legal_masks(model, state0.placed, state0.next_comp, spec)
mask, _info = la.lookahead_mask(model, state0.placed, state0.comp, state0.next_comp,
                                spec, legal=L0, legal_next=L1)
assert mask.shape == L0.shape and not (mask & ~L0).any(), "掩膜必须 ⊆ L"
assert mask.any()

# ④ 走一步（BoardState 是唯一状态载体）⇒ 完成度 / HPWL / 自检都要动
k, i, j = (int(v) for v in np.argwhere(mask)[0])
state1 = state0.place(k, i, j, spec=spec, legal=mask)
assert state1.index == 1 and len(state1.placed) == 1
r1 = table.of(state1)
assert r1.count_ratio == 1.0 / n
h1 = float(model.hpwl(refs=set(state1.placed_refs), poses=state1.poses))
assert np.isfinite(h1) and h1 >= 0.0
audit = state1.audit()
assert audit.clean, audit.describe()                # 注：`clean` 是 property（不是方法）

# ⑤ 判定器（violations）：整板源布局判一次（纯几何）
vio.judge_model(model)

print("SMOKE " + json.dumps({
    "board": board,
    "n_total": n,
    "L0_legal": int(L0.sum()),
    "mask_kept": int(mask.sum()),
    "completion_after_1": r1.count_ratio,
    "hpwl_after_1_mm": round(h1, 6),
    "legacy_modules": legacy,
    "boards_in_index": len(BOARDS),
}, ensure_ascii=False))
'''


# ===================== A. 无 KiCad 端到端 =====================

def test_training_path_runs_without_kicad():
    """子进程里（无 `KICAD_PATH`、`pcbnew`/`wx` 被拦截）跑完整条训练侧链路。

    失败信息里会带子进程 stdout/stderr，便于定位是"哪一步需要 KiCad"还是"哪里打了噪音"。
    """
    if not os.path.isfile(os.path.join(ROOT, "data", "model_cache", "index.json")):
        pytest.skip("没有 data/model_cache/index.json（先跑 scripts/dump_models.py）")

    env = dict(os.environ)
    env.pop("KICAD_PATH", None)                    # 训练机口径
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    p = subprocess.run([sys.executable, "-c", _SMOKE], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)

    assert p.returncode == 0, (
        f"无 KiCad 冒烟失败（退出码 {p.returncode}）\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")

    # 噪音守门（③）：惰性导出被破坏时这里会先红
    both = (p.stdout or "") + (p.stderr or "")
    noise = [ln for ln in both.splitlines()
             if "[INFO]" in ln or "[WARN]" in ln or "KICAD_PATH" in ln]
    assert not noise, "训练机路径出现 KiCad 噪音（__init__ 惰性导出被破坏？）：" + repr(noise)

    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("SMOKE ")]
    assert len(lines) == 1, f"未拿到唯一 SMOKE 行：\n{p.stdout}"
    d = json.loads(lines[0][len("SMOKE "):])

    assert d["legacy_modules"] == [], f"上一代模块被连带导入：{d['legacy_modules']}"
    assert d["L0_legal"] > 0 and d["mask_kept"] > 0
    assert d["n_total"] >= 2
    assert d["completion_after_1"] == pytest.approx(1.0 / d["n_total"])
    print(f"\n[无 KiCad 冒烟] {d}")


# ===================== B. 缓存里每块板都能离线加载 =====================

def test_every_cached_board_loads_without_build():
    """训练机的**真实门槛**：`index.json` 里的每块板都能 `allow_build=False` 加载。

    （`test_model_cache.py` 用的是"现 dump 到 tmp"的缓存；本条用的是**仓库里的正式缓存**。）
    """
    from utils.kicad_utils import model_cache as mc

    if not os.path.isfile(os.path.join(mc.cache_dir(), "index.json")):
        pytest.skip("没有 data/model_cache/index.json（先跑 scripts/dump_models.py）")

    boards = [e["board"] for e in mc.read_index()["boards"]]
    assert boards, "缓存索引里没有板"
    sizes = {}
    for board in boards:
        m = mc.load_model(board, allow_build=False)          # SH 源文件不在也能跑；在则校验 SHA1
        assert len(m.footprints) > 0 and len(m.pads) > 0, board
        sizes[board] = len(m.footprints)
    assert len(sizes) == len(boards)


# ===================== C. 惰性导出不破坏旧链路 =====================

def test_lazy_exports_keep_legacy_links_and_reject_typos():
    """PEP 562 惰性导出：旧符号仍可取用（`from utils.kicad_utils import KiCadUtils` 等），
    且拼错的名字必须抛 `AttributeError`（不能静默吞掉）。"""
    import utils.kicad_utils as ku
    import env as env_pkg

    for pkg in (ku, env_pkg):
        with pytest.raises(AttributeError):
            getattr(pkg, "NoSuchSymbol")

    # 旧链路可用（惰性导入在访问时发生）
    assert set(ku.__all__) == {"Component", "Net", "KiCadUtils"}
    assert isinstance(ku.KiCadUtils, type)
    assert isinstance(ku.Net, type) and isinstance(ku.Component, type)

    assert set(env_pkg.__all__) == {"PCBUtils", "PCBLayoutEnv"}
    assert isinstance(env_pkg.PCBUtils, type)          # 不依赖 gymnasium
    pytest.importorskip("gymnasium", reason="上一代 env 需要 gymnasium；与本条守门无关")
    assert isinstance(env_pkg.PCBLayoutEnv, type)
