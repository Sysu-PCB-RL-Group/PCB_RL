# -*- coding: utf-8 -*-
"""固定放置顺序（`env/order_plan.py`）的回归测试（真实 20 板）。

规则（Owner 2026-09-17 拍案）：`score = 0.7·面积分 + 0.3·连通性分`（连通性**每步重算**：
与已放集合共享的**非电源 net** 数），电源专属件用"共享电源网数"兜底；
tie-break = 面积↓ → pad 数↓ → ref↑（**全序**）。

三层断言（越往下越严）：
  A. `test_order_golden`           逐板顺序 SHA1 金表（任何规则/口径改动都会红）
  B. `test_order_deterministic`    打乱元件输入顺序 5 次，顺序必须**逐位一致**
                                   （防 Python str 哈希随机化导致跨进程漂移）
  C. `test_order_metrics_aggregate` 20 板聚合指标 ≈ 金表
                                   （迷茫A / 首→次步距 / 前25%步面积 / 大件位置 / 电源件位置）

运行（需 KiCad 环境）：
    $env:KICAD_PATH="E:\\Kicad"; python -m pytest tests/kicad/test_order_plan.py -v
"""
import hashlib
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import order_plan as op  # noqa: E402

# ⚠️ 必须**模块级**导入 `pcb_model`：`tests/conftest.py` 的 autouse fixture 会把
# `sys.modules["utils.kicad_utils"]` 换成 MagicMock（全局粘性）⇒ 模块级导入早于污染。
from utils.kicad_utils import pcb_model as _PCB_MODEL  # noqa: E402

pytestmark = pytest.mark.require_kicad

# ---- w=0.7 的顺序金表（probe147 实测；hash, 件数, 首件, 末件） ----
ORDER_GOLDEN = {
    "bliysfyt-night-light-af2cf507--Board2-f9911fd2": ("322f8fe5ec3d", 11, "USB1", "R1"),
    "climbsnail-spotwelder-45c7d2f7--MiniMos板切分-f02d78a5": ("344204186b7c", 24, "U5", "R32"),
    "climbsnail-spotwelder-45c7d2f7--Mini控制板切分-b0eefa2f": ("0bf1515079c6", 84, "U3", "R5"),
    "eda_uahyxtbgs-165w-three-way-multi-protocol-fa-dc91a91e--300W四路彩屏桌面电源（彩屏版）-56ddf4ed":
        ("f28fc0a30c07", 97, "CN2", "R22"),
    "elisaday-high-color-value-low-cost-reflow-24a201e4--LCD转接板-956a8225": ("6d71b7f5e291", 10, "U15", "R23"),
    "esp-college-project_bkymsryt-aeef78a5--ShuttleBoard-BMI270&BMM350-V1_2-b0ba0089":
        ("91d1496abcdd", 21, "H1", "R4"),
    "fei1554973967-sw3538-dff8db20--底板带串口-a82dd2f4": ("a1fc1c365126", 78, "CN1", "D3"),
    "iamdt-guang-chui-60-40a2c53b--主板v1(不建议用于新复刻)-385a2a21": ("df82bace13b2", 86, "U10", "D"),
    "jackt-tong-yong-xiao-che-kong-zhi-qi-zhuan-li-948f377a--4.IMU小板v1_1-09db0370":
        ("d92bca2bb40a", 10, "FPC1", "R2"),
    "li-chuang-kai-fa-ban-li-chuang-coloreasypico2-kai-fa-ban-7911d14e": ("0ff1ef43d359", 65, "J1", "R12"),
    "li-chuang-kai-fa-ban-li-chuang-esp32s3r8n8-kai-fa-ban-cb10375b": ("81759a3acf5b", 53, "J2", "R13"),
    "li-chuang-kai-fa-ban-li-chuang-gd32vw553-kai-fa-ban-b5ed9764": ("aa641492fa97", 50, "J4", "R13"),
    "li-chuang-kai-fa-ban-pid-jian-yi-ru-men-tao-jian-c5ce5a79--V1.0.0-94ebd22e":
        ("3ca259548bd7", 14, "U10", "H2"),
    "lr1388-esp32-camera-photometer-7ba3d19c--2感光板-63aa5ecd": ("f03e0464d55d", 36, "USBC1", "R9"),
    "lyg0927-cyberwand-stm32-convolutional-ne-ac55f5c2--Main-52a399c5": ("f3a8fcbc373f", 60, "Slot", "R10"),
    "realtix-sakura_nfc-d968f2f9--执行器-18c91766": ("1bc6bdf652f2", 17, "U2", "LED1"),
    "travellium-12v-desktop-power-supply-cc8d8467--Z作废留念-MK9218-29253164": ("7c4ce6dec8f6", 43, "L1", "R5"),
    "trumpx-ipadrescreen-ipad34-c6ab9950--IPAD3_4_TouchScreen_Board-56329e46":
        ("5cda38b5d1c9", 21, "U1", "LED1"),
    "verysunshine-34e06a1e8bfa4ca7a126286b764d3f7b-42c390bd": ("b15c7977fe52", 39, "U8", "R23"),
    "wzh2814685-voltpen_v0-5-3f4d7379--主控板-5bd81944": ("8fc7beddd31b", 38, "TFT-1", "R4"),
}

# ---- 20 板聚合指标金表（probe147 精确值；对照：旧"纯面积降序" 13.0% / 7.8 / 79.1% / 0.026） ----
AGG_GOLDEN = dict(fog_a=0.072684, delay_med=3.85, a25=0.769121, big_pos=0.041343, pw_pos=0.382520)
WEIGHT_GOLDEN = dict(area=0.7, conn=0.3)


def _real_pcb_model():
    if _PCB_MODEL is None or not getattr(_PCB_MODEL, "__file__", None):
        pytest.fail("检测到 Mock 污染：本文件必须**单独运行**（或不被 tests/ 根目录的 mock 测试污染）")
    return _PCB_MODEL


@pytest.fixture(scope="session")
def models_by_tag(boards_with_models):
    """`tag -> model`（tag = 板目录名）。"""
    out = {}
    for path, _board, model, _raw in boards_with_models:
        out[os.path.basename(os.path.dirname(path))] = model
    return out


def _id_of(tag):
    return tag if len(tag) <= 40 else tag[:37] + "..."


@pytest.mark.parametrize("tag", sorted(ORDER_GOLDEN), ids=_id_of)
def test_order_golden(tag, models_by_tag):
    """A：逐板顺序金表（SHA1 + 件数 + 首末件）。"""
    model = models_by_tag[tag]
    order = op.plan_order(model, w_area=WEIGHT_GOLDEN["area"])
    want_hash, want_n, want_first, want_last = ORDER_GOLDEN[tag]
    got_hash = hashlib.sha1("|".join(order).encode("utf-8")).hexdigest()[:12]
    assert len(order) == want_n, f"{tag}：件数 {len(order)} ≠ 金表 {want_n}"
    assert (order[0], order[-1]) == (want_first, want_last), \
        f"{tag}：首/末件 {(order[0], order[-1])} ≠ 金表 {(want_first, want_last)}"
    assert got_hash == want_hash, \
        f"{tag}：顺序变了（hash {got_hash} ≠ {want_hash}）——若是有意改动，请重跑 probe147 更新金表"


@pytest.mark.parametrize("tag", sorted(ORDER_GOLDEN), ids=_id_of)
def test_order_deterministic(tag, models_by_tag):
    """B：打乱元件输入顺序 5 次，结果必须逐位一致（防 str 哈希随机化/迭代顺序依赖）。"""
    model = models_by_tag[tag]
    base = op.plan_order(model, w_area=WEIGHT_GOLDEN["area"])
    for seed in range(5):
        comps = list(model.footprints)
        random.Random(seed).shuffle(comps)
        assert op.plan_order(model, comps, w_area=WEIGHT_GOLDEN["area"]) == base, \
            f"{tag}：打乱输入顺序后顺序变了（seed={seed}）——tie-break 可能不是全序"


def test_order_metrics_aggregate(models_by_tag):
    """C：20 板聚合指标 ≈ 金表（迷茫A / 首→次步距 / 前25%步面积 / 大件位置 / 电源件位置）。"""
    acc = defaultdict(list)
    for tag, model in models_by_tag.items():
        if tag not in ORDER_GOLDEN:
            continue
        m = op.order_metrics(model, op.plan_order(model, w_area=WEIGHT_GOLDEN["area"]))
        for k in ("fog_a", "delay_med", "a25", "big_pos", "pw_pos"):
            acc[k].append(m[k])
    assert len(acc["fog_a"]) == len(ORDER_GOLDEN), "参与统计的板数不对"

    got = dict(fog_a=float(np.mean(acc["fog_a"])),
               delay_med=float(np.mean(acc["delay_med"])),
               a25=float(np.mean(acc["a25"])),
               big_pos=float(np.mean(acc["big_pos"])),
               pw_pos=float(np.nanmean(acc["pw_pos"])))
    for k, want in AGG_GOLDEN.items():
        assert got[k] == pytest.approx(want, abs=1e-3), \
            f"{k} 变了：{got[k]:.4f} ≠ 金表 {want}（全套 {got}）"
    # 与旧规则（纯面积降序）对比：迷茫与步距应显著变好，大件先放不应明显变差
    assert got["fog_a"] < 0.10, "迷茫步占比应低于 10%（旧规则 13.0%）"
    assert got["delay_med"] < 5.0, "首→次步距应低于 5 步（旧规则 7.8）"
    assert got["a25"] > 0.75, "前 25% 步已放面积不应低于 75%（旧规则 79.1%）"
    assert got["big_pos"] < 0.06, "前 10% 大件应仍在最前面 6% 内（旧规则 0.026）"
