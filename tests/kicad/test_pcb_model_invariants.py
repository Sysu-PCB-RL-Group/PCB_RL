"""转换数据模型的**不变量测试**（真实 20 板；与 `test_pcb_model_positions.py` 分工）

本文件覆盖"非位置/角度"的模型契约（位置链路见另一个文件）：
  1. `is_power_net` 单点定义（HPWL 排除电源/地网的口径不得分叉）
  2. `pads_by_key` **不丢件**（同封装同号多 pad 必须全部保留）
  3. 禁布区构成 = 板轮廓开槽 + 零 pad 封装 hull；`slots` 与 `keepouts` 是同一批对象
  4. `nets` 不可变（frozen 快照名副其实）
  5. `pads` 展平缓存完整

设计原则：**能失败** —— 每条断言都给出"模型值 vs 独立算出/实测值"。
"""
import os

import pytest

from utils.kicad_utils import pcb_model  # noqa: F401  触发 bootstrap（DLL + wx 日志抑制）

pcbnew = pytest.importorskip("pcbnew", reason="需要 KiCad 环境（KICAD_PATH）")

# 电源网判定的金表（测试内独立给出，不依赖被测实现）
# 2026-09-17 修补后：新增"电压名尾 + 轨子串"判据（3V3 / P60V / PGND / MOTOR-VCC…），
# 并明确排除"使能 / 参考 / 检测"类信号（_EN / _REF / PWRC…）。
POWER_NET_GOLDEN = (
    # —— 旧判据命中（保持不变）
    ("GND", True), ("VCC", True), ("VDD", True), ("+3V3", True), ("+3.3V", True),
    ("VIN", True), ("VBUS", True), ("AGND", True),
    # —— 修补后新增命中（真实供电轨；probe139 实测：10 条 / 264 pad 曾被漏判）
    ("3V3", True), ("3.3V", True), ("P60V", True), ("P12V", True), ("1V1", True),
    ("24V", True), ("USB_VBUS", True), ("MOTOR-VCC", True), ("A_3V3", True), ("PGND", True),
    # —— 必须仍判为信号（不得被"电压名 / 轨子串"误伤）
    ("SW", False), ("KEY_L", False), ("SPI2_CS", False), ("PB0_CUS", False), ("OUT+", False),
    ("IN", False), ("SDA", False), ("3V3_EN", False), ("ADC_VREF", False),
    ("DCDC_PWR_EN", False), ("JDY_PWRC", False),
)


@pytest.mark.require_kicad
class TestModelInvariants:

    def test_power_net_single_source(self, boards_with_models):
        """两份实现（legacy `Net.is_power_net` 与 `pcb_model.is_power_net`）必须同值。"""
        from utils.kicad_utils.kicad_utils import Net as LegacyNet

        # (a) 金表：被测实现必须符合"独立给出的"期望
        bad = [
            f"{n} → {pcb_model.is_power_net(n)}，期望 {want}"
            for n, want in POWER_NET_GOLDEN
            if pcb_model.is_power_net(n) != want
        ]
        assert not bad, "电源网判定与金表不符：\n   " + "\n   ".join(bad)

        # (b) 一致性：数据集里全部 net 名，两份实现必须逐一同值（防"两份词表漂移"）
        n = 0
        for _path, _board, model, _raw in boards_with_models:
            for name in model.nets:
                assert pcb_model.is_power_net(name) == LegacyNet.is_power_net(name), (
                    f"net {name!r}：两处判定不一致 "
                    f"（pcb_model={pcb_model.is_power_net(name)}, legacy={LegacyNet.is_power_net(name)}）"
                )
                n += 1
        print(f"\n[电源网] 金表 {len(POWER_NET_GOLDEN)} 项 + 数据集 {n} 个 net 名，两份实现一致")

    def test_pads_by_key_no_loss(self, boards_with_models):
        """`pads_by_key` 必须保留同号多 pad 的**全部**实例（不得静默取第一片）。"""
        total = 0
        pid_h1_seen = None
        for path, _board, model, _raw in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            by_key = model.pads_by_key
            kept = sum(len(v) for v in by_key.values())
            assert kept == len(model.pads), (
                f"{tag}：pads_by_key 丢件 —— 索引内 {kept} 个 pad ≠ 模型 {len(model.pads)} 个"
            )
            total += kept
            if "pid-jian-yi" in tag:
                # §1.1 实测：PID 板 `H1`（m3 铜柱）在一个封装内有 7 个同号 `pad "1"`
                pid_h1_seen = len(by_key.get(("H1", "1"), ()))
        assert total == 2648, f"pad 总数异常：{total}（期望 2648）"
        assert pid_h1_seen == 7, (
            f"PID 板 H1 的同号 pad 数 = {pid_h1_seen}（期望 7，§1.1 实测）"
            f" ⇒ 索引丢件或数据异常"
        )
        print(f"\n[pads_by_key] 索引内 {total} 个 pad；PID H1 号「1」保留 {pid_h1_seen} 片（同号多 pad 完整）")

    def test_keepouts_are_slots_plus_motor_hulls(self, boards_with_models):
        """禁布区构成 = 板轮廓开槽 + 零 pad 封装 hull；`slots` 与 `keepouts` 必须是同一批对象。"""
        for path, _board, model, _raw in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            outline = model.board_outline
            assert len(model.keepouts) == len(outline.slots) + len(model.zero_pad_footprints), (
                f"{tag}：禁布区构成异常 keepouts={len(model.keepouts)} "
                f"slots={len(outline.slots)} 零pad封装={len(model.zero_pad_footprints)}"
            )
            # 同一批对象（用 id 判等，防"复制一份导致重复计数"）
            keep_ids = {id(k) for k in model.keepouts}
            missing = [k.kind for k in outline.slots if id(k) not in keep_ids]
            assert not missing, (
                f"{tag}：BoardOutline.slots 里 {len(missing)} 个开槽不在 PcbModel.keepouts 中"
                f"（同一批对象的约定被破坏 ⇒ 判定可能重复计数）"
            )

    def test_nets_are_immutable(self, boards_with_models):
        """`nets` 必须冻结：`frozen=True` 不能只挡字段赋值、却放任容器被就地修改。"""
        _path, _board, model, _raw = boards_with_models[0]
        with pytest.raises(TypeError):
            model.nets["__injected__"] = None          # type: ignore[index]
        # 字段本身也不可赋值
        with pytest.raises(Exception):
            model.pads = ()                            # type: ignore[misc]

    def test_pads_cache_complete(self, boards_with_models):
        """`pads` 必须是完整展平缓存（不重算、不漏件）。"""
        total = 0
        for path, _board, model, _raw in boards_with_models:
            tag = os.path.basename(os.path.dirname(path))
            expect = sum(len(f.pads) for f in model.footprints)
            assert isinstance(model.pads, tuple), f"{tag}：pads 不是 tuple"
            assert len(model.pads) == expect, (
                f"{tag}：pads 缓存不完整 {len(model.pads)} ≠ 各 footprint 之和 {expect}"
            )
            total += len(model.pads)
        assert total == 2648, f"pad 总数异常：{total}（期望 2648）"
