"""pytest 共享 fixtures：Mock KiCad 注入 + 合成 env。

各测试文件可直接使用 `patched_env` / `patched_env_from_json` fixture，
无需重复 mock 注入代码。

注意：Mock 注入从模块级别移到 fixture 级别（autouse），以便
tests/kicad/ 下的真实 KiCad 测试可以独立运行而不受 Mock 影响。
"""

import sys
import os
from unittest import mock

import pytest


# ---- Mock KiCad 注入（fixture 级别，按需注入） ----
_global_mock_injected = False


def _inject_mock_kicad():
    """注入 MockKiCadUtils 到 sys.modules。

    幂等操作：多次调用不会重复注入。
    """
    global _global_mock_injected
    if _global_mock_injected:
        return
    _global_mock_injected = True

    _m = mock.MagicMock()
    sys.modules["utils.kicad_utils"] = _m

    from env.mock_pcb import MockKiCadUtils, MockComponent, MockNet  # noqa: E402
    import utils  # noqa: E402
    utils.kicad_utils = sys.modules["utils.kicad_utils"]
    utils.kicad_utils.KiCadUtils = MockKiCadUtils
    utils.kicad_utils.Component = MockComponent
    utils.kicad_utils.Net = MockNet


@pytest.fixture(autouse=True)
def _mock_kicad_for_tests(request):
    """自动为所有非真实 KiCad 测试注入 Mock。

    跳过 tests/kicad/ 目录下的测试（标记为 require_kicad 或路径包含 kicad/）。
    """
    node_path = str(request.node.fspath)
    if "kicad" in node_path.replace("\\", "/").split("/tests/")[-1].split("/")[0]:
        return  # kicad 目录下的测试不使用 Mock
    _inject_mock_kicad()


@pytest.fixture
def patched_env():
    """合成 PCB env（5 movable + 2 fixed，100×80mm）。"""
    _inject_mock_kicad()
    from env.mock_pcb import create_patched_env  # noqa: E402
    env = create_patched_env()
    env.reset()
    return env


@pytest.fixture
def patched_env_from_json():
    """从 pcb_001.json 创建的 env（5 movable + 2 fixed，120×80mm）。"""
    _inject_mock_kicad()
    from env.mock_pcb import create_env_from_json  # noqa: E402
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "模拟", "pcb_001.json"
    )
    env = create_env_from_json(json_path)
    env.reset()
    return env
