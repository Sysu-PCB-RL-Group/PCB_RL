"""pytest 共享 fixtures：Mock KiCad 注入 + 合成 env。

各测试文件可直接使用 `patched_env` / `patched_env_from_json` fixture，
无需重复 mock 注入代码。
"""

import sys
import os
from unittest import mock

import pytest

# ---- Mock KiCad 注入（必须在任何 import env 之前） ----
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

from env.mock_pcb import (
    create_patched_env, create_env_from_json,
    MockKiCadUtils, MockComponent, MockNet,
)
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet


@pytest.fixture
def patched_env():
    """合成 PCB env（5 movable + 2 fixed，100×80mm）。"""
    env = create_patched_env()
    env.reset()
    return env


@pytest.fixture
def patched_env_from_json():
    """从 pcb_001.json 创建的 env（5 movable + 2 fixed，120×80mm）。"""
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "模拟", "pcb_001.json"
    )
    env = create_env_from_json(json_path)
    env.reset()
    return env
