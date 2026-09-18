"""真实 KiCad 集成测试的共享 fixtures。

直接使用真实 pcbnew API（不经过 Mock）。
所有 fixture 在 KICAD_PATH 未设置时自动 skip。

注意：mock 测试（tests/ 根目录）先运行时会将 MockKiCadUtils 注入
sys.modules["utils.kicad_utils"]，进而污染 env.* 模块的导入缓存。
此处通过清除缓存模块和强制重载，确保真实 KiCad 代码加载。
"""

import os
import sys
import pytest


# ---- 清除 mock 测试污染的模块缓存，确保 env 模块加载真实 KiCadUtils ----
def _clean_mock_cache():
    """删除可能在 mock 测试中被污染缓存的模块，让真实 KiCad 测试重新加载。"""
    polluted = [k for k in list(sys.modules.keys()) if k.startswith("env.")]
    for key in polluted:
        del sys.modules[key]
    # 同时清除 kicad_utils（可能被 mock 注入）
    if "utils.kicad_utils" in sys.modules:
        del sys.modules["utils.kicad_utils"]
    # 清理 utils 上的 mock 属性引用
    import utils  # noqa: E402
    if hasattr(utils, "kicad_utils"):
        delattr(utils, "kicad_utils")


_clean_mock_cache()


# ---- KiCad 环境检查 ----
def _get_kicad_path():
    return os.environ.get("KICAD_PATH", "")


def _init_kicad():
    """在首次调用时初始化 KiCad Python 环境。"""
    kicad_path = _get_kicad_path()
    if not kicad_path or not os.path.isdir(kicad_path):
        return False

    try:
        os.add_dll_directory(os.path.join(kicad_path, "bin"))
        sys.path.insert(0, os.path.join(kicad_path, "bin", "Lib", "site-packages"))
        sys.path.insert(0, os.path.join(kicad_path, "bin"))
        return True
    except OSError:
        return False


def pytest_configure(config):
    """注册 kicad 自定义 mark。"""
    config.addinivalue_line("markers", "require_kicad: 需要真实 KiCad 环境的测试")


def pytest_runtest_setup(item):
    """在 kicad 标记的测试运行前检查 KiCad 环境。"""
    if item.get_closest_marker("require_kicad"):
        if not _get_kicad_path() or not os.path.isdir(_get_kicad_path()):
            pytest.skip("KICAD_PATH not set or does not exist")
        if not _init_kicad():
            pytest.skip("Failed to initialize KiCad Python environment")


# ---- 共享 fixture ----

@pytest.fixture(scope="session")
def kicad_pcb_path():
    """返回一个可用于测试的真实 .kicad_pcb 文件路径。

    优先使用最小的板子（减少测试耗时），找不到则 skip。
    """
    # 查找 data/kicad-native-batch 下的所有 .kicad_pcb
    batch_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kicad-native-batch")
    candidates = []
    for root, dirs, files in os.walk(batch_dir):
        for f in files:
            if f.endswith(".kicad_pcb") and not f.endswith("best_layout.kicad_pcb"):
                candidates.append(os.path.join(root, f))

    if not candidates:
        pytest.skip("No .kicad_pcb files found in data/kicad-native-batch/")

    # 优先选择最小的板子
    candidates.sort(key=lambda p: os.path.getsize(p))
    return os.path.abspath(candidates[0])


@pytest.fixture(scope="session")
def boards_with_models(all_kicad_pcb_paths):
    """加载全部真实板并构建转换数据模型（session 级，供 tests/kicad 下的模型测试共享）。

    返回 `[(path, board, model, raw_angles), ...]`：
      - `raw_angles[i]` = **加载瞬间**第 i 个（有 pad 的）footprint 的文件原始角（未被任何测试旋转污染）
      - 测试可临时旋转 footprint，但必须自负责恢复原姿态
    """
    from utils.kicad_utils import pcb_model  # noqa: F401  触发 bootstrap（DLL 路径 + wx 日志抑制）

    pcbnew = pcb_model.pcbnew
    out = []
    for path in all_kicad_pcb_paths:
        board = pcbnew.LoadBoard(path)
        raw_angles = [
            float(fp.GetOrientationDegrees())
            for fp in board.GetFootprints()
            if list(fp.Pads())
        ]
        out.append((path, board, pcb_model.build_model(board, path), raw_angles))
    return out


@pytest.fixture(scope="session")
def all_kicad_pcb_paths():
    """返回 data/kicad-native-batch 下**全部**真实 .kicad_pcb（默认 20 块）。

    与 `kicad_pcb_path`（只取最小一块）相对：用于需要"全量真实数据"的回归测试。
    可用环境变量 `PCB_RL_TEST_MAX_BOARDS` 限制数量（调试用），默认不限制。
    """
    batch_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kicad-native-batch")
    candidates = []
    for root, _dirs, files in os.walk(batch_dir):
        for f in files:
            if f.endswith(".kicad_pcb") and not f.endswith("best_layout.kicad_pcb"):
                candidates.append(os.path.abspath(os.path.join(root, f)))

    if not candidates:
        pytest.skip("No .kicad_pcb files found in data/kicad-native-batch/")

    candidates.sort()
    limit = os.environ.get("PCB_RL_TEST_MAX_BOARDS", "").strip()
    if limit.isdigit() and int(limit) > 0:
        candidates = candidates[: int(limit)]
    return candidates
