"""LoadBoard 预检（utils/kicad_utils/board_preflight.py）的回归测试。

背景：pcbnew.LoadBoard 在文本 `(face "…")` 无法解析到已装字体时会**无界阻塞**
（实测某板阻塞 168 分钟、CPU≈0%），线程与 signal 都打不断，会让训练永久挂死。
预检用独立子进程 + 硬超时把这类板挡住。

本文件大部分用例**不需要 pcbnew**：子进程调用被 mock 掉，
因此在 conda python3.12 环境下也能真实执行（不会被 conftest 跳过）。
只有标了 require_kicad 的用例才需要真实 KiCad 环境。
"""

import hashlib
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.kicad_utils import board_preflight as bp   # noqa: E402


# ---------------------------------------------------------------- 工具

def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class _FakeProc:
    """冒充 subprocess.Popen 的返回值。

    hang=True 时真的睡一会儿再抛 TimeoutExpired —— 真 Popen 的
    communicate(timeout=…) 是真等满才抛的，不睡的话 ProbeResult.seconds
    会恒为 0，测不出计时是否接上了。
    """

    def __init__(self, returncode=0, stdout="", stderr="", hang=False,
                 hang_seconds=0.05):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self._hang_seconds = hang_seconds
        self.killed = False

    def communicate(self, timeout=None, **kwargs):
        if self._hang and not self.killed:
            time.sleep(self._hang_seconds)
            raise subprocess.TimeoutExpired(cmd="pcbnew", timeout=timeout)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """清空进程内缓存 + 保证选得到解释器，避免用例互相污染。

    解释器钉成 sys.executable 而不是字符串 "python"：mock 用例会替换掉
    Popen，这个值根本用不到；而 require_kicad 用例必须拿到真的 KiCad
    解释器，写 "python" 会走 PATH 撞到 conda 3.12（import pcbnew 必失败）。
    """
    bp._cache.clear()
    monkeypatch.setattr(bp, "_child_interpreter", lambda: sys.executable)
    yield
    bp._cache.clear()


@pytest.fixture
def board_file(tmp_path):
    """一个带 (face ...) 声明的伪板文件。"""
    p = tmp_path / "Fake.kicad_pcb"
    p.write_text(
        '(kicad_pcb\n'
        '  (version 20260206)\n'
        '  (gr_text "hi"\n'
        '    (at 1 2 0)\n'
        '    (effects\n'
        '      (font\n'
        '        (face "得意黑 斜体")\n'
        '        (size 1 1)\n'
        '      )\n'
        '    )\n'
        '  )\n'
        ')\n',
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------- 状态判定

def test_timeout_is_flagged_and_child_killed(monkeypatch, board_file):
    """卡住的子进程必须被强杀，并报成 timeout 而不是普通失败。"""
    procs = []

    def fake_popen(cmd, **kwargs):
        procs.append(_FakeProc(hang=True))
        return procs[-1]

    monkeypatch.setattr(bp.subprocess, "Popen", fake_popen)

    r = bp.probe_board(board_file, timeout=1.0, use_cache=False)

    assert r.status == bp.PROBE_TIMEOUT
    assert r.ok is False
    assert r.seconds >= 0.04, "耗时没被计时"
    assert len(procs) == 1
    assert procs[0].killed, "超时后必须 kill 子进程，否则会泄漏挂死进程"


def test_ok_path(monkeypatch, board_file):
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(returncode=0))
    r = bp.probe_board(board_file, timeout=1.0, use_cache=False)
    assert r.status == bp.PROBE_OK and r.ok is True


def test_none_return_is_error(monkeypatch, board_file):
    """LoadBoard 返回 None（文件损坏）与超时是不同的失败类型。"""
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(returncode=3))
    r = bp.probe_board(board_file, timeout=1.0, use_cache=False)
    assert r.status == bp.PROBE_ERROR
    assert r.ok is False
    assert "None" in r.detail


def test_child_error_reports_last_stderr_line(monkeypatch, board_file):
    monkeypatch.setattr(
        bp.subprocess, "Popen",
        lambda cmd, **kw: _FakeProc(returncode=1, stderr="a\nboom\n"))
    r = bp.probe_board(board_file, timeout=1.0, use_cache=False)
    assert r.status == bp.PROBE_ERROR
    assert r.detail == "boom"


def test_missing_file_is_error_not_crash(tmp_path):
    r = bp.probe_board(str(tmp_path / "nope.kicad_pcb"), timeout=1.0)
    assert r.status == bp.PROBE_ERROR
    assert "不可读" in r.detail


def test_no_interpreter_is_not_blocking(monkeypatch, board_file):
    """找不到解释器时放行（unavailable），不能把预检变成新的失败来源。"""
    monkeypatch.setattr(bp, "_child_interpreter", lambda: None)
    r = bp.probe_board(board_file, timeout=1.0, use_cache=False)
    assert r.status == bp.PROBE_UNAVAILABLE
    assert r.ok is True


def test_skip_line_format():
    """日志行格式：给运维 grep 用（板名不带 .kicad_pcb 扩展名）。"""
    r = bp.ProbeResult("/x/y/Button-cbd1bfb1.kicad_pcb",
                       bp.PROBE_TIMEOUT, 15.0, timeout=15.0)
    assert bp.format_skip_line(r) == "[SKIP] Button-cbd1bfb1：LoadBoard超过15秒"


def test_reason_reports_budget_not_measured_time(monkeypatch, board_file):
    """日志里的秒数必须是【配置的超时预算】，不是实测耗时。

    实测值受 kill 时机与系统负载影响，曾出现 "LoadBoard超过0秒" 这种
    完全误导的日志（假 Popen 瞬间抛超时时就会这样）。
    """
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(hang=True,
                                                    hang_seconds=0.01))
    r = bp.probe_board(board_file, timeout=7.0, use_cache=False)

    assert r.status == bp.PROBE_TIMEOUT
    assert r.seconds < 1.0, "前提：实测耗时远小于预算"
    assert "超过7秒" in r.reason
    assert "超过0秒" not in r.reason


# ---------------------------------------------------------------- 缓存

def test_cache_avoids_reprobe(monkeypatch, board_file):
    """同一文件第二次预检不得再付一次超时代价。"""
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc(returncode=0)

    monkeypatch.setattr(bp.subprocess, "Popen", fake_popen)

    bp.probe_board(board_file, timeout=1.0)
    bp.probe_board(board_file, timeout=1.0)
    assert len(calls) == 1


def test_cache_invalidated_when_file_changes(monkeypatch, board_file):
    calls = []
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: (calls.append(cmd), _FakeProc())[1])

    bp.probe_board(board_file, timeout=1.0)
    os.utime(board_file, (0, 0))            # 改 mtime
    bp.probe_board(board_file, timeout=1.0)
    assert len(calls) == 2


def test_use_cache_false_forces_reprobe(monkeypatch, board_file):
    calls = []
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: (calls.append(cmd), _FakeProc())[1])
    bp.probe_board(board_file, timeout=1.0)
    bp.probe_board(board_file, timeout=1.0, use_cache=False)
    assert len(calls) == 2


# ---------------------------------------------------------------- 开关 / 超时

def test_preflight_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PCB_PREFLIGHT", "0")
    assert bp.preflight_enabled() is False
    monkeypatch.setenv("PCB_PREFLIGHT", "1")
    assert bp.preflight_enabled() is True


def test_timeout_env_override(monkeypatch):
    monkeypatch.setenv("PCB_LOAD_TIMEOUT", "3.5")
    assert bp.load_timeout() == 3.5


def test_bad_timeout_env_falls_back(monkeypatch):
    monkeypatch.setenv("PCB_LOAD_TIMEOUT", "abc")
    assert bp.load_timeout() == bp.DEFAULT_TIMEOUT
    monkeypatch.setenv("PCB_LOAD_TIMEOUT", "-1")
    assert bp.load_timeout() == bp.DEFAULT_TIMEOUT


# ---------------------------------------------------------------- 字体修复

def test_normalize_fonts_removes_face_without_touching_source(board_file, tmp_path):
    """核心安全属性：只写副本，原文件一个字节都不许动。"""
    before = _md5(board_file)
    dst = str(tmp_path / "out.kicad_pcb")

    made = bp.normalize_fonts(board_file, dst)

    assert made == dst
    assert "(face " not in open(dst, encoding="utf-8").read()
    assert _md5(board_file) == before, "normalize_fonts 修改了源文件！"
    # 除字体声明外，其它内容必须原样保留
    src_txt = open(board_file, encoding="utf-8").read()
    dst_txt = open(dst, encoding="utf-8").read()
    assert '(gr_text "hi"' in dst_txt
    assert "(size 1 1)" in dst_txt
    assert len(dst_txt) < len(src_txt)


def test_normalize_fonts_noop_without_face(tmp_path):
    p = tmp_path / "plain.kicad_pcb"
    p.write_text("(kicad_pcb (version 1))\n", encoding="utf-8")
    assert bp.normalize_fonts(str(p), str(tmp_path / "o.kicad_pcb")) is None


def test_normalize_fonts_strips_all_faces(tmp_path):
    p = tmp_path / "multi.kicad_pcb"
    p.write_text('(kicad_pcb\n'
                 '  (gr_text "a"\n    (font\n      (face "X 1")\n    )\n  )\n'
                 '  (gr_text "b"\n    (font\n      (face "Y 2")\n    )\n  )\n)\n',
                 encoding="utf-8")
    dst = str(tmp_path / "o.kicad_pcb")
    bp.normalize_fonts(str(p), dst)
    txt = open(dst, encoding="utf-8").read()
    assert "(face " not in txt
    assert txt.count("(gr_text") == 2


def test_ensure_loadable_returns_original_when_ok(monkeypatch, board_file, tmp_path):
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(returncode=0))
    path, r = bp.ensure_loadable(board_file, timeout=1.0, work_dir=str(tmp_path))
    assert path == board_file and r.status == bp.PROBE_OK


def test_ensure_loadable_repairs_after_timeout(monkeypatch, board_file, tmp_path):
    """第一次超时 -> 产出修复副本 -> 复检通过 -> 返回副本路径。"""
    before = _md5(board_file)
    state = {"n": 0}

    def fake_popen(cmd, **kwargs):
        state["n"] += 1
        # 原始（带 face）的超时；修复副本（已去 face）的成功
        return _FakeProc(hang=(state["n"] == 1), returncode=0)

    monkeypatch.setattr(bp.subprocess, "Popen", fake_popen)

    path, r = bp.ensure_loadable(board_file, timeout=1.0, work_dir=str(tmp_path))

    assert r.status == bp.PROBE_OK
    assert path is not None and os.path.dirname(path) == str(tmp_path)
    assert os.path.basename(path).endswith("_normalized.kicad_pcb")
    assert "(face " not in open(path, encoding="utf-8").read()
    assert _md5(board_file) == before, "ensure_loadable 不能改动源文件"


def test_ensure_loadable_gives_up_if_error(monkeypatch, board_file, tmp_path):
    """非超时类失败不尝试修复（修复手段只针对字体导致的阻塞）。"""
    monkeypatch.setattr(bp.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(returncode=3))
    path, r = bp.ensure_loadable(board_file, timeout=1.0, work_dir=str(tmp_path))
    assert path is None and r.status == bp.PROBE_ERROR


def test_probe_boards_reports_each_bad_board(monkeypatch, tmp_path, capsys):
    """批量预检必须逐块打印 [SKIP] 行。"""
    good = tmp_path / "Good.kicad_pcb"
    bad = tmp_path / "Bad.kicad_pcb"
    good.write_text("(kicad_pcb)\n", encoding="utf-8")
    bad.write_text("(kicad_pcb)\n", encoding="utf-8")

    def fake_popen(cmd, **kwargs):
        return _FakeProc(hang=("Bad" in cmd[-1]), returncode=0)

    monkeypatch.setattr(bp.subprocess, "Popen", fake_popen)

    res = bp.probe_boards([str(good), str(bad)], timeout=1.0)
    out = capsys.readouterr().out
    assert len(res) == 2
    assert "[SKIP] Bad：" in out
    assert "跳过 1 块" in out


# ---------------------------------------------------------------- 真实 KiCad（可选）

@pytest.mark.require_kicad
def test_real_good_board_probes_ok(tmp_path):
    """真实环境下正常板必须秒过（预检不能误伤好板）。"""
    batch = os.path.join(os.path.dirname(__file__), "..", "..",
                         "data", "kicad-native-batch")
    cands = []
    for root, _, files in os.walk(batch):
        for f in files:
            if f.endswith(".kicad_pcb") and "Button" not in f:
                cands.append(os.path.join(root, f))
    if not cands:
        pytest.skip("无测试板")
    cands.sort(key=os.path.getsize)

    r = bp.probe_board(cands[0], timeout=15.0)
    assert r.status == bp.PROBE_OK, r.reason
    assert r.seconds < 10.0


def _button_board():
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "kicad-native-batch",
        "htx-studio-project_kpilekig-42ad306f--Button-cbd1bfb1",
        "htx-studio-project_kpilekig-42ad306f--Button-cbd1bfb1.kicad_pcb")


@pytest.mark.require_kicad
def test_real_board_probe_is_bounded(monkeypatch):
    """预检的真实契约：**绝不无界**（不是「必须挂死」）。

    这块板（引用了本机没装的字体 得意黑 斜体）曾稳定挂死 >168 分钟，
    但后来又能在 2.4s 内加载 —— 挂/不挂取决于 KiCad 侧的缓存状态，
    不可复现。所以这里**不能**断言「必须超时」，否则测试会随环境飘。

    真正要钉住的是需求本身：无论走哪个分支，都必须在超时预算内返回，
    且两种结果都有正确的语义。子进程被无限卡住的机制由 mock 用例
    test_timeout_is_flagged_and_child_killed 确定性覆盖。
    """
    src = _button_board()
    if not os.path.isfile(src):
        pytest.skip("Button 板不在数据集里")

    budget = 5.0
    monkeypatch.setenv("PCB_LOAD_TIMEOUT", str(int(budget)))
    r = bp.probe_board(src, timeout=budget, use_cache=False)

    assert r.status in (bp.PROBE_OK, bp.PROBE_TIMEOUT), r.status
    assert r.seconds <= budget + 10.0, f"预检无界：{r.seconds:.1f}s"
    if r.status == bp.PROBE_TIMEOUT:
        assert "LoadBoard超过5秒" in bp.format_skip_line(r)
    else:
        assert r.ok is True


@pytest.mark.require_kicad
def test_real_normalized_copy_loads(tmp_path):
    """确定性覆盖修复路径：去掉 (face ...) 的真实板副本必须能加载。

    与上一条不同，这条不依赖「原板此刻挂不挂」：去掉字体声明后 KiCad 回退
    到自带笔画字体，任何机器上都能加载。
    """
    src = _button_board()
    if not os.path.isfile(src):
        pytest.skip("Button 板不在数据集里")

    dst = str(tmp_path / "normalized.kicad_pcb")
    assert bp.normalize_fonts(src, dst) == dst
    assert "(face " not in open(dst, encoding="utf-8").read()

    r = bp.probe_board(dst, timeout=15.0, use_cache=False)
    assert r.status == bp.PROBE_OK, r.reason
    assert r.seconds < 10.0

    # 用例内导入：模块级导入会把 pcbnew 拖进所有环境（conda 3.12 下会打警告）
    from utils.kicad_utils.kicad_utils import KiCadUtils
    ki = KiCadUtils()
    assert ki.load_board(dst) is True
    assert ki.components, "修复副本加载后应能解析出元件"
