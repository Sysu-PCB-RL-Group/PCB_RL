"""PCB 文件可加载性预检：独立子进程 + 硬超时。

为什么必须用独立进程
--------------------
`pcbnew.LoadBoard()` 存在**无界阻塞**（不是变慢，是永不返回）：
实测 Button 板阻塞 **168 分钟**仍未返回，累计 CPU 仅 9.2s（≈0% 占用），
同目录另外 4 块板 0.010~0.013s。

阻塞发生在 C++ 扩展内部，此时 Python 解释器不可达：
`signal.alarm` / `threading` / `concurrent.futures` 的 timeout 都**打不断**
（timeout 只在 GIL 可获取时才可能触发）。只能靠进程级隔离 + 杀进程。
实测 `TerminateProcess` 有效 —— 外部 `timeout` 能稳定杀掉（退出码 124）。

根因（实测定位，见 verification_output/bisect_button_hang.py）
-----------------------------------------------------------
触发条件是 **文本的 `(face "…")` 字体声明无法完整解析到已装字体**，
与 CJK / render_cache / 文件体积都无关：

    变体                             结果
    原文件（face "得意黑 斜体"）      挂死
    删 render_cache，保留字体声明     挂死      ← render_cache 是红鲱鱼
    只删 (face ...)，保留 render_cache 0.029s   ← 决定性
    face "得意黑"（缺家族）           挂死
    face "NoSuchFontXYZ123"（ASCII）  挂死      ← 与 CJK 无关
    face "微软雅黑 斜体"（缺样式）     挂死
    face "微软雅黑"（可解析）          0.044s

即：**任一第三方板只要引用了本机没装的字体（缺家族或缺样式），加载就会挂死。**
板文件来自多个第三方项目，这个风险是普遍存在的。

同一文件 `kicad-cli pcb export svg` 能正常解析并提示
`找不到字体 '得意黑 斜体'; 用 '微软雅黑 Regular' 代替`，说明是 python
`LoadBoard` 路径特有的阻塞，不是文件损坏。

用法
----
    from utils.kicad_utils.board_preflight import probe_board, probe_boards

    r = probe_board("xxx.kicad_pcb")          # 15s 超时
    if not r.ok:
        print(f"[SKIP] {r.name}：{r.reason}")

超时可用环境变量 `PCB_LOAD_TIMEOUT` 覆盖（秒，默认 15）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# ---- 状态 ----
PROBE_OK = "ok"                  # 子进程加载成功
PROBE_TIMEOUT = "timeout"        # 超过 timeout 被杀（根因通常就是字体）
PROBE_ERROR = "error"            # 子进程报错 / 返回 None
PROBE_UNAVAILABLE = "unavailable"  # 找不到可用解释器，跳过预检（不阻断流程）

DEFAULT_TIMEOUT = 15.0

# 子进程要跑的最小程序：只做 LoadBoard，用退出码回报结果。
# 单独一份而不是复用 time_loadboard.py，是为了让 utils/ 不依赖
# verification_output/（那是验证脚本目录，不属于运行时代码）。
_CHILD_CODE = r"""
import os, sys
kp = os.environ.get("KICAD_PATH", "")
if kp:
    try:
        os.add_dll_directory(os.path.join(kp, "bin"))
    except (OSError, AttributeError):
        pass
    for p in (os.path.join(kp, "bin", "Lib", "site-packages"),
              os.path.join(kp, "bin")):
        if p not in sys.path:
            sys.path.insert(0, p)
import pcbnew
board = pcbnew.LoadBoard(sys.argv[1])
sys.exit(0 if board is not None else 3)
"""

# 进程内缓存：(abspath, mtime_ns, size) -> ProbeResult
_cache: Dict[tuple, "ProbeResult"] = {}


def preflight_enabled() -> bool:
    """预检总开关，`PCB_PREFLIGHT=0` 可关闭（回到逐板直接 LoadBoard 的老行为）。"""
    return os.environ.get("PCB_PREFLIGHT", "1").strip().lower() not in (
        "0", "false", "no", "off")


def load_timeout() -> float:
    """预检超时秒数，可用 PCB_LOAD_TIMEOUT 覆盖。"""
    raw = os.environ.get("PCB_LOAD_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        v = float(raw)
    except ValueError:
        print(f"[WARN] PCB_LOAD_TIMEOUT={raw!r} 不是数字，用默认 {DEFAULT_TIMEOUT}s")
        return DEFAULT_TIMEOUT
    return v if v > 0 else DEFAULT_TIMEOUT


@dataclass
class ProbeResult:
    """单块板的预检结果。

    seconds 是**实测**耗时，timeout 是本次使用的**超时预算**。
    日志里报的是 timeout（预算是确定的，实测会被 kill 时机影响，报出来会漂）。
    """

    path: str
    status: str
    seconds: float
    detail: str = ""
    timeout: float = 0.0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def ok(self) -> bool:
        """是否可继续加载。

        UNAVAILABLE（没找到解释器）算 ok：预检无法进行时不应阻断原有流程，
        交回给调用方按老路径处理。
        """
        return self.status in (PROBE_OK, PROBE_UNAVAILABLE)

    @property
    def reason(self) -> str:
        if self.status == PROBE_TIMEOUT:
            # 报超时预算而不是实测秒数：预算是配置值，稳定且可复现；
            # 实测值受 kill 时机/系统负载影响（曾出现 "超过0秒" 的误导日志）。
            budget = self.timeout if self.timeout > 0 else self.seconds
            return f"LoadBoard超过{budget:.0f}秒"
        if self.status == PROBE_ERROR:
            return f"LoadBoard失败: {self.detail}" if self.detail else "LoadBoard失败"
        return ""


def _child_interpreter() -> Optional[str]:
    """选一个能 import pcbnew 的解释器。

    优先 sys.executable：父进程能把 pcbnew 导进来（否则也不会走到这里），
    说明当前解释器就是对的（通常是 KiCad 自带 python 3.11）。
    父进程在 conda 3.12 下 pcbnew 导入失败时，回退到 KICAD_PATH 里的 python。
    """
    if _parent_has_pcbnew():
        return sys.executable

    kicad_path = os.environ.get("KICAD_PATH", "")
    if kicad_path:
        cand = os.path.join(kicad_path, "bin",
                            "python.exe" if os.name == "nt" else "python")
        if os.path.isfile(cand):
            return cand
    return None


def _parent_has_pcbnew() -> bool:
    try:
        import pcbnew  # noqa: F401
        return True
    except Exception:
        return False


def probe_board(path: str, timeout: Optional[float] = None,
                use_cache: bool = True) -> ProbeResult:
    """在独立子进程里试加载一块板，返回预检结果。

    Args:
        path: .kicad_pcb 路径
        timeout: 超时秒数，默认 load_timeout()（15s）
        use_cache: 命中 (路径, mtime, size) 缓存时直接返回

    Returns:
        ProbeResult；超时/失败时 result.ok 为 False，用 result.reason 取原因。
    """
    timeout = load_timeout() if timeout is None else float(timeout)
    path = os.path.abspath(path)

    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError as e:
        return ProbeResult(path, PROBE_ERROR, 0.0, f"文件不可读: {e}", timeout)

    if use_cache and key in _cache:
        return _cache[key]

    result = _run_probe(path, timeout)
    _cache[key] = result
    return result


def _run_probe(path: str, timeout: float) -> ProbeResult:
    interp = _child_interpreter()
    if interp is None:
        return ProbeResult(path, PROBE_UNAVAILABLE, 0.0,
                           "找不到可 import pcbnew 的解释器", timeout)

    cmd = [interp, "-c", _CHILD_CODE, path]
    env = dict(os.environ)

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError as e:
        return ProbeResult(path, PROBE_ERROR, 0.0, f"无法启动子进程: {e}", timeout)

    try:
        out, err = proc.communicate(timeout=timeout)
        dt = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        # 卡在 C++ 里的进程只能强杀；TerminateProcess 无需目标配合，实测有效
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        dt = time.perf_counter() - t0
        return ProbeResult(path, PROBE_TIMEOUT, dt,
                           "子进程已被强制终止（根因通常是字体无法解析）",
                           timeout)

    if proc.returncode == 0:
        return ProbeResult(path, PROBE_OK, dt, timeout=timeout)
    if proc.returncode == 3:
        return ProbeResult(path, PROBE_ERROR, dt, "LoadBoard 返回 None", timeout)
    tail = (err or out or "").strip().splitlines()
    return ProbeResult(path, PROBE_ERROR, dt,
                       tail[-1] if tail else f"退出码 {proc.returncode}", timeout)


def probe_boards(paths: Iterable[str], timeout: Optional[float] = None,
                 verbose: bool = True) -> Dict[str, ProbeResult]:
    """批量预检，返回 {abspath: ProbeResult}。

    串行执行：每块板本身很快（~0.03s），只有坏板才会吃满 timeout；
    并发会让"哪块板在耗时间"难以归因，训练启动阶段不值得。
    """
    results: Dict[str, ProbeResult] = {}
    for p in paths:
        ap = os.path.abspath(p)
        r = probe_board(ap, timeout=timeout)
        results[ap] = r
        if verbose and not r.ok:
            # 必须与 load_board 走同一个格式化函数，否则两条路径的日志
            # 一处带扩展名一处不带，grep 会漏
            print(format_skip_line(r))
    if verbose and results:
        bad = [r for r in results.values() if not r.ok]
        print(f"[PREFLIGHT] 预检 {len(results)} 块板，跳过 {len(bad)} 块")
    return results


def normalize_fonts(src: str, dst: str) -> Optional[str]:
    """把无法解析的 `(face "…")` 声明去掉，生成一份可加载的副本。

    **只写 dst，绝不原地覆盖 src。** 去掉 face 后 KiCad 回退到自带笔画字体
    （永远存在），因此副本必定可加载；代价是文字外观与原作者机器上的渲染不同。
    这是"让训练能跑完"的修复，不是"还原原作者渲染"的修复。

    只做文本级括号配对删除，不重新序列化整块板，避免引入其它差异。

    Returns:
        写成功返回 dst，失败返回 None。
    """
    import re

    with open(src, "r", encoding="utf-8") as f:
        text = f.read()

    n = len(re.findall(r"\(face ", text))
    if n == 0:
        return None

    # 连同前导缩进与行尾换行一起删，避免留下空行
    new = re.sub(r"[ \t]*\(face \"[^\"]*\"\)\r?\n", "", text)
    if new == text:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(new)
    print(f"[REPAIR] 去掉 {n} 处 (face ...) 声明: {os.path.basename(src)} "
          f"-> {os.path.basename(dst)} "
          f"({os.path.getsize(src)} -> {os.path.getsize(dst)} bytes)")
    return dst


def ensure_loadable(path: str, timeout: Optional[float] = None,
                    work_dir: Optional[str] = None) -> tuple:
    """预检 + 必要时修复，返回 (可加载路径, ProbeResult)。

    预检通过 → 原路径；超时 → 生成 `<name>_normalized.kicad_pcb` 副本并复检。
    修复副本只在内存目录/指定 work_dir 落地，不污染数据集。

    Returns:
        (path_to_load, result)。两者都不可用时返回 (None, result)。
    """
    r = probe_board(path, timeout=timeout)
    if r.ok:
        return path, r

    if r.status != PROBE_TIMEOUT:
        return None, r

    src = os.path.abspath(path)
    base = os.path.splitext(os.path.basename(src))[0]
    out_dir = work_dir or os.path.join(os.path.dirname(src), "_normalized")
    dst = os.path.join(out_dir, f"{base}_normalized.kicad_pcb")

    if normalize_fonts(src, dst) is None:
        return None, r

    r2 = probe_board(dst, timeout=timeout, use_cache=False)
    if r2.ok:
        print(f"[REPAIR] 修复副本可加载（{r2.seconds:.3f}s）：{os.path.basename(dst)}")
        return dst, r2

    print(f"[REPAIR] 修复副本仍不可加载：{os.path.basename(dst)} —— {r2.reason}")
    return None, r2


def format_skip_line(result: ProbeResult) -> str:
    """统一的跳过日志行，便于日志里 grep。"""
    return f"[SKIP] {os.path.splitext(result.name)[0]}：{result.reason}"
