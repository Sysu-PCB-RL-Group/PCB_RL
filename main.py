# -*- coding: utf-8 -*-
"""项目统一入口（仓库根）：**一个 main 函数，用第一个参数选择「训练 / 推理」**（Owner 2026-09-24）。

为什么有它
----------
上一代三个入口（根目录 `main.py` / `run.py` / `run_inference.py`）已删除 —— 它们是
SPEC v2.1 / v2.2 的 **5 步动作栈**（`env/pcb_layout_env.py` + `dqn/`），与现行口径
（动作 =（旋转 k, cell i, j）、`L`/`G` = 4×200×200、违规被 mask 禁止）无关。
现行的训练 / 推理各自**已经是独立可跑的脚本**，本文件只把它们**串起来**。

设计（三条，别改坏）
--------------------
```text
① 模式 = **argv[0]**（`train` / `infer` / `help`），**其余参数原样透传**给对应脚本的 `main()`
② 参数权威**只在子脚本**（`train_dqn.py` / `infer_dqn.py` 的 argparse）⇒ 本文件**零 flag**，
   不会出现"同一个旋钮两处定义、改了一处忘另一处"
③ 退出码原样返回（两个子脚本的 `main(argv) -> int`）
```

用法（照抄可直接粘）
```text
& python main.py train                                     # 训练（默认 1000 局 → output/）
& python main.py train --episodes 5 --eval-every-episodes 5 --log-every 1   # 冒烟
& python main.py train --boards <板名>                     # 只在一两块板上跑
& python main.py infer                                     # 推理（output/best.pt，16 板）
& python main.py infer --ckpt output/ckpt_ep50.pt --boards <板名> --no-viz
& python main.py infer --net-boxes                         # 打开 HPWL 包围框
& python main.py help                                      # 本页
& python main.py train --help                              # 子命令的 --help（它自己打印）
```

⚠️ 训练/推理都是**长任务**，一律在**你自己的终端**跑（本机 conda 环境 `pcb_rl_311`）：
`& E:\\miniconda\\envs\\pcb_rl_311\\python.exe main.py train`
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

# 本文件就在仓库根 ⇒ 直接运行时 `sys.path[0]` 已经是它（这一行是幂等的）；
# 保留是为了 `import main`（从别处调用）时也能找到 `scripts/` 包。
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODES: Dict[str, str] = {
    "train": "训练（Double DQN，台账 §9.2；产物 → output/：train.log / log.jsonl / ckpt_epN.pt / best.pt）",
    "infer": "推理（纯贪心、mask = L；产物 → result/infer_<权重名>_ep<局号>/：日志 + summary.json + 图）",
}


def _usage() -> str:
    lines = ["项目统一入口 —— 用法：python main.py <模式> [该模式的参数…]", ""]
    lines += ["  %-6s %s" % (k, v) for k, v in MODES.items()]
    lines += ["  %-6s %s" % ("help", "打印本页"), ""]
    lines += ["各模式的**全部参数**由对应脚本自己定义，原样透传；`<模式> --help` 看它的选项。",
              "  train → scripts/train_dqn.py",
              "  infer → scripts/infer_dqn.py"]
    return "\n".join(lines)


def _runtime_error() -> Optional[str]:
    """预检解释器有没有 `torch`（**本机有三种 python**，这是最容易踩的坑）。

    训练与推理都靠它 ⇒ 用错解释器的症状是"莫名其妙 ImportError"，这里提前给出可执行的下一步。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return ("这个解释器没有 torch（多半是默认的 python 3.14 core）—— 训练/推理必须用 conda 环境：\n"
                "  & E:\\miniconda\\envs\\pcb_rl_311\\python.exe main.py <模式> …\n"
                "（⚠️ 不要往 KiCad 自带的裸 python 里装 torch）")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """分发：`argv[0]` 选模式，其余透传（`--help` / 无参 ⇒ 打印本页）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_usage())
        return 2
    mode, rest = args[0], args[1:]
    if mode in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if mode not in MODES:
        print("未知模式：%r\n\n%s" % (mode, _usage()), file=sys.stderr)
        return 2
    err = _runtime_error()
    if err is not None:                       # 解释器不对 ⇒ 直接说清怎么跑，别抛一堆 traceback
        print(err, file=sys.stderr)
        return 3
    # 惰性 import：`help` / 模式拼错时不必付 torch 的启动代价（两个子脚本顶层都 import torch）
    if mode == "train":
        from scripts import train_dqn
        return int(train_dqn.main(rest))
    from scripts import infer_dqn
    return int(infer_dqn.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
