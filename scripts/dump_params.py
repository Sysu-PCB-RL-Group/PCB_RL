# -*- coding: utf-8 -*-
"""参数一览的口子：打印 / 生成 / 校验 `PARAMS.md`（薄封装 `env/params.py`）。

用法

```text
& python scripts/dump_params.py            # 打印到终端（看"现在配置的是什么"）
& python scripts/dump_params.py --write    # 覆盖写 `PARAMS.md`（改了参数后跑一次）
& python scripts/dump_params.py --check    # 校验（不一致 ⇒ 退出码 1）
```

设计：真源仍是各模块里的常数（本脚本**不定义任何参数**）；`PARAMS.md` 是生成物。
详见 `env/params.py` 的模块说明与 `tests/kicad/test_params_single_source.py`。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env import params as P     # noqa: E402

if __name__ == "__main__":
    raise SystemExit(P.main())
