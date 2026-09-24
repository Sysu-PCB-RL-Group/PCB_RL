# PCB_RL

基于强化学习的 PCB 自动布局（Double DQN）。

> ⚠️ 本项目 **2026-09-22 起整体重构**：旧的 5 步动作栈（`env/pcb_layout_env.py` + `dqn/` +
> `config/argument.txt` + `main.py/run.py/run_inference.py` 那套）**已废弃并删除**。
> 本文以下内容全部是现行架构。

## 先读这三份

- **技术总览（新人从这里开始，约 300 行）**：[`docs/项目技术文档.md`](docs/项目技术文档.md)
- **当前参数一览（现在到底配的是什么）**：[`PARAMS.md`](PARAMS.md)
- **设计台账（唯一事实来源，2400+ 行，含每个数字的由来）**：[`docs/导入信息重构.md`](docs/导入信息重构.md)

（重构前的旧文档在 `docs/旧版结构文档/`，**已作废**，仅作历史对照。）

## 最快跑起来

```powershell
cd E:\PCB_AI\PCB_RL

& E:\miniconda\envs\pcb_rl_311\python.exe main.py train        # 训练（默认 1000 局 → output/，约 4.4 h）
& E:\miniconda\envs\pcb_rl_311\python.exe main.py infer        # 推理（output/best.pt，16 板 → result/）
& E:\miniconda\envs\pcb_rl_311\python.exe -m pytest tests -q   # 测试（340 passed / 13 skipped）
```

`main.py train|infer` 的参数**原样透传**给 `scripts/train_dqn.py` / `scripts/infer_dqn.py`
⇒ `python main.py train --help` 看全部选项。

**训练/推理不需要 KiCad**（几何走仓库内的 `data/model_cache/`）；只有"重建缓存"
（`scripts/dump_models.py`）和"读真实板的测试"（`pytest tests/kicad`）需要本机装 KiCad 10.0+ 并设 `KICAD_PATH`。
