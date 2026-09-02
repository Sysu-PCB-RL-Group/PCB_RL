此目录存放 PCB RL 项目的训练/推理数据与生成产物。

## 目录结构
- kicad-native-batch/   立创 EDA 导出的真实 KiCad 项目（.kicad_pcb），供训练循环扫描。
                        env/pcb_layout_env.py 的 get_all_files() 会递归扫描此目录。
- expert_transitions/   专家轨迹生成产物：
                          * <board>_expert.json     量化后的 transition 缓存（训练注入用）
                          * <board>_expert_raw.json  raw 轨迹缓存（含 shift / base / 变异类型 /
                                                    拆分前完整 reward，供 analyze_expert.py 审查）
- README.txt / .gitkeep 本说明与空目录占位文件。

## 当前状态（2026-08-24）
- 唯一训练板：开关小板
  （ran-pang-airlink-86231b34--开关小板-883285b3，FP=6, Net=3, grid 36×36）。
- 自 2026-08-01 起统一使用真实 KiCad 数据，不再用 env/mock_pcb.py 合成数据。
- 专家轨迹生成：python generate_expert_only.py --board 开关小板
- 轨迹质量审查：python analyze_expert.py --board airlink
