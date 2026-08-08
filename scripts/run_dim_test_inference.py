r"""固定维度测试 — 推理脚本。

功能：
  1. 加载训练保存的 _best.pt 权重
  2. 完整放置全部 movable 元件（无早停）
  3. 导出最终布局为 .kicad_pcb 文件

用法：
  D:\KiCad\bin\python.exe scripts\run_dim_test_inference.py
  D:\KiCad\bin\python.exe scripts\run_dim_test_inference.py --checkpoint result/xxx_best.pt --output result/my_layout.kicad_pcb
"""

import os, sys, json, argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

os.environ.setdefault("KICAD_PATH", r"D:\KiCad")

from env.pcb_layout_env import Config as _Config  # noqa: E402 — 必须在 KICAD_PATH 之后


def main():
    parser = argparse.ArgumentParser(description="固定维度推理")
    parser.add_argument("--pcb", type=str,
                        default=r"kicad-native-batch\ran-pang-airlink-86231b34--开关小板-883285b3\ran-pang-airlink-86231b34--开关小板-883285b3.kicad_pcb",
                        help="PCB 文件路径")
    parser.add_argument("--checkpoint", type=str,
                        default=r"result\ran-pang-airlink-86231b34--开关小板-883285b3_best.pt",
                        help="训练保存的权重文件")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 .kicad_pcb 路径（默认 result/best_layout/xxx_inference.kicad_pcb）")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ---- 导入模块 ----
    from env.pcb_layout_env import PCBLayoutEnv
    from dqn import DQNAgent

    device = torch.device(args.device)

    # ---- 1. 创建环境 ----
    print(f"Loading PCB: {args.pcb}")
    env = PCBLayoutEnv(args.pcb)

    # ---- 2. 创建 Agent 并加载权重 ----
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = DQNAgent(
        n_movable=_Config.MAX_MOVABLE,
        grid_w=_Config.GRID_W,
        grid_h=_Config.GRID_H,
        device=device,
    )
    agent.load(args.checkpoint)
    agent.q_net.eval()

    # ---- 3. 推理：逐个放置全部元件 ----
    env.reset()
    env._reset_pending()
    done = False
    step_count = 0

    print(f"\n开始推理: N_movable={env.N_movable}, grid={env.grid_width}x{env.grid_height}")
    print(f"固定维度: MAX_MOVABLE={_Config.MAX_MOVABLE}, GRID_W={_Config.GRID_W}, GRID_H={_Config.GRID_H}")

    while not done:
        # 5 步自回归
        for step_id in range(5):
            mask = env._build_mask_for_step(step_id)
            obs = env._get_obs_with_pending(step_id)
            action = agent.select_action(obs, mask, step_id, epsilon=0.0)
            env._update_pending(action, step_id)

        # 组装 → clamp 兜底 → step
        full_action = env._assemble_action()
        full_action = _clamp_action(full_action, env)
        obs_next, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated
        step_count += 1

        placed = info.get("placed_count", 0)
        print(f"  Step {step_count:3d}: reward={reward:8.2f}, placed={placed}/{env.N_movable}"
              f"  {'[完成]' if terminated else ''}")

        env._reset_pending()

    # ---- 4. 导出 ----
    if args.output is None:
        os.makedirs("result/best_layout", exist_ok=True)
        args.output = "result/best_layout/ran-pang-airlink-86231b34--开关小板-883285b3_inference.kicad_pcb"

    # 修正：确保所有组件有有效 bbox
    for comp in env.movable_list:
        if comp.is_placed and (comp.bbox == (0.0, 0.0, 0.0, 0.0) or any(v == 0.0 for v in comp.bbox)):
            # 无效 bbox，给一个默认位置
            comp.bbox = (env.valid_area[0], env.valid_area[1],
                         env.valid_area[0] + comp.footprint_w,
                         env.valid_area[1] + comp.footprint_h)

    ok = env.kicad.export_layout(args.output, env.kicad.components)
    if ok:
        print(f"\n布局导出成功: {args.output}")
    else:
        print("\n导出失败")

    # ---- 5. 汇总 ----
    print(f"\n===== 推理结果 =====")
    print(f"总步数: {step_count}")
    print(f"已放置: {len(env.placed_list)} / {env.N_movable}")
    for comp in env.movable_list:
        if comp.is_placed:
            x1, y1, x2, y2 = comp.bbox
            print(f"  {comp.id:12s} ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"
                  f" rot={comp.rotation*90}° layer={'TOP' if comp.layer==0 else 'BOT'}")

    env.close()


def _clamp_action(action, env):
    """Clamp 兜底：将动作坐标限制在 valid_area 内（复用统一实现）。

    统一使用 PCBUtils.clamp_to_board 保持 AABB 宽高不变，仅平移，
    避免直接 min/max 截断导致元件尺寸变化触发 AABB 校验失败。
    """
    from scripts.run_inference import _clamp_action_to_board
    return _clamp_action_to_board(env, *action)


if __name__ == "__main__":
    main()
