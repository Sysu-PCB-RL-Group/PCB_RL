# -*- coding: utf-8 -*-
"""冒烟测试：确认 BOARD_MARGIN=0 后各板环境能初始化、Button 的 U1 能放下、grid 安全。"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KICAD_PATH", "D:\\KICAD")
from env.pcb_layout_env import PCBLayoutEnv, Config

Config.load_from_file("config/argument.txt")
Config.load_dims_from_file("config/admin.txt")
print(f"BOARD_MARGIN={Config.BOARD_MARGIN}  GRID_W={Config.GRID_W} GRID_H={Config.GRID_H}")

boards = sorted(glob.glob("data/kicad-native-batch/*/*.kicad_pcb"))
for pcb in boards:
    name = os.path.basename(os.path.dirname(pcb))
    try:
        env = PCBLayoutEnv(pcb)
        gw, gh = env.grid_width, env.grid_height
        # 检查最大可动元件所需格数是否 <= grid
        max_h = max(int(c.footprint_h / 1.0) for c in env.movable_list)
        max_w = max(int(c.footprint_w / 1.0) for c in env.movable_list)
        fit = "OK" if (gh >= max_h and gw >= max_w) else "NO"
        print(f"  {name[:42]:42s} movable={env.N_movable} grid={gw}x{gh} "
              f"需格w/h={max_w}x{max_h} 放置={fit} "
              f"(grid<=100: {'OK' if max(gw,gh)<=Config.GRID_W else 'OVER!'})")
        env.reset()
        env.close() if hasattr(env, "close") else None
    except Exception as e:
        print(f"  {name[:42]:42s} !! 失败: {e}")
