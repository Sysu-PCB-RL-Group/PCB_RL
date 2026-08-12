# -*- coding: utf-8 -*-
"""Replace kicad-native-batch with 10 simple boards from source."""
import os, shutil

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"
SOURCE = r"E:\PCB_AI\环境重构\kicad-native-batch_extracted\kicad-native-batch"

# 10 simple boards to copy
BOARDS = [
    "ran-pang-airlink-86231b34--开关小板-883285b3",                         # FP=6, Net=3
    "dimsmary-micron-profilometer-micro-profil-51c806c0--Connector-3e5024e3",  # FP=8, Net=3
    "honglin-first_crt_smartphone-e4cb491d--灯板2-3cba64b3",                  # FP=6, Net=3
    "htx-studio-project_kpilekig-42ad306f--Button-cbd1bfb1",                  # FP=4, Net=3
    "htx-studio-project_kpilekig-42ad306f--MotorLimit-301a4056",               # FP=6, Net=4
    "lovelessing-d133-mobile-phone-development-bo-91bf7213--按键侧板V0.1-c790c8d4",  # FP=2, Net=3
    "lovelessing-d133-mobile-phone-development-bo-91bf7213--按键侧板V0.2-152707d2",  # FP=6, Net=4
    "tuyaopen-graffiti-t5pocket-pocket-machine-c240388e--2.-Pogo-Pin-4P-FPC-5c4f009b",  # FP=2, Net=4
    "double_mumu-zhi-neng-yu-yin-jia-re-bei-dian-9be43d66--加热底板-6d1a9314",  # FP=2, Net=2
    "fei1554973967-sw3538-dff8db20--后面板-3d1a59a5",                        # FP=4, Net=1
]

# Step 1: Clear target
print("=== Clearing target ===")
for name in os.listdir(TARGET):
    path = os.path.join(TARGET, name)
    if os.path.isdir(path):
        print(f"  Removing: {name}")
        shutil.rmtree(path)

# Step 2: Copy from source
print("\n=== Copying boards ===")
ok, fail = 0, 0
for name in BOARDS:
    src = os.path.join(SOURCE, name)
    dst = os.path.join(TARGET, name)
    if not os.path.isdir(src):
        print(f"  MISSING: {name}")
        fail += 1
        continue
    shutil.copytree(src, dst)
    print(f"  Copied: {name}")
    ok += 1

print(f"\nDone: {ok} copied, {fail} failed")
