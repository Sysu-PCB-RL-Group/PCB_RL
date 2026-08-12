"""删除超出 21x21 grid 的 5 个板子目录"""
import os, shutil

TARGET = r"e:\PCB_AI\PCB_RL\data\kicad-native-batch"

# 保留的 5 个:
#   ran-pang-airlink-86231b34--开关小板-883285b3         (FP=6, 6x3)
#   dimsmary-micron-profilometer-micro-profil-51c806c0--Connector-3e5024e3  (FP=8, 13x13)
#   honglin-first_crt_smartphone-e4cb491d--灯板2-3cba64b3 (FP=6, 12x10)
#   htx-studio-project_kpilekig-42ad306f--Button-cbd1bfb1 (FP=4, 16x10)
#   lovelessing-d133-mobile-phone-development-bo-91bf7213--按键侧板V0.1-c790c8d4 (FP=2, 2x16)

# 删除的 5 个（超出 21x21 grid）:
DELETE_DIRS = [
    "double_mumu-zhi-neng-yu-yin-jia-re-bei-dian-9be43d66--加热底板-6d1a9314",      # 42x40
    "fei1554973967-sw3538-dff8db20--后面板-3d1a59a5",                              # 104x53
    "htx-studio-project_kpilekig-42ad306f--MotorLimit-301a4056",                   # 161x5
    "lovelessing-d133-mobile-phone-development-bo-91bf7213--按键侧板V0.2-152707d2",# 2x32
    "tuyaopen-graffiti-t5pocket-pocket-machine-c240388e--2.-Pogo-Pin-4P-FPC-5c4f009b", # 30x44
]

for d in DELETE_DIRS:
    path = os.path.join(TARGET, d)
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"DELETED: {d}")
    else:
        print(f"NOT FOUND: {d}")

# 验证剩余
remaining = [d for d in os.listdir(TARGET) if os.path.isdir(os.path.join(TARGET, d))]
print(f"\n剩余 {len(remaining)} 个板子:")
for d in sorted(remaining):
    print(f"  - {d}")
