"""临时烟测：验证 BC 正则化锚定（init_bc_anchor + _bc_regularization + learn 集成）。

关键点：锚定后 q_net 与 bc_net 权重相同 → 首次 learn 的 KL 恰为 0（正确）。
需多次更新让 q_net 偏离 bc_net 后，reg 才 > 0。
"""
import numpy as np
import torch
from env.pcb_layout_env import Config
from dqn.agent import DQNAgent

Config.GRID_W = 10
Config.GRID_H = 10
Config.MAX_MOVABLE = 8
Config.BC_REG_LAMBDA = 10.0
Config.BC_REG_TEMP = 1.0

head_dims = [8, 10, 10, 4, 2]


def make_batch(B=16, M=8, seed=0):
    rng = np.random.default_rng(seed)
    # 避免全掩码边界：placed_flags(第0维)保持小值(<0.5)，grid_w/grid_h 取 [0.5,1]→[5,10]
    slots = rng.standard_normal((B, M, 8)) * 0.2
    gf = rng.random((B, 8))
    gf[:, 0] = 0.5 + 0.5 * rng.random(B)
    gf[:, 1] = 0.5 + 0.5 * rng.random(B)
    obs = {
        "component_slots": slots.astype("float32"),
        "global_features": gf.astype("float32"),
    }
    next_obs = {k: v.copy() for k, v in obs.items()}
    step_ids = rng.integers(0, 5, size=B).astype("int64")
    next_step_ids = rng.integers(0, 5, size=B).astype("int64")
    actions = np.array([rng.integers(0, head_dims[s]) for s in step_ids], dtype="int64")
    rewards = rng.standard_normal(B).astype("float32")
    dones = np.zeros(B, dtype="float32"); dones[0] = 1.0
    return (obs, actions, rewards, next_obs, [None] * B, dones, step_ids, next_step_ids)


agent = DQNAgent(8, 10, 10, torch.device("cpu"))
# 锚定前：reg 应关闭
b = make_batch(seed=1)
l0, _ = agent.learn(b, None)
assert agent.last_bc_reg == 0.0 and np.isfinite(l0)
print("before anchor ok, loss=", round(l0, 4))

# 初始化 BC 锚点
agent.init_bc_anchor()
assert agent.bc_net_active and agent.bc_reg_lambda == 10.0

# 多次更新让 q_net 偏离 bc_net → 验证 KL 与值 L2 均生效且有限
regs = []
vals = []
for i in range(6):
    b = make_batch(seed=10 + i)
    l, _ = agent.learn(b, None)
    assert np.isfinite(l)
    regs.append(agent.last_bc_reg)
    vals.append(agent.last_bc_val)
print("after anchor kl=", [round(r, 4) for r in regs], "val=", [round(v, 3) for v in vals])
assert any(r > 0 for r in regs), "BC 策略 KL 应 >0（q_net 已偏离 bc_net 后被锚定拉回）"
assert any(v > 0 for v in vals), "BC 值 L2 应 >0（Q 值已偏离 BC Q 值）"

print("BC regularization smoke OK")
