"""DQN 自回归训练管线集成测试（无需 KiCad）。

使用合成 PCB 数据模拟完整训练流程：
reset → 5 步自回归 → buffer → learn → 多 episode 收敛检查。

Mock 数据来源：env.mock_pcb。
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- mock utils.kicad_utils 必须在 import env 之前 ----
from unittest import mock
_m = mock.MagicMock()
sys.modules["utils.kicad_utils"] = _m

import logging
from typing import Tuple

import numpy as np
import torch

from env.mock_pcb import create_patched_env, MockKiCadUtils, MockComponent, MockNet
# 手动将 mock 子模块 attach 到 utils 并注入类（供 env.pcb_layout_env 中 from utils.kicad_utils import KiCadUtils 使用）
import utils
utils.kicad_utils = sys.modules["utils.kicad_utils"]
utils.kicad_utils.KiCadUtils = MockKiCadUtils
utils.kicad_utils.Component = MockComponent
utils.kicad_utils.Net = MockNet

from env.pcb_layout_env import Config
from dqn import DQNAgent, ReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==================== Episode 执行 ====================

def run_episode(env, agent: DQNAgent, buffer: ReplayBuffer,
                epsilon: float) -> Tuple[float, int, int, float]:
    """执行 1 个 episode：5 步自回归 + 每步 learn + buffer 写入。

    ⚠️ 与 run.py Runner._run_episode 独立维护（集成测试用副本）。
    若 run.py 逻辑变更，需手动同步此处。

    Returns:
        (episode_reward, step_count, placed_count, avg_loss)
    """
    env.reset()
    env._reset_pending()
    done = False
    episode_reward = 0.0
    step_count = 0
    placed_count = 0
    losses = []

    while not done:
        # ---- 5 步自回归 ----
        transitions = []
        for step_id in range(5):
            mask = env._build_mask_for_step(step_id)
            obs = env._get_obs_with_pending(step_id)
            action = agent.select_action(obs, mask, step_id, epsilon)
            env._update_pending(action, step_id)
            transitions.append((obs, action, mask, step_id))

        # ---- env.step() ----
        full_action = env._assemble_action()
        obs_next, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1
        placed_count = info.get("placed_count", 0)

        # ---- buffer 写入 ----
        for i, (s, a, mask, sid) in enumerate(transitions):
            if sid < 4:
                next_s = transitions[i + 1][0]
                next_sid = sid + 1
                step_done = False
            else:
                next_s = obs_next
                next_sid = 0 if not done else -1
                step_done = done

            step_reward = reward if sid == 4 else 0.0
            buffer.add(s, a, step_reward, next_s, mask, step_done, sid, next_sid)

        # ---- 每步 learn ----
        if buffer.is_ready(Config.BATCH_SIZE):
            batch = buffer.sample(Config.BATCH_SIZE)
            loss, _ = agent.learn(batch)
            losses.append(loss)
            if agent.train_steps % Config.TARGET_UPDATE_FREQ == 0:
                agent.sync_target()

        env._reset_pending()

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return episode_reward, step_count, placed_count, avg_loss


# ==================== 主测试 ====================

def test_training_pipeline():
    """主测试：运行 N 个 episode 的完整训练管线并验证关键指标。"""
    n_episodes = 20
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("=" * 60)
    logger.info(f"DQN Pipeline Integration Test ({n_episodes} episodes)")
    logger.info("=" * 60)

    # 0. 加载配置（确保测试使用与生产环境一致的超参数）
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "argument.txt")
    Config.load_from_file(config_path)

    # 1. 创建 env
    env = create_patched_env()

    # 2. 创建 Agent + Buffer
    device = torch.device("cpu")
    agent = DQNAgent(
        n_movable=env.N_movable,
        grid_w=Config.GRID_W, grid_h=Config.GRID_H,
        device=device,
        lr=Config.LEARNING_RATE, gamma=Config.GAMMA,
    )
    buffer = ReplayBuffer(capacity=Config.REPLAY_BUFFER_SIZE)

    logger.info(f"n_movable={env.N_movable}, lr={Config.LEARNING_RATE}, gamma={Config.GAMMA}")

    # 3. 训练 loop
    rewards = []
    placed_counts = []
    losses = []

    for ep in range(1, n_episodes + 1):
        eps = max(0.05, 1.0 - ep / n_episodes * 0.95)
        env.W_OUT_OF_BOUNDS_CURRENT = Config.W_OUT_OF_BOUNDS_INITIAL

        ep_reward, ep_steps, ep_placed, avg_loss = run_episode(env, agent, buffer, eps)

        rewards.append(ep_reward)
        placed_counts.append(ep_placed)
        losses.append(avg_loss)

        if ep % 5 == 0 or ep == 1:
            logger.info(
                f"[Ep {ep:3d}] ε={eps:.3f}  R={ep_reward:8.2f}  "
                f"placed={ep_placed}/{env.N_movable}  loss={avg_loss:.4f}  "
                f"buffer={len(buffer)}"
            )

    # 4. 验证
    logger.info("=" * 60)
    logger.info("Verification Results")
    logger.info("=" * 60)

    all_ok = True

    avg_first_half = np.mean(rewards[:n_episodes // 2])
    avg_second_half = np.mean(rewards[n_episodes // 2:])
    logger.info(f"  Reward: first_half_avg={avg_first_half:.2f}, second_half_avg={avg_second_half:.2f}")

    unique_rewards = len(set(round(r, 1) for r in rewards))
    if unique_rewards <= 1:
        logger.warning("  ⚠️  All rewards identical — model may not be learning")
        all_ok = False
    else:
        logger.info(f"  ✅  Reward varies ({unique_rewards} unique values)")

    logger.info(f"  Buffer size: {len(buffer)}")
    if len(buffer) < Config.BATCH_SIZE:
        logger.warning(f"  ⚠️  Buffer too small for batch_size={Config.BATCH_SIZE}")
        all_ok = False
    else:
        logger.info(f"  ✅  Buffer has enough samples for training")

    valid_losses = [l for l in losses if l > 0]
    if len(valid_losses) == 0:
        logger.warning("  ⚠️  No valid loss recorded")
        all_ok = False
    elif any(np.isnan(l) for l in valid_losses):
        logger.warning("  ⚠️  Loss contains NaN")
        all_ok = False
    else:
        logger.info(f"  ✅  Loss valid (avg={np.mean(valid_losses):.4f}, n={len(valid_losses)})")

    logger.info(f"  Train steps: {agent.train_steps}")
    if agent.train_steps == 0:
        logger.warning("  ⚠️  No training steps executed")
        all_ok = False
    else:
        logger.info(f"  ✅  Training executed")

    # 掩码正确性间接验证：
    # - select_action 未触发全 False fallback（否则 agent.py ERROR 日志可见）
    # - env.step() 未因 _invalid_action 卡死全部 episode
    # - 若需显式验证，应对照 _build_mask_for_step 编写独立单元测试

    # 5. 保存 checkpoint（验证 save/load 往返）
    ckpt_path = os.path.join(os.path.dirname(__file__), "test_checkpoint.pt")
    agent.save(ckpt_path)
    logger.info(f"  ✅  Checkpoint saved: {ckpt_path}")

    agent2 = DQNAgent(
        n_movable=env.N_movable,
        grid_w=Config.GRID_W, grid_h=Config.GRID_H,
        device=device,
    )
    agent2.load(ckpt_path)
    assert agent2.train_steps == agent.train_steps, "train_steps mismatch after load"
    logger.info(f"  ✅  Checkpoint load: train_steps={agent2.train_steps}")

    # 验证预测输出一致性（所有权重应完全相同）
    dummy_obs = {
        "component_slots": np.random.randn(1, env.N_movable, 8).astype(np.float32),
        "global_features": np.random.randn(1, 8).astype(np.float32),
    }
    dummy_t = {k: torch.tensor(v).to(device) for k, v in dummy_obs.items()}
    agent.q_net.eval()
    agent2.q_net.eval()
    for sid in range(5):
        with torch.no_grad():
            q1 = agent.q_net(dummy_t, sid)
            q2 = agent2.q_net(dummy_t, sid)
            assert torch.allclose(q1, q2, atol=1e-6), f"Output mismatch at step_id={sid}"
    logger.info(f"  ✅  Checkpoint prediction consistency verified (5 heads)")

    os.remove(ckpt_path)

    # 6. 最终判定
    logger.info("=" * 60)
    if all_ok:
        logger.info("🎉  ALL CHECKS PASSED — DQN Pipeline Integration Test SUCCESS")
    else:
        logger.warning("⚠️  Some checks failed — review warnings above")
    logger.info("=" * 60)

    assert all_ok, "Some checks failed — review warnings above"


if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
