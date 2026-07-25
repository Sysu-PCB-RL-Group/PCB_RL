import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .pcb_layout_env import PCBLayoutEnv
from .pcb_routing_env import PCBRoutingEnv

class PCBJointEnv(gym.Env):
    """布局+布线联合训练环境"""
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, config_path="pcb_config.json"):
        super().__init__()
        # 初始化子环境
        self.layout_env = PCBLayoutEnv(config_path)
        self.routing_env = PCBRoutingEnv(config_path, layout_env=self.layout_env)
        
        # 联合动作空间：布局动作 + 布线动作
        self.action_space = spaces.Tuple([
            self.layout_env.action_space,
            self.routing_env.action_space
        ])
        
        # 联合状态空间：布局状态 + 布线状态
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.layout_env.observation_space.shape[0] + self.routing_env.observation_space.shape[0],),
            dtype=np.float32
        )
        
        self.max_steps = max(self.layout_env.max_steps, self.routing_env.max_steps)
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        layout_obs = self.layout_env.reset()
        routing_obs = self.routing_env.reset()
        # 归一化布局状态后拼接
        layout_obs = layout_obs / max(self.layout_env.config["board_spec"]["width"], self.layout_env.config["board_spec"]["height"])
        joint_obs = np.concatenate([layout_obs, routing_obs])
        return joint_obs
    
    def step(self, action):
        # 拆分动作：布局动作 + 布线动作
        layout_action, routing_action = action
        
        # 执行布局动作
        layout_obs, layout_reward, layout_done, layout_info = self.layout_env.step(layout_action)
        
        # 更新布线环境的布局状态，执行布线动作
        self.routing_env.layout_env = self.layout_env
        routing_obs, routing_reward, routing_done, routing_info = self.routing_env.step(routing_action)
        
        # 联合奖励（加权）
        joint_reward = 0.4 * layout_reward + 0.6 * routing_reward
        
        # 联合结束条件
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (layout_done and routing_done)
        
        # 联合观测
        layout_obs = layout_obs / max(self.layout_env.config["board_spec"]["width"], self.layout_env.config["board_spec"]["height"])
        joint_obs = np.concatenate([layout_obs, routing_obs])
        
        # 联合信息
        joint_info = {
            "step": self.current_step,
            "layout_info": layout_info,
            "routing_info": routing_info,
            "joint_reward": joint_reward
        }
        
        return joint_obs, joint_reward, done, joint_info
    
    def render(self, mode="human"):
        print("===== Joint Env Status =====")
        self.layout_env.render(mode)
        self.routing_env.render(mode)
        print("============================")