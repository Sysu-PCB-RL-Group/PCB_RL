import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .pcb_utils import PCBUtils

class PCBLayoutEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, config_path="pcb_config.json"):
        super().__init__()
        self.utils = PCBUtils(config_path)
        self.config = self.utils.config
        self.max_steps = self.config["rl_config"]["max_steps"]
        
        # 1. 定义动作空间：(元器件ID, 目标x坐标, 目标y坐标)
        # 离散化坐标（可根据精度调整）
        self.x_bins = np.linspace(0, self.config["board_spec"]["width"], 50)
        self.y_bins = np.linspace(0, self.config["board_spec"]["height"], 50)
        comp_list = self.utils.get_component_list()
        self.comp_ids = [comp["id"] for comp in comp_list if comp["id"] not in self.config["component_constraints"]["fixed_components"]]
        
        self.action_space = spaces.Tuple([
            spaces.Discrete(len(self.comp_ids)),  # 可选元器件
            spaces.Discrete(len(self.x_bins)),    # x坐标
            spaces.Discrete(len(self.y_bins))     # y坐标
        ])
        
        # 2. 定义状态空间：元器件位置+类型+网络需求+约束
        # 简化版：状态 = [所有元器件x/y, 未布线网络数, 约束违规数]
        state_dim = 2 * len(comp_list) + 2
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.config["board_spec"]["width"], self.config["board_spec"]["height"]),
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # 环境内部状态
        self.current_step = 0
        self.component_positions = {}  # {comp_id: (x,y)}
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.current_step = 0
        # 初始化元器件位置（固定元器件按配置，其他随机）
        comp_list = self.utils.get_component_list()
        for comp in comp_list:
            if comp["id"] in self.config["component_constraints"]["fixed_components"]:
                self.component_positions[comp["id"]] = (comp["x"], comp["y"])
            else:
                rand_x = np.random.uniform(0, self.config["board_spec"]["width"])
                rand_y = np.random.uniform(0, self.config["board_spec"]["height"])
                # 应该移动到板框外
                self.component_positions[comp["id"]] = (rand_x, rand_y)
        # 返回初始状态
        return self._get_obs()
    
    def _get_obs(self):
        """构建环境观测值"""
        # 1. 元器件位置展平
        pos_flat = []
        for comp in self.utils.get_component_list():
            pos = self.component_positions[comp["id"]]
            pos_flat.extend([pos[0], pos[1]])
        # 2. 未布线网络数
        net_state = self.utils.get_netlist_state()
        unrouted_num = len(net_state["unrouted_nets"])
        # 3. 约束违规数
        violations = self.utils.check_layout_constraints(self.component_positions)
        violation_num = len(violations)
        # 拼接状态
        obs = np.array(pos_flat + [unrouted_num, violation_num], dtype=np.float32)
        return obs
    
    def _calculate_reward(self):
        """计算当前步骤奖励"""
        # 1. 布局紧凑度奖励
        compactness = self.utils.calculate_layout_compactness(self.component_positions)
        compactness_reward = compactness * self.config["rl_config"]["reward_weights"]["layout_compactness"]
        
        # 2. 约束违规惩罚
        violations = self.utils.check_layout_constraints(self.component_positions)
        violation_penalty = len(violations) * self.config["rl_config"]["reward_weights"]["violation_penalty"]
        
        # 3. 效率奖励（步骤越少越好）
        efficiency_reward = 1 / (self.current_step + 1) * self.config["rl_config"]["reward_weights"]["efficiency"]
        
        # 总奖励
        total_reward = compactness_reward + violation_penalty + efficiency_reward
        return total_reward
    
    def step(self, action):
        """执行动作，返回 (obs, reward, done, info)"""
        # 解析动作：(元器件索引, x_bin索引, y_bin索引)
        comp_idx, x_bin_idx, y_bin_idx = action
        comp_id = self.comp_ids[comp_idx]
        target_x = self.x_bins[x_bin_idx]
        target_y = self.y_bins[y_bin_idx]
        
        # 执行布局动作（更新元器件位置）
        if comp_id not in self.config["component_constraints"]["fixed_components"]:
            self.component_positions[comp_id] = (target_x, target_y)
        
        # 检查是否结束
        self.current_step += 1
        done = self.current_step >= self.max_steps
        # 额外结束条件：无布局违规且紧凑度达标
        violations = self.utils.check_layout_constraints(self.component_positions)
        compactness = self.utils.calculate_layout_compactness(self.component_positions)
        if len(violations) == 0 and compactness > 0.8:
            done = True
        
        # 计算奖励和观测
        reward = self._calculate_reward()
        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "violations": violations,
            "compactness": compactness,
            "component_positions": self.component_positions
        }
        
        return obs, reward, done, info
    
    def render(self, mode="human"):
        """可视化布局状态（可对接KiCad预览）"""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Component Positions: {self.component_positions}")
            print(f"Violations: {self.utils.check_layout_constraints(self.component_positions)}")
        # 可选：实现rgb_array模式，输出PCB布局图像