import copy
import datetime
import logging
import numpy as np
import os
import torch
from env.pcb_joint_env import PCBJointEnv



class Runner(object):
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        self.max_episode_step = self.args.max_episode_step
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        self.input_folder = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data"
                )
            )
        print(f"input_folder: {self.input_folder}")
        

        self.env = PCBJointEnv(self.input_folder)  # training
        self.evaluate_env = PCBJointEnv(self.input_folder)  # evaluate

        # get all files and initialize file scores
        self.all_files = self.env.get_all_files()
        self.file_index = 0
        self.file_scores = {f: 0 for f in self.all_files}

        # initialize RL agent and replay buffer
        # TODO: Initialize RL agent and replay buffer here
    
    def run(self):
        self.all_files.sort(key=lambda f: os.path.getsize(os.path.join(self.input_folder, f)))

        sorted_files = self.all_files[::-1]
        logging.info("{}".format(sorted_files))
        for file_tmp in sorted_files: 
            file_path = os.path.join(self.input_folder, file_tmp)
            # TODO: run an episode for the current file

