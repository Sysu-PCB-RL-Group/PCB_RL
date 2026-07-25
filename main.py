from run import Runner
import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")

    parser.add_argument('--env_name', type=str, default='torch') 
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=114514, help="random seed")

    # save path
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--task_name", type=str, default="default_task_name")
    
    args = parser.parse_args()
    return args

def run():
    args = get_config()
    runner = Runner(args)
    runner.run()

if __name__ == '__main__':
    run()


