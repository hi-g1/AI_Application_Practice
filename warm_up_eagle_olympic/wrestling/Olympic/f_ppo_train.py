import os
import sys
from pathlib import Path
project_path = str(Path(__file__).resolve().parent.parent)

# Add ppo module in system path and project path
sys.path.append("AI_Application_Practice/06.PPO/")
sys.path.append(project_path)

from b_ppo import PPOAgent
import numpy as np
import torch
import random

import gymnasium as gym

from d_wrappers import CompetitionOlympicsEnvWrapper
import argparse
from config import args_olympic_wrestling

from termproject_olympic.env.chooseenv import make

DEVICE = None

class GlobalConfig:
    def __init__(self):
        self.seed = 555
        self.path2save_train_history = "train_history"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

config = GlobalConfig()
seed_everything(config.seed)


def make_env(env_name, agent=None, config=None):
    # environment
    env = make(env_type="olympics-integrated", game_name=env_name)
    env = CompetitionOlympicsEnvWrapper(env, args=config)

    return env


def main(args, evaluation=False):
    if not evaluation:
        # ppo agent
        agent = PPOAgent(make_env, args)
        # ppo train
        agent.train()
    else:
        agent = PPOAgent(make_env, args)
        agent.load_predtrain_model(args)
        agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="olympics-wrestling")
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    args = args_olympic_wrestling.get_args(rest_args)

    args.path2save_train_history = config.path2save_train_history

    if not os.path.exists(args.path2save_train_history):
        try:
            os.mkdir(args.path2save_train_history)
        except:
            dir_path_head, dir_path_tail = os.path.split(args.path2save_train_history)
            if len(dir_path_tail) == 0:
                dir_path_head, dir_path_tail = os.path.split(dir_path_head)
            os.mkdir(dir_path_head)
            os.mkdir(args.path2save_train_history)

    main(args, args.is_evaluate)
