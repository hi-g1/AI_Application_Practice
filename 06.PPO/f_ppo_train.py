import os
import sys

# Add ppo module in system path
sys.path.append("AI_Application_Practice/06.PPO/")

from b_ppo import PPOAgent
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

import gymnasium as gym

from d_wrappers import (AcrobotActionWrapper,
                        PendulumActionNormalizer,
                        BipedalWalkerRewardWrapper)
from e_utils import show_video, convert_gif
import argparse
from config import args_ppo_pendulum_v1, args_ppo_acrobot_v1, args_ppo_cartpole_v1, args_ppo_lunarlander_continuous_v2, \
    args_ppo_lunarlander_v2, args_ppo_bipedalwalker_v3, args_ppo_mountaincar_continuous_v0


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


def make_env(env_name, render_mode=None):
    # environment
    env_id = env_name
    env = gym.make(env_id, render_mode=render_mode)
    if env_id == "Acrobot-v1":
        env = AcrobotActionWrapper(env)
    elif env_id == "Pendulum-v1":
        env = PendulumActionNormalizer(env)
    elif env_id == "BipedalWalker-v3":
        env = BipedalWalkerRewardWrapper(env)
    else:
        pass
    if hasattr(env, 'seed'):
        env.seed(config.seed)
    return env


def main(args, evaluation=False):
    if not evaluation:
        # ppo agent
        agent = PPOAgent(make_env, args)
        # ppo train
        agent.train()
    else:
        agent = PPOAgent(make_env, args)
        agent.load_predtrain_model(f"{args.path2save_train_history}/actor.pth", f"{args.path2save_train_history}/critic.pth")
        agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="MountainCarContinuous-v0")
    # "CartPole-v1", "Pendulum-v1", "Acrobot-v1", "LunarLanderContinuous-v2"
    # "LunarLander-v2", "BipedalWalker-v3", "MountainCarContinuous-v0"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    if env_name == "CartPole-v1":
        args = args_ppo_cartpole_v1.get_args(rest_args)
    elif env_name == "Pendulum-v1":
        args = args_ppo_pendulum_v1.get_args(rest_args)
    elif env_name == "Acrobot-v1":
        args = args_ppo_acrobot_v1.get_args(rest_args)
    elif env_name == "LunarLanderContinuous-v2":
        args = args_ppo_lunarlander_continuous_v2.get_args(rest_args)
    elif env_name == "LunarLander-v2":
        args = args_ppo_lunarlander_v2.get_args(rest_args)
    elif env_name == "BipedalWalker-v3":
        args = args_ppo_bipedalwalker_v3.get_args(rest_args)
    elif env_name == "MountainCarContinuous-v0":
        args = args_ppo_mountaincar_continuous_v0.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

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
