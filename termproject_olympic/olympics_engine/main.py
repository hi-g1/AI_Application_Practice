import sys
from pathlib import Path

base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)
print(sys.path)
import argparse
from termproject_olympic.olympics_engine.agent import *
import time
from termproject_olympic.env.chooseenv import make
import json
from warm_up_1_olympic.g_test_agent import running_1_agent, wrestling_1_agent
from warm_up_eagle_olympic.running.rlmain.Olympic.running_2_agent import eagle_running_agent
from warm_up_eagle_olympic.wrestling.Olympic.g_test_agent import eagle_wrestling_agent
from warm_up_link_olympic.running.g_test_agent import link_running_agent
from warm_up_link_olympic.wrestling.g_test_agent import link_wrestling_agent

import numpy as np
import random

from collections import deque

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

def run_running(agent_1, agent_2):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="running")
    # "CartPole-v1", "Pendulum-v1", "Acrobot-v1", "LunarLanderContinuous-v2"
    # "LunarLander-v2", "BipedalWalker-v3", "MountainCarContinuous-v0"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    if env_name == 'running':
        game = make(env_type="olympics-integrated", game_name='olympics-running')
        agent_num = 2
    elif env_name == 'wrestling':
        game = make(env_type="olympics-integrated", game_name='olympics-wrestling')
        agent_num = 2

    obs = game.reset()
    print(obs)
    done = False
    step = 0
    if RENDER:
        game.env_core.render()

    time_epi_s = time.time()
    while not done:
        time.sleep(0.05)
        step += 1

        if agent_num == 2:
            action1 = agent_1.act(obs[0])
            action2 = agent_2.act(obs[1])
            action = [action1, action2]

        obs, reward, done, _, _ = game.step(action)
        # print('obs = ', obs)
        # plt.imshow(obs[0])
        # plt.show()
        if RENDER:
            game.env_core.render()
            # time.sleep(0.1)

    return reward

def run_wrestling(agent_1, agent_2):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="wrestling")
    # "CartPole-v1", "Pendulum-v1", "Acrobot-v1", "LunarLanderContinuous-v2"
    # "LunarLander-v2", "BipedalWalker-v3", "MountainCarContinuous-v0"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    frame_skipping = 3
    frame_skipping_num = 0
    prev_action = None
    if env_name == 'running':
        game = make(env_type="olympics-integrated", game_name='olympics-running')
        agent_num = 2
    elif env_name == 'wrestling':
        game = make(env_type="olympics-integrated", game_name='olympics-wrestling')
        agent_num = 2

    obs = game.reset()
    done = False
    step = 0
    if RENDER:
        game.env_core.render()

    time_epi_s = time.time()
    while not done:
        time.sleep(0.05)
        step += 1

        if agent_num == 2:
            action1 = agent_1.act(obs[0])
            action2 = agent_2.act(obs[1])
            action = [action1, action2]

        obs, reward, done, _, _ = game.step(action)
        # print('obs = ', obs)
        # plt.imshow(obs[0])
        # plt.show()
        if RENDER:
            game.env_core.render()

    return reward

if __name__ == "__main__":
    running_agent_0 = link_running_agent()
    running_agent_1 = running_1_agent()
    running_agent_2 = eagle_running_agent()
    wrestling_agent_0 = link_wrestling_agent()
    wrestling_agent_1 = wrestling_1_agent()
    wrestling_agent_2 = eagle_wrestling_agent()

    game_score = {'running_team_link': 0, 'running_team_1': 0, 'running_team_eagle': 0,
                  'wreatling_team_link': 0, 'wrestling_team_1': 0, 'wreatling_team_eagle': 0}

    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': link, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': team_1')
    for _ in range(3):
        reward = run_running(running_agent_0, running_agent_1)
        if reward[0] > reward[1]:
            game_score["running_team_link"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_1"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': team_1, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': link')
    for _ in range(3):
        reward = run_running(running_agent_1, running_agent_0)
        if reward[0] > reward[1]:
            game_score["running_team_1"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_link"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': link, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': eagle')
    for _ in range(3):
        reward = run_running(running_agent_0, running_agent_2)
        if reward[0] > reward[1]:
            game_score["running_team_link"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_eagle"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': eagle, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': link')
    for _ in range(3):
        reward = run_running(running_agent_2, running_agent_0)
        if reward[0] > reward[1]:
            game_score["running_team_eagle"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_link"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': team_1, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': eagle')
    for _ in range(3):
        reward = run_running(running_agent_1, running_agent_2)
        if reward[0] > reward[1]:
            game_score["running_team_1"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_eagle"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print("Game: running, " + '\033[91m' + 'agent_0' + '\033[0m' + ': eagle, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': team_1')
    for _ in range(3):
        reward = run_running(running_agent_2, running_agent_1)
        if reward[0] > reward[1]:
            game_score["running_team_eagle"] += 1
        elif reward[1] > reward[0]:
            game_score["running_team_1"] += 1
    print("GAME SCORE: ", game_score)

    print("GAME CHANGE TO WRESTLING!!!!!!")
    time.sleep(1)

    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': link, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': team_1')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_0, wrestling_agent_1)
        if reward[0] > reward[1]:
            game_score["wreatling_team_link"] += 1
        elif reward[1] > reward[0]:
            game_score["wrestling_team_1"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': team_1, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': link')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_1, wrestling_agent_0)
        if reward[0] > reward[1]:
            game_score["wrestling_team_1"] += 1
        elif reward[1] > reward[0]:
            game_score["wreatling_team_link"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': link, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': eagle')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_0, wrestling_agent_2)
        if reward[0] > reward[1]:
            game_score["wreatling_team_link"] += 1
        elif reward[1] > reward[0]:
            game_score["wreatling_team_eagle"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': eagle, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': link')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_2, wrestling_agent_0)
        if reward[0] > reward[1]:
            game_score["wreatling_team_eagle"] += 1
        elif reward[1] > reward[0]:
            game_score["wreatling_team_link"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': team_1, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': eagle')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_1, wrestling_agent_2)
        if reward[0] > reward[1]:
            game_score["wrestling_team_1"] += 1
        elif reward[1] > reward[0]:
            game_score["wreatling_team_eagle"] += 1
    print("GAME SCORE: ", game_score)
    time.sleep(1)
    print(
        "Game: wrestling, " + '\033[91m' + 'agent_0' + '\033[0m' + ': eagle, ' + '\033[34m' + 'agent_1' + '\033[0m' + ': team_1')
    for _ in range(3):
        reward = run_wrestling(wrestling_agent_2, wrestling_agent_1)
        if reward[0] > reward[1]:
            game_score["wreatling_team_eagle"] += 1
        elif reward[1] > reward[0]:
            game_score["wrestling_team_1"] += 1
    print("GAME SCORE: ", game_score)
    print("===================================================")
    print("FINAL GAME SCORE: ", game_score)
    print("===================================================")