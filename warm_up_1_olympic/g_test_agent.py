import random


from warm_up_1_olympic.a_actor_critic import CnnEncoder, ContinuousActor, init_weights
import numpy as np
from collections import deque
import torch


class running_1_agent:
    def __init__(self):
        device = "cpu"
        self.device = torch.device(device)
        self.encoder = CnnEncoder().apply(init_weights).to(torch.device(device))
        self.actor = ContinuousActor(self.encoder, torch.device(device)).apply(init_weights).to(torch.device(device))

        self.actor.load_state_dict(torch.load(
            "/Users/zzzanghun/git/AI_Application_Practice/warm_up_1_olympic/train_history/running/11_27_12_53/actor_2500000.pth",
            map_location=self.device))
        self.encoder.load_state_dict(torch.load(
            "/Users/zzzanghun/git/AI_Application_Practice/warm_up_1_olympic/train_history/running/11_27_12_53/encoder_2500000.pth",
            map_location=self.device))

        self.frame_stack = 4
        self.frames_controlled = deque([], maxlen=self.frame_stack)

    def act(self, obs):
        obs = self.obs_preprocessing(obs['obs']['agent_obs'])
        obs = np.reshape(obs, (1, 4, 40, 40)).astype(np.float64)
        obs = torch.FloatTensor(obs).to(self.device)
        action, _ = self.actor(obs)
        action = self.get_scaled_action(action)
        return action

    def obs_preprocessing(self, obs):
        obs = np.where((obs == 8) | (obs == 10), 0, obs)
        obs = np.expand_dims(obs, axis=0)

        # obs = obs / 10.0  # Normalize
        if len(self.frames_controlled) != self.frame_stack:
            for _ in range(self.frame_stack):
                self.frames_controlled.append(obs)
        else:
            self.frames_controlled.append(obs)
        obs = np.concatenate(list(self.frames_controlled), axis=0)

        return [obs]

    def get_scaled_action(self, action):
        action = list(action.detach().cpu().numpy()).pop()
        action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -100 + (action[0] + 1) / 2 * (170 - (-100))
        scaled_action_1 = -30 + (action[1] + 1) / 2 * (30 - (-30))

        return np.asarray([[scaled_action_0], [scaled_action_1]])


class wrestling_1_agent:
    def __init__(self):
        device = "cpu"
        self.device = torch.device(device)
        self.encoder = CnnEncoder().apply(init_weights).to(torch.device(device))
        self.actor = ContinuousActor(self.encoder, torch.device(device)).apply(init_weights).to(torch.device(device))

        self.actor.load_state_dict(torch.load(
            "/Users/zzzanghun/git/AI_Application_Practice/warm_up_1_olympic/train_history/wrestling/11_26_13_28/actor_1800000.pth",
            map_location=self.device))
        self.encoder.load_state_dict(torch.load(
            "/Users/zzzanghun/git/AI_Application_Practice/warm_up_1_olympic/train_history/wrestling/11_26_13_28/encoder_1800000.pth",
            map_location=self.device))

        self.frame_stack = 4
        self.frames_controlled = deque([], maxlen=self.frame_stack)

    def act(self, obs):
        obs = self.obs_preprocessing(obs['obs']['agent_obs'])
        obs = np.reshape(obs, (1, 4, 40, 40)).astype(np.float64)
        obs = torch.FloatTensor(obs).to(self.device)
        action, _ = self.actor(obs)
        action = self.get_scaled_action(action)
        return action

    def obs_preprocessing(self, obs):
        if int(obs[32][19]) == 10:
            obs = np.where((obs == 8), 9, obs)
            obs = np.where((obs == 10), 8, obs)
            obs = np.where((obs == 9), 10, obs)

        obs = np.expand_dims(obs, axis=0)

        # obs = obs / 10.0  # Normalize
        if len(self.frames_controlled) != self.frame_stack:
            for _ in range(self.frame_stack):
                self.frames_controlled.append(obs)
        else:
            self.frames_controlled.append(obs)
        obs = np.concatenate(list(self.frames_controlled), axis=0)

        return [obs]

    def get_scaled_action(self, action):
        action = list(action.detach().cpu().numpy()).pop()
        action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -100 + (action[0] + 1) / 2 * (170 - (-100))
        scaled_action_1 = -30 + (action[1] + 1) / 2 * (30 - (-30))

        return np.asarray([[scaled_action_0], [scaled_action_1]])

class random_agent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        force = 0
        angle = 0

        return [[force], [angle]]



a = running_1_agent()
