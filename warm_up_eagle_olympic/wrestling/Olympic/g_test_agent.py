import sys
from pathlib import Path
project_path = str(Path(__file__).resolve().parent)
sys.path.append(project_path)
from a_actor_critic import CnnEncoder, ContinuousActor, init_weights
import numpy as np
from collections import deque
import torch


class eagle_wrestling_agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.encoder = CnnEncoder().apply(init_weights).to(torch.device("cpu"))
        self.actor = ContinuousActor(self.encoder, torch.device("cpu")).apply(init_weights).to(torch.device("cpu"))

        self.actor.load_state_dict(torch.load("/Users/zzzanghun/git/AI_Application_Practice/warm_up_eagle_olympic/wrestling/train_history/olympics-wrestling/12_6_11_43/actor.pth", map_location=self.device))
        self.encoder.load_state_dict(torch.load("/Users/zzzanghun/git/AI_Application_Practice/warm_up_eagle_olympic/wrestling/train_history/olympics-wrestling/12_6_11_43/encoder.pth", map_location=self.device))

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
        obs = np.expand_dims(obs, axis=0)

        obs = obs / 10.0  # Normalize
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

        scaled_action_0 = -60 + (action[0] + 1) / 2 * (60 - (-60))
        scaled_action_1 = -30 + (action[1] + 1) / 2 * (30 - (-30))

        return np.asarray([[scaled_action_0], [scaled_action_1]])
