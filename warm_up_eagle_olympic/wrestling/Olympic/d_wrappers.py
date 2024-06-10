
from a_actor_critic import CnnEncoder, ContinuousActor, init_weights
import torch
import gym
import numpy
import numpy as np
import random

from collections import deque

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class CompetitionOlympicsEnvWrapper(gym.Wrapper):
    def __init__(self, env, agent=None, args=None):
        super().__init__(env)

        self.args = args
        assert self.args

        self.controlled_agent_index = args.controlled_agent_index

        # for frame_stack
        self.frame_stack = args.frame_stack
        assert self.frame_stack > 0 or isinstance(self.frame_stack, int)
        self.frames_controlled = deque([], maxlen=self.frame_stack)
        self.frames_opponent = deque([], maxlen=self.frame_stack)

        self.sub_game = args.env_name
        self.device = args.device

        self.episode_steps = 0
        self.total_steps = 0

        # Opponent
        self.opponent_agent = ContinuousActor(encoder=CnnEncoder().apply(init_weights).to(self.device), device=torch.device("cpu"))
        self.opponent_agent.load_state_dict(torch.load("train_history/olympics-wrestling/12_6_11_43/actor.pth", map_location=self.device))

    def reset(self):
        self.episode_steps = 0
        observation = self.env.reset()

        observation_opponent_agent = np.expand_dims(observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        observation_controlled_agent = np.expand_dims(observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0)

        # print(f"{self.sub_game = }")
        # print(observation_controlled_agent)
        # print(f"before frame stacking!! {observation_controlled_agent.shape = }")

        ######### frame stack #########
        observation_opponent_agent = self.frame_stacking(self.frames_opponent, observation_opponent_agent)
        observation_controlled_agent = self.frame_stacking(self.frames_controlled, observation_controlled_agent)

        # print(f"after frame stacking!! {observation_controlled_agent.shape = }")
        ################################

        # Opponent
        self.observation_opponent_agent = [observation_opponent_agent]

        return [observation_controlled_agent], None

    def step(self, action_controlled):

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        # Opponent
        state = np.array(self.observation_opponent_agent)
        state = torch.FloatTensor(state).to(self.device)
        action_opponent, _ = self.opponent_agent(state)
        action_opponent = self.get_scaled_action(list(action_opponent.detach().cpu().numpy()).pop())

        action_controlled = self.get_scaled_action(action_controlled)
        # action_opponent = self.get_opponent_action() # Random Action
        # print(action_controlled, f"{action_controlled.shape = }")

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        # print(f"after expanding dims!! {action_controlled}, {action_controlled.shape = }")

        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [action_controlled, action_opponent]
        # print(f"final action!! {action}")

        next_observation, reward, done, info_before, info_after = self.env.step(action)

        next_observation_opponent_agent = np.expand_dims(next_observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        next_observation_controlled_agent = np.expand_dims(next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0)

        # HAN
        count3 = np.sum(next_observation_controlled_agent[:, 27:38, 15:26] == 4.0)
        count4 = np.sum(next_observation_controlled_agent[:, 27:38, 15:26] == 1.0)
        # count5 = np.sum(next_observation_controlled_agent[:, 25:40, 13:28] == 1.0)
        # count6 = np.sum(next_observation_controlled_agent[:, 25:40, 13:28] == 4.0)
        # front = np.sum(next_observation_controlled_agent[:, 0:28, 19:22] == 4.0)
        
        # 12_3_23_54
        # reward[self.controlled_agent_index] += 1

        # 12_4_17_50
        # reward[self.controlled_agent_index] += 1
        # reward[self.controlled_agent_index] += count3

        # 12_4_23_9
        # reward[self.controlled_agent_index] += 1
        # reward[self.controlled_agent_index] += count3
        # reward[self.controlled_agent_index] = 0 if count4 > 0 else reward[self.controlled_agent_index]

        # New
        reward[self.controlled_agent_index] += 1
        reward[self.controlled_agent_index] += count3 * 2
        reward[self.controlled_agent_index] = 0 if count4 > 0 else reward[self.controlled_agent_index]

        # END

        ######### frame stack #########
        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)

        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################
        
        reward_controlled = reward[self.controlled_agent_index]

        info = {}

        # Opponent
        self.observation_opponent_agent = [next_observation_opponent_agent]

        return [next_observation_controlled_agent], reward_controlled, done, False, info

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self):
        force = random.uniform(-100, 200) * 0.001
        angle = random.uniform(-30, 30) * 0.001
        opponent_scaled_actions = np.asarray([force, angle])

        return opponent_scaled_actions

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -60 + (clipped_action[0] + 1) / 2 * (60 - (-60))
        scaled_action_1 = -30 + (clipped_action[1] + 1) / 2 * (30 - (-30))

        return numpy.asarray([scaled_action_0, scaled_action_1])

    def frame_stacking(self, deque, obs):
        for _ in range(self.frame_stack):
            deque.append(obs)
        obs = self._transform_observation(deque)
        return obs

    def _transform_observation(self, frames):
        assert len(frames) == self.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        return obs
