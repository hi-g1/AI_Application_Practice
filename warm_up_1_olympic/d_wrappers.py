import math

import gym
import numpy
import numpy as np
import random

from collections import deque

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class CompetitionOlympicsEnvWrapper(gym.Wrapper):
    metadata = {}

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

    def reset(self):
        self.episode_steps = 0
        observation = self.env.reset()

        observation_opponent_agent = np.expand_dims(
            observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )
        observation_controlled_agent = np.expand_dims(
            observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        ######### frame stack #########
        observation_opponent_agent = self.frame_stacking(self.frames_opponent, observation_opponent_agent)
        observation_controlled_agent = self.frame_stacking(self.frames_controlled, observation_controlled_agent)

        # print(f"after frame stacking!! {observation_controlled_agent.shape = }")
        ################################

        return [observation_controlled_agent], None

    # 나는 8이다.
    def step(self, action_controlled, ep):

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        action_opponent = self.get_opponent_action(ep)
        # print(action_controlled, f"{action_controlled.shape = }")

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        # print(f"after expanding dims!! {action_controlled}, {action_controlled.shape = }")

        if np.any(self._transform_observation(self.frames_controlled)[-1] == 10):
            action_controlled *= 0.4
        else:
            action_controlled *= 0.15

        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [
            action_controlled, action_opponent]

        next_observation, reward, done, info_before, info_after = self.env.step(action)

        next_observation_opponent_agent = np.expand_dims(
            next_observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        next_observation_controlled_agent = np.expand_dims(
            next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        ######### frame stack #########
        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)

        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################

        reward_controlled = reward[self.controlled_agent_index]

        if reward_controlled == 1:
            reward_controlled = 30

        adjusted_reward = self.adjust_reward(next_observation_controlled_agent, reward_controlled)
        info = {}

        return [next_observation_controlled_agent], adjusted_reward, done, False, info

    def adjust_reward(self, observation, original_reward):
        if self.closest_distance_to_one(observation[-1], 1, 8) < self.closest_distance_to_one(observation[-1], 1, 10):
            original_reward -= 0.3
        else:
            original_reward += 0.3

        if self.closest_distance_to_one(observation[-1], 1) >= self.closest_distance_to_one(observation[-2], 1):
            original_reward += 0.1

        if self.check_condition(observation[-1]):
            original_reward -= 2

        if np.any(observation[-1] == 10):
            original_reward += 0.15
        else:
            original_reward -= 0.1


        return original_reward

    def closest_distance_to_one(self, array, target_value, object=8):
        target_positions = np.argwhere(array == object)
        one_positions = np.argwhere(array == target_value)

        if target_positions.size == 0 or one_positions.size == 0:
            return 9999

        # Calculate all pairwise distances and find the minimum
        distances = np.sqrt((target_positions[:, None, 0] - one_positions[:, 0]) ** 2 +
                            (target_positions[:, None, 1] - one_positions[:, 1]) ** 2)
        min_distance = np.min(distances)

        return min_distance

    def check_condition(self, array):
        # ndarray를 반복하면서 8 값 주변에 1이 있는지 확인
        for i in range(1, array.shape[0] - 1):
            for j in range(1, array.shape[1] - 1):
                if array[i, j] == 8 and (array[i - 1:i + 2, j] == 1).any() or (array[i, j - 1:j + 2] == 1).any():
                    return True
        return False

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self, ep):

        if ep % 11 == 0:
            angle = random.uniform(-5, 5)
            force = random.uniform(50, 200)
        elif ep % 9 == 0:
            angle = random.uniform(-20, 20)
            force = random.uniform(0, 0.5)
        elif ep % 7 == 0:
            if self.episode_steps < 30:
                angle = random.uniform(-2, -5)
                force = random.uniform(0, 0)
            elif self.episode_steps < 250:
                angle = random.uniform(1.7, 1.7)
                force = random.uniform(0, 5)
            else:
                angle = random.uniform(-1, -4)
                force = random.uniform(0, 1)
        elif ep % 5 == 0:
            if self.episode_steps < 30:
                angle = random.uniform(2, 5)
                force = random.uniform(0, 0)
            elif self.episode_steps < 250:
                angle = random.uniform(-1.7, -1.7)
                force = random.uniform(0, 5)
            else:
                angle = random.uniform(1, 4)
                force = random.uniform(0, 1)
        elif ep % 3 == 0:
            angle = random.uniform(-15, 15)
            force = random.uniform(-1, 10)
        else:
            angle = random.uniform(-7, 7)
            force = random.uniform(0, 4)

        opponent_scaled_actions = np.asarray([force, angle])

        return opponent_scaled_actions

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -100 + (clipped_action[0] + 1) / 2 * (200 - (-100))
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
