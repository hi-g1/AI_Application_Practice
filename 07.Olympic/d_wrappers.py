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

    def reset(self):
        self.episode_steps = 0
        observation = self.env.reset()

        # observation_opponent_agent = np.expand_dims(
        #     observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0
        # )
        #
        # observation_controlled_agent = np.expand_dims(
        #     observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        # )

        # opponent
        you_obs = observation[1 - self.controlled_agent_index]['obs']['agent_obs']
        you_energy = observation[1 - self.controlled_agent_index]['obs']['energy']
        you_obs = np.where((you_obs == 10), 11, you_obs)
        you_obs = np.where((you_obs == 8), 10, you_obs)
        you_obs = np.where((you_obs == 11), 8, you_obs)
        you_obs[39][39] = you_energy
        observation_opponent_agent = np.expand_dims(you_obs, axis=0)

        # controlled
        myobs = observation[self.controlled_agent_index]['obs']['agent_obs']
        myenergy = observation[self.controlled_agent_index]['obs']['energy']
        #myobs = np.where((myobs == 8) | (myobs == 10), 0, myobs)
        observation_controlled_agent = np.expand_dims(myobs, axis=0)

        # print(f"{observation_controlled_agent =}")

        # print(f"{self.sub_game = }")
        # print(observation_controlled_agent)
        # print(f"before frame stacking!! {observation_controlled_agent.shape = }")

        ######### frame stack #########
        observation_opponent_agent = self.frame_stacking(self.frames_opponent, observation_opponent_agent)
        observation_controlled_agent = self.frame_stacking(self.frames_controlled, observation_controlled_agent)

        # print(f"after frame stacking!! {observation_controlled_agent.shape = }")
        ################################

        # return [observation_controlled_agent], None
        return [observation_controlled_agent], None, [observation_opponent_agent]

    def step(self, action_controlled, action_opponent):

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        # action_opponent = self.get_opponent_action()
        # print(f"{action_opponent[0] =}")
        action_opponent = self.get_scaled_action(action_opponent)

        # print(action_controlled, f"{action_controlled.shape = }")

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        # print(f"after expanding dims!! {action_controlled}, {action_controlled.shape = }")

        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [
            action_controlled, action_opponent]
        # print(f"final action!! {action}")

        next_observation, reward, done, info_before, info_after = self.env.step(action)

        next_observation_opponent_agent = np.expand_dims(
            next_observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        # next_observation_controlled_agent = np.expand_dims(
        #     next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        # )

        # opponent
        you_obs = next_observation[1-self.controlled_agent_index]['obs']['agent_obs']
        you_energy = next_observation[1-self.controlled_agent_index]['obs']['energy']
        # you_obs = np.where((you_obs == 8) | (you_obs == 10), 0, you_obs)
        you_obs = np.where((you_obs == 10), 11, you_obs)
        you_obs = np.where((you_obs == 8), 10, you_obs)
        you_obs = np.where((you_obs == 11), 8, you_obs)
        you_obs[39][39] = you_energy
        next_observation_opponent_agent = np.expand_dims(you_obs, axis=0)

        # controlled
        myobs = next_observation[self.controlled_agent_index]['obs']['agent_obs']
        myenergy = next_observation[self.controlled_agent_index]['obs']['energy']
        # myobs = np.where((myobs == 8) | (myobs == 10), 0, myobs)
        myobs[39][39] = myenergy
        next_observation_controlled_agent = np.expand_dims(myobs, axis=0)

        # print(f"{next_observation_controlled_agent =}")

        reward_controlled = reward[self.controlled_agent_index]

        # 레슬링임~~~!!!

        if reward_controlled == 100:
            reward_controlled = 0

        # 중심점과 사각형의 가로, 세로 너비 설정
        center_x, center_y = 32, 19
        width = 10

        # 사각형 범위 계산
        start_x = center_x - width // 2
        end_x = center_x + width // 2
        start_y = center_y - width // 2
        end_y = center_y + width // 2

        # 범위를 벗어나지 않도록 보정
        start_x = max(start_x, 0)
        end_x = min(end_x, 39)
        start_y = max(start_y, 0)
        end_y = min(end_y, 39)

        temp_reward = 0
        frame = next_observation_controlled_agent[0]
        on_center = 0
        on_out = 0
        near_center = 0
        near_out = 0
        very_near_out = 0
        left_count = 0
        right_count = 0
        ones_in_square = 0

        cent_plus_reward = 0
        near_center_plus_reward = 0
        out_minus_reward = 0
        near_out_minus_reward = 0
        very_near_out_minus_reward = 0
        left_out_minus_reward = 0
        right_out_minus_reward = 0
        low_energy_minus = 0
        around_minus_reward = 0
        for i in range(40):
            if (26 <= i <= 28):
                for j in range(16, 25):
                    if int(frame[i][j] == 4):
                        on_center += 1

            if (10 <= i <= 25):
                for j in range(16, 25):
                    if int(frame[i][j] == 4):
                        near_center += 1

                    if int(frame[i][j] == 1):
                        near_out += 1

            if (37 <= i <= 39):
                for j in range(40):
                    if int(frame[i][j] == 1):
                        on_out += 1

            if (22 <= i <= 27):
                for j in range(40):
                    if int(frame[i][j] == 1):
                        very_near_out += 1

        for j in range(6):
            left_count += np.count_nonzero(frame[:, j] == 1)

        for j in range(34, 40):
            right_count += np.count_nonzero(frame[:, j] == 1)

        ones_in_square = np.count_nonzero(frame[start_x:end_x + 1, start_y:end_y + 1] == 1)
        # print(f"around {ones_in_square}")

        # print(f"오른쪽 {left_count}   왼쪽{right_count}\n")

        if on_center > 15:
            cent_plus_reward = 0.05
        if near_center > 15:
            near_center_plus_reward = 0.01
        if on_out > 7:
            # print("바로 뒤에 아웃이네..?")
            out_minus_reward = -0.15
        if near_out > 15:
            near_out_minus_reward = -0.05
        if very_near_out > 15:
            # print("바로 앞에 아웃이네..?")
            very_near_out_minus_reward = -0.15
        if right_count >= 10:
            right_out_minus_reward = -0.15
            # print("오른쪽에 아웃이네..?")
        if left_count >= 10:
            left_out_minus_reward = -0.15
            # print("왼쪽에 아웃이네..?")
        # if ones_in_square >= 10:
        #     around_minus_reward = -0.5

        if 10 < int(myobs[39][39]) < 80:
            low_energy_minus = -0.5

        myobs[39][39] = 0

        temp_reward += (
                cent_plus_reward + near_center_plus_reward + out_minus_reward + near_out_minus_reward + low_energy_minus + \
                very_near_out_minus_reward + left_out_minus_reward + right_out_minus_reward + around_minus_reward
        )
        # print(
        #     f"center: {cent_plus_reward}  near_center: {near_center_plus_reward}  out: {out_minus_reward}  near_out: {near_out_minus_reward}  hp: {low_energy_minus}")

        reward_controlled = (reward_controlled + temp_reward)



        ######### frame stack #########
        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)

        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################



        info = {}

        # return [next_observation_controlled_agent], reward_controlled, done, False, info
        return [next_observation_controlled_agent], reward_controlled, done, False, info, [next_observation_opponent_agent]

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self):
        force = random.uniform(-100, 200)
        angle = random.uniform(-30, 30)
        force = 0
        angle = 0
        opponent_scaled_actions = np.asarray([force, angle])

        return opponent_scaled_actions

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -100 + (clipped_action[0] + 1) / 2 * (200 - (-100))
        scaled_action_1 = -30 + (clipped_action[1] + 1) / 2 * (30 - (-30))

        if action[0] == 0.0 and action[1] == 0.0:
            scaled_action_0 = 0.0
            scaled_action_1 = 0.0
        
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
