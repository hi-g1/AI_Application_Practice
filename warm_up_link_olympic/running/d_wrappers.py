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

        self.prev_farthest_4 = None

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
    def step(self, action_controlled):

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        action_opponent = self.get_opponent_action()
        # print(action_controlled, f"{action_controlled.shape = }")

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        # print(f"after expanding dims!! {action_controlled}, {action_controlled.shape = }")

        action = [action_opponent * 0.2, action_controlled * 0.2] if self.args.controlled_agent_index == 1 else [
            action_controlled * 0.2, action_opponent * 0.2]

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
            reward_controlled = 8

        adjusted_reward = self.adjust_reward(next_observation_controlled_agent, reward_controlled)
        info = {}

        return [next_observation_controlled_agent], adjusted_reward, done, False, info

    def adjust_reward(self, observation, original_reward):
        farthest_info = self.find_farthest_4(observation[-1])

        if farthest_info is not None:
            current_farthest_4, current_distance = farthest_info
        else:
            current_farthest_4 = None
            current_distance = 0

        if hasattr(self, 'prev_farthest_4') and self.prev_farthest_4 is not None:
            agent_position = np.where(observation == 8)
            prev_distance = self.calculate_distance(self.prev_farthest_4, (agent_position[0][0], agent_position[1][0]))

            if np.any(current_distance < prev_distance):
                original_reward += 0.0001
            else:
                original_reward -= 0.0001
        self.prev_farthest_4 = current_farthest_4

        if self.isRightArrow(observation[-1]):
            original_reward += 0.01
        else:
            original_reward -= 0.06

        if not self.check_double_adjacency(observation[-1]):
            original_reward -= 0.04

        if np.sum(observation[-1] == 4) < np.sum(observation[-2] == 4):
            original_reward -= 0.01

        original_reward -= 0.005

        if self.check_collision(observation[-1]):
            original_reward -= 0.5

        return original_reward

    # 벽의 기울기에 따라 조건을 준 함수. (정상 동작 확인되지 않아 주석 처리)
    # def calculate_average_slope_of_sixes(self, positions):
    #     # Calculate slope between every pair of positions
    #     slopes = []
    #     for i in range(len(positions)):
    #         for j in range(i + 1, len(positions)):
    #             y_diff = positions[j][0] - positions[i][0]
    #             x_diff = positions[j][1] - positions[i][1]
    #
    #             # Only calculate slope if points are adjacent (either horizontally, vertically, or diagonally)
    #             if abs(y_diff) <= 1 and abs(x_diff) <= 1:
    #                 # Avoid division by zero for vertical lines
    #                 if x_diff == 0 and y_diff > 0:
    #                     slope = 30
    #                 elif x_diff == 0 and y_diff < 0:
    #                     slope = -30
    #                 else:
    #                     slope = y_diff / x_diff
    #                 slopes.append(slope)
    #
    #     # Calculate the average of the slopes, ignoring infinities
    #     finite_slopes = [s for s in slopes if np.isfinite(s)]
    #     if finite_slopes:
    #         average_slope = np.mean(finite_slopes)
    #     else:
    #         average_slope = None  # Return None if there are no finite slopes
    #
    #     return average_slope

    def check_double_adjacency(self, array, value1=8, value2=6):
        rows, cols = array.shape
        # Define only direct directions for adjacency
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(rows):
            for j in range(cols):
                if array[i, j] == value1:
                    adjacency_count = 0
                    # Check adjacent cells in direct directions only
                    for dx, dy in directions:
                        adj_x, adj_y = i + dx, j + dy
                        if 0 <= adj_x < rows and 0 <= adj_y < cols and array[adj_x, adj_y] == value2:
                            adjacency_count += 1
                            if adjacency_count >= 2:
                                return False
        return True

    def find_rows_with_exactly_two_fours(self, array):
        rows_with_two_fours = []
        for row in range(40):
            # Count the number of '4's in each row
            count_fours = np.sum(array[row, :] == 4)

            # If exactly two '4's are found in a row, add it to the list
            if count_fours == 2:
                rows_with_two_fours.append(row)

        return rows_with_two_fours

    def isRightArrow(self, array, target_value=8):
        # Find the rows with exactly two '4's
        rows_with_two_fours = self.find_rows_with_exactly_two_fours(array)

        # Initialize variables to track the closest row and its nearest neighbor
        closest_row_distance = float('inf')
        closest_row = None
        nearest_neighbor_row = None

        # Find the indices of the target value (10) in the array
        target_indices = np.argwhere(array == target_value)

        # If no target values are found, we cannot proceed
        if len(target_indices) == 0:
            return False

        for row in rows_with_two_fours:
            # Calculate the distance of this row to the nearest '10'
            min_distance_to_ten = np.min(np.abs(target_indices[0][0] - row))

            # Update closest row
            if min_distance_to_ten < closest_row_distance:
                closest_row_distance = min_distance_to_ten
                closest_row = row

        # Now find the nearest neighbor row to the closest row
        nearest_neighbor_distance = float('inf')
        for row in rows_with_two_fours:
            if row != closest_row:
                distance_to_closest_row = np.abs(row - closest_row)
                if distance_to_closest_row < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance_to_closest_row
                    nearest_neighbor_row = row

        # If a closest row and its nearest neighbor are found, compare their distances
        if closest_row is not None and nearest_neighbor_row is not None:
            closest_row_fours = np.where(array[closest_row, :] == 4)[0]
            nearest_neighbor_row_fours = np.where(array[nearest_neighbor_row, :] == 4)[0]

            closest_row_distance_between_fours = np.abs(closest_row_fours[1] - closest_row_fours[0])
            nearest_neighbor_row_distance_between_fours = np.abs(
                nearest_neighbor_row_fours[1] - nearest_neighbor_row_fours[0])

            return closest_row_distance_between_fours > nearest_neighbor_row_distance_between_fours

        return False

    def calculate_distance(self, agent_pos, target_pos):
        return np.abs(agent_pos - target_pos)

    def find_farthest_4(self, observation):
        agent_position = np.where(observation == 8)
        target_positions = np.where(observation == 4)

        if len(agent_position[0]) == 0 or len(target_positions[0]) == 0:
            return None

        agent_pos = np.array([agent_position[0][0], agent_position[1][0]])
        target_pos = np.array(list(zip(target_positions[0], target_positions[1])))

        distances = np.sum(np.abs(target_pos - agent_pos), axis=1)
        farthest_index = np.argmax(distances)

        farthest_4_pos = target_pos[farthest_index]

        return farthest_4_pos, distances[farthest_index]

    def check_collision(self, array):
        rows, cols = array.shape
        for i in range(rows):
            for j in range(cols - 1):
                if array[i, j] == 8 and array[i, j + 1] == 6:
                    return True
        return False

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self):
        force = random.uniform(-100, 200)
        angle = random.uniform(-30, 30)
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
