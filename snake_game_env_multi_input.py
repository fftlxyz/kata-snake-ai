import math

import gymnasium as gym
import numpy as np

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(4)  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN

        self.observation_space = gym.spaces.Dict({
            # rgb 3 * 84 * 84, stable_baselines3.common.torch_layers.NatureCNN 的结构
            "image": gym.spaces.Box(
                low=0, high=255,
                shape=(3, 84, 84),
                dtype=np.uint8),
            # 通过判断下一个snake head pos是否存在到snail tail的路径判断
            "next_direction_safe": gym.spaces.MultiBinary(4)})

        self.board_size = board_size
        self.grid_size = board_size ** 2  # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 4  # More than enough steps to get the food.
        else:
            self.step_limit = 1e9  # Basically no limit.
        self.reward_step_counter = 0

    def reset(self, *, seed=None, options=None):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs, {}

    def step(self, action):
        self.done, info = self.game.step(
            action)  # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size:  # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1  # Victory reward
            self.done = True
            return obs, reward, self.done, False, info

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        if self.done:  # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (
                    self.grid_size - info["snake_size"]) / self.max_growth)  # (-max_growth, -1)
            reward = reward * 0.1
            return obs, reward, self.done, False, info

        elif info["food_obtained"]:  # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0  # Reset reward step counter

        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(
                    info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        # 如果snake头尾存在通路, reward加倍，惩罚减半, 不存在通路容易把自己搞死
        if self._exist_way_from_pos_to_tail(self.game.snake[0]):
            if reward > 0:
                reward = reward * 2
            else:
                reward = reward / 2

        return obs, reward, self.done, False, info

    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])

    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return False
            else:
                col += 1

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                    (row, col) in snake_list  # The snake won't pop the last cell if it ate food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )
        else:
            game_over = (
                    (row, col) in snake_list[:-1]  # The snake will pop the last cell if it did not eat food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):

        # revers the obs shape for easy access
        image_obs = np.zeros((84, 84, 3), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        image_obs[tuple(np.transpose(self.game.snake))] = [(i, i, i) for i in
                                                           np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)]

        # Set the snake head to green and the tail to blue
        image_obs[tuple(self.game.snake[0])] = [0, 255, 0]
        image_obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        image_obs[self.game.food] = [0, 0, 255]

        # return a (3, 84, 84) image obs
        image_obs = np.transpose(image_obs, (2, 0, 1))

        next_direction_safe = self._check_next_direction_safe()

        return {
            "image": image_obs,
            "next_direction_safe": next_direction_safe
        }

    def _check_next_direction_safe(self):

        next_direction_safe = [0, 0, 0, 0]

        # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        direction = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        for i in range(0, 4):
            if not self._check_action_validity(i):
                continue
            row, col = self.game.snake[0]
            if self._exist_way_from_pos_to_tail((row + direction[i][0], col + direction[i][1])):
                next_direction_safe[i] = 1

        return next_direction_safe

    def _exist_way_from_pos_to_tail(self, head_pos):

        visited = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        visited[tuple(np.transpose(self.game.snake))] = 1
        visited[head_pos] = 1
        visited[self.game.snake[-1]] = 0

        return self._do_exists_way_from_pos_to_tail(head_pos, visited)

    def _do_exists_way_from_pos_to_tail(self, pos, visited):

        if pos == self.game.snake[-1]:
            return True

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + direction[0], pos[1] + direction[1])
            if 0 <= next_pos[0] < self.board_size \
                    and 0 <= next_pos[1] < self.board_size \
                    and visited[next_pos] == 0:
                visited[next_pos] = 1
                if self._do_exists_way_from_pos_to_tail(next_pos, visited):
                    return True
        return False
