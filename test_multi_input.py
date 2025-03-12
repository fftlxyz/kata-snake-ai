import time
import random

from sb3_contrib import MaskablePPO

from snake_game_env_multi_input import SnakeEnv


MODEL_PATH = r"trained_models/ppo_snake_final.zip"
NUM_EPISODE = 10

RENDER = False
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0
max_snake_size = 0

for episode in range(NUM_EPISODE):
    obs, _ = env.reset()
    episode_reward = 0
    done = False

    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    per_food_step_limit = env.game.grid_size * 4
    per_food_step = 0
    while not done:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction
        num_step += 1
        per_food_step += 1
        obs, reward, done, _, info = env.step(action)
        snake_size = info["snake_size"]

        if done:
            if info["snake_size"] == env.game.grid_size:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. "
                  f"Step Reward: {sum_step_reward:.4f} Snake Size: {snake_size}")
            sum_step_reward = 0
            per_food_step = 0
        else:
            sum_step_reward += reward

        episode_reward += reward

        if per_food_step > per_food_step_limit:
            print(f"Gameover, exceed per food step limit, steps:{per_food_step}")
            break
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    if snake_size > max_snake_size:
        max_snake_size = snake_size

    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, "
          f"Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, "
      f"Max Snake Size:{max_snake_size}  Average reward: {total_reward / NUM_EPISODE}")
