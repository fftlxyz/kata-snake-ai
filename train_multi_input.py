import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_env_multi_input import SnakeEnv

LOAD_LATEST_MODEL = False
MODEL_SAVE_DIR = "trained_models"
FINAL_MODEL_FILE_NAME = "ppo_snake_final.zip"

NUM_ENV = 8
BATCH_SIZE = 512
DEVICE = torch.cpu.current_device()
# try use accelerator
if torch.accelerator.is_available():
    NUM_ENV = 32
    BATCH_SIZE = 512 * 8
    DEVICE = torch.accelerator.current_accelerator()

TENSOR_BOARD_LOG_DIR = "tensor_board_logs"

os.makedirs(TENSOR_BOARD_LOG_DIR, exist_ok=True)


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed, silent_mode=True)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        return env

    return _init


def main():
    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = DummyVecEnv([make_env(seed=s) for s in seed_set])

    latest_model_path = os.path.join(MODEL_SAVE_DIR, FINAL_MODEL_FILE_NAME)
    # try use the latest trained model
    if LOAD_LATEST_MODEL and os.path.exists(latest_model_path):
        model = MaskablePPO.load(latest_model_path, env)
        print("latest model loaded...")
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            device=DEVICE,
            verbose=1,
            n_steps=2048,
            batch_size=BATCH_SIZE,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=TENSOR_BOARD_LOG_DIR
        )

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    checkpoint_interval = 156250  # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=MODEL_SAVE_DIR,
                                             name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(MODEL_SAVE_DIR, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(100_000_000),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(MODEL_SAVE_DIR, "ppo_snake_final.zip"))


if __name__ == "__main__":
    main()
