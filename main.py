import gymnasium as gym

from stable_baselines3 import AVEC_PPO, PPO, CORRECTED_AVEC_PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import wandb


if __name__ == "__main__":
    env_name = "Ant-v4"
    for mode in ["AVEC", "CORRECTED_AVEC", "PPO"]:
        for i in range(10):
            set_random_seed(i)
            run = wandb.init(
                # Set the project where this run will be logged
                project="avec experiments",
                # Track hyperparameters and run metadata
                sync_tensorboard=True,
                config={
                    "mode": mode,
                    "env": env_name,
                },
            )
            env = gym.make(env_name)
            if mode == "AVEC":
                model = AVEC_PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
            elif mode == "CORRECTED_AVEC":
                model = CORRECTED_AVEC_PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
            else:
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
            model.learn(total_timesteps=1_000_000, callback=WandbCallback())
            run.finish()
