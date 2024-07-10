from stable_baselines3 import AVEC_PPO, PPO, CORRECTED_AVEC_PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import wandb
import sys

if __name__ == "__main__":

    seed = int(sys.argv[1])
    env_name = str(sys.argv[2])
    mode = str(sys.argv[3])
    set_random_seed(seed)
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
    env = make_vec_env(env_name, n_envs=4)
    if mode == "AVEC":
        model = AVEC_PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    elif mode == "CORRECTED_AVEC":
        model = CORRECTED_AVEC_PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=1_000_000, callback=WandbCallback())
    run.finish()
