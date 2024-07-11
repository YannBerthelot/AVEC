#!/home/yberthel/AVEC/venv/bin/python
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

    batch_sizes = [32, 64, 128, 256]
    n_timesteps = int(1e6)
    num_envs = 4

    agents_dict = {"AVEC_PPO": AVEC_PPO, "CORRECTED_AVEC_PPO": CORRECTED_AVEC_PPO, "PPO": PPO}

    set_random_seed(seed)

    for batch_size in batch_sizes:
        run = wandb.init(
            # Set the project where this run will be logged
            project="avec experiments",
            # Track hyperparameters and run metadata
            sync_tensorboard=True,
            config={
                "agent": "PPO",
                "mode": mode,
                "env": env_name,
                "seed": seed,
                "batch size": batch_size,
                "num envs": num_envs,
                "num timesteps": n_timesteps,
            },
        )
        env = make_vec_env(env_name, n_envs=num_envs)
        agent = agents_dict[mode]
        model = agent("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", batch_size=batch_size)
        model.learn(total_timesteps=n_timesteps, callback=WandbCallback())
        run.finish()
