#!/home/yberthel/AVEC/venv/bin/python
from stable_baselines3 import AVEC_PPO, PPO, CORRECTED_AVEC_PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import wandb
import sys
import torch
import torch.nn as nn
import yaml

DEFAULT_BATCH_SIZE = 64


def read_hyperparams_data(file_name):
    with open(file_name) as stream:
        try:
            hyperparams_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return hyperparams_data


def parse_hyperparams(env_name, hyperparams_data):
    hyperparams = hyperparams_data[env_name]
    if "normalize" in hyperparams.keys():
        hyperparams["normalize_advantage"] = hyperparams.pop("normalize")
    if "n_envs" in hyperparams.keys():
        n_envs = hyperparams.pop("n_envs")
    else:
        n_envs = 1
    if "policy" in hyperparams.keys():
        policy = hyperparams.pop("policy")
    else:
        policy = "MlpPolicy"
    if "n_timesteps" in hyperparams.keys():
        n_timesteps = hyperparams.pop("n_timesteps")
    if "policy_kwargs" in hyperparams.keys():
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
            assert isinstance(hyperparams["policy_kwargs"], dict)
    return n_envs, policy, hyperparams


if __name__ == "__main__":
    env_name = str(sys.argv[2])
    seed = int(sys.argv[1])
    mode = str(sys.argv[3])
    batch_size_factor = int(sys.argv[4])
    n_timesteps = int(1e6)

    num_threads = 2
    torch.set_num_threads(num_threads)
    set_random_seed(seed)
    agents_dict = {"AVEC_PPO": AVEC_PPO, "CORRECTED_AVEC_PPO": CORRECTED_AVEC_PPO, "PPO": PPO}

    hyperparams_data = read_hyperparams_data("/home/yberthel/AVEC/ppo.yml")
    n_envs, policy, hyperparams = parse_hyperparams(env_name, hyperparams_data)  # TODO : change batch_size with batch_factor
    if "batch_size" in hyperparams.keys():
        hyperparams["batch_size"] *= batch_size_factor
    else:
        hyperparams["batch_size"] = DEFAULT_BATCH_SIZE * batch_size_factor
    run = wandb.init(
        project="avec experiments 2",
        sync_tensorboard=True,
        config={
            "agent": "PPO",
            "mode": mode,
            "env": env_name,
            "seed": seed,
            "batch size factor": batch_size_factor,
        },
    )
    env = make_vec_env(env_name, n_envs=n_envs)
    agent = agents_dict[mode]
    model = agent(policy, env, tensorboard_log=f"runs/{run.id}", **hyperparams)
    model.learn(total_timesteps=n_timesteps, callback=WandbCallback())
    run.finish()
