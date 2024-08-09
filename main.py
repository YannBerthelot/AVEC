#!/home/yberthel/AVEC/venv/bin/python
from stable_baselines3 import AVEC_PPO, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import set_random_seed
import wandb
import sys
import torch
import torch.nn as nn
import yaml
import os
import psutil
from copy import deepcopy
from functools import partial

DEFAULT_N_STEPS = 2048


def read_hyperparams_data(file_name):
    with open(file_name) as stream:
        try:
            hyperparams_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return hyperparams_data


def linear_schedule(x, init_x):
    return x * init_x


def parse_hyperparams(env_name, hyperparams_data):
    hyperparams = hyperparams_data[env_name]
    for key, value in hyperparams.items():
        if isinstance(value, str):
            if "lin" in value:
                true_val = float(value.split("_")[1])
                hyperparams[key] = partial(linear_schedule, init_x=true_val)
    if "normalize" in hyperparams.keys():
        normalize = hyperparams.pop("normalize")
    else:
        normalize = False
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
    else:
        n_timesteps = None
    if "policy_kwargs" in hyperparams.keys():
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
            assert isinstance(hyperparams["policy_kwargs"], dict)
    return n_envs, policy, hyperparams, normalize, n_timesteps


if __name__ == "__main__":
    seed = int(sys.argv[1])
    env_name = str(sys.argv[2])
    mode = str(sys.argv[3])
    n_steps_factor = float(sys.argv[4])
    network_size_factor = float(sys.argv[5])
    alpha = float(sys.argv[6])
    n_timesteps_user = int(eval(sys.argv[7]))
    N_EVAL_TIMESTEPS = int(eval(sys.argv[8]))
    N_SAMPLES_MC = int(sys.argv[9])
    N_EVAL_ENVS = int(sys.argv[10])

    num_threads = int(psutil.cpu_count() / psutil.cpu_count(logical=False))
    torch.set_num_threads(num_threads)
    set_random_seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # hyperparams_data = read_hyperparams_data("/home/yberthel/AVEC/ppo.yml")
    hyperparams_data = read_hyperparams_data(os.path.join(dir_path, "ppo.yml"))
    n_envs, policy, hyperparams, normalize, n_timesteps = parse_hyperparams(
        env_name, hyperparams_data
    )  # TODO : change batch_size with batch_factor
    if "batch_size" in hyperparams.keys():
        hyperparams["n_steps"] = int(n_steps_factor * hyperparams["n_steps"])
    else:
        hyperparams["n_steps"] = int(DEFAULT_N_STEPS * n_steps_factor)
    if "policy_kwargs" in hyperparams.keys():
        policy_kwargs = hyperparams["policy_kwargs"]
        if "net_arch" in policy_kwargs.keys():
            net_arch = policy_kwargs["net_arch"]
            net_arch["vf"] = [int(x * network_size_factor) for x in net_arch["vf"]]
            policy_kwargs["net_arch"] = net_arch
        hyperparams["policy_kwargs"] = policy_kwargs
    else:
        hyperparams["policy_kwargs"] = dict(
            log_std_init=0,
            ortho_init=True,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[int(64 * network_size_factor), int(64 * network_size_factor)]),
        )

    run = wandb.init(
        project="avec experiments ranking local",
        sync_tensorboard=True,
        config={
            "agent": "PPO",
            "mode": mode,
            "env": env_name,
            "seed": seed,
            "rollout size factor": n_steps_factor,
            "critic network size factor": network_size_factor,
            "alpha": alpha,
        },
    )
    env = make_vec_env(env_name, n_envs=n_envs)
    if normalize:
        env = VecNormalize(env, gamma=hyperparams["gamma"] if "gamma" in hyperparams.keys() else 0.99)
    if mode == "PPO":
        agent = PPO
    else:
        hyperparams["env_name"] = env_name
        hyperparams["alpha"] = alpha
        hyperparams["n_eval_timesteps"] = N_EVAL_TIMESTEPS
        hyperparams["n_samples_MC"] = N_SAMPLES_MC
        hyperparams["n_eval_envs"] = N_EVAL_ENVS
        if mode == "AVEC_PPO":
            agent = AVEC_PPO
        elif mode == "CORRECTED_AVEC_PPO":
            hyperparams["correction"] = True
            agent = AVEC_PPO

    model = agent(
        policy,
        env,
        tensorboard_log=f"runs/{run.id}",
        **hyperparams,
        seed=seed,
    )
    model.learn(total_timesteps=n_timesteps if n_timesteps is not None else n_timesteps_user, callback=WandbCallback())
    run.finish()
