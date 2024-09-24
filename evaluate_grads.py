#!/home/yberthel/AVEC/venv/bin/python
from stable_baselines3.common import utils
import pdb
import numpy as np
from math import ceil
from stable_baselines3.common.callbacks import WandbCheckpointCallback
from stable_baselines3 import AVEC_PPO, PPO, AVEC_SAC, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.avec_utils import read_from_pickle, save_to_json, copy_to_host_and_delete
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
import shutil
import math
from pathlib import Path

DEFAULT_N_STEPS = 2048
DEFAULT_BUFFER_SIZE = 1000000


def copy_from_host(destination_file: Path, host: str, host_file: Path) -> None:
    os.system(f"scp -r {host}:{host_file} {destination_file}")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


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
    assert "PPO" in mode or "SAC" in mode, f"Unrecognized mode {mode}"
    n_steps_factor = float(sys.argv[4])
    network_size_factor = float(sys.argv[5])
    alpha = float(sys.argv[6])
    n_timesteps_user = int(eval(sys.argv[7]))
    N_EVAL_TIMESTEPS = int(eval(sys.argv[8]))
    N_SAMPLES_MC = int(sys.argv[9])
    N_EVAL_ENVS = int(sys.argv[10])
    flag = int(sys.argv[11])

    num_threads = int(psutil.cpu_count() / psutil.cpu_count(logical=False))
    torch.set_num_threads(num_threads)
    set_random_seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    hyperparams_data = read_hyperparams_data(os.path.join(dir_path, "ppo.yml" if "PPO" in mode else "sac.yml"))
    n_envs, policy, hyperparams, normalize, n_timesteps = parse_hyperparams(
        env_name, hyperparams_data
    )  # TODO : change batch_size with batch_factor
    if "PPO" in mode:
        if "n_steps" in hyperparams.keys():
            hyperparams["n_steps"] = int(n_steps_factor * hyperparams["n_steps"])
        else:
            hyperparams["n_steps"] = int(DEFAULT_N_STEPS * n_steps_factor)
    elif "SAC" in mode:
        if "buffer_size" in hyperparams.keys():
            hyperparams["buffer_size"] = int(n_steps_factor * hyperparams["buffer_size"])
        else:
            hyperparams["buffer_size"] = int(DEFAULT_BUFFER_SIZE * n_steps_factor)

    if "policy_kwargs" in hyperparams.keys():
        policy_kwargs = hyperparams["policy_kwargs"]
        if "net_arch" in policy_kwargs.keys():
            net_arch = policy_kwargs["net_arch"]
            if "PPO" in mode:
                net_arch["vf"] = [int(x * network_size_factor) for x in net_arch["vf"]]
            else:
                net_arch["qf"] = [int(x * network_size_factor) for x in net_arch["qf"]]
            policy_kwargs["net_arch"] = net_arch
        hyperparams["policy_kwargs"] = policy_kwargs
    else:
        if "PPO" in mode:
            hyperparams["policy_kwargs"] = dict(
                log_std_init=0,
                ortho_init=True,
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[64, 64], vf=[int(64 * network_size_factor), int(64 * network_size_factor)]),
            )
        else:  # SAC
            hyperparams["policy_kwargs"] = dict(
                net_arch=dict(pi=[256, 256], qf=[int(256 * network_size_factor), int(256 * network_size_factor)]),
                activation_fn=nn.ReLU,
            )

    run = wandb.init(
        project="avec experiments sac 5 local",
        sync_tensorboard=True,
        config={
            "agent": mode,
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
    elif (mode == "AVEC_PPO") or (mode == "CORRECTED_AVEC_PPO"):
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
    elif mode == "SAC":
        agent = SAC
    elif "AVEC_SAC" in mode:  # or (mode == "CORRECTED_AVEC_PPO"):
        hyperparams["env_name"] = env_name
        hyperparams["alpha"] = alpha
        hyperparams["n_eval_timesteps"] = N_EVAL_TIMESTEPS
        hyperparams["n_samples_MC"] = N_SAMPLES_MC
        hyperparams["n_eval_envs"] = N_EVAL_ENVS
        if mode == "AVEC_SAC":
            agent = AVEC_SAC
        elif mode == "CORRECTED_AVEC_SAC":
            hyperparams["correction"] = True
            agent = AVEC_SAC
    model = agent(
        policy,
        env,
        tensorboard_log=f"runs/{run.id}",
        **hyperparams,
        seed=seed,
    )
    true_n_timesteps = n_timesteps if n_timesteps is not None else n_timesteps_user
    # true_n_timesteps = int(1e4)
    # Save a checkpoint every 1000 steps
    folder = "./models"
    os.makedirs(folder, exist_ok=True)
    number_of_flags = 10
    n_steps = model.n_steps if "PPO" in mode else model.train_freq.frequency
    save_freq = ceil((true_n_timesteps / n_steps) * (1 / number_of_flags)) * n_steps
    target_folder = os.path.join("/mnt/nfs_disk/yberthel/data", env_name, mode)
    filename = f"{env_name}_{mode}_{alpha}_{seed}_{int(save_freq*flag)}"
    copy_from_host(
        os.path.join(folder, filename + ".zip"),
        "flanders.gw",
        os.path.join(target_folder, "models", filename + ".zip"),
    )
    assert os.path.exists(os.path.join(folder, filename + ".zip")), f"download failed for {filename}"

    states_filename = f"states_{env_name}_{mode}_{alpha}_{seed}_{int(save_freq*flag)}"
    copy_from_host(
        os.path.join(folder, states_filename + ".pkl"),
        "flanders.gw",
        os.path.join(target_folder, "states", states_filename + ".pkl"),
    )
    assert os.path.exists(os.path.join(folder, states_filename + ".pkl")), f"download failed for {states_filename}"
    states = read_from_pickle(os.path.join(folder, states_filename))
    os.remove(os.path.join(folder, states_filename + ".pkl"))
    buffer_size = model.buffer_size
    old_alt_params = deepcopy(model.get_parameters()["policy"]["alternate_critic.qf0.0.weight"])
    model = model.load(os.path.join(folder, filename))
    assert not torch.equal(
        old_alt_params, model.get_parameters()["policy"]["alternate_critic.qf0.0.weight"]
    ), "alt critic didn't change after loading"
    model.set_env(env)
    # model._setup_learn(
    #     0,
    #     WandbCallback(),
    #     False,
    #     "run",
    #     False,

    # )
    model._logger = utils.configure_logger(0, model.tensorboard_log, "run", False)

    number_of_files_needed = ceil(n_steps / model.replay_buffer.buffer_size)
    buffer_files = []
    for flag_idx in range(1, number_of_files_needed + 1):
        flag_val = (1 / number_of_files_needed) * number_of_flags
        closest_flag_val = find_nearest(list(range(1, number_of_flags + 1)), flag_val)
        buffer_filename = f"replay_buffer_{env_name}_{mode}_{alpha}_{seed}_{closest_flag_val*save_freq}"
        # artifact = wandb.use_artifact(f"{buffer_filename}:latest")
        # datadir = artifact.download()
        if not os.path.exists(os.path.join(folder, buffer_filename + ".pkl")):
            copy_from_host(
                os.path.join(folder, buffer_filename + ".pkl"),
                "flanders.gw",
                os.path.join(target_folder, "replay_buffers", buffer_filename + ".pkl"),
            )
            assert os.path.exists(os.path.join(folder, buffer_filename + ".pkl")), f"download failed for {buffer_filename}"
        buffer_files.append(buffer_filename)

    number_of_files_needed = ceil(flag * save_freq / model.replay_buffer.buffer_size)
    for i in range(number_of_files_needed):
        buffer_filename = buffer_files[i]
        temp_model = agent.load(os.path.join(folder, filename))
        temp_model.load_replay_buffer(os.path.join(folder, buffer_filename))
        # os.remove(os.path.join(folder, buffer_filename + ".pkl"))
        lower_idx = max(0, flag * save_freq - buffer_size * (i + 1))
        uppder_idx = min(flag * save_freq, buffer_size * (i + 1))
        for values in ["next_observations", "actions", "observations", "rewards", "dones", "timeouts"]:
            model.replay_buffer.__dict__[values][lower_idx:uppder_idx] = temp_model.replay_buffer.__dict__[values][
                lower_idx:uppder_idx
            ]
        model.replay_buffer.pos = temp_model.replay_buffer.pos
        model.replay_buffer.full = temp_model.replay_buffer.full
        del temp_model
        os.remove(os.path.join(folder, buffer_filename + ".pkl"))
    os.remove(os.path.join(folder, filename + ".zip"))
    if len(os.listdir(folder)) == 0:
        shutil.rmtree(folder)
    for param, value in hyperparams.items():
        model.__dict__[param] = value
    model.__dict__["n_eval_rollout_steps"] = N_EVAL_TIMESTEPS
    model.__dict__["n_eval_rollout_envs"] = N_EVAL_ENVS
    model._last_obs = model.replay_buffer.__dict__["next_observations"][-1]
    grads, alternate_grads, true_grads, alternate_true_grads, var_grad, alternate_var_grad = model.collect_rollouts_for_grads(
        n_flags=flag,
        number_of_flags=number_of_flags,
        alpha=model.alpha,
        timesteps=flag * save_freq,
        n_rollout_timesteps=N_EVAL_TIMESTEPS,
    )

    def grad_to_list(grads):
        if isinstance(grads[0], torch.Tensor):
            return [grad.tolist() for grad in grads]
        else:
            for i, grad in enumerate(grads):
                grads[i] = grad_to_list(grad)
            return grads

    grads_dict = {
        "grads": grad_to_list(grads),
        "alternate_grads": grad_to_list(alternate_grads),
        "true_grads": grad_to_list(true_grads),
        "alternate_true_grads": grad_to_list(alternate_true_grads),
    }
    filename = f"grads_{env_name}_{mode}_{alpha}_{seed}_{flag*save_freq}"
    save_to_json(grads_dict, filename)
    import pdb

    pdb.set_trace()
    copy_to_host_and_delete(
        filename + ".json",
        "yberthel@flanders.gw",
        os.path.join(target_folder, "grads", filename + ".json"),
    )