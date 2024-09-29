#!/home/yberthel/AVEC/venv/bin/python

from stable_baselines3.common.avec_utils import set_model, copy_to_host_and_delete, save_to_json

import wandb
import sys
import torch
import os
from dotenv import load_dotenv
from copy import deepcopy

DEFAULT_N_STEPS = 2048
DEFAULT_BUFFER_SIZE = 1000000


if __name__ == "__main__":
    seed = int(sys.argv[1])
    env_name = str(sys.argv[2])
    mode = str(sys.argv[3])
    assert "PPO" in mode or "SAC" in mode, f"Unrecognized mode {mode}"
    n_steps_factor = float(sys.argv[4])
    network_size_factor = float(sys.argv[5])
    alpha = float(sys.argv[6])
    n_timesteps_user = int(eval(sys.argv[7]))
    n_eval_timesteps = int(eval(sys.argv[8]))
    n_samples_MC = int(sys.argv[9])
    n_eval_envs = int(sys.argv[10])
    asked_flag = int(sys.argv[11])
    load_dotenv("/home/yberthel/AVEC/.env")
    LOCAL = eval(os.environ.get("LOCAL"))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(dir_path, "ppo.yml" if "PPO" in mode else "sac.yml")

    agent_name = "PPO" if "PPO" in mode else "SAC"
    number = 7 if "PPO" in mode else 6
    prefix = " local" if LOCAL else ""
    run = wandb.init(
        project=f"avec experiments {agent_name} {number}{prefix}",
        sync_tensorboard=True,
        config={
            "agent": mode,
            "mode": mode,
            "env": env_name,
            "seed": seed,
            "rollout size factor": n_steps_factor,
            "critic network size factor": network_size_factor,
            "alpha": alpha,
            "type_of_job": "evaluate",
        },
        mode="offline",
    )
    for flag in range(1, 11):
        if asked_flag != 0:
            if flag != asked_flag:
                continue
        model, states, number_of_flags, save_freq, env, folder, buffer_filename, target_folder = set_model(
            seed,
            env_name,
            mode,
            flag,
            alpha,
            n_steps_factor,
            network_size_factor,
            n_eval_timesteps,
            n_eval_envs,
            n_samples_MC,
            run,
            yaml_path,
        )
        grads, alternate_grads, true_grads, alternate_true_grads, var_grad, alternate_var_grad = (
            model.collect_rollouts_for_grads(
                n_flags=flag,
                number_of_flags=number_of_flags,
                alpha=model.alpha,
                timesteps=flag * save_freq,
                n_rollout_timesteps=n_eval_timesteps,
                replay_buffer=deepcopy(model.replay_buffer),
            )
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
        copy_to_host_and_delete(
            filename + ".json",
            "yberthel@flanders.gw",
            os.path.join(target_folder, "grads", filename + ".json"),
        )
        run_path = "/" + os.path.join(*run.dir.split("/")[:-1])
        if not (os.path.exists(run_path)):
            wandb_folder = "/" + os.path.join(*run.dir.split("/")[:-2])
            if f"run-{run.id}.wandb" in os.listdir(os.path.join(wandb_folder, "latest-run")):
                run_path = os.path.join(wandb_folder, "latest-run")
            else:
                raise ValueError(f"Run {run.id} could not be found")
    run.finish()
    if LOCAL:
        os.system(f"wandb sync {run_path}")
    else:
        os.system(f"source /home/yberthel/AVEC/venv/bin/activate && wandb sync {run_path} && wandb artifact cache cleanup 1GB")
        os.system(f". /home/yberthel/AVEC/venv/bin/activate && wandb sync {run_path} && wandb artifact cache cleanup 1GB")
    # os.system(f"wandb sync --clean --include-offline --clean-force {run_path}")
    # else:
    #     os.system(f"wandb sync --sync-tensorboard {run_path} --append --id {run.id}")
