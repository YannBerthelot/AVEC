#!/home/yberthel/AVEC/venv/bin/python
from stable_baselines3.common.avec_utils import set_model
import wandb
import sys
import os
from dotenv import load_dotenv


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

        model, states, number_of_flags, save_freq, env, folder, buffer_filename, _ = set_model(
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
        if "SAC" in mode:
            model.collect_rollouts_for_eval(
                states,
                model.replay_buffer,
                n_flags=flag,
                number_of_flags=number_of_flags,
                alpha=model.alpha,
                timesteps=flag * save_freq,
            )
        else:
            model.collect_rollouts_for_eval(
                env,
                rollout_buffer=model.rollout_buffer,
                n_rollout_steps=model.n_steps,
                alpha=model.alpha,
                n_flags=flag,
                timesteps=flag * save_freq,
            )

        run_path = "/" + os.path.join(*run.dir.split("/")[:-1])
        if not (os.path.exists(run_path)):
            wandb_folder = "/" + os.path.join(*run.dir.split("/")[:-2])
            if f"run-{run.id}.wandb" in os.listdir(os.path.join(wandb_folder, "latest-run")):
                run_path = os.path.join(wandb_folder, "latest-run")
            else:
                raise ValueError(f"Run {run.id} could not be found")
    run.finish()
    if "SAC" in mode:
        os.remove(os.path.join(folder, buffer_filename + ".pkl"))
    os.system(f"source /home/yberthel/AVEC/venv/bin/activate && wandb sync {run_path} && wandb artifact cache cleanup 1GB")
    os.system(f". /home/yberthel/AVEC/venv/bin/activate && wandb sync {run_path} && wandb artifact cache cleanup 1GB")
