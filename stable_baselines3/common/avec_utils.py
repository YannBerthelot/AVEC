import os
import wandb
from typing import Optional, Callable
import json
import pickle
from math import ceil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gymnasium import Wrapper, Env
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import mujoco
from stable_baselines3.common.buffers import EvaluationAvecRolloutBuffer
from gymnasium import spaces
from copy import deepcopy
from stable_baselines3.common.utils import obs_as_tensor
from scipy.stats import kendalltau
from stable_baselines3.common.type_aliases import RolloutReturn
from pathlib import Path


def copy_to_host_and_delete(source_file: Path, host: str, host_file: Path) -> None:
    os.system("source /home/yberthel/AVEC/venv/bin/activate")
    os.system(f"scp -r {source_file} {host}:{host_file}")
    if os.path.exists(source_file):
        os.remove(source_file)


def compute_pairwise_from_grads(grads_1, grads_2) -> list:
    similarities = []
    for g_1, g_2 in zip(grads_1, grads_2):
        if g_1.ndim == 1:
            assert g_2.ndim == g_1.ndim
            similarities.append(cosine_similarity(g_1.reshape(1, -1), g_2.reshape(1, -1)).mean())
        else:
            similarities.append(cosine_similarity(g_1, g_2).mean())
    return similarities


def get_state(env):
    is_mujoco = "data" in dir(env.envs[0].unwrapped)
    if is_mujoco:
        state = [{"qvel": env.unwrapped.data.qvel, "qpos": env.unwrapped.data.qpos} for env in env.envs]
    else:
        state = [env.unwrapped.state for env in env.envs]
    return state


def save_to_pickle(obj, filename):
    with open(f"{filename}.pkl", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{filename}.pkl"


def read_from_pickle(filename):
    with open(f"{filename}.pkl", "rb") as handle:
        return pickle.load(handle)


def save_to_json(obj, filename):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def read_from_json(filename):
    with open(f"{filename}.json") as data_file:
        return json.load(data_file)


MUJOCO_NAMES = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "HumanoidStandup",
    "Reacher",
    "Walker2D",
    "Swimmer",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
]


def get_fixed_reset_state_env(env_name: str, num_envs: int, states):
    # env = gym.make(env_name, max_episode_steps=int(1e3))
    is_mujoco = env_name.split("-")[0] in MUJOCO_NAMES
    if is_mujoco:
        wrapper = MujocoResetWrapper
    else:
        wrapper = ClassicControlWrapper
    env = make_vec_env(env_id=env_name, n_envs=num_envs, wrapper_class=wrapper, wrapper_kwargs={"state": states})

    # env = make_vec_env(MujocoResetWrapper, n_envs=num_envs, env_kwargs={"env": env, "state": states})
    return env


class ClassicControlWrapper(Wrapper):
    def __init__(self, env: Env, state):
        super().__init__(env)
        self.state = state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.env.reset(seed=seed)

        self.env.unwrapped.state = self.state
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        if "_get_obs" in dir(self.env.unwrapped):
            obs = self.env.unwrapped._get_obs()
        elif "_get_ob" in dir(self.env.unwrapped):
            obs = self.env.unwrapped._get_ob()
        else:
            obs = np.array(self.state, dtype=np.float32)
        return obs, {}


class MujocoResetWrapper(Wrapper):
    def __init__(self, env: Env, state):
        super().__init__(env)
        self.state = state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.env.reset(seed=seed)

        mujoco.mj_resetData(self.unwrapped.model, self.unwrapped.data)

        ob = self.reset_model()
        info = self.env.unwrapped._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def reset_model(self):
        qpos = self.state["qpos"]
        qvel = self.state["qvel"]
        self.env.unwrapped.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()
        return observation


def load_artifacts(
    number_of_flags,
    env_name,
    correction,
    alpha,
    seed,
    true_algo_name: str,
    mode: str = "grads",
    grads_folder: str = "grads",
    value_folder: str = "value",
):
    if mode == "grads":
        folder = grads_folder
    elif mode == "value":
        folder = value_folder
    algo_name = f"CORRECTED_AVEC_{true_algo_name}" if correction else f"AVEC_{true_algo_name}"
    for training_frac in range(1, number_of_flags + 1):
        filename = f"{mode}_{env_name}_{algo_name}_{alpha}_{seed}_{training_frac*10}"
        try:
            artifact = wandb.use_artifact(f"{filename}:latest")
            datadir = artifact.download()
            os.rename(os.path.join(datadir, filename + ".pkl"), os.path.join(folder, filename + ".pkl"))

        except:  # wandb.errors.ArtifactNotLoggedError: somehow this does not exist despite being in the doc
            continue


def compute_true_grads(self, alpha: float) -> list:
    old_last_obs = deepcopy(self._last_obs)
    old_episode_starts = deepcopy(self._last_episode_starts)
    true_grads = get_true_grads_from_policy(
        self, alpha, env_name=self.env_name, num_envs=self.n_eval_rollout_envs, n_steps=self.n_eval_rollout_steps
    )
    self._last_obs, self._last_episode_starts = old_last_obs, old_episode_starts
    return true_grads


def get_true_grads_from_policy(self, alpha: float, env_name: str, num_envs: int = 1, n_steps: int = int(1e5)):
    env = make_vec_env(env_id=env_name, n_envs=num_envs)
    self._last_obs = env.reset()
    n_rollout_steps = ceil(n_steps / num_envs)
    if "PPO" in self.true_algo_name:
        rollout_buffer = self.rollout_buffer_class(
            n_rollout_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=num_envs,
            **self.rollout_buffer_kwargs,
        )
        self.collect_rollouts(
            env=env,
            callback=None,
            n_rollout_steps=n_rollout_steps,
            rollout_buffer=rollout_buffer,
            flag=False,
            update=False,
            value_function_eval=False,
        )
        self.train(update=False, n_epochs=1, alpha=alpha)
    else:
        rollout_buffer = self.replay_buffer_class(
            n_rollout_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            n_envs=num_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs,
        )
        self.collect_rollouts(
            env=env,
            callback=None,
            train_freq=n_rollout_steps,
            replay_buffer=rollout_buffer,
            flag=False,
            update=False,
            value_function_eval=False,
        )
        self.train(update=False, gradient_steps=1, alpha=alpha, batch_size=self.batch_size)

    return deepcopy(self.grads)


def compute_or_load_true_grads(
    self,
    n_flags: int,
    number_of_flags: int,
    alpha: float,
) -> list:
    os.makedirs(self.grads_folder, exist_ok=True)
    training_frac = int((n_flags - 1) * 100 / number_of_flags)
    algo_name = f"CORRECTED_AVEC_{self.true_algo_name}" if self.correction else f"AVEC_{self.true_algo_name}"
    filename = f"grads_{self.env_name}_{algo_name}_{alpha}_{alpha}_{self.seed}_{training_frac}"
    if filename not in os.listdir(self.grads_folder):
        true_grads = compute_true_grads(self, alpha=alpha)
        grads_path = os.path.join(self.grads_folder, filename)
        save_to_pickle(true_grads, grads_path)
        wandb.log_artifact(artifact_or_path=grads_path + ".pkl", name=filename, type="grads")
    else:
        true_grads = read_from_pickle(filename)
    return true_grads


def evaluate_and_log_grads(
    self,
    num_rollout,
    pairwise_similarities,
    avec_pairwise_similarities,
    update,
    n_gradient_rollouts,
    true_grads,
    avec_true_grads,
):
    if "PPO" in self.true_algo_name:
        self.train(update=False, n_epochs=1)
        grads = deepcopy(self.grads)
        self.train(update=False, n_epochs=1, alpha=0.0)
        avec_grads = deepcopy(self.grads)
    else:
        self.train(update=False, gradient_steps=1)
        grads = deepcopy(self.grads)
        self.train(update=False, gradient_steps=1, alpha=0.0)
        avec_grads = deepcopy(self.grads)
    if self.old_grads is not None:
        pairwise_cosine_sim = compute_pairwise_from_grads(grads, self.old_grads)
        avec_pairwise_cosine_sim = compute_pairwise_from_grads(avec_grads, self.old_avec_grads)
        pairwise_similarities.append(np.mean(pairwise_cosine_sim))
        avec_pairwise_similarities.append(np.mean(avec_pairwise_cosine_sim))
    if num_rollout == 1:  # TODO : check that it goes as intended
        if "PPO" in self.true_algo_name:
            self.train(update=False, n_epochs=1, alpha=self.alpha)
        else:
            self.train(update=False, batch_size=self.batch_size, gradient_steps=1, alpha=self.alpha)
        true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, true_grads)
        average_true_gradient_pairwise_cosine_sim = np.mean(true_gradient_pairwise_cosine_sim)
        if "PPO" in self.true_algo_name:
            self.train(update=False, n_epochs=1, alpha=self.alpha)
        else:
            self.train(update=False, batch_size=self.batch_size, gradient_steps=1, alpha=self.alpha)
        avec_true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, true_grads)
        avec_average_true_gradient_pairwise_cosine_sim = np.mean(avec_true_gradient_pairwise_cosine_sim)
        avec_avec_true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, avec_true_grads)
        avec_avec_average_true_gradient_pairwise_cosine_sim = np.mean(avec_avec_true_gradient_pairwise_cosine_sim)
        self.logger.record("gradients/convergence to the true gradients", average_true_gradient_pairwise_cosine_sim)
        self.logger.record("gradients/avec convergence to the true gradients", avec_average_true_gradient_pairwise_cosine_sim)
        self.logger.record(
            "gradients/avec convergence to the true avec gradients",
            avec_avec_average_true_gradient_pairwise_cosine_sim,
        )
    if update:
        assert len(pairwise_similarities) == n_gradient_rollouts - 1, f"{len(pairwise_similarities)}"
        self.logger.record("gradients/average pairwise cosine sim", np.mean(pairwise_similarities))
        self.logger.record("gradients/avec average pairwise cosine sim", np.mean(avec_pairwise_similarities))

        self.logger.dump(step=self.num_timesteps)
    return grads, avec_grads, pairwise_similarities, avec_pairwise_similarities


def compute_true_values(self, state, action=None):
    state = state[0]  # TODO : find how to fix this?
    if "PPO" in self.true_algo_name:
        eval_buffer = collect_rollouts_MC_from_state(
            self,
            self.n_eval_rollout_envs,
            state,
            n_rollout_steps=ceil(self.n_eval_rollout_steps / self.n_eval_rollout_envs),
        )
    else:
        assert action is not None, f"action is none for sac : {action=}"
        eval_buffer = collect_rollouts_MC_from_state_and_actions(
            self,
            self.n_eval_rollout_envs,
            state,
            action,
            n_rollout_steps=self.n_eval_rollout_steps,
        )
    states_values_MC = eval_buffer.returns[eval_buffer.episode_starts.astype("bool")]
    nb_full_episodes = eval_buffer.episode_starts.sum() - 1
    states_values_MC = (
        states_values_MC[: -eval_buffer.episode_starts.shape[1]]
        if eval_buffer.episode_starts.ndim > 1
        else states_values_MC[:-1]
    )  # remove last one(s) as it is not a full episode

    MC_episode_lengths = [
        len(x) + 1 for x in "".join(eval_buffer.episode_starts.T.flatten().astype("int").astype("str")).split("1")[1:]
    ]
    return states_values_MC, MC_episode_lengths, nb_full_episodes


def compute_or_load_true_values(
    self, state, n_flags: int, number_of_flags: int, alpha: float, state_idx: int, true_algo_name: str, action=None
) -> list:
    os.makedirs(self.value_folder, exist_ok=True)
    training_frac = int((n_flags - 1) * 100 / number_of_flags)
    algo_name = f"CORRECTED_AVEC_{true_algo_name}" if self.correction else f"AVEC_{true_algo_name}"
    filename = f"value_{self.env_name}_{algo_name}_{alpha}_{self.seed}_{training_frac}_{state_idx}_{self.n_eval_rollout_envs}_{self.n_eval_rollout_steps}"
    if filename not in os.listdir(self.value_folder):
        states_values_MC, MC_episode_lengths, nb_full_episodes = compute_true_values(self, state, action)
        values_path = os.path.join(self.value_folder, filename)

        save_to_json(
            {
                "value": states_values_MC if isinstance(states_values_MC, list) else states_values_MC.tolist(),
                "episode_length": (
                    MC_episode_lengths if isinstance(MC_episode_lengths, list) else MC_episode_lengths.tolist()
                ),
                "nb_full_episodes": nb_full_episodes,
            },
            values_path,
        )
        wandb.log_artifact(
            artifact_or_path=values_path + ".json", name=filename, type="value"
        )  # Logs the artifact version "my_data" as a dataset with data from dataset.h5
    else:
        states_values_MC, MC_episode_lengths, nb_full_episodes = read_from_json(filename).values()
    return states_values_MC, MC_episode_lengths, nb_full_episodes


def collect_rollouts_MC_from_state(
    self,
    num_envs,
    states,
    n_rollout_steps: int,
) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param rollout_buffer: Buffer to fill with rollouts
    :param n_rollout_steps: Number of experiences to collect per environment
    :return: True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    """
    assert self._last_obs is not None, "No previous observation was provided"
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)
    _last_episode_starts = np.ones((num_envs,), dtype=bool)
    n_steps = 0
    env = get_fixed_reset_state_env(self.env_name, num_envs, states)
    _last_obs = env.reset()
    evaluation_rollout_buffer = EvaluationAvecRolloutBuffer(
        buffer_size=n_rollout_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        gamma=self.gamma,
        n_envs=num_envs,
        gae_lambda=1.0,  # MC
    )
    evaluation_rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if self.use_sde:
        self.policy.reset_noise(env.num_envs)

    while n_steps < n_rollout_steps:

        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(_last_obs, self.device)
            if "PPO" in self.true_algo_name:
                actions = self.policy(obs_tensor)[0]
                actions = actions.cpu().numpy()
            else:
                actions = self._sample_action(self.learning_starts, self.action_noise, env.num_envs)[0]

        # Rescale and perform action
        clipped_actions = actions

        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions = self.policy.unscale_action(clipped_actions)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        n_steps += num_envs

        if isinstance(self.action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(dones):
            if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += self.gamma * terminal_value
        evaluation_rollout_buffer.add(
            _last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            _last_episode_starts,  # type: ignore[arg-type]
        )

        _last_obs = new_obs  # type: ignore[assignment]
        _last_episode_starts = dones

    evaluation_rollout_buffer.compute_returns_and_advantage(dones=dones)

    return evaluation_rollout_buffer


def collect_rollouts_MC_from_state_and_actions(
    self,
    num_envs,
    states,
    first_actions,
    n_rollout_steps: int,
) -> RolloutReturn:
    """
    Collect experiences and store them into a ``ReplayBuffer``.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param train_freq: How much experience to collect
        by doing rollouts of current policy.
        Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
        or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
        with ``<n>`` being an integer greater than 0.
    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param replay_buffer:
    :param log_interval: Log data every ``log_interval`` episodes
    :return:
    """
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)

    _last_episode_starts = np.ones((num_envs,), dtype=bool)
    n_steps = 0
    env = get_fixed_reset_state_env(self.env_name, num_envs, states)
    _last_obs = env.reset()
    evaluation_rollout_buffer = EvaluationAvecRolloutBuffer(
        buffer_size=ceil(n_rollout_steps / num_envs),
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_envs=num_envs,
        device=self.device,
        gamma=self.gamma,
    )
    evaluation_rollout_buffer.reset()

    num_collected_steps, num_collected_episodes = 0, 0

    if self.use_sde:
        self.actor.reset_noise(env.num_envs)
    from tqdm import tqdm

    for n_steps in tqdm(range(0, n_rollout_steps, num_envs)):
        # while n_steps < n_rollout_steps:
        if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        # actions, buffer_actions = self._sample_action(self.learning_starts, self.action_noise, env.num_envs)
        # actions, _ = self.actor.action_log_prob(th.Tensor(_last_obs))
        unscaled_actions, _ = self.predict(_last_obs, deterministic=False)
        if isinstance(self.action_space, spaces.Box):
            scaled_actions = self.policy.scale_action(unscaled_actions)

            # Add noise to the action (improve exploration)
            if self.action_noise is not None:
                scaled_actions = np.clip(scaled_actions + self.action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_actions = scaled_actions
            actions = self.policy.unscale_action(scaled_actions)
        else:
            # Discrete case, no need to normalize or clip
            buffer_actions = unscaled_actions
            actions = buffer_actions
        # Rescale and perform action
        num_actions = len(actions)
        actual_actions = np.array(
            [first_actions[0] if _last_episode_starts[i] else actions[i] for i in range(num_actions)]
        ).reshape(num_envs, self.action_space.shape[0])

        new_obs, rewards, dones, infos = env.step(actual_actions)
        # n_steps += num_envs

        # if update:
        #     # Give access to local variables
        #     callback.update_locals(locals())
        #     # Only stop training if return value is False, not when it is None.
        #     if not callback.on_step():
        #         return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

        # Store data in replay buffer (normalized action and unnormalized observation)
        evaluation_rollout_buffer.add(
            _last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            _last_episode_starts,  # type: ignore[arg-type]
        )

        _last_obs = new_obs  # type: ignore[assignment]
        _last_episode_starts = dones

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1

                if self.action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    self.action_noise.reset(**kwargs)
    evaluation_rollout_buffer.compute_returns_and_advantage(dones=dones)
    return evaluation_rollout_buffer


def evaluate_value_function(
    self,
    state,
    n_flags,
    number_of_flags,
    alpha,
    n_steps,
    true_algo_name,
    action=None,
):
    states_values_MC, MC_episode_lengths, nb_full_episodes = compute_or_load_true_values(
        self, state, n_flags, number_of_flags, alpha, state_idx=n_steps, true_algo_name=true_algo_name, action=action
    )  # TODO : adapt for q values
    return states_values_MC.mean(axis=0)
    assert (np.sum(np.isnan(states_values_MC)) == 0) and len(states_values_MC) > 0, f"{states_values_MC=}"
    MC_values = np.concatenate((MC_values, states_values_MC), axis=None)
    predicted_values.append(values.detach().numpy()[0])
    true_value = states_values_MC.mean(axis=0)
    true_values.append(true_value)
    value_error = (true_value - values.detach().numpy()) ** 2
    value_errors.append(value_error)
    normalized_value_errors.append(value_error / (MC_values.mean(axis=0) ** 2))
    return (
        predicted_values,
        true_values,
        value_errors,
        normalized_value_errors,
        MC_values,
    )


def ranking_and_error_logging(
    self,
    predicted_values,
    true_values,
    deltas,
    normalized_value_errors,
    value_errors,
    alternate_values=None,
    alternate_value_errors=None,
    alternate_normalized_value_errors=None,
    alternate_deltas=None,
    timesteps=None,
):
    kendal_tau = kendalltau(np.array(predicted_values), np.array(true_values))
    kendal_tau_stat = kendal_tau.statistic
    assert not np.isnan(kendal_tau_stat), f"{predicted_values=} {true_values=}"
    self.logger.record("ranking/Kendal Tau statistic", kendal_tau_stat)
    self.logger.record("ranking/Kendal Tau p-value", kendal_tau.pvalue)

    if deltas is not None:
        self.logger.record("errors/value estimation error mean", abs(np.mean(deltas)))
        self.logger.record("errors/value estimation error std", np.std(deltas))
        error_difference = np.mean((np.mean(value_errors) - np.mean(deltas)) ** 2)
        self.logger.record("errors/errors difference", error_difference)

    self.logger.record("errors/normalized value approximation error mean", np.mean(normalized_value_errors))
    self.logger.record("errors/normalized value approximation error std", np.std(normalized_value_errors))
    self.logger.record("errors/value approximation error mean", np.mean(value_errors))
    self.logger.record("errors/value approximation error std", np.std(value_errors))

    self.logger.record("values/predicted value mean", np.mean(predicted_values))
    self.logger.record("values/predicted value std", np.std(predicted_values))
    self.logger.record("values/true value mean", np.mean(true_values))
    self.logger.record("values/true value std", np.std(true_values))

    outlyingness = (true_values - np.median(true_values)) / np.median(abs(true_values - np.median(true_values)))

    self.logger.record("outliers/mean abs outlyingness", np.mean(abs(outlyingness)))
    self.logger.record("outliers/std abs outlyingness", np.std(abs(outlyingness)))
    self.logger.record("outliers/min outlyingness", np.min(outlyingness))
    self.logger.record("outliers/max outlyingness", np.max(outlyingness))
    self.logger.record("outliers/median abs outlyingness", np.median(outlyingness))
    self.logger.record("outliers/25p outlyingness", np.quantile(outlyingness, 0.25))
    self.logger.record("outliers/75p outlyingness", np.quantile(outlyingness, 0.75))

    if alternate_values is not None:
        kendal_tau = kendalltau(np.array(alternate_values), np.array(true_values))
        kendal_tau_stat = kendal_tau.statistic
        assert not np.isnan(kendal_tau_stat), f"{alternate_values=} {true_values=}"
        self.logger.record("ranking/alternate Kendal Tau statistic", kendal_tau_stat)
        self.logger.record("ranking/alternate Kendal Tau p-value", kendal_tau.pvalue)
        if alternate_deltas is not None:
            self.logger.record("errors/alternate value estimation error mean", abs(np.mean(alternate_deltas)))
            self.logger.record("errors/alternate value estimation error std", np.std(alternate_deltas))
            error_difference = np.mean((np.mean(alternate_value_errors) - np.mean(alternate_deltas)) ** 2)
            self.logger.record("errors/alternate errors difference", error_difference)

        self.logger.record(
            "errors/alternate normalized value approximation error mean", np.mean(alternate_normalized_value_errors)
        )
        self.logger.record(
            "errors/alternate normalized value approximation error std", np.std(alternate_normalized_value_errors)
        )
        self.logger.record("errors/alternate value approximation error mean", np.mean(alternate_value_errors))
        self.logger.record("errors/alternate value approximation error std", np.std(alternate_value_errors))

        self.logger.record("values/alternate predicted value mean", np.mean(alternate_values))
        self.logger.record("values/alternate predicted value std", np.std(alternate_values))

    self.logger.dump(step=self.num_timesteps)
    corrected = "CORRECTED_" if self.correction else ""
    mode = f"{corrected}AVEC_{self.true_algo_name}"
    name_prefix = f"{self.env_name}_{mode}_{self.alpha}_{self.seed}"
    filename = f"{name_prefix}_{timesteps}"
    save_to_pickle(true_values, filename)
    copy_to_host_and_delete(
        filename + ".pkl",
        "flanders.gw",
        os.path.join("/mnt/data/yberthel/data", "true_values_" + filename + ".pkl"),
    )


def get_grad_from_net(net, grads=None) -> list:
    if grads is None:
        grads = []

    for parameter in net.parameters():
        grads.append(parameter.grad)
    return grads
