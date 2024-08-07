import os
import pickle
import pdb
import sys
import time
import wandb
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from math import ceil, floor
import numpy as np
import torch as th
from sklearn.metrics.pairwise import cosine_similarity
from gymnasium import spaces
import gymnasium as gym
import wandb.errors
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from math import ceil
from copy import deepcopy, copy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, AvecRolloutBuffer, RolloutBuffer, EvaluationAvecRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

from stable_baselines3.common.env_util import make_vec_env

from gymnasium import Wrapper

GRADS_FOLDER = "grads"
N_GRADIENT_ROLLOUTS = 10
VALUE_FUNCTION_EVAL = False
number_of_flags = 10


def compute_pairwise_from_grads(grads_1, grads_2) -> list:
    similarities = []
    for g_1, g_2 in zip(grads_1, grads_2):
        if g_1.ndim == 1:
            assert g_2.ndim == g_1.ndim
            similarities.append(cosine_similarity(g_1.reshape(1, -1), g_2.reshape(1, -1)).mean())
        else:
            similarities.append(cosine_similarity(g_1, g_2).mean())
    return similarities


def save_to_pickle(obj, filename):
    with open(f"{filename}.pkl", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(filename):
    with open(f"{filename}.pkl", "rb") as handle:
        return pickle.load(handle)


def get_fixed_reset_state_env(env_name: str, num_envs: int, states):
    # env = gym.make(env_name, max_episode_steps=int(1e3))
    env = make_vec_env(env_id=env_name, n_envs=num_envs, wrapper_class=MujocoResetWrapper, wrapper_kwargs={"state": states})

    # env = make_vec_env(MujocoResetWrapper, n_envs=num_envs, env_kwargs={"env": env, "state": states})
    return env


def get_state_from_mujoco(env):
    return env.unwrapped.data


import mujoco
from gymnasium import Env


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

        mujoco.mj_resetData(self.model, self.data)

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


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.old_grads = None
        self.grads = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
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

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

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

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        N_GRADIENT_ROLLOUTS = 11
        pairwise_similarities = []
        while self.num_timesteps < total_timesteps:
            for num_rollout in range(1, N_GRADIENT_ROLLOUTS + 1):
                update = num_rollout == N_GRADIENT_ROLLOUTS
                continue_training = self.collect_rollouts(
                    self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, update=update
                )

                if not continue_training:
                    break

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    assert self.ep_info_buffer is not None
                    self._dump_logs(iteration)

                self.train(update=update)
                if self.old_grads is not None:
                    average_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, self.old_grads).mean()
                    pairwise_similarities.append(average_pairwise_cosine_sim)
                    self.old_grads = self.grads
                if self.old_policy_params is not None:
                    pairwise_sims = get_pairwise_sim_from_nets_params(self.policy.parameters(), self.old_policy_params)
                    pdb.set_trace()
                self.old_policy_params = deepcopy(self.policy.parameters())
        assert len(pairwise_similarities == N_GRADIENT_ROLLOUTS)
        if len(pairwise_similarities) > 0:
            self.logger.record(
                "gradients/average pairwise cosine sim", np.mean(pairwise_similarities), iteration=self.num_timesteps
            )
        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


class AvecOnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: AvecRolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[AvecRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        alpha: float = 0.0,
        correction: bool = False,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        n_eval_rollout_steps: int = int(1e5),
        n_eval_rollout_envs: int = 32,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.alpha = alpha
        self.correction = correction
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.n_eval_rollout_steps = n_eval_rollout_steps
        self.n_eval_rollout_envs = n_eval_rollout_envs
        self.num_eval_timesteps = 0
        self.grads = None
        self.old_grads = None
        self.old_policy_params = None
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = AvecRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            correction=self.correction,
            alpha=self.alpha,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AvecRolloutBuffer,
        n_rollout_steps: int,
        flag: bool,
        update: bool = True,
        value_function_eval: bool = False,
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
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        if update:
            callback.on_rollout_start()

        N_SAMPLES = 100
        if flag and value_function_eval:
            pbar = tqdm(total=N_SAMPLES, desc="Collecting")
        samples = np.random.choice(n_rollout_steps, N_SAMPLES)
        value_errors = []
        normalized_value_errors = []
        predicted_values = []
        MC_values = np.array([])
        # pbar = tqdm(total=n_rollout_steps, desc="Collecting")
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

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
            if update:
                self.num_timesteps += env.num_envs
            if flag and (n_steps in samples) and value_function_eval:
                self.num_eval_timesteps += 1
                pbar.update(1)
                state = [{"qvel": env.unwrapped.data.qvel, "qpos": env.unwrapped.data.qpos} for env in env.envs]
                state = state[0]  # TODO : find how to fix this?
                eval_buffer = self.collect_rollouts_MC_from_state(
                    self.n_eval_rollout_envs,
                    state,
                    n_rollout_steps=ceil(self.n_eval_rollout_steps / self.n_eval_rollout_envs),
                )
                states_values_MC = eval_buffer.returns[eval_buffer.episode_starts.astype("bool")]
                nb_full_episodes = eval_buffer.episode_starts.sum() - 1
                states_values_MC = (
                    states_values_MC[: -eval_buffer.episode_starts.shape[1]]
                    if eval_buffer.episode_starts.ndim > 1
                    else states_values_MC[:-1]
                )  # remove last one(s) as it is not a full episode
                MC_values = np.concatenate((MC_values, states_values_MC), axis=None)
                MC_episode_lengths = [
                    len(x) + 1
                    for x in "".join(eval_buffer.episode_starts.T.flatten().astype("int").astype("str")).split("1")[1:]
                ]
                predicted_values.append(values.detach().numpy())
                value_error = (states_values_MC.mean(axis=0) - values.detach().numpy()) ** 2
                value_errors.append(value_error)
                normalized_value_errors.append(value_error / (MC_values.mean(axis=0) ** 2))
                self.logger.record("MC/MC episode mean length", np.mean(MC_episode_lengths))
                self.logger.record("MC/MC episode std length", np.std(MC_episode_lengths))
                self.logger.record("MC/number of complete trajectories", nb_full_episodes)
                self.logger.record("value/value MC mean", np.mean(MC_values))
                self.logger.record("value/value MC std", np.std(MC_values))
                self.logger.record("value/value std (variance)", np.std(predicted_values))
                self.logger.record(
                    "value/normalized value std (variance)", np.std(predicted_values) / np.mean(predicted_values)
                )
                self.logger.record("value/value mean", np.mean(predicted_values))
                self.logger.record("value/eval step", self.num_eval_timesteps)
                self.logger.record("errors/error std", np.std(value_errors))
                self.logger.record("errors/error mean (bias)", np.mean(value_errors))
                self.logger.record("errors/normalized error mean (bias)", np.mean(normalized_value_errors))
                self.logger.record("errors/normalized error std", np.std(normalized_value_errors))
                self.logger.dump(step=self.num_timesteps)

            # Give access to local variables
            if update:
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
            if update:
                self._update_info_buffer(infos, dones)
            n_steps += env.num_envs
            # pbar.update(env.num_envs)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            # TODO : measure how long the episodes that are kept are

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if flag and value_function_eval:
            self.logger.record("errors/normalized value approximation error mean", np.mean(normalized_value_errors))
            self.logger.record("errors/normalized value approximation error std", np.std(normalized_value_errors))
            self.logger.record("errors/value estimation error mean", np.mean(rollout_buffer.deltas))
            self.logger.record("errors/value approximation error mean", np.mean(value_errors))
            self.logger.record("errors/value approximation error std", np.std(value_errors))
            self.logger.record("errors/value estimation error std", np.std(rollout_buffer.deltas))
            error_difference = np.mean((np.mean(value_errors) - np.mean(rollout_buffer.deltas)) ** 2)
            self.logger.record("errors/errors difference", error_difference)

        if update:
            callback.update_locals(locals())
            callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def get_true_grads_from_policy(self, alpha: float, env_name: str, num_envs: int = 32, n_steps: int = int(1e6)):
        env = make_vec_env(env_id=env_name, n_envs=num_envs)
        self._last_obs = env.reset()
        rollout_buffer = self.rollout_buffer_class(
            ceil(n_steps / num_envs),
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=num_envs,
            **self.rollout_buffer_kwargs,
        )
        self.collect_rollouts(
            env,
            None,
            rollout_buffer,
            n_rollout_steps=n_steps,
            flag=False,
            update=False,
            value_function_eval=False,
        )
        self.train(update=False, n_epochs=1, alpha=alpha)
        return deepcopy(self.grads)

    def load_artifacts(self, number_of_flags):
        algo_name = "CORRECTED_AVEC_PPO" if self.correction else "AVEC_PPO"
        for training_frac in range(1, number_of_flags + 1):
            filename = f"{self.env_name}_{algo_name}_{self.alpha}_{self.seed}_{training_frac*10}"
            try:
                artifact = wandb.use_artifact(f"{filename}:latest")
                datadir = artifact.download()
                os.rename(os.path.join(datadir, filename + ".pkl"), os.path.join(GRADS_FOLDER, filename + ".pkl"))

            except:  # wandb.errors.ArtifactNotLoggedError: somehow this does not exist despite being in the doc
                continue

    def compute_true_grads(self, alpha) -> list:
        old_last_obs = deepcopy(self._last_obs)
        old_episode_starts = deepcopy(self._last_episode_starts)
        true_grads = self.get_true_grads_from_policy(alpha, env_name=self.env_name)
        self._last_obs = old_last_obs
        self._last_episode_starts = old_episode_starts
        return true_grads

    def compute_or_load_true_grads(self, n_flags: int, number_of_flags: int, alpha: float) -> list:
        os.makedirs(GRADS_FOLDER, exist_ok=True)
        training_frac = int((n_flags - 1) * 100 / number_of_flags)
        algo_name = "CORRECTED_AVEC_PPO" if self.correction else "AVEC_PPO"
        filename = f"{self.env_name}_{algo_name}_{self.alpha}_{alpha}_{self.seed}_{training_frac}"
        if filename not in os.listdir(GRADS_FOLDER):
            true_grads = self.compute_true_grads(alpha)
            grads_path = os.path.join(GRADS_FOLDER, filename)
            save_to_pickle(true_grads, grads_path)
            wandb.log_artifact(
                artifact_or_path=grads_path + ".pkl", name=filename, type="dataset"
            )  # Logs the artifact version "my_data" as a dataset with data from dataset.h5
        else:
            true_grads = read_from_pickle(filename)
        return true_grads

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        self.load_artifacts(number_of_flags)

        TOTAL_UPDATES = total_timesteps // self.n_steps

        n_flags = 1
        while self.num_timesteps < total_timesteps:
            flag = (
                (self.num_timesteps // (int((n_flags / number_of_flags) * total_timesteps)) > 0) and self.num_timesteps > 0
            ) or (self._n_updates / self.n_epochs) == (TOTAL_UPDATES - 1)
            pairwise_similarities = []
            avec_pairwise_similarities = []
            self.old_grads = None
            self.old_avec_grads = None
            self.old_policy_params = None
            grads = None
            avec_grads = None
            n_iterations = 2 if (VALUE_FUNCTION_EVAL or not (flag)) else N_GRADIENT_ROLLOUTS + 1
            if flag:
                n_flags += 1
                true_grads = self.compute_or_load_true_grads(n_flags, number_of_flags, alpha=self.alpha)
                avec_true_grads = self.compute_or_load_true_grads(n_flags, number_of_flags, alpha=0.0)

            for num_rollout in range(1, n_iterations):
                update = num_rollout == n_iterations - 1
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                    flag=flag,
                    update=update,
                    value_function_eval=VALUE_FUNCTION_EVAL,
                )

                if not continue_training:
                    break
                if update:
                    iteration += 1
                    self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    assert self.ep_info_buffer is not None
                    self._dump_logs(iteration)
                if flag:
                    self.train(update=False, epoch=1)
                    grads = deepcopy(self.grads)
                    self.train(update=False, epoch=1, alpha=0.0)
                    avec_grads = deepcopy(self.grads)
                if self.old_grads is not None and flag:
                    assert self.old_avec_grads is not None
                    pairwise_cosine_sim = compute_pairwise_from_grads(grads, self.old_grads)
                    avec_pairwise_cosine_sim = compute_pairwise_from_grads(avec_grads, self.old_avec_grads)
                    pairwise_similarities.append(np.mean(pairwise_cosine_sim))
                    avec_pairwise_similarities.append(np.mean(avec_pairwise_cosine_sim))
                if flag and num_rollout == 1:  # TODO : check that it goes as intended
                    self.train(update=False, n_epochs=1, alpha=self.alpha)
                    true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, true_grads)
                    average_true_gradient_pairwise_cosine_sim = np.mean(true_gradient_pairwise_cosine_sim)
                    self.train(update=False, n_epochs=1, alpha=0.0)
                    avec_true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, true_grads)
                    avec_average_true_gradient_pairwise_cosine_sim = np.mean(avec_true_gradient_pairwise_cosine_sim)
                    avec_avec_true_gradient_pairwise_cosine_sim = compute_pairwise_from_grads(self.grads, avec_true_grads)
                    avec_avec_average_true_gradient_pairwise_cosine_sim = np.mean(avec_avec_true_gradient_pairwise_cosine_sim)
                if flag and update:
                    assert len(pairwise_similarities) == N_GRADIENT_ROLLOUTS - 1, f"{len(pairwise_similarities)}"
                    self.logger.record("fraction of training steps", int((n_flags - 1) * 100 / number_of_flags))
                    self.logger.record("gradients/average pairwise cosine sim", np.mean(pairwise_similarities))
                    self.logger.record("gradients/avec average pairwise cosine sim", np.mean(pairwise_similarities))
                    self.logger.record(
                        "gradients/convergence to the true gradients", average_true_gradient_pairwise_cosine_sim
                    )
                    self.logger.record(
                        "gradients/avec convergence to the true gradients", avec_average_true_gradient_pairwise_cosine_sim
                    )
                    self.logger.record(
                        "gradients/avec convergence to the true avec gradients",
                        avec_avec_average_true_gradient_pairwise_cosine_sim,
                    )
                    self.logger.dump(step=self.num_timesteps)
                if update:
                    self.train(update=update)
                if flag:
                    self.old_grads = deepcopy(grads)
                    self.old_avec_grads = deepcopy(avec_grads)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

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
        num_timesteps = 0
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
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
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
            # TODO : investigate why no environment fail early?
            n_steps += 1
            num_timesteps += num_envs

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
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
