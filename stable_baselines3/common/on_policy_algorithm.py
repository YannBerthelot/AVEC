import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from tqdm import tqdm

from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, AvecRolloutBuffer, RolloutBuffer, EvaluationAvecRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.avec_utils import (
    load_artifacts,
    compute_or_load_true_grads,
    evaluate_and_log_grads,
    evaluate_value_function,
    ranking_and_error_logging,
    get_state,
)
from numpy.random import default_rng

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


N_GRADIENT_ROLLOUTS = 10
VALUE_FUNCTION_EVAL = False
GRAD_EVAL = False
TRUE_ALGO_NAME = "PPO"
number_of_flags = 10


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
            self.states.append(get_state(env))
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

            # Handle timeout by bootstrapping with value function
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

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

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
        n_eval_rollout_steps: int = int(1e6),
        n_eval_rollout_envs: int = 32,
        n_samples_MC: int = 100,
        grads_folder: str = "grads",
        value_folder: str = "values",
        grad_eval: bool = False,
        value_eval: bool = False,
    ):
        self.true_algo_name = TRUE_ALGO_NAME
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
        self.n_samples_MC = n_samples_MC
        self.grads_folder = grads_folder
        self.value_folder = value_folder
        self.true_algo_name = TRUE_ALGO_NAME
        self.previous_update_start = 0
        self.samples = []
        self.states = []

        self.value_errors = []
        self.normalized_value_errors = []
        self.predicted_values = []
        self.predicted_alternate_values = []
        self.alternate_value_errors = []
        self.alternate_normalized_value_errors = []
        self.true_values = []
        self.MC_values = []
        self.previous_buffer_size = 0
        self.alternate_critic = None

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
        self.alternate_policy = deepcopy(self.policy)
        self.policy = self.policy.to(self.device)
        self.alternate_policy = self.alternate_policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AvecRolloutBuffer,
        n_rollout_steps: int,
        update: bool = True,
        collect_states: bool = False,
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

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                _, alternate_values, _ = self.alternate_policy(obs_tensor)
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
            if collect_states:
                self.states.append(get_state(env))
            if update:
                self.num_timesteps += env.num_envs
            # Give access to local variables
            if update:
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
            if update:
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
                alternate_value=alternate_values,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            # TODO : measure how long the episodes that are kept are

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones, alternate_last_values=alternate_values)
        if update:
            callback.update_locals(locals())
            callback.on_rollout_end()
        return True

    def collect_rollouts_for_eval(
        self,
        env: VecEnv,
        rollout_buffer: AvecRolloutBuffer,
        n_rollout_steps: int,
        alpha: float = None,
        n_flags: int = None,
        timesteps=None,
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
        # Switch to eval mode (this affects batch norm / dropout)
        self._last_obs = env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        self.policy.set_training_mode(False)

        self.collect_rollouts(env, None, rollout_buffer, n_rollout_steps, update=False, collect_states=True)
        rng = default_rng(self.seed)
        self.samples = rng.choice(
            len(self.states) - 1, self.n_samples_MC
        )  # TODO : find how to handle with update at each step
        deltas = []
        alternate_deltas = []
        MC_episode_lengths = []
        nb_full_episodes = []

        for i in self.samples:
            state = self.states[i]
            MC_values, MC_episode_length, nb_full_episode = evaluate_value_function(
                self,
                state,
                n_flags,
                number_of_flags,
                alpha,
                self.num_timesteps + i,
                TRUE_ALGO_NAME,
            )
            MC_episode_lengths.append(MC_episode_length)
            nb_full_episodes.append(nb_full_episode)

            true_value = MC_values.mean(axis=0)
            value_error = (true_value - rollout_buffer.values[i]) ** 2
            normalized_value_error = value_error / (true_value**2)
            deltas.append(rollout_buffer.deltas[i])
            self.predicted_values.append(rollout_buffer.values[i])
            self.true_values.append(true_value)
            self.MC_values.append(MC_values)
            self.value_errors.append(value_error)
            self.normalized_value_errors.append(normalized_value_error)

            alternate_value_error = (true_value - rollout_buffer.alternate_values[i]) ** 2
            alternate_normalized_value_error = alternate_value_error / (true_value**2)
            alternate_deltas.append(rollout_buffer.alternate_advantages[i])
            self.predicted_alternate_values.append(rollout_buffer.alternate_values[i])
            self.alternate_value_errors.append(alternate_value_error)
            self.alternate_normalized_value_errors.append(alternate_normalized_value_error)

        ranking_and_error_logging(
            self,
            predicted_values=self.predicted_values,
            true_values=self.true_values,
            deltas=deltas,
            normalized_value_errors=self.normalized_value_errors,
            value_errors=self.value_errors,
            timesteps=timesteps,
            MC_episode_lengths=MC_episode_lengths,
            nb_full_episodes=nb_full_episodes,
            alternate_deltas=alternate_deltas,
            alternate_normalized_value_errors=self.alternate_normalized_value_errors,
            alternate_value_errors=self.alternate_value_errors,
            alternate_values=self.predicted_alternate_values,
            MC_values=self.MC_values,
        )

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
        assert not (GRAD_EVAL and VALUE_FUNCTION_EVAL), "Can only use either grad or value eval"
        if GRAD_EVAL:
            mode = "grads"
        elif VALUE_FUNCTION_EVAL:
            mode = "value"
        else:
            mode = None

        TOTAL_UPDATES = total_timesteps // self.n_steps

        n_flags = 1
        while self.num_timesteps < total_timesteps:
            flag = (
                ((self.num_timesteps + self.n_steps) // (int((n_flags / number_of_flags) * total_timesteps)) > 0)
                and self.num_timesteps > 0
            ) or (self.num_timesteps + self.n_steps >= total_timesteps)
            if flag:
                n_flags += 1
                fraction_of_training_steps = int((n_flags - 1) * 100 / number_of_flags)
                print(self.num_timesteps, n_flags, flag, fraction_of_training_steps)
            pairwise_similarities = []
            avec_pairwise_similarities = []
            self.old_grads = None
            self.old_avec_grads = None
            self.old_policy_params = None
            grads = None
            avec_grads = None
            if VALUE_FUNCTION_EVAL:
                assert not GRAD_EVAL
                n_iterations = 1
            elif GRAD_EVAL and flag:
                n_iterations = N_GRADIENT_ROLLOUTS
            else:
                n_iterations = 1

            if flag:

                if GRAD_EVAL:
                    true_grads = compute_or_load_true_grads(
                        self,
                        n_flags,
                        number_of_flags,
                        alpha=self.alpha,
                    )
                    avec_true_grads = compute_or_load_true_grads(
                        self,
                        n_flags,
                        number_of_flags,
                        alpha=0.0,
                    )

            for num_rollout in range(1, n_iterations + 1):
                update = num_rollout == n_iterations
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                    update=update,
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
                if GRAD_EVAL and flag:
                    grads, avec_grads, pairwise_similarities, avec_pairwise_similarities = evaluate_and_log_grads(
                        self,
                        num_rollout,
                        pairwise_similarities,
                        avec_pairwise_similarities,
                        update,
                        N_GRADIENT_ROLLOUTS,
                        true_grads,
                        avec_true_grads,
                    )

                if update:
                    self.train(update=update)
                if flag:
                    self.logger.record("fraction of training steps", fraction_of_training_steps)
                    self.logger.dump(step=self.num_timesteps)
                    if GRAD_EVAL:
                        self.old_grads = deepcopy(grads)
                        self.old_avec_grads = deepcopy(avec_grads)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def collect_rollouts_for_grads(
        self,
        timesteps: int,
        alpha: float = None,
        n_flags: int = None,
        number_of_flags: int = None,
        n_iterations: int = 10,
        alternate_alpha: float = 0.5,
        n_rollout_timesteps: float = int(1e6),
    ):
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
        self.n_eval_rollout_steps = n_rollout_timesteps
        print("Computing true_grads")
        # self.replay_buffer = deepcopy(replay_buffer)
        true_grads = compute_or_load_true_grads(
            self,
            n_flags,
            number_of_flags,
            alpha=alpha,
        )
        alternate_true_grads = compute_or_load_true_grads(
            self,
            n_flags,
            number_of_flags,
            alpha=alternate_alpha,
        )
        print("Finished Computing true_grads")
        pairwise_similarities, alternate_pairwise_similarities = [], []
        grads_list, alternate_grads_list = [], []
        print("Computing consecutive grads")
        for num_rollout in tqdm(range(1, n_iterations + 1)):
            update = num_rollout == n_iterations
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # self.replay_buffer = deepcopy(replay_buffer)

            rollout = self.collect_rollouts(
                self.env,
                callback=None,
                rollout_buffer=self.rollout_buffer,
                update=False,
                n_rollout_steps=self.n_eval_rollout_steps,
            )

            if self.num_timesteps > 0:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = 1
                # Special case when the user passes `gradient_steps=0`

                grads, alternate_grads, pairwise_similarities, alternate_pairwise_similarities = evaluate_and_log_grads(
                    self,
                    num_rollout,
                    pairwise_similarities,
                    alternate_pairwise_similarities,
                    update,
                    N_GRADIENT_ROLLOUTS,
                    true_grads,
                    alternate_true_grads,
                    timesteps=timesteps,
                )
                grads_list.append(grads)
                alternate_grads_list.append(alternate_grads)
                self.train(n_epochs=gradient_steps)
                self.old_grads = deepcopy(grads)
                self.old_alternate_grads = deepcopy(alternate_grads)
        var_grad = np.var(pairwise_similarities)
        self.logger.record(
            "gradients/variance of the gradients",
            var_grad,
        )
        alternate_var_grad = np.var(alternate_pairwise_similarities)
        self.logger.record(
            "gradients/variance of the alternate gradients",
            alternate_var_grad,
        )
        return grads_list, alternate_grads_list, true_grads, alternate_true_grads, var_grad, alternate_var_grad
