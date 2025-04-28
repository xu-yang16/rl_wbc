#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.actor_observations = None
            self.actor_next_observations = None
            self.actor_observation_history = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        actor_obs_history_shape,
        critic_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device

        self.actor_obs_shape = actor_obs_shape
        self.actor_obs_history_shape = actor_obs_history_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.actor_observations = torch.zeros(
            num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device
        )
        self.actor_next_observations = torch.zeros(
            num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device
        )
        self.actor_observation_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actor_obs_history_shape,
            device=self.device,
        )
        self.critic_observations = torch.zeros(
            num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.actor_observations[self.step].copy_(transition.actor_observations)
        self.actor_next_observations[self.step].copy_(
            transition.actor_next_observations
        )
        self.actor_observation_history[self.step].copy_(
            transition.actor_observation_history
        )
        self.critic_observations[self.step].copy_(transition.critic_observations)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values

        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        actor_observations = self.actor_observations.flatten(0, 1)
        actor_next_observations = self.actor_next_observations.flatten(0, 1)
        actor_observation_history = self.actor_observation_history.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                actor_observations_batch = actor_observations[batch_idx]
                actor_next_observations_batch = actor_next_observations[batch_idx]
                actor_observation_history_batch = actor_observation_history[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]

                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                yield actor_observations_batch, actor_next_observations_batch, actor_observation_history_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch

    def encoder_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        actor_next_observations = self.actor_next_observations.flatten(0, 1)
        actor_obs_history = self.actor_observation_history.flatten(0, 1)
        critic_obs = self.critic_observations.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                actor_next_obs_batch = actor_next_observations[batch_idx]
                actor_obs_history_batch = actor_obs_history[batch_idx]
                critic_obs_batch = critic_obs[batch_idx]
                yield actor_next_obs_batch, actor_obs_history_batch, critic_obs_batch
