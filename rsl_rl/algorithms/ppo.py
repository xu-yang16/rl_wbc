#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, MLP_Encoder
from rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        encoder: MLP_Encoder,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_learning_rate=1e-2,
        min_learning_rate=1e-5,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate

        # PPO components
        self.actor_critic = actor_critic.to(self.device)

        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.encoder = encoder
        if encoder is not None:
            self.encoder = encoder.to(self.device)
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=learning_rate
            )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        actor_obs_history_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            actor_obs_history_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, actor_obs, actor_obs_history, critic_obs):
        # act
        if self.encoder is not None:
            encoder_out = self.encoder.encode(actor_obs_history)
            self.transition.actions = self.actor_critic.act(
                torch.cat((actor_obs, encoder_out), dim=-1)
            ).detach()
            # evaluate
            critic_obs_include_latent = torch.cat((critic_obs, encoder_out), dim=-1)
            self.transition.values = self.actor_critic.evaluate(
                critic_obs_include_latent
            ).detach()
            self.transition.critic_observations = critic_obs_include_latent
        else:
            self.transition.actions = self.actor_critic.act(actor_obs).detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
            self.transition.critic_observations = critic_obs

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_observations = actor_obs
        self.transition.actor_observation_history = actor_obs_history

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, actor_next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.transition.actor_next_observations = actor_next_obs
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_entropy_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            actor_obs_batch,
            _,  # actor_next_obs_batch
            actor_obs_history_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            if self.encoder is not None:
                encoder_out_batch = self.encoder.encode(actor_obs_history_batch)
                self.actor_critic.act(
                    torch.cat((actor_obs_batch, encoder_out_batch), dim=-1)
                )
            else:
                self.actor_critic.act(actor_obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            self.min_learning_rate, self.learning_rate / 1.5
                        )
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            self.max_learning_rate, self.learning_rate * 1.5
                        )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            # if kl_mean > 1.5 * self.desired_kl:  # early stopping
            #     print("early stop, num_updates =", num_updates)
            #     break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += self.value_loss_coef * value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += -self.entropy_coef * entropy_batch.mean().item()
            mean_kl += kl_mean.item()

        mean_encoder_loss = 0
        if self.encoder is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                actor_next_obs_batch,
                actor_obs_history_batch,
                critic_obs_batch,
            ) in generator:
                self.encoder.encode(actor_obs_history_batch)
                encode_batch = self.encoder.get_encoder_out()

                encoder_loss = (
                    (
                        encode_batch
                        - actor_next_obs_batch[:, : self.encoder.num_output_dim]
                    )
                    .pow(2)
                    .mean()
                )

                self.encoder_optimizer.zero_grad()
                encoder_loss.backward()
                self.encoder_optimizer.step()

                mean_encoder_loss += encoder_loss.item()

        mean_value_loss /= num_updates
        mean_encoder_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_encoder_loss,
            mean_surrogate_loss,
            mean_entropy_loss,
            mean_kl,
        )
