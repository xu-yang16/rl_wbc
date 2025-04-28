"""Environment wrappers for normalizing RL environments."""

from absl import logging

import torch


class AttributeModifier:
    def __getattr__(self, name):
        return getattr(self._env, name)

    def set_attribute(self, name, value):
        set_attr = getattr(self._env, "set_attribute", None)
        if callable(set_attr):
            self._env.set_attribute(name, value)
        else:
            setattr(self._env, name, value)

    @property
    def episode_length_buf(self):
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._env.episode_length_buf = new_length


class RangeNormalize(AttributeModifier):
    def __init__(self, env):
        self._env = env
        self._device = self._env.device

        self.obs_history = torch.zeros(
            self._env.num_envs,
            self._env.history_length * self._env.num_actor_obs,
            device=self._device,
        )
        self.num_obs = self._env.history_length * self._env.num_actor_obs

    @property
    def observation_space(self):
        low, high = self._env.observation_space
        return -torch.ones_like(low), torch.ones_like(low)

    @property
    def action_space(self):
        low, high = self._env.action_space
        return -torch.ones_like(low), torch.ones_like(low)

    def step(self, action):
        action = self._denormalize_action(action)
        observ, privileged_obs, reward, done, info = self._env.step(action)
        observ = self._normalize_observ(observ)

        self.obs_history = torch.cat(
            (self.obs_history[:, self._env.num_obs :], observ), dim=-1
        )
        return self.obs_history, privileged_obs, reward, done, info

    def reset(self):
        actor_obs, actor_obs_history, critic_obs = self._env.reset()
        observ = self._normalize_observ(observ)
        self.obs_history = torch.cat(
            (self.obs_history[:, self._env.num_obs :], observ), dim=-1
        )
        return self.obs_history, privileged_obs

    def _denormalize_action(self, action):
        min_ = self._env.action_space[0]
        max_ = self._env.action_space[1]
        action = (action + 1) / 2 * (max_ - min_) + min_
        return action

    def _normalize_observ(self, observ):
        min_ = self._env.observation_space[0]
        max_ = self._env.observation_space[1]
        observ = 2 * (observ - min_) / (max_ - min_) - 1
        return observ

    def get_observations(self):
        current_obs = self._normalize_observ(self._env.get_observations())
        self.obs_history = torch.cat(
            (self.obs_history[:, self._env.num_obs :], current_obs), dim=-1
        )
        return self.obs_history

    def close(self):
        self._env.close()


class ClipAction(AttributeModifier):
    """Clip out of range actions to the action space of the environment."""

    def __init__(self, env):
        self._env = env

    def step(self, action):
        action_space = self._env.action_space
        action = torch.clip(action, action_space[0], action_space[1])
        return self._env.step(action)
