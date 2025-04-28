#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os, time
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, MLP_Encoder, EmpiricalNormalization
from rsl_rl.env import VecEnv


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.encoder_cfg = train_cfg["encoder"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        encoder_output_dim = self.encoder_cfg["num_output_dim"]
        num_actor_obs_with_latent = self.env.num_actor_obs + encoder_output_dim
        num_actor_obs_history = self.env.num_actor_obs * self.env.history_length
        num_critic_obs_with_latent = self.env.num_critic_obs + encoder_output_dim

        # normalizer
        self.actor_obs_normalizer = EmpiricalNormalization(
            shape=[self.env.num_actor_obs], until=1.0e8
        ).to(self.device)
        self.actor_obs_history_normalizer = EmpiricalNormalization(
            shape=[num_actor_obs_history], until=1.0e8
        ).to(self.device)
        self.critic_obs_normalizer = EmpiricalNormalization(
            shape=[self.env.num_critic_obs], until=1.0e8
        ).to(self.device)

        if encoder_output_dim != 0:
            encoder: MLP_Encoder = MLP_Encoder(
                num_input_dim=num_actor_obs_history,
                num_output_dim=encoder_output_dim,
                **self.policy_cfg,
            ).to(self.device)
        else:
            encoder = None
        actor_critic: ActorCritic = ActorCritic(
            num_actor_obs_with_latent,
            num_critic_obs_with_latent,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self.alg: PPO = PPO(
            actor_critic=actor_critic,
            encoder=encoder,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_actor_obs],
            [num_actor_obs_history],
            [num_critic_obs_with_latent],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        actor_obs, actor_obs_history, critic_obs = self.env.get_all_observations()
        actor_obs, actor_obs_history, critic_obs = self.normalize_obs(
            actor_obs, actor_obs_history, critic_obs
        )
        actor_obs, actor_obs_history, critic_obs = (
            actor_obs.to(self.device),
            actor_obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, actor_obs_history, critic_obs)
                    (
                        actor_obs,
                        actor_obs_history,
                        critic_obs,
                        rewards,
                        dones,
                        infos,
                        *_,
                    ) = self.env.step(actions)
                    actor_obs, actor_obs_history, critic_obs = self.normalize_obs(
                        actor_obs, actor_obs_history, critic_obs
                    )
                    actor_obs, actor_obs_history, critic_obs, rewards, dones = (
                        actor_obs.to(self.device),
                        actor_obs_history.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(
                        rewards, dones, infos, actor_next_obs=actor_obs
                    )

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.alg.encoder is not None:
                    encoder_out = self.alg.encoder.encode(actor_obs_history)
                    self.alg.compute_returns(
                        torch.cat((critic_obs, encoder_out), dim=-1)
                    )
                else:
                    self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_encoder_loss,
                mean_surrogate_loss,
                mean_entropy_loss,
                mean_kl,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.env.close()
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_latest.pt"))

    def normalize_obs(self, actor_obs, actor_obs_history, critic_obs):
        # normalize obs
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actor_obs_history = self.actor_obs_history_normalizer(actor_obs_history)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return actor_obs, actor_obs_history, critic_obs

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # curriculum
        # ic(locs["ep_infos"])
        self.writer.add_scalar(
            "Curriculum/lin_x_command", locs["ep_infos"][0]["lin_x_command"], locs["it"]
        )
        self.writer.add_scalar(
            "Curriculum/lin_y_command", locs["ep_infos"][0]["lin_y_command"], locs["it"]
        )
        self.writer.add_scalar(
            "Curriculum/ang_yaw_command",
            locs["ep_infos"][0]["ang_yaw_command"],
            locs["it"],
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_encoder_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "actor_obs_norm_state_dict": self.actor_obs_normalizer.state_dict(),
                "actor_obs_history_norm_state_dict": self.actor_obs_history_normalizer.state_dict(),
                "critic_obs_norm_state_dict": self.critic_obs_normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location="cpu", weights_only=False)
        # -- Load model
        resumed_training = self.alg.actor_critic.load_state_dict(
            loaded_dict["model_state_dict"]
        )
        if resumed_training:
            # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
            # and the critic/teacher normalizer is loaded for the critic/teacher
            self.actor_obs_normalizer.load_state_dict(
                loaded_dict["actor_obs_norm_state_dict"]
            )
            self.actor_obs_history_normalizer.load_state_dict(
                loaded_dict["actor_obs_history_norm_state_dict"]
            )
            self.critic_obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.policy.act_inference
        if device is not None:
            self.actor_obs_normalizer.to(device)
        policy = lambda x: self.alg.policy.act_inference(
            self.actor_obs_normalizer(x)
        )  # noqa: E731
        return policy

    def train_mode(self):
        # switch to train mode (for dropout for example)
        self.alg.actor_critic.train()

        # normalizer
        self.actor_obs_normalizer.train()
        self.actor_obs_history_normalizer.train()
        self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()

        # normalizer
        self.actor_obs_normalizer.eval()
        self.actor_obs_history_normalizer.eval()
        self.critic_obs_normalizer.eval()
