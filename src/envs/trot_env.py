"""Policy outputs desired CoM speed for Go1 to track the desired speed."""

import time, sys
from typing import Sequence, Tuple

from ml_collections import ConfigDict
import numpy as np
import torch

from src.utilities.torch_utils import to_torch, torch_rand_float
from src.controllers import qp_torque_optimizer
from src.controllers import phase_gait_generator
from src.controllers import raibert_swing_leg_controller
from src.envs import go1_rewards
from src.robots.common.motors import MotorControlMode

from icecream import ic
from loguru import logger


class TrotEnv:
    def __init__(
        self,
        num_envs: int,
        config: ConfigDict,
        device: str = "cuda",
        show_gui: bool = False,
        use_real_robot: int = 0,  # 0 for isaacgym; 1 for mujoco; 2 for real robot
        robot_name: str = "go1",
    ):
        self._num_envs = num_envs
        self._device = device
        self._show_gui = show_gui
        self._config = config
        self._use_real_robot = use_real_robot

        self._num_actor_obs = config.num_actor_obs
        self._history_length = config.history_length
        self._num_critic_obs = config.num_critic_obs
        self._num_actions = config.num_actions
        self._actor_obs_buf = None
        self._critic_obs_buf = None
        self._action_lb = to_torch(self._config.action_lb, device=self._device)
        self._action_ub = to_torch(self._config.action_ub, device=self._device)

        self._goal_lb = to_torch(self._config.goal_lb, device=self._device)
        self._goal_ub = to_torch(self._config.goal_ub, device=self._device)
        self._construct_observation_and_action_space()

        self._actor_obs_buf = torch.zeros(
            (self._num_envs, self._num_actor_obs), device=self._device
        )
        self._actor_obs_history_buf = torch.zeros(
            (self._num_envs, self._num_actor_obs * self._history_length),
            device=self._device,
        )
        self._critic_obs_buf = torch.zeros(
            (self._num_envs, self._num_critic_obs), device=self._device
        )

        # Set up robot and controller
        use_gpu = "cuda" in device

        if self._use_real_robot == 0:
            from src.utilities.isaacgym_utils import create_sim

            from configs.sim_config import isaac_config
            from src.robots.go1_isaac import Go1Isaac

            self._sim_conf = isaac_config.get_sim_config(
                use_gpu=use_gpu,
                show_gui=show_gui,
                use_penetrating_contact=self._config.get("use_penetrating_contact"),
            )
            self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
            self._robot = Go1Isaac(
                num_envs=self._num_envs,
                sim=self._sim,
                viewer=self._viewer,
                sim_config=self._sim_conf,
                motor_control_mode=MotorControlMode.HYBRID,
                motor_torque_delay_steps=self._config.get("motor_torque_delay_steps"),
                num_actions=self._num_actions,
                num_actor_obs=self._num_actor_obs,
                domain_rand=self._config.get("randomized"),
                terrain_type=self._config.get("terrain"),
                robot_name=robot_name,
            )

        elif self._use_real_robot == 1:
            from src.robots import go1_mujoco

            self._num_envs = 1
            self._robot = go1_mujoco.Go1Mujoco(
                num_envs=self._num_envs,
                device=self._device,
                motor_control_mode=MotorControlMode.HYBRID,
                motor_torque_delay_steps=self._config.get(
                    "motor_torque_delay_steps", 0
                ),  # useless
                robot_name=robot_name,
            )

        elif self._use_real_robot == 2:
            from src.robots import go1_robot

            self._num_envs = 1
            self._robot = go1_robot.Go1Robot(
                num_envs=self._num_envs,
                motor_control_mode=MotorControlMode.HYBRID,
            )

        else:
            logger.error("wrong robot_class")
            exit(-1)

        if self._use_real_robot == 0:
            # Need to set frictions twice to make it work on GPU... 😂
            self._robot.set_foot_frictions(0.01)
            self._robot.set_foot_frictions(self._config.get("foot_friction", 1.0))
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
            self._robot, self._config.gait
        )
        self._swing_leg_controller = (
            raibert_swing_leg_controller.RaibertSwingLegController(
                self._robot,
                self._gait_generator,
                foot_height=self._config.get("swing_foot_height", 0.0),
                foot_landing_clearance=self._config.get(
                    "swing_foot_landing_clearance", 0.0
                ),
            )
        )
        logger.error(f"iter={self._config.get('qp_iter')}")
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            robot=self._robot,
            base_position_kp=self._config.get("base_position_kp"),
            base_position_kd=self._config.get("base_position_kd"),
            base_orientation_kp=self._config.get("base_orientation_kp"),
            base_orientation_kd=self._config.get("base_orientation_kd"),
            weight_ddq=self._config.get("qp_weight_ddq"),
            foot_friction_coef=self._config.get("qp_foot_friction_coef"),
            # body_mass=self._config.get("qp_body_mass"),
            # body_inertia=self._config.get("qp_body_inertia"),
            body_mass=self.robot.robot_param_manager.robot.mass,
            body_inertia=self.robot.robot_param_manager.robot.inertia,
            desired_body_height=self.robot.robot_param_manager.robot.desired_body_height,
            warm_up=self._config.get("qp_warm_up"),
            iter=self._config.get("qp_iter"),
            solver_type=self._config.get("solver_type"),
            friction_type=self._config.get("friction_type"),
        )

        self.common_step_count = 0
        self._steps_count = torch.zeros(self._num_envs, device=self._device)
        self._episode_length = self._config.episode_length_s / self._config.env_dt

        # for smooth actions
        self._action = torch.zeros(
            (self._num_envs, self._num_actions), device=self._device
        )
        self._last_action = self._action.clone()

        self._desired_cmd = torch.zeros((self._num_envs, 3), device=self._device)
        self._resample_command(torch.arange(self._num_envs, device=self._device))

        self._rewards = go1_rewards.Go1Rewards(self)
        self._prepare_rewards()
        self._extras = dict()

        # output range
        logger.warning(
            f"goal_lb={self._config.get('goal_lb')}, goal_ub={self._config.get('goal_ub')}"
        )
        logger.warning(f"qp_body_inertia={self._config.get('qp_body_inertia')}")

        # Running a few steps with dummy commands to ensure JIT compilation
        if self._num_envs == 1 and self._use_real_robot == 2:
            self._robot._robot_interface.send_command(np.zeros(60), 2)
            time.sleep(0.1)
            self._robot._receive_observation()
            self._robot._post_physics_step()
            for state in range(16):
                desired_contact_state = torch.tensor(
                    [[(state & (1 << i)) != 0 for i in range(4)]],
                    dtype=torch.bool,
                    device=self._device,
                )
                for _ in range(3):
                    self._gait_generator.update()
                    self._swing_leg_controller.update()
                    desired_foot_positions = (
                        self._swing_leg_controller.desired_foot_positions
                    )
                    self._torque_optimizer.get_action(
                        foot_contact_state=desired_contact_state,
                        swing_foot_position=desired_foot_positions,
                    )

    # ----------------- Init -----------------
    def _construct_observation_and_action_space(self):
        self._action_lb = to_torch(self._config.action_lb, device=self._device)
        self._action_ub = to_torch(self._config.action_ub, device=self._device)
        robot_lb = to_torch(
            [0.0, -3.14, -3.14, -4.0, -4.0, -10.0, -3.14, -3.14, -3.14]
            + [-0.4, -0.4, -0.7] * 4
            + [-2.0, -2.0, -2.0] * 4
            + [-0.5, -0.5, -0.5] * 4
            + [-3.0, -3.0, -3.0] * 4,
            device=self._device,
        )
        robot_ub = to_torch(
            [0.6, 3.14, 3.14, 4.0, 4.0, 10.0, 3.14, 3.14, 3.14]
            + [0.4, 0.4, 0.7] * 4
            + [2.0, 2.0, 2.0] * 4
            + [0.5, 0.5, 0.5] * 4
            + [3.0, 3.0, 3.0] * 4,
            device=self._device,
        )

        task_lb = to_torch([-2.0, -2.0, -2.0] + [-1.0] * 8, device=self._device)
        task_ub = to_torch([2.0, 2.0, 2.0] + [1.0] * 8, device=self._device)
        self._observation_lb = torch.concatenate((task_lb, robot_lb))
        self._observation_ub = torch.concatenate((task_ub, robot_ub))
        # action smoothness
        self._observation_lb = torch.concatenate(
            (
                self._observation_lb,
                self._action_lb,
            )
        )
        self._observation_ub = torch.concatenate(
            (
                self._observation_ub,
                self._action_ub,
            )
        )

    def _prepare_rewards(self):
        self._reward_names, self._reward_fns, self._reward_scales = [], [], []
        self._episode_sums = dict()
        self._reward_name_scale = dict()
        for name, scale in self._config.rewards:
            self._reward_name_scale[name] = scale
            self._reward_names.append(name)
            self._reward_fns.append(getattr(self._rewards, name + "_reward"))
            self._reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs, device=self._device)

        (
            self._terminal_reward_names,
            self._terminal_reward_fns,
            self._terminal_reward_scales,
        ) = ([], [], [])
        for name, scale in self._config.terminal_rewards:
            self._terminal_reward_names.append(name)
            self._terminal_reward_fns.append(getattr(self._rewards, name + "_reward"))
            self._terminal_reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs, device=self._device)

    # ----------------- Reset -----------------
    def reset(self) -> torch.Tensor:
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids) -> torch.Tensor:
        self._last_action[env_ids, :] = 0.0
        # Aggregate rewards
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            if self._use_real_robot == 0 and self._num_envs > 1:
                self.update_command_curriculum(env_ids)  # command curriculum
            self._extras["episode"] = {}
            self._extras["episode"]["lin_x_command"] = torch.mean(
                torch.abs(self._desired_cmd[:, 0])
            )
            self._extras["episode"]["lin_y_command"] = torch.mean(
                torch.abs(self._desired_cmd[:, 1])
            )
            self._extras["episode"]["ang_yaw_command"] = torch.mean(
                torch.abs(self._desired_cmd[:, 2])
            )

            for reward_name in self._episode_sums.keys():
                if reward_name in self._reward_names:
                    # Normalize by time
                    self._extras["episode"]["reward_{}".format(reward_name)] = (
                        torch.mean(self._episode_sums[reward_name][env_ids])
                    )

                if reward_name in self._terminal_reward_names:
                    self._extras["episode"]["reward_{}".format(reward_name)] = (
                        torch.mean(self._episode_sums[reward_name][env_ids])
                    )

                self._episode_sums[reward_name][env_ids] = 0

            self._steps_count[env_ids] = 0
            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)
            self._resample_command(env_ids)

            self._compute_all_observations()
            self._actor_obs_buf = self.get_actor_observations()
            self._actor_obs_history_buf[env_ids] = self._actor_obs_buf[env_ids].repeat(
                1, self._history_length
            )
            self._critic_obs_buf = self._actor_obs_buf.clone()

        return self._actor_obs_buf, self._actor_obs_history_buf, self._critic_obs_buf

    # ----------------- Step -----------------
    def step(self, action: torch.Tensor):
        # suppose action \in [-1, 1], we need to rescale to action_lb and action_ub
        # action = torch.zeros_like(action)
        min_ = self.action_space[0]
        max_ = self.action_space[1]
        self._last_action = torch.clone(action)
        action = (action + 1) / 2 * (max_ - min_) + min_

        self._steps_count += 1
        self.common_step_count += 1
        self._action = torch.clip(action, self._action_lb, self._action_ub)
        com_action, foot_action = self._split_action(self._action)

        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        logs = []

        desired_lin_vel = torch.zeros((self._num_envs, 3), device=self._device)
        desired_lin_vel[:, :2] = self._desired_cmd[:, :2]
        desired_ang_vel = torch.zeros((self._num_envs, 3), device=self._device)
        desired_ang_vel[:, 2] = self._desired_cmd[:, 2]
        self._torque_optimizer._desired_linear_velocity = desired_lin_vel
        self._torque_optimizer._desired_angular_velocity = desired_ang_vel
        self._torque_optimizer.rl_gains = com_action[:, :]

        for step in range(
            max(int(self._config.env_dt / self._robot.control_timestep), 1)
        ):
            if self._use_real_robot == 2:
                self._robot.state_estimator.update_foot_contact(
                    self._gait_generator.desired_contact_state
                )
                self._robot.update_desired_foot_contact(
                    self._gait_generator.desired_contact_state
                )

            self.gait_generator.update()
            self._swing_leg_controller.update()

            desired_foot_positions = (
                self._swing_leg_controller.with_residuals_desired_foot_positions(
                    rl_action_residuals=foot_action
                )
            )

            (motor_action, self._desired_acc, self._solved_acc, *_) = (
                self._torque_optimizer.get_action(
                    self._gait_generator.desired_contact_state,
                    swing_foot_position=desired_foot_positions,
                )
            )

            if self._use_real_robot == 2 or self._use_real_robot == 1:
                logs.append(
                    dict(
                        timestamp=self._robot.time_since_reset,
                        base_position=torch.clone(self._robot.base_position_world),
                        base_orientation_rpy=torch.clone(
                            self._robot.base_orientation_rpy
                        ),
                        base_velocity=torch.clone(self._robot.base_velocity_body_frame),
                        desired_vel=torch.clone(self._desired_cmd),
                        base_angular_velocity=torch.clone(
                            self._robot.base_angular_velocity_body_frame
                        ),
                        motor_positions=torch.clone(self._robot.motor_positions),
                        motor_velocities=torch.clone(self._robot.motor_velocities),
                        motor_action=motor_action,
                        motor_torques=self._robot.motor_torques,
                        foot_contact_state=self._gait_generator.desired_contact_state,
                        foot_contact_force=self._robot.foot_contact_forces,
                        desired_swing_foot_position=desired_foot_positions,
                        desired_acc_body_frame=self._desired_acc,
                        solved_acc_body_frame=self._solved_acc,
                        foot_positions_in_base_frame=self._robot.foot_positions_in_base_frame,
                        env_action=action,
                        env_obs=torch.clone(self._actor_obs_buf),
                    )
                )
                logs[-1]["base_acc"] = np.array(
                    self._robot.raw_state.imu.accelerometer
                )  # pytype: disable=attribute-error
                logs[-1]["base_gyro"] = np.array(self._robot.raw_state.imu.gyroscope)
                logs[-1]["quat"] = np.array(self._robot.raw_state.imu.quaternion)
                logs[-1]["contact_force"] = self._robot.foot_contact_forces_numpy

            self._robot.step(motor_action)

            rewards = self.get_reward()
            dones = torch.logical_or(dones, self._is_done())
            sum_reward += rewards * torch.logical_not(dones)

        self._extras["logs"] = logs

        # resample commands
        sample_interval = int(self._config.resampling_time / self._config.env_dt)
        env_ids = (
            (self._steps_count % sample_interval == 0).nonzero(as_tuple=False).flatten()
        )
        self._resample_command(env_ids)

        is_terminal = dones.clone()
        if is_terminal.any():
            sum_reward += self.get_terminal_reward(is_terminal, dones)
        if self._use_real_robot == 0:
            self.reset_idx(dones.nonzero(as_tuple=False).flatten())

            if self._show_gui:
                self._robot.render()
                self._robot._action = self._action
                self._robot._actor_obs = self._actor_obs_buf

        self._compute_all_observations()

        # logger.debug(f"self._actor_obs_buf[0]={self._actor_obs_buf[0]}")
        # import ipdb

        # ipdb.set_trace()

        return (
            self._actor_obs_buf,
            self._actor_obs_history_buf,
            self._critic_obs_buf,
            sum_reward,
            dones,
            self._extras,
        )

    def _split_action(self, action):
        com_action = action[:, :6]
        foot_action = action[:, 6:]
        return com_action, foot_action

    # ----------------- Commands -----------------
    def _resample_command(self, env_ids):
        if self._use_real_robot != 0:
            return
        if env_ids.shape[0] == 0:
            return
        self._desired_cmd[env_ids] = torch_rand_float(
            self._goal_lb,
            self._goal_ub,
            [env_ids.shape[0], 3],
            device=self._device,
        )
        self._desired_cmd[env_ids, :2] *= (
            torch.norm(self._desired_cmd[env_ids, :2], dim=1) > 0.1
        ).unsqueeze(1)

    def update_command_curriculum(self, env_ids):
        if (
            self._config.get("use_command_curriculum", False)
            and self.common_step_count > 0
        ):
            if self.common_step_count % self._episode_length == 0:
                lin_vel_reward_value = torch.mean(
                    self._episode_sums["legged_gym_tracking_lin_vel"][env_ids]
                    / self._episode_length
                    / max(int(self._config.env_dt / self._robot.control_timestep), 1)
                )
                reward_scale = self._reward_name_scale["legged_gym_tracking_lin_vel"]
                if (
                    lin_vel_reward_value
                    >= 0.8 * self._reward_name_scale["legged_gym_tracking_lin_vel"]
                ):
                    ic(lin_vel_reward_value, 0.8 * reward_scale)
                    with self._config.unlocked():
                        self._config.goal_ub[0] += 0.1
                        self._config.goal_lb[0] -= 0.1
                    ic(self._config.goal_ub[0])
                ang_vel_reward_value = torch.mean(
                    self._episode_sums["legged_gym_tracking_ang_vel"][env_ids]
                    / self._episode_length
                    / max(int(self._config.env_dt / self._robot.control_timestep), 1)
                )
                if (
                    ang_vel_reward_value
                    >= 0.8 * self._reward_name_scale["legged_gym_tracking_ang_vel"]
                ):
                    pass
                    # ic(ang_vel_reward_value)
                    # with self._config.unlocked():
                    #     self._config.goal_ub[2] += 0.1
                    #     self._config.goal_lb[2] -= 0.1
                    # ic(self._config.goal_ub[2])

    # ----------------- Observations -----------------
    def _compute_all_observations(self):
        # if self._use_real_robot == 1:
        #     self._desired_cmd[:] = 0.0
        if self._use_real_robot == 2:  # for real robot, keep still
            self._desired_cmd[:] = 0.0
            cmd_from_joy = self._robot.get_desired_cmd()
            # print(f"cmd_from_joy: {cmd_from_joy}")
            cmd_from_joy *= np.abs(cmd_from_joy) > 0.1
            self._desired_cmd[:, 0] = cmd_from_joy[0] * 1.0
            self._desired_cmd[:, 1] = cmd_from_joy[1] * 0.6
            self._desired_cmd[:, 2] = cmd_from_joy[2] * 1.0

        # cmd dim: 3
        cmd_obs = self._desired_cmd.clone()
        # phase dim: 2
        phase_obs = torch.stack(
            (
                torch.cos(self._gait_generator.all_phases),
                torch.sin(self._gait_generator.all_phases),
            ),
            dim=1,
        ).view(self.num_envs, -1)
        base_height = self._robot.base_position_world[:, 2:]
        roll = self._robot.base_orientation_rpy[:, 0:1]
        pitch = self._robot.base_orientation_rpy[:, 1:2]
        # base vel
        base_lin_vel = self._robot.base_velocity_body_frame
        # ang vel
        base_ang_vel = self._robot.base_angular_velocity_body_frame
        # foot state
        foot_pos = self._robot.foot_positions_in_base_frame.reshape(
            (self._num_envs, 12)
        )
        foot_vel = (
            self._robot.foot_velocities_in_base_frame.reshape((self._num_envs, 12))
            * 0.1
        )
        # dof pos and dof vel
        dof_pos = self._robot.motor_positions - self._robot.init_motor_angles
        dof_vel = self._robot.motor_velocities * 0.1

        if self._use_real_robot == 0:
            # base state
            noise = 2 * torch.rand_like(base_height) - 1
            base_height += noise * 0.02
            noise = 2 * torch.rand_like(roll) - 1
            roll += noise * 0.05
            noise = 2 * torch.rand_like(pitch) - 1
            pitch += noise * 0.05

            # base vel
            noise = 2 * torch.rand_like(base_lin_vel) - 1
            base_lin_vel += noise * 0.1
            noise = 2 * torch.rand_like(base_ang_vel) - 1
            base_ang_vel += noise * 0.1

            # foot state
            foot_pos_noise = 2 * torch.rand_like(foot_pos) - 1
            foot_pos += foot_pos_noise * 0.005
            foot_vel_noise = 2 * torch.rand_like(foot_vel) - 1
            foot_vel += foot_vel_noise * 0.2 * 0.1

            # dof state
            dof_pos_noise = 2 * torch.rand_like(dof_pos) - 1
            dof_pos += dof_pos_noise * 0.01
            dof_vel_noise = 2 * torch.rand_like(dof_vel) - 1
            dof_vel += dof_vel_noise * 1.5 * 0.1

        robot_obs = torch.concatenate(
            (
                # 3+3+3+12+12+12+12+num_actions
                base_height,  # Base height
                roll,  # Base roll
                pitch,  # Base Pitch
                base_lin_vel,  # Base velocity
                base_ang_vel,  # Base yaw rate
                dof_pos,
                dof_vel,
                foot_pos,
                foot_vel,
                self._action,
            ),
            dim=1,
        )
        self._actor_obs_buf = torch.concatenate((robot_obs, cmd_obs, phase_obs), dim=1)
        self._actor_obs_history_buf = torch.cat(
            (
                self._actor_obs_history_buf[:, self._num_actor_obs :],
                self._actor_obs_buf,
            ),
            dim=-1,
        )
        self._critic_obs_buf = self._actor_obs_buf.clone()

    def get_all_observations(self):
        return self._actor_obs_buf, self._actor_obs_history_buf, self._critic_obs_buf

    def get_actor_observations(self):
        return self._actor_obs_buf

    def get_actor_observations_history(self):
        return self._actor_obs_history_buf.view(
            self._num_envs, self._history_length, self._num_actor_obs
        )

    def get_critic_observations(self):
        return self._critic_obs_buf

    # ----------------- Rewards -----------------
    def get_reward(self):
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        for idx in range(len(self._reward_names)):
            reward_name = self._reward_names[idx]
            reward_fn = self._reward_fns[idx]
            reward_value = reward_fn()
            reward_scale = self._reward_scales[idx]
            reward_item = reward_scale * reward_value
            self._episode_sums[reward_name] += reward_item
            sum_reward += reward_item

        if self._config.clip_negative_reward:
            sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward

    def get_terminal_reward(self, is_terminal, dones):
        early_term = torch.logical_and(
            dones, torch.logical_not(self._episode_terminated())
        )
        coef = torch.where(
            early_term, self._gait_generator.cycle_progress, torch.ones_like(early_term)
        )

        sum_reward = torch.zeros(self._num_envs, device=self._device)
        for idx in range(len(self._terminal_reward_names)):
            reward_name = self._terminal_reward_names[idx]
            reward_fn = self._terminal_reward_fns[idx]
            reward_scale = self._terminal_reward_scales[idx]
            reward_item = reward_scale * reward_fn() * is_terminal * coef
            self._episode_sums[reward_name] += reward_item
            sum_reward += reward_item

        if self._config.clip_negative_terminal_reward:
            sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward

    # ----------------- Termination -----------------
    def _episode_terminated(self):
        return self._steps_count >= self._episode_length

    def _is_done(self):
        is_unsafe = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        # is_unsafe = self._robot.base_position_world[:, 2] < self._config.get(
        #     "terminate_on_height", 0.15
        # )
        if self._config.get("terminate_on_body_contact", False):
            is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)

        if self._config.get("terminate_on_limb_contact", False):
            limb_contact = torch.logical_or(
                self._robot.calf_contacts, self._robot.thigh_contacts
            )
            limb_contact = torch.sum(limb_contact, dim=1)
            is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)
        return torch.logical_or(self._episode_terminated(), is_unsafe)

    def close(self):
        ic(torch.mean(self._action, dim=0))
        ic(torch.mean(self._actor_obs_buf, dim=0))

    @property
    def robot(self):
        return self._robot

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def max_episode_length(self):
        return self._episode_length

    @property
    def episode_length_buf(self):
        return self._steps_count

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._steps_count = to_torch(new_length, device=self._device)

    # ----------------- obs and actions -----------------
    @property
    def action_space(self):
        return self._action_lb, self._action_ub

    @property
    def observation_space(self):
        return self._observation_lb, self._observation_ub

    # ----------------- Common -----------------
    @property
    def device(self):
        return self._device

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_actor_obs(self):
        return self._num_actor_obs

    @property
    def num_critic_obs(self):
        return self._num_critic_obs

    @property
    def history_length(self):
        return self._history_length

    @property
    def num_actions(self):
        return self._num_actions
