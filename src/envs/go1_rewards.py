"""Set of rewards for the Go1 robot."""

import torch


class Go1Rewards:
    """Set of rewards for Go1 robot."""

    def __init__(self, env):
        self._env = env
        self._robot = self._env.robot
        self._gait_generator = self._env.gait_generator
        self._num_envs = self._env.num_envs
        self._device = self._env.device

    # ---------------------- pos ---------------------- #
    def height_reward(self):
        return -torch.square(self._robot.base_position_world[:, 2] - 0.26)

    # ---------------------- orientation ---------------------- #
    def upright_reward(self):
        return self._robot.projected_gravity[:, 2]

    # ---------------------- velocity ---------------------- #
    def speed_tracking_reward(self):
        actual_speed = torch.concatenate(
            (
                self._robot.base_velocity_body_frame[:, :2],
                self._robot.base_angular_velocity_body_frame[:, 2:],
            ),
            dim=1,
        )
        return -torch.sum(torch.square(self._env._desired_cmd - actual_speed), dim=1)

    def exp_speed_tracking_reward(self):
        actual_speed = torch.concatenate(
            (
                self._robot.base_velocity_body_frame[:, :2],
                self._robot.base_angular_velocity_body_frame[:, 2:],
            ),
            dim=1,
        )
        return torch.exp(
            -4 * torch.sum(torch.square(self._env._desired_cmd - actual_speed), dim=1)
        )

    def lin_vel_z_reward(self):
        return -torch.square(self._robot.base_velocity_body_frame[:, 2])

    def ang_vel_xy_reward(self):
        return -torch.sum(
            torch.square(self._robot.base_angular_velocity_body_frame[:, :2]), dim=1
        )

    def forward_speed_reward(self):
        return self._robot.base_velocity_body_frame[:, 0]

    def roll_pitch_reward(self):
        rpy = self._robot.base_orientation_rpy
        return -torch.sum(torch.square(rpy[:, :2]), dim=1)

    # ---------------------- action penalty ---------------------- #
    def action_penalty_reward(self):
        return -torch.sum(torch.square(self._env._action), dim=1)

    def acc_action_penalty_reward(self):
        return -torch.sum(torch.square(self._env._action[:, :6]), dim=1)

    def foot_action_penalty_reward(self):
        return -torch.sum(torch.square(self._env._action[:, -6:]), dim=1)

    # ---------------------- alive ---------------------- #
    def alive_reward(self):
        return torch.ones(self._num_envs, device=self._device)

    def foot_slipping_reward(self):
        foot_slipping = (
            torch.sum(
                self._gait_generator.desired_contact_state
                * torch.sum(
                    torch.square(self._robot.foot_velocities_in_world_frame[:, :, :2]),
                    dim=2,
                ),
                dim=1,
            )
            / 4
        )
        foot_slipping = torch.clip(foot_slipping, 0, 1)
        return -foot_slipping

    def foot_clearance_reward(self, foot_height_thres=0.02):
        desired_contacts = self._gait_generator.desired_contact_state
        foot_height = self._robot.foot_height - 0.02  # Foot radius
        # print(f"Foot height: {foot_height}")
        foot_height = torch.clip(foot_height, 0, foot_height_thres) / foot_height_thres
        foot_clearance = (
            torch.sum(torch.logical_not(desired_contacts) * foot_height, dim=1) / 4
        )

        return foot_clearance

    def foot_clearance_v2_reward(self, foot_height_thres=0.02):
        desired_contacts = self._gait_generator.desired_contact_state
        foot_height = self._robot.foot_height - 0.1
        foot_height = torch.clip(foot_height, max=0)
        return -torch.sum(torch.logical_not(desired_contacts) * foot_height, dim=1) / 4

    def foot_force_reward(self):
        """Swing leg should not have contact force."""
        foot_forces = torch.norm(self._robot.foot_contact_forces, dim=2)
        calf_forces = torch.norm(self._robot.calf_contact_forces, dim=2)
        thigh_forces = torch.norm(self._robot.thigh_contact_forces, dim=2)
        limb_forces = (foot_forces + calf_forces + thigh_forces).clip(max=10)
        foot_mask = torch.logical_not(self._gait_generator.desired_contact_state)

        return -torch.sum(limb_forces * foot_mask, dim=1) / 4

    def cost_of_transport_reward(self):
        motor_power = torch.abs(
            0.3 * self._robot.motor_torques**2
            + self._robot.motor_torques * self._robot.motor_velocities
        )
        commanded_vel = torch.sqrt(torch.sum(self._env.command[:, :2] ** 2, dim=1))
        return -torch.sum(motor_power, dim=1) / commanded_vel

    def legged_gym_torques_reward(self):
        return -torch.sum(torch.square(self._robot.motor_torques), dim=1)

    def legged_gym_tracking_lin_vel_reward(self):
        lin_vel_error = torch.sum(
            torch.square(
                self._robot.base_velocity_body_frame[:, :2]
                - self._env._desired_cmd[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error * 4)

    def legged_gym_tracking_ang_vel_reward(self):
        ang_vel_error = torch.sum(
            torch.square(
                self._robot.base_angular_velocity_body_frame[:, 2:]
                - self._env._desired_cmd[:, 2:]
            ),
            dim=1,
        )
        return torch.exp(-ang_vel_error * 4)

    def legged_gym_lin_vel_z_reward(self):
        return -torch.square(self._robot.base_velocity_body_frame[:, 2])

    def legged_gym_ang_vel_xy_reward(self):
        return -torch.sum(
            torch.square(self._robot.base_angular_velocity_body_frame[:, :2]), dim=1
        )

    def legged_gym_dof_acc_reward(self):
        return -torch.sum(
            torch.square(
                (self._robot.motor_velocities - self._robot._last_motor_velocities)
                / 0.01
            ),
            dim=1,
        )

    def legged_gym_feet_air_time_reward(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self._robot.foot_contact_forces[:, :, 2] > 1.0
        contact_filt = torch.logical_or(contact, self._robot._last_contacts)
        self._robot._last_contacts = contact
        first_contact = (self._robot.feet_air_time > 0.0) * contact_filt
        self._robot.feet_air_time += 0.01
        rew_airTime = torch.sum(
            (self._robot.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self._env._desired_cmd[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self._robot.feet_air_time *= ~contact_filt
        return rew_airTime

    def legged_gym_action_rate_reward(self):
        return -torch.sum(
            torch.square(self._env._last_action[:, :6] - self._env._action[:, :6])
            / 0.1,
            dim=1,
        )

    def legged_gym_feet_clearance_reward(self):
        feet_height_error = torch.square(self._robot.foot_height - 0.1)
        return -torch.sum(
            feet_height_error
            * torch.norm(self._robot._foot_velocities[:, :, :2], dim=2),
            dim=1,
        )

    def energy_consumption_reward(self):
        motor_power = torch.clip(
            0.3 * self._robot.motor_torques**2
            + self._robot.motor_torques * self._robot.motor_velocities,
            min=0,
        )
        return -torch.clip(torch.sum(motor_power, dim=1), min=0.0, max=2000.0)

    def contact_consistency_reward(self):
        desired_contact = self._gait_generator.desired_contact_state
        actual_contact = torch.logical_or(
            self._robot.foot_contacts, self._robot.calf_contacts
        )
        actual_contact = torch.logical_or(actual_contact, self._robot.thigh_contacts)
        # print(f"Actual contact: {actual_contact}")
        return torch.sum(desired_contact == actual_contact, dim=1) / 4

    def distance_to_goal_reward(self):
        base_position = self._robot.base_position_world
        return -torch.sqrt(
            torch.sum(
                torch.square(
                    base_position[:, :2] - self._env.desired_landing_position[:, :2]
                ),
                dim=1,
            )
        )

    def swing_foot_vel_reward(self):
        foot_vel = torch.sum(self._robot.foot_velocities_in_base_frame**2, dim=2)
        contact_mask = torch.logical_not(self._env.gait_generator.desired_contact_state)
        return -torch.sum(foot_vel * contact_mask, dim=1) / (
            torch.sum(contact_mask, dim=1) + 0.001
        )

    def heading_reward(self):
        # print(self._robot.base_orientation_rpy[:, 2])
        # input("Any Key...")
        return -self._robot.base_orientation_rpy[:, 2] ** 2

    def out_of_bound_action_reward(self):
        exceeded_action = torch.maximum(
            self._env._action_lb - self._env._last_action,
            self._env._last_action - self._env._action_ub,
        )
        exceeded_action = torch.clip(exceeded_action, min=0.0)
        normalized_excess = exceeded_action / (
            self._env._action_ub - self._env._action_lb
        )
        return -torch.sum(torch.square(normalized_excess), dim=1)

    def swing_residual_reward(self):
        return -torch.mean(torch.square(self._env._last_action[:, -6:]), axis=1)

    def knee_contact_reward(self):
        rew = (
            -(
                (
                    torch.sum(
                        torch.logical_or(
                            self._env.robot.thigh_contacts,
                            self._env.robot.calf_contacts,
                        ),
                        dim=1,
                    )
                ).float()
            )
            / 4
        )
        return rew

    def body_contact_reward(self):
        return -self._robot.has_body_contact.float()
