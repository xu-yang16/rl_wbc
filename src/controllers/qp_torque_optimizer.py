"""Solves the centroidal QP to compute desired foot torques."""

import time
import numpy as np
import torch
from src.controllers.qp_manager import QPManager
from src.robots.common.motors import MotorCommand
from src.utilities.torch_utils import (
    quat_from_euler_xyz,
    to_torch,
    compute_orientation_error,
)
from src.utilities.rotation_utils import quat_to_rot_mat
from loguru import logger


@torch.jit.script
def compute_desired_acc(
    base_orientation_rpy: torch.Tensor,
    base_position: torch.Tensor,
    base_angular_velocity_body_frame: torch.Tensor,
    base_velocity_body_frame: torch.Tensor,
    desired_base_orientation_rpy: torch.Tensor,
    desired_base_position: torch.Tensor,
    desired_angular_velocity: torch.Tensor,
    desired_linear_velocity: torch.Tensor,
    desired_angular_acceleration: torch.Tensor,
    desired_linear_acceleration: torch.Tensor,
    base_position_kp: torch.Tensor,
    base_position_kd: torch.Tensor,
    base_orientation_kp: torch.Tensor,
    base_orientation_kd: torch.Tensor,
    device: str = "cuda",
):
    base_quat = quat_from_euler_xyz(
        base_orientation_rpy[:, 0],
        base_orientation_rpy[:, 1],
        torch.zeros_like(base_orientation_rpy[:, 0], device=device),
    )
    base_rot_mat = quat_to_rot_mat(base_quat)
    base_rot_mat_t = torch.transpose(base_rot_mat, 1, 2)

    lin_pos_error = desired_base_position - base_position
    lin_pos_error[:, :2] = 0
    lin_vel_error = (
        desired_linear_velocity
        - torch.matmul(base_rot_mat, base_velocity_body_frame[:, :, None])[:, :, 0]
    )
    desired_lin_acc_gravity_frame = (
        base_position_kp * lin_pos_error
        + base_position_kd * lin_vel_error
        + desired_linear_acceleration
    )

    ang_pos_error = compute_orientation_error(
        desired_base_orientation_rpy, base_quat, device=device
    )
    ang_vel_error = (
        desired_angular_velocity
        - torch.matmul(base_rot_mat, base_angular_velocity_body_frame[:, :, None])[
            :, :, 0
        ]
    )
    desired_ang_acc_gravity_frame = (
        base_orientation_kp * ang_pos_error
        + base_orientation_kd * ang_vel_error
        + desired_angular_acceleration
    )

    desired_lin_acc_body_frame = torch.matmul(
        base_rot_mat_t, desired_lin_acc_gravity_frame[:, :, None]
    )[:, :, 0]
    desired_ang_acc_body_frame = torch.matmul(
        base_rot_mat_t, desired_ang_acc_gravity_frame[:, :, None]
    )[:, :, 0]

    return torch.concatenate(
        (desired_lin_acc_body_frame, desired_ang_acc_body_frame), dim=1
    )


@torch.jit.script
def convert_to_skew_symmetric_batch(foot_positions: torch.Tensor):
    """
    Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
    """
    n = foot_positions.shape[0]
    x = foot_positions[:, :, 0]
    y = foot_positions[:, :, 1]
    z = foot_positions[:, :, 2]
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape(
        (n, 3, 3, 4)
    )
    return torch.concatenate(
        [skew[:, :, :, 0], skew[:, :, :, 1], skew[:, :, :, 2], skew[:, :, :, 3]], dim=2
    )


@torch.jit.script
def construct_mass_mat(
    foot_positions,
    inv_mass,
    inv_inertia,
    device: str = "cuda",
):
    num_envs = foot_positions.shape[0]
    mass_mat = torch.zeros((num_envs, 6, 12), device=device)
    # Construct mass matrix
    inv_mass_concat = torch.concatenate([inv_mass] * 4, dim=1)
    mass_mat[:, :3] = inv_mass_concat[None, :, :]
    px = convert_to_skew_symmetric_batch(foot_positions)
    mass_mat[:, 3:6] = torch.matmul(inv_inertia, px)
    return mass_mat


class QPTorqueOptimizer:
    """Centroidal QP controller to optimize for joint torques."""

    def __init__(
        self,
        robot,
        base_position_kp=np.array([0.0, 0.0, 50]),
        base_position_kd=np.array([10.0, 10.0, 10.0]),
        base_orientation_kp=np.array([50.0, 50.0, 0.0]),
        base_orientation_kd=np.array([10.0, 10.0, 10.0]),
        weight_ddq=np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 1.0]),
        weight_grf=1e-4,
        body_mass=13.076,
        body_inertia=np.array([0.14, 0.35, 0.35]) * 1.5,
        desired_body_height=0.26,
        foot_friction_coef=0.7,
        warm_up=True,  # if using pdhg
        iter=20,  # if using pdhg
        solver_type="pdhg",
        friction_type="cone",
    ):
        """Initializes the controller with desired weights and gains."""
        self._robot = robot
        self._device = self._robot._device
        self._num_envs = self._robot.num_envs
        self._warm_up = warm_up
        self._iter = iter

        # pd gains
        self._base_orientation_kp = to_torch(base_orientation_kp, device=self._device)
        self._base_orientation_kp = torch.stack(
            [self._base_orientation_kp] * self._num_envs, dim=0
        )
        self._base_orientation_kd = to_torch(base_orientation_kd, device=self._device)
        self._base_orientation_kd = torch.stack(
            [self._base_orientation_kd] * self._num_envs, dim=0
        )
        self._base_position_kp = to_torch(base_position_kp, device=self._device)
        self._base_position_kp = torch.stack(
            [self._base_position_kp] * self._num_envs, dim=0
        )
        self._base_position_kd = to_torch(base_position_kd, device=self._device)
        self._base_position_kd = torch.stack(
            [self._base_position_kd] * self._num_envs, dim=0
        )

        # rl output extra gains
        self.rl_gains = torch.zeros((self._num_envs, 6), device=self._device)

        # desired base state
        self._desired_base_orientation_rpy = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_base_position = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_base_position[:, 2] = desired_body_height
        self._desired_linear_velocity = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_angular_velocity = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_linear_acceleration = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._desired_angular_acceleration = torch.zeros(
            (self._num_envs, 3), device=self._device
        )

        # qp weights
        self._Wq = to_torch(weight_ddq, device=self._device, dtype=torch.float32)
        self._Wf = to_torch(weight_grf, device=self._device)

        # robot dynamics model
        self._foot_friction_coef = foot_friction_coef
        self._inv_mass = torch.eye(3, device=self._device) / body_mass
        self._inv_inertia = torch.linalg.inv(
            to_torch(body_inertia, device=self._device)
        )

        self.qp_manager = QPManager(
            num_envs=self._num_envs,
            weight_ddq=weight_ddq,
            Wr=weight_grf,
            Wt=weight_grf,
            foot_friction_coef=foot_friction_coef,
            max_z_force=120.0,
            min_z_force=0.0,
            max_joint_torque=15.0,
            device=self._device,
            iter=iter,
            warm_start=warm_up,
            friction_type=friction_type,
            solver_type=solver_type,
        )

    def solve_joint_torques(
        self,
        foot_contact_state: torch.tensor,
        desired_com_ddq: torch.tensor,
    ):
        """Solves centroidal QP to find desired joint torques."""
        self._mass_mat = construct_mass_mat(
            self._robot.foot_positions_in_base_frame,
            self._inv_mass,
            self._inv_inertia,
            device=self._device,
        )

        # Solve QP
        grf, solved_acc, solver_time = self.qp_manager.solve_grf(
            mass_mat=self._mass_mat,
            desired_acc=desired_com_ddq,
            base_rot_mat_t=self._robot.base_rot_mat_t,
            all_foot_jacobian=self._robot.all_foot_jacobian,
            dq=self._robot.motor_velocities,
            foot_contact_state=foot_contact_state,
        )

        motor_torques = -torch.bmm(grf[:, None, :], self._robot.all_foot_jacobian)[:, 0]
        return motor_torques, solved_acc, grf, solver_time

    def get_action(
        self, foot_contact_state: torch.Tensor, swing_foot_position: torch.Tensor
    ):
        """Computes motor actions."""
        base_position_kp = self._base_position_kp.clone()
        base_position_kd = self._base_position_kd.clone()
        base_orientation_kp = self._base_orientation_kp.clone()
        base_orientation_kd = self._base_orientation_kd.clone()

        # rl comes in
        base_position_kp[:, 2] += self.rl_gains[:, 2]
        base_position_kd[:, :2] += self.rl_gains[:, :2]
        base_orientation_kp[:, :2] += self.rl_gains[:, 3:5]
        base_orientation_kd[:, 2] += self.rl_gains[:, 5]

        # Compute desired acceleration
        desired_acc_body_frame = compute_desired_acc(
            base_orientation_rpy=self._robot.base_orientation_rpy,
            base_position=self._robot.base_position_world,
            base_angular_velocity_body_frame=self._robot.base_angular_velocity_body_frame,
            base_velocity_body_frame=self._robot.base_velocity_body_frame,
            desired_base_orientation_rpy=self._desired_base_orientation_rpy,
            desired_base_position=self._desired_base_position,
            desired_angular_velocity=self._desired_angular_velocity,
            desired_linear_velocity=self._desired_linear_velocity,
            desired_angular_acceleration=self._desired_angular_acceleration,
            desired_linear_acceleration=self._desired_linear_acceleration,
            base_position_kp=base_position_kp,
            base_position_kd=base_position_kd,
            base_orientation_kp=base_orientation_kp,
            base_orientation_kd=base_orientation_kd,
            device=self._device,
        )
        desired_acc_body_frame = torch.clip(
            desired_acc_body_frame,
            to_torch([-30, -30, -10, -20, -20, -20], device=self._device),
            to_torch([30, 30, 30, 20, 20, 20], device=self._device),
        )

        # Solve for joint torques of the stance legs
        motor_torques, solved_acc, grf, solver_time = self.solve_joint_torques(
            foot_contact_state, desired_acc_body_frame
        )

        # Solver for joint pos of the swing legs
        foot_position_local = torch.bmm(
            self._robot.base_rot_mat_t, swing_foot_position.transpose(1, 2)
        ).transpose(1, 2)
        foot_position_local[:, :, 2] = torch.clip(
            foot_position_local[:, :, 2], min=-0.35, max=-0.1
        )

        desired_motor_position = self._robot.get_motor_angles_from_foot_positions(
            foot_position_local
        )

        # Combine the motor commands for the stance and swing legs
        contact_state_expanded = foot_contact_state.repeat_interleave(3, dim=1)
        desired_position = torch.where(
            contact_state_expanded, self._robot.motor_positions, desired_motor_position
        )
        desired_velocity = torch.where(
            contact_state_expanded,
            self._robot.motor_velocities,
            torch.zeros_like(motor_torques),
        )
        desired_torque = torch.where(
            contact_state_expanded, motor_torques, torch.zeros_like(motor_torques)
        )
        desired_torque = torch.clip(
            desired_torque,
            max=self._robot.motor_group.max_torques,
            min=self._robot.motor_group.min_torques,
        )

        return (
            MotorCommand(
                desired_position=desired_position,
                kp=torch.ones_like(self._robot.motor_group.kps) * 30,
                desired_velocity=desired_velocity,
                kd=torch.ones_like(self._robot.motor_group.kds) * 1,
                desired_extra_torque=desired_torque,
            ),
            desired_acc_body_frame,
            solved_acc,
            grf,
            solver_time,
        )

    @property
    def desired_base_position(self) -> torch.Tensor:
        return self._desired_base_position

    @desired_base_position.setter
    def desired_base_position(self, base_position: float):
        self._desired_base_position = to_torch(base_position, device=self._device)

    @property
    def desired_base_orientation_rpy(self) -> torch.Tensor:
        return self._desired_base_orientation_rpy

    @desired_base_orientation_rpy.setter
    def desired_base_orientation_rpy(self, orientation_rpy: torch.Tensor):
        self._desired_base_orientation_rpy = to_torch(
            orientation_rpy, device=self._device
        )

    @property
    def desired_linear_velocity(self) -> torch.Tensor:
        return self._desired_linear_velocity

    @desired_linear_velocity.setter
    def desired_linear_velocity(self, desired_linear_velocity: torch.Tensor):
        self._desired_linear_velocity = to_torch(
            desired_linear_velocity, device=self._device
        )

    @property
    def desired_angular_velocity(self) -> torch.Tensor:
        return self._desired_angular_velocity

    @desired_angular_velocity.setter
    def desired_angular_velocity(self, desired_angular_velocity: torch.Tensor):
        self._desired_angular_velocity = to_torch(
            desired_angular_velocity, device=self._device
        )

    @property
    def desired_linear_acceleration(self):
        return self._desired_linear_acceleration

    @desired_linear_acceleration.setter
    def desired_linear_acceleration(self, desired_linear_acceleration: torch.Tensor):
        self._desired_linear_acceleration = to_torch(
            desired_linear_acceleration, device=self._device
        )

    @property
    def desired_angular_acceleration(self):
        return self._desired_angular_acceleration

    @desired_angular_acceleration.setter
    def desired_angular_acceleration(self, desired_angular_acceleration: torch.Tensor):
        self._desired_angular_acceleration = to_torch(
            desired_angular_acceleration, device=self._device
        )
