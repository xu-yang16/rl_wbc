"""Vectorized Go1 robot in Isaac Gym."""

from typing import Any, Sequence, Union

from src.utilities.torch_utils import to_torch
import ml_collections
import torch

from src.robots.isaac_gym_robot import IsaacGymRobot
from src.robots.common.motors import MotorControlMode, MotorGroup, MotorModel
from src.robots.common.robot_params import RobotParamManager


class Go1Isaac(IsaacGymRobot):
    """Go1 robot in simulation."""

    def __init__(
        self,
        sim: Any,
        viewer: Any,
        sim_config: ml_collections.ConfigDict(),
        num_envs: int,
        motor_control_mode: MotorControlMode,
        motor_torque_delay_steps: int = 0,
        num_actions: int = 6,
        num_actor_obs: int = 10,
        domain_rand: bool = False,
        terrain_type: str = "flat",
        robot_name: str = "go1",
    ):
        motors = MotorGroup(
            device=sim_config.sim_device,
            num_envs=num_envs,
            motors=(
                MotorModel(
                    name="FR_hip_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.0,
                    min_position=-0.802851455917,
                    max_position=0.802851455917,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="FR_thigh_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.9,
                    min_position=-1.0471975512,
                    max_position=4.18879020479,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="FR_calf_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=-1.8,
                    min_position=-2.6965336943,
                    max_position=-0.916297857297,
                    min_velocity=-20,
                    max_velocity=20,
                    min_torque=-35.55,
                    max_torque=35.55,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="FL_hip_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.0,
                    min_position=-0.802851455917,
                    max_position=0.802851455917,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="FL_thigh_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.9,
                    min_position=-1.0471975512,
                    max_position=4.18879020479,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="FL_calf_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=-1.8,
                    min_position=-1.0471975512,
                    max_position=4.18879020479,
                    min_velocity=-20,
                    max_velocity=20,
                    min_torque=-35.55,
                    max_torque=35.55,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RR_hip_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.0,
                    min_position=-0.802851455917,
                    max_position=0.802851455917,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RR_thigh_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.9,
                    min_position=-1.0471975512,
                    max_position=4.18879020479,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RR_calf_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=-1.8,
                    min_position=-2.6965336943,
                    max_position=-0.916297857297,
                    min_velocity=-20,
                    max_velocity=20,
                    min_torque=-35.55,
                    max_torque=35.55,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RL_hip_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.0,
                    min_position=-0.802851455917,
                    max_position=0.802851455917,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RL_thigh_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=0.9,
                    min_position=-1.0471975512,
                    max_position=4.18879020479,
                    min_velocity=-30,
                    max_velocity=30,
                    min_torque=-23.7,
                    max_torque=23.7,
                    kp=100,
                    kd=1,
                ),
                MotorModel(
                    name="RL_calf_joint",
                    motor_control_mode=motor_control_mode,
                    init_position=-1.8,
                    min_position=-2.6965336943,
                    max_position=-0.916297857297,
                    min_velocity=-20,
                    max_velocity=20,
                    min_torque=-35.55,
                    max_torque=35.55,
                    kp=100,
                    kd=1,
                ),
            ),
            torque_delay_steps=motor_torque_delay_steps,
        )
        self.robot_name = robot_name
        self.robot_param_manager = RobotParamManager(robot_name)
        self._init_motor_angles = to_torch(
            self.robot_param_manager.robot.initial_pos.reshape(
                12,
            ),
            device=sim_config.sim_device,
        )
        self._hip_offset = to_torch(
            self.robot_param_manager.robot.hip_offset, device=sim_config.sim_device
        )
        self._hip_positions_in_body_frame = to_torch(
            self.robot_param_manager.robot.normal_stand, device=sim_config.sim_device
        ).repeat(num_envs, 1, 1)
        self.l_up = self.robot_param_manager.robot.l_up
        self.l_low = self.robot_param_manager.robot.l_low
        self.foot_size = self.robot_param_manager.robot.foot_size
        self.l_hip = to_torch(
            self.robot_param_manager.robot.l_hip, device=sim_config.sim_device
        )

        super().__init__(
            sim=sim,
            viewer=viewer,
            num_envs=num_envs,
            urdf_path=self.robot_param_manager.robot.urdf_path,
            sim_config=sim_config,
            motors=motors,
            feet_names=[
                "1_FR_foot",
                "2_FL_foot",
                "3_RR_foot",
                "4_RL_foot",
            ],
            calf_names=[
                "1_FR_calf",
                "2_FL_calf",
                "3_RR_calf",
                "4_RL_calf",
            ],
            thigh_names=[
                "1_FR_thigh",
                "2_FL_thigh",
                "3_RR_thigh",
                "4_RL_thigh",
            ],
            num_actions=num_actions,
            num_actor_obs=num_actor_obs,
            domain_rand=domain_rand,
            terrain_type=terrain_type,
        )
        if self.robot_name == "go2":
            self.env_origins[:, 2] = 0.30

    @property
    def hip_positions_in_body_frame(self):
        return self._hip_positions_in_body_frame

    @property
    def hip_offset(self):
        """Position of hip offset in base frame, used for IK only."""
        return self._hip_offset

    def get_motor_angles_from_foot_positions(self, foot_local_positions: torch.Tensor):
        foot_positions_in_hip_frame = foot_local_positions - self.hip_offset
        l_up = self.l_up
        l_low = self.l_low + self.foot_size
        l_hip = self.l_hip

        x = foot_positions_in_hip_frame[:, :, 0]
        y = foot_positions_in_hip_frame[:, :, 1]
        z = foot_positions_in_hip_frame[:, :, 2]
        theta_knee = -torch.arccos(
            torch.clip(
                (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2)
                / (2 * l_low * l_up),
                -1,
                1,
            )
        )
        l = torch.sqrt(
            torch.clip(
                l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee), 1e-7, 1
            )
        )
        theta_hip = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_knee / 2
        c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
        s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = torch.arctan2(s1, c1)

        # thetas: num_envs x 4
        joint_angles = torch.stack([theta_ab, theta_hip, theta_knee], dim=2)
        return joint_angles.reshape((-1, 12))

    @property
    def time_since_reset_scalar(self):
        return self._time_since_reset[0].item()
