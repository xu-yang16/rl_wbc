import numpy as np
import torch
from src.utilities.torch_utils import to_torch

from .common.motors import MotorControlMode, MotorGroup, MotorModel
from .common.robot_params import RobotParamManager


class Robot:
    """
    Properties:
    - control_timestep
    - hip_offset
    - hip_positions_in_body_frame
    - num_dof
    - motor_group
    - num_envs
    - device
    - motor_control_mode
    - init_motor_angles
    - base_orientation_quat
    - base_rot_mat
    - base_rot_mat_t
    - foot_positions_in_base_frame
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        motor_control_mode: MotorControlMode = MotorControlMode.HYBRID,
        motor_torque_delay_steps: int = 0,
        action_repeat: int = 1,
        robot_name: str = "go1",
    ):
        self.robot_param_manager = RobotParamManager(robot_name)
        self._num_envs = num_envs
        self._device = device
        self._motor_control_mode = motor_control_mode

        self._base_quat_torch = torch.zeros((num_envs, 4), device=device)
        self._base_rot_mat_torch = (
            torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        )
        self._base_rot_mat_t_torch = (
            torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        )
        self._foot_positions_in_base_frame = np.zeros((4, 3))

        self.init_motors(motor_control_mode, motor_torque_delay_steps)
        self._init_motor_angles = to_torch(
            self.robot_param_manager.robot.initial_pos.reshape(
                12,
            ),
            device=device,
        )
        self._action_repeat = action_repeat
        self._sim_dt = 0.002
        self._num_dof = 12

        # for go1
        self._hip_offset = torch.tensor(
            self.robot_param_manager.robot.hip_offset, device=device
        ).float()
        self._hip_positions_in_body_frame = (
            torch.tensor(self.robot_param_manager.robot.normal_stand, device=device)
            .repeat(self.num_envs, 1, 1)
            .float()
        )
        self.l_up, self.l_low, self.foot_size = (
            self.robot_param_manager.robot.l_up,
            self.robot_param_manager.robot.l_low,
            self.robot_param_manager.robot.foot_size,
        )
        self.l_hip = self.robot_param_manager.robot.l_hip
        self.l_hip_torch = torch.tensor(
            self.robot_param_manager.robot.l_hip, device=device
        ).float()

    def init_motors(
        self,
        motor_control_mode: MotorControlMode = MotorControlMode.HYBRID,
        motor_torque_delay_steps: int = 0,
    ):
        self._motors = MotorGroup(
            device=self.device,
            num_envs=self.num_envs,
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

    # ----------------------- common -----------------------
    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device

    @property
    def num_dof(self):
        return self._num_dof

    @property
    def control_timestep(self):
        return self._sim_dt * self._action_repeat

    # ----------------------- kinematics -----------------------
    @property
    def mpc_body_height(self):
        return self.robot_param_manager.robot.desired_body_height

    @property
    def hip_offset(self):
        """Position of hip offset in base frame, used for IK only."""
        return self._hip_offset

    @property
    def hip_positions_in_body_frame(self):
        return self._hip_positions_in_body_frame

    # ----------------------- motors -----------------------
    @property
    def motor_group(self):
        return self._motors

    @property
    def motor_control_mode(self):
        return self._motor_control_mode

    @property
    def init_motor_angles(self):
        return self._init_motor_angles

    # ----------------------- base state -----------------------
    @property
    def base_orientation_quat(self):
        return self._base_quat_torch

    @property
    def base_rot_mat(self):
        return self._base_rot_mat_torch

    @property
    def base_rot_mat_t(self):
        return self._base_rot_mat_t_torch

    @property
    def foot_positions_in_base_frame(self):
        return to_torch(self._foot_positions_in_base_frame, device=self.device)

    # ----------------------- need numpy (motor position) -----------------------
    def _foot_positions_in_hip_frame(self, compute_tip=False):
        motor_positions = self.motor_positions_numpy.reshape((4, 3))
        theta_ab = motor_positions[:, 0]
        theta_hip = motor_positions[:, 1]
        theta_knee = motor_positions[:, 2]
        l_up = self.l_up
        l_low = self.l_low + self.foot_size if compute_tip else self.l_low
        l_hip = self.l_hip
        leg_distance = np.sqrt(
            l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee)
        )
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * np.sin(eff_swing)
        off_z_hip = -leg_distance * np.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
        off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip

        return np.stack([off_x, off_y, off_z], axis=1)

    def get_motor_angles_from_foot_positions(self, foot_local_positions: torch.Tensor):
        foot_positions_in_hip_frame = foot_local_positions - self.hip_offset
        l_up = self.l_up
        l_low = self.l_low + self.foot_size
        l_hip = self.l_hip_torch

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

    def _compute_all_foot_jacobian(self, compute_tip=False):
        motor_positions = self.motor_positions_numpy.reshape((4, 3))
        l_up = 0.213
        l_low = 0.233 if compute_tip else 0.213
        l_hip = np.array([-1, 1, -1, 1]) * 0.08

        t1, t2, t3 = motor_positions[:, 0], motor_positions[:, 1], motor_positions[:, 2]
        l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
        t_eff = t2 + t3 / 2
        J = np.zeros((4, 3, 3))
        J[:, 0, 0] = 0
        J[:, 0, 1] = -l_eff * np.cos(t_eff)
        J[:, 0, 2] = (
            l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff
            - l_eff * np.cos(t_eff) / 2
        )
        J[:, 1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
        J[:, 1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
        J[:, 1, 2] = (
            -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(t_eff) / l_eff
            - l_eff * np.sin(t1) * np.sin(t_eff) / 2
        )
        J[:, 2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
        J[:, 2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
        J[:, 2, 2] = (
            l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(t_eff) / l_eff
            + l_eff * np.sin(t_eff) * np.cos(t1) / 2
        )

        flattened_jacobian = np.zeros((12, 12))
        flattened_jacobian[:3, :3] = J[0]
        flattened_jacobian[3:6, 3:6] = J[1]
        flattened_jacobian[6:9, 6:9] = J[2]
        flattened_jacobian[9:12, 9:12] = J[3]
        return flattened_jacobian
