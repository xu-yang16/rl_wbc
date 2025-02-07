"""Mujoco simulation for Go1 robot."""

import os, time
import os.path as osp
from typing import Any, Optional, Union

import mujoco, mujoco_viewer
from scipy.spatial.transform import Rotation as R

from src.utilities.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.robots import robot
from src.robots.estimator import robot_state_estimator
from src.robots.common.motors import MotorCommand, MotorControlMode
from src.robots.common.robot_params import RobotParamManager
from src.utilities.rotation_utils import getMatrixFromQuaternion

from icecream import ic
from loguru import logger


class IMUState:
    def __init__(self):
        self.quaternion = np.array([0, 0, 0, 1.0])
        self.rpy = np.zeros(3)
        self.accelerometer = np.zeros(3)
        self.gyroscope = np.zeros(3)


class Motor:
    def __init__(self):
        self.q = 0.0
        self.dq = 0.0
        self.tauEst = 0.0


class RawState:
    def __init__(self):
        self.imu = IMUState()
        self.motorState = [Motor() for _ in range(12)]
        self.footForce = np.zeros(4)


class Go1Mujoco(robot.Robot):
    """Go1 robot class."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        motor_control_mode: MotorControlMode,
        motor_torque_delay_steps: int = 0,
        gui=True,
        robot_name: str = "go1",
    ):
        del motor_torque_delay_steps  # unused

        self._num_envs = num_envs
        self._device = device

        self.apply_disturb = False
        self.disturb_force = np.array([0, 0, -15])

        self.contact_forces_3d = np.zeros(12)
        self.gui = gui
        super().__init__(
            num_envs=num_envs,
            device=device,
            motor_control_mode=motor_control_mode,
            motor_torque_delay_steps=0,
            action_repeat=1,
            robot_name=robot_name,
        )
        # init mujoco environment
        self.load_model(
            xml_path=osp.join(
                osp.dirname(__file__),
                "../../",
                self.robot_param_manager.robot.urdf_path.replace("urdf", "xml"),
            )
        )

        self._raw_state = RawState()
        self._contact_force_threshold = np.zeros(4)
        self._state_estimator = robot_state_estimator.RobotStateEstimator(
            self, use_external_contact_estimator=False
        )
        self._last_reset_time = self.data.time
        self._motor_control_mode = motor_control_mode
        self._foot_contact_force_histories = [[], [], [], []]

        self._jacobians = np.zeros((12, 12))
        self._jacobians_tip = np.zeros((12, 12))

        self._base_rot_mat = np.eye(3)
        self.v = np.zeros(3)
        self.acc = np.zeros(6)
        self.ff_acc = np.zeros(6)
        self.plot_choice = 0

    def update_desired_foot_contact(self, desired_contact):
        foot_forces = self.foot_contact_forces_numpy
        desired_contact = desired_contact.cpu().numpy().flatten()
        for idx in range(4):
            if desired_contact[idx]:
                if len(self._foot_contact_force_histories[idx]) > 0:
                    self._contact_force_threshold[idx] = np.mean(
                        self._foot_contact_force_histories[idx]
                    ) + 0.5 * np.std(self._foot_contact_force_histories[idx])
                    self._foot_contact_force_histories[idx] = []
            else:
                self._foot_contact_force_histories[idx].append(foot_forces[idx])

    def reset(self, reset_time: float = 1.5):
        print("About to reset the robot...")
        self.data.qpos[-12:] = self._motors.init_positions.cpu().numpy()
        self.data.qpos[0:3] = np.array([0, 0, 0.27])
        self.data.qpos[3:7] = np.array([1.0, 0, 0, 0])
        self.data.qvel[0:6] = np.zeros(6)
        stand_foot_forces = []
        for _ in range(80):
            action = MotorCommand(
                desired_position=to_torch(
                    self._motors.init_positions[None, :], device=self._device
                ),
                kp=self._motors.kps,
                desired_velocity=torch.zeros((self._num_envs, 12), device=self._device),
                kd=self._motors.kds,
            )
            self.step(action, motor_control_mode=MotorControlMode.POSITION)
            stand_foot_forces.append(self.foot_contact_forces_numpy)
        # Calibrate foot force sensors
        stand_foot_forces = np.mean(stand_foot_forces, axis=0)
        self._contact_force_threshold = stand_foot_forces * 0.8
        ic(self._contact_force_threshold)

        self._last_reset_time = self.data.time
        self._state_estimator.reset()
        self._post_physics_step()
        print("init finished...")

    def reset_idx(self, env_ids):
        del env_ids  # unused
        self.reset()

    def _apply_action(
        self,
        action: MotorCommand,
        motor_control_mode: Optional[MotorControlMode] = None,
    ) -> None:
        if motor_control_mode is None:
            motor_control_mode = self._motor_control_mode
        command = np.zeros(60, dtype=np.float32)
        if motor_control_mode == MotorControlMode.POSITION:
            for motor_id in range(self._num_dof):
                command[motor_id * 5] = action.desired_position.cpu().numpy()[
                    0, motor_id
                ]
                command[motor_id * 5 + 1] = action.kp.cpu().numpy()[motor_id]
                command[motor_id * 5 + 3] = action.kd.cpu().numpy()[motor_id]
        elif motor_control_mode == MotorControlMode.HYBRID:
            command[0::5] = action.desired_position[0].cpu().numpy()
            command[1::5] = action.kp[0].cpu().numpy()
            command[2::5] = action.desired_velocity[0].cpu().numpy()
            command[3::5] = action.kd[0].cpu().numpy()
            command[4::5] = action.desired_extra_torque[0].cpu().numpy()
        else:
            raise ValueError(
                f"Unknown motor control mode for Go1 robot: {motor_control_mode}."
            )

        # if self.viewer.is_alive:
        self.send_command(command)
        self.receive_observation()
        # else:
        # exit(-1)
        if self.viewer is not None and not self.viewer.is_alive:
            exit(-1)

    def step(
        self,
        action: MotorCommand,
        desired_acc: torch.Tensor = None,
        solved_acc: torch.Tensor = None,
        ff_acc: torch.Tensor = None,
        motor_control_mode: Optional[MotorControlMode] = None,
    ):
        for _ in range(self._action_repeat):
            self._apply_action(action, motor_control_mode)
            self._post_physics_step()
            self._state_estimator.update(self._raw_state)

            if motor_control_mode == MotorControlMode.POSITION:
                continue
            elif (
                desired_acc is not None
                and solved_acc is not None
                and ff_acc is not None
            ):
                self.plot(
                    desired_acc.squeeze(0).detach().cpu().numpy(),
                    solved_acc.squeeze(0).detach().cpu().numpy(),
                    self.acc,
                    ff_acc.squeeze(0).detach().cpu().numpy(),
                )
            if self.viewer is not None:
                self.viewer.render()

    def _post_physics_step(self):
        q = self._raw_state.imu.quaternion  # w, x, y, z
        base_quat = np.array([q[1], q[2], q[3], q[0]])  # to x, y, z, w
        self._base_rot_mat = getMatrixFromQuaternion(q[1], q[2], q[3], q[0])
        self._base_rot_mat_torch = to_torch(
            self._base_rot_mat[None, :, :], device=self._device
        )
        self._base_rot_mat_t_torch = to_torch(
            (self._base_rot_mat.T)[None, :, :], device=self._device
        )
        self._base_quat_torch = to_torch([base_quat], device=self._device)

        self._jacobians = self._compute_all_foot_jacobian(compute_tip=False)
        self._jacobians_tip = self._compute_all_foot_jacobian(compute_tip=True)

        self._foot_positions_in_base_frame = (
            self._foot_positions_in_hip_frame(compute_tip=True)
            + self.hip_offset.cpu().numpy()
        )
        self._foot_center_positions_in_base_frame = (
            self._foot_positions_in_hip_frame(compute_tip=False)
            + self.hip_offset.cpu().numpy()
        )

    def compute_foot_jacobian(self, leg_id):
        return self._jacobians[leg_id * 3 : leg_id * 3 + 3, leg_id * 3 : leg_id * 3 + 3]

    def compute_foot_jacobian_tip(self, leg_id):
        return self._jacobians_tip[
            leg_id * 3 : leg_id * 3 + 3, leg_id * 3 : leg_id * 3 + 3
        ]

    def apply_disturbance_force(self, force, body_index=1):
        if self.apply_disturb:
            self.data.xfrc_applied[body_index, 0:3] = force

    # --- mujoco interfaces ---#
    def load_model(self, xml_path=""):
        os.environ["HOME"] = osp.join(osp.dirname(__file__), "../../")

        # xml_path = osp.join(osp.dirname(__file__), "../../data/go1/meshes/mjmodel.xml")
        ic(xml_path)
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = 0.002
        data = mujoco.MjData(model)

        mujoco.mj_step(model, data)
        if not self.gui:
            return model, data, None
        viewer = mujoco_viewer.MujocoViewer(model, data)
        self._counter = 0
        self._last_shadows = False
        self.offset = 0

        import pathlib

        ic(viewer.CONFIG_PATH)

        # options
        import glfw

        ic(viewer._contacts)
        viewer._key_callback(
            window=None, key=glfw.KEY_C, scancode=None, action=glfw.RELEASE, mods=None
        )
        # viewer._key_callback(
        #     window=None, key=glfw.KEY_TAB, scancode=None, action=glfw.RELEASE, mods=None
        # )
        ic(viewer._contacts)

        fig_names = ["acc0", "acc1", "lin_acc2"]
        line_names = ["desired", "solved", "actual"]
        new_line_names = [
            "ff_lin_acc_x",
            "ff_lin_acc_y",
            "ff_lin_acc_z",
            "ff_ang_acc_x",
            "ff_ang_acc_y",
            "ff_ang_acc_z",
        ]
        for i in range(3):
            for j in range(3):
                viewer.add_line_to_fig(line_name=line_names[j], fig_idx=i)
            for k in range(6):
                viewer.add_line_to_fig(line_name=new_line_names[k], fig_idx=i)
            fig = viewer.figs[i]
            fig.title = fig_names[i]
            fig.flg_legend = True
            fig.xlabel = "Timesteps"
            fig.figurergba[0:4] = np.array([0.2, 0, 0.2, 0.01])
            fig.gridsize[0:2] = np.array([5, 5])

        self.model, self.data, self.viewer = model, data, viewer

    def send_command(self, command):
        p_des, kp, v_des, kv, ff = (
            command[0::5],
            command[1::5],
            command[2::5],
            command[3::5],
            command[4::5],
        )
        p = self.data.qpos[-12:]
        v = self.data.qvel[-12:]
        tau = kp * (p_des - p) + kv * (v_des - v) + ff
        max_tau = np.array([23.7, 23.7, 35.55] * 4)
        tau = np.clip(tau, -max_tau, max_tau)
        # tau = np.clip(tau, -15.0, 15.0)
        # logger.debug(f"tau={tau}")
        self.data.ctrl = tau

        # apply disturbance
        self.apply_disturbance_force(force=self.disturb_force, body_index=1)

        mujoco.mj_step(self.model, self.data)

    def plot(self, desired_acc, solved_acc, actual_acc, ff_acc):
        if self.viewer is None:
            return
        # plots for acc
        if self.viewer._shadows != self._last_shadows:
            self.plot_choice += 1
            self.plot_choice %= 3
            self._last_shadows = self.viewer._shadows
            print(f"plot_choice={self.plot_choice}")
            if self.plot_choice == 0:
                self.offset = 0
                for i in range(3):
                    self.viewer.figs[i].title = f"lin_acc{i}"
                    self.viewer.figs[i].linepnt = 0
            elif self.plot_choice == 1:
                self.offset = 3
                for i in range(3):
                    self.viewer.figs[i].title = f"ang_acc{i}"
                    self.viewer.figs[i].linepnt = 0
            elif self.plot_choice == 2:
                self.offset = 6
                for i in range(3):
                    self.viewer.figs[i].title = f"ff_acc{i}"
                    self.viewer.figs[i].linepnt = 0
        if self.plot_choice == 0 or self.plot_choice == 1:
            for i in range(3):
                self.viewer.add_data_to_line(
                    line_name="desired",
                    line_data=desired_acc[i + self.offset],
                    fig_idx=i,
                )
                self.viewer.add_data_to_line(
                    line_name="solved", line_data=solved_acc[i + self.offset], fig_idx=i
                )
                self.viewer.add_data_to_line(
                    line_name="actual",
                    line_data=actual_acc[i + self.offset],
                    fig_idx=i,
                )
        elif self.plot_choice == 2:
            for i in range(3):
                self.viewer.add_data_to_line(
                    line_name=f"ff_lin_acc_x",
                    line_data=ff_acc[0],
                    fig_idx=0,
                )
                self.viewer.add_data_to_line(
                    line_name=f"ff_lin_acc_y",
                    line_data=ff_acc[1],
                    fig_idx=0,
                )
                self.viewer.add_data_to_line(
                    line_name=f"ff_lin_acc_z",
                    line_data=ff_acc[2],
                    fig_idx=0,
                )
                self.viewer.add_data_to_line(
                    line_name=f"ff_ang_acc_x",
                    line_data=ff_acc[3],
                    fig_idx=1,
                )
                self.viewer.add_data_to_line(
                    line_name=f"ff_ang_acc_y",
                    line_data=ff_acc[4],
                    fig_idx=1,
                )
                self.viewer.add_data_to_line(
                    line_name=f"ff_ang_acc_z",
                    line_data=ff_acc[5],
                    fig_idx=1,
                )

    def receive_observation(self):
        # read state from mujoco
        q = self.data.qpos.astype(np.double)[-12:]
        dq = self.data.qvel.astype(np.double)[-12:]
        quat = self.data.sensor("imu_quat").data.astype(np.double)  # w, x, y, z
        rotation = R.from_quat(
            np.array([quat[1], quat[2], quat[3], quat[0]])
        )  # R.from_quat requires [x, y, z, w]
        rpy = rotation.as_euler(
            "xyz", degrees=False
        )  # Set degrees=True if you want the angles in degrees
        v = rotation.apply(self.data.qvel[:3], inverse=True).astype(
            np.double
        )  # In the base frame
        v_truth = self.data.sensor("frame_vel").data.astype(np.double)
        self.v = v_truth
        acc = self.data.sensor("imu_acc").data.astype(np.double)
        omega = self.data.sensor("imu_gyro").data.astype(np.double)

        # self.acc[:3] = acc - np.array([0, 0, 9.81])
        # self.acc[2] *= -1
        # self.acc[3:] = rotation.apply(
        #     self.data.sensor("imu_ang_acc").data.astype(np.double), inverse=True
        # ).astype(
        #     np.double
        # )  # In the base frame

        raw_state = RawState()
        raw_state.imu.quaternion = quat  # w, x, y, z
        raw_state.imu.rpy = rpy
        raw_state.imu.accelerometer = acc
        raw_state.imu.gyroscope = omega
        for i in range(12):
            raw_state.motorState[i].q = q[i]
            raw_state.motorState[i].dq = dq[i]
            raw_state.motorState[i].tauEst = self.data.ctrl[i]

        f = np.zeros(12)
        footname2id = {"FR_foot": 0, "FL_foot": 3, "RR_foot": 6, "RL_foot": 9}
        for i in range(self.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.data.contact[i]
            geom_name = self.model.geom(contact.geom2).name
            if "foot" not in geom_name:
                continue
            result = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, result)
            grf_glo = np.zeros(3)
            mujoco.mju_mulMatTVec(grf_glo, contact.frame.reshape((3, 3)), result[:3])
            # print("contact", i, " result=", result, " grf_glo=", grf_glo)
            # print("force=", self.data.efc_force[contact.efc_address])
            for i in range(3):
                f[footname2id[geom_name] + i] = grf_glo[i]

        raw_state.footForce = np.array(
            [
                np.linalg.norm(
                    f[2:3]
                ),  # self.data.sensor("FR_foot_force_sensor").data.astype(np.double)[0],
                np.linalg.norm(
                    f[5:6]
                ),  # self.data.sensor("FL_foot_force_sensor").data.astype(np.double)[0],
                np.linalg.norm(
                    f[8:9]
                ),  # self.data.sensor("RR_foot_force_sensor").data.astype(np.double)[0],
                np.linalg.norm(
                    f[11:12]
                ),  # self.data.sensor("RL_foot_force_sensor").data.astype(np.double)[0],
            ]
        )
        self.contact_forces_3d[:] = f[:]
        # logger.info(f"foot force: {f}")
        # ic(raw_state.footForce)
        self._raw_state = raw_state

        return raw_state

    @property
    def foot_positions_in_base_frame(self):
        return to_torch(
            self._foot_positions_in_base_frame[None, :, :], device=self._device
        )

    @property
    def all_foot_jacobian(self):
        return to_torch(self._jacobians[None, :, :], device=self._device)

    @property
    def base_position_world(self):
        return to_torch([self._state_estimator.estimated_position], device=self._device)

    @property
    def base_orientation_rpy(self):
        return to_torch([self._raw_state.imu.rpy], device=self._device)

    @property
    def projected_gravity(self):
        return self.base_rot_mat[:, :, 2]

    @property
    def base_rot_mat_numpy(self):
        return self._base_rot_mat.copy()

    @property
    def base_velocity_world_frame(self):
        return to_torch(
            self._state_estimator.estimated_velocity[None, :], device=self._device
        )

    @property
    def base_velocity_body_frame(self):
        return to_torch(
            self._base_rot_mat.T.dot(self.data.qvel[:3])[None, :],
            device=self._device,
        )
        return to_torch(
            self._base_rot_mat.T.dot(self._state_estimator.estimated_velocity)[None, :],
            device=self._device,
        )

    @property
    def base_angular_velocity_world_frame(self):
        return to_torch(
            self._base_rot_mat.T.dot(self._state_estimator.angular_velocity)[None, :],
            device=self._device,
        )

    @property
    def base_angular_velocity_body_frame(self):
        """Smoothed using moving-window filter"""
        return to_torch(
            self._base_rot_mat.T.dot(self.data.qvel[3:6])[None, :],
            device=self._device,
        )
        return to_torch(
            self._state_estimator.angular_velocity[None, :], device=self._device
        )

    @property
    def motor_positions(self):
        return to_torch(
            [[motor.q for motor in self._raw_state.motorState[:12]]],
            device=self._device,
        )

    @property
    def motor_positions_numpy(self):
        return np.array([motor.q for motor in self._raw_state.motorState[:12]])

    @property
    def motor_velocities(self):
        return to_torch(
            [[motor.dq for motor in self._raw_state.motorState[:12]]],
            device=self._device,
        )

    @property
    def motor_velocities_numpy(self):
        return np.array([motor.dq for motor in self._raw_state.motorState[:12]])

    @property
    def motor_torques(self):
        return to_torch(
            [[motor.tauEst for motor in self._raw_state.motorState[:12]]],
            device=self._device,
        )

    @property
    def foot_center_positions_in_base_frame_numpy(self):
        return self._foot_center_positions_in_base_frame.copy()

    @property
    def foot_height(self):
        return torch.where(self.foot_contact, 0.02, 0.05)

    @property
    def foot_velocities_in_world_frame(self):
        # TODO: logging.warning("World-frame foot velocity is not yet implemented.")
        raise NotImplementedError
        return torch.zeros((self._num_envs, 4, 3))

    @property
    def foot_velocities_in_base_frame(self):
        foot_vels = torch.bmm(
            self.all_foot_jacobian, self.motor_velocities[:, :, None]
        ).squeeze()
        return foot_vels.reshape((self._num_envs, 4, 3))

    @property
    def foot_contact(self):
        return torch.tensor(self.foot_contact_numpy[None, :], device=self._device)

    @property
    def foot_contact_numpy(self):
        return self.foot_contact_forces_numpy > self._contact_force_threshold

    @property
    def has_nonfoot_contact(self):
        return to_torch([False], device=self._device)

    @property
    def foot_contact_forces(self):
        return to_torch([self._raw_state.footForce], device=self._device)

    @property
    def foot_contact_forces_numpy(self):
        return np.array(self._raw_state.footForce)

    @property
    def time_since_reset(self):
        return to_torch([self.data.time - self._last_reset_time], device=self._device)

    @property
    def time_since_reset_scalar(self):
        return self.data.time - self._last_reset_time

    @property
    def raw_state(self):
        return self._raw_state

    @property
    def state_estimator(self):
        return self._state_estimator

    @property
    def control_timestep(self):
        return self.model.opt.timestep

    @property
    def has_body_contact(self):
        return torch.zeros(self._num_envs, device=self._device)
