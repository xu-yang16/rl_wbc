"""Example of running the phase gait generator."""

import time
from typing import Sequence
import ml_collections
import argparse

import numpy as np
import scipy

import isaacgym
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.robots.common.motors import MotorControlMode
from src.utilities.torch_utils import to_torch

from tqdm import tqdm
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--total_time_secs", type=float, default=60.0)
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--show_gui", type=bool, default=True)
parser.add_argument("--use_real_robot", type=int, default=1)
args = parser.parse_args()


def create_sim(sim_conf):
    from isaacgym import gymapi, gymutil

    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    sim = gym.create_sim(
        sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params
    )

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
    else:
        viewer = None

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 0.0
    plane_params.dynamic_friction = 0.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    return sim, viewer


def get_init_positions(
    num_envs: int, distance: float = 1.0, device: str = "cpu"
) -> Sequence[float]:
    num_cols = int(np.sqrt(num_envs))
    init_positions = np.zeros((num_envs, 3))
    for idx in range(num_envs):
        init_positions[idx, 0] = idx // num_cols * distance
        init_positions[idx, 1] = idx % num_cols * distance
        init_positions[idx, 2] = 0.34
    return to_torch(init_positions, device=device)


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 0.6
    vy = 0.2
    wz = 0.8

    time_points = (0, 3, 6, 9, 12, 15)
    speed_points = (
        (0, 0, 0),
        (vx, 0, 0),
        (0, -vy, 0),
        (0, 0, wz),
        (0, 0, -wz),
        (0, 0, 0),
    )

    speed = scipy.interpolate.interp1d(
        time_points, speed_points, kind="nearest", fill_value="extrapolate", axis=0
    )(t)

    return speed


def get_gait_config():
    config = ml_collections.ConfigDict()
    config.stepping_frequency = 2.0  # 1
    config.initial_offset = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32) * (
        2 * np.pi
    )  # FR, FL, RR, RL
    config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32) * 1.0
    return config


if __name__ == "__main__":
    if args.use_real_robot == 0:
        import isaacgym

    import torch

    if args.use_real_robot == 0:
        from configs.sim_config.isaac_config import get_asset_config, get_sim_config
        from src.robots.go1_isaac import Go1Isaac

        sim_conf = get_sim_config()
        sim, viewer = create_sim(sim_conf)
        robot = Go1Isaac(
            num_envs=args.num_envs,
            sim=sim,
            viewer=viewer,
            sim_config=sim_conf,
            motor_control_mode=MotorControlMode.HYBRID,
            motor_torque_delay_steps=5,
        )
    elif args.use_real_robot == 1:
        from src.robots import go1_mujoco

        robot = go1_mujoco.Go1Mujoco(
            num_envs=args.num_envs,
            device="cuda" if args.use_gpu else "cpu",
            motor_control_mode=MotorControlMode.HYBRID,
            motor_torque_delay_steps=0,
        )
    elif args.use_real_robot == 2:
        from src.robots import go1_robot

        robot = go1_robot.Go1Robot(
            num_envs=args.num_envs,
            motor_control_mode=MotorControlMode.HYBRID,
        )

    gait_config = get_gait_config()
    gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator, foot_landing_clearance=0.0, foot_height=0.1
    )
    torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        robot,
        desired_body_height=0.26,
        weight_ddq=np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 1.0]),
        body_mass=13.076,
        body_inertia=np.diag(np.array([0.14, 0.35, 0.35]) * 1.5),
        foot_friction_coef=0.4,
        solver_type="baseline",
        friction_type="pyramid",
        iter=20,
        warm_up=True,
    )

    robot.reset()
    num_envs, num_dof = robot.num_envs, robot.num_dof
    steps_count = 0

    start_time = time.time()
    pbar = tqdm(total=args.total_time_secs)

    contact_force = np.zeros((0, 4), dtype=float)
    joint_torques = np.zeros((0, 12), dtype=float)

    with torch.inference_mode():
        while robot.time_since_reset <= args.total_time_secs:
            if args.use_real_robot == 2:
                robot.state_estimator.update_foot_contact(
                    gait_generator.desired_contact_state
                )  # pytype: disable=attribute-error
            gait_generator.update()
            swing_leg_controller.update()

            # Update speed comand
            command = _generate_example_linear_angular_speed(
                robot.time_since_reset_scalar
            )
            robot.set_desired_velocity(command)
            # print(lin_command, ang_command)
            torque_optimizer.desired_linear_velocity = [
                command[0],
                command[1],
                0.0,
            ]
            torque_optimizer.desired_angular_velocity = [0.0, 0.0, command[2]]

            motor_action, desired_acc, solved_acc, grf, solver_time = (
                torque_optimizer.get_action(
                    gait_generator.desired_contact_state,
                    swing_foot_position=swing_leg_controller.desired_foot_positions,
                )
            )
            robot.step(motor_action)
            steps_count += 1
            # logger.debug(f"grf={grf}")
            if args.use_real_robot == 0:  # for isaac gym
                robot.render()

            if steps_count % 50 == 1:
                print(
                    f"hz={steps_count / (time.time() - start_time)}, time={robot.time_since_reset_scalar}"
                )

            # contact_force = np.vstack(
            #     [contact_force, robot.foot_contact_forces[0, :].cpu().numpy()]
            # )
            # joint_torques = np.vstack(
            #     [joint_torques, robot.motor_torques[0].cpu().numpy()]
            # )
            # logger.debug(f"contact force={robot.foot_contact_forces}")

            # if robot.time_since_reset > 10 or steps_count > 600:
            #     break
        exit(0)
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(contact_force)
        # plt.show()

        plt.figure(2)
        plt.plot(joint_torques)
        plt.show()

        print("Wallclock time: {}".format(time.time() - start_time))
