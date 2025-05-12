"""Config for Go1 speed tracking environment."""

from ml_collections import ConfigDict
import numpy as np
import torch


def get_gait_config():
    gait_config = ConfigDict()
    gait_config.stepping_frequency = 2
    gait_config.initial_offset = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32) * (
        2 * np.pi
    )

    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    return gait_config


def get_terrain_config():
    config = ConfigDict()
    config.terrain = "trimesh"
    config.border_size = 25
    config.terrain_length = 8
    config.terrain_width = 8
    config.num_rows = 10
    config.num_cols = 10
    config.horizontal_scale = 0.1
    config.vertical_scale = 0.005
    config.slope_threshold = 0.75
    # slope_smooth, slope_rough, stairs_up, stairs_down, discrete
    config.terrain_proportions = [0.3, 0.3, 0.0, 0.0, 0.4]
    config.max_init_level = 1
    config.randomize_steps = False
    config.randomize_step_width = True
    # Curriculum setup
    config.curriculum = True
    config.restitution = 0.0

    return config


def get_controller_config():
    config = ConfigDict()

    # RL control frequency
    config.env_dt = 0.02

    # terrain
    config.terrain = "trimesh"
    config.foot_friction = 0.7  # 0.7

    # command
    config.episode_length_s = 20.0
    config.resampling_time = 10.0
    config.use_command_curriculum = False
    config.goal_lb = torch.tensor(
        [0.0, 0.0, -0.0], dtype=torch.float
    )  # Lin_x, Lin_y, Rot_z
    config.goal_ub = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)

    # action
    config.action_lb = np.array(
        [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0] + [-0.05, -0.03, -0.05] * 1
    )
    config.action_ub = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0] + [0.05, 0.03, 0.05] * 1)
    config.action_lb[:6] *= 2
    config.action_ub[:6] *= 2

    config.num_actions = config.action_lb.shape[0]
    config.num_actor_obs = 77
    config.history_length = 10
    config.num_critic_obs = 77

    config.base_position_kp = np.array([0.0, 0.0, 50.0]) * 1.0
    config.base_position_kd = np.array([10.0, 10.0, 10.0]) * 1.0
    config.base_orientation_kp = np.array([50.0, 50.0, 0.0]) * 1.0
    config.base_orientation_kd = np.array([10.0, 10.0, 10.0]) * 1.0
    config.qp_foot_friction_coef = 0.4
    config.qp_weight_ddq = np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 1.0])
    config.qp_body_mass = 13.076
    config.qp_body_inertia = np.diag(np.array([0.14, 0.35, 0.35]) * 1.5)

    # solver config
    config.qp_warm_up = True
    config.qp_iter = 20
    config.solver_type = "pdhg"
    config.friction_type = "pyramid"

    config.swing_foot_height = 0.1
    config.swing_foot_landing_clearance = 0.0

    return config


def get_rl_config():
    config = ConfigDict()
    config.randomized = True
    # randomize_payload_mass [-1, 2]; randomize_com_displacement [-0.05, 0.05]
    # randomize_friction [0.2, 1.25]; randomize_motor_strength [0.9, 1.1]
    # disturbance [-30.0, 30.0], 8s;
    # push_robots 1.0, 15s

    # motor
    config.motor_torque_delay_steps = 0

    config.terminate_on_body_contact = True
    config.terminate_on_limb_contact = False
    config.terminate_on_height = 0.09
    config.use_penetrating_contact = False

    config.rewards = [
        ("roll_pitch", 0.010),
        ("height", 0.02),
        # ("lin_vel_z", 0.02),
        ("ang_vel_xy", 2e-6),
        # ("out_of_bound_action", 0.01),
        ("legged_gym_tracking_lin_vel", 2.0 * 1e-3),
        ("legged_gym_tracking_ang_vel", 1.0 * 1e-3),
        ("legged_gym_action_rate", 0.01 * 1e-3),
        # ("alive", 0.01),
        # ("acc_action_penalty", 0.01 * 1e-3),
    ]
    config.clip_negative_reward = False
    config.normalize_reward_by_phase = False

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False

    return config


def get_config():
    config = ConfigDict()

    config.gait = get_gait_config()

    config.update(get_controller_config())
    config.update(get_rl_config())

    config.update(get_terrain_config())

    return config
