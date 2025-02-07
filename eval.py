"""Evaluate a trained policy."""

from absl import app
from absl import flags


from datetime import datetime
import os, time, copy, pickle
import isaacgym
import torch
import numpy as np
import scipy
from loguru import logger

flags.DEFINE_string("logdir", "logs/", "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_integer("use_real_robot", 1, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
flags.DEFINE_string("name", "go1", "robot name.")
FLAGS = flags.FLAGS


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    actor_path = os.path.join(path, "actor.jit")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(actor_path)
    logger.warning(f"Exported actor to {actor_path}")


def find_latest_pt(logdir):
    # 检查是否指定了具体的文件
    if os.path.isfile(logdir) and logdir.endswith(".pt"):
        return logdir

    # 初始化最新文件变量
    latest_pt = None
    latest_time = -1

    # 遍历所有子目录
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_pt = file_path

    return latest_pt


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 1
    vy = 0.6
    wz = 0.8

    # time_points = (0, 1, 9, 10, 15, 20, 25, 30)
    # speed_points = ((0, 0, 0, 0), (0, 0.6, 0, 0), (0, 0.6, 0, 0), (vx, 0, 0, 0),
    #                 (0, 0, 0, -wz), (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    time_points = (0, 3, 6, 9, 12, 15)
    speed_points = (
        (0, 0, 0, 0),
        (0, 0, 0, wz),
        (vx, 0, 0, 0),
        (0, 0, -vy, 0),
        (vx, 0, 0, wz),
        (0, 0, 0, 0),
    )

    speed = scipy.interpolate.interp1d(
        time_points, speed_points, kind="nearest", fill_value="extrapolate", axis=0
    )(t)

    return speed[0:3], speed[3]


def main(argv):
    del argv  # unused

    # for record and compute mse
    desired_vel = np.empty((0, 3), dtype=float)
    actual_vel = np.empty((0, 3), dtype=float)
    joint_vel = np.empty((0, 12), dtype=float)
    joint_torque = np.empty((0, 12), dtype=float)

    contact_force = np.zeros((0, 4), dtype=float)

    # if use isaac gym
    if FLAGS.use_real_robot == 0:
        import isaacgym

    from src.envs.trot_env import TrotEnv
    from src.envs.trot_env_e2e import TrotEnvE2E
    import torch
    from rsl_rl.runners import OnPolicyRunner
    import yaml
    from src.envs import env_wrappers

    torch.set_printoptions(precision=2, sci_mode=False)

    device = "cuda" if FLAGS.use_gpu else "cpu"

    # Find the latest policy pt
    policy_path = find_latest_pt(FLAGS.logdir)
    logger.warning(f"latest policy path: {policy_path}")
    reward_config_path = os.path.join(
        os.path.dirname(policy_path), "config/reward_config.yaml"
    )
    training_config_path = os.path.join(
        os.path.dirname(policy_path), "config/training_config.yaml"
    )

    with open(reward_config_path, "r", encoding="utf-8") as f:
        reward_config = yaml.load(f, Loader=yaml.Loader)
    with open(training_config_path, "r", encoding="utf-8") as f:
        training_config = yaml.load(f, Loader=yaml.Loader)

    reward_config.randomized = False
    logger.debug(f"reward_config={reward_config}")
    env = TrotEnv(
        num_envs=FLAGS.num_envs,
        device=device,
        config=reward_config,
        show_gui=FLAGS.show_gui,
        use_real_robot=FLAGS.use_real_robot,
        robot_name=FLAGS.name,
    )
    env = env_wrappers.RangeNormalize(env)
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (
            not FLAGS.use_contact_sensor
        )

    # Retrieve policy
    runner = OnPolicyRunner(env, training_config, policy_path, device=device)
    runner.load(policy_path)

    policy = runner.get_inference_policy()
    # runner.alg.actor_critic.train()

    export_policy_as_jit(runner.alg.actor_critic, os.path.dirname(policy_path))

    # Reset environment
    state, _ = env.reset()
    total_reward = torch.zeros(FLAGS.num_envs, device=device)
    steps_count = 0

    start_time = time.time()
    logs = []
    with torch.inference_mode():
        # try:
        while True:
            steps_count += 1
            action = policy(state)

            lin_command, ang_command = _generate_example_linear_angular_speed(
                env.robot.time_since_reset
            )
            env._env._desired_cmd[:, 0] = lin_command[0]
            env._env._desired_cmd[:, 1] = lin_command[1]
            env._env._desired_cmd[:, 2] = ang_command
            # logger.debug(f"cmd: {lin_command}, {ang_command}")

            state, _, reward, done, info = env.step(action)

            # current_sim_time = env.robot.time_since_reset_scalar
            # if steps_count % 100 == 1:
            #     print(
            #         f"hz={steps_count / current_sim_time}, sim_time={current_sim_time}, real_time={time.time() - start_time}"
            #     )

            # # for data record
            # desired_vel = np.vstack(
            #     [
            #         desired_vel,
            #         np.array([lin_command[0], lin_command[1], ang_command]),
            #     ]
            # )
            # # logger.debug(
            # #     f"env._env._robot.base_velocity_body_frame={env._env._robot.base_velocity_body_frame.shape}"
            # # )
            # actual_vel = np.vstack(
            #     [
            #         actual_vel,
            #         np.array(
            #             [
            #                 env._env._robot.base_velocity_body_frame[0, 0],
            #                 env._env._robot.base_velocity_body_frame[0, 1],
            #                 env._env._robot.base_angular_velocity_body_frame[0, 2],
            #             ]
            #         ),
            #     ]
            # )
            # joint_vel = np.vstack([joint_vel, env._env._robot.motor_velocities[0, :]])
            # joint_torque = np.vstack(
            #     [joint_torque, env._env._robot.motor_torques[0, :]]
            # )
            # contact_force = np.vstack(
            #     [contact_force, env._env._robot.foot_contact_forces_numpy]
            # )

            # if current_sim_time > 15:
            #     logger.warning("Time out! current_sim_time = {}", current_sim_time)
            #     break
            #   total_reward += reward
            # logs.extend(info["logs"])
        # except Exception as e:
        #     print(e)

    print(f"Time elapsed: {time.time() - start_time}")

    import matplotlib.pyplot as plt

    plt.plot(contact_force[:, 0], label="FR")
    plt.plot(contact_force[:, 1], label="FL")
    plt.plot(contact_force[:, 2], label="RR")
    plt.plot(contact_force[:, 3], label="RL")
    plt.show()
    # compute mse
    all_mse = np.mean((desired_vel - actual_vel) ** 2, axis=0)
    total_mse = np.sum(all_mse)

    power = np.sum(np.abs(joint_torque * joint_vel)) / 15
    # save to npz
    exp_name = FLAGS.logdir.split("/")[-1]

    file_name = f"sim_plots/rl_wbc/{exp_name}_mse{total_mse}_power{power}.npz"
    logger.warning(f"Saving to {file_name}")
    np.savez(
        file_name,
        desired_vel=desired_vel,
        actual_vel=actual_vel,
        joint_vel=joint_vel,
        joint_torque=joint_torque,
    )

    # if FLAGS.use_real_robot == 2 or True:
    #     if FLAGS.use_real_robot == 2:
    #         mode = "real"
    #     elif FLAGS.use_real_robot == 1:
    #         mode = "mujoco"
    #     elif FLAGS.use_real_robot == 0:
    #         mode = "isaac"
    #     output_dir = f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{policy_path.split('/')[1]}.pkl"
    #     output_path = os.path.join("1125_mj_data", output_dir)

    #     with open(output_path, "wb") as fh:
    #         pickle.dump(logs, fh)
    #     print(f"Data logged to: {output_path}")
    env.close()


if __name__ == "__main__":
    app.run(main)
