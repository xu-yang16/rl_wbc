"""Simple state estimator for Go1 robot."""

import rospy, time, threading
from geometry_msgs.msg import PoseStamped  # ros msg PoseStamped
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

import numpy as np


def convert_to_skew_symmetric(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def normalize_angle(angle):
    """Normalize the angle to the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class RobotStateVicon:
    """Estimates base velocity of A1 robot.
    The velocity estimator consists of a state estimator for CoM velocity.
    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.
    """

    def __init__(self, robot=None):
        rospy.init_node("robot_state_vicon", anonymous=True)
        # rospy.Subscriber(
        #     f"/vrpn_client_node/xdog/pose", PoseStamped, self.callback, queue_size=10
        # )
        # self.pub = rospy.Publisher(
        #     "robot_state_vicon", Float64MultiArray, queue_size=10
        # )

        # time related
        self._counter = 0
        self.start_time = time.time()
        self._last_timestamp = 0.0
        self.dt = 1 / 100.0
        self.raw_vel_matrix = np.zeros((15, 3))

        self.raw_ang_vel = np.zeros(3)
        self.z = np.zeros(3)
        # KF variables
        self.world_lin_x_variable = np.zeros(3)
        self.world_lin_y_variable = np.zeros(3)
        self.world_lin_z_variable = np.zeros(3)
        self.world_ang_x_variable = np.zeros(3)
        self.world_ang_y_variable = np.zeros(3)
        self.world_ang_z_variable = np.zeros(3)

        self._base_frame = np.eye(3)
        self._euler = np.zeros(3)
        self._estimated_position = np.array([0.0, 0.0, 0.26])

        self._world_lin_vel = np.zeros(3)
        self._world_ang_vel = np.zeros(3)
        self._local_lin_vel = np.zeros(3)
        self._local_ang_vel = np.zeros(3)

        # system and measurement model
        self.F = np.array(
            [[1, self.dt, self.dt * self.dt], [0, 1.0, self.dt], [0, 0, 1]]
        )
        self.H = np.array([[1.0, 0, 0]])

        self.process_std = 1.0
        self.Q = (
            np.array(
                [
                    [0.25 * self.dt**4, 0.5 * self.dt**3, 0.5 * self.dt**2],
                    [0.5 * self.dt**3, self.dt**2, self.dt],
                    [0.5 * self.dt**2, self.dt, 1],
                ]
            )
            * self.process_std**2
        )
        self.rts_std = 0.01
        self.R = np.eye(1) * self.rts_std**2

        # kf related
        self.P = np.eye(3) * 500.0

        self.reset()
        # self.poll_running = True
        # self.run_thread = threading.Thread(target=self.spin, daemon=False)
        # self.run_thread.start()

    def reset(self):
        self._last_timestamp = 0
        self._world_lin_vel = np.zeros(3)
        self._world_ang_vel = np.zeros(3)
        self._base_frame = np.eye(3)
        self._estimated_position = np.array([0.0, 0.0, 0.26])
        self._local_ang_vel = np.zeros(3)
        self._local_lin_vel = np.zeros(3)

    def callback(self, data, dt=0.01):
        self._counter += 1
        if self._counter % 15 == 0:
            pass
            print(f"vicon est hz={self._counter / (time.time() - self.start_time)}")
        self._estimated_position = np.array(
            [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        )
        orientation = R.from_quat(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w,
            ]
        )

        self._euler = orientation.as_euler("xyz", degrees=False)
        self._base_frame = orientation.as_matrix()

        # if first time
        if self._last_timestamp == 0.0:
            self.world_lin_x_variable[0] = self._estimated_position[0]
            self.world_lin_y_variable[0] = self._estimated_position[1]
            self.world_lin_z_variable[0] = self._estimated_position[2]
            self.world_ang_x_variable[0] = self._euler[0]
            self.world_ang_y_variable[0] = self._euler[1]
            self.world_ang_z_variable[0] = self._euler[2]
        # update velocity
        if self._last_timestamp == 0.0:
            # First timestamp received, return an estimated delta_time.
            self.dt = 1 / 100.0
        else:
            self.dt = dt
        self._last_timestamp = data.header.stamp.to_sec()
        # print(f"state est dt={self.dt}")
        self.kf_update(self._estimated_position, self._euler)
        # self.publish()

    def kf_update(self, world_pos, world_euler):
        self.F = np.array([[1, self.dt, self.dt**2], [0, 1.0, self.dt], [0, 0, 1]])

        self.Q = (
            np.array(
                [
                    [0.25 * self.dt**4, 0.5 * self.dt**3, 0.5 * self.dt**2],
                    [0.5 * self.dt**3, self.dt**2, self.dt],
                    [0.5 * self.dt**2, self.dt, 1],
                ]
            )
            * self.process_std**2
        )

        # self.Q = np.diag([1, 5, 10]) * self.process_std**2

        # update variables
        x_pred_lin_x = self.F.dot(self.world_lin_x_variable)
        x_pred_lin_y = self.F.dot(self.world_lin_y_variable)
        x_pred_lin_z = self.F.dot(self.world_lin_z_variable)
        x_pred_ang_x = self.F.dot(self.world_ang_x_variable)
        x_pred_ang_y = self.F.dot(self.world_ang_y_variable)
        x_pred_ang_z = self.F.dot(self.world_ang_z_variable)
        P_pred = self.F.dot(self.P).dot(self.F.T) + self.Q
        S = self.H.dot(P_pred).dot(self.H.T) + self.R
        K = P_pred.dot(self.H.T).dot(np.linalg.pinv(S))
        # print(f"K={K}")
        self.K = K

        y_lin_x = world_pos[0] - self.H.dot(x_pred_lin_x)
        y_lin_y = world_pos[1] - self.H.dot(x_pred_lin_y)
        y_lin_z = world_pos[2] - self.H.dot(x_pred_lin_z)
        y_ang_x = normalize_angle(world_euler[0] - self.H.dot(x_pred_ang_x))
        y_ang_y = normalize_angle(world_euler[1] - self.H.dot(x_pred_ang_y))
        y_ang_z = normalize_angle(world_euler[2] - self.H.dot(x_pred_ang_z))
        # TODO: ang pos [-\pi, \pi]
        # self.raw_ang_vel[0] = y_ang_x / self.dt
        # self.raw_ang_vel[1] = y_ang_y / self.dt
        # self.raw_ang_vel[2] = y_ang_z / self.dt
        # self.z = np.array([y_lin_x[0], y_lin_y[0], y_lin_z[0]])

        self.world_lin_x_variable = x_pred_lin_x + K.dot(y_lin_x)
        self.world_lin_y_variable = x_pred_lin_y + K.dot(y_lin_y)
        self.world_lin_z_variable = x_pred_lin_z + K.dot(y_lin_z)
        self.world_ang_x_variable = x_pred_ang_x + K.dot(y_ang_x)
        self.world_ang_y_variable = x_pred_ang_y + K.dot(y_ang_y)
        self.world_ang_z_variable = x_pred_ang_z + K.dot(y_ang_z)
        self.world_ang_x_variable[0] = normalize_angle(self.world_ang_x_variable[0])
        self.world_ang_y_variable[0] = normalize_angle(self.world_ang_y_variable[0])
        self.world_ang_z_variable[0] = normalize_angle(self.world_ang_z_variable[0])

        self.P = (np.eye(3) - K.dot(self.H)).dot(P_pred)

        # aggregate
        self._world_lin_vel = np.array(
            [
                self.world_lin_x_variable[1],
                self.world_lin_y_variable[1],
                self.world_lin_z_variable[1],
            ]
        )
        self._world_ang_vel = np.array(
            [
                self.world_ang_x_variable[1],
                self.world_ang_y_variable[1],
                self.world_ang_z_variable[1],
            ]
        )
        # print(f"_estimated_position={self._estimated_position}")
        # print(
        #     "world_lin_vel = {{{: >+6.4f}, {: >+6.4f}, {: >+6.4f}}}".format(
        #         self._world_lin_vel[0], self._world_lin_vel[1], self._world_lin_vel[2]
        #     )
        # )

        # print(
        #     "local_lin_vel = {{{: >+6.2f}, {: >+6.2f}, {: >+6.2f}}}".format(
        #         self._local_lin_vel[0], self._local_lin_vel[1], self._local_lin_vel[2]
        #     )
        # )
        # print(
        #    "local_ang_vel = {{{: >+6.2f}, {: >+6.2f}, {: >+6.2f}}}".format(
        #        self._local_ang_vel[0], self._local_ang_vel[1], self._local_ang_vel[2]
        #    )
        # )

        # print(f"euler={self._euler}")

    def publish(self):
        msg = Float64MultiArray()
        msg.data = np.concatenate(
            (
                self._estimated_position,
                self._euler,
                self._world_lin_vel,
                self._world_ang_vel,
                self._local_lin_vel,
                self._local_ang_vel,
            )
        )
        self.pub.publish(msg)
        # print(f"published: {msg.data}")

    def update_foot_contact(self, foot_contact):
        self._foot_contact = foot_contact.cpu().numpy().reshape(4)

    @property
    def estimated_velocity(self):
        # FIXME: this ie wrong
        euler = self._euler.copy()
        rot_mat = R.from_euler("xyz", euler).as_matrix()
        return rot_mat.T.dot(self._world_lin_vel)

    @property
    def local_estimated_velocity(self):
        self._local_lin_vel = self._base_frame.T.dot(self._world_lin_vel)
        # self._local_lin_vel = self._base_frame.T.dot(
        #     np.mean(self.raw_vel_matrix, axis=0)
        # )
        return self._local_lin_vel.copy()

    @property
    def estimated_position(self):
        return self._estimated_position - np.array([0.0, 0.0, 0.09])

    @property
    def local_angular_velocity(self):
        self._local_ang_vel = self._base_frame.T.dot(self._world_ang_vel)
        return self._local_ang_vel.copy()


if __name__ == "__main__":
    robot_state_vicon = RobotStateVicon()
    data = PoseStamped()
    length = 500
    truth = np.zeros((length, 3))
    estimate = np.zeros((length, 3))
    truth_pos = np.zeros((length, 3))
    estimate_pos = np.zeros((length, 3))
    duration = 0
    for i in range(length):
        dt = np.random.rand() * 0.005 + 0.018
        duration += dt
        data.pose.position.x = (
            np.sin(3 * duration) + np.cos(2 * duration) + np.random.randn() * 0.001
        )
        data.pose.position.y = (
            np.cos(3 * duration) + np.sin(5 * duration) + np.random.randn() * 0.001
        )
        data.pose.position.z = 0.26 + np.random.randn() * 0.001
        data.pose.orientation.x = 0
        data.pose.orientation.y = 0
        data.pose.orientation.z = 0
        data.pose.orientation.w = 1
        data.header.stamp = rospy.Time.now()
        robot_state_vicon.callback(data, dt)
        truth[i, :] = np.array(
            [
                np.cos(3 * duration) * 3 - 2 * np.sin(2 * duration),
                -np.sin(3 * duration) * 3 + 5 * np.cos(5 * duration),
                0.0,
            ]
        )
        estimate[i] = robot_state_vicon._world_lin_vel
        truth_pos[i] = np.array(
            [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        )
        estimate_pos[i] = robot_state_vicon._estimated_position
    print(f"K={robot_state_vicon.K}")

    x = truth[:, 0]
    y = estimate[:, 0]
    correlation = np.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    # Find the lag with the maximum correlation
    import matplotlib.pyplot as plt

    lag_max = lags[np.argmax(correlation)]
    print(f"lag={lag_max}")
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlation)
    plt.title("Cross-Correlation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation coefficient")
    plt.axvline(
        lag_max, color="red", linestyle="--", label=f"Max correlation at lag={lag_max}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot traj (x, y) in plane
    plt.plot(truth_pos[:, 0], truth_pos[:, 1], label="truth")
    plt.plot(estimate_pos[:, 0], estimate_pos[:, 1], label="estimate")
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(truth[:, 0], label="truth")
    axs[0].plot(estimate[:, 0], label="estimate")
    axs[0].set_title("x")
    axs[1].plot(truth[:, 1], label="truth")
    axs[1].plot(estimate[:, 1], label="estimate")
    axs[1].set_title("y")
    axs[2].plot(truth[:, 2], label="truth")
    axs[2].plot(estimate[:, 2], label="estimate")
    axs[2].set_title("z")
    plt.show()
    plt.legend()
