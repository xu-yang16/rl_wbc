import numpy as np
import os.path as osp


class RobotParamManager:
    def __init__(self, robot_name: str):
        if robot_name == "go1":
            self.robot = Go1()
        elif robot_name == "a1":
            self.robot = A1()
        elif robot_name == "mini_cheetah":
            self.robot = MiniCheetah()
        elif robot_name == "anymal":
            self.robot = Anymal()
        elif robot_name == "go2":
            self.robot = Go2()
        else:
            raise ValueError("Invalid robot name")


class Go1:
    def __init__(self):
        self.urdf_path = "data/go1/urdf/go1.urdf"
        self.hip_offset = np.array(
            [
                [0.1881, -0.04675, 0.0],
                [0.1881, 0.04675, 0.0],
                [-0.1881, -0.04675, 0.0],
                [-0.1881, 0.04675, 0.0],
            ]
        ) - np.array([0.0116053, 0.00442221, 0.000106692])
        self.l_hip = np.array([-1, 1, -1, 1]) * 0.08
        self.l_up = 0.213
        self.l_low = 0.213
        self.foot_size = 0.02

        # desired behavior
        self.normal_stand = np.array(
            [
                [0.1835, -0.131, 0.0],
                [0.1835, 0.122, 0.0],
                [-0.1926, -0.131, 0.0],
                [-0.1926, 0.122, 0.0],
            ]
        )
        self.desired_body_height = 0.26

        # inertia params
        self.mass = 5.204 + (0.591 + 0.92 + 0.135862) * 4
        self.inertia = np.diag(np.array([0.14, 0.35, 0.35]) * 1.5)

        # initial pos
        self.initial_pos = np.array(
            [[0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8]]
        )
        self.kp_standup, self.kd_standup = 100, 1
        self.kp_wbc, self.kd_wbc = 30, 1


class A1:
    def __init__(self):
        self.urdf_path = "data/a1/urdf/a1.urdf"
        self.hip_offset = np.array(
            [
                [0.183, -0.047, 0.0],
                [0.183, 0.047, 0.0],
                [-0.183, -0.047, 0.0],
                [-0.183, 0.047, 0.0],
            ]
        )  # - np.array([0.0127283, 0.00218554, 0.000514891])
        self.l_hip = np.array([-1, 1, -1, 1]) * 0.08505
        self.l_up = 0.2
        self.l_low = 0.2
        self.foot_size = 0.02

        # desired behavior
        self.normal_stand = np.array(
            [
                [0.1805, -0.1308, 0.0],
                [0.1805, 0.1308, 0.0],
                [-0.1805, -0.1308, 0.0],
                [-0.1805, 0.1308, 0.0],
            ]
        )
        self.desired_body_height = 0.27

        # inertia params
        self.mass = 4.714 + (0.696 + 1.013 + 0.226) * 4
        self.inertia = np.diag(np.array([0.14, 0.35, 0.35]) * 1.5)

        # initial pos
        self.initial_pos = np.array(
            [[0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8]]
        )
        self.kp_standup, self.kd_standup = 100, 1
        self.kp_wbc, self.kd_wbc = 30, 1


class Go2:
    def __init__(self):
        self.urdf_path = "data/go2/urdf/go2.urdf"
        self.hip_offset = np.array(
            [
                [0.1934, -0.0465, 0.0],
                [0.1934, 0.0465, 0.0],
                [-0.1934, -0.0465, 0.0],
                [-0.1934, 0.0465, 0.0],
            ]
        ) + np.array([0.0, -0.0, 0.0])
        self.l_hip = np.array([-1, 1, -1, 1]) * 0.0955
        self.l_up = 0.213
        self.l_low = 0.213
        self.foot_size = 0.021

        # desired behavior
        self.normal_stand = np.array(
            [
                [0.1934, -0.1700, 0.0],
                [0.1934, 0.1700, 0.0],
                [-0.1934, -0.1700, 0.0],
                [-0.1934, 0.1700, 0.0],
            ]
        )
        self.desired_body_height = 0.32

        # inertia params
        self.mass = 6.921 + (0.678 + 1.152 + 0.241352) * 4 + 3
        self.inertia = np.diag(np.array([0.5, 0.35, 0.35]) * 1.5)

        # initial pos
        self.initial_pos = np.array(
            [[-0.1, 0.9, -1.8], [0.1, 0.9, -1.8], [-0.1, 0.9, -1.8], [0.1, 0.9, -1.8]]
        )
        self.initial_pos[:, 0] *= 1
        self.kp_standup, self.kd_standup = 100, 1
        self.kp_wbc, self.kd_wbc = 40, 1


class MiniCheetah:
    def __init__(self):
        self.urdf_path = "data/mini_cheetah/urdf/mini_cheetah.urdf"
        self.hip_offset = np.array(
            [
                [0.19, -0.049, 0.0],
                [0.19, 0.049, 0.0],
                [-0.19, -0.049, 0.0],
                [-0.19, 0.049, 0.0],
            ]
        ) + np.array([-0.05, -0.01, 0.0])
        self.l_hip = np.array([-1, 1, -1, 1]) * 0.062
        self.l_up = 0.209
        self.l_low = 0.18
        self.foot_size = 0.015

        # desired behavior
        self.normal_stand = np.array(
            [
                [0.19, -0.14, 0.0],
                [0.19, 0.14, 0.0],
                [-0.19, -0.14, 0.0],
                [-0.19, 0.14, 0.0],
            ]
        )
        self.desired_body_height = 0.26

        # inertia params
        self.mass = 3.3 + (0.54 + 0.634 + 0.214) * 4
        self.inertia = np.diag(np.array([0.14, 0.35, 0.35]) * 1)

        # initial pos
        self.initial_pos = np.array(
            [[0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8], [0, 0.9, -1.8]]
        )
        self.kp_standup, self.kd_standup = 100, 1
        self.kp_wbc, self.kd_wbc = 30, 1


class Anymal:
    def __init__(self):
        self.urdf_path = "data/anymal/urdf/anymal.urdf"
        self.hip_offset = np.array(
            [
                [0.277, -0.116, 0.0],
                [0.277, 0.116, 0.0],
                [-0.277, -0.116, 0.0],
                [-0.277, 0.116, 0.0],
            ]
        ) + np.array([0.03, 0.0, 0.0])
        self.l_hip = np.array([-1, 1, -1, 1]) * 0.0
        self.l_up = 0.27
        self.l_low = 0.29
        self.foot_size = 0.01

        # desired behavior
        self.normal_stand = np.array(
            [
                [0.34, -0.19, 0.0],
                [0.34, 0.19, 0.0],
                [-0.34, -0.19, 0.0],
                [-0.34, 0.19, 0.0],
            ]
        )
        self.desired_body_height = 0.38

        # inertia params
        self.mass = 19.2035 + (1.42462 + 1.63498 + 0.472163) * 4
        self.inertia = np.diag(np.array([0.14, 0.35, 0.35]) * 10)

        # initial pos
        self.initial_pos = np.array(
            [[0, 0.7, -1.3], [0, 0.7, -1.3], [0, 0.7, -1.3], [0, 0.7, -1.3]]
        )
        self.kp_standup, self.kd_standup = 100, 1
        self.kp_wbc, self.kd_wbc = 100, 1
