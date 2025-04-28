"""General class for (vectorized) robots."""

import os, sys
import sys
from typing import Any, List

from isaacgym import gymapi, gymtorch

import ml_collections
import numpy as np
import torch

from configs.sim_config.isaac_config import get_asset_config
from src.utilities.terrain import Terrain
from src.utilities.torch_utils import to_torch, quat_apply, torch_rand_float
from src.utilities.rotation_utils import (
    quat_to_rot_mat,
    get_euler_xyz_from_quaternion,
    angle_normalize,
)
from icecream import ic
from loguru import logger


class IsaacGymRobot:
    """General class for simulated quadrupedal robot.
    Functions:
        - reset: resets the robot to its initial state
        - step: applies the given action to the robot for a number of simulation steps
        - render: renders the robot in the simulation
        - set_foot_friction: sets the friction coefficient of the feet
        - set_foot_frictions: sets the friction coefficient of the feet for multiple environments
        - get_motor_angles_from_foot_positions: computes the motor angles from the foot positions
        - update_init_positions: updates the initial positions of the robot
        - _create_terrain: creates the terrain (plane or trimesh)
    Properties:
        - base_position: the position of the base of the robot
        - base_position_world: the position of the base of the robot in world frame
        - base_orientation_rpy: the orientation of the base of the robot in rpy
        - base_orientation_quat: the orientation of the base of the robot in quaternion
        - base_rot_mat: the rotation matrix of the base of the robot
        - base_rot_mat_t: the transpose of the rotation matrix of the base of the robot
        - projected_gravity: the gravity projected on the base of the robot

        - base_velocity_world_frame: the linear velocity of the base of the robot in world frame
        - base_velocity_body_frame: the linear velocity of the base of the robot in local frame
        - base_angular_velocity_world_frame: the angular velocity of the base of the robot in world frame
        - base_angular_velocity_body_frame: the angular velocity of the base of the robot in local frame

        - motor_positions: the positions of the motors of the robot
        - motor_velocities: the velocities of the motors of the robot
        - motor_torques: the torques of the motors of the robot

        - foot_height: the heights of the feet of the robot
        - foot_positions_in_base_frame: the positions of the feet of the robot in the base frame
        - foot_positions_in_world_frame: the positions of the feet of the robot in the world frame
        - foot_velocities_in_base_frame: the velocities of the feet of the robot in the base frame
        - foot_velocities_in_world_frame: the velocities of the feet of the robot in the world frame

        - contact_forces: the contact forces of the robot
        - motor_positions: the positions of the motors of the robot
        - motor_velocities: the velocities of the motors of the robot
        - motor_torques: the torques of the motors of the robot
        - torques: the torques of the robot

    """

    def __init__(
        self,
        sim: Any,
        viewer: Any,
        num_envs: int,
        urdf_path: str,
        sim_config: ml_collections.ConfigDict,
        motors: Any,
        feet_names: List[str],
        calf_names: List[str],
        thigh_names: List[str],
        num_actions: int = 6,
        num_actor_obs: int = 10,
        domain_rand: bool = False,
        terrain_type: str = "flat",
    ):
        """Initializes the robot class."""
        self.domain_rand = domain_rand
        self._gym = gymapi.acquire_gym()
        self._sim = sim
        self._viewer = viewer
        self._enable_viewer_sync = True
        self._sim_config = sim_config
        self._device = self._sim_config.sim_device
        self._num_envs = num_envs
        self._motors = motors
        self._feet_names = feet_names
        self._calf_names = calf_names
        self._thigh_names = thigh_names

        # terrain
        self.env_origins = torch.zeros(self._num_envs, 3, device=self._device)
        self._create_terrain(terrain_type)

        self._base_init_state = self._compute_base_init_state(self.env_origins)
        self._envs = []
        self._actors = []
        self._time_since_reset = torch.zeros(self._num_envs, device=self._device)

        if "cuda" in self._device:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

        self._load_urdf(urdf_path)
        self._gym.prepare_sim(self._sim)
        self._init_buffers()

        self._post_physics_step()
        # self.reset()

        # # for visualization
        self.subscribe_keyboard_event()
        self.enable_viewer_sync = True
        self.lock_viewer_to_robot = 1
        self.follow_robot_index = 0

        # # for debug
        self._action = torch.zeros(
            self._num_envs, num_actions, device=self._device, requires_grad=False
        )
        self._actor_obs = torch.zeros(
            self._num_envs, num_actor_obs, device=self._device, requires_grad=False
        )

    def _create_terrain(self, terrain_type: str):
        """Creates terrains.

        Note that we set the friction coefficient to all 0 here. This is because
        Isaac seems to pick the larger friction out of a contact pair as the
        actual friction coefficient. We will set the corresponding friction coef
        in robot friction.
        """
        if terrain_type == "flat":
            ic("terrain: flat ground")
            self._create_plane()
        elif terrain_type == "trimesh":
            ic("terrain: trimesh")
            self._create_trimesh()
        else:
            raise ValueError("Unknown terrain type")

    def get_terrain_config(self):
        from ml_collections import ConfigDict
        from src.utilities.terrain import GenerationMethod

        config = ConfigDict()
        # config.type = 'plane'
        config.type = "trimesh"
        config.terrain_length = 10
        config.terrain_width = 10
        config.border_size = 15
        config.num_rows = 10
        config.num_cols = 10
        config.horizontal_scale = 0.05
        config.vertical_scale = 0.005
        config.move_up_distance = 4.5
        config.move_down_distance = 2.5
        config.slope_threshold = 0.75
        config.generation_method = GenerationMethod.CURRICULUM
        config.max_init_level = 1
        config.terrain_proportions = dict(
            slope_smooth=0.0,
            slope_rough=0.0,
            stair=0.4,
            obstacles=0.0,
            stepping_stones=0.4,
            gap=0.1,
            pit=0.1,
        )
        config.randomize_steps = False
        config.randomize_step_width = True
        # Curriculum setup
        config.curriculum = True
        config.restitution = 0.0
        return config

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg."""
        terrain_config = self.get_terrain_config()
        terrain = Terrain(terrain_config, device=self._device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = terrain.vertices.shape[0]
        tm_params.nb_triangles = terrain.triangles.shape[0]

        tm_params.transform.p.x = -terrain_config.border_size
        tm_params.transform.p.y = -terrain_config.border_size
        tm_params.transform.p.z = 0.0

        tm_params.static_friction = 0.99
        tm_params.dynamic_friction = 0.99
        tm_params.restitution = terrain_config.restitution
        self._gym.add_triangle_mesh(
            self._sim,
            terrain.vertices.flatten(order="C"),
            terrain.triangles.flatten(order="C"),
            tm_params,
        )

        terrain_levels = torch.randint(
            0, terrain_config.num_rows, (self.num_envs,), device=self.device
        )
        terrain_types = torch.div(
            torch.arange(self._num_envs, device=self.device),
            (self._num_envs / terrain_config.num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        terrain_origins = terrain.env_origins[terrain_levels, terrain_types]

        # robot initial positions
        init_positions = terrain_origins.clone()
        init_positions[:, 2] += 0.268 + 0.04
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        sobol_points = sobol_engine.draw(self._num_envs)
        sobol_points = (sobol_points - 0.5) * 2.5
        init_positions[:, :2] += to_torch(sobol_points, device=self._device)

        self.env_origins = init_positions

    def _create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self._gym.add_ground(self._sim, plane_params)

        num_cols = int(np.sqrt(self._num_envs))

        distance = 2.0
        indices = torch.arange(self._num_envs)
        self.env_origins[:, 0] = (indices // num_cols) * distance
        self.env_origins[:, 1] = (indices % num_cols) * distance
        self.env_origins[:, 2] = 0.268

    def _compute_base_init_state(self, env_origins: torch.Tensor):
        """Computes desired init state for CoM (position and velocity)."""
        init_state_list = (
            [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0]
        )
        init_states = np.stack([init_state_list] * self.num_envs, axis=0)
        init_states = to_torch(init_states, device=self._device)
        init_states[:, :3] = env_origins
        return to_torch(init_states, device=self._device)

    def _load_urdf(self, urdf_path):
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        asset_config = get_asset_config()
        if "go2" in urdf_path:
            asset_config.asset_options.flip_visual_attachments = True
        self._robot_asset = self._gym.load_asset(
            self._sim, asset_root, asset_file, asset_config.asset_options
        )
        self._num_dof = self._gym.get_asset_dof_count(self._robot_asset)
        self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)

        # domain randomization
        self.payload = torch.zeros(
            self._num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.com_displacement = torch.zeros(
            self._num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.domain_rand:
            self.payload = torch_rand_float(
                -1.0, 2.0, (self._num_envs, 1), device=self.device
            )
            self.com_displacement = torch_rand_float(
                -0.05, 0.05, (self._num_envs, 3), device=self.device
            )
        rigid_shape_props_asset = self._gym.get_asset_rigid_shape_properties(
            self._robot_asset
        )
        friction_coeffs = torch.ones(
            self._num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        restituion_coeffs = torch.zeros(
            self._num_envs,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.domain_rand:
            friction_coeffs = torch_rand_float(
                0.2, 1.25, (self._num_envs, 1), device=self.device
            )
            restituion_coeffs = torch_rand_float(
                0.0, 0.5, (self._num_envs, 1), device=self.device
            )
            # for idx in range(len(rigid_shape_props_asset)):
            #     rigid_shape_props_asset[idx].friction = friction_coeffs[idx]
            #     rigid_shape_props_asset[idx].restitution = restituion_coeffs[idx]

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        for i in range(self._num_envs):
            env_handle = self._gym.create_env(
                self._sim, env_lower, env_upper, int(np.sqrt(self._num_envs))
            )
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(*self._base_init_state[i, :3])

            # for s in range(len(rigid_shape_props_asset)):
            #     rigid_shape_props_asset[s].friction = friction_coeffs[i]
            #     rigid_shape_props_asset[s].restitution = restituion_coeffs[i]
            # self._gym.set_asset_rigid_shape_properties(
            #     self._robot_asset, rigid_shape_props_asset
            # )
            actor_handle = self._gym.create_actor(
                env_handle,
                self._robot_asset,
                start_pose,
                "actor",
                i,
                asset_config.self_collisions,
                0,
            )

            # randomize
            body_props = self._gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )

            body_props[0].mass += self.payload[i, 0]
            body_props[0].com += gymapi.Vec3(
                self.com_displacement[i, 0],
                self.com_displacement[i, 1],
                self.com_displacement[i, 2],
            )
            self._gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )

            self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self._envs.append(env_handle)
            self._actors.append(actor_handle)

        self._feet_indices = torch.zeros(
            len(self._feet_names),
            dtype=torch.long,
            device=self._device,
            requires_grad=False,
        )
        for i in range(len(self._feet_names)):
            self._feet_indices[i] = self._gym.find_actor_rigid_body_handle(
                self._envs[0], self._actors[0], self._feet_names[i]
            )

        self._calf_indices = torch.zeros(
            len(self._calf_names),
            dtype=torch.long,
            device=self._device,
            requires_grad=False,
        )
        for i in range(len(self._calf_names)):
            self._calf_indices[i] = self._gym.find_actor_rigid_body_handle(
                self._envs[0], self._actors[0], self._calf_names[i]
            )

        self._thigh_indices = torch.zeros(
            len(self._thigh_names),
            dtype=torch.long,
            device=self._device,
            requires_grad=False,
        )
        for i in range(len(self._thigh_names)):
            self._thigh_indices[i] = self._gym.find_actor_rigid_body_handle(
                self._envs[0], self._actors[0], self._thigh_names[i]
            )

        self._body_indices = torch.zeros(
            self._num_bodies
            - len(self._feet_names)
            - len(self._thigh_names)
            - len(self._calf_names),
            dtype=torch.long,
            device=self._device,
        )
        ic(
            self._num_bodies,
            len(self._feet_names),
            len(self._thigh_names),
            len(self._calf_names),
        )
        all_body_names = self._gym.get_actor_rigid_body_names(self._envs[0], 0)
        ic(all_body_names)
        self._body_names = []
        limb_names = self._thigh_names + self._calf_names + self._feet_names
        idx = 0
        for name in all_body_names:
            if name not in limb_names:
                self._body_indices[idx] = self._gym.find_actor_rigid_body_handle(
                    self._envs[0], self._actors[0], name
                )
                idx += 1
                self._body_names.append(name)

    def set_foot_friction(self, friction_coef, env_id=0):
        rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
            self._envs[env_id], self._actors[env_id]
        )
        for idx in range(len(rigid_shape_props)):
            rigid_shape_props[idx].friction = friction_coef
        self._gym.set_actor_rigid_shape_properties(
            self._envs[env_id], self._actors[env_id], rigid_shape_props
        )
        # import pdb
        # pdb.set_trace()

    def set_foot_frictions(self, friction_coefs, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self._num_envs)
        friction_coefs = friction_coefs * np.ones(self._num_envs)
        for env_id, friction_coef in zip(env_ids, friction_coefs):
            self.set_foot_friction(friction_coef, env_id=env_id)

    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        dof_force = self._gym.acquire_dof_force_tensor(self._sim)
        jacobians = self._gym.acquire_jacobian_tensor(self._sim, "actor")

        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

        # Robot state buffers
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[
            : self._num_envs * self._num_bodies, :
        ]
        self._jacobian = gymtorch.wrap_tensor(jacobians)
        self._motor_positions = self._dof_state.view(self._num_envs, self._num_dof, 2)[
            ..., 0
        ]
        self._motor_velocities = self._dof_state.view(self._num_envs, self._num_dof, 2)[
            ..., 1
        ]
        self._last_motor_velocities = self._motor_velocities.clone()
        self._last_contacts = torch.zeros(
            self._num_envs,
            len(self._feet_indices),
            dtype=torch.bool,
            device=self._device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self._num_envs,
            len(self._feet_indices),
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )

        # base state
        self._base_quat = self._root_states[:, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        self._gravity_vec = torch.stack(
            [to_torch([0.0, 0.0, 1.0], device=self._device)] * self._num_envs
        )

        # foot state
        self._foot_positions = self._rigid_body_state.view(
            self._num_envs, self._num_bodies, 13
        )[:, self._feet_indices, 0:3]
        self._foot_velocities = self._rigid_body_state.view(
            self._num_envs, self._num_bodies, 13
        )[:, self._feet_indices, 7:10]
        self._all_foot_jacobian = torch.zeros(
            (self._num_envs, 12, 12), device=self._device
        )

        # forces
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self._num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis
        self._motor_torques = gymtorch.wrap_tensor(dof_force).view(
            self._num_envs, self._num_dof
        )

        # Other useful buffers
        self._torques = torch.zeros(
            self._num_envs,
            self._num_dof,
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        self.motor_strength_factors = torch.ones(
            self._num_envs,
            12,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.domain_rand:
            self.motor_strength_factors = torch_rand_float(
                0.9,
                1.1,
                (self._num_envs, 12),
                device=self.device,
            )
        self.common_step_counter = 0

    def reset(self):
        self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._time_since_reset[env_ids] = 0

        # Reset root states:
        self._root_states[env_ids] = self._base_init_state[env_ids]
        # Reset root velocity:
        self._root_states[env_ids, 7:13] = torch_rand_float(
            -0.0, 0.0, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel

        self._gym.set_actor_root_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Reset dofs
        self._motor_positions[env_ids] = to_torch(
            self._init_motor_angles, device=self._device, dtype=torch.float
        )
        # randomize initial dofs
        # self._motor_positions[env_ids] *= torch_rand_float(
        #     0.5, 1.5, (len(env_ids), 12), device=self._device
        # )
        self._motor_velocities[env_ids] = 0.0
        self._last_motor_velocities[env_ids] = 0.0

        self._gym.set_dof_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        if len(env_ids) == self._num_envs:
            self._gym.simulate(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._post_physics_step()

    def step(self, action):
        for _ in range(self._sim_config.action_repeat):
            self._torques, _ = self.motor_group.convert_to_torque(
                action, self._motor_positions, self._motor_velocities
            )
            torques = self._torques * self.motor_strength_factors
            self._gym.set_dof_actuation_force_tensor(
                self._sim,
                gymtorch.unwrap_tensor(torques),
            )
            self._gym.simulate(self._sim)
            if self._device == "cpu":
                self._gym.fetch_results(self._sim, True)

            self._last_motor_velocities = self._motor_velocities.clone()
            self._gym.refresh_dof_state_tensor(self._sim)
            self._time_since_reset += self._sim_config.sim_params.dt

        self._post_physics_step()
        self._post_domain_rand()

    def _post_physics_step(self):
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
        self._gym.refresh_jacobian_tensors(self._sim)

        # update robot state
        self._base_quat[:] = self._root_states[:, 3:7]
        self._base_rot_mat = quat_to_rot_mat(self._base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)

        # foot state
        self._foot_velocities = self._rigid_body_state.view(
            self._num_envs, self._num_bodies, 13
        )[:, self._feet_indices, 7:10]
        self._foot_positions = self._rigid_body_state.view(
            self._num_envs, self._num_bodies, 13
        )[:, self._feet_indices, 0:3]

        # update foot jacobian
        self._all_foot_jacobian[:, :3, :3] = torch.bmm(
            self.base_rot_mat_t, self._jacobian[:, 4, :3, 6:9]
        )
        self._all_foot_jacobian[:, 3:6, 3:6] = torch.bmm(
            self.base_rot_mat_t, self._jacobian[:, 8, :3, 9:12]
        )
        self._all_foot_jacobian[:, 6:9, 6:9] = torch.bmm(
            self.base_rot_mat_t, self._jacobian[:, 12, :3, 12:15]
        )
        self._all_foot_jacobian[:, 9:12, 9:12] = torch.bmm(
            self.base_rot_mat_t, self._jacobian[:, 16, :3, 15:18]
        )

    def _post_domain_rand(self):
        self.common_step_counter += 1
        push_interval_t = 15  # s
        disturbance_interval_t = 8  # s
        push_interval = np.ceil(push_interval_t / self._sim_config.dt)
        disturbance_interval = np.ceil(disturbance_interval_t / self._sim_config.dt)

        if self.domain_rand:
            if self.common_step_counter % push_interval == 0:
                max_vel = 1.0
                self._root_states[:, 7:9] = torch_rand_float(
                    -max_vel, max_vel, (self._num_envs, 2), device=self.device
                )  # lin vel x/y
                self._gym.set_actor_root_state_tensor(
                    self._sim, gymtorch.unwrap_tensor(self._root_states)
                )
            if self.common_step_counter % disturbance_interval == 0:
                injected_disturbance = torch_rand_float(
                    -30.0, 30.0, (self._num_envs, 3), device=self.device
                )
                disturbance = torch.zeros(
                    self._num_envs,
                    self._num_bodies,
                    3,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                disturbance[:, 0, :] = injected_disturbance
                self._gym.apply_rigid_body_force_tensors(
                    self._sim,
                    forceTensor=gymtorch.unwrap_tensor(disturbance),
                    space=gymapi.CoordinateSpace.LOCAL_SPACE,
                )

    # for visualization
    def subscribe_keyboard_event(self):
        # subscribe keyboard events
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ENTER, "lock viewer to robot"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_UP, "lock viewer to last robot"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_DOWN, "lock viewer to next robot"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_R, "reset the environment"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_A, "print abs_action mean"
        )
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_B, "print abs_obs mean"
        )

    def check_keyboard_event(self, action, value):
        if action == "QUIT" and value > 0:
            sys.exit()
        elif action == "toggle_viewer_sync" and value > 0:
            self.enable_viewer_sync = not self.enable_viewer_sync
        elif action == "lock viewer to robot" and value > 0:
            self.lock_viewer_to_robot = (self.lock_viewer_to_robot + 1) % 3
        elif action == "lock viewer to next robot" and value > 0:
            if self.follow_robot_index >= 0:
                self.follow_robot_index += 1
                if self.follow_robot_index >= self._num_envs:
                    self.follow_robot_index = 0
        elif action == "lock viewer to last robot" and value > 0:
            if self.follow_robot_index >= 0:
                self.follow_robot_index -= 1
                if self.follow_robot_index < 0:
                    self.follow_robot_index = self._num_envs - 1
        elif action == "print abs_action mean" and value > 0:
            print(f"abs_action mean: {torch.mean(torch.abs(self._action), dim=0)}")
        elif action == "print abs_obs mean" and value > 0:
            print(f"abs_obs mean: {torch.mean(torch.abs(self._obs), dim=0)}")

    def _viewer_follow(self):
        """Callback called before rendering the scene
        Default behaviour: Follow robot
        """
        if self.lock_viewer_to_robot == 0:
            return
        distance = 0
        if self.lock_viewer_to_robot == 1:
            distance = quat_apply(
                self.base_orientation_quat[
                    self.follow_robot_index : (self.follow_robot_index + 1), :
                ],
                torch.tensor([-3.5, 0, 1.4], device=self.device, requires_grad=False),
            )
            distance[2] = 1.5
        elif self.lock_viewer_to_robot == 2:
            distance = quat_apply(
                self.base_orientation_quat[
                    self.follow_robot_index : (self.follow_robot_index + 1), :
                ],
                torch.tensor([0, -3.5, 1.4], device=self.device, requires_grad=False),
            )
            distance[2] = 0.8
        pos = self.base_position_world[self.follow_robot_index, :] + distance
        lookat = self.base_position_world[self.follow_robot_index, :]
        cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time=True):
        if self._viewer:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

            for evt in self._gym.query_viewer_action_events(self._viewer):
                self.check_keyboard_event(evt.action, evt.value)

            self._viewer_follow()

            if self._device != "cpu":
                self._gym.fetch_results(self._sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
                if sync_frame_time:
                    self._gym.sync_frame_time(self._sim)

    # ik
    def get_motor_angles_from_foot_positions(self, foot_local_positions: torch.Tensor):
        raise NotImplementedError()

    @property
    def init_motor_angles(self):
        return self._init_motor_angles

    # ----------------- base properties -----------------
    @property
    def base_position_world(self):
        return self._root_states[:, :3]

    @property
    def base_orientation_rpy(self):
        return angle_normalize(get_euler_xyz_from_quaternion(self._root_states[:, 3:7]))

    @property
    def base_orientation_quat(self):
        return self._root_states[:, 3:7]

    @property
    def projected_gravity(self):
        return torch.bmm(self.base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]

    @property
    def base_rot_mat(self):
        return self._base_rot_mat

    @property
    def base_rot_mat_t(self):
        return self._base_rot_mat_t

    @property
    def base_velocity_world_frame(self):
        return self._root_states[:, 7:10]

    @property
    def base_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._root_states[:, 7:10, None])[
            :, :, 0
        ]

    @property
    def base_angular_velocity_world_frame(self):
        return self._root_states[:, 10:13]

    @property
    def base_angular_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._root_states[:, 10:13, None])[
            :, :, 0
        ]

    # ----------------- motor Properties -----------------
    @property
    def motor_positions(self):
        return torch.clone(self._motor_positions)

    @property
    def motor_velocities(self):
        return torch.clone(self._motor_velocities)

    @property
    def motor_torques(self):
        return torch.clone(self._torques)

    # ----------------- foot Properties -----------------
    @property
    def foot_height(self):
        return self._foot_positions[:, :, 2]

    @property
    def foot_positions_in_base_frame(self):
        foot_positions_world_frame = self._foot_positions
        base_position_world_frame = self._root_states[:, :3]
        # num_env x 4 x 3
        foot_position = (
            foot_positions_world_frame - base_position_world_frame[:, None, :]
        )
        return torch.matmul(
            self._base_rot_mat_t, foot_position.transpose(1, 2)
        ).transpose(1, 2)

    @property
    def foot_positions_in_world_frame(self):
        return torch.clone(self._foot_positions)

    @property
    def foot_velocities_in_base_frame(self):
        foot_vels = torch.bmm(
            self.all_foot_jacobian, self.motor_velocities[:, :, None]
        ).squeeze()
        return foot_vels.reshape((self._num_envs, 4, 3))

    @property
    def foot_velocities_in_world_frame(self):
        return self._foot_velocities

    @property
    def foot_contacts(self):
        return self._contact_forces[:, self._feet_indices, 2] > 1.0

    @property
    def foot_contact_forces(self):
        return self._contact_forces[:, self._feet_indices, :]

    # ----------------- other Properties -----------------
    @property
    def calf_contacts(self):
        return self._contact_forces[:, self._calf_indices, 2] > 1.0

    @property
    def calf_contact_forces(self):
        return self._contact_forces[:, self._calf_indices, :]

    @property
    def thigh_contacts(self):
        return self._contact_forces[:, self._thigh_indices, 2] > 1.0

    @property
    def thigh_contact_forces(self):
        return self._contact_forces[:, self._thigh_indices, :]

    @property
    def has_body_contact(self):
        return torch.any(
            torch.norm(self._contact_forces[:, self._body_indices, :], dim=-1) > 1.0,
            dim=1,
        )

    # ----------------- kinematics Properties -----------------
    @property
    def hip_positions_in_body_frame(self):
        raise NotImplementedError()

    @property
    def all_foot_jacobian(self):
        return self._all_foot_jacobian

    # ----------------- common Properties -----------------
    @property
    def motor_group(self):
        return self._motors

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_dof(self):
        return self._num_dof

    @property
    def device(self):
        return self._device

    @property
    def time_since_reset(self):
        return self._gym.get_sim_time(self._sim)

    @property
    def control_timestep(self):
        return self._sim_config.dt * self._sim_config.action_repeat
