"""Generates different terrains for IsaacGym."""

from typing import Sequence

import ml_collections  # type: ignore
import numpy as np
import scipy
import torch

from isaacgym import terrain_utils
from src.utilities.torch_utils import to_torch


class Terrain:
    def __init__(
        self, config: ml_collections.ConfigDict, device: str = "cuda", random_seed=0
    ):
        self._config = config
        self._device = device
        np.random.seed(random_seed)

        # derived
        self.num_sub_terrains = self._config.num_rows * self._config.num_cols
        self.length_per_env_pixels = int(
            self._config.terrain_length / self._config.horizontal_scale
        )
        self.width_per_env_pixels = int(
            self._config.terrain_width / self._config.horizontal_scale
        )

        self.border = int(self._config.border_size / self._config.horizontal_scale)
        self.total_cols = (
            int(self._config.num_cols * self.width_per_env_pixels) + 2 * self.border
        )
        self.total_rows = (
            int(self._config.num_rows * self.length_per_env_pixels) + 2 * self.border
        )
        self.height_samples_numpy = np.zeros(
            (self.total_rows, self.total_cols), dtype=np.int16
        )

        self.proportions = [
            np.sum(self._config.terrain_proportions[: i + 1])
            for i in range(len(self._config.terrain_proportions))
        ]

        self.env_origins_numpy = np.zeros(
            (self._config.num_rows, self._config.num_cols, 3)
        )

        self.curriculum()

    def curriculum(self):
        for j in range(self._config.num_cols):
            for i in range(self._config.num_rows):
                difficulty = i / self._config.num_rows
                choice = j / self._config.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self._config.vertical_scale,
            horizontal_scale=self._config.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31, step_height=step_height, platform_size=3.0
            )
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_samples_numpy[start_x:end_x, start_y:end_y] = (
            terrain.height_field_raw
        )

        env_origin_x = (i + 0.5) * self._config.terrain_length
        env_origin_y = (j + 0.5) * self._config.terrain_width
        x1 = int((self._config.terrain_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self._config.terrain_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self._config.terrain_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self._config.terrain_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins_numpy[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
