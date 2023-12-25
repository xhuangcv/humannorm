import bisect
import math
import random
import numpy as np
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 1024
    eval_width: int = 1024
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    random_test: bool = False
    sampling_type: str = "head_only"
    # output path for testing view and camera parameters
    test_save_path: str = "./.threestudio_cache"


class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        self.fullbody_part_ratio = [0.7, 0.1, 0.1, 0.1]

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)
        if global_step < 10000:
            self.fullbody_part_ratio = [0.7, 0.1, 0.1, 0.1]
        else:
            self.fullbody_part_ratio = [0.1, 0.3, 0.3, 0.3]

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]
        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )


        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)

        # random scale the focal length by [1.0, 2.0]
        if self.cfg.sampling_type == "head_only":
            human_part = 1
            # get directions by dividing directions_unit_focal by focal length
            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                None, :, :, :
            ].repeat(self.batch_size, 1, 1, 1)
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.width / self.height, 0.1, 1000.0
            )  # FIXME: hard-coded near and far

        elif self.cfg.sampling_type == "half_body":
            human_part = np.random.choice([1, 2], p=[0.3, 0.7])
            if human_part == 1: # head
                focal_scale = 1.5
                focal_scale = torch.full_like(focal_length, focal_scale)
                focal_length *= focal_scale
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)
                center[:,1] += 0.3
            elif human_part == 2: # upper body
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)

            intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.batch_size, 1, 1)
            intrinsic[:, 0, 0] = focal_length
            intrinsic[:, 1, 1] = focal_length
            intrinsic[:, 0, 2] = cx
            intrinsic[:, 1, 2] = cy

            proj_mtx = []
            directions = []
            for i in range(self.batch_size):
                proj = convert_proj(intrinsic[i], self.cfg.height, self.cfg.width, 0.1, 1000.0)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                proj_mtx.append(proj)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.cfg.height,
                    self.cfg.width,
                    (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
                    (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
                    use_pixel_centers=False,
                )
                directions.append(direction)

            proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
            directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)
        elif self.cfg.sampling_type == "full_body":
            human_part = np.random.choice([0, 1, 2, 3], p=self.fullbody_part_ratio)
            if human_part == 0: # full
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)
            elif human_part == 1: # head
                focal_scale = 2
                focal_scale = torch.full_like(focal_length, focal_scale)
                focal_length *= focal_scale
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)
                center[:,2] += 0.6
            elif human_part == 2: # upper body
                focal_scale = 2.0
                focal_scale = torch.full_like(focal_length, focal_scale)
                focal_length *= focal_scale
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)
                center[:,2] += 0.3
            elif human_part == 3: # lower body
                focal_scale = 2.0
                focal_scale = torch.full_like(focal_length, focal_scale)
                focal_length *= focal_scale
                cx = torch.full_like(focal_length, self.cfg.width / 2)
                cy = torch.full_like(focal_length, self.cfg.height / 2)
                center[:,2] -= 0.5

            intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.batch_size, 1, 1)
            intrinsic[:, 0, 0] = focal_length
            intrinsic[:, 1, 1] = focal_length
            intrinsic[:, 0, 2] = cx
            intrinsic[:, 1, 2] = cy

            proj_mtx = []
            directions = []
            for i in range(self.batch_size):
                proj = convert_proj(intrinsic[i], self.cfg.height, self.cfg.width, 0.1, 1000.0)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                proj_mtx.append(proj)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.cfg.height,
                    self.cfg.width,
                    (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
                    (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
                    use_pixel_centers=False,
                )
                directions.append(direction)

            proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
            directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        human_part_out: Int[Tensor, "B"] = torch.tensor(human_part, dtype=torch.long)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "focal": focal_length,
            "human_part": human_part_out,
        }


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        if self.cfg.random_test:
            if self.cfg.sampling_type == "head_only":
                n_views_part = self.n_views
            elif self.cfg.sampling_type == "half_body":
                assert self.n_views % 2 == 0
                n_views_part = self.n_views // 2
            elif self.cfg.sampling_type == "full_body":
                assert self.n_views % 4 == 0
                n_views_part = self.n_views // 4

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            if self.cfg.random_test:
                # azimuth_deg = torch.rand(self.n_views) * 360.0
                azimuth_deg = torch.linspace(0, 360.0, n_views_part)
                azimuth_deg = azimuth_deg.repeat(self.n_views//n_views_part)
            else:
                azimuth_deg = torch.linspace(0, 360.0, self.n_views)

        if self.split == "test" and self.cfg.random_test:
            elevation_deg: Float[Tensor, "B"] = (
                torch.rand(self.n_views)
                * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0])
                + self.cfg.elevation_range[0]
            )

            camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.n_views)
                * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0])
                + self.cfg.camera_distance_range[0]
            )

        else:     
            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.cfg.eval_elevation_deg
            )
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_camera_distance
            )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )

        # random scale the focal length by [1.0, 2.0]
        if self.split == "test" and self.cfg.random_test:
            if self.cfg.sampling_type == "head_only":
                cx = torch.full_like(focal_length, self.cfg.eval_width / 2)
                cy = torch.full_like(focal_length, self.cfg.eval_height / 2)

            elif self.cfg.sampling_type == "half_body":
                focal_scale_p1 = torch.full([n_views_part], 1.5)
                focal_scale_p2 = torch.full([n_views_part], 1.0)
                focal_scale = torch.cat([focal_scale_p1, focal_scale_p2], 0)
                focal_length *= focal_scale

                cx = torch.full_like(focal_length, self.cfg.eval_width / 2)
                cy = torch.full_like(focal_length, self.cfg.eval_height / 2)

                center[0*n_views_part:1*n_views_part, 2] += 0.3
                center[1*n_views_part:2*n_views_part, 2] += 0.0

            elif self.cfg.sampling_type == "full_body":
                focal_scale_p1 = torch.full([n_views_part], 1.0)
                focal_scale_p2 = torch.full([n_views_part], 4.0)
                focal_scale_p3 = torch.full([n_views_part], 2.0)
                focal_scale_p4 = torch.full([n_views_part], 2.0)
                focal_scale = torch.cat([focal_scale_p1, focal_scale_p2, focal_scale_p3, focal_scale_p4], 0)
                focal_length *= focal_scale

                cx = torch.full_like(focal_length, self.cfg.eval_width / 2)
                cy = torch.full_like(focal_length, self.cfg.eval_height / 2)

                center[0*n_views_part:1*n_views_part, 2] += 0
                center[1*n_views_part:2*n_views_part, 2] += 0.6
                center[2*n_views_part:3*n_views_part, 2] += 0.3
                center[3*n_views_part:4*n_views_part, 2] -= 0.7

            intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(self.n_views, 1, 1)
            intrinsic[:, 0, 0] = focal_length
            intrinsic[:, 1, 1] = focal_length
            intrinsic[:, 0, 2] = cx
            intrinsic[:, 1, 2] = cy

            proj_mtx = []
            directions = []
            for i in range(self.n_views):
                proj = convert_proj(intrinsic[i], self.cfg.eval_height, self.cfg.eval_width, 0.1, 1000.0)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                proj_mtx.append(proj)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.cfg.eval_height,
                    self.cfg.eval_width,
                    (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
                    (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
                    use_pixel_centers=False,
                )
                directions.append(direction)

            proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
            directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)
            

        else:
            cx: Float[Tensor, "B"] = torch.full_like(focal_length, self.cfg.eval_width / 2)
            cy: Float[Tensor, "B"] = torch.full_like(focal_length, self.cfg.eval_height / 2)
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )
            directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
                None, :, :, :
            ].repeat(self.n_views, 1, 1, 1)
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            )  # FIXME: hard-coded near and far

            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "focal": self.focal_length[index],
            "cx": self.cx[index],
            "cy": self.cy[index],
            "n_views": self.n_views,
            "test_save_path": self.cfg.test_save_path,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
