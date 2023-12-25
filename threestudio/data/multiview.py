import json
import math
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays, get_projection_matrix
from threestudio.utils.typing import *


def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]


def inter_pose(pose_0, pose_1, ratio):
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


@dataclass
class MultiviewsDataModuleConfig:
    dataroot: str = ""
    train_downsample_resolution: int = 4
    eval_downsample_resolution: int = 4
    train_data_interval: int = 1
    eval_data_interval: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    camera_layout: str = "around"
    camera_distance: float = -1
    eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)
    eval_width: int = 512
    eval_height: int = 512
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.
    eval_elevation_deg: float = 0.
    n_views: int = 120


class MultiviewIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        scale = self.cfg.train_downsample_resolution

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        frames = frames[:: self.cfg.train_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []
        elevation_deg = []
        azimuth_deg = []
        camera_distances = []

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        c2w_list = []
        for frame in tqdm(frames):
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            c2w = extrinsic
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)

        if self.cfg.camera_layout == "around":
            # c2w_list[:, :3, 3] -= torch.mean(c2w_list[:, :3, 3], dim=0).unsqueeze(0)
            pass
        elif self.cfg.camera_layout == "front":
            assert self.cfg.camera_distance > 0
            c2w_list[:, :3, 3] -= torch.mean(c2w_list[:, :3, 3], dim=0).unsqueeze(0)
            z_vector = torch.zeros(c2w_list.shape[0], 3, 1)
            z_vector[:, 2, :] = -1
            rot_z_vector = c2w_list[:, :3, :3] @ z_vector
            rot_z_vector = torch.mean(rot_z_vector, dim=0).unsqueeze(0)
            c2w_list[:, :3, 3] -= rot_z_vector[:, :, 0] * self.cfg.camera_distance
        else:
            raise ValueError(
                f"Unknown camera layout {self.cfg.camera_layout}. Now support only around and front."
            )

        for idx, frame in tqdm(enumerate(frames)):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"] / scale
            intrinsic[1, 1] = frame["fl_y"] / scale
            intrinsic[0, 2] = frame["cx"] / scale
            intrinsic[1, 2] = frame["cy"] / scale

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.frame_w, self.frame_h))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False,
            )

            c2w = c2w_list[idx]
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)

            # load camera positions for view dependent prompts
            elevation_deg.append(torch.tensor(frame['elevation']))
            azimuth_deg.append(torch.tensor(frame['azimuth']))
            camera_distances.append(torch.tensor(frame['camera_distances']))
        threestudio.info("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction, self.frames_c2w, keepdim=True
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

        self.elevation_deg: Float[Tensor, "B 1"] = torch.stack(elevation_deg, dim=0)
        self.azimuth_deg: Float[Tensor, "B 1"] = torch.stack(azimuth_deg, dim=0)
        self.camera_distances: Float[Tensor, "B 1"] = torch.stack(camera_distances, dim=0)
        self.human_part_out: Int[Tensor, "B 1"] = torch.full_like(self.camera_distances, torch.tensor(0, dtype=torch.long))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        index = torch.randint(0, self.n_frames, (1,)).item()
        return {
            "index": index,
            "rays_o": self.rays_o[index : index + 1],
            "rays_d": self.rays_d[index : index + 1],
            "mvp_mtx": self.mvp_mtx[index : index + 1],
            "c2w": self.frames_c2w[index : index + 1],
            "camera_positions": self.frames_position[index : index + 1],
            "light_positions": self.light_positions[index : index + 1],
            "gt_rgb": self.frames_img[index : index + 1],
            "height": self.frame_h,
            "width": self.frame_w,
            "elevation": self.elevation_deg[index : index + 1],
            "azimuth": self.azimuth_deg[index : index + 1],
            "camera_distances": self.camera_distances[index : index + 1],
            "human_part": self.human_part_out[index : index + 1],
        }


class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.eval_batch_size == 1
        scale = self.cfg.eval_downsample_resolution

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        frames = frames[:: self.cfg.eval_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        c2w_list = []
        for frame in tqdm(frames):
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                frame["transform_matrix"], dtype=torch.float32
            )
            c2w = extrinsic
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)

        if self.cfg.camera_layout == "around":
            # c2w_list[:, :3, 3] -= torch.mean(c2w_list[:, :3, 3], dim=0).unsqueeze(0)
            pass
        elif self.cfg.camera_layout == "front":
            assert self.cfg.camera_distance > 0
            c2w_list[:, :3, 3] -= torch.mean(c2w_list[:, :3, 3], dim=0).unsqueeze(0)
            z_vector = torch.zeros(c2w_list.shape[0], 3, 1)
            z_vector[:, 2, :] = -1
            rot_z_vector = c2w_list[:, :3, :3] @ z_vector
            rot_z_vector = torch.mean(rot_z_vector, dim=0).unsqueeze(0)
            c2w_list[:, :3, 3] -= rot_z_vector[:, :, 0] * self.cfg.camera_distance
        else:
            raise ValueError(
                f"Unknown camera layout {self.cfg.camera_layout}. Now support only around and front."
            )

        if not (self.cfg.eval_interpolation is None):
            idx0 = self.cfg.eval_interpolation[0]
            idx1 = self.cfg.eval_interpolation[1]
            eval_nums = self.cfg.eval_interpolation[2]
            frame = frames[idx0]
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"] / scale
            intrinsic[1, 1] = frame["fl_y"] / scale
            intrinsic[0, 2] = frame["cx"] / scale
            intrinsic[1, 2] = frame["cy"] / scale
            for ratio in np.linspace(0, 1, eval_nums):
                img: Float[Tensor, "H W 3"] = torch.zeros(
                    (self.frame_h, self.frame_w, 3)
                )
                frames_img.append(img)
                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.frame_h,
                    self.frame_w,
                    (intrinsic[0, 0], intrinsic[1, 1]),
                    (intrinsic[0, 2], intrinsic[1, 2]),
                    use_pixel_centers=False,
                )

                c2w = torch.FloatTensor(
                    inter_pose(c2w_list[idx0], c2w_list[idx1], ratio)
                )
                camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

                near = 0.1
                far = 1000.0
                proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                frames_proj.append(proj)
                frames_c2w.append(c2w)
                frames_position.append(camera_position)
                frames_direction.append(direction)
        else:
            for idx, frame in tqdm(enumerate(frames)):
                intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
                intrinsic[0, 0] = frame["fl_x"] / scale
                intrinsic[1, 1] = frame["fl_y"] / scale
                intrinsic[0, 2] = frame["cx"] / scale
                intrinsic[1, 2] = frame["cy"] / scale

                frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
                img = cv2.imread(frame_path)[:, :, ::-1].copy()
                img = cv2.resize(img, (self.frame_w, self.frame_h))
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
                frames_img.append(img)

                direction: Float[Tensor, "H W 3"] = get_ray_directions(
                    self.frame_h,
                    self.frame_w,
                    (intrinsic[0, 0], intrinsic[1, 1]),
                    (intrinsic[0, 2], intrinsic[1, 2]),
                    use_pixel_centers=False,
                )

                c2w = c2w_list[idx]
                camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

                near = 0.1
                far = 1000.0
                K = intrinsic
                proj = [
                    [
                        2 * K[0, 0] / self.frame_w,
                        -2 * K[0, 1] / self.frame_w,
                        (self.frame_w - 2 * K[0, 2]) / self.frame_w,
                        0,
                    ],
                    [
                        0,
                        -2 * K[1, 1] / self.frame_h,
                        (self.frame_h - 2 * K[1, 2]) / self.frame_h,
                        0,
                    ],
                    [
                        0,
                        0,
                        (-far - near) / (far - near),
                        -2 * far * near / (far - near),
                    ],
                    [0, 0, -1, 0],
                ]
                proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
                frames_proj.append(proj)
                frames_c2w.append(c2w)
                frames_position.append(camera_position)
                frames_direction.append(direction)
        threestudio.info("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(
            self.frames_direction, self.frames_c2w, keepdim=True
        )
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

    def __len__(self):
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        return batch


class MultiviewTestDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.eval_batch_size == 1

        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        assert camera_dict["camera_model"] == "OPENCV"
        assert len(camera_dict["frames"]) == self.cfg.n_views

        frames = camera_dict["frames"]
        self.frames_img = []
        for idx, frame in tqdm(enumerate(frames)):
            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.cfg.eval_width, self.cfg.eval_height))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            self.frames_img.append(img)

        azimuth_deg = torch.linspace(0, 360.0, self.cfg.n_views)
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

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.cfg.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
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


    def __len__(self):
        return self.cfg.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch
    

@register("multiview-camera-datamodule")
class MultiviewDataModule(pl.LightningDataModule):
    cfg: MultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewTestDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewTestDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
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
