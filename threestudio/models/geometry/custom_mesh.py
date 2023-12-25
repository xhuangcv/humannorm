import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *


@threestudio.register("custom-mesh")
class CustomMesh(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: str = ""
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        use_3d_noise: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )
        self.pos_encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.pos_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )

        if self.cfg.use_3d_noise:
            self.noise_cube = torch.randn([1,4,128,128,128]).to('cuda')
            self.noise_map = torch.randn([1,512,512, 4]).to('cuda')

        # Initialize custom mesh
        if self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            v_pos = torch.tensor(mesh.vertices, dtype=torch.float32)
            v_pos = v_pos / 0.9
            v_pos = v_pos.to(self.device)
            t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)
            self.mesh = Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
            self.register_buffer(
                "v_buffer",
                v_pos,
            )
            self.register_buffer(
                "t_buffer",
                t_pos_idx,
            )

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )
        print(self.mesh.v_pos.device)
        self.delta = None

    def isosurface(self) -> Mesh:
        points = contract_to_unisphere(self.v_buffer, self.bbox)  # points normalized to (0, 1)
        enc = self.pos_encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.pos_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        self.delta = features
        # print(self.delta.shape)
        self.mesh = Mesh(v_pos=self.v_buffer + self.delta, t_pos_idx=self.t_buffer)
        return self.mesh
        if hasattr(self, "mesh"):
            return self.mesh
        elif hasattr(self, "v_buffer"):
            self.mesh = Mesh(v_pos=self.v_buffer, t_pos_idx=self.t_buffer)
            return self.mesh
        else:
            raise ValueError(f"custom mesh is not initialized")

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, selector = None
    ) -> Dict[str, Float[Tensor, "..."]]:
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out = {"features": features, "noise_map": None}

        if self.cfg.use_3d_noise:
            # get noise map
            points_unscaled = points_unscaled / self.bbox.max() # covert to [-1, 1]
            points_unscaled = points_unscaled.view(1,1,1,points_unscaled.shape[0],points_unscaled.shape[1])
            noise_fg = F.grid_sample(self.noise_cube, points_unscaled, mode="nearest").squeeze().permute(1,0)
            noise_map = self.noise_map.clone()

            noise_map[selector] = noise_fg
            noise_map = F.interpolate(noise_map.permute(0,3,1,2), [64, 64], mode="nearest")
            out.update({"noise_map":noise_map})

        return out

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
