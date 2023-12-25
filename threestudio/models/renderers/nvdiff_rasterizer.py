from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        normal_type: str = 'world'
        use_sdf_loss: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        if self.cfg.use_sdf_loss:
            mesh, sdf_loss = self.geometry.isosurface()
        else:
            mesh = self.geometry.isosurface()
            sdf_loss = None

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh, 'sdf_loss': sdf_loss}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)

        if self.cfg.normal_type == 'world':
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif self.cfg.normal_type == 'camera':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            bg_normal = torch.zeros_like(gb_normal)
            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif self.cfg.normal_type == 'controlnet':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            # nerf coord to a special coord for controlnet
            gb_normal = torch.cat([-gb_normal[:,:,:,0:1], gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3]], -1)
            bg_normal = torch.zeros_like(gb_normal)
            bg_normal[~mask[..., 0]] = torch.Tensor([[126/255, 107/255, 1.0]]).to(bg_normal)

            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        else:
            raise ValueError(f"Unknown normal type: {self.cfg.normal_type}")

        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # gb_depth = rast[..., 2:3]
        # gb_depth = 1./(gb_depth + 1e-7)
        gb_depth, _ = self.ctx.interpolate_one(v_pos_clip[0,:, :3].contiguous(), rast, mesh.t_pos_idx)
        gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
        max_depth = torch.max(gb_depth[mask[..., 0]])
        min_depth = torch.min(gb_depth[mask[..., 0]])
        gb_depth_aa = torch.lerp(
                torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
            )
        gb_depth_aa = self.ctx.antialias(
            gb_depth_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_depth":gb_depth_aa})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        return out
