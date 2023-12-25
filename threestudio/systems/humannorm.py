from dataclasses import dataclass, field

import os
import json
import torch
import torch.nn.functional as F
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.misc import cleanup, get_device

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import imageio
import numpy as np


@threestudio.register("humannorm-system")
class HumanNorm(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture: bool = False
        start_sdf_loss_step: int = -1

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.frames = []
        self.transforms = {
                "camera_model": "OPENCV",
                "orientation_override": "none",
            }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # addtional prompt processor and guiance such as depth stable diffusion model
        self.prompt_processor_add = None
        self.guidance_add = None
        if len(self.cfg.prompt_processor_type_add) > 0:
            self.prompt_processor_add = threestudio.find(self.cfg.prompt_processor_type_add)(
                self.cfg.prompt_processor_add
            )

        if len(self.cfg.guidance_type_add) > 0:
            self.guidance_add = threestudio.find(self.cfg.guidance_type_add)(self.cfg.guidance_add)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()

        if self.true_global_step == self.cfg.start_sdf_loss_step:
            np.save('.threestudio_cache/mesh_v_pos.npy', out['mesh'].v_pos.detach().cpu().numpy())
            np.save('.threestudio_cache/mesh_t_pos_idx.npy', out['mesh'].t_pos_idx.detach().cpu().numpy())

        if not self.cfg.texture:  # geometry training

            # normal SDS loss
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch
            )

            # depth SDS loss
            if self.prompt_processor_add is not None:
                prompt_utils = self.prompt_processor_add()

            if self.guidance_add is not None and self.C(self.cfg.loss.lambda_sds_add) > 0:
                guidance_inp = out["comp_depth"].repeat(1,1,1,3)
                guidance_out_add = self.guidance_add(
                    guidance_inp, prompt_utils, **batch
                )
                guidance_out.update({"loss_sds_add":guidance_out_add["loss_sds"]})
            else:
                guidance_out.update({"loss_sds_add":0})

            # SDF loss
            if out['sdf_loss'] is not None:
                guidance_out.update({"loss_sdf": out['sdf_loss']})

            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
                # print('loss_lap', loss_laplacian_smoothness * self.C(self.cfg.loss.lambda_laplacian_smoothness)) # for debugging

        else:  # texture training
            guidance_inp = out["comp_rgb"]
            if isinstance(
                self.guidance,
                (
                    threestudio.models.guidance.controlnet_guidance.ControlNetGuidance,
                    threestudio.models.guidance.controlnet_vsd_guidance.ControlNetVSDGuidance,
                    threestudio.models.guidance.sds_du_controlnet_guidance.SDSDUControlNetGuidance,
                ),
            ):
                cond_inp = out["comp_normal"] # conditon for controlnet
                guidance_out = self.guidance(
                    guidance_inp, cond_inp, prompt_utils, **batch
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch
                )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                # print(name, value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])) # for debugging

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-val-color/{batch['index'][0]}.jpg",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if self.cfg.texture
                    else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )

        if not self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-val-normal/{batch['index'][0]}.jpg",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ],
                name="validation_step",
                step=self.true_global_step,
            )


    def on_validation_epoch_end(self):
        if self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-val-color",
                f"it{self.true_global_step}-val-color",
                "(\d+)\.jpg",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
            )
        if not self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-val-normal",
                f"it{self.true_global_step}-val-normal",
                "(\d+)\.jpg",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        out = self(batch)
    
        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-test-color/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )

        self.save_image_grid(
            f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

        # save camera parameters and views in coarse texture stage for multi-step SDS loss in fine texture stage
        if 'focal' in batch and self.cfg.texture:
            # save camera parameters
            c2w = batch['c2w'][0].cpu().numpy()

            down_scale = batch['width'] / 512 # ensure the resolution is set to 512

            frame = {
                    "fl_x": float(batch['focal'][0].cpu()) / down_scale,
                    "fl_y": float(batch['focal'][0].cpu()) / down_scale ,
                    "cx": float(batch['cx'][0].cpu()) / down_scale,
                    "cy": float(batch['cy'][0].cpu()) / down_scale,
                    "w": int(batch['width'].cpu() / down_scale),
                    "h": int(batch['height'].cpu() / down_scale),
                    "file_path": f"./image/{batch['index'][0]}.png",
                    "transform_matrix": c2w.tolist(),
                    "elevation": float(batch['elevation'][0].cpu()),
                    "azimuth": float(batch['azimuth'][0].cpu()),
                    "camera_distances": float(batch['camera_distances'][0].cpu()),
                }
            self.frames.append(frame)

            if batch['index'][0] == (batch['n_views'][0]-1):
                os.makedirs(f"{batch['test_save_path'][0]}", exist_ok=True)
                self.transforms["frames"] = self.frames
                with open(os.path.join(batch['test_save_path'][0], 'transforms.json'), 'w') as f:
                    f.write(json.dumps(self.transforms, indent=4))

                # init
                self.frames.clear()

            save_img = out["comp_rgb"]
            save_img = F.interpolate(save_img.permute(0,3,1,2), (512, 512), mode="bilinear", align_corners=False)
            os.makedirs(f"{batch['test_save_path'][0]}/image", exist_ok=True)
            imageio.imwrite(f"{batch['test_save_path'][0]}/image/{batch['index'][0]}.png", (save_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255).astype(np.uint8))


    def on_test_epoch_end(self):
        if self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-color",
                f"it{self.true_global_step}-test-color",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step,
            )

        self.save_img_sequence(
            f"it{self.true_global_step}-test-normal",
            f"it{self.true_global_step}-test-normal",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )