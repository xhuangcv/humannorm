import os
from dataclasses import dataclass
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from controlnet_aux import NormalBaeDetector, CannyDetector

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import parse_version, C
from threestudio.utils.typing import *
from threestudio.utils.perceptual import PerceptualLoss


@threestudio.register("sds-du-controlnet-guidance")
class SDSDUControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "SG161222/Realistic_Vision_V2.0"
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        control_type: str = 'normal' # normal/input_normal/canny
        controlnet_name_or_path: str = './pretrained_models/controlnet-normal' # normal/input_normal/canny

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float  = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20
        per_editing_step: int = 10
        use_sds: bool = False

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100
    
    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        if self.cfg.control_type == 'normal' or self.cfg.control_type == 'input_normal':
            controlnet_name_or_path: str = self.cfg.controlnet_name_or_path
        elif self.cfg.control_type == 'canny':
            controlnet_name_or_path: str = "lllyasviel/control_v11p_sd15_canny"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir
        }

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path, 
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=controlnet,
            **pipe_kwargs).to(self.device)

        self.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, 
            torch_dtype=self.weights_dtype, 
            cache_dir=self.cfg.cache_dir)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()

        if self.cfg.control_type == 'normal':
            self.preprocessor = NormalBaeDetector.from_pretrained("/home/xinhuang/.cache/huggingface/hub/models--lllyasviel--Annotators/snapshots/9a7d84251d487d11c4834466779de6b0d2c44486")
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == 'canny':
            self.preprocessor = CannyDetector()
        elif self.cfg.control_type == 'input_normal':
            self.preprocessor = None

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.edited_images = {}
        self.grad_clip_val: Optional[float] = None
        self.condition_scale = self.cfg.condition_scale

        threestudio.info(f"Loaded ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs, 
        down_block_additional_residuals, 
        mid_block_additional_residual
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual
        ).sample.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(t.item() // 25 + 1)
        print(f" Denoising steps: {t.item() // 25}, multi-step denoising ...")
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            threestudio.debug("Start editing...")
            for i, t in enumerate(self.scheduler.timesteps):

                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 2)
                    down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                        latent_model_input, t, 
                        encoder_hidden_states=text_embeddings,
                        image_cond=image_cond,
                        condition_scale=self.condition_scale
                    )

                    noise_pred = self.forward_control_unet(latent_model_input, t, 
                        encoder_hidden_states=text_embeddings, 
                        cross_attention_kwargs=None, 
                        down_block_additional_residuals=down_block_res_samples, 
                        mid_block_additional_residual=mid_block_res_sample)
                # perform classifier-free guidance
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = (
                    noise_pred_uncond 
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
                )
                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents
    
    def prepare_image_cond(
        self, 
        cond_rgb: Float[Tensor, "B H W C"]
    ):
        if self.cfg.control_type == 'normal':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy()*255).astype(np.uint8).copy()
            detected_map = self.preprocessor(cond_rgb)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == 'canny':
            cond_rgb = (cond_rgb[0].detach().cpu().numpy()*255).astype(np.uint8).copy()
            blurred_img = cv2.blur(cond_rgb,ksize=(5,5))
            detected_map = self.preprocessor(blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound)
            control = torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            # control = control.unsqueeze(-1).repeat(1, 1, 3)
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "input_normal":
            control = cond_rgb.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unknown control type: {self.cfg.control_type}")
            
        return F.interpolate(
            control, (512, 512), mode="bilinear", align_corners=False
        )
    
    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"]
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input, t, 
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond,
                condition_scale=self.cfg.condition_scale
            )

            noise_pred = self.forward_control_unet(latent_model_input, t, 
                encoder_hidden_states=text_embeddings, 
                cross_attention_kwargs=None, 
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample)

        # perform classifier-free guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond 
            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        human_part: Int[Tensor, "B"],
        index=None,
        gt_rgb=None,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )

        latents = self.encode_images(rgb_BCHW_512)

        image_cond = self.prepare_image_cond(cond_rgb)

        # temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(elevation, azimuth, camera_distances, human_part, True)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sds: # SDS loss
            grad = self.compute_grad_sds(text_embeddings, latents, image_cond, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
            }
        else: # multi-step SDS loss
            if index in self.edited_images:
                gt_rgb = self.edited_images[index]
                gt_rgb_BCHW = torch.nn.functional.interpolate(
                    gt_rgb, (H, W), mode="bilinear", align_corners=False
                )

            if (self.global_step % self.cfg.per_editing_step == 0) or index not in self.edited_images:
                edit_latents = self.edit_latents(text_embeddings, latents, image_cond, t)
                edit_images = self.decode_latents(edit_latents)
                gt_rgb_BCHW = F.interpolate(edit_images, (H, W), mode='bilinear')
                self.edited_images[index] = gt_rgb_BCHW.detach()

                temp = (gt_rgb_BCHW[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(".threestudio_cache/debug.jpg", temp[:, :, ::-1])

            loss_l1 = torch.nn.functional.l1_loss(rgb_BCHW_512, gt_rgb_BCHW.detach(), reduction='mean') / batch_size
            loss_p = self.perceptual_loss(rgb_BCHW_512, gt_rgb_BCHW.detach()).sum() / batch_size

            return {
                "loss_l1": loss_l1,
                "loss_p": loss_p,
            }


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
        self.global_step = global_step
        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        self.condition_scale=C(self.cfg.condition_scale, epoch, global_step)