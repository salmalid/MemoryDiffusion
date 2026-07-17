"""Stable Diffusion 1.5 + IP-Adapter generation backend.

Everything here is inference-only. The IP-Adapter's own image encoder
(OpenCLIP ViT-H, loaded automatically by `load_ip_adapter`) produces the
*conditioning* embeddings stored in the memory bank — retrieval embeddings
come from models/encoders.py and live in different spaces.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from conditioning.cross_attention import apply_ip_adapter_scale
from conditioning.ip_adapter import build_ip_adapter_embeds


class DiffusionGenerator:
    def __init__(
        self,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "models",
        ip_adapter_weights: str = "ip-adapter_sd15.bin",
        device: str | None = None,
        low_vram: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Also loads the ViT-H image encoder + feature extractor from the repo.
        pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder=ip_adapter_subfolder,
            weight_name=ip_adapter_weights,
        )

        if low_vram and self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        self.pipe = pipe

    # ------------------------------------------------------------------ #
    # Enrollment side: images -> conditioning embeddings                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_condition_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode reference images with the IP-Adapter's image encoder.

        Returns float32 [n, 1024] (un-normalized — the adapter was trained on
        raw image_embeds, so we must NOT L2-normalize these).
        """
        pixels = self.pipe.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixels = pixels.to(device=self._execution_device(), dtype=self.dtype)
        embeds = self.pipe.image_encoder(pixels).image_embeds
        return embeds.float().cpu().numpy()

    def _execution_device(self) -> torch.device:
        try:
            return self.pipe._execution_device  # handles cpu-offload hooks
        except Exception:
            return torch.device(self.device)

    # ------------------------------------------------------------------ #
    # Generation side: prompt + memory embeddings -> image                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        cond_embeds: np.ndarray,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        ip_adapter_scale: float = 0.7,
        scale_preset: str = "uniform",
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        apply_ip_adapter_scale(self.pipe, ip_adapter_scale, scale_preset)

        do_cfg = guidance_scale > 1.0
        embeds = build_ip_adapter_embeds(cond_embeds, do_cfg, self._execution_device(), self.dtype)

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            ip_adapter_image_embeds=[embeds],
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            height=int(height),
            width=int(width),
        )
        return result.images[0]
