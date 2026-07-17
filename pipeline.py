"""MemoryDiffusion — top-level orchestrator.

    enroll(subject, images)   images -> embeddings -> memory bank      (no training)
    generate(prompt, subject) prompt -> retrieve -> aggregate -> diffuse

Heavy models (CLIP/DINOv2 encoders, SD 1.5 + IP-Adapter) load lazily on first
use, so constructing the pipeline (and inspecting the bank) is instant.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageOps

from models.diffusion import DiffusionGenerator
from models.encoders import VisionEncoders
from retrieval.aggregation import aggregate_condition_embeds
from retrieval.memory_bank import MemoryBank

DEFAULTS = {
    "device": None,
    "low_vram": False,
    "encoders": {
        "clip_model": "openai/clip-vit-large-patch14",
        "dino_model": "facebook/dinov2-base",
    },
    "diffusion": {
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "ip_adapter_repo": "h94/IP-Adapter",
        "ip_adapter_subfolder": "models",
        "ip_adapter_weights": "ip-adapter_sd15.bin",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "negative_prompt": (
            "lowres, bad anatomy, worst quality, low quality, watermark, blurry, deformed"
        ),
    },
    "retrieval": {"top_k": 4, "aggregation": "softmax", "temperature": 0.1},
    "memory": {"store_dir": "memory_store"},
    "generation": {
        "ip_adapter_scale": 0.7,
        "scale_preset": "uniform",
        "height": 512,
        "width": 512,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _prepare(images: List[Image.Image]) -> List[Image.Image]:
    """Normalize inputs: RGB + EXIF orientation fixed."""
    return [ImageOps.exif_transpose(img).convert("RGB") for img in images]


class MemoryDiffusion:
    def __init__(self, config_path: str | Path | None = None) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
        self.cfg = cfg

        self.bank = MemoryBank(cfg["memory"]["store_dir"])
        self._encoders: Optional[VisionEncoders] = None
        self._generator: Optional[DiffusionGenerator] = None

    # ------------------------------------------------------------------ #
    # lazy heavy models                                                   #
    # ------------------------------------------------------------------ #
    @property
    def encoders(self) -> VisionEncoders:
        if self._encoders is None:
            enc = self.cfg["encoders"]
            self._encoders = VisionEncoders(
                clip_model=enc["clip_model"],
                dino_model=enc["dino_model"],
                device=self.cfg["device"],
            )
        return self._encoders

    @property
    def generator(self) -> DiffusionGenerator:
        if self._generator is None:
            dif = self.cfg["diffusion"]
            self._generator = DiffusionGenerator(
                base_model=dif["base_model"],
                ip_adapter_repo=dif["ip_adapter_repo"],
                ip_adapter_subfolder=dif["ip_adapter_subfolder"],
                ip_adapter_weights=dif["ip_adapter_weights"],
                device=self.cfg["device"],
                low_vram=self.cfg["low_vram"],
            )
        return self._generator

    # ------------------------------------------------------------------ #
    # enrollment (one-time, per subject)                                  #
    # ------------------------------------------------------------------ #
    def enroll(self, subject: str, images: List[Image.Image]) -> int:
        """Encode reference images into the memory bank. Returns subject count."""
        images = _prepare(images)
        retrieval_embeds = self.encoders.encode_images(images)
        cond_embeds = self.generator.encode_condition_images(images)
        return self.bank.add(
            subject,
            images,
            clip_embeds=retrieval_embeds["clip"],
            dino_embeds=retrieval_embeds["dino"],
            cond_embeds=cond_embeds,
        )

    # ------------------------------------------------------------------ #
    # generation (per prompt)                                             #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        subject: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        aggregation: Optional[str] = None,
        ip_adapter_scale: Optional[float] = None,
        scale_preset: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[Image.Image, List[dict]]:
        """Generate an image conditioned on retrieved visual memories.

        Returns (image, hits) where hits = [{"index", "record", "score"}]
        for the UI to visualize what the model "remembered".
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if len(self.bank) == 0:
            raise ValueError("Memory bank is empty — enroll a subject first")

        rcfg, dcfg, gcfg = self.cfg["retrieval"], self.cfg["diffusion"], self.cfg["generation"]

        query = self.encoders.encode_text(prompt)
        hits = self.bank.query(
            query,
            space="clip",  # text queries live in CLIP space
            subject=subject,
            top_k=top_k if top_k is not None else rcfg["top_k"],
        )
        if not hits:
            raise ValueError(f"No memories found for subject '{subject}'")

        cond = self.bank.cond_embeds([h["index"] for h in hits])
        scores = np.array([h["score"] for h in hits], dtype=np.float32)
        aggregated = aggregate_condition_embeds(
            cond,
            scores,
            mode=aggregation if aggregation is not None else rcfg["aggregation"],
            temperature=rcfg["temperature"],
        )

        image = self.generator.generate(
            prompt=prompt,
            cond_embeds=aggregated,
            negative_prompt=(
                negative_prompt if negative_prompt is not None else dcfg["negative_prompt"]
            ),
            num_inference_steps=(
                num_inference_steps if num_inference_steps is not None
                else dcfg["num_inference_steps"]
            ),
            guidance_scale=(
                guidance_scale if guidance_scale is not None else dcfg["guidance_scale"]
            ),
            ip_adapter_scale=(
                ip_adapter_scale if ip_adapter_scale is not None else gcfg["ip_adapter_scale"]
            ),
            scale_preset=scale_preset if scale_preset is not None else gcfg["scale_preset"],
            seed=seed,
            height=height if height is not None else gcfg["height"],
            width=width if width is not None else gcfg["width"],
        )
        return image, hits
