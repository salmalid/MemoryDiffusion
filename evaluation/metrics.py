"""Standard personalization metrics (same protocol as the DreamBooth paper).

  CLIP-I : mean cosine between generated and reference images in CLIP space
           -> identity, semantic level.
  DINO   : same, in DINOv2 space -> identity, fine-grained (stricter).
  CLIP-T : mean cosine between each generated image and its prompt
           -> prompt adherence / editability.

All inputs are PIL images; the encoders come from models.encoders.VisionEncoders.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

from models.encoders import VisionEncoders


def _mean_cross_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of all pairwise cosines between rows of `a` and rows of `b`
    (rows are already L2-normalized by the encoders)."""
    return float((a @ b.T).mean())


def clip_i(encoders: VisionEncoders, generated: List[Image.Image],
           references: List[Image.Image]) -> float:
    gen = encoders.encode_images(generated)["clip"]
    ref = encoders.encode_images(references)["clip"]
    return _mean_cross_cosine(gen, ref)


def dino_i(encoders: VisionEncoders, generated: List[Image.Image],
           references: List[Image.Image]) -> float:
    gen = encoders.encode_images(generated)["dino"]
    ref = encoders.encode_images(references)["dino"]
    return _mean_cross_cosine(gen, ref)


def clip_t(encoders: VisionEncoders, generated: List[Image.Image],
           prompts: List[str]) -> float:
    if len(generated) != len(prompts):
        raise ValueError("Need one prompt per generated image")
    img_emb = encoders.encode_images(generated)["clip"]
    txt_emb = np.stack([encoders.encode_text(p) for p in prompts])
    return float((img_emb * txt_emb).sum(axis=1).mean())
