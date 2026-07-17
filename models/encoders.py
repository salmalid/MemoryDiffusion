"""Vision encoders for MemoryDiffusion.

Two complementary embedding spaces:
  - CLIP ViT-L/14 : text-aligned semantics -> used to retrieve memories from a *prompt*.
  - DINOv2 (base) : fine-grained visual identity -> used for identity metrics and
                    image-to-image retrieval.

Both are inference-only (no gradients, frozen weights).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize along `axis` (safe for zero vectors)."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class VisionEncoders:
    """Frozen CLIP + DINOv2 feature extractors."""

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        dino_model: str = "facebook/dinov2-base",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)

        self.dino = AutoModel.from_pretrained(dino_model).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """Encode PIL images.

        Returns dict with L2-normalized arrays:
          "clip": [n, 768]  (ViT-L projection space, comparable with encode_text)
          "dino": [n, 768]  (DINOv2 CLS token)
        """
        clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        clip_emb = self.clip.get_image_features(**clip_inputs).float().cpu().numpy()

        dino_inputs = self.dino_processor(images=images, return_tensors="pt").to(self.device)
        dino_emb = self.dino(**dino_inputs).last_hidden_state[:, 0].float().cpu().numpy()

        return {"clip": l2_normalize(clip_emb), "dino": l2_normalize(dino_emb)}

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a prompt into CLIP text space. Returns L2-normalized [768]."""
        inputs = self.clip_processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        emb = self.clip.get_text_features(**inputs).float().cpu().numpy()[0]
        return l2_normalize(emb)
