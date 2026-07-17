"""IP-Adapter conditioning: turn retrieved memory embeddings into the tensor
format the diffusers pipeline expects.

diffusers contract for `ip_adapter_image_embeds` (one entry per loaded adapter):
  - without classifier-free guidance: tensor of shape [batch, num_images, dim]
  - with CFG: negative embeds concatenated FIRST -> [2 * batch, num_images, dim]

We generate with batch=1, so shapes are [1, k, 1024] / [2, k, 1024], where k is
the number of memory embeddings injected (1 after mean/softmax aggregation,
top-k after "concat" aggregation). dim=1024 is the OpenCLIP ViT-H image-embed
size used by ip-adapter_sd15.
"""
from __future__ import annotations

import numpy as np
import torch


def build_ip_adapter_embeds(
    cond_embeds: np.ndarray,
    do_classifier_free_guidance: bool,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pack [k, dim] (or [dim]) memory embeddings into the pipeline format.

    The negative branch is the zero embedding — the same convention diffusers
    itself uses when preparing IP-Adapter embeds internally.
    """
    arr = np.atleast_2d(np.asarray(cond_embeds, dtype=np.float32))  # [k, dim]
    pos = torch.from_numpy(arr).to(device=device, dtype=dtype).unsqueeze(0)  # [1, k, dim]

    if do_classifier_free_guidance:
        neg = torch.zeros_like(pos)
        return torch.cat([neg, pos], dim=0)  # [2, k, dim], negative first
    return pos
