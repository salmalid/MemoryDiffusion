"""Layer-wise control of where the visual memory is injected in the U-Net.

IP-Adapter adds a decoupled image cross-attention branch to every cross-attention
layer. diffusers lets us scale that branch *per layer block*, which gives coarse
control over WHAT the memory transfers:

  - "uniform"          : same strength everywhere (full identity transfer).
  - "style_only"       : only the up-block that carries style/appearance
                         (the InstantStyle observation).
  - "style_and_layout" : style up-block + the down-block that carries layout.

Presets store *relative* weights in [0, 1]; `apply_ip_adapter_scale` multiplies
them by the user's global strength so the UI slider keeps working in every mode.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

SCALE_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "uniform": None,  # plain scalar scale on every layer
    "style_only": {"up": {"block_0": [0.0, 1.0, 0.0]}},
    "style_and_layout": {
        "down": {"block_2": [0.0, 1.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    },
}


def _scale_nested(config: Any, strength: float) -> Any:
    """Recursively multiply every numeric leaf of a preset by `strength`."""
    if isinstance(config, dict):
        return {key: _scale_nested(value, strength) for key, value in config.items()}
    if isinstance(config, list):
        return [_scale_nested(value, strength) for value in config]
    return float(config) * strength


def apply_ip_adapter_scale(pipe, strength: float, preset: str = "uniform") -> None:
    """Apply `strength` (0..1+) through the chosen preset onto the pipeline."""
    if preset not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale preset '{preset}'. Options: {list(SCALE_PRESETS)}")

    template = SCALE_PRESETS[preset]
    if template is None:
        pipe.set_ip_adapter_scale(float(strength))
    else:
        pipe.set_ip_adapter_scale(_scale_nested(copy.deepcopy(template), float(strength)))
