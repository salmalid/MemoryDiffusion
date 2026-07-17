"""Fuse the top-k retrieved memory embeddings into the conditioning signal.

Strategies (an ablation axis of the project — see README RQ3):
  mean    : average of the k embeddings (robust, blurs details).
  softmax : retrieval-score-weighted average (prompt-relevant memories dominate).
  best    : single highest-scoring memory (sharp, may lose identity coverage).
  concat  : keep all k embeddings as separate image tokens (richest, slowest).
"""
from __future__ import annotations

import numpy as np


def aggregate_condition_embeds(
    cond_embeds: np.ndarray,
    scores: np.ndarray,
    mode: str = "softmax",
    temperature: float = 0.1,
) -> np.ndarray:
    """cond_embeds: [k, dim] conditioning embeddings of the retrieved memories.
    scores: [k] retrieval similarities (any monotonic scale).
    Returns [m, dim] with m=1 (mean/softmax/best) or m=k (concat).
    """
    cond = np.atleast_2d(np.asarray(cond_embeds, dtype=np.float32))
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if cond.shape[0] != scores.shape[0]:
        raise ValueError(f"{cond.shape[0]} embeddings but {scores.shape[0]} scores")

    if mode == "mean":
        return cond.mean(axis=0, keepdims=True)

    if mode == "softmax":
        logits = scores / max(temperature, 1e-6)
        logits -= logits.max()  # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum()
        return (weights[:, None] * cond).sum(axis=0, keepdims=True)

    if mode == "best":
        return cond[int(scores.argmax())][None, :]

    if mode == "concat":
        return cond

    raise ValueError(f"Unknown aggregation mode '{mode}' (mean|softmax|best|concat)")
