"""Persistent multi-subject visual memory bank.

Each memory record = one reference image with three embeddings:
  clip [768]  — retrieval from text prompts (CLIP ViT-L space)
  dino [768]  — fine-grained identity (DINOv2 space)
  cond [1024] — conditioning signal for the diffusion model (IP-Adapter ViT-H)

Storage layout (memory_store/):
  records.json  — subject / image path metadata
  embeds.npz    — the three embedding matrices, row-aligned with records
  images/<subject-slug>/<uuid>.png — the reference images themselves

Search is FAISS inner-product over L2-normalized retrieval embeddings
(== cosine similarity). Text queries against the "fused" space are zero-padded
on the DINO block, which reduces exactly to CLIP similarity (uniform 1/sqrt(2)
scale, ranking unchanged).
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from PIL import Image

_EMB_KEYS = ("clip", "dino", "cond")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "subject"


@dataclass
class MemoryRecord:
    record_id: str
    subject: str
    image_path: str  # relative to store_dir


class MemoryBank:
    def __init__(self, store_dir: str | Path = "memory_store") -> None:
        self.store_dir = Path(store_dir)
        (self.store_dir / "images").mkdir(parents=True, exist_ok=True)
        self.records: List[MemoryRecord] = []
        self._embeds: Dict[str, Optional[np.ndarray]] = {k: None for k in _EMB_KEYS}
        self._load()

    # ------------------------------------------------------------------ #
    # persistence                                                         #
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        rec_file = self.store_dir / "records.json"
        emb_file = self.store_dir / "embeds.npz"
        if rec_file.exists() and emb_file.exists():
            self.records = [MemoryRecord(**r) for r in json.loads(rec_file.read_text("utf-8"))]
            with np.load(emb_file) as data:
                self._embeds = {k: data[k].astype(np.float32) for k in _EMB_KEYS}

    def _save(self) -> None:
        (self.store_dir / "records.json").write_text(
            json.dumps([asdict(r) for r in self.records], indent=2), "utf-8"
        )
        if self.records:
            np.savez(self.store_dir / "embeds.npz", **self._embeds)
        else:
            (self.store_dir / "embeds.npz").unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # basic accessors                                                     #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, record: MemoryRecord) -> Path:
        return self.store_dir / record.image_path

    def subjects(self) -> Dict[str, List[MemoryRecord]]:
        """Subject -> its records, in insertion order."""
        grouped: Dict[str, List[MemoryRecord]] = {}
        for rec in self.records:
            grouped.setdefault(rec.subject, []).append(rec)
        return grouped

    def cond_embeds(self, indices: Sequence[int]) -> np.ndarray:
        return self._embeds["cond"][list(indices)]

    # ------------------------------------------------------------------ #
    # mutation                                                            #
    # ------------------------------------------------------------------ #
    def add(
        self,
        subject: str,
        images: List[Image.Image],
        clip_embeds: np.ndarray,
        dino_embeds: np.ndarray,
        cond_embeds: np.ndarray,
    ) -> int:
        """Add reference images for `subject`. Returns the subject's new count."""
        subject = subject.strip()
        if not subject:
            raise ValueError("Subject name must not be empty")
        n = len(images)
        for name, arr in (("clip", clip_embeds), ("dino", dino_embeds), ("cond", cond_embeds)):
            if arr.shape[0] != n:
                raise ValueError(f"{name} embeddings ({arr.shape[0]}) != images ({n})")

        subject_dir = self.store_dir / "images" / _slugify(subject)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            rel = Path("images") / subject_dir.name / f"{uuid.uuid4().hex[:12]}.png"
            img.save(self.store_dir / rel)
            self.records.append(
                MemoryRecord(record_id=rel.stem, subject=subject, image_path=str(rel))
            )

        new = {"clip": clip_embeds, "dino": dino_embeds, "cond": cond_embeds}
        for key in _EMB_KEYS:
            block = np.asarray(new[key], dtype=np.float32)
            old = self._embeds[key]
            self._embeds[key] = block if old is None else np.concatenate([old, block], axis=0)

        self._save()
        return len(self.subjects()[subject])

    def remove_subject(self, subject: str) -> int:
        """Delete a subject's memories (records, embeddings, image files)."""
        keep = np.array([rec.subject != subject for rec in self.records], dtype=bool)
        removed = int((~keep).sum())
        if removed == 0:
            return 0

        self.records = [rec for rec, k in zip(self.records, keep) if k]
        for key in _EMB_KEYS:
            arr = self._embeds[key]
            if arr is not None:
                self._embeds[key] = arr[keep] if keep.any() else None
        shutil.rmtree(self.store_dir / "images" / _slugify(subject), ignore_errors=True)
        self._save()
        return removed

    # ------------------------------------------------------------------ #
    # retrieval                                                           #
    # ------------------------------------------------------------------ #
    def _retrieval_matrix(self, space: str, indices: Sequence[int]) -> np.ndarray:
        if space in ("clip", "dino"):
            return self._embeds[space][list(indices)]
        if space == "fused":
            clip = self._embeds["clip"][list(indices)]
            dino = self._embeds["dino"][list(indices)]
            return np.concatenate([clip, dino], axis=1) / np.sqrt(2.0)
        raise ValueError(f"Unknown retrieval space '{space}' (clip|dino|fused)")

    def query(
        self,
        query_vec: np.ndarray,
        space: str = "clip",
        subject: Optional[str] = None,
        top_k: int = 4,
    ) -> List[dict]:
        """Top-k memories by cosine similarity.

        Returns [{"index", "record", "score"}] sorted by score, filtered to
        `subject` when given.
        """
        candidates = [
            i for i, rec in enumerate(self.records)
            if subject is None or rec.subject == subject
        ]
        if not candidates:
            return []

        matrix = np.ascontiguousarray(self._retrieval_matrix(space, candidates), dtype=np.float32)
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query.shape[0] < matrix.shape[1]:  # text query vs fused space
            query = np.concatenate(
                [query, np.zeros(matrix.shape[1] - query.shape[0], dtype=np.float32)]
            )

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        k = min(int(top_k), len(candidates))
        scores, local_idx = index.search(query[None, :], k)

        return [
            {"index": candidates[j], "record": self.records[candidates[j]], "score": float(s)}
            for s, j in zip(scores[0], local_idx[0])
            if j >= 0
        ]
