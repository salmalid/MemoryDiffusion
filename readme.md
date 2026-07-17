# MemoryDiffusion

### Personalized Image Generation Without Training

> **Research question:** *Can an external visual memory replace parameter updates (fine-tuning) for diffusion-model personalization?*

MemoryDiffusion is a lightweight, **training-free** personalization system for text-to-image diffusion models. Instead of fine-tuning a model on a subject (DreamBooth, LoRA, Textual Inversion), it builds an external **memory bank** of visual embeddings from a handful of reference images and *retrieves* the relevant visual features at inference time to condition the diffusion process.

The result: give the model a subject (a person, an object, a style) once, and generate it in new contexts — **no gradient steps, no per-subject checkpoints, low VRAM, seconds instead of minutes.**

---

## 1. Motivation

### The problem with current personalization

| Method | Requires training | Per-subject artifact | Typical cost |
|---|---|---|---|
| DreamBooth | Yes (full/partial fine-tune) | New checkpoint (GBs) | 10–30 min, high VRAM |
| LoRA | Yes (adapter training) | LoRA weights (10s–100s MB) | 5–15 min, medium VRAM |
| Textual Inversion | Yes (embedding optimization) | New token embedding | 20–60 min |

All three share the same bottleneck: **they encode the subject into model parameters**, which means:

- Training time per new subject (poor UX, no instant personalization).
- GPU cost and specialized setup.
- Storage that grows linearly with the number of subjects.
- Risk of overfitting / catastrophic forgetting of the base model's priors.

### The MemoryDiffusion alternative

Treat personalization as a **retrieval problem**, not an optimization problem. The subject lives in an external, editable memory store. Adding a new subject = adding vectors to a database. This is instant, composable, and cheap.

**Analogy:** Fine-tuning is like memorizing a fact by rewiring your brain; MemoryDiffusion is like writing it in a notebook you can consult whenever you need it — the same idea that made RAG (Retrieval-Augmented Generation) successful for LLMs, applied to diffusion.

---

## 2. Core Idea

```
                 ┌─────────────────────────────────────────────┐
   ENROLLMENT    │  User reference images (3–10 photos)         │
   (one time)    └───────────────────────┬─────────────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────┐
                           │  Vision Encoder           │
                           │  (CLIP ViT-L / DINOv2)    │
                           └──────────────┬────────────┘
                                          │  embeddings
                                          ▼
                           ┌──────────────────────────┐
                           │  Memory Bank              │
                           │  (FAISS vector index)     │
                           └──────────────┬────────────┘
                                          │
   ─────────────────────────────────────────────────────────────
                                          │
   INFERENCE                              ▼
   (per prompt)     Prompt ──►  ┌──────────────────────────┐
   "the car driving             │  Retriever               │
    through Tokyo    ──────────►│  (top-k relevant vectors)│
    at night"                   └──────────────┬────────────┘
                                          │  retrieved visual memory
                                          ▼
                           ┌──────────────────────────┐
                           │  Conditioning Module      │
                           │  (IP-Adapter / cross-attn)│
                           └──────────────┬────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────┐
                           │  Diffusion Model (SD 1.5) │
                           └──────────────┬────────────┘
                                          │
                                          ▼
                                   Personalized image
```

### Walkthrough example

1. **Enroll:** User uploads 5 photos of a vintage car → encoder produces embeddings → stored in the memory bank.
2. **Prompt:** `"The car driving through Tokyo at night"`.
3. **Retrieve:** System pulls the memory vectors that best capture the car's identity — shape, color, texture, fine details.
4. **Inject:** Those visual features condition the diffusion U-Net (via IP-Adapter-style image prompt or cross-attention injection).
5. **Generate:** SD 1.5 renders the *same car* in a new scene it never saw during enrollment.

---

## 3. Technical Stack

| Component | Choice | Role |
|---|---|---|
| **Vision encoder** | CLIP ViT-L/14 and/or DINOv2 | Turn reference images into embeddings. CLIP = semantic/text-aligned; DINOv2 = fine-grained visual identity. |
| **Memory / retrieval** | FAISS | Store embeddings, fast top-k similarity search. |
| **Diffusion backbone** | Stable Diffusion 1.5 | Base generator (open, lightweight, well-supported). |
| **Conditioning** | IP-Adapter *or* cross-attention injection | Inject retrieved visual memory into the denoising process. |
| **Demo UI** | Gradio | Upload images, type prompt, view results. |
| **Hardware** | ~8 GB VRAM, **inference only** | Runs on modest consumer GPUs. |

### Encoder choice — a design question worth studying

- **CLIP** embeddings align with text, which pairs naturally with prompt conditioning but can be weaker on identity detail.
- **DINOv2** captures fine visual structure/identity better but is not text-aligned.
- A promising direction: **use both** — DINOv2 for identity fidelity, CLIP for prompt/semantic alignment — and fuse them. This is itself an experimental contribution.

### Conditioning — the two candidate mechanisms

1. **IP-Adapter style:** Treat the retrieved embedding as an "image prompt" fed through a decoupled cross-attention adapter. Mature, robust, minimal changes to the base model.
2. **Cross-attention injection:** Inject retrieved features directly into the U-Net's cross-attention layers. More experimental, more control, more risk.

Recommendation: **start with IP-Adapter** (fastest path to a working baseline), then explore custom injection as a research extension.

---

## 4. Research Contributions

The central claim to validate:

> **Can external visual memory replace parameter updates for diffusion personalization?**

Sub-questions that make the study richer and more publishable:

- **RQ1 — Fidelity:** How close does retrieval-based conditioning get to LoRA/DreamBooth on identity preservation?
- **RQ2 — Encoders:** Which encoder (CLIP vs. DINOv2 vs. fused) best preserves subject identity?
- **RQ3 — Retrieval design:** How do the number of reference images, top-k, and aggregation strategy (mean-pool vs. attention-weighted vs. concat) affect quality?
- **RQ4 — Editability trade-off:** Does stronger conditioning improve identity at the cost of prompt-following (the classic identity ↔ editability tension)?
- **RQ5 — Scalability:** Does the approach hold when the memory bank contains *many* subjects (multi-subject retrieval, cross-subject contamination)?

---

## 5. Evaluation Plan

### Baselines to compare against

| Method | Training | VRAM | Identity | Storage/subject | Time/subject |
|---|---|---|---|---|---|
| DreamBooth | Yes | High | Excellent | GBs | ~10–30 min |
| LoRA | Yes | Medium | Good | 10s–100s MB | ~5–15 min |
| Textual Inversion | Yes | Medium | Fair–Good | KBs | ~20–60 min |
| **MemoryDiffusion** | **No** | **Low** | **Good (target)** | **KBs–MBs (vectors)** | **~seconds** |

### Metrics

- **CLIP-I / CLIP-T similarity** — image-image identity similarity, and image-text prompt alignment (measures editability).
- **DINO similarity** — fine-grained identity preservation (a stronger identity signal than CLIP).
- **Human evaluation** — pairwise preference on (a) identity fidelity and (b) prompt adherence.
- **Efficiency metrics** — enrollment time, inference latency, peak VRAM, storage per subject.

### Suggested datasets

- **DreamBooth dataset** (30 subjects, standard for this task) for direct comparability.
- A small custom set (faces / objects / styles) to stress-test generalization.
- Report results per category — identity behaves very differently for faces vs. objects vs. styles.

---

## 6. Project Structure

```
MemoryDiffusion/
├── pipeline.py        # Orchestrator: enroll() / generate() (lazy model loading)
├── models/            # Vision encoders + diffusion backbone wrappers
│   ├── encoders.py        #   CLIP ViT-L + DINOv2 embedding (retrieval spaces)
│   └── diffusion.py       #   SD 1.5 + IP-Adapter wrapper (conditioning space)
├── retrieval/         # Memory bank
│   ├── memory_bank.py     #   persistent multi-subject FAISS bank (3 spaces/record)
│   └── aggregation.py     #   top-k fusion: mean | softmax | best | concat
├── conditioning/      # Injection into the diffusion process
│   ├── ip_adapter.py      #   memory embeds -> ip_adapter_image_embeds format
│   └── cross_attention.py #   layer-wise injection presets (style/layout control)
├── demo/
│   └── gradio_app.py      # Dark demo UI: enroll → prompt → generate + retrieval panel
├── evaluation/
│   ├── metrics.py         #   CLIP-I, CLIP-T, DINO similarity
│   └── run_benchmark.py   #   benchmark harness + results.json + summary chart
├── configs/
│   └── default.yaml       # all knobs (encoder, k, aggregation, scales…)
├── data/              # (gitignored) reference images / datasets
├── requirements.txt
├── README.md
└── paper.md           # research write-up / PFE report draft
```

### Quickstart (Windows / PowerShell)

```powershell
cd c:\Users\emadi\PFE-project\MemoryDiffusion
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1. PyTorch with CUDA (use cpu wheel index if you have no NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Everything else
pip install -r requirements.txt

# 3. Launch the demo (first run downloads ~7 GB of model weights)
python demo\gradio_app.py
```

Benchmark (after placing subjects under `data\benchmark\<subject>\*.jpg`):

```powershell
python evaluation\run_benchmark.py --data data\benchmark --out outputs\benchmark
```

---

## 7. Roadmap (suggested phases)

- [ ] **Phase 0 — Setup:** Environment, SD 1.5 + CLIP + FAISS loading, sanity generation.
- [ ] **Phase 1 — Memory bank:** Encode reference images, build FAISS index, top-k retrieval.
- [ ] **Phase 2 — Baseline conditioning:** Integrate IP-Adapter, generate a personalized image end-to-end.
- [ ] **Phase 3 — Demo:** Gradio app (upload → prompt → result).
- [ ] **Phase 4 — Evaluation:** Implement metrics, run against LoRA/DreamBooth baselines.
- [ ] **Phase 5 — Research extensions:** Encoder fusion, retrieval/aggregation ablations, multi-subject memory.
- [ ] **Phase 6 — Write-up:** Consolidate results into `paper.md`.

---

## 8. Known Challenges & Risks

- **Identity vs. editability trade-off** — the core tension of all personalization; needs careful conditioning-strength tuning.
- **Encoder mismatch** — CLIP/DINOv2 embeddings live in a different space than the diffusion model's conditioning space; a projection/adapter layer is usually required (IP-Adapter provides one).
- **Few-shot robustness** — with only 3–5 images, retrieval may capture background/pose rather than the subject; consider foreground segmentation during enrollment.
- **Multi-subject interference** — retrieval must not leak features between subjects in a shared bank.
- **Fair comparison** — baselines must be tuned reasonably, or the comparison is not credible.

---

## 9. Related Work (to cite in `paper.md`)

- **DreamBooth**, **LoRA**, **Textual Inversion** — fine-tuning-based personalization.
- **IP-Adapter** — decoupled cross-attention for image-prompt conditioning (closest mechanism).
- **RAG / Retrieval-Augmented Generation** — the conceptual parent from the LLM world.
- **Retrieval-Augmented Diffusion Models (RDM)** — prior work using retrieval to condition diffusion (positions this project's novelty: *personalization*-focused, training-free).

---

## 10. Positioning / Novelty

Retrieval-augmented diffusion exists; training-free personalization adapters (IP-Adapter) exist. **MemoryDiffusion's contribution is combining them into a persistent, editable, multi-subject memory system for personalization, and rigorously measuring whether it can substitute for fine-tuning.** The novelty is the *system framing + the empirical answer*, not any single component.

---

*Final-year project (PFE). Status: design phase.*
