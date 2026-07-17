"""MemoryDiffusion benchmark harness.

Data layout (one folder per subject, images directly inside):

    data/benchmark/
    ├── vintage-car/   *.jpg / *.png   (3-10 reference images)
    ├── luna-cat/      ...
    └── prompts.txt    optional — one template per line, "{}" = subject name

Run from the project root:
    python evaluation/run_benchmark.py --data data/benchmark --out outputs/benchmark

Produces per-subject generations, results.json, and a dark-themed summary chart
(benchmark_chart.png) of CLIP-I / CLIP-T / DINO.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from evaluation.metrics import clip_i, clip_t, dino_i
from pipeline import MemoryDiffusion

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_PROMPTS = [
    "a photo of {} in Tokyo at night, neon lights",
    "a photo of {} on a beach at sunset",
    "a photo of {} in a snowy forest",
    "an oil painting of {}",
]

# ---- dark chart tokens (validated reference palette, dark mode) ----------- #
PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
BASELINE = "#383835"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
SERIES = {"CLIP-I": "#3987e5", "CLIP-T": "#008300", "DINO": "#d55181"}  # slots 1-3


def load_references(subject_dir: Path) -> list[Image.Image]:
    paths = sorted(p for p in subject_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]


def plot_results(results: dict[str, dict[str, float]], out_path: Path) -> None:
    """Grouped bars — subjects on x, one bar per metric (all cosine, one axis)."""
    subjects = list(results.keys())
    metrics = list(SERIES.keys())
    x = np.arange(len(subjects))
    width = 0.24

    fig, ax = plt.subplots(figsize=(1.6 + 2.2 * len(subjects), 4.6), dpi=180)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(SURFACE)

    for i, metric in enumerate(metrics):
        values = [results[s][metric] for s in subjects]
        ax.bar(x + (i - 1) * (width + 0.02), values, width,
               label=metric, color=SERIES[metric], edgecolor=SURFACE, linewidth=1.5)

    ax.set_ylim(0, 1)
    ax.set_ylabel("cosine similarity", color=MUTED, fontsize=9)
    ax.set_xticks(x, subjects, color=INK_2, fontsize=10)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_title("MemoryDiffusion — identity & prompt adherence",
                 color=INK, fontsize=12, pad=14, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right", ncols=3)
    for text in legend.get_texts():
        text.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MemoryDiffusion benchmark")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for reproducibility")
    args = parser.parse_args()

    subject_dirs = sorted(
        d for d in args.data.iterdir() if d.is_dir() and load_references(d)
    ) if args.data.exists() else []
    if not subject_dirs:
        raise SystemExit(
            f"No subject folders with images found under {args.data}.\n"
            "Expected data/benchmark/<subject>/*.jpg — see the module docstring."
        )

    prompts_file = args.data / "prompts.txt"
    templates = (
        [line.strip() for line in prompts_file.read_text("utf-8").splitlines() if line.strip()]
        if prompts_file.exists() else DEFAULT_PROMPTS
    )

    args.out.mkdir(parents=True, exist_ok=True)
    md = MemoryDiffusion(args.config)
    # Isolated memory store so benchmarks never pollute the demo's bank.
    md.cfg["memory"]["store_dir"] = str(args.out / "memory_store")
    from retrieval.memory_bank import MemoryBank
    md.bank = MemoryBank(md.cfg["memory"]["store_dir"])

    results: dict[str, dict[str, float]] = {}
    timings: dict[str, dict[str, float]] = {}

    for subject_dir in subject_dirs:
        subject = subject_dir.name.replace("-", " ").replace("_", " ")
        references = load_references(subject_dir)
        print(f"\n=== {subject} ({len(references)} references) ===")

        t0 = time.perf_counter()
        md.enroll(subject, references)
        enroll_s = time.perf_counter() - t0
        print(f"  enrolled in {enroll_s:.1f}s (no training)")

        generated, prompts = [], []
        gen_dir = args.out / subject_dir.name
        gen_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for i, template in enumerate(templates):
            prompt = template.format(subject) if "{}" in template else template
            image, _hits = md.generate(prompt=prompt, subject=subject, seed=args.seed + i)
            image.save(gen_dir / f"{i:02d}.png")
            generated.append(image)
            prompts.append(prompt)
            print(f"  [{i + 1}/{len(templates)}] {prompt}")
        gen_s = (time.perf_counter() - t0) / max(len(templates), 1)

        results[subject] = {
            "CLIP-I": clip_i(md.encoders, generated, references),
            "CLIP-T": clip_t(md.encoders, generated, prompts),
            "DINO": dino_i(md.encoders, generated, references),
        }
        timings[subject] = {"enroll_s": enroll_s, "s_per_image": gen_s}
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in results[subject].items()))

    payload = {
        "config": md.cfg,
        "prompt_templates": templates,
        "results": results,
        "timings": timings,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2), "utf-8")
    plot_results(results, args.out / "benchmark_chart.png")

    print(f"\nSaved: {args.out / 'results.json'}")
    print(f"Saved: {args.out / 'benchmark_chart.png'}")
    means = {m: float(np.mean([r[m] for r in results.values()])) for m in SERIES}
    print("Overall  " + "  ".join(f"{k}={v:.3f}" for k, v in means.items()))


if __name__ == "__main__":
    main()
