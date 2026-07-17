"""MemoryDiffusion — Gradio demo.

Run from the project root:
    python demo/gradio_app.py

First launch downloads SD 1.5 + IP-Adapter + CLIP + DINOv2 (~7 GB) into the
Hugging Face cache; later launches are instant (models load lazily on first
enroll/generate).
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image, ImageOps

from pipeline import MemoryDiffusion

MD = MemoryDiffusion(ROOT / "configs" / "default.yaml")

ALL_SUBJECTS = "✦ all memories"

EXAMPLE_PROMPTS = [
    "driving through Tokyo at night, neon reflections, cinematic",
    "on a beach at golden hour, soft light, 35mm photo",
    "in a snowy forest, morning fog, ultra detailed",
    "as an oil painting in the style of Van Gogh",
    "in a cyberpunk city, rain, dramatic lighting",
]

# --------------------------------------------------------------------------- #
# Look & feel — dark tokens from the validated reference palette              #
# --------------------------------------------------------------------------- #
CSS = """
:root {
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --hair: rgba(255,255,255,0.10);
  --accent: #3987e5;
  --accent-2: #9085e9;
}
.gradio-container {
  background:
    radial-gradient(1100px 480px at 12% -8%, rgba(57,135,229,0.14), transparent 60%),
    radial-gradient(900px 420px at 92% -6%, rgba(144,133,233,0.10), transparent 60%),
    var(--page) !important;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
  color: var(--ink-2);
}
/* ---- hero ---- */
#hero { text-align: center; padding: 26px 0 6px; }
#hero h1 {
  margin: 0;
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(100deg, #3987e5, #9085e9, #3987e5);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: hero-shift 7s ease-in-out infinite;
}
@keyframes hero-shift { 0%,100% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} }
#hero p { margin: 8px 0 0; color: var(--muted); font-size: 1.02rem; }
#hero .pill {
  display: inline-block; margin-top: 14px; padding: 5px 14px;
  border: 1px solid var(--hair); border-radius: 999px;
  color: var(--ink-2); font-size: 0.8rem; background: rgba(26,26,25,0.6);
}
/* ---- glass cards ---- */
.glass {
  background: rgba(26,26,25,0.72) !important;
  border: 1px solid var(--hair) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 14px 42px rgba(0,0,0,0.42);
  padding: 14px !important;
}
/* ---- primary button ---- */
.btn-primary {
  background: linear-gradient(100deg, #3987e5, #9085e9) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  transition: box-shadow 0.25s ease, transform 0.15s ease;
}
.btn-primary:hover { box-shadow: 0 0 26px rgba(57,135,229,0.45); transform: translateY(-1px); }
/* ---- memory / retrieval panels (single-series magnitude bars) ---- */
.mem-panel { display: flex; flex-direction: column; gap: 10px; }
.mem-row {
  display: flex; align-items: center; gap: 12px;
  background: var(--surface); border: 1px solid var(--hair);
  border-radius: 12px; padding: 8px 12px;
}
.mem-row img {
  width: 64px; height: 64px; object-fit: cover;
  border-radius: 10px; border: 1px solid var(--hair);
}
.mem-meta { flex: 1; min-width: 0; }
.mem-name {
  color: var(--ink); font-size: 0.86rem; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mem-track { height: 6px; background: var(--grid); border-radius: 4px; margin-top: 7px; }
.mem-fill  { height: 6px; background: var(--accent); border-radius: 4px; }
.mem-score { color: var(--ink-2); font-size: 0.78rem; margin-top: 5px;
             font-variant-numeric: tabular-nums; }
.panel-title { color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
               letter-spacing: 0.09em; margin: 2px 0 4px; }
/* ---- subject cards in the bank tab ---- */
.bank-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.bank-card { background: var(--surface); border: 1px solid var(--hair); border-radius: 14px; padding: 14px; }
.bank-card h4 { margin: 0 0 4px; color: var(--ink); font-size: 1rem; }
.bank-card .count { color: var(--muted); font-size: 0.8rem; margin-bottom: 10px; }
.bank-thumbs { display: flex; gap: 6px; flex-wrap: wrap; }
.bank-thumbs img { width: 52px; height: 52px; object-fit: cover; border-radius: 8px; border: 1px solid var(--hair); }
.empty-note { color: var(--muted); font-style: italic; padding: 14px 4px; }
footer { display: none !important; }
#foot { text-align: center; color: var(--muted); font-size: 0.8rem; padding: 18px 0 10px; }
"""

FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.toString());
  }
}
"""


# --------------------------------------------------------------------------- #
# HTML helpers                                                                #
# --------------------------------------------------------------------------- #
def _thumb_b64(path: Path, size: int = 128) -> str:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def retrieval_html(hits: list[dict]) -> str:
    """The 'what the model remembered' panel: thumbnail + similarity bar."""
    if not hits:
        return "<div class='empty-note'>No memories retrieved.</div>"
    top = max(h["score"] for h in hits)
    top = top if top > 1e-6 else 1.0
    rows = []
    for rank, h in enumerate(hits, start=1):
        rel = max(0.0, min(1.0, h["score"] / top))
        src = _thumb_b64(MD.bank.image_path(h["record"]))
        rows.append(
            f"<div class='mem-row'>"
            f"  <img src='{src}' alt='memory {rank}'/>"
            f"  <div class='mem-meta'>"
            f"    <div class='mem-name'>#{rank} · {h['record'].subject}</div>"
            f"    <div class='mem-track'><div class='mem-fill' style='width:{rel * 100:.1f}%'></div></div>"
            f"    <div class='mem-score'>cosine {h['score']:.3f}</div>"
            f"  </div>"
            f"</div>"
        )
    return (
        "<div class='mem-panel'>"
        "<div class='panel-title'>Retrieved memories — what the model remembered</div>"
        + "".join(rows)
        + "</div>"
    )


def bank_overview_html() -> str:
    grouped = MD.bank.subjects()
    if not grouped:
        return "<div class='empty-note'>Memory bank is empty — enroll your first subject.</div>"
    cards = []
    for subject, records in grouped.items():
        thumbs = "".join(
            f"<img src='{_thumb_b64(MD.bank.image_path(r), 96)}' alt=''/>" for r in records[:6]
        )
        more = f" +{len(records) - 6}" if len(records) > 6 else ""
        cards.append(
            f"<div class='bank-card'>"
            f"  <h4>🧠 {subject}</h4>"
            f"  <div class='count'>{len(records)} memories{more}</div>"
            f"  <div class='bank-thumbs'>{thumbs}</div>"
            f"</div>"
        )
    return "<div class='bank-grid'>" + "".join(cards) + "</div>"


def _subject_choices() -> list[str]:
    return [ALL_SUBJECTS] + list(MD.bank.subjects().keys())


# --------------------------------------------------------------------------- #
# Event handlers                                                              #
# --------------------------------------------------------------------------- #
def on_enroll(subject: str, files: list[str] | None):
    subject = (subject or "").strip()
    if not subject:
        raise gr.Error("Give your subject a name first.")
    if not files:
        raise gr.Error("Upload at least one reference image.")
    images = [Image.open(f) for f in files]
    count = MD.enroll(subject, images)
    status = (
        f"<div class='panel-title'>✅ Enrolled — “{subject}” now holds "
        f"{count} visual memories. No training happened.</div>"
    )
    choices = _subject_choices()
    return (
        status,
        gr.update(choices=choices, value=subject),
        gr.update(choices=choices[1:], value=None),
        bank_overview_html(),
    )


def on_generate(
    subject: str,
    prompt: str,
    negative: str,
    top_k: float,
    scale: float,
    preset: str,
    steps: float,
    guidance: float,
    seed: float,
    progress=gr.Progress(track_tqdm=True),
):
    if not (prompt or "").strip():
        raise gr.Error("Write a prompt first.")
    if len(MD.bank) == 0:
        raise gr.Error("Memory bank is empty — enroll a subject in the first tab.")
    subject_filter = None if subject in (None, "", ALL_SUBJECTS) else subject
    try:
        image, hits = MD.generate(
            prompt=prompt.strip(),
            subject=subject_filter,
            negative_prompt=negative.strip() or None,
            top_k=int(top_k),
            ip_adapter_scale=float(scale),
            scale_preset=preset,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
        )
    except ValueError as err:
        raise gr.Error(str(err))
    return image, retrieval_html(hits)


def on_delete(subject: str):
    if not subject:
        raise gr.Error("Pick a subject to forget.")
    removed = MD.bank.remove_subject(subject)
    status = f"<div class='panel-title'>🗑️ Forgot “{subject}” ({removed} memories erased).</div>"
    choices = _subject_choices()
    return (
        status,
        gr.update(choices=choices, value=ALL_SUBJECTS),
        gr.update(choices=choices[1:], value=None),
        bank_overview_html(),
    )


def on_refresh():
    choices = _subject_choices()
    return (
        gr.update(choices=choices),
        gr.update(choices=choices[1:]),
        bank_overview_html(),
    )


def on_files_change(files: list[str] | None):
    return files or []


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, css=CSS, js=FORCE_DARK_JS, title="MemoryDiffusion") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>MemoryDiffusion</h1>
          <p>Personalized image generation without training — the model consults a visual memory bank instead of fine-tuning.</p>
          <span class="pill">SD 1.5 · IP-Adapter · CLIP + DINOv2 · FAISS · zero gradient steps</span>
        </div>
        """
    )

    with gr.Tabs():
        # ------------------------------ enroll ---------------------------- #
        with gr.Tab("🧠 Enroll a memory"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["glass"]):
                    subject_in = gr.Textbox(
                        label="Subject name",
                        placeholder="e.g. my vintage car, Luna the cat, brand mascot…",
                    )
                    files_in = gr.File(
                        label="Reference images (3–10 recommended)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    enroll_btn = gr.Button("⚡ Memorize", elem_classes=["btn-primary"])
                    enroll_status = gr.HTML()
                with gr.Column(scale=5, elem_classes=["glass"]):
                    preview_gallery = gr.Gallery(
                        label="Preview", columns=4, height=360, object_fit="cover"
                    )

        # ----------------------------- generate --------------------------- #
        with gr.Tab("✨ Generate"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["glass"]):
                    gen_subject = gr.Dropdown(
                        label="Subject",
                        choices=_subject_choices(),
                        value=ALL_SUBJECTS,
                    )
                    prompt_in = gr.Textbox(
                        label="Prompt",
                        placeholder="the subject driving through Tokyo at night…",
                        lines=2,
                    )
                    gr.Examples(examples=[[p] for p in EXAMPLE_PROMPTS], inputs=[prompt_in])
                    negative_in = gr.Textbox(
                        label="Negative prompt",
                        value=MD.cfg["diffusion"]["negative_prompt"],
                        lines=2,
                    )
                    with gr.Accordion("⚙️ Memory & diffusion controls", open=False):
                        topk_in = gr.Slider(1, 8, value=MD.cfg["retrieval"]["top_k"], step=1,
                                            label="Memories retrieved (top-k)")
                        scale_in = gr.Slider(0.0, 1.2, value=MD.cfg["generation"]["ip_adapter_scale"],
                                             step=0.05,
                                             label="Memory strength (identity ↔ editability)")
                        preset_in = gr.Radio(
                            choices=["uniform", "style_only", "style_and_layout"],
                            value=MD.cfg["generation"]["scale_preset"],
                            label="Injection preset (which U-Net layers receive memory)",
                        )
                        steps_in = gr.Slider(10, 60, value=MD.cfg["diffusion"]["num_inference_steps"],
                                             step=1, label="Inference steps")
                        guidance_in = gr.Slider(1.0, 15.0, value=MD.cfg["diffusion"]["guidance_scale"],
                                                step=0.5, label="Guidance scale (CFG)")
                        seed_in = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    generate_btn = gr.Button("🎨 Generate from memory", elem_classes=["btn-primary"])
                with gr.Column(scale=5, elem_classes=["glass"]):
                    output_image = gr.Image(label="Result", type="pil", height=430)
                    retrieval_panel = gr.HTML(value="<div class='empty-note'>Retrieved memories will appear here.</div>")

        # ---------------------------- memory bank ------------------------- #
        with gr.Tab("🗂️ Memory bank"):
            with gr.Column(elem_classes=["glass"]):
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                bank_overview = gr.HTML(value=bank_overview_html())
                with gr.Row():
                    del_subject = gr.Dropdown(
                        label="Forget a subject",
                        choices=_subject_choices()[1:],
                    )
                    delete_btn = gr.Button("🗑️ Forget", variant="stop")
                delete_status = gr.HTML()

    gr.HTML("<div id='foot'>MemoryDiffusion · PFE research demo · external memory instead of fine-tuning</div>")

    # ------------------------------- wiring ------------------------------- #
    files_in.change(on_files_change, inputs=[files_in], outputs=[preview_gallery])
    enroll_btn.click(
        on_enroll,
        inputs=[subject_in, files_in],
        outputs=[enroll_status, gen_subject, del_subject, bank_overview],
    )
    generate_btn.click(
        on_generate,
        inputs=[gen_subject, prompt_in, negative_in, topk_in, scale_in, preset_in,
                steps_in, guidance_in, seed_in],
        outputs=[output_image, retrieval_panel],
    )
    delete_btn.click(
        on_delete,
        inputs=[del_subject],
        outputs=[delete_status, gen_subject, del_subject, bank_overview],
    )
    refresh_btn.click(on_refresh, outputs=[gen_subject, del_subject, bank_overview])


if __name__ == "__main__":
    demo.queue().launch()
