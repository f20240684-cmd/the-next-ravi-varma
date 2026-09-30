#!/usr/bin/env python3
"""The Next Ravi Varma -- Gradio web dashboard.

Usage:
    python app/app.py [--config ../configs/generation.yaml] [--share]

If a trained LoRA checkpoint does not exist yet, the app still launches and
clearly tells you so in the Generate tab rather than pretending it exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ravi_varma.config import GenerationConfig
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.generation.prompt_engine import PromptExpansionEngine
from ravi_varma.utils.logging import setup_logging

from components import ABOUT_MARKDOWN, lora_status_message, run_evaluation, run_generation

logger = setup_logging("app")


def _require_gradio():
    try:
        import gradio as gr

        return gr
    except ImportError:
        logger.error(
            "Gradio is not installed in this environment. Run `pip install gradio` "
            "(see requirements.txt) to launch the web dashboard. All underlying pipeline "
            "code (dataset/training/generation/evaluation) works independently of Gradio "
            "via the scripts/ CLI tools."
        )
        raise SystemExit(2)


def build_app(config_path: str):
    gr = _require_gradio()
    config = GenerationConfig.load(config_path)
    generator = RaviVarmaGenerator(config, allow_missing_lora=True)
    prompt_engine = PromptExpansionEngine(trigger_token=config.style.trigger_token, negative_prompt=config.negative_prompt)

    css_path = Path(__file__).parent / "styles.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else None

    with gr.Blocks(css=css, title="The Next Ravi Varma") as demo:
        gr.HTML(
            "<div class='rv-header'><h1>🎨 The Next Ravi Varma</h1>"
            "<p>Generate novel, Ravi Varma-inspired artwork from Indian mythological &amp; historical scenes.</p></div>"
        )
        gr.Markdown(f"**LoRA status:** {lora_status_message(generator)}")

        with gr.Tabs():
            # ---------------------------------------------------------- #
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        scene = gr.Textbox(
                            label="Scene Description",
                            placeholder="Yudhishthira weeping over the fallen body of Karna on the battlefield of Kurukshetra.",
                            lines=3,
                        )
                        style_strength = gr.Slider(0.0, 1.5, value=config.generation.lora_scale, step=0.05, label="Style Strength (LoRA scale)")
                        seed = gr.Number(label="Seed (blank = random)", value=None, precision=0)
                        steps = gr.Slider(10, 60, value=config.generation.steps, step=1, label="Steps")
                        guidance = gr.Slider(1.0, 15.0, value=config.generation.guidance_scale, step=0.5, label="Guidance Scale")
                        width = gr.Dropdown([384, 448, 512, 576, 640, 768], value=config.generation.width, label="Width")
                        height = gr.Dropdown([384, 448, 512, 576, 640, 768], value=config.generation.height, label="Height")
                        negative_prompt = gr.Textbox(label="Negative Prompt", value=config.negative_prompt, lines=2)

                        with gr.Accordion("Optional: pose/composition reference (ControlNet)", open=False):
                            controlnet_enabled = gr.Checkbox(label="Enable ControlNet", value=False)
                            pose_image = gr.Image(label="Pose Reference", type="pil")
                            controlnet_type = gr.Radio(["openpose", "canny"], value=config.controlnet.type, label="ControlNet Type")
                            controlnet_strength = gr.Slider(0.0, 1.5, value=config.controlnet.strength, step=0.05, label="ControlNet Strength")

                        generate_btn = gr.Button("Generate Artwork", variant="primary")

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image")
                        expanded_prompt_box = gr.Textbox(label="Expanded Prompt", interactive=False, lines=3)
                        warning_box = gr.Markdown()
                        metadata_box = gr.Markdown(label="Metadata")

                generate_btn.click(
                    fn=lambda *a: run_generation(generator, prompt_engine, *a),
                    inputs=[
                        scene, style_strength, seed, steps, guidance, width, height,
                        negative_prompt, pose_image, controlnet_type, controlnet_strength, controlnet_enabled,
                    ],
                    outputs=[output_image, expanded_prompt_box, metadata_box, warning_box],
                )

            # ---------------------------------------------------------- #
            with gr.Tab("Advanced Controls"):
                gr.Markdown(
                    f"**Base model:** `{config.model.base_model}`  \n"
                    f"**Model family:** `{config.model.model_family}`  \n"
                    f"**Scheduler:** `{config.model.scheduler}`  \n"
                    f"**LoRA checkpoint path:** `{config.model.lora_path}`  \n\n"
                    "To change the base model, scheduler, or resolution beyond the presets above, "
                    "edit `configs/generation.yaml` and restart the app -- this keeps unsupported/"
                    "dangerous combinations (e.g. an SDXL LoRA against an SD1.5 base) out of the UI."
                )

            # ---------------------------------------------------------- #
            with gr.Tab("Evaluation"):
                gr.Markdown("Upload a generated image (or reuse the one above) and compare it against the reference corpus.")
                with gr.Row():
                    eval_image = gr.Image(label="Generated Image", type="pil")
                    with gr.Column():
                        eval_prompt = gr.Textbox(label="Prompt used to generate it")
                        reference_dir = gr.Textbox(label="Reference corpus directory", value="data/reference")
                        eval_btn = gr.Button("Calculate Metrics")
                eval_output = gr.Markdown()
                eval_btn.click(fn=run_evaluation, inputs=[eval_image, eval_prompt, reference_dir], outputs=[eval_output])

            # ---------------------------------------------------------- #
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MARKDOWN)

        gr.HTML(
            "<div class='rv-disclaimer'>Ravi Varma-inspired generative artwork. AI-generated; "
            "not an authentic Raja Ravi Varma painting.</div>"
        )

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "generation.yaml"))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_app(args.config)
    demo.launch(share=args.share, server_port=args.server_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
