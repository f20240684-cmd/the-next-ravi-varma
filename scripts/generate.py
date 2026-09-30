#!/usr/bin/env python3
"""Generate a Ravi Varma-inspired image from a scene description.

Usage:
    python scripts/generate.py --prompt "Arjuna standing before Krishna on the battlefield of Kurukshetra"
    python scripts/generate.py --prompt "..." --pose-image path/to/pose.jpg --controlnet-type openpose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from ravi_varma.config import GenerationConfig
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.generation.prompt_engine import PromptExpansionEngine
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Natural-language scene description.")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--no-style", action="store_true", help="Disable the Ravi Varma trigger token / style injection.")
    parser.add_argument("--no-prompt-expansion", action="store_true", help="Use the raw prompt as-is, skipping the expansion engine.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--pose-image", default=None, help="Optional reference image for ControlNet.")
    parser.add_argument("--controlnet-type", choices=["openpose", "canny"], default=None)
    parser.add_argument("--controlnet-strength", type=float, default=None)
    args = parser.parse_args()

    logger = setup_logging("generate")
    config = GenerationConfig.load(args.config)

    if args.no_prompt_expansion:
        final_prompt = args.prompt
        expanded = None
    else:
        engine = PromptExpansionEngine(trigger_token=config.style.trigger_token, negative_prompt=config.negative_prompt)
        expanded = engine.expand(args.prompt, enable_style=not args.no_style)
        final_prompt = expanded.final_prompt()
        logger.info("Expanded prompt (mode=%s): %s", expanded.mode_used, final_prompt)

    control_image = None
    if args.pose_image:
        control_image = Image.open(args.pose_image).convert("RGB")

    generator = RaviVarmaGenerator(config, allow_missing_lora=True)
    try:
        result = generator.generate(
            prompt=final_prompt,
            negative_prompt=expanded.negative_prompt if expanded else None,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            lora_scale=args.lora_scale,
            control_image=control_image,
            control_type=args.controlnet_type,
            controlnet_strength=args.controlnet_strength,
        )
    except RuntimeError as e:
        logger.error("%s", e)
        return 2

    print(f"Saved image:    {result.image_path}")
    print(f"Saved metadata: {result.metadata_path}")
    print(json.dumps(result.metadata, indent=2))
    if not result.metadata["lora_loaded"]:
        print(
            "\nWARNING: no trained LoRA checkpoint was found -- this image was generated with the "
            "BASE model only and does not reflect Ravi Varma's style. Run scripts/train_lora.py first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
