#!/usr/bin/env python3
"""Train a Ravi Varma-style LoRA on top of a Stable Diffusion base model.

Usage:
    python scripts/train_lora.py --config configs/lora_sd15.yaml
    python scripts/train_lora.py --config configs/lora_sd15.yaml --max-train-steps 50   # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import LoraTrainingConfig
from ravi_varma.training.lora_trainer import LoraTrainer, check_ml_stack_available
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a lora_sd15.yaml / lora_sdxl.yaml file.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override training.max_train_steps (useful for a quick smoke test).")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Override checkpointing.resume_from_checkpoint.")
    args = parser.parse_args()

    logger = setup_logging("train_lora")

    config = LoraTrainingConfig.load(args.config)
    if args.max_train_steps is not None:
        config.training.max_train_steps = args.max_train_steps
    if args.resume_from_checkpoint is not None:
        config.checkpointing.resume_from_checkpoint = args.resume_from_checkpoint

    missing = check_ml_stack_available()
    if missing:
        logger.error(missing)
        logger.error(
            "This machine cannot run real LoRA training. Use Google Colab "
            "(notebooks/03_lora_training.ipynb) or a CUDA machine with requirements.txt installed."
        )
        return 2

    trainer = LoraTrainer(config)
    try:
        output_dir = trainer.train()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info("Training complete. LoRA checkpoint saved to %s", output_dir)
    logger.info("Next: python scripts/generate.py --prompt \"...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
