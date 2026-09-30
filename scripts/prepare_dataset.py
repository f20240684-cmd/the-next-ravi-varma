#!/usr/bin/env python3
"""Resize/crop validated images and build the training manifest.

Usage:
    python scripts/prepare_dataset.py [--config configs/dataset.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.data.dataset import prepare_dataset
from ravi_varma.data.validation import DatasetValidator
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("prepare_dataset")
    config = DatasetConfig.load(args.config)

    if not args.skip_validation:
        report = DatasetValidator(config).validate()
        if not report.is_valid:
            logger.error(
                "Dataset has 0 valid samples; refusing to prepare. Run "
                "scripts/validate_dataset.py for details, or pass --skip-validation to override."
            )
            return 1
        if report.error_count:
            logger.warning(
                "Proceeding with %d valid samples despite %d error-level validation issues; "
                "invalid samples will simply be skipped below.", report.valid_samples, report.error_count,
            )

    manifest = prepare_dataset(config)
    if not manifest:
        logger.error("No samples were prepared (missing captions/images?). See warnings above.")
        return 1

    logger.info("Prepared %d samples -> %s", len(manifest), config.manifest_file)
    logger.info("Next: python scripts/train_lora.py --config configs/lora_sd15.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
