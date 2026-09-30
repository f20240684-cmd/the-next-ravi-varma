#!/usr/bin/env python3
"""Validate the raw Ravi Varma image dataset + metadata.

Usage:
    python scripts/validate_dataset.py [--config configs/dataset.yaml] [--json-out data/metadata/validation_report.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.data.validation import DatasetValidator
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--json-out", default="data/metadata/validation_report.json")
    args = parser.parse_args()

    logger = setup_logging("validate_dataset")

    try:
        config = DatasetConfig.load(args.config)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 2

    validator = DatasetValidator(config)
    report = validator.validate()
    validator.save_report(report, args.json_out)

    print(report.summary())
    print(f"\nFull JSON report: {args.json_out}")

    if not report.is_valid:
        logger.error("Dataset is not trainable: 0 valid samples. See errors above.")
        return 1
    if report.error_count:
        logger.warning("Dataset has %d valid samples but also %d error-level issues to review.", report.valid_samples, report.error_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
