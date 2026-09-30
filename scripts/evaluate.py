#!/usr/bin/env python3
"""Evaluate generated images: CLIP text-image alignment + style similarity
against a reference corpus.

Usage:
    python scripts/evaluate.py --input outputs/generated --reference data/reference
    python scripts/evaluate.py --image outputs/generated/12345_ab12cd34.png --prompt "..." --reference data/reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.evaluation.clip_score import clip_available
from ravi_varma.evaluation.report import batch_evaluate, evaluate_image
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/generated", help="Directory of generated images to batch-evaluate.")
    parser.add_argument("--image", default=None, help="Evaluate a single image instead of a whole directory.")
    parser.add_argument("--prompt", default=None, help="Prompt for --image (defaults to the sidecar .json's 'prompt' field).")
    parser.add_argument("--reference", default="data/reference", help="Directory of reference Ravi Varma paintings.")
    parser.add_argument("--output-csv", default="outputs/evaluations/report.csv")
    args = parser.parse_args()

    logger = setup_logging("evaluate")

    availability = clip_available()
    if not availability.available:
        logger.warning("CLIP unavailable (%s). Metrics will be reported as null, not fabricated.", availability.reason)

    if args.image:
        result = evaluate_image(args.image, args.reference, prompt=args.prompt)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    results = batch_evaluate(args.input, args.reference, args.output_csv)
    if not results:
        logger.error("No images found to evaluate in %s", args.input)
        return 1

    print(f"Evaluated {len(results)} image(s). CSV report -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
