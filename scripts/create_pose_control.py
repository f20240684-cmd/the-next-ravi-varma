#!/usr/bin/env python3
"""Precompute a ControlNet conditioning map (OpenPose or Canny) from a
reference image, useful for inspecting the conditioning before generation.

Usage:
    python scripts/create_pose_control.py --image ref.jpg --type openpose --out pose_map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from ravi_varma.generation.controlnet import make_conditioning_image
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--type", choices=["openpose", "canny"], default="canny")
    parser.add_argument("--low-threshold", type=int, default=100)
    parser.add_argument("--high-threshold", type=int, default=200)
    parser.add_argument("--out", default=None, help="Defaults to <image>_<type>.png")
    args = parser.parse_args()

    logger = setup_logging("create_pose_control")

    image = Image.open(args.image).convert("RGB")
    conditioning = make_conditioning_image(
        image, control_type=args.type, canny_low_threshold=args.low_threshold, canny_high_threshold=args.high_threshold
    )

    out_path = Path(args.out) if args.out else Path(args.image).with_name(f"{Path(args.image).stem}_{args.type}.png")
    conditioning.save(out_path)
    logger.info("Saved %s conditioning map -> %s", args.type, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
