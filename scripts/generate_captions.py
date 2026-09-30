#!/usr/bin/env python3
"""Generate dense captions for every image in the dataset.

Writes one sidecar `data/captions/<stem>.json` per image (structured
caption fields) and, when --write-metadata is passed, also fills the
`caption` field of data/metadata/metadata.jsonl with a flattened caption
suitable for training.

Usage:
    python scripts/generate_captions.py [--vlm-model Salesforce/blip-image-captioning-large] [--no-vlm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.data.captions import get_captioner
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--vlm-model", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-vlm", action="store_true", help="Skip the VLM and use the offline heuristic captioner only.")
    parser.add_argument("--write-metadata", action="store_true", help="Also fill the `caption` field in metadata.jsonl.")
    args = parser.parse_args()

    logger = setup_logging("generate_captions")
    config = DatasetConfig.load(args.config)

    raw_dir = Path(config.raw_dir)
    captions_dir = Path(config.captions_dir)
    captions_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(config.validation.allowed_formats)
    images = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
    if not images:
        logger.error("No images found in %s. Run scripts/validate_dataset.py first to check the dataset.", raw_dir)
        return 1

    captioner = get_captioner(prefer_vlm=not args.no_vlm, model_id=args.vlm_model, device=args.device)
    logger.info("Using captioner: %s (source=%s)", type(captioner).__name__, getattr(captioner, "available", True))

    flat_captions: dict[str, str] = {}
    for img_path in images:
        rel = str(img_path.relative_to(raw_dir))
        structured = captioner.caption(img_path)
        out_path = captions_dir / f"{img_path.stem}.json"
        out_path.write_text(
            json.dumps({**structured.to_dict(), "flat_caption": structured.as_flat_caption()}, indent=2),
            encoding="utf-8",
        )
        flat_captions[rel] = structured.as_flat_caption()
        logger.info("Captioned %s -> %s", rel, out_path)

    if args.write_metadata:
        metadata_file = Path(config.metadata_file)
        records = []
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        by_image = {r["image"]: r for r in records}
        for rel, caption in flat_captions.items():
            if rel in by_image:
                by_image[rel]["caption"] = caption
            else:
                by_image[rel] = {"image": rel, "caption": caption, "artist": config.style.artist_name}
        with metadata_file.open("w", encoding="utf-8") as f:
            for r in by_image.values():
                f.write(json.dumps(r) + "\n")
        logger.info("Updated captions in %s", metadata_file)

    logger.info("Captioned %d images -> %s", len(images), captions_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
