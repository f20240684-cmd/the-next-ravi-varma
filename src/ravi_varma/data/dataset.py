"""Dataset preparation: turns validated raw images + captions into a
processed, resized image directory and a JSONL manifest suitable both for
Hugging Face `datasets.Dataset.from_json` and for the lightweight
`RaviVarmaImageCaptionDataset` used directly by the LoRA trainer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ravi_varma.config import DatasetConfig
from ravi_varma.utils.image import resize_for_training, safe_open_image
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ManifestRecord:
    image: str
    text: str
    artist: str = "Raja Ravi Varma"
    source: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "text": self.text,
            "artist": self.artist,
            "source": self.source,
            "year": self.year,
            "title": self.title,
        }


def _load_metadata_records(metadata_file: Path) -> list[dict]:
    if not metadata_file.exists():
        return []
    records = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_caption_for_image(rel_image_path: str, captions_dir: Path, metadata_record: Optional[dict]) -> Optional[str]:
    """Caption resolution order: metadata JSONL `caption` field takes
    priority (it may itself have been populated by the captioning script);
    otherwise fall back to a sidecar .txt/.json file in captions_dir with
    the same stem as the image."""
    if metadata_record and metadata_record.get("caption"):
        return str(metadata_record["caption"]).strip()

    stem = Path(rel_image_path).stem
    txt_path = captions_dir / f"{stem}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()

    json_path = captions_dir / f"{stem}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Support both a flat "flat_caption" key and a raw string file.
        if isinstance(data, dict):
            return data.get("flat_caption") or data.get("raw_caption")
        return None
    return None


def prepare_dataset(config: DatasetConfig) -> list[ManifestRecord]:
    """Read raw images + captions, resize/crop to `target_resolution`, write
    them to `processed_dir`, and return the manifest records (also written
    to `manifest_file`)."""
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    captions_dir = Path(config.captions_dir)
    metadata_file = Path(config.metadata_file)
    manifest_file = Path(config.manifest_file)

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_records = _load_metadata_records(metadata_file)
    metadata_by_image = {str(Path(r["image"])): r for r in metadata_records}

    allowed = set(config.validation.allowed_formats)
    image_paths = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed)

    manifest: list[ManifestRecord] = []
    skipped = 0
    for idx, img_path in enumerate(image_paths):
        rel = str(img_path.relative_to(raw_dir))
        record = metadata_by_image.get(rel)

        caption = _load_caption_for_image(rel, captions_dir, record)
        if not caption:
            skipped += 1
            logger.warning("Skipping %s: no caption found (metadata or captions/ sidecar file).", rel)
            continue

        img = safe_open_image(img_path)
        if img is None:
            skipped += 1
            logger.warning("Skipping %s: image is unreadable/corrupt.", rel)
            continue

        processed = resize_for_training(
            img,
            target_resolution=config.preprocessing.target_resolution,
            mode=config.preprocessing.resize_mode,
            interpolation=config.preprocessing.interpolation,
        )
        out_name = f"{idx:05d}.jpg"
        out_path = processed_dir / out_name
        processed.save(out_path, format="JPEG", quality=95)

        prefixed_caption = config.style.caption_prefix_template.format(
            trigger_token=config.style.trigger_token, caption=caption
        )

        manifest.append(
            ManifestRecord(
                image=str(out_path),
                text=prefixed_caption,
                artist=(record or {}).get("artist") or config.style.artist_name,
                source=(record or {}).get("source"),
                year=(record or {}).get("year"),
                title=(record or {}).get("title"),
            )
        )

    with manifest_file.open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec.to_dict()) + "\n")

    logger.info(
        "Prepared %d training samples (%d skipped) -> %s", len(manifest), skipped, manifest_file
    )
    return manifest


def load_manifest(manifest_file: "str | Path") -> list[ManifestRecord]:
    manifest_file = Path(manifest_file)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_file}. Run scripts/prepare_dataset.py first."
        )
    records = []
    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                records.append(ManifestRecord(**d))
    return records


def iter_hf_dataset_dicts(manifest: list[ManifestRecord]) -> Iterator[dict]:
    """Yield plain dicts in the shape `datasets.Dataset.from_list` (or
    `Dataset.from_generator`) expects: {"image": <path>, "text": <caption>}.
    Kept as a thin adapter so `datasets` remains an optional dependency for
    everything except the actual training script."""
    for rec in manifest:
        yield {"image": rec.image, "text": rec.text}


class RaviVarmaImageCaptionDataset:
    """A minimal torch Dataset over the manifest, used by the LoRA trainer.

    Implemented without inheriting from `torch.utils.data.Dataset` at import
    time so this module stays importable without torch; the class only
    requires torch inside `__getitem__`.
    """

    def __init__(self, manifest: list[ManifestRecord], resolution: int = 512, center_crop: bool = True, random_flip: bool = True):
        self.manifest = manifest
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        import random

        import numpy as np
        import torch
        from torchvision import transforms

        rec = self.manifest[idx]
        img = safe_open_image(rec.image)
        if img is None:
            raise RuntimeError(f"Could not read processed image {rec.image}; re-run prepare_dataset.py.")

        tfs = [transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.LANCZOS)]
        tfs.append(
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution)
        )
        if self.random_flip and random.random() < 0.5:
            tfs.append(transforms.RandomHorizontalFlip(p=1.0))
        tfs += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

        pixel_values = transforms.Compose(tfs)(img)
        return {"pixel_values": pixel_values, "text": rec.text}
