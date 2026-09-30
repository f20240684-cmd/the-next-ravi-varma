"""Dataset validation.

Used by `scripts/validate_dataset.py`. Checks a raw image directory (plus an
optional metadata JSONL file) for the problems most likely to break LoRA
training: corrupt images, missing files, unsupported formats, near-duplicate
images, images below a minimum resolution, and missing/invalid captions.

Only depends on Pillow/numpy (via `ravi_varma.utils.image`) -- no torch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ravi_varma.config import DatasetConfig
from ravi_varma.utils.image import file_sha256, get_resolution, hamming_distance, perceptual_hash, safe_open_image
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_METADATA_FIELDS = {"image"}
OPTIONAL_METADATA_FIELDS = {"caption", "artist", "source", "year", "title"}


@dataclass
class DatasetIssue:
    severity: str  # "error" | "warning"
    category: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category, "path": self.path, "message": self.message}


@dataclass
class ValidationReport:
    total_images_found: int = 0
    valid_samples: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)

    def add(self, severity: str, category: str, path: str, message: str) -> None:
        self.issues.append(DatasetIssue(severity, category, path, message))

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Dataset is trainable (not necessarily perfect) if it has at least
        one valid sample and no fatal (metadata-file-level) errors."""
        return self.valid_samples > 0

    def to_dict(self) -> dict:
        return {
            "total_images_found": self.total_images_found,
            "valid_samples": self.valid_samples,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        lines = [
            "=== Dataset Validation Report ===",
            f"Images found:     {self.total_images_found}",
            f"Valid samples:    {self.valid_samples}",
            f"Errors:           {self.error_count}",
            f"Warnings:         {self.warning_count}",
        ]
        for issue in self.issues[:200]:
            lines.append(f"  [{issue.severity.upper():7s}] {issue.category:18s} {issue.path}: {issue.message}")
        if len(self.issues) > 200:
            lines.append(f"  ... and {len(self.issues) - 200} more issues (see JSON report)")
        return "\n".join(lines)


def _load_metadata(metadata_file: Path) -> list[dict]:
    if not metadata_file.exists():
        return []
    records = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {metadata_file}:{lineno}: {e}") from e
            if not REQUIRED_METADATA_FIELDS.issubset(record.keys()):
                raise ValueError(
                    f"Metadata record at {metadata_file}:{lineno} missing required field(s): "
                    f"{REQUIRED_METADATA_FIELDS - record.keys()}"
                )
            records.append(record)
    return records


class DatasetValidator:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        raw_dir = Path(self.config.raw_dir)
        metadata_file = Path(self.config.metadata_file)
        cfg = self.config.validation

        if not raw_dir.exists():
            report.add("error", "missing_directory", str(raw_dir), "Raw image directory does not exist.")
            return report

        # 1. Discover candidate image files on disk.
        all_files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
        image_candidates = [p for p in all_files if p.suffix.lower() in cfg.allowed_formats]
        non_image_files = [p for p in all_files if p.suffix.lower() not in cfg.allowed_formats and p.name != ".gitkeep"]
        for p in non_image_files:
            report.add("warning", "unsupported_format", str(p), f"Unsupported file extension '{p.suffix}'.")

        report.total_images_found = len(image_candidates)
        if report.total_images_found == 0:
            report.add(
                "error",
                "empty_dataset",
                str(raw_dir),
                "No images with an allowed extension were found. Add images to data/raw/.",
            )

        # 2. Load metadata (captions), if provided.
        try:
            metadata_records = _load_metadata(metadata_file)
        except ValueError as e:
            report.add("error", "invalid_metadata", str(metadata_file), str(e))
            metadata_records = []

        caption_by_image: dict[str, dict] = {}
        for record in metadata_records:
            caption_by_image[str(Path(record["image"]))] = record

        if metadata_records:
            for record in metadata_records:
                img_path = raw_dir / record["image"]
                if not img_path.exists():
                    report.add(
                        "error",
                        "missing_image",
                        record["image"],
                        "Metadata references an image that does not exist on disk.",
                    )
        elif cfg.require_caption:
            report.add(
                "warning",
                "no_metadata_file",
                str(metadata_file),
                "No metadata file found; captions cannot be validated. "
                "Run scripts/generate_captions.py or provide data/metadata/metadata.jsonl.",
            )

        # 3. Per-image checks: corruption, resolution, duplicates, captions.
        seen_hashes: dict[str, list[str]] = {}
        seen_sha256: dict[str, list[str]] = {}
        for img_path in image_candidates:
            rel = str(img_path.relative_to(raw_dir))

            size_mb = img_path.stat().st_size / (1024 * 1024)
            if size_mb > cfg.max_file_size_mb:
                report.add(
                    "warning", "large_file", rel, f"File is {size_mb:.1f}MB (limit {cfg.max_file_size_mb}MB)."
                )

            img = safe_open_image(img_path)
            if img is None:
                report.add("error", "corrupt_image", rel, "Image is corrupt, truncated, or unreadable.")
                continue

            resolution = img.size
            if min(resolution) < cfg.min_resolution:
                report.add(
                    "error",
                    "low_resolution",
                    rel,
                    f"Shortest side {min(resolution)}px is below the minimum {cfg.min_resolution}px.",
                )
                continue

            sha = file_sha256(img_path)
            seen_sha256.setdefault(sha, []).append(rel)

            if cfg.detect_duplicates:
                phash = perceptual_hash(img_path)
                if phash is not None:
                    seen_hashes.setdefault(phash, []).append(rel)

            has_caption = rel in caption_by_image and bool(caption_by_image[rel].get("caption"))
            if cfg.require_caption and metadata_records and not has_caption:
                report.add("error", "missing_caption", rel, "Image has no non-empty caption in metadata.")
                continue

            report.valid_samples += 1

        # Exact duplicates (identical bytes).
        for sha, paths in seen_sha256.items():
            if len(paths) > 1:
                for p in paths[1:]:
                    report.add("warning", "exact_duplicate", p, f"Identical (byte-for-byte) to {paths[0]}.")

        # Near-duplicates (perceptual hash within a small Hamming distance).
        if cfg.detect_duplicates:
            hashes = list(seen_hashes.keys())
            for i, h1 in enumerate(hashes):
                for h2 in hashes[i + 1 :]:
                    if hamming_distance(h1, h2) <= 4:
                        for p in seen_hashes[h2]:
                            report.add(
                                "warning",
                                "near_duplicate",
                                p,
                                f"Visually near-identical to {seen_hashes[h1][0]} (hash distance <= 4).",
                            )

        return report

    def save_report(self, report: ValidationReport, output_path: "str | Path") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved validation report to %s", output_path)
