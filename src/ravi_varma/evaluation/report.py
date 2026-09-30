"""Aggregates text-image alignment, style similarity, and optional
image-quality metrics into a single evaluation report per generated image,
plus batch evaluation across a directory with a CSV summary.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ravi_varma.evaluation.clip_score import text_image_similarity
from ravi_varma.evaluation.style_similarity import compute_style_similarity
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_LIMITATIONS = {
    "text_alignment": (
        "CLIP text-image similarity measures coarse semantic alignment between the prompt and "
        "image; it does not verify fine-grained details (e.g. exact pose, number of fingers) "
        "and is not a proxy for aesthetic quality."
    ),
    "style_similarity": (
        "Embedding similarity to a reference corpus is a rough, comparative signal, not a "
        "validated measure of artistic style transfer or authenticity."
    ),
    "ssim": "SSIM measures pixel-level structural similarity and is only meaningful when compared against a specific reference image, not a style corpus in general.",
    "lpips": "LPIPS is a learned perceptual distance; lower is 'more similar' but the metric was not trained specifically on paintings and may not track painterly style well.",
}


@dataclass
class ImageEvaluation:
    image_path: str
    prompt: Optional[str]
    text_alignment: Optional[float]
    style_similarity_mean: Optional[float]
    style_similarity_median: Optional[float]
    style_similarity_nearest: Optional[float]
    style_similarity_nearest_ref: Optional[str]
    num_references_used: int
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _read_prompt_from_sidecar(image_path: Path) -> Optional[str]:
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return data.get("prompt")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def evaluate_image(
    image_path: "str | Path",
    reference_dir: "str | Path",
    prompt: Optional[str] = None,
    device: str = "cpu",
) -> ImageEvaluation:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    prompt = prompt or _read_prompt_from_sidecar(image_path)

    text_alignment = text_image_similarity(image_path, prompt, device=device) if prompt else None
    if prompt is None:
        logger.warning("No prompt available for %s (pass --prompt or ensure a sidecar .json exists); skipping text_alignment.", image_path)

    style = compute_style_similarity(image_path, reference_dir, device=device)

    notes = None
    if text_alignment is None or style.mean_similarity is None:
        notes = "Some metrics could not be computed -- see logs. No score was fabricated."

    return ImageEvaluation(
        image_path=str(image_path),
        prompt=prompt,
        text_alignment=text_alignment,
        style_similarity_mean=style.mean_similarity,
        style_similarity_median=style.median_similarity,
        style_similarity_nearest=style.nearest_similarity,
        style_similarity_nearest_ref=style.nearest_reference,
        num_references_used=style.num_references_used,
        notes=notes,
    )


def batch_evaluate(
    input_dir: "str | Path",
    reference_dir: "str | Path",
    output_csv: "str | Path",
    device: str = "cpu",
) -> list[ImageEvaluation]:
    input_dir = Path(input_dir)
    images = sorted(p for p in input_dir.glob("*.png")) + sorted(p for p in input_dir.glob("*.jpg"))
    if not images:
        logger.warning("No .png/.jpg images found in %s", input_dir)

    results = [evaluate_image(img, reference_dir, device=device) for img in images]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ImageEvaluation.__dataclass_fields__.keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    logger.info("Wrote batch evaluation CSV (%d rows) -> %s", len(results), output_csv)
    return results
