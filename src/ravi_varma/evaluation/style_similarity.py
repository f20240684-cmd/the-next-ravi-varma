"""Style similarity: compares a generated image against a reference corpus
of real Ravi Varma paintings using CLIP image embeddings.

This is an *image-embedding similarity* metric, not a validated measure of
artistic style or authenticity -- see the caveats returned alongside every
result and repeated in the README/App "About" tab.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ravi_varma.evaluation.clip_score import image_embedding
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class StyleSimilarityResult:
    mean_similarity: Optional[float]
    median_similarity: Optional[float]
    nearest_similarity: Optional[float]
    nearest_reference: Optional[str]
    num_references_used: int
    caveat: str = (
        "This score reflects CLIP image-embedding cosine similarity to a reference corpus, "
        "not a validated measure of artistic style or authenticity. It should be read as a "
        "rough, comparative signal only."
    )


def _load_reference_embeddings(reference_dir: "str | Path", device: str = "cpu") -> dict[str, np.ndarray]:
    reference_dir = Path(reference_dir)
    embeddings = {}
    if not reference_dir.exists():
        return embeddings
    for path in sorted(reference_dir.rglob("*")):
        if path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            emb = image_embedding(path, device=device)
            if emb is not None:
                embeddings[str(path)] = emb
    return embeddings


def compute_style_similarity(
    generated_image: "str | Path",
    reference_dir: "str | Path",
    device: str = "cpu",
) -> StyleSimilarityResult:
    generated_embedding = image_embedding(generated_image, device=device)
    if generated_embedding is None:
        logger.warning("CLIP unavailable; cannot compute style similarity.")
        return StyleSimilarityResult(None, None, None, None, 0)

    reference_embeddings = _load_reference_embeddings(reference_dir, device=device)
    if not reference_embeddings:
        logger.warning(
            "No reference images found in %s; cannot compute style similarity. "
            "Populate it with licensed/public-domain Ravi Varma paintings.", reference_dir,
        )
        return StyleSimilarityResult(None, None, None, None, 0)

    sims = {
        path: float(np.dot(generated_embedding, emb))
        for path, emb in reference_embeddings.items()
    }
    nearest_path = max(sims, key=sims.get)

    return StyleSimilarityResult(
        mean_similarity=statistics.mean(sims.values()),
        median_similarity=statistics.median(sims.values()),
        nearest_similarity=sims[nearest_path],
        nearest_reference=nearest_path,
        num_references_used=len(sims),
    )
