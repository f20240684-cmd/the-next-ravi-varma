"""Dense caption generation.

Wraps a VLM (default: Salesforce/blip-image-captioning-large, configurable to
a BLIP-2/LLaVA-style model) to produce structured captions per the schema in
the project spec (Subject / Composition / Pose / Clothing / Lighting /
Environment / Visual characteristics).

`transformers`/`torch` are imported lazily inside `VLMCaptioner` so that
`scripts/generate_captions.py --help`, dataset validation, and the rest of
the app work without them installed. If they are unavailable, or a specific
model fails to load, `VLMCaptioner.available` is False and callers should
fall back to `HeuristicCaptioner`, which never invents details -- it only
reports what it can measure directly from pixel data (palette, aspect
ratio, brightness) and leaves everything else explicitly "unknown".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

CAPTION_FIELDS = [
    "subject",
    "composition",
    "pose",
    "clothing",
    "lighting",
    "environment",
    "visual_characteristics",
]


@dataclass
class StructuredCaption:
    subject: str = "unknown"
    composition: str = "unknown"
    pose: str = "unknown"
    clothing: str = "unknown"
    lighting: str = "unknown"
    environment: str = "unknown"
    visual_characteristics: str = "unknown"
    raw_caption: Optional[str] = None  # unstructured VLM output, if available
    source: str = "heuristic"  # "vlm" | "heuristic"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "composition": self.composition,
            "pose": self.pose,
            "clothing": self.clothing,
            "lighting": self.lighting,
            "environment": self.environment,
            "visual_characteristics": self.visual_characteristics,
            "raw_caption": self.raw_caption,
            "source": self.source,
        }

    def as_text_block(self) -> str:
        """Render in the structured text format shown in the project spec."""
        return (
            f"Subject:\n{self.subject}\n\n"
            f"Composition:\n{self.composition}\n\n"
            f"Pose:\n{self.pose}\n\n"
            f"Clothing:\n{self.clothing}\n\n"
            f"Lighting:\n{self.lighting}\n\n"
            f"Environment:\n{self.environment}\n\n"
            f"Visual characteristics:\n{self.visual_characteristics}\n"
        )

    def as_flat_caption(self) -> str:
        """A single-line caption suitable for training (dataset manifest),
        built only from fields that are not "unknown"."""
        parts = [self.raw_caption] if self.raw_caption else []
        for f in ("subject", "clothing", "pose", "environment", "lighting", "visual_characteristics"):
            v = getattr(self, f)
            if v and v != "unknown":
                parts.append(v)
        return ", ".join(dict.fromkeys(p.strip() for p in parts if p and p.strip()))


class HeuristicCaptioner:
    """Zero-dependency fallback captioner.

    Deliberately conservative: it only reports properties it can actually
    measure from the pixels (rough brightness/contrast, dominant color
    family, aspect ratio -> likely composition), and leaves subject/pose/
    clothing/environment as "unknown" rather than guessing. This matches the
    spec's requirement to never hallucinate details a model cannot verify.
    """

    available = True

    def caption(self, image_path: "str | Path") -> StructuredCaption:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        thumb = img.resize((64, 64))
        pixels = list(thumb.getdata())
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)
        brightness = (avg_r + avg_g + avg_b) / 3

        if brightness < 85:
            lighting = "predominantly dark/low-key lighting (measured, not confirmed as artistic chiaroscuro)"
        elif brightness > 170:
            lighting = "predominantly bright/high-key lighting"
        else:
            lighting = "moderate mid-tone lighting"

        if avg_r > avg_g and avg_r > avg_b:
            palette = "warm color palette, red/orange dominant tones"
        elif avg_b > avg_r and avg_b > avg_g:
            palette = "cool color palette, blue-dominant tones"
        else:
            palette = "balanced color palette"

        aspect = w / h
        if 0.9 <= aspect <= 1.1:
            composition = "roughly square composition"
        elif aspect > 1.1:
            composition = "landscape-oriented composition, wider than tall"
        else:
            composition = "portrait-oriented composition, taller than wide"

        return StructuredCaption(
            subject="unknown (no VLM available -- run with --vlm to auto-detect subjects)",
            composition=composition,
            pose="unknown",
            clothing="unknown",
            lighting=lighting,
            environment="unknown",
            visual_characteristics=palette,
            raw_caption=None,
            source="heuristic",
        )


class VLMCaptioner:
    """BLIP-family captioner. Loads lazily; `available` reports whether the
    model actually loaded (handles missing deps, missing network access to
    the Hugging Face Hub, or an unsupported model id gracefully)."""

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self.available = self._try_load()

    def _try_load(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as e:  # broad: covers ImportError and broken/partial installs (e.g. missing shared libs)
            logger.warning("torch is not usable in this environment (%s); VLM captioning unavailable. Use HeuristicCaptioner.", e)
            return False

        try:
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            return True
        except Exception as e:  # network errors, gated repo, disk space, etc.
            logger.warning("Could not load VLM captioner '%s': %s. Falling back to heuristic captions.", self.model_id, e)
            return False

    def _raw_caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        import torch

        inputs = (
            self._processor(image, prompt, return_tensors="pt")
            if prompt
            else self._processor(image, return_tensors="pt")
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=60)
        return self._processor.decode(out[0], skip_special_tokens=True)

    def caption(self, image_path: "str | Path") -> StructuredCaption:
        if not self.available:
            raise RuntimeError("VLMCaptioner is not available; use HeuristicCaptioner instead.")

        img = Image.open(image_path).convert("RGB")
        base_caption = self._raw_caption(img)

        # Ask targeted follow-up questions to populate structured fields.
        # BLIP (base captioning model) doesn't do VQA, so for fields it can't
        # answer we degrade gracefully to "unknown" rather than fabricating.
        structured = StructuredCaption(
            subject=base_caption,
            composition="unknown",
            pose="unknown",
            clothing="unknown",
            lighting="unknown",
            environment="unknown",
            visual_characteristics="unknown",
            raw_caption=base_caption,
            source="vlm",
        )
        return structured


def get_captioner(prefer_vlm: bool = True, model_id: str = "Salesforce/blip-image-captioning-large", device: str = "cpu"):
    """Return the best available captioner, preferring a real VLM but never
    failing outright if one cannot be loaded."""
    if prefer_vlm:
        vlm = VLMCaptioner(model_id=model_id, device=device)
        if vlm.available:
            return vlm
        logger.info("Falling back to HeuristicCaptioner (no VLM available in this environment).")
    return HeuristicCaptioner()
