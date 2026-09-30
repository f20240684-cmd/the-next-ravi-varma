"""ControlNet structural conditioning.

Canny edge conditioning only needs OpenCV (always available in this repo's
dependency set) and works with zero extra downloads. OpenPose conditioning
needs `controlnet_aux` (wraps a small pose-estimation network) plus network
access to download that detector the first time; if it isn't available, we
tell the caller clearly rather than silently doing nothing.

The actual ControlNet-conditioned diffusion pass lives in
`generation/pipeline.py`; this module only produces the conditioning image.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from PIL import Image

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

ControlType = Literal["openpose", "canny"]


def make_canny_conditioning(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Pure-OpenCV Canny edge map, always available offline."""
    arr = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, low_threshold, high_threshold)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


class OpenPoseDetector:
    """Lazy wrapper around `controlnet_aux.OpenposeDetector`. `available`
    reports whether the detector could actually be loaded (package
    installed + weights reachable)."""

    def __init__(self):
        self._detector = None
        self.available = self._try_load()

    def _try_load(self) -> bool:
        try:
            from controlnet_aux import OpenposeDetector

            self._detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return True
        except Exception as e:
            logger.warning(
                "OpenPose detector unavailable (%s). Install `controlnet_aux` and ensure network "
                "access to the Hugging Face Hub, or use ControlNet type='canny' instead.", e,
            )
            return False

    def detect(self, image: Image.Image) -> Image.Image:
        if not self.available:
            raise RuntimeError("OpenPose detector is not available in this environment.")
        return self._detector(image)


def make_conditioning_image(
    image: Image.Image,
    control_type: ControlType = "canny",
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
) -> Image.Image:
    """Dispatch to the requested conditioning method. Falls back to Canny
    (which has no extra dependencies) if OpenPose is requested but the
    detector cannot be loaded, and clearly logs the fallback."""
    if control_type == "canny":
        return make_canny_conditioning(image, canny_low_threshold, canny_high_threshold)

    if control_type == "openpose":
        detector = OpenPoseDetector()
        if detector.available:
            return detector.detect(image)
        logger.warning("Falling back to Canny conditioning because OpenPose is unavailable.")
        return make_canny_conditioning(image, canny_low_threshold, canny_high_threshold)

    raise ValueError(f"Unknown control_type: {control_type}")


CONTROLNET_MODEL_IDS = {
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}
