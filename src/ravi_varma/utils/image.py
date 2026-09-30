"""Image utilities used by dataset validation/preprocessing and ControlNet
conditioning. Depends only on Pillow + numpy (both lightweight, always
installed), never on torch/diffusers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

RESAMPLE = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


def safe_open_image(path: "str | Path") -> Optional[Image.Image]:
    """Open an image, returning None (never raising) if it is corrupt,
    truncated, or not a recognizable image format."""
    try:
        img = Image.open(path)
        img.load()  # force-read pixel data to catch truncated files
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def get_resolution(path: "str | Path") -> Optional[tuple[int, int]]:
    img = safe_open_image(path)
    if img is None:
        return None
    return img.size  # (width, height)


def perceptual_hash(path: "str | Path", hash_size: int = 8) -> Optional[str]:
    """A minimal average-hash (aHash) implementation for near-duplicate
    detection, avoiding an extra dependency like `imagehash`."""
    img = safe_open_image(path)
    if img is None:
        return None
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    # Pack bits into a hex string.
    bit_str = "".join("1" if b else "0" for b in bits)
    return f"{int(bit_str, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    int_a, int_b = int(hash_a, 16), int(hash_b, 16)
    return bin(int_a ^ int_b).count("1")


def file_sha256(path: "str | Path", chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resize_for_training(
    img: Image.Image,
    target_resolution: int = 512,
    mode: Literal["center_crop", "pad", "stretch"] = "center_crop",
    interpolation: str = "lanczos",
) -> Image.Image:
    """Resize/crop an image to a square `target_resolution`, matching the
    strategy used by most Diffusers LoRA training scripts."""
    resample = RESAMPLE.get(interpolation, Image.LANCZOS)
    w, h = img.size

    if mode == "stretch":
        return img.resize((target_resolution, target_resolution), resample)

    if mode == "pad":
        scale = target_resolution / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), resample)
        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        canvas.paste(resized, ((target_resolution - new_w) // 2, (target_resolution - new_h) // 2))
        return canvas

    # center_crop (default): scale so the shortest side == target, then crop.
    scale = target_resolution / min(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), resample)
    left = (new_w - target_resolution) // 2
    top = (new_h - target_resolution) // 2
    return resized.crop((left, top, left + target_resolution, top + target_resolution))
