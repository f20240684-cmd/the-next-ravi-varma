"""Hardware capability detection and memory-optimization decisions.

torch is imported lazily and defensively: every function in this module
works (returning honest "unavailable" answers) even if torch is not
installed, which lets the rest of the codebase -- dataset validation,
prompt expansion, config parsing, tests -- run on a plain CPU-only
environment with no ML stack at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)


def _try_import_torch():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:  # broad: covers ImportError and broken/partial installs (e.g. missing shared libs)
        return None


@dataclass
class DeviceInfo:
    device: str  # "cuda" | "mps" | "cpu"
    torch_available: bool
    cuda_available: bool
    mps_available: bool
    gpu_name: Optional[str] = None
    total_vram_gb: Optional[float] = None
    recommended_dtype: str = "float32"

    def supports_fp16(self) -> bool:
        return self.device == "cuda"

    def supports_bf16(self) -> bool:
        torch = _try_import_torch()
        if torch is None or self.device != "cuda":
            return False
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:  # pragma: no cover - defensive
            return False


def detect_device(preferred: str = "auto") -> DeviceInfo:
    """Detect the best available compute device.

    Args:
        preferred: "auto" | "cuda" | "mps" | "cpu". If a specific device is
            requested but unavailable, falls back to the best available
            option and logs a warning.
    """
    torch = _try_import_torch()
    if torch is None:
        if preferred not in ("auto", "cpu"):
            logger.warning("torch is not installed; forcing device to 'cpu'.")
        return DeviceInfo(
            device="cpu",
            torch_available=False,
            cuda_available=False,
            mps_available=False,
            recommended_dtype="float32",
        )

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if preferred == "cuda" and not cuda_available:
        logger.warning("CUDA requested but not available; falling back.")
        preferred = "auto"
    if preferred == "mps" and not mps_available:
        logger.warning("MPS requested but not available; falling back.")
        preferred = "auto"

    if preferred == "cuda" or (preferred == "auto" and cuda_available):
        device = "cuda"
    elif preferred == "mps" or (preferred == "auto" and mps_available):
        device = "mps"
    else:
        device = "cpu"

    gpu_name = None
    total_vram_gb = None
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:  # pragma: no cover - defensive
            pass

    recommended_dtype = "float16" if device == "cuda" else "float32"

    return DeviceInfo(
        device=device,
        torch_available=True,
        cuda_available=cuda_available,
        mps_available=mps_available,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        recommended_dtype=recommended_dtype,
    )


@dataclass
class MemoryOptimizationPlan:
    """Which memory-saving features to enable for a given device+config.

    Every field defaults to the *safe* choice. `resolve_optimizations`
    downgrades anything the current hardware/library versions cannot
    actually support, rather than blindly enabling everything requested.
    """

    attention_slicing: bool = False
    vae_slicing: bool = False
    vae_tiling: bool = False
    model_cpu_offload: bool = False
    gradient_checkpointing: bool = False
    use_xformers: bool = False
    mixed_precision: str = "no"  # "fp16" | "bf16" | "no"


def resolve_optimizations(
    device_info: DeviceInfo,
    requested_attention_slicing: "bool | str" = "auto",
    requested_vae_slicing: bool = True,
    requested_vae_tiling: bool = False,
    requested_model_cpu_offload: "bool | str" = "auto",
    requested_mixed_precision: str = "fp16",
    requested_xformers: bool = True,
) -> MemoryOptimizationPlan:
    """Turn user-requested optimizations into a plan that is actually valid
    for the detected hardware, never enabling something incompatible."""
    torch = _try_import_torch()

    is_low_vram = bool(
        device_info.device == "cuda"
        and device_info.total_vram_gb is not None
        and device_info.total_vram_gb < 8.0
    )

    if requested_attention_slicing == "auto":
        attention_slicing = device_info.device in ("cuda", "mps") and (is_low_vram or device_info.device == "mps")
    else:
        attention_slicing = bool(requested_attention_slicing)

    if requested_model_cpu_offload == "auto":
        model_cpu_offload = is_low_vram
    else:
        model_cpu_offload = bool(requested_model_cpu_offload) and device_info.device == "cuda"

    mixed_precision = requested_mixed_precision
    if mixed_precision == "fp16" and not device_info.supports_fp16():
        mixed_precision = "no"
    elif mixed_precision == "bf16" and not device_info.supports_bf16():
        mixed_precision = "fp16" if device_info.supports_fp16() else "no"

    use_xformers = False
    if requested_xformers and device_info.device == "cuda" and torch is not None:
        try:
            import xformers  # noqa: F401

            use_xformers = True
        except Exception:
            use_xformers = False

    plan = MemoryOptimizationPlan(
        attention_slicing=attention_slicing,
        vae_slicing=requested_vae_slicing,
        vae_tiling=requested_vae_tiling,
        model_cpu_offload=model_cpu_offload,
        gradient_checkpointing=True,
        use_xformers=use_xformers,
        mixed_precision=mixed_precision,
    )
    logger.info(
        "Resolved memory plan for device=%s (vram=%.1fGB if known): %s",
        device_info.device,
        device_info.total_vram_gb or -1,
        plan,
    )
    return plan
