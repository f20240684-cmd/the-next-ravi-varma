"""The Next Ravi Varma -- a LoRA-based generative art pipeline inspired by
Raja Ravi Varma's paintings.

This package intentionally keeps heavy ML dependencies (torch, diffusers,
transformers, peft) as *optional* imports throughout, so that lightweight
parts of the system (dataset validation, prompt expansion, configuration)
work even on machines without a GPU or without those packages installed.
See `src/ravi_varma/utils/device.py` for the central capability-detection
logic used across the codebase.
"""

__version__ = "0.1.0"
