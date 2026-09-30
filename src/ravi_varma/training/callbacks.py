"""Training-time callbacks: checkpoint save/resume bookkeeping and periodic
validation-image generation. Kept separate from `lora_trainer.py` so the
training loop itself stays readable.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Saves numbered checkpoints under `output_dir/checkpoint-<step>` and
    prunes old ones beyond `checkpoints_total_limit`."""

    def __init__(self, output_dir: "str | Path", checkpoints_total_limit: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_total_limit = checkpoints_total_limit

    def checkpoint_dirs(self) -> list[Path]:
        dirs = [p for p in self.output_dir.glob("checkpoint-*") if p.is_dir()]
        return sorted(dirs, key=lambda p: int(p.name.split("-")[-1]))

    def latest_checkpoint(self) -> Optional[Path]:
        dirs = self.checkpoint_dirs()
        return dirs[-1] if dirs else None

    def resolve_resume_path(self, resume_from_checkpoint: Optional[str]) -> Optional[Path]:
        if not resume_from_checkpoint:
            return None
        if resume_from_checkpoint == "latest":
            latest = self.latest_checkpoint()
            if latest is None:
                logger.info("resume_from_checkpoint='latest' requested but no checkpoints found; starting fresh.")
            return latest
        path = Path(resume_from_checkpoint)
        if not path.exists():
            logger.warning("Requested resume checkpoint %s does not exist; starting fresh.", path)
            return None
        return path

    def save(self, accelerator, step: int) -> Path:
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(ckpt_dir))
        logger.info("Saved checkpoint at step %d -> %s", step, ckpt_dir)
        self._prune()
        return ckpt_dir

    def _prune(self) -> None:
        dirs = self.checkpoint_dirs()
        excess = len(dirs) - self.checkpoints_total_limit
        for old_dir in dirs[:max(excess, 0)]:
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.info("Pruned old checkpoint %s (checkpoints_total_limit=%d)", old_dir, self.checkpoints_total_limit)


class ValidationImageCallback:
    """Generates a handful of sample images at `validation_steps` intervals
    so training progress can be inspected visually / in TensorBoard."""

    def __init__(self, prompt: str, num_images: int, output_dir: "str | Path"):
        self.prompt = prompt
        self.num_images = num_images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, pipeline, step: int, tb_writer=None, seed: int = 0) -> list[Path]:
        import torch

        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        saved = []
        for i in range(self.num_images):
            image = pipeline(self.prompt, generator=generator, num_inference_steps=25).images[0]
            path = self.output_dir / f"val_step{step}_{i}.png"
            image.save(path)
            saved.append(path)
            if tb_writer is not None:
                import numpy as np

                tb_writer.add_image(f"validation/{i}", np.asarray(image).transpose(2, 0, 1), global_step=step)
        logger.info("Saved %d validation images at step %d -> %s", len(saved), step, self.output_dir)
        return saved


class LossLogger:
    """Thin wrapper around TensorBoard's SummaryWriter that degrades to
    plain logging if tensorboard is not installed."""

    def __init__(self, logging_dir: "str | Path", report_to: str = "tensorboard"):
        self.writer = None
        if report_to == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                Path(logging_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(logging_dir))
            except Exception:
                logger.warning("tensorboard not installed or not usable; falling back to console-only loss logging.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        logger.info("step=%d %s=%.5f", step, tag, value)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
