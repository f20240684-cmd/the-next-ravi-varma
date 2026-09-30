"""LoRA fine-tuning of a Stable Diffusion (1.5 or XL) UNet using Diffusers +
PEFT + Accelerate.

This mirrors the structure of Diffusers' official `train_text_to_image_lora`
examples, wrapped in a class so it can be driven both from
`scripts/train_lora.py` and from the Colab notebook, and adapted to:
  * load its own dataset (`RaviVarmaImageCaptionDataset`) built from our
    JSONL manifest,
  * inject the configurable style trigger token,
  * respect the shared `MemoryOptimizationPlan` from `utils/device.py`,
  * checkpoint/resume/validate via `training/callbacks.py`.

All heavy imports (torch, diffusers, peft, accelerate, bitsandbytes,
xformers) are performed lazily inside methods, never at module import time,
so `from ravi_varma.training.lora_trainer import LoraTrainer` succeeds even
on a machine without the ML stack installed (e.g. to unit-test config
handling). Actually calling `.train()` without those packages installed
raises a clear `RuntimeError` rather than failing with a confusing
`ImportError` deep in the call stack.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from ravi_varma.config import LoraTrainingConfig
from ravi_varma.data.dataset import RaviVarmaImageCaptionDataset, load_manifest
from ravi_varma.training.callbacks import CheckpointManager, LossLogger, ValidationImageCallback
from ravi_varma.utils.device import detect_device, resolve_optimizations
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_PACKAGES = ("torch", "diffusers", "transformers", "peft", "accelerate")


def check_ml_stack_available() -> Optional[str]:
    """Return None if everything needed for real training is importable,
    otherwise a human-readable message naming what's missing (or broken --
    e.g. a partial/corrupted install missing shared libraries)."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception as e:  # broad: broken installs raise OSError/RuntimeError, not just ImportError
            missing.append(f"{pkg} ({e.__class__.__name__}: {e})")
    if missing:
        return (
            "Cannot run LoRA training: missing or broken packages "
            f"{missing}. Install requirements.txt cleanly on a GPU machine or Colab (see "
            "notebooks/03_lora_training.ipynb)."
        )
    return None


class LoraTrainer:
    def __init__(self, config: LoraTrainingConfig):
        self.config = config
        self.device_info = detect_device(preferred="auto")
        self.plan = resolve_optimizations(
            self.device_info,
            requested_mixed_precision=config.training.mixed_precision,
            requested_xformers=config.training.enable_xformers,
        )
        self._accelerator = None
        self._unet = None
        self._text_encoder = None
        self._vae = None
        self._tokenizer = None
        self._noise_scheduler = None
        self._pipeline_cls = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_models(self):
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        cfg = self.config
        kwargs = {"revision": cfg.revision}
        if cfg.variant:
            kwargs["variant"] = cfg.variant

        if cfg.model_family == "sdxl":
            # SDXL has two text encoders/tokenizers; kept as a documented
            # extension point -- the primary supported path is SD1.5.
            raise NotImplementedError(
                "SDXL training path is scaffolded via configs/lora_sdxl.yaml but requires "
                ">= 16GB VRAM and the dual text-encoder handling in Diffusers' "
                "train_text_to_image_lora_sdxl.py. Contributions welcome -- see README."
            )

        self._tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", **kwargs)
        self._text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder", **kwargs)
        self._vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae", **kwargs)
        self._unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet", **kwargs)
        self._noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
        self._pipeline_cls = StableDiffusionPipeline

    def _apply_lora(self):
        from peft import LoraConfig as PeftLoraConfig

        lora_cfg = self.config.lora
        unet_lora_config = PeftLoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
        )
        self._vae.requires_grad_(False)
        self._text_encoder.requires_grad_(False)
        self._unet.requires_grad_(False)
        self._unet.add_adapter(unet_lora_config)

        trainable_params = [p for p in self._unet.parameters() if p.requires_grad]

        if lora_cfg.train_text_encoder:
            text_lora_config = PeftLoraConfig(
                r=lora_cfg.rank,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self._text_encoder.add_adapter(text_lora_config)
            trainable_params += [p for p in self._text_encoder.parameters() if p.requires_grad]

        return trainable_params

    def _build_optimizer(self, trainable_params):
        opt_cfg = self.config.optimizer
        optimizer_cls = None
        if opt_cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("use_8bit_adam=True but bitsandbytes is not installed; using standard AdamW.")
        if optimizer_cls is None:
            import torch

            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_params,
            lr=opt_cfg.learning_rate,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            weight_decay=opt_cfg.adam_weight_decay,
            eps=opt_cfg.adam_epsilon,
        )

    def print_startup_summary(self, dataset_size: int, trainable_params: list, total_params: int) -> None:
        trainable_count = sum(p.numel() for p in trainable_params)
        pct = 100 * trainable_count / max(total_params, 1)
        logger.info("=== The Next Ravi Varma -- LoRA Training ===")
        logger.info("Device: %s (%s)", self.device_info.device, self.device_info.gpu_name or "n/a")
        if self.device_info.total_vram_gb:
            logger.info("VRAM: %.1f GB", self.device_info.total_vram_gb)
        logger.info("Dataset size: %d samples", dataset_size)
        logger.info("Trainable parameters: %d (%.3f%% of %d total)", trainable_count, pct, total_params)
        logger.info("Mixed precision: %s | xFormers: %s", self.plan.mixed_precision, self.plan.use_xformers)
        logger.info("Config: %s", self.config.model_dump())

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self) -> Path:
        missing = check_ml_stack_available()
        if missing:
            raise RuntimeError(missing)

        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        cfg = self.config
        set_seed(cfg.training.seed)

        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=self.plan.mixed_precision if self.plan.mixed_precision != "no" else "no",
            log_with="tensorboard" if cfg.logging.report_to == "tensorboard" else None,
            project_dir=cfg.logging.logging_dir,
        )
        self._accelerator = accelerator

        self._load_models()
        trainable_params = self._apply_lora()

        if cfg.training.gradient_checkpointing:
            self._unet.enable_gradient_checkpointing()
        if self.plan.use_xformers:
            try:
                self._unet.enable_xformers_memory_efficient_attention()
            except Exception as e:  # pragma: no cover - hardware dependent
                logger.warning("Could not enable xFormers attention: %s", e)

        manifest = load_manifest(cfg.dataset.manifest_file)
        if not manifest:
            raise RuntimeError(
                f"Manifest {cfg.dataset.manifest_file} is empty. Run scripts/prepare_dataset.py first."
            )
        dataset = RaviVarmaImageCaptionDataset(
            manifest,
            resolution=cfg.dataset.resolution,
            center_crop=cfg.dataset.center_crop,
            random_flip=cfg.dataset.random_flip,
        )

        def collate_fn(examples):
            pixel_values = torch.stack([e["pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
            input_ids = self._tokenizer(
                [e["text"] for e in examples],
                padding="max_length",
                truncation=True,
                max_length=self._tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )

        optimizer = self._build_optimizer(trainable_params)

        steps_per_epoch = math.ceil(len(dataloader) / cfg.training.gradient_accumulation_steps)
        max_train_steps = cfg.training.max_train_steps
        if cfg.training.num_train_epochs:
            max_train_steps = steps_per_epoch * cfg.training.num_train_epochs

        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.optimizer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.optimizer.lr_warmup_steps * cfg.training.gradient_accumulation_steps,
            num_training_steps=max_train_steps * cfg.training.gradient_accumulation_steps,
        )

        self._unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            self._unet, optimizer, dataloader, lr_scheduler
        )
        weight_dtype = torch.float16 if self.plan.mixed_precision == "fp16" else torch.float32
        self._vae.to(accelerator.device, dtype=weight_dtype)
        self._text_encoder.to(accelerator.device, dtype=weight_dtype)

        total_params = sum(p.numel() for p in self._unet.parameters())
        self.print_startup_summary(len(dataset), trainable_params, total_params)

        checkpoint_manager = CheckpointManager(cfg.checkpointing.output_dir, cfg.checkpointing.checkpoints_total_limit)
        loss_logger = LossLogger(cfg.logging.logging_dir, cfg.logging.report_to)

        global_step = 0
        resume_path = checkpoint_manager.resolve_resume_path(cfg.checkpointing.resume_from_checkpoint)
        if resume_path is not None:
            accelerator.load_state(str(resume_path))
            global_step = int(str(resume_path).split("-")[-1])
            logger.info("Resumed from checkpoint %s at step %d", resume_path, global_step)

        progress_start = time.time()
        self._unet.train()
        done = False
        while not done:
            for batch in dataloader:
                with accelerator.accumulate(self._unet):
                    latents = self._vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * self._vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()
                    noisy_latents = self._noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = self._text_encoder(batch["input_ids"].to(accelerator.device))[0]

                    model_pred = self._unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self._noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self._noise_scheduler.config.prediction_type == "v_prediction":
                        target = self._noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type {self._noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % cfg.logging.log_every_n_steps == 0:
                        elapsed = time.time() - progress_start
                        loss_logger.log_scalar("train/loss", loss.detach().item(), global_step)
                        loss_logger.log_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                        logger.info(
                            "step %d/%d | loss %.4f | %.1fs elapsed",
                            global_step, max_train_steps, loss.detach().item(), elapsed,
                        )

                    if accelerator.is_main_process and global_step % cfg.checkpointing.checkpointing_steps == 0:
                        checkpoint_manager.save(accelerator, global_step)

                    if (
                        accelerator.is_main_process
                        and cfg.validation.validation_steps
                        and global_step % cfg.validation.validation_steps == 0
                    ):
                        self._run_validation(accelerator, cfg, global_step, loss_logger)

                    if global_step >= max_train_steps:
                        done = True
                        break

        accelerator.wait_for_everyone()
        final_path = self._save_final_lora(accelerator, cfg)
        loss_logger.close()
        return final_path

    def _run_validation(self, accelerator, cfg: LoraTrainingConfig, step: int, loss_logger: LossLogger) -> None:
        try:
            pipeline = self._pipeline_cls.from_pretrained(
                cfg.base_model,
                unet=accelerator.unwrap_model(self._unet),
                text_encoder=self._text_encoder,
                vae=self._vae,
                tokenizer=self._tokenizer,
                safety_checker=None,
            ).to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            callback = ValidationImageCallback(
                cfg.validation.validation_prompt,
                cfg.validation.num_validation_images,
                Path(cfg.logging.logging_dir) / "validation_images",
            )
            callback.run(pipeline, step, tb_writer=getattr(loss_logger, "writer", None), seed=cfg.training.seed)
            del pipeline
        except Exception as e:  # pragma: no cover - hardware dependent
            logger.warning("Validation image generation failed at step %d: %s", step, e)

    def _save_final_lora(self, accelerator, cfg: LoraTrainingConfig) -> Path:
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        output_dir = Path(cfg.checkpointing.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(self._unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        self._pipeline_cls.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        # Persist the exact config used, so `generate.py` can sanity-check
        # base_model / trigger_token compatibility later.
        (output_dir / "training_config.yaml").write_text(
            __import__("yaml").safe_dump(cfg.model_dump()), encoding="utf-8"
        )
        logger.info("Saved final LoRA weights -> %s", output_dir)
        return output_dir
