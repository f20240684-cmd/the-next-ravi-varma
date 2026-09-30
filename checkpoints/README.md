# `checkpoints/lora/`

Trained LoRA weights are saved here by `scripts/train_lora.py`, e.g.:

```
checkpoints/lora/ravi_varma/
├── pytorch_lora_weights.safetensors
├── training_config.yaml       # exact config used to produce this checkpoint
└── checkpoint-<step>/         # intermediate resumable checkpoints (accelerator state)
```

This directory is intentionally excluded from version control (see `.gitignore`) -- checkpoints
are large binary artifacts that belong in a model registry, cloud storage bucket, or the Hugging
Face Hub, not in git.

**If this directory is empty**, `scripts/generate.py` and the Gradio app will still run, but will
generate with the *base* Stable Diffusion model only and will clearly say so in both the console
output and the UI -- they will never silently pretend a Ravi Varma LoRA is loaded.
