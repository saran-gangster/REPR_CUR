# REPR_CUR copilot guide

## Architecture overview
- Core trainer lives in `src/lightning/lit_jepa.py`, wrapping a `DecoderOnlyTransformer` with the JEPA objective.
- `src/models/transformer.py` defines the autoregressive stack with RMSNorm, SwiGLU MLPs, RoPE, and a gated "lm_bridge" residual.
- `src/models/jepa.py` houses the latent predictor, EMA teacher, and VICReg-style regularizers fed from transformer tap layers.
- `src/utils` contains shared helpers: horizon sampling, variance/covariance penalties, and EMA updates used across training.

## Data pipeline
- `src/train.py` instantiates `UnifiedDataModule`, which resolves dataset aliases via `src/data/registry.py` (e.g. `tiny_shakespeare`, `wikitext2`).
- Dataset modules expose Hugging Face BPE tokenizers for vocab sizing; `UnifiedDataModule` forwards tokenizer + `vocab_size` back to the model on `setup`.
- Tiny Shakespeare downloads raw text from `configs/tiny_jepa.yaml::data.download_url`; WikiText streams from `datasets.load_dataset` and trains a cached BPE.
- Most loader kwargs (block size, workers, fractions) are declared in module signatures so LightningCLI validation passes without custom parsing.

## Training workflow
- Run training with LightningCLI, e.g. `python -m src.train fit --config configs/tiny_jepa.yaml`; swap configs to target other datasets.
- The CLI injects both TensorBoard and WandB loggers; set `WANDB_MODE=disabled` or override `trainer.logger` in YAML if offline.
- Grad clip, precision, devices, and strategy are controlled through the `trainer` section inside the config files.
- Optimizer hyperparameters sit under `model.optimizer`; `lr_head` is respected inside `LitJEPA.configure_optimizers` (see file for parameter grouping when extending).
- Model vocabulary automatically resizes on `on_fit_start` when the datamodule exposes `vocab_size`, so regenerate tokenizers before changing configs.

## Model & objectives
- JEPA taps an intermediate transformer layer (`model.jepa.tap_layer`, negative indices allowed) and can detach gradients below the tap via `grad_barrier`.
- Horizon sampling is configured with `horizons`, `horizon_probs`, and `pairs_per_seq`; these feed `sample_anchor_target_pairs` to balance short/long predictions.
- EMA momentum can follow a cosine schedule (`ema_schedule`, `ema_schedule_steps`), and `lm_weight` may ramp using warmup controls if `lm_weight_use_scheduler`.
- Weight tying adds a learnable temperature (`logit_scale`) and bias; ensure `weight_tying: true` only when tokenizer vocab will stay fixed.
- JEPA losses combine InfoNCE with VICReg terms; adjust `gamma_var`/`gamma_cov` or the predictor regularizers (`gamma_var_pred`, `gamma_cov_pred`) when tuning collapse behavior.
- Baseline LM-only runs set `run_baseline: true`, zeroing JEPA weights but still using the same transformer stack for ablations.

## Testing & validation
- Preflight coverage lives in `tests/test_preflight.py`; run `pytest -q tests/test_preflight.py` from repo root before committing changes.
- Horizon sampling, EMA updates, RoPE causality, and a BF16 smoke test are asserted; the mixed-precision test skips automatically when CUDA BF16 is unavailable.
- Keep tests deterministic by seeding via `_set_seed`; new tests should follow the same helper to avoid flaky randomness.

## Conventions & tips
- Treat configs as the single source of truth: prefer adding knobs there rather than hardcoding defaults inside modules.
- When adding datamodules, expose Lightning-friendly constructor args and register aliases via `registry._available_datamodule_classes` patterns.
- The transformer expects token IDs shaped `(batch, seq)`; dataloaders already return long tensors, so normalize inputs upstream instead of inside the model.
- Avoid breaking `UnifiedDataModule.setup` contract—ensure new datamodules set `self.tokenizer` and `self.vocab_size` for automatic resizing.
- For fast experiments, lower `train_fraction` on WikiText2; the streaming loader keeps memory usage manageable without manual chunking.
- W&B logging is optional but enabled by default; remember to set `WANDB_API_KEY` or disable it in configs to prevent hangs in CI environments.
  