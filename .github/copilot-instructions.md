# REPR_CUR — Copilot guide for AI coding agents

This repo trains a JEPA-augmented decoder-only transformer via LightningCLI. Use this as your quick map to the moving parts, how to run, and the house rules that matter here.

## Current Implementation: Minimal LeJEPA

**The minimal LeJEPA approach has replaced the original complex implementation as the default.**

- Location: `src/models/jepa.py`, `src/lightning/lit_jepa.py`
- Philosophy: Strip to essentials following LeJEPA paper principles
- Core: Simple prediction loss + one geometric regularizer (VICReg or SIGReg)
- Config: `configs/tiny_jepa.yaml`, `configs/wikitext.yaml`
- Documentation: See `MINIMAL_LEJEPA.md` for full details
- Backups: Original complex version saved as `*_backup.py` files

### What Changed
- **Removed**: EMA teacher, Soft-NCE, κ-modulation, cycle/composition losses, gradient barriers, complex scheduling
- **Kept**: Simple prediction (cosine/L2) + geometry regularization (VICReg/SIGReg)
- **Simplified**: From 20+ hyperparameters to 2 core knobs (α and λ)

## Big picture (Current Minimal LeJEPA)
- Entry point: `src/train.py` wires `LitJEPA` + `UnifiedDataModule` into LightningCLI
- Core model: `src/models/transformer.py` implements an autoregressive stack with RMSNorm, SwiGLU MLPs, and RoPE
- JEPA objective: `src/models/jepa.py` provides projector, predictor, horizon embeddings, and geometric regularizers
- Geometric regularizers: `src/utils/vicreg.py` contains VICReg and SIGReg (isotropic Gaussian via random projections)
- Utilities: `src/utils/sampling.py` handles horizon pair sampling

## Data and tokenization
- Use `UnifiedDataModule` (`src/data/unified.py`) with `data.module` set to an alias or class path. Aliases resolve via `src/data/registry.py` (e.g., `tiny_shakespeare`, `wikitext2`, `wiki`, `wt2`, `tiny`).
- Each datamodule must set `self.tokenizer` and `self.vocab_size` in `setup`; `UnifiedDataModule` forwards these to the model for auto vocab resizing.
- Tiny Shakespeare uses `configs/tiny_jepa.yaml:data.download_url`; WikiText2 streams via Hugging Face and trains a cached BPE (`src/data/hf_tokenizer.py`). For quick runs on WikiText2, lower `data.train_fraction`.

## How to run, log, and debug
- Train: `python -m src.train fit --config configs/tiny_jepa.yaml` (swap configs to change dataset/model sizes).
- Logging: both TB and W&B are enabled. To avoid W&B in CI/offline, set `WANDB_MODE=disabled` or override `--trainer.logger` in the YAML.
- Precision/devices/strategy/grad-clip are controlled under the `trainer:` section in configs. Optional speed-ups: set `TORCH_COMPILE=1` (CUDA) and TF32 is auto-enabled when available.

## What the trainer does for you (`src/lightning/lit_jepa.py`)
- Minimal loss combining: `total_loss = lm_loss + alpha * jepa_loss` where `jepa_loss = (1-lambda)*pred + lambda*geom`
- JEPA tap: Tap with `jepa.tap_layer` (negatives allowed) to extract representations from intermediate layers
- Optimizer grouping: heads (JEPA projector/predictor, horizon emb, LM head or tied token emb) get `lr_head`; backbone gets `lr`. Weight decay applies only to >=2D params. LR schedule = warmup+cosine per-step.
- Auto vocab resize on `on_fit_start`: when the datamodule exposes `vocab_size`, token embedding and LM head are rebuilt and weight tying extras reinitialized.

## JEPA, horizons, and diagnostics
- Pair sampling: `src/utils/sampling.py::sample_anchor_target_pairs` draws anchors/targets using `jepa.horizons`, `jepa.horizon_probs`, and `jepa.pairs_per_seq`.
- Losses: JEPA combines prediction loss (cosine or L2) and geometric regularization (VICReg or SIGReg).
- Validation logs: PPL, JEPA loss, prediction and geometry components, variance/covariance metrics; validation pairs are cached per epoch for determinism.

## Tests you should keep green
- Run: `pytest -q tests/test_preflight.py`.
- Covered invariants: horizon distribution, RoPE causality/shape checks, VICReg/SIGReg regularizers, minimal JEPA forward pass, and a CUDA BF16 forward/backward smoke test (auto-skips on non‑Ampere GPUs).

## Conventions and extension points
- Treat YAML configs as source of truth; prefer adding knobs to configs rather than hardcoding.
- Model input is token IDs shaped `(batch, seq)` (dtype long). Dataloaders already return longs; don’t normalize inside the model.
- Adding a dataset: implement a LightningDataModule in `src/data/<name>.py`, ensure Lightning-friendly constructor args, set `vocab_size`/`tokenizer` in `setup`, and it will be discoverable via `registry.py` aliasing.
- Baseline runs: set `jepa.run_baseline: true` to disable JEPA and train LM-only with the same stack.

Notes and pitfalls
- W&B is on by default; set `WANDB_API_KEY` or disable it to avoid hangs in CI.
- If you enable `weight_tying: true`, keep tokenizer vocab fixed across runs or rely on the auto-resize at fit start.

## Known issues and quick workarounds
- JEPA loss can be noisy/unstable on small corpora (e.g., Tiny Shakespeare) with the current settings (off_diag_scale=0.005, pairs_per_seq=64, kl_future_weight=0.25). We’re investigating; meanwhile:
	- Try `jepa.off_diag_scale: 0.02` and/or reduce `jepa.pairs_per_seq` to 32–48 to stabilize batch-normalized Barlow stats.
	- Start with a higher EMA (e.g., `ema_schedule: [0.995, 0.9998]`) and ensure `jepa.tap_norm: true`, `jepa.grad_barrier: true`.
	- Lower `optimizer.lr_head` relative to `optimizer.lr` (e.g., 0.5x) or briefly freeze `latent_to_hidden` for the first few hundred steps.
- Rising `kl_future` over training: the current KL compares student logits from `latent_to_hidden(z_pred)` (aligned to hidden at tpos) against teacher “next-token” logits gathered at `tpos-1`. This misaligns targets (student ≈ token at tpos+1 vs. teacher ≈ token at tpos), which can inflate KL even as features improve.
	- Quick mitigation: set `jepa.kl_future_weight: 0.0` until alignment is fixed, or switch the teacher gather to `tpos` to compare like-for-like next-token distributions.
	- Also consider temperature-tuning via `logit_scale` (with weight tying) and monitoring `val/top1_acc` and `val/margin` to verify JEPA progress when KL is disabled.

### KL future metric: why it can rise and how to read it
- KL is computed as KL(p_teacher || q_student) = CE − H. As the LM teacher sharpens during training, its entropy H drops. Even if the student improves (CE decreases), KL can still increase because H shrinks faster. That’s a metric/teacher effect, not necessarily a bug.
- Log CE and teacher entropy alongside KL to see the right trend:
	- ce_future = −E_p_teacher[log q_student]
	- teacher_entropy = −E_p_teacher[log p_teacher]
	- kl_future = ce_future − teacher_entropy
- Use a soft teacher with temperature T>1 during both training and validation for stability, and scale KL by T^2 (Hinton distillation):
	- Add knobs: `jepa.kl_temp: 2.0` (try 2–4), `jepa.kl_future_weight: 0.25–0.5`.
	- Freeze LM head weights in the KL path so grads only tune the latent→hidden mapper:
		- Implement a helper (pseudo): student_logits = linear(h_latent, lm_head.weight.detach(), bias=None); apply detached logit_scale/output_bias if tied.
	- In validation, log `ce_future`, `teacher_entropy`, and `kl_future` (with T and T^2 scaling) using the same pairs as training.
- Sanity checklist for KL path:
	- Detach teacher logits; detach LM head (student side) or freeze it; use the same temperature on both sides; scale KL by T^2; warm up `kl_future_weight` if the head is unstable early.

