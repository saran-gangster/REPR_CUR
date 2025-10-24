# REPR_CUR — Copilot guide for AI coding agents

This repo trains a JEPA-augmented decoder-only transformer via LightningCLI. Use this as your quick map to the moving parts, how to run, and the house rules that matter here.

## Big picture
- Entry point: `src/train.py` wires `LitJEPA` + `UnifiedDataModule` into LightningCLI and injects both TensorBoard and W&B loggers by default.
- Core model: `src/models/transformer.py` implements an autoregressive stack with RMSNorm, SwiGLU MLPs, RoPE, and a gated `lm_bridge` residual path. Weight tying optionally exposes `logit_scale` and `output_bias`.
- JEPA objective: `src/models/jepa.py` provides online/EMA towers, predictor, horizon embeddings, and VICReg-style regularizers. Tap features come from an intermediate transformer layer.
- Utilities: `src/utils/{sampling.py,vicreg.py,ema.py}` contain horizon pair sampling, variance/covariance penalties, and EMA update helpers.

## Data and tokenization
- Use `UnifiedDataModule` (`src/data/unified.py`) with `data.module` set to an alias or class path. Aliases resolve via `src/data/registry.py` (e.g., `tiny_shakespeare`, `wikitext2`, `wiki`, `wt2`, `tiny`).
- Each datamodule must set `self.tokenizer` and `self.vocab_size` in `setup`; `UnifiedDataModule` forwards these to the model for auto vocab resizing.
- Tiny Shakespeare uses `configs/tiny_jepa.yaml:data.download_url`; WikiText2 streams via Hugging Face and trains a cached BPE (`src/data/hf_tokenizer.py`). For quick runs on WikiText2, lower `data.train_fraction`.

## How to run, log, and debug
- Train: `python -m src.train fit --config configs/tiny_jepa.yaml` (swap configs to change dataset/model sizes).
- Logging: both TB and W&B are enabled. To avoid W&B in CI/offline, set `WANDB_MODE=disabled` or override `--trainer.logger` in the YAML.
- Precision/devices/strategy/grad-clip are controlled under the `trainer:` section in configs. Optional speed-ups: set `TORCH_COMPILE=1` (CUDA) and TF32 is auto-enabled when available.

## What the trainer does for you (`src/lightning/lit_jepa.py`)
- JEPA tap and scheduling:
	- Tap with `jepa.tap_layer` (negatives allowed); set `jepa.grad_barrier: true` to decouple grads below the tap; optional `jepa.tap_norm`.
	- EMA momentum can cosine-ramp via `jepa.ema_schedule` + `jepa.ema_schedule_steps`; JEPA momentum is updated every train batch.
	- `jepa.lm_weight` can warm up if `jepa.lm_weight_use_scheduler: true` (defaults on unless `run_baseline`).
- Optimizer grouping: heads (JEPA projector/predictor, horizon emb, LM head or tied token emb) get `lr_head`; backbone gets `lr`. Weight decay applies only to >=2D params. LR schedule = warmup+cosine per-step.
- Gradient clipping is split at the tap: lower (token emb, blocks ≤ tap, norm_tap, JEPA heads) and upper (bridge/norm_f/lm_head) are clipped separately.
- Auto vocab resize on `on_fit_start`: when the datamodule exposes `vocab_size`, token embedding and LM head are rebuilt and weight tying extras reinitialized.

## JEPA, horizons, and diagnostics
- Pair sampling: `src/utils/sampling.py::sample_anchor_target_pairs` draws anchors/targets using `jepa.horizons`, `jepa.horizon_probs`, and `jepa.pairs_per_seq`.
- Losses: JEPA combines alignment/Barlow-style terms; optional future LM KL (`kl_future_weight`) compares predicted-latent-derived logits to teacher next-token logits.
- Validation logs: PPL, JEPA loss, alignment/Barlow metrics, std gap, retrieval Top‑1 and margin; validation pairs are cached per epoch for determinism.

## Tests you should keep green
- Run: `pytest -q tests/test_preflight.py`.
- Covered invariants: horizon distribution, EMA update, RoPE causality/shape checks, and a CUDA BF16 forward/backward smoke test (auto-skips on non‑Ampere GPUs).

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
  