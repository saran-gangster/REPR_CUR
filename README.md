# Unified EXIT-JEPA (Core JEPA-only Prototype)

This repo trains a JEPA objective on a standard decoder-only Transformer using Tiny Shakespeare.
No EXIT sampling, no internal scratchpad yet — perfect for quick iteration.

## Quickstart

1) Install
   pip install -r requirements.txt

2) Train (PyTorch Lightning CLI)
   python -m src.train --config configs/tiny_jepa.yaml

3) TensorBoard
   tensorboard --logdir lightning_logs

## Notes
- JEPA predicts future hidden states in a latent space with cosine + VICReg regularization.
- Target projector is an EMA of the online projector (BYOL-style).
- Transformer uses RoPE, RMSNorm, and SwiGLU.