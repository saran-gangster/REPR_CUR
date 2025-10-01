import math
import os
import sys
from typing import List

import numpy as np
import torch
import pytest

# Ensure "src" is importable when running from repo root
sys.path.append(os.path.abspath("."))

from src.utils.sampling import sample_anchor_target_pairs
from src.models.jepa import JEPAObjective
from src.models.transformer import DecoderOnlyTransformer
from src.lightning.lit_jepa import LitJEPA


def _set_seed(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # BF16 generally supported on Ampere+ (compute capability >= 8.0)
    try:
        major, minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


# -----------------------------
# 1) Pair sampler invariants
# -----------------------------

def test_sampler_invariants_positions_and_shapes():
    _set_seed(42)
    B = 8
    T = 128
    pairs_per_seq = 32
    horizons = [2, 8, 32, 64]
    horizon_probs = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float)

    g = torch.Generator(device="cpu").manual_seed(123)
    b_idx, t_idx, tpos, k_ids = sample_anchor_target_pairs(
        batch_size=B,
        seq_len=T,
        pairs_per_seq=pairs_per_seq,
        horizon_values=horizons,
        horizon_probs=horizon_probs,
        device=torch.device("cpu"),
        generator=g,
    )

    N = B * pairs_per_seq
    assert b_idx.shape == (N,)
    assert t_idx.shape == (N,)
    assert tpos.shape == (N,)
    assert k_ids.shape == (N,)

    # tpos should be exactly t_idx + k (no need to clamp because sampler enforces t_idx <= T-1-k)
    k_vals = torch.tensor(horizons, dtype=torch.long)[k_ids]
    assert torch.all(tpos == (t_idx + k_vals))

    # All positions in range
    assert torch.all(t_idx >= 0)
    assert torch.all(tpos >= 0)
    assert torch.all(tpos <= T - 1)

    # Batch index counts should be uniform: each batch appears pairs_per_seq times
    counts_b = torch.bincount(b_idx, minlength=B)
    assert torch.all(counts_b == pairs_per_seq)


def test_sampler_horizon_distribution_matches_probs():
    _set_seed(123)
    # One big draw to get tight concentration without many loops (fast)
    B = 512
    T = 256
    pairs_per_seq = 512
    horizons = [2, 8, 32, 64]
    probs = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)

    g = torch.Generator(device="cpu").manual_seed(999)
    _, _, _, k_ids = sample_anchor_target_pairs(
        batch_size=B,
        seq_len=T,
        pairs_per_seq=pairs_per_seq,
        horizon_values=horizons,
        horizon_probs=torch.tensor(probs, dtype=torch.float32),
        device=torch.device("cpu"),
        generator=g,
    )

    N = B * pairs_per_seq
    obs = torch.bincount(k_ids, minlength=len(horizons)).to(torch.float64).cpu().numpy()
    p_emp = obs / float(N)

    # Each horizon should be present in a large sample
    assert np.all(obs > 100), f"Degenerate horizon counts: {obs.tolist()}"

    # Relative error per bin should be small (allow 2% tolerance)
    rel_err = np.abs(p_emp - probs) / np.maximum(probs, 1e-12)
    assert np.all(rel_err < 0.02), f"Empirical probs {p_emp.tolist()} deviate too much from config {probs.tolist()}"


# -----------------------------
# 2) EMA update correctness
# -----------------------------

def test_ema_update_moves_target_towards_online():
    _set_seed(7)
    d_model = 32
    latent_dim = 32
    jepa = JEPAObjective(
        d_model=d_model,
        latent_dim=latent_dim,
        predictor_hidden_multiplier=1.5,
        horizons=[4],
        horizon_probs=[1.0],
        pairs_per_seq=4,
        ema_momentum=0.9,  # we will test this exact momentum
    )

    # Initialize target = online exactly
    jepa._init_target_from_online()

    # Nudge online parameters by adding noise
    with torch.no_grad():
        for p in jepa.online_proj.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Keep a copy of old target, and new online
    old_target_params = [p.detach().clone() for p in jepa.target_proj.parameters()]
    new_online_params = [p.detach().clone() for p in jepa.online_proj.parameters()]

    # Perform EMA update with known momentum
    m = 0.9
    jepa.ema_momentum = m
    jepa.momentum_update()

    # New target should be: m * old_target + (1 - m) * online
    with torch.no_grad():
        for t_param, old_t, new_o in zip(jepa.target_proj.parameters(), old_target_params, new_online_params):
            expected = m * old_t + (1.0 - m) * new_o
            assert torch.allclose(t_param, expected, rtol=1e-5, atol=1e-7), "EMA update incorrect"


# -----------------------------------------------
# 3) RoPE cache, forward pass, and causality test
# -----------------------------------------------

@pytest.mark.parametrize("T", [64, 256, 512])
def test_transformer_rope_forward_and_causality(T: int):
    _set_seed(2024)
    vocab_size = 1024
    d_model = 64
    n_heads = 8   # head_dim = 8 (even) → satisfies RoPE pairing
    n_layers = 3
    B = 2

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,         # deterministic for test
        ff_multiplier=2,
        use_rope=True,
        rope_base=10000.0,
        weight_tying=False,
    ).eval()  # eval → no dropout

    x1 = torch.randint(0, vocab_size, (B, T), dtype=torch.long)
    # Replace future tokens only, keep prefix identical
    x2 = x1.clone()
    cut = T // 2
    if cut < T:
        x2[:, cut:] = torch.randint(0, vocab_size, (B, T - cut), dtype=torch.long)

    # Get both tap and final states to stress different codepaths
    h_tap_1, h_final_1 = model(x1, tap_layer=-2, return_tap=True, grad_barrier=False, tap_norm=True)
    h_tap_2, h_final_2 = model(x2, tap_layer=-2, return_tap=True, grad_barrier=False, tap_norm=True)

    # Shape checks
    assert h_tap_1.shape == (B, T, d_model)
    assert h_final_1.shape == (B, T, d_model)
    assert h_tap_2.shape == (B, T, d_model)
    assert h_final_2.shape == (B, T, d_model)

    # Causality: prefix outputs must be identical when only future tokens change
    assert torch.allclose(h_final_1[:, :cut, :], h_final_2[:, :cut, :], rtol=1e-5, atol=1e-6), \
        "Causality violated: earlier positions changed when only future tokens were modified"

    # Also check tap activations for causality
    assert torch.allclose(h_tap_1[:, :cut, :], h_tap_2[:, :cut, :], rtol=1e-5, atol=1e-6), \
        "Causality violated at tap layer"


# --------------------------------------------------------------
# 4) Mixed precision (BF16) forward/backward one-step smoke test
# --------------------------------------------------------------

@pytest.mark.skipif(not _bf16_supported(), reason="BF16 mixed precision requires CUDA Ampere+ (compute capability >= 8.0)")
def test_bf16_mixed_precision_one_step_cuda():
    _set_seed(31415)

    device = torch.device("cuda")

    vocab_size = 2048
    d_model = 64
    n_layers = 2
    n_heads = 4
    B = 2
    T = 128

    # Keep JEPA head small and light for speed
    jepa_cfg = dict(
        latent_dim=64,
        predictor_hidden_multiplier=1.0,
        horizons=[2, 4],
        horizon_probs=[0.5, 0.5],
        pairs_per_seq=8,
        ema_momentum=0.996,
        gamma_var=1.0,
        gamma_cov=1.0,
        tau=0.1,
        tap_layer=-1,
        grad_barrier=True,
        tap_norm=True,
        jepa_weight=1.0,
        lm_weight=0.5,
        lm_weight_use_scheduler=False,
    )

    model = LitJEPA(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.0,
        ff_multiplier=2,
        use_rope=True,
        rope_base=10000.0,
        weight_tying=False,
        jepa=jepa_cfg,
        optimizer=dict(lr=1.0e-3, lr_head=1.0e-3, weight_decay=0.0, betas=(0.9, 0.95), warmup_steps=10),
    ).to(device)
    model.train()

    # Build optimizer (no scheduler since we don’t have a trainer)
    opt = model.configure_optimizers()
    if isinstance(opt, dict):
        optimizer = opt["optimizer"]
    else:
        optimizer = opt

    x = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model.training_step(x, batch_idx=0)
    assert torch.isfinite(loss).item(), "Loss is not finite in BF16 mixed precision"

    loss.backward()

    # Confirm we actually produced gradients
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p is not None and p.grad is not None:
            g = p.grad.detach()
            if torch.isfinite(g).all():
                total_norm += g.norm().item()
                count += 1
    assert count > 0 and total_norm > 0.0, "No finite gradients produced in BF16 mixed precision"

    optimizer.step()