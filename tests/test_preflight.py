import os
import sys
import numpy as np
import torch
import pytest
import lightning as L
from torch.utils.data import Dataset, DataLoader 

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

    k_vals = torch.tensor(horizons, dtype=torch.long)[k_ids]
    assert torch.all(tpos == (t_idx + k_vals))
    assert torch.all(t_idx >= 0)
    assert torch.all(tpos <= T - 1)
    counts_b = torch.bincount(b_idx, minlength=B)
    assert torch.all(counts_b == pairs_per_seq)


def test_sampler_horizon_distribution_matches_probs():
    _set_seed(123)
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
    assert np.all(obs > 100)
    rel_err = np.abs(p_emp - probs) / np.maximum(probs, 1e-12)
    assert np.all(rel_err < 0.02)


# -----------------------------
# 2) EMA update correctness
# -----------------------------

def test_ema_update_moves_target_towards_online():
    _set_seed(7)
    jepa = JEPAObjective(d_model=32, latent_dim=32, horizons=[4], horizon_probs=[1.0])
    jepa._init_target_from_online()
    with torch.no_grad():
        for p in jepa.online_proj.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    old_target_params = [p.detach().clone() for p in jepa.target_proj.parameters()]
    new_online_params = [p.detach().clone() for p in jepa.online_proj.parameters()]
    m = 0.9
    jepa.ema_momentum = m
    jepa.momentum_update()
    with torch.no_grad():
        for t_param, old_t, new_o in zip(jepa.target_proj.parameters(), old_target_params, new_online_params):
            expected = m * old_t + (1.0 - m) * new_o
            assert torch.allclose(t_param, expected, rtol=1e-5, atol=1e-7)


# -----------------------------------------------
# 3) RoPE cache, forward pass, and causality test
# -----------------------------------------------

@pytest.mark.parametrize("T", [64, 256, 512])
def test_transformer_rope_forward_and_causality(T: int):
    _set_seed(2024)
    model = DecoderOnlyTransformer(vocab_size=1024, d_model=64, n_layers=3, n_heads=8, use_rope=True).eval()
    x1 = torch.randint(0, 1024, (2, T), dtype=torch.long)
    x2 = x1.clone()
    cut = T // 2
    if cut < T:
        x2[:, cut:] = torch.randint(0, 1024, (2, T - cut), dtype=torch.long)
    tap_out_1, h_final_1 = model(x1, tap_layer=-2, return_tap=True)
    tap_out_2, h_final_2 = model(x2, tap_layer=-2, return_tap=True)

    if isinstance(tap_out_1, tuple):
        h_tap_1 = tap_out_1[0]
    else:
        h_tap_1 = tap_out_1

    if isinstance(tap_out_2, tuple):
        h_tap_2 = tap_out_2[0]
    else:
        h_tap_2 = tap_out_2
    assert h_tap_1.shape == (2, T, 64)
    assert h_final_1.shape == (2, T, 64)
    assert torch.allclose(h_final_1[:, :cut, :], h_final_2[:, :cut, :], rtol=1e-5, atol=1e-6)
    assert torch.allclose(h_tap_1[:, :cut, :], h_tap_2[:, :cut, :], rtol=1e-5, atol=1e-6)

# --------------------------------------------------------------
# 4) Mixed precision (BF16) forward/backward one-step smoke test
# --------------------------------------------------------------

@pytest.mark.skipif(not _bf16_supported(), reason="BF16 mixed precision requires CUDA Ampere+ (compute capability >= 8.0)")
def test_bf16_mixed_precision_one_step_cuda():
    _set_seed(31415)
    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    jepa_cfg = dict(
        latent_dim=64, predictor_hidden_multiplier=1.0, horizons=[2, 4], horizon_probs=[0.5, 0.5],
        pairs_per_seq=8, ema_momentum=0.996,
        loss_type="barlow", off_diag_scale=0.02, align_scale=1.0,
        tap_layer=-1, grad_barrier=True, tap_norm=True, jepa_weight=0.1,
        lm_weight=0.5, lm_weight_use_scheduler=False,
    )
    model = LitJEPA(
        vocab_size=2048, d_model=64, n_layers=2, n_heads=4, dropout=0.0, ff_multiplier=2,
        use_rope=True, weight_tying=False, jepa=jepa_cfg,
        optimizer=dict(lr=1.0e-3, lr_head=1.0e-3, weight_decay=0.0, betas=(0.9, 0.95), warmup_steps=10),
    )

    class DummyDataset(Dataset):
        def __init__(self, num_samples, seq_len, vocab_size):
            self.num_samples = num_samples
            self.data = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return self.data[idx]

    dl = DataLoader(DummyDataset(num_samples=8, seq_len=128, vocab_size=2048), batch_size=2)

    try:
        trainer.fit(model, train_dataloaders=dl)
    except Exception as e:
        pytest.fail(f"BF16 mixed precision training step failed unexpectedly: {e}")

    # Final check for finite gradients
    total_norm, count = 0.0, 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            assert torch.isfinite(g).all(), "Found non-finite gradients."
            total_norm += g.norm().item()
            count += 1
    assert count > 0 and total_norm > 0.0, "No gradients were produced."


