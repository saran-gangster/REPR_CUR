"""
Quick test to verify baseline mode (alpha=0.0 or run_baseline=True) works correctly.
"""
import torch
from src.lightning.lit_jepa import LitJEPA


def test_baseline_mode_with_alpha_zero():
    """Test that alpha=0.0 disables JEPA computation."""
    model = LitJEPA(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        jepa={"alpha": 0.0},
    )
    
    # Should be in baseline mode
    assert model.alpha == 0.0
    
    # Create dummy batch
    x = torch.randint(0, 1000, (2, 64))
    
    # Training step should work without JEPA
    loss = model.training_step(x, 0)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    print(f"✓ Baseline with alpha=0.0 works, loss: {loss.item():.4f}")


def test_baseline_mode_with_run_baseline_flag():
    """Test that run_baseline=True disables JEPA computation."""
    model = LitJEPA(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        jepa={"run_baseline": True, "alpha": 0.5},  # alpha should be ignored
    )
    
    # Should be in baseline mode with alpha forced to 0
    assert model.run_baseline is True
    assert model.alpha == 0.0  # Should override the 0.5
    
    # Create dummy batch
    x = torch.randint(0, 1000, (2, 64))
    
    # Training step should work without JEPA
    loss = model.training_step(x, 0)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    print(f"✓ Baseline with run_baseline=True works, loss: {loss.item():.4f}")


def test_jepa_mode_still_works():
    """Test that JEPA mode (alpha>0) still works."""
    model = LitJEPA(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        jepa={"alpha": 0.1, "latent_dim": 64},
    )
    
    assert model.alpha == 0.1
    assert model.run_baseline is False
    
    # Create dummy batch
    x = torch.randint(0, 1000, (2, 64))
    
    # Training step should compute JEPA loss
    loss = model.training_step(x, 0)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    print(f"✓ JEPA mode with alpha=0.1 works, loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("Testing baseline mode...")
    test_baseline_mode_with_alpha_zero()
    test_baseline_mode_with_run_baseline_flag()
    test_jepa_mode_still_works()
    print("\n✅ All baseline mode tests passed!")
