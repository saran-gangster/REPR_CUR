"""
Geometric regularizers for latent space shaping.

Two main approaches:
1. VICReg-style: variance + covariance regularization
2. SIGReg-style: isotropic Gaussian via random 1D projections
"""

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, eps: float = 1e-4):
    # Use unbiased=False for efficiency and stability across accelerators
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.relu(1.0 - std).mean()

def covariance_loss(z: torch.Tensor):
    # center
    z = z - z.mean(dim=0, keepdim=True)
    N, D = z.shape
    if N <= 1:
        return z.new_zeros(())
    # covariance matrix (D x D)
    cov = (z.T @ z) / (N - 1)
    # off-diagonal squared sum without materializing a full off-diagonal mask
    off_diag_sum = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
    c = off_diag_sum / D
    return c


def vicreg_regularizer(z: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.1, eps: float = 1e-4) -> tuple[torch.Tensor, dict]:
    """
    VICReg-style geometric regularizer.
    
    Enforces:
    - High variance in each dimension (std >= 1)
    - Low covariance between dimensions
    
    Args:
        z: Latent vectors (N, D)
        var_weight: Weight for variance loss
        cov_weight: Weight for covariance loss
        eps: Epsilon for numerical stability
        
    Returns:
        loss: Combined regularization loss
        metrics: Dict with 'var_loss', 'cov_loss', 'mean_std', 'mean_cov'
    """
    if z.numel() == 0:
        zero = z.new_zeros(())
        return zero, {
            "var_loss": zero,
            "cov_loss": zero,
            "mean_std": zero,
            "mean_cov": zero,
        }
    
    N, D = z.shape
    if N <= 1:
        zero = z.new_zeros(())
        return zero, {
            "var_loss": zero,
            "cov_loss": zero,
            "mean_std": zero,
            "mean_cov": zero,
        }
    
    # Center
    z_centered = z - z.mean(dim=0, keepdim=True)
    
    # Variance loss: penalize std < 1
    std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + eps)
    var_loss = torch.relu(1.0 - std).mean()
    
    # Covariance loss: penalize off-diagonal correlations
    cov = (z_centered.T @ z_centered) / (N - 1)
    off_diag = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
    cov_loss = off_diag / D if D > 1 else z.new_zeros(())
    
    loss = var_weight * var_loss + cov_weight * cov_loss
    
    metrics = {
        "var_loss": var_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "mean_std": std.mean().detach(),
        "mean_cov": (off_diag / (D * (D - 1))).sqrt().detach() if D > 1 else z.new_zeros(()),
    }
    
    return loss, metrics


def sigreg_regularizer(
    z: torch.Tensor,
    num_directions: int = 128,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    SIGReg-style isotropic Gaussian regularizer using random 1D projections.
    
    Based on Cramér-Wold theorem: a distribution is N(0, I) if all 1D projections
    are N(0, 1). We sample random directions and penalize deviations from
    standard normal moments.
    
    Args:
        z: Latent vectors (N, D)
        num_directions: Number of random directions to sample
        eps: Epsilon for numerical stability
        
    Returns:
        loss: Gaussianity loss
        metrics: Dict with 'moment2_loss', 'moment4_loss', 'mean_std'
    """
    if z.numel() == 0:
        zero = z.new_zeros(())
        return zero, {
            "moment2_loss": zero,
            "moment4_loss": zero,
            "mean_std": zero,
        }
    
    N, D = z.shape
    if N <= 1 or D <= 0:
        zero = z.new_zeros(())
        return zero, {
            "moment2_loss": zero,
            "moment4_loss": zero,
            "mean_std": zero,
        }
    
    device = z.device
    
    # Sample random unit directions
    dirs = torch.randn(num_directions, D, device=device)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + eps)  # (M, D)
    
    # Compute projections: (N, M)
    u = z @ dirs.T
    
    # Standardize each direction
    mean = u.mean(dim=0, keepdim=True)
    std = u.std(dim=0, keepdim=True, unbiased=False) + eps
    u_std = (u - mean) / std
    
    # Compute moments: for N(0,1), E[x^2]=1, E[x^4]=3
    m2 = (u_std ** 2).mean(dim=0)  # (M,) should be ~1
    m4 = (u_std ** 4).mean(dim=0)  # (M,) should be ~3
    
    # Loss: penalize deviations from Gaussian moments
    moment2_loss = ((m2 - 1.0) ** 2).mean()
    moment4_loss = ((m4 - 3.0) ** 2).mean()
    
    loss = moment2_loss + moment4_loss
    
    # Compute overall std for monitoring
    mean_std = z.std(dim=0, unbiased=False).mean()
    
    metrics = {
        "moment2_loss": moment2_loss.detach(),
        "moment4_loss": moment4_loss.detach(),
        "mean_std": mean_std.detach(),
    }
    
    return loss, metrics