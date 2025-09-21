import torch

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