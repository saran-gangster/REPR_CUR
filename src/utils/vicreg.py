import torch

def variance_loss(z: torch.Tensor, eps: float = 1e-4):
    # z: (N, D)
    std = torch.sqrt(z.var(dim=0) + eps)  # (D,)
    v = torch.mean(torch.relu(1.0 - std))
    return v

def covariance_loss(z: torch.Tensor):
    # center
    z = z - z.mean(dim=0, keepdim=True)
    N, D = z.shape
    # covariance matrix (D x D)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    c = (off_diag ** 2).sum() / D
    return c