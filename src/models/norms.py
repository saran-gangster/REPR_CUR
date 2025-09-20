import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.shape[-1] ** 0.5))
        x_normed = x / (norm + self.eps)
        return self.weight * x_normed