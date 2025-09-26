import torch

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    # Rotary embedding as in RoPE paper
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    freqs = pos * theta  # (T, head_dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # interleave for even-odd pairs
    cos = torch.stack((cos, cos), dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack((sin, sin), dim=-1).reshape(seq_len, head_dim)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, n_heads, T, head_dim)
    # cos/sin: (T, head_dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos.unsqueeze(0).unsqueeze(0) + x_rot * sin.unsqueeze(0).unsqueeze(0)