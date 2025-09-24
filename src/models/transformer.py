from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm
from .rope import build_rope_cache, apply_rope

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = hidden_mult * d_model
        # Fuse two projections into one matmul for speed
        self.w12 = nn.Linear(d_model, 2 * inner, bias=False)
        self.w3 = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.w12(x).chunk(2, dim=-1)
        x = F.silu(u) * v
        x = self.w3(x)
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_rope: bool = True, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.rope_base = rope_base

        # Fuse QKV into a single projection for speed
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._rope_cache = None
        self._rope_cache_len = 0

    def maybe_build_rope(self, seq_len, device, dtype):
        if not self.use_rope:
            return None
        if (
            self._rope_cache is None
            or self._rope_cache_len < seq_len
            or self._rope_cache[0].device != device
            or self._rope_cache[0].dtype != dtype
        ):
            cos, sin = build_rope_cache(seq_len, self.head_dim, base=self.rope_base, device=device, dtype=dtype)
            self._rope_cache = (cos, sin)
            self._rope_cache_len = seq_len

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(0, 2, 3, 1, 4)  # (B, 3, H, T, D)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, H, T, D)

        if self.use_rope:
            self.maybe_build_rope(T, x.device, x.dtype)
            cos, sin = self._rope_cache
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # scaled dot-product attention with causal masking
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
        )  # (B, H, T, D)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 ff_multiplier: int = 4, use_rope: bool = True, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_rope, rope_base)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, hidden_mult=ff_multiplier, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 dropout: float = 0.1, ff_multiplier: int = 4, use_rope: bool = True, rope_base: float = 10000.0,
                 weight_tying: bool = True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, ff_multiplier, use_rope, rope_base)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.n_layers = n_layers

        # LM head (weight tying now optional)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.weight_tying = bool(weight_tying)
        if self.weight_tying:
            self.lm_head.weight = self.token_emb.weight  # optional tie

        # Optional normalization at tap to stabilize features
        self.norm_tap = RMSNorm(d_model)

        # LM-only bridge to adapt detached lower features for the upper LM stack
        self.lm_bridge = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, bias=True),
        )
        # NEW: learnable gate for the bridge, initialized near zero (sigmoid(-2) ~ 0.12)
        self.lm_bridge_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, idx, tap_layer: Optional[int] = None, return_tap: bool = False,
                grad_barrier: bool = False, tap_norm: bool = False):
        """
        idx: (B, T) token ids
        tap_layer: if not None, index of block output to return for JEPA (supports negative index)
        return_tap: if True, returns (h_tap, h_final). Else returns h_final.
        grad_barrier: if True and tap_layer is set, detach the graph at the tap so
                      gradients from layers above the tap do not flow below it.
        tap_norm: if True, apply a small RMSNorm at the tap before returning it.
        """
        # idx -> embeddings
        x = self.token_emb(idx)  # (B, T, C)
        x = self.drop(x)

        tap_idx = None
        h_tap = None
        if tap_layer is not None:
            # normalize negative indexing and clamp into range
            tl = tap_layer if tap_layer >= 0 else self.n_layers + tap_layer
            tl = max(0, min(self.n_layers - 1, tl))
            tap_idx = tl

        # bridge gate (scalar in [0,1])
        g = torch.sigmoid(self.lm_bridge_gate)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if tap_idx is not None and i == tap_idx:
                # Capture JEPA tap BEFORE any detach/bridge
                h_tap = x
                if tap_norm:
                    h_tap = self.norm_tap(h_tap)
                # Detach the LM path to block gradients flowing into lower layers
                if grad_barrier:
                    x = x.detach()
                # Apply gated LM-only bridge after the detach
                x = x + g * self.lm_bridge(x)

        h_final = self.norm_f(x)

        if return_tap:
            if h_tap is None:
                # if tap requested but index invalid, fall back to using final pre-norm (rare)
                h_tap = x
                if tap_norm:
                    h_tap = self.norm_tap(h_tap)
            return h_tap, h_final
        else:
            return h_final