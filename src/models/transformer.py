from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import RMSNorm
from .rope import build_rope_cache, apply_rope
import math

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
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    weight_tying: bool = False,
    simple_recurrence_steps: int = 0,
    lm_bridge_enabled: bool = True,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, ff_multiplier, use_rope, rope_base)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.n_layers = n_layers

        self.weight_tying = bool(weight_tying)
        self.simple_recurrence_steps = max(0, int(simple_recurrence_steps))
        self.lm_bridge_enabled = bool(lm_bridge_enabled)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if self.weight_tying:
            # tie weights
            self.lm_head.weight = self.token_emb.weight

        # Tied softmax stabilizers: learnable temperature + output bias
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model))) if self.weight_tying else None
        self.output_bias = nn.Parameter(torch.zeros(vocab_size)) if self.weight_tying else None
        self.norm_tap = RMSNorm(d_model)

        # LM-only bridge on the upper stack
        if self.lm_bridge_enabled:
            self.lm_bridge = nn.Sequential(
                RMSNorm(d_model),
                nn.Linear(d_model, d_model, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model, bias=True),
            )
            self.lm_bridge_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.lm_bridge = None
            self.register_parameter("lm_bridge_gate", None)

    def tap_index(self, tap_layer: Optional[int]) -> Optional[int]:
        if tap_layer is None:
            return None
        tl = tap_layer if tap_layer >= 0 else self.n_layers + tap_layer
        return max(0, min(self.n_layers - 1, tl))

    def simple_recurrence(self, h: torch.Tensor, tap_layer: int, steps: Optional[int] = None) -> torch.Tensor:
        if steps is None:
            steps = self.simple_recurrence_steps
        steps = 0 if steps is None else max(0, int(steps))
        if steps <= 0:
            return h
        tl = self.tap_index(tap_layer)
        if tl is None:
            return h
        blk = self.blocks[tl]
        x = h
        for _ in range(steps):
            x = blk(x)
        return x

    def forward(self, idx, tap_layer: Optional[int] = None, return_tap: bool = False,
                grad_barrier: bool = False, tap_norm: bool = False,
                simple_recurrence_steps: Optional[int] = None):
        """
        idx: (B, T) token ids
        tap_layer: if not None, index of block output to return for JEPA (supports negative index)
        return_tap: if True, returns (h_tap, h_final). Else returns h_final.
        grad_barrier: if True and tap_layer is set, detach the graph at the tap so
                      gradients from layers above the tap do not flow below it.
        tap_norm: if True, apply a small RMSNorm at the tap before returning it.
        """
        # idx -> embeddings
        x = self.token_emb(idx)
        x = self.drop(x)

        tap_idx = self.tap_index(tap_layer)
        steps = self.simple_recurrence_steps if simple_recurrence_steps is None else max(0, int(simple_recurrence_steps))
        h_tap_original = None
        h_tap_recurrent = None

        bridge_active = self.lm_bridge_enabled and self.lm_bridge is not None and self.lm_bridge_gate is not None
        g = torch.sigmoid(self.lm_bridge_gate) if bridge_active else None

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if tap_idx is not None and i == tap_idx:
                h_tap_original = x
                h_tap_recurrent = self.simple_recurrence(h_tap_original, tap_idx, steps)

                x_for_upper = h_tap_original
                if grad_barrier:
                    x_for_upper = x_for_upper.detach()

                if bridge_active:
                    # Existing LM-only bridge
                    x = x_for_upper + g * self.lm_bridge(x_for_upper)
                else:
                    x = x_for_upper

        h_final = self.norm_f(x)

        if return_tap:
            if h_tap_original is None:
                h_tap_original = x
            if h_tap_recurrent is None:
                h_tap_recurrent = h_tap_original
            if tap_norm:
                h_tap_original = self.norm_tap(h_tap_original)
                h_tap_recurrent = self.norm_tap(h_tap_recurrent)
            return (h_tap_recurrent, h_tap_original), h_final
        else:
            return h_final