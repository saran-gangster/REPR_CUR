from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.sampling import sample_anchor_target_pairs
from src.utils.ema import update_ema_

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(round(hidden_mult * out_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)

class JEPAObjective(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int = 256,
        predictor_hidden_multiplier: float = 2.0,
        horizons: List[int] = [8, 32, 128],
        horizon_probs: List[float] = [0.5, 0.3, 0.2],
        pairs_per_seq: int = 64,
        ema_momentum: float = 0.996,
        dropout: float = 0.0,
        loss_type: str = "barlow",
        off_diag_scale: float = 0.01,
        align_scale: float = 1.0,
    ):
        super().__init__()
        assert len(horizons) == len(horizon_probs)
        self.pairs_per_seq = pairs_per_seq
        self.ema_momentum = ema_momentum
        self.loss_type = loss_type
        self.off_diag_scale = off_diag_scale
        self.align_scale = align_scale

        # Register sampling buffers
        probs = torch.tensor(horizon_probs, dtype=torch.float)
        probs = probs / probs.sum()
        self.register_buffer("horizon_probs", probs)
        self.register_buffer("horizon_values", torch.tensor(horizons, dtype=torch.long))
        self.horizons = horizons  # for logging

        # Horizon embedding directly in latent space
        self.horizon_emb_latent = nn.Embedding(len(horizons), latent_dim)

        # Online and target projectors share architecture (EMA teacher)
        self.online_proj = MLP(in_dim=d_model, out_dim=latent_dim,
                               hidden_mult=predictor_hidden_multiplier, dropout=dropout)
        self.target_proj = MLP(in_dim=d_model, out_dim=latent_dim,
                               hidden_mult=predictor_hidden_multiplier, dropout=0.0)

        # Freeze EMA teacher params
        self.target_proj.requires_grad_(False)

        # Predictor in latent space over [z_anchor, z_k]
        self.predictor = MLP(in_dim=2 * latent_dim, out_dim=latent_dim,
                             hidden_mult=predictor_hidden_multiplier, dropout=dropout)

        self._init_target_from_online()

    def _barlow_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute standard Barlow Twins loss with internal batch normalisation."""
        if z_a.numel() == 0:
            zero = z_a.new_zeros(())
            return zero, zero, zero

        N, D = z_a.shape
        if N <= 1:
            zero = z_a.new_zeros(())
            return zero, zero, zero

        a_norm = (z_a - z_a.mean(dim=0, keepdim=True)) / (z_a.std(dim=0, unbiased=False, keepdim=True) + eps)
        b_norm = (z_b - z_b.mean(dim=0, keepdim=True)) / (z_b.std(dim=0, unbiased=False, keepdim=True) + eps)

        c = (a_norm.T @ b_norm) / float(N)

        on_diag = (1.0 - torch.diagonal(c)).pow(2).mean()

        if D <= 1:
            off_diag = c.new_zeros(())
        else:
            off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum()) / (D * (D - 1))

        loss = self.align_scale * on_diag + self.off_diag_scale * off_diag
        return loss, on_diag, off_diag

    def _init_target_from_online(self):
        with torch.no_grad():
            for tp, op in zip(self.target_proj.parameters(), self.online_proj.parameters()):
                if tp.shape == op.shape:
                    tp.data.copy_(op.data)

    @torch.no_grad()
    def momentum_update(self):
        # EMA update for target projector from online projector
        update_ema_(self.target_proj, self.online_proj, self.ema_momentum)
    
    def _sample_pairs(self, B: int, T: int, device, generator: torch.Generator | None = None):
        return sample_anchor_target_pairs(
            batch_size=B,
            seq_len=T,
            pairs_per_seq=self.pairs_per_seq,
            horizon_values=self.horizon_values,
            horizon_probs=self.horizon_probs,
            device=device,
            generator=generator,
        )

    def _resolve_pairs(
        self,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        B: int,
        T: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if pairs is None:
            b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device, generator=generator)
        else:
            b_idx, t_idx, tpos, k_ids = pairs
            if b_idx.device != device:
                b_idx = b_idx.to(device, non_blocking=True)
                t_idx = t_idx.to(device, non_blocking=True)
                tpos = tpos.to(device, non_blocking=True)
                k_ids = k_ids.to(device, non_blocking=True)
        return b_idx, t_idx, tpos, k_ids

    def forward(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_latents: bool = False,
        teacher_h: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        h: (B, T, C) student hidden states from tapped transformer layer (anchor source)
        teacher_h: optional (B, T, C) teacher hidden states (target source). Defaults to `h` if None.
        pairs: optional (b_idx, t_idx, tpos, k_ids). If None, freshly sample pairs.
        Returns dict with loss and diagnostics.
        """
        B, T, C = h.shape
        device = h.device

        # Use fixed pairs if provided; else sample
        b_idx, t_idx, tpos, k_ids = self._resolve_pairs(pairs, B, T, device)

        # gather hidden states
        h_anchor = h[b_idx, t_idx, :]           # (N, C)
        if teacher_h is None:
            h_target = h[b_idx, tpos, :]
        else:
            if teacher_h.device != device:
                teacher_h = teacher_h.to(device, non_blocking=True)
            h_target = teacher_h[b_idx, tpos, :]

        # project to latent space
        z_anchor = self.online_proj(h_anchor)   # (N, D_latent)
        z_tgt = self.target_proj(h_target).detach()  # (N, D_latent), stop-grad teacher

        # horizon embedding in latent space
        z_k = self.horizon_emb_latent(k_ids)    # (N, D_latent)

        # predictor in latent space
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D_latent)

        if z_pred.numel() == 0:
            zero = z_pred.new_zeros(())
            loss = zero
            align_loss = zero
            barlow_diag = zero
            barlow_off = zero
            mse_pred_tgt = zero
        else:
            if getattr(self, "loss_type", "barlow") == "barlow":
                align_loss, barlow_diag, barlow_off = self._barlow_loss(z_pred, z_tgt)
            else:
                raise ValueError(f"Unsupported JEPA loss_type: {self.loss_type}")

            loss = align_loss
            mse_pred_tgt = F.mse_loss(z_pred, z_tgt).detach()

        # Keep only essential diagnostics
        std_anchor = z_anchor.std(dim=0, unbiased=False).mean()
        std_pred = z_pred.std(dim=0, unbiased=False).mean()

        out = {
            "loss": loss,
            "align_loss": align_loss.detach(),
            "barlow_on_diag": barlow_diag.detach(),
            "barlow_off_diag": barlow_off.detach(),
            "mse_pred_tgt": mse_pred_tgt,
            "std_anchor": std_anchor.detach(),
            "std_pred": std_pred.detach(),
        }
        if return_latents:
            out["z_pred"] = z_pred
            out["pairs"] = (b_idx, t_idx, tpos, k_ids)
        return out

    @torch.no_grad()
    def compute_latents(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        teacher_h: torch.Tensor | None = None,
    ):
        """
        Utility for validation: return prediction/target latents using either:
        - provided fixed pairs (b_idx, t_idx, tpos, k_ids), or
        - freshly sampled pairs if pairs is None.
        Returns:
        z_pred: (N, D)
        z_tgt:  (N, D)
        k_ids:  (N,) horizon category ids
        """
        B, T, C = h.shape
        device = h.device

        b_idx, t_idx, tpos, k_ids = self._resolve_pairs(pairs, B, T, device)

        h_anchor = h[b_idx, t_idx, :]    # (N, C)
        if teacher_h is None:
            h_target = h[b_idx, tpos, :]
        else:
            if teacher_h.device != device:
                teacher_h = teacher_h.to(device, non_blocking=True)
            h_target = teacher_h[b_idx, tpos, :]

        z_anchor = self.online_proj(h_anchor)       # (N, D)
        z_k = self.horizon_emb_latent(k_ids)        # (N, D)
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D)
        z_tgt = self.target_proj(h_target)          # (N, D)

        return z_pred, z_tgt, k_ids