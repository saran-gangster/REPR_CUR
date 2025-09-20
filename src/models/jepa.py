from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.sampling import sample_anchor_target_pairs
from src.utils.vicreg import variance_loss, covariance_loss
from src.utils.ema import update_ema_

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * out_dim
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
        predictor_hidden_multiplier: int = 2,
        horizons: List[int] = [8, 32, 128],
        horizon_probs: List[float] = [0.5, 0.3, 0.2],
        pairs_per_seq: int = 64,
        ema_momentum: float = 0.996,
        gamma_var: float = 1.0,
        gamma_cov: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(horizons) == len(horizon_probs)
        self.horizons = horizons
        self.horizon_probs = torch.tensor(horizon_probs, dtype=torch.float)
        self.horizon_probs = self.horizon_probs / self.horizon_probs.sum()
        self.pairs_per_seq = pairs_per_seq
        self.ema_momentum = ema_momentum
        self.gamma_var = gamma_var
        self.gamma_cov = gamma_cov

        self.horizon_emb = nn.Embedding(len(horizons), d_model)
        self.predictor = MLP(in_dim=d_model * 2, out_dim=latent_dim,
                             hidden_mult=predictor_hidden_multiplier, dropout=dropout)
        # target projector is EMA copy recipient
        self.target_proj = MLP(in_dim=d_model, out_dim=latent_dim,
                               hidden_mult=predictor_hidden_multiplier, dropout=0.0)
        self._init_target_from_predictor()

    def _init_target_from_predictor(self):
        # initialize target_proj to same structure (but from predictor's "output space")
        # We only copy shapes that match
        with torch.no_grad():
            for tp, pp in zip(self.target_proj.parameters(), self.predictor.parameters()):
                if tp.shape == pp.shape:
                    tp.data.copy_(pp.data)

    @torch.no_grad()
    def momentum_update(self):
        # EMA update for target projector
        update_ema_(self.target_proj, self.predictor, self.ema_momentum)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: (B, T, C) final hidden states from transformer
        Returns dict with loss and diagnostics
        """
        B, T, C = h.shape
        device = h.device

        # sample anchor/target pairs
        (b_idx, t_idx, tpos, k_ids) = sample_anchor_target_pairs(
            batch_size=B,
            seq_len=T,
            pairs_per_seq=self.pairs_per_seq,
            horizon_values=self.horizons,
            horizon_probs=self.horizon_probs,
            device=device
        )
        # gather hidden states
        h_anchor = h[b_idx, t_idx, :]           # (N, C)
        h_target = h[b_idx, tpos, :]            # (N, C)
        # horizon embeddings
        hk = self.horizon_emb(k_ids)            # (N, C)
        predictor_in = torch.cat([h_anchor, hk], dim=-1)    # (N, 2C)
        z_pred = self.predictor(predictor_in)   # (N, D_latent)

        with torch.no_grad():
            z_tgt = self.target_proj(h_target)  # (N, D_latent)
            z_tgt = z_tgt.detach()

        # cosine distance
        p = F.normalize(z_pred, dim=-1)
        y = F.normalize(z_tgt, dim=-1)
        cos_loss = 1.0 - (p * y).sum(dim=-1)
        cos_loss = cos_loss.mean()

        # VICReg-style variance and covariance terms (to prevent collapse)
        var_loss = variance_loss(z_pred) + variance_loss(z_tgt)
        cov_loss = covariance_loss(z_pred) + covariance_loss(z_tgt)

        loss = cos_loss + self.gamma_var * var_loss + self.gamma_cov * cov_loss

        return {
            "loss": loss,
            "cos_loss": cos_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "num_pairs": torch.tensor(h_anchor.shape[0], device=device, dtype=torch.float)
        }