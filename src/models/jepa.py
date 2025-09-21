from typing import Dict, List
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
        self.pairs_per_seq = pairs_per_seq
        self.ema_momentum = ema_momentum
        self.gamma_var = gamma_var
        self.gamma_cov = gamma_cov

        # Register sampling buffers to avoid per-step device transfers/allocations
        probs = torch.tensor(horizon_probs, dtype=torch.float)
        probs = probs / probs.sum()
        self.register_buffer("horizon_probs", probs)
        self.register_buffer("horizon_values", torch.tensor(horizons, dtype=torch.long))
        self.horizons = horizons  # keep for logging

        # Horizon embedding directly in latent space
        self.horizon_emb_latent = nn.Embedding(len(horizons), latent_dim)

        # BYOL-correct: online and target projectors share architecture
        self.online_proj = MLP(in_dim=d_model, out_dim=latent_dim,
                               hidden_mult=predictor_hidden_multiplier, dropout=dropout)
        self.target_proj = MLP(in_dim=d_model, out_dim=latent_dim,
                               hidden_mult=predictor_hidden_multiplier, dropout=0.0)

        # Predictor operates purely in latent space over [z_anchor, z_k]
        self.predictor = MLP(in_dim=2 * latent_dim, out_dim=latent_dim,
                             hidden_mult=predictor_hidden_multiplier, dropout=dropout)

        self._init_target_from_online()

    def _init_target_from_online(self):
        with torch.no_grad():
            for tp, op in zip(self.target_proj.parameters(), self.online_proj.parameters()):
                if tp.shape == op.shape:
                    tp.data.copy_(op.data)

    @torch.no_grad()
    def momentum_update(self):
        # EMA update for target projector from online projector
        update_ema_(self.target_proj, self.online_proj, self.ema_momentum)

    def _sample_pairs(self, B: int, T: int, device):
        return sample_anchor_target_pairs(
            batch_size=B,
            seq_len=T,
            pairs_per_seq=self.pairs_per_seq,
            horizon_values=self.horizon_values,
            horizon_probs=self.horizon_probs,
            device=device
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: (B, T, C) final hidden states from transformer
        Returns dict with loss and diagnostics
        """
        B, T, C = h.shape
        device = h.device

        # sample anchor/target pairs
        b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device)

        # gather hidden states
        h_anchor = h[b_idx, t_idx, :]           # (N, C)
        h_target = h[b_idx, tpos, :]            # (N, C)

        # project to latent space
        z_anchor = self.online_proj(h_anchor)   # (N, D_latent)
        z_tgt = self.target_proj(h_target).detach()  # (N, D_latent)

        # horizon embedding in latent space
        z_k = self.horizon_emb_latent(k_ids)    # (N, D_latent)

        # predictor in latent space
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D_latent)

        # cosine loss
        p = F.normalize(z_pred, dim=-1)
        y = F.normalize(z_tgt, dim=-1)
        cos_loss = 1.0 - (p * y).sum(dim=-1)
        cos_loss = cos_loss.mean()

        # VICReg terms (on unnormalized latents)
        var_loss = variance_loss(z_pred) + variance_loss(z_tgt)
        cov_loss = covariance_loss(z_pred) + covariance_loss(z_tgt)

        loss = cos_loss + self.gamma_var * var_loss + self.gamma_cov * cov_loss

        return {
            "loss": loss,
            "cos_loss": cos_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "num_pairs": torch.tensor(h_anchor.shape[0], device=device, dtype=torch.float),
        }

    @torch.no_grad()
    def compute_latents(self, h: torch.Tensor):
        """
        Utility for validation: sample pairs and return prediction/target latents.
        Returns:
          z_pred: (N, D)
          z_tgt:  (N, D)
          k_ids:  (N,) horizon category ids
        """
        B, T, C = h.shape
        device = h.device

        b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device)

        h_anchor = h[b_idx, t_idx, :]    # (N, C)
        h_target = h[b_idx, tpos, :]     # (N, C)

        z_anchor = self.online_proj(h_anchor)       # (N, D)
        z_k = self.horizon_emb_latent(k_ids)        # (N, D)
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D)
        z_tgt = self.target_proj(h_target)          # (N, D)

        return z_pred, z_tgt, k_ids