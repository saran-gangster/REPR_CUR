from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.sampling import sample_anchor_target_pairs
from src.utils.vicreg import variance_loss, covariance_loss
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
        gamma_var: float = 1.0,
        gamma_cov: float = 1.0,
        dropout: float = 0.0,
        tau: float = 0.2,
    ):
        super().__init__()
        assert len(horizons) == len(horizon_probs)
        self.pairs_per_seq = pairs_per_seq
        self.ema_momentum = ema_momentum
        self.gamma_var = gamma_var
        self.gamma_cov = gamma_cov
        self.tau = tau

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
        h: (B, T, C) hidden states from tapped transformer layer
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
        z_tgt = self.target_proj(h_target).detach()  # (N, D_latent), stop-grad teacher

        # horizon embedding in latent space
        z_k = self.horizon_emb_latent(k_ids)    # (N, D_latent)

        # predictor in latent space
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D_latent)

        # normalize for similarity
        p = F.normalize(z_pred, dim=-1)
        y = F.normalize(z_tgt, dim=-1)

        # InfoNCE per-horizon (in-batch negatives within each horizon bucket)
        tau = self.tau
        unique_hids = torch.unique(k_ids)
        total_n = p.size(0)
        info_loss = torch.zeros((), device=device)
        for hid in unique_hids:
            idx = (k_ids == hid).nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                continue
            sim = (p[idx] @ y[idx].T) / tau  # (n, n)
            target = torch.arange(n, device=device)
            loss_h = F.cross_entropy(sim, target)
            info_loss = info_loss + loss_h * (n / max(1, total_n))

        # diagnostics: cosine distance (not optimized)
        cos_loss = 1.0 - (p * y).sum(dim=-1)
        cos_loss = cos_loss.mean()

        # VICReg-style regularization on the ONLINE PROJECTOR's output to prevent collapse
        var_loss = variance_loss(z_anchor)
        cov_loss = covariance_loss(z_anchor)

        loss = info_loss + self.gamma_var * var_loss + self.gamma_cov * cov_loss

        # diagnostics: stds of latents
        std_tgt = z_tgt.std(dim=0, unbiased=False).mean()
        std_anchor = z_anchor.std(dim=0, unbiased=False).mean()
        std_pred = z_pred.std(dim=0, unbiased=False).mean()

        return {
            "loss": loss,
            "info_nce_loss": info_loss.detach(),
            "cos_loss": cos_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "std_tgt": std_tgt.detach(),
            "std_anchor": std_anchor.detach(),
            "std_pred": std_pred.detach(),
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

        z_anchor = self.online_proj(h_anchor)       # (N, D) (unused here; kept for potential future diagnostics)
        z_k = self.horizon_emb_latent(k_ids)        # (N, D)
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D)
        z_tgt = self.target_proj(h_target)          # (N, D)

        return z_pred, z_tgt, k_ids