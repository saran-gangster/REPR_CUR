"""
Minimal JEPA objective inspired by LeJEPA principles.

Core design:
1. Simple prediction loss (cosine or L2) between predicted and target latents
2. One geometric regularizer (VICReg or SIGReg) to prevent collapse
3. No teacher-student, no EMA, no contrastive NCE, no semantic coupling

Total loss: (1 - λ) * prediction_loss + λ * geometry_loss
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.sampling import sample_anchor_target_pairs
from src.utils.vicreg import vicreg_regularizer, sigreg_regularizer


class MLP(nn.Module):
    """Simple 2-layer MLP with GELU activation."""
    
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
    """
    Minimal JEPA objective: predict future latents + geometric regularization.
    
    Architecture:
    - Projector: h_t -> z_t (hidden to latent)
    - Predictor: (z_t, horizon_emb) -> z_pred_{t+k} (predict future latent)
    - Horizon embeddings: learnable embeddings for each horizon value
    
    Loss:
    - Prediction: cosine or L2 distance between z_pred and z_target
    - Geometry: VICReg or SIGReg to prevent collapse
    """
    
    def __init__(
        self,
        d_model: int,
        latent_dim: int = 256,
        predictor_hidden_multiplier: float = 2.0,
        horizons: List[int] = [2, 8, 32, 64],
        horizon_probs: List[float] = [0.4, 0.3, 0.2, 0.1],
        pairs_per_seq: int = 64,
        dropout: float = 0.0,
        prediction_loss_type: str = "cosine",  # "cosine" or "l2"
        geometry_regularizer: str = "vicreg",  # "vicreg" or "sigreg"
        geometry_lambda: float = 0.2,  # Weight for geometry loss
        vicreg_var_weight: float = 1.0,
        vicreg_cov_weight: float = 0.1,
        sigreg_num_directions: int = 128,
    ):
        super().__init__()
        assert len(horizons) == len(horizon_probs)
        
        self.latent_dim = latent_dim
        self.pairs_per_seq = pairs_per_seq
        self.prediction_loss_type = prediction_loss_type
        self.geometry_regularizer = geometry_regularizer
        self.geometry_lambda = geometry_lambda
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        self.sigreg_num_directions = sigreg_num_directions
        
        # Register sampling buffers
        probs = torch.tensor(horizon_probs, dtype=torch.float)
        probs = probs / probs.sum()
        self.register_buffer("horizon_probs", probs)
        self.register_buffer("horizon_values", torch.tensor(horizons, dtype=torch.long))
        self.horizons = horizons
        
        # Horizon embeddings
        self.horizon_emb = nn.Embedding(len(horizons), latent_dim)
        
        # Projector: hidden -> latent
        self.projector = MLP(
            in_dim=d_model,
            out_dim=latent_dim,
            hidden_mult=predictor_hidden_multiplier,
            dropout=dropout,
        )
        
        # Predictor: (z_anchor, z_horizon) -> z_pred
        self.predictor = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_mult=predictor_hidden_multiplier,
            dropout=dropout,
        )
        
    def _sample_pairs(
        self,
        B: int,
        T: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ):
        """Sample (anchor, target) pairs with horizons."""
        return sample_anchor_target_pairs(
            batch_size=B,
            seq_len=T,
            pairs_per_seq=self.pairs_per_seq,
            horizon_values=self.horizon_values,
            horizon_probs=self.horizon_probs,
            device=device,
            generator=generator,
        )
    
    def _prediction_loss(self, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction loss between predicted and target latents.
        
        Args:
            z_pred: Predicted latents (N, D)
            z_tgt: Target latents (N, D)
            
        Returns:
            loss: Scalar loss
        """
        if z_pred.numel() == 0 or z_tgt.numel() == 0:
            return z_pred.new_zeros(())
        
        if self.prediction_loss_type == "cosine":
            # Cosine similarity loss: 1 - cos(z_pred, z_tgt)
            z_pred_norm = F.normalize(z_pred, dim=-1, eps=1e-8)
            z_tgt_norm = F.normalize(z_tgt, dim=-1, eps=1e-8)
            loss = (1.0 - (z_pred_norm * z_tgt_norm).sum(dim=-1)).mean()
        elif self.prediction_loss_type == "l2":
            # L2 loss
            loss = F.mse_loss(z_pred, z_tgt)
        else:
            raise ValueError(f"Unknown prediction_loss_type: {self.prediction_loss_type}")
        
        return loss
    
    def _geometry_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute geometric regularization loss.
        
        Args:
            z: Latent vectors to regularize (can be anchors, predictions, or both)
            
        Returns:
            loss: Scalar loss
            metrics: Dict of diagnostic metrics
        """
        if self.geometry_regularizer == "vicreg":
            return vicreg_regularizer(
                z,
                var_weight=self.vicreg_var_weight,
                cov_weight=self.vicreg_cov_weight,
            )
        elif self.geometry_regularizer == "sigreg":
            return sigreg_regularizer(
                z,
                num_directions=self.sigreg_num_directions,
            )
        else:
            raise ValueError(f"Unknown geometry_regularizer: {self.geometry_regularizer}")
    
    def forward(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute minimal JEPA objective.
        
        Args:
            h: Hidden states from transformer (B, T, C)
            pairs: Optional pre-sampled pairs (b_idx, t_idx, tpos, k_ids)
            
        Returns:
            Dictionary with:
                - loss: Total JEPA loss
                - pred_loss: Prediction loss
                - geom_loss: Geometry loss
                - Additional metrics from geometry regularizer
        """
        B, T, C = h.shape
        device = h.device
        
        # Sample or use provided pairs
        if pairs is None:
            b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device)
        else:
            b_idx, t_idx, tpos, k_ids = pairs
            if b_idx.device != device:
                b_idx = b_idx.to(device, non_blocking=True)
                t_idx = t_idx.to(device, non_blocking=True)
                tpos = tpos.to(device, non_blocking=True)
                k_ids = k_ids.to(device, non_blocking=True)
        
        # Gather hidden states
        h_anchor = h[b_idx, t_idx, :]  # (N, C)
        h_target = h[b_idx, tpos, :]   # (N, C)
        
        # Project to latent space
        z_anchor = self.projector(h_anchor)  # (N, D)
        z_tgt = self.projector(h_target).detach()  # (N, D), stop grad on target
        
        # Get horizon embeddings
        z_horizon = self.horizon_emb(k_ids)  # (N, D)
        
        # Predict future latent
        z_pred = self.predictor(torch.cat([z_anchor, z_horizon], dim=-1))  # (N, D)
        
        # Compute losses
        pred_loss = self._prediction_loss(z_pred, z_tgt)
        
        # Apply geometry regularizer to both predictions and targets
        z_all = torch.cat([z_pred, z_tgt], dim=0)  # (2N, D)
        geom_loss, geom_metrics = self._geometry_loss(z_all)
        
        # Combined loss
        total_loss = (1.0 - self.geometry_lambda) * pred_loss + self.geometry_lambda * geom_loss
        
        # Collect outputs
        out = {
            "loss": total_loss,
            "pred_loss": pred_loss.detach(),
            "geom_loss": geom_loss.detach(),
            "pairs": (b_idx, t_idx, tpos, k_ids),
        }
        
        # Add geometry metrics
        for key, val in geom_metrics.items():
            out[f"geom_{key}"] = val
        
        return out
    
    @torch.no_grad()
    def compute_latents(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Utility for validation: compute predicted and target latents.
        
        Args:
            h: Hidden states (B, T, C)
            pairs: Optional pre-sampled pairs
            
        Returns:
            z_pred: Predicted latents (N, D)
            z_tgt: Target latents (N, D)
            k_ids: Horizon category ids (N,)
        """
        B, T, C = h.shape
        device = h.device
        
        # Sample or use provided pairs
        if pairs is None:
            b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device)
        else:
            b_idx, t_idx, tpos, k_ids = pairs
            if b_idx.device != device:
                b_idx = b_idx.to(device, non_blocking=True)
                t_idx = t_idx.to(device, non_blocking=True)
                tpos = tpos.to(device, non_blocking=True)
                k_ids = k_ids.to(device, non_blocking=True)
        
        h_anchor = h[b_idx, t_idx, :]
        h_target = h[b_idx, tpos, :]
        
        z_anchor = self.projector(h_anchor)
        z_horizon = self.horizon_emb(k_ids)
        z_pred = self.predictor(torch.cat([z_anchor, z_horizon], dim=-1))
        z_tgt = self.projector(h_target)
        
        return z_pred, z_tgt, k_ids
