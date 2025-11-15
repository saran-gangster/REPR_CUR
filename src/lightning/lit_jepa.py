"""
Minimal Lightning module for JEPA-augmented LM training.

Simplified design following LeJEPA principles:
- LM loss + α * JEPA loss (single scalar trade-off)
- No gradient barrier (both losses backprop through entire network)
- No complex scheduling or teacher coupling
- Minimal logging for clarity
"""

from typing import Any, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from src.models.transformer import DecoderOnlyTransformer
from src.models.jepa import JEPAObjective


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Warmup + cosine decay learning rate schedule."""
    
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = float(step) / max(1, self.warmup_steps)
        else:
            progress = float(step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


class LitJEPA(L.LightningModule):
    """
    Minimal JEPA-augmented language model.
    
    Total loss: lm_loss + α * jepa_loss
    
    Where jepa_loss = (1 - λ) * prediction_loss + λ * geometry_loss
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        use_rope: bool = True,
        rope_base: float = 10000,
        weight_tying: bool = True,
        jepa: Dict[str, Any] = None,
        optimizer: Dict[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        jepa_cfg = jepa or {}
        
        # Transformer LM backbone
        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            ff_multiplier=ff_multiplier,
            use_rope=use_rope,
            rope_base=rope_base,
            weight_tying=weight_tying,
        )
        
        # Minimal JEPA objective
        self.jepa = JEPAObjective(
            d_model=d_model,
            latent_dim=jepa_cfg.get("latent_dim", d_model),
            predictor_hidden_multiplier=jepa_cfg.get("predictor_hidden_multiplier", 2.0),
            horizons=jepa_cfg.get("horizons", [2, 8, 32, 64]),
            horizon_probs=jepa_cfg.get("horizon_probs", [0.4, 0.3, 0.2, 0.1]),
            pairs_per_seq=jepa_cfg.get("pairs_per_seq", 64),
            dropout=dropout,
            prediction_loss_type=jepa_cfg.get("prediction_loss_type", "cosine"),
            geometry_regularizer=jepa_cfg.get("geometry_regularizer", "vicreg"),
            geometry_lambda=jepa_cfg.get("geometry_lambda", 0.2),
            vicreg_var_weight=jepa_cfg.get("vicreg_var_weight", 1.0),
            vicreg_cov_weight=jepa_cfg.get("vicreg_cov_weight", 0.1),
            sigreg_num_directions=jepa_cfg.get("sigreg_num_directions", 128),
        )
        
        # JEPA weight (α)
        self.alpha = float(jepa_cfg.get("alpha", 0.1))
        
        # Tap layer for JEPA
        self.jepa_tap_layer = jepa_cfg.get("tap_layer", -3)
        self.jepa_tap_norm = bool(jepa_cfg.get("tap_norm", False))
        
        # Optimizer config
        self.optim_cfg = optimizer or {
            "lr": 3e-4,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
            "warmup_steps": 200,
        }
        
    def _lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Compute LM logits with weight tying if enabled."""
        logits = self.model.lm_head(h)
        if getattr(self.model, "weight_tying", False):
            ls = getattr(self.model, "logit_scale", None)
            if ls is not None:
                logits = logits * ls
            b = getattr(self.model, "output_bias", None)
            if b is not None:
                logits = logits + b
        return logits
    
    def _forward_with_tap(self, x: torch.Tensor):
        """Forward pass with tap for JEPA."""
        return self.model(
            x,
            tap_layer=self.jepa_tap_layer,
            return_tap=True,
            grad_barrier=False,  # No gradient barrier
            tap_norm=self.jepa_tap_norm,
        )
    
    def forward(self, x):
        """Standard forward for inference."""
        return self.model(x)
    
    def _lm_loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss."""
        targets = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch  # (B, T)
        
        # Forward with tap for JEPA
        (h_tap, _), h_final = self._forward_with_tap(x)
        
        # LM loss
        logits_final = self._lm_logits(h_final)
        lm_logits = logits_final[:, :-1, :]
        lm_loss = self._lm_loss(lm_logits, x)
        
        # JEPA loss
        jepa_out = self.jepa(h_tap)
        jepa_loss = jepa_out["loss"]
        
        # Total loss
        total_loss = lm_loss + self.alpha * jepa_loss
        
        # Logging
        self.log("train/lm_loss", lm_loss, on_step=True)
        self.log("train/jepa_loss", jepa_loss, on_step=True)
        self.log("train/pred_loss", jepa_out["pred_loss"], on_step=True)
        self.log("train/geom_loss", jepa_out["geom_loss"], on_step=True)
        self.log("train/total_loss", total_loss, on_step=True)
        
        # Geometry metrics
        if "geom_mean_std" in jepa_out:
            self.log("train/geom_mean_std", jepa_out["geom_mean_std"], on_step=True)
        if "geom_var_loss" in jepa_out:
            self.log("train/geom_var_loss", jepa_out["geom_var_loss"], on_step=True)
        if "geom_cov_loss" in jepa_out:
            self.log("train/geom_cov_loss", jepa_out["geom_cov_loss"], on_step=True)
        if "geom_moment2_loss" in jepa_out:
            self.log("train/geom_moment2_loss", jepa_out["geom_moment2_loss"], on_step=True)
        if "geom_moment4_loss" in jepa_out:
            self.log("train/geom_moment4_loss", jepa_out["geom_moment4_loss"], on_step=True)
        
        return total_loss
    
    def on_validation_epoch_start(self):
        """Cache validation pairs for deterministic validation."""
        pass
    
    def _get_or_make_val_pairs(self, B: int, T: int, device: torch.device):
        """Get or create cached validation pairs."""
        pairs = getattr(self, "_val_pairs", None)
        if pairs is not None:
            b_idx, t_idx, tpos, _ = pairs
            shape_ok = (
                b_idx.numel() == self.jepa.pairs_per_seq * B
                and t_idx.max().item() < T
                and tpos.max().item() < T
            )
            device_ok = b_idx.device == device
            if shape_ok and device_ok:
                return pairs
        
        # Create new cached pairs
        g = torch.Generator(device=device)
        g.manual_seed(12345)
        self._val_pairs = self.jepa._sample_pairs(B, T, device, generator=g)
        return self._val_pairs
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch
        B, T = x.shape
        
        # Forward with tap
        (h_tap, _), h_final = self._forward_with_tap(x)
        
        # LM loss
        logits_final = self._lm_logits(h_final)
        lm_logits = logits_final[:, :-1, :]
        lm_loss = self._lm_loss(lm_logits, x)
        
        # JEPA loss with cached pairs
        val_pairs = self._get_or_make_val_pairs(B, T, x.device)
        jepa_out = self.jepa(h_tap, pairs=val_pairs)
        jepa_loss = jepa_out["loss"]
        
        # Logging
        self.log("val/ppl", torch.exp(lm_loss), on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/lm_loss", lm_loss, on_epoch=True, sync_dist=True)
        self.log("val/jepa_loss", jepa_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/pred_loss", jepa_out["pred_loss"], on_epoch=True, sync_dist=True)
        self.log("val/geom_loss", jepa_out["geom_loss"], on_epoch=True, sync_dist=True)
        
        # Geometry metrics
        if "geom_mean_std" in jepa_out:
            self.log("val/geom_mean_std", jepa_out["geom_mean_std"], on_epoch=True, sync_dist=True)
        if "geom_var_loss" in jepa_out:
            self.log("val/geom_var_loss", jepa_out["geom_var_loss"], on_epoch=True, sync_dist=True)
        if "geom_cov_loss" in jepa_out:
            self.log("val/geom_cov_loss", jepa_out["geom_cov_loss"], on_epoch=True, sync_dist=True)
        if "geom_moment2_loss" in jepa_out:
            self.log("val/geom_moment2_loss", jepa_out["geom_moment2_loss"], on_epoch=True, sync_dist=True)
        if "geom_moment4_loss" in jepa_out:
            self.log("val/geom_moment4_loss", jepa_out["geom_moment4_loss"], on_epoch=True, sync_dist=True)
    
    def on_fit_start(self):
        """Setup at training start."""
        # Match vocab to datamodule if needed
        try:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "vocab_size", None):
                new_V = int(dm.vocab_size)
                old_V = self.model.lm_head.out_features
                if new_V != old_V:
                    d_model = self.model.token_emb.embedding_dim
                    device = self.device
                    
                    # Recreate token embedding
                    tok = nn.Embedding(new_V, d_model).to(device)
                    self.model.token_emb = tok
                    
                    # Recreate LM head
                    head = nn.Linear(d_model, new_V, bias=False).to(device)
                    if getattr(self.model, "weight_tying", False):
                        head.weight = self.model.token_emb.weight
                    self.model.lm_head = head
                    
                    # Rebuild tied extras
                    if getattr(self.model, "weight_tying", False):
                        scale_init = float(getattr(self.model, "logit_scale_init", 1.0))
                        self.model.logit_scale = nn.Parameter(torch.tensor(scale_init, device=device))
                        self.model.output_bias = nn.Parameter(torch.zeros(new_V, device=device))
                    else:
                        self.model.logit_scale = None
                        self.model.output_bias = None
                    
                    self.hparams.vocab_size = new_V
        except Exception:
            pass
        
        # Enable TF32 for speed
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # W&B model watching
        try:
            from lightning.pytorch.loggers import WandbLogger
            from lightning.pytorch.loggers.logger import LoggerCollection
            logger = self.logger
            if isinstance(logger, LoggerCollection):
                for lg in logger:
                    if isinstance(lg, WandbLogger):
                        lg.watch(self, log="gradients", log_freq=200)
            elif isinstance(logger, WandbLogger):
                logger.watch(self, log="gradients", log_freq=200)
        except Exception:
            pass
    
    def configure_optimizers(self):
        """Setup optimizer and scheduler."""
        wd = self.optim_cfg.get("weight_decay", 0.1)
        betas = self.optim_cfg.get("betas", (0.9, 0.95))
        lr = self.optim_cfg.get("lr", 3e-4)
        warmup = self.optim_cfg.get("warmup_steps", 200)
        
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []
        
        for p in self.parameters():
            if p.requires_grad:
                if p.ndim < 2:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        
        param_groups = [
            {"params": decay_params, "lr": lr, "weight_decay": wd},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        
        if self.trainer.max_steps is None:
            return optimizer
        
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=warmup,
            max_steps=self.trainer.max_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
