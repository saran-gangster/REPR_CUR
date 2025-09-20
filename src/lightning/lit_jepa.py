from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from src.models.transformer import DecoderOnlyTransformer
from src.models.jepa import JEPAObjective

class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
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
            scale = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return [base_lr * scale for base_lr in self.base_lrs]

class LitJEPA(L.LightningModule):
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
        jepa: Dict[str, Any] = None,
        optimizer: Dict[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['optimizer'])
        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, ff_multiplier=ff_multiplier, use_rope=use_rope, rope_base=rope_base
        )
        jepa_cfg = jepa or {}
        self.jepa = JEPAObjective(
            d_model=d_model,
            latent_dim=jepa_cfg.get("latent_dim", d_model),
            predictor_hidden_multiplier=jepa_cfg.get("predictor_hidden_multiplier", 2),
            horizons=jepa_cfg.get("horizons", [8, 32, 128]),
            horizon_probs=jepa_cfg.get("horizon_probs", [0.5, 0.3, 0.2]),
            pairs_per_seq=jepa_cfg.get("pairs_per_seq", 64),
            ema_momentum=jepa_cfg.get("ema_momentum", 0.996),
            gamma_var=jepa_cfg.get("gamma_var", 1.0),
            gamma_cov=jepa_cfg.get("gamma_cov", 1.0),
            dropout=dropout,
        )
        self.lambda_weight = jepa_cfg.get("lambda_weight", 0.1)
        self.optim_cfg = optimizer or {"lr": 3e-4, "weight_decay": 0.1, "betas": (0.9, 0.95), "warmup_steps": 200}

    def forward(self, x):
        return self.model(x)  # (B, T, C)

    def _lm_loss(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # h: (B, T, C), predict x[:, 1:] from h[:, :-1, :]
        logits = self.model.lm_head(h[:, :-1, :])            # (B, T-1, V)
        targets = x[:, 1:]                                   # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        x = batch  # (B,T) token ids
        h = self(x)

        # Losses
        lm_loss = self._lm_loss(h, x)
        jepa_out = self.jepa(h)
        total_loss = lm_loss + self.lambda_weight * jepa_out["loss"]

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/lm_loss", lm_loss, on_step=True)
        self.log("train/jepa_loss", jepa_out["loss"], on_step=True)
        self.log("train/cos_loss", jepa_out["cos_loss"], on_step=True)
        self.log("train/var_loss", jepa_out["var_loss"], on_step=True)
        self.log("train/cov_loss", jepa_out["cov_loss"], on_step=True)
        self.log("train/num_pairs", jepa_out["num_pairs"], on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        with torch.no_grad():
            h = self(x)
            lm_loss = self._lm_loss(h, x)
            jepa_out = self.jepa(h)
            total_loss = lm_loss + self.lambda_weight * jepa_out["loss"]

        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/lm_loss", lm_loss, on_epoch=True)
        self.log("val/jepa_loss", jepa_out["loss"], on_epoch=True)
        self.log("val/cos_loss", jepa_out["cos_loss"], on_epoch=True)
        self.log("val/var_loss", jepa_out["var_loss"], on_epoch=True)
        self.log("val/cov_loss", jepa_out["cov_loss"], on_epoch=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update after optimizer step
        self.jepa.momentum_update()

    def configure_optimizers(self):
        wd = self.optim_cfg.get("weight_decay", 0.1)
        betas = self.optim_cfg.get("betas", (0.9, 0.95))
        lr = self.optim_cfg.get("lr", 3e-4)
        warmup = self.optim_cfg.get("warmup_steps", 200)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd)

        if self.trainer.max_steps is None:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            return optimizer
        else:
            scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup, max_steps=self.trainer.max_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                },
            }