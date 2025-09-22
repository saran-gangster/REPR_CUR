from typing import Any, Dict
import os
import math
import inspect
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
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
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
            predictor_hidden_multiplier=jepa_cfg.get("predictor_hidden_multiplier", 2.0),
            horizons=jepa_cfg.get("horizons", [8, 32, 128]),
            horizon_probs=jepa_cfg.get("horizon_probs", [0.5, 0.3, 0.2]),
            pairs_per_seq=jepa_cfg.get("pairs_per_seq", 64),
            ema_momentum=jepa_cfg.get("ema_momentum", 0.996),
            gamma_var=jepa_cfg.get("gamma_var", 1.0),
            gamma_cov=jepa_cfg.get("gamma_cov", 1.0),
            dropout=dropout,
            tau=jepa_cfg.get("tau", 0.2),
        )
        # New: independent weights (back-compat with lambda_weight)
        self.jepa_weight = jepa_cfg.get("jepa_weight", jepa_cfg.get("lambda_weight", 0.1))
        self.lm_weight = jepa_cfg.get("lm_weight", 1.0)

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
        if self.lm_weight > 0.0:
            lm_loss = self._lm_loss(h, x)
        else:
            lm_loss = torch.tensor(0.0, device=self.device)

        jepa_out = self.jepa(h)
        total_loss = self.lm_weight * lm_loss + self.jepa_weight * jepa_out["loss"]

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/lm_loss", lm_loss, on_step=True)
        self.log("train/jepa_loss", jepa_out["loss"], on_step=True)
        self.log("train/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_step=True)
        self.log("train/cos_loss", jepa_out["cos_loss"], on_step=True)
        self.log("train/var_loss", jepa_out["var_loss"], on_step=True)
        self.log("train/cov_loss", jepa_out["cov_loss"], on_step=True)
        self.log("train/std_tgt", jepa_out["std_tgt"], on_step=True)
        self.log("train/std_pred", jepa_out["std_pred"], on_step=True)
        self.log("train/num_pairs", jepa_out["num_pairs"], on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        with torch.no_grad():
            h = self(x)
            if self.lm_weight > 0.0:
                lm_loss = self._lm_loss(h, x)
            else:
                lm_loss = torch.tensor(0.0, device=self.device)

            jepa_out = self.jepa(h)
            total_loss = self.lm_weight * lm_loss + self.jepa_weight * jepa_out["loss"]

            # Imposter validation
            z_pred, z_tgt, k_ids = self.jepa.compute_latents(h)
            p = F.normalize(z_pred, dim=-1)
            y = F.normalize(z_tgt, dim=-1)
            d_true_all = 1.0 - (p * y).sum(dim=-1)  # (N,)

            # "Imposter" targets: roll by 1 (alternative: random permutation)
            y_imp = torch.roll(y, shifts=1, dims=0)
            d_imp_all = 1.0 - (p * y_imp).sum(dim=-1)

            dist_true = d_true_all.mean()
            dist_imposter = d_imp_all.mean()
            imposter_acc = (d_true_all < d_imp_all).float().mean()

        # Epoch-level logs must sync across devices
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/lm_loss", lm_loss, on_epoch=True, sync_dist=True)
        self.log("val/jepa_loss", jepa_out["loss"], on_epoch=True, sync_dist=True)
        self.log("val/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_epoch=True, sync_dist=True)
        self.log("val/cos_loss", jepa_out["cos_loss"], on_epoch=True, sync_dist=True)
        self.log("val/var_loss", jepa_out["var_loss"], on_epoch=True, sync_dist=True)
        self.log("val/cov_loss", jepa_out["cov_loss"], on_epoch=True, sync_dist=True)
        self.log("val/std_tgt", jepa_out["std_tgt"], on_epoch=True, sync_dist=True)
        self.log("val/std_pred", jepa_out["std_pred"], on_epoch=True, sync_dist=True)

        # New logs
        self.log("val/imposter_acc", imposter_acc, on_epoch=True, sync_dist=True)
        self.log("val/dist_true", dist_true, on_epoch=True, sync_dist=True)
        self.log("val/dist_imposter", dist_imposter, on_epoch=True, sync_dist=True)
        if self.lm_weight > 0.0:
            self.log("val/ppl", torch.exp(lm_loss), on_epoch=True, sync_dist=True)

        # Per-horizon breakdown (diagnostics)
        with torch.no_grad():
            for hid, horizon in enumerate(self.jepa.horizons):
                mask = (k_ids == hid)
                if mask.any():
                    acc_h = (d_true_all[mask] < d_imp_all[mask]).float().mean()
                    self.log(f"val/imposter_acc_h{horizon}", acc_h, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.jepa.momentum_update()

    def on_fit_start(self):
        # match vocab to datamodule automatically (so BPE/synthetic "just work")
        try:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "vocab_size", None):
                new_V = int(dm.vocab_size)
                # The model has an LM head even if LM loss is off; tie to token_emb.
                old_V = self.model.lm_head.out_features
                if new_V != old_V:
                    d_model = self.model.token_emb.embedding_dim
                    device = self.device
                    tok = nn.Embedding(new_V, d_model).to(device)
                    self.model.token_emb = tok
                    head = nn.Linear(d_model, new_V, bias=False).to(device)
                    head.weight = self.model.token_emb.weight  # tie weights
                    self.model.lm_head = head
                    # Update saved hyperparams for clarity
                    self.hparams.vocab_size = new_V
        except Exception:
            pass

        # Enable TF32 on Ampere+ GPUs for additional speed
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # If WandbLogger is active, watch the model to log gradients/params
        try:
            from lightning.pytorch.loggers import WandbLogger
            from lightning.pytorch.loggers.logger import LoggerCollection
        except Exception:
            return
        logger = self.logger
        if isinstance(logger, LoggerCollection):
            for lg in logger:
                if isinstance(lg, WandbLogger):
                    lg.watch(self, log="gradients", log_freq=200)
        elif isinstance(logger, WandbLogger):
            logger.watch(self, log="gradients", log_freq=200)

    def setup(self, stage: str):
        # Optional compilation for extra speed on CUDA: export TORCH_COMPILE=1
        if hasattr(torch, "compile") and os.getenv("TORCH_COMPILE", "0") == "1":
            if torch.cuda.is_available():
                mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
                try:
                    self.model = torch.compile(self.model, mode=mode)
                except Exception:
                    pass

    def configure_optimizers(self):
        wd = self.optim_cfg.get("weight_decay", 0.1)
        betas = self.optim_cfg.get("betas", (0.9, 0.95))
        lr = self.optim_cfg.get("lr", 3e-4)
        warmup = self.optim_cfg.get("warmup_steps", 200)

        AdamW = torch.optim.AdamW
        optimizer = None
        try:
            # Use fused AdamW on CUDA if available
            if "fused" in inspect.signature(AdamW).parameters and torch.cuda.is_available():
                optimizer = AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd, fused=True)
        except Exception:
            optimizer = None
        if optimizer is None:
            optimizer = AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd)

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