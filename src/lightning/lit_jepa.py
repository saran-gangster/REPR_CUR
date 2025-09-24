from typing import Any, Dict
import os
import math
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
        # optional weight tying (default True for back-compat; config sets it False for true decoupling)
        weight_tying: bool = True,
        jepa: Dict[str, Any] = None,
        optimizer: Dict[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['optimizer'])
        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, ff_multiplier=ff_multiplier, use_rope=use_rope, rope_base=rope_base,
            weight_tying=weight_tying,
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
        # Independent weights (back-compat with lambda_weight)
        self.jepa_weight = jepa_cfg.get("jepa_weight", jepa_cfg.get("lambda_weight", 0.1))
        self.lm_weight_final = jepa_cfg.get("lm_weight", 1.0)

        # Gradient decoupling controls
        self.jepa_tap_layer = jepa_cfg.get("tap_layer", -2)
        self.jepa_grad_barrier = bool(jepa_cfg.get("grad_barrier", True))
        self.jepa_tap_norm = bool(jepa_cfg.get("tap_norm", False))

        # Schedules
        self.lm_warmup_steps = int(jepa_cfg.get("lm_warmup_steps", 0))
        ema_sched = jepa_cfg.get("ema_schedule", None)
        if isinstance(ema_sched, (list, tuple)) and len(ema_sched) == 2:
            self.ema_start, self.ema_end = float(ema_sched[0]), float(ema_sched[1])
        else:
            self.ema_start, self.ema_end = self.jepa.ema_momentum, self.jepa.ema_momentum
        self.ema_sched_steps = int(jepa_cfg.get("ema_schedule_steps", self.lm_warmup_steps if self.lm_warmup_steps > 0 else 1))

        self.optim_cfg = optimizer or {"lr": 3e-4, "weight_decay": 0.1, "betas": (0.9, 0.95), "warmup_steps": 200}

    def forward(self, x):
        return self.model(x)  # (B, T, C)

    def _lm_loss(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(h[:, :-1, :])            # (B, T-1, V)
        targets = x[:, 1:]                                   # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return loss

    def _current_lm_weight(self, step: int) -> float:
        if self.lm_weight_final <= 0.0:
            return 0.0
        warm = max(0, self.lm_warmup_steps)
        max_steps = self.trainer.max_steps or (warm + 1)
        if step < warm:
            return 0.0
        frac = float(step - warm) / max(1, max_steps - warm)
        frac = max(0.0, min(1.0, frac))
        return self.lm_weight_final * frac

    def _current_ema_momentum(self, step: int) -> float:
        if self.ema_sched_steps <= 0 or self.ema_start == self.ema_end:
            return self.jepa.ema_momentum
        t = max(0.0, min(1.0, float(step) / float(self.ema_sched_steps)))
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))  # cosine ramp 0->1
        return self.ema_start + (self.ema_end - self.ema_start) * alpha

    def training_step(self, batch, batch_idx):
        x = batch  # (B,T) token ids

        # Decoupled features: JEPA from tap layer, LM from final layer
        h_jepa, h_final = self.model(
            x, tap_layer=self.jepa_tap_layer, return_tap=True,
            grad_barrier=self.jepa_grad_barrier, tap_norm=self.jepa_tap_norm
        )

        # Losses
        lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=self.device)
        jepa_out = self.jepa(h_jepa)

        # Scheduled LM weight
        eff_lm_w = self._current_lm_weight(self.global_step)
        total_loss = eff_lm_w * lm_loss + self.jepa_weight * jepa_out["loss"]

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/lm_loss", lm_loss, on_step=True)
        self.log("train/jepa_loss", jepa_out["loss"], on_step=True)
        self.log("train/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_step=True)
        self.log("train/cos_loss", jepa_out["cos_loss"], on_step=True)
        self.log("train/var_loss", jepa_out["var_loss"], on_step=True)
        self.log("train/cov_loss", jepa_out["cov_loss"], on_step=True)
        self.log("train/std_tgt", jepa_out["std_tgt"], on_step=True)
        if "std_anchor" in jepa_out:
            self.log("train/std_anchor", jepa_out["std_anchor"], on_step=True)
        self.log("train/std_pred", jepa_out["std_pred"], on_step=True)
        self.log("train/num_pairs", jepa_out["num_pairs"], on_step=True)
        self.log("train/lm_weight", torch.tensor(eff_lm_w, device=self.device), on_step=True)
        self.log("train/ema_momentum", torch.tensor(self.jepa.ema_momentum, device=self.device), on_step=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        with torch.no_grad():
            h_jepa, h_final = self.model(
                x, tap_layer=self.jepa_tap_layer, return_tap=True,
                grad_barrier=self.jepa_grad_barrier, tap_norm=self.jepa_tap_norm
            )
            lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=self.device)
            jepa_out = self.jepa(h_jepa)

            eff_lm_w = self._current_lm_weight(self.global_step)
            total_loss = eff_lm_w * lm_loss + self.jepa_weight * jepa_out["loss"]

            # Compute JEPA validation diagnostics with within-horizon negatives
            z_pred, z_tgt, k_ids = self.jepa.compute_latents(h_jepa)  # (N,D), (N,D), (N,)
            p = F.normalize(z_pred, dim=-1)
            y = F.normalize(z_tgt, dim=-1)

            unique_hids = torch.unique(k_ids)
            total_n = 0
            correct_sum = 0.0
            dtrue_sum = 0.0
            dimp_sum = 0.0

            # Per-horizon logging and aggregation
            for hid in unique_hids:
                mask = (k_ids == hid)
                n_h = int(mask.sum().item())
                horizon_idx = (k_ids == hid).nonzero(as_tuple=True)[0]
                if n_h >= 2:
                    p_h = p[horizon_idx]
                    y_h = y[horizon_idx]
                    y_imp_h = torch.roll(y_h, shifts=1, dims=0)

                    d_true_h = 1.0 - (p_h * y_h).sum(dim=-1)
                    d_imp_h = 1.0 - (p_h * y_imp_h).sum(dim=-1)

                    acc_h = (d_true_h < d_imp_h).float().mean()
                    self.log(f"val/imposter_acc_h{int(self.jepa.horizons[int(hid)])}", acc_h, on_epoch=True, sync_dist=True)

                    correct_sum += (d_true_h < d_imp_h).float().sum()
                    dtrue_sum += d_true_h.sum()
                    dimp_sum += d_imp_h.sum()
                    total_n += n_h
                else:
                    self.log(f"val/imposter_acc_h{int(self.jepa.horizons[int(hid)])}", torch.tensor(float('nan'), device=self.device), on_epoch=True, sync_dist=True)

            if total_n >= 2:
                dist_true = dtrue_sum / total_n
                dist_imposter = dimp_sum / total_n
                imposter_acc = correct_sum / total_n
            else:
                y_imp = torch.roll(y, shifts=1, dims=0)
                d_true_all = 1.0 - (p * y).sum(dim=-1)
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
        if "std_anchor" in jepa_out:
            self.log("val/std_anchor", jepa_out["std_anchor"], on_epoch=True, sync_dist=True)
        self.log("val/std_pred", jepa_out["std_pred"], on_epoch=True, sync_dist=True)

        # Aggregated within-horizon metrics
        self.log("val/imposter_acc", imposter_acc, on_epoch=True, sync_dist=True)
        self.log("val/dist_true", dist_true, on_epoch=True, sync_dist=True)
        self.log("val/dist_imposter", dist_imposter, on_epoch=True, sync_dist=True)
        if self.lm_weight_final > 0.0:
            self.log("val/ppl", torch.exp(lm_loss), on_epoch=True, sync_dist=True)
        self.log("val/lm_weight", torch.tensor(eff_lm_w, device=self.device), on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        try:
            self.jepa.ema_momentum = float(self._current_ema_momentum(self.global_step))
        except Exception:
            pass
        self.jepa.momentum_update()

    def on_fit_start(self):
        # Toggle grads for modules that are fully unused to avoid DDP unused-param errors
        try:
            if self.lm_weight_final <= 0.0:
                for p in self.model.lm_head.parameters():
                    p.requires_grad = False
            if self.jepa_weight <= 0.0:
                for p in self.jepa.parameters():
                    p.requires_grad = False
        except Exception:
            pass

        # match vocab to datamodule automatically (so BPE/synthetic "just work")
        try:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "vocab_size", None):
                new_V = int(dm.vocab_size)
                old_V = self.model.lm_head.out_features
                if new_V != old_V:
                    d_model = self.model.token_emb.embedding_dim
                    device = self.device
                    tok = nn.Embedding(new_V, d_model).to(device)
                    self.model.token_emb = tok
                    head = nn.Linear(d_model, new_V, bias=False).to(device)
                    if getattr(self.model, "weight_tying", False):
                        head.weight = self.model.token_emb.weight
                    self.model.lm_head = head
                    self.hparams.vocab_size = new_V
        except Exception:
            pass

        # Ensure a sane EMA schedule length if unspecified or degenerate
        try:
            if getattr(self, "ema_sched_steps", None) in (None, 0, 1):
                try:
                    self.ema_sched_steps = int(self.trainer.max_steps or 1000)
                except Exception:
                    self.ema_sched_steps = 1000
        except Exception:
            self.ema_sched_steps = 1000

        # Initialize EMA to schedule start
        try:
            self.jepa.ema_momentum = float(self._current_ema_momentum(0))
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
        lr_backbone = self.optim_cfg.get("lr", 3e-4)  # backbone LR
        lr_head = self.optim_cfg.get("lr_head", lr_backbone / 10.0)  # head/embedding LR
        warmup = self.optim_cfg.get("warmup_steps", 200)

        # Parameter partition
        head_params = list(self.model.token_emb.parameters()) + list(self.model.lm_head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.parameters() if id(p) not in head_ids]

        def split_decay(params):
            decay, no_decay = [], []
            for p in params:
                if p is None or not p.requires_grad:
                    continue
                # 1D params (norm scales, biases) -> no weight decay
                if p.ndim == 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        head_decay, head_no_decay = split_decay(head_params)
        bb_decay, bb_no_decay = split_decay(backbone_params)

        param_groups = [
            {"params": head_decay, "lr": lr_head, "weight_decay": wd},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
            {"params": bb_decay, "lr": lr_backbone, "weight_decay": wd},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        ]

        AdamW = torch.optim.AdamW
        optimizer = AdamW(param_groups, betas=betas)  # lr is per-group now

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
            
    # Per-branch gradient clipping; compatible with Lightning variants (with or without optimizer_idx arg)
    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx: int | None = None,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
        *args,
        **kwargs,
    ):
        # Some Lightning versions pass only (optimizer, gradient_clip_val, gradient_clip_algorithm)
        # If so, shift arguments accordingly
        if optimizer_idx is not None and isinstance(optimizer_idx, (float, int)) and gradient_clip_val is None:
            # Called as (optimizer, gradient_clip_val, gradient_clip_algorithm)
            gradient_clip_val = float(optimizer_idx)
            optimizer_idx = None
            gradient_clip_algorithm = kwargs.get("gradient_clip_algorithm", gradient_clip_algorithm)

        try:
            clip_val = float(gradient_clip_val) if gradient_clip_val is not None else 0.0
        except Exception:
            clip_val = 0.0
        if clip_val is None or clip_val <= 0.0:
            return  # nothing to do

        try:
            # Determine tap index as used in the forward path
            n_layers = getattr(self.model, "n_layers", None)
            tap_layer = self.jepa_tap_layer
            if n_layers is None or tap_layer is None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val)
                return
            tap_idx = tap_layer if tap_layer >= 0 else n_layers + tap_layer
            tap_idx = max(0, min(n_layers - 1, tap_idx))

            lower_params = []
            upper_params = []

            # Lower: token_emb + blocks[0..tap_idx] + norm_tap + JEPA heads
            lower_params += list(self.model.token_emb.parameters())
            for i, blk in enumerate(self.model.blocks):
                if i <= tap_idx:
                    lower_params += list(blk.parameters())
                else:
                    upper_params += list(blk.parameters())
            # norms/bridges
            lower_params += list(self.model.norm_tap.parameters())
            upper_params += list(self.model.lm_bridge.parameters())
            upper_params += list(self.model.norm_f.parameters())
            upper_params += list(self.model.lm_head.parameters())

            # JEPA heads belong to lower branch
            lower_params += list(self.jepa.online_proj.parameters())
            lower_params += list(self.jepa.predictor.parameters())
            lower_params += list(self.jepa.horizon_emb_latent.parameters())
            # target_proj is EMA/no-grad

            # Deduplicate to be safe
            def uniq(params):
                seen = set()
                out = []
                for p in params:
                    if p is None:
                        continue
                    pid = id(p)
                    if pid not in seen:
                        seen.add(pid)
                        out.append(p)
                return out

            lower_params = uniq(lower_params)
            upper_params = uniq(upper_params)

            # Apply per-branch clipping
            torch.nn.utils.clip_grad_norm_(lower_params, max_norm=clip_val, norm_type=2.0)
            torch.nn.utils.clip_grad_norm_(upper_params, max_norm=clip_val, norm_type=2.0)

        except Exception:
            try:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val, norm_type=2.0)
            except Exception:
                pass