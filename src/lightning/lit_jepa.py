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
        weight_tying: bool = True,
        jepa: Dict[str, Any] = None,
        optimizer: Dict[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['optimizer'])

        jepa_cfg = jepa or {}

        # Transformer
        self.simple_recurrence_steps = int(jepa_cfg.get("simple_recurrence_steps", 0))
        self.recur_steps = self.simple_recurrence_steps
        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, ff_multiplier=ff_multiplier, use_rope=use_rope, rope_base=rope_base,
            weight_tying=weight_tying,
            simple_recurrence_steps=0,
        )
        self.model.simple_recurrence_steps = self.simple_recurrence_steps

        # JEPA
        self.jepa = JEPAObjective(
            d_model=d_model,
            latent_dim=jepa_cfg.get("latent_dim", d_model),
            predictor_hidden_multiplier=jepa_cfg.get("predictor_hidden_multiplier", 2.0),
            horizons=jepa_cfg.get("horizons", [1, 2, 8, 32, 64]),
            horizon_probs=jepa_cfg.get("horizon_probs", [0.5, 0.25, 0.15, 0.06, 0.04]),
            pairs_per_seq=jepa_cfg.get("pairs_per_seq", 64),
            ema_momentum=jepa_cfg.get("ema_momentum", 0.996),
            gamma_var=jepa_cfg.get("gamma_var", 1.0),
            gamma_cov=jepa_cfg.get("gamma_cov", 0.2),
            dropout=dropout,
            tau=jepa_cfg.get("tau", 0.2),
            gamma_var_pred=jepa_cfg.get("gamma_var_pred", 0.0),
            gamma_cov_pred=jepa_cfg.get("gamma_cov_pred", 0.0),
        )

        # JEPA/LM weights
        self.jepa_weight = jepa_cfg.get("jepa_weight", jepa_cfg.get("lambda_weight", 0.1))
        self.lm_weight_final = float(jepa_cfg.get("lm_weight", 1.0))

        # Baseline toggle (disables JEPA)
        self.run_baseline = bool(jepa_cfg.get("run_baseline", False))
        if self.run_baseline:
            self.jepa_weight = 0.0
            self.lm_weight_final = 1.0

        # LM weight scheduling
        self.lm_weight_use_scheduler = bool(jepa_cfg.get("lm_weight_use_scheduler", not self.run_baseline))
        self.lm_warmup_steps = int(jepa_cfg.get("lm_warmup_steps", 0))

        # Gradient decoupling controls
        self.jepa_tap_layer = jepa_cfg.get("tap_layer", -2)
        self.jepa_grad_barrier = bool(jepa_cfg.get("grad_barrier", True))
        self.jepa_tap_norm = bool(jepa_cfg.get("tap_norm", False))
        self.recur_at = int(jepa_cfg.get("recur_at", self.jepa_tap_layer))

        # EMA schedule
        ema_sched = jepa_cfg.get("ema_schedule", None)
        if isinstance(ema_sched, (list, tuple)) and len(ema_sched) == 2:
            self.ema_start, self.ema_end = float(ema_sched[0]), float(ema_sched[1])
        else:
            self.ema_start, self.ema_end = self.jepa.ema_momentum, self.jepa.ema_momentum
        self.ema_sched_steps = int(jepa_cfg.get("ema_schedule_steps", self.lm_warmup_steps if self.lm_warmup_steps > 0 else 1))

        # Optim config
        self.optim_cfg = optimizer or {"lr": 3e-4, "weight_decay": 0.1, "betas": (0.9, 0.95), "warmup_steps": 200}
    
    def _lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        # Base logits
        logits = self.model.lm_head(h)
        # If tied, apply learnable temperature and output bias
        if getattr(self.model, "weight_tying", False):
            ls = getattr(self.model, "logit_scale", None)
            if ls is not None:
                logits = logits * ls
            b = getattr(self.model, "output_bias", None)
            if b is not None:
                logits = logits + b
        return logits
    
    def forward(self, x):
        return self.model(x)  # (B, T, C)

    def _lm_loss(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        logits = self._lm_logits(h[:, :-1, :])               # (B, T-1, V)
        targets = x[:, 1:]                                   # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return loss

    def _current_lm_weight(self, step: int) -> float:
        # If scheduler is disabled, use lm_weight as a constant from step 0
        if not self.lm_weight_use_scheduler:
            return max(0.0, float(self.lm_weight_final))

        if self.lm_weight_final <= 0.0:
            return 0.0
        warm = max(0, self.lm_warmup_steps)
        max_steps = self.trainer.max_steps or (warm + 1)
        if step < warm:
            return 0.0
        frac = float(step - warm) / max(1, max_steps - warm)
        frac = max(0.0, min(1.0, frac))
        return float(self.lm_weight_final) * frac

    def _current_ema_momentum(self, step: int) -> float:
        if self.ema_sched_steps <= 0 or self.ema_start == self.ema_end:
            return self.jepa.ema_momentum
        t = max(0.0, min(1.0, float(step) / float(self.ema_sched_steps)))
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))  # cosine ramp 0->1
        return self.ema_start + (self.ema_end - self.ema_start) * alpha

    def training_step(self, batch, batch_idx):
        x = batch  # (B,T)
        use_jepa = self.jepa_weight > 0.0

        if use_jepa:
            (h_pred, h_teacher), h_final = self.model(
                x,
                tap_layer=self.jepa_tap_layer,
                return_tap=True,
                grad_barrier=self.jepa_grad_barrier,
                tap_norm=self.jepa_tap_norm,
                simple_recurrence_steps=self.recur_steps,
            )
        else:
            h_final = self.model(x, tap_layer=None, return_tap=False)
            h_pred = None
            h_teacher = None

        lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=self.device)

        if use_jepa:
            jepa_out = self.jepa(h_pred, teacher_h=h_teacher)
            jepa_loss = jepa_out["loss"]
        else:
            jepa_out = {}
            jepa_loss = torch.tensor(0.0, device=self.device)

        eff_lm_w = self._current_lm_weight(self.global_step)
        total_loss = eff_lm_w * lm_loss + self.jepa_weight * jepa_loss

        self.log("train/lm_loss", lm_loss, on_step=True)

        if use_jepa:
            self.log("train/jepa_loss", jepa_out["loss"], on_step=True)
            self.log("train/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_step=True)
            g = torch.sigmoid(self.model.lm_bridge_gate)
            self.log("train/bridge_gate", g, on_step=True)
            self.log("train/recur_steps", torch.tensor(float(self.recur_steps), device=self.device), on_step=True)
        else:
            zero = torch.tensor(0.0, device=self.device)
            self.log("train/jepa_loss", zero, on_step=True)
            self.log("train/info_nce_loss", zero, on_step=True)

        return total_loss
    
    def on_validation_epoch_start(self):
        pass
    
    def _get_or_make_val_pairs(self, B: int, T: int, device: torch.device):
        pairs = getattr(self, "_val_pairs", None)
        if pairs is not None:
            b_idx, t_idx, tpos, _ = pairs
            shape_ok = (b_idx.numel() == self.jepa.pairs_per_seq * B) and (t_idx.max().item() < T) and (tpos.max().item() < T)
            device_ok = (b_idx.device == device)
            if shape_ok and device_ok:
                return pairs

        g = torch.Generator(device=device)
        g.manual_seed(12345)
        self._val_pairs = self.jepa._sample_pairs(B, T, device, generator=g)
        return self._val_pairs
    
    def validation_step(self, batch, batch_idx):
        x = batch
        use_jepa = self.jepa_weight > 0.0

        with torch.no_grad():
            if use_jepa:
                (h_pred, h_teacher), h_final = self.model(
                    x,
                    tap_layer=self.jepa_tap_layer,
                    return_tap=True,
                    grad_barrier=self.jepa_grad_barrier,
                    tap_norm=self.jepa_tap_norm,
                    simple_recurrence_steps=self.recur_steps,
                )
            else:
                h_final = self.model(x, tap_layer=None, return_tap=False)
                h_pred = None
                h_teacher = None

            lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=self.device)

            if use_jepa:
                B, T = x.shape[0], x.shape[1]
                pairs = self._get_or_make_val_pairs(B, T, x.device)
                jepa_out = self.jepa(h_pred, pairs=pairs, teacher_h=h_teacher)
                jepa_loss = jepa_out["loss"]

                # Compute latents and per-horizon metrics
                z_pred, z_tgt, k_ids = self.jepa.compute_latents(h_pred, pairs=pairs, teacher_h=h_teacher)
                p = torch.nn.functional.normalize(z_pred, dim=-1)
                y = torch.nn.functional.normalize(z_tgt, dim=-1)
                unique_hids = torch.unique(k_ids)
                total_n, correct_sum, dtrue_sum, dneg_sum = 0, 0.0, 0.0, 0.0

                # Aggregators for normalized top1
                weighted_norm_sum = 0.0
                count_sum = 0

                for hid in unique_hids:
                    horizon_idx = (k_ids == hid).nonzero(as_tuple=True)[0]
                    n_h = int(horizon_idx.numel())
                    hval = int(self.jepa.horizons[int(hid)])
                    if n_h >= 2:
                        p_h, y_h = p[horizon_idx], y[horizon_idx]
                        sim = p_h @ y_h.T
                        pos_sim = sim.diag()

                        sim_neg = sim.clone()
                        sim_neg.fill_diagonal_(-float("inf"))
                        max_neg_sim, _ = sim_neg.max(dim=1)

                        acc_h = (pos_sim > max_neg_sim).float().mean()
                        self.log(f"val/top1_acc_h{hval}", acc_h, on_epoch=True, sync_dist=True)

                        # Normalized top-1
                        chance_h = 1.0 / float(n_h)
                        norm_top1_h = (acc_h - chance_h) / max(1.0 - chance_h, 1e-8)
                        self.log(f"val/norm_top1_h{hval}", norm_top1_h, on_epoch=True, sync_dist=True)

                        weighted_norm_sum += norm_top1_h * n_h
                        count_sum += n_h

                        # Accumulate for global top1 and distances
                        correct_sum += (pos_sim > max_neg_sim).float().sum()
                        dtrue_sum += (1.0 - pos_sim).sum()
                        dneg_sum += (1.0 - max_neg_sim).sum()
                        total_n += n_h

                if total_n >= 1:
                    top1_acc = correct_sum / total_n
                    dist_true = dtrue_sum / total_n
                    dist_imposter = dneg_sum / total_n
                    # Compute margin: positive margin means better separation
                    margin = dist_imposter - dist_true
                else:
                    top1_acc = torch.tensor(0.0, device=self.device)
                    margin = torch.tensor(0.0, device=self.device)

                # Global normalized top-1
                if count_sum > 0:
                    self.log("val/norm_top1", torch.tensor(weighted_norm_sum / count_sum, device=self.device),
                            on_epoch=True, sync_dist=True)

                jepa_loss = jepa_out["loss"]
            else:
                jepa_out, jepa_loss = {}, torch.tensor(0.0, device=self.device)
                top1_acc = torch.tensor(0.0, device=self.device)
                margin = torch.tensor(0.0, device=self.device)

        # Core validation metrics only
        self.log("val/top1_acc", top1_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ppl", torch.exp(lm_loss), on_epoch=True, sync_dist=True)
        
        if use_jepa:
            self.log("val/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_epoch=True, sync_dist=True)
            self.log("val/std_pred", jepa_out.get("std_pred", torch.tensor(0.0, device=self.device)), on_epoch=True, sync_dist=True)
            self.log("val/std_anchor", jepa_out.get("std_anchor", torch.tensor(0.0, device=self.device)), on_epoch=True, sync_dist=True)
            self.log("val/jepa_loss", jepa_loss, on_epoch=True, sync_dist=True)
            self.log("val/margin", margin, on_epoch=True, sync_dist=True)
                        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Only update EMA if JEPA is active
        if self.jepa_weight > 0.0:
            try:
                self.jepa.ema_momentum = float(self._current_ema_momentum(self.global_step))
                self.log("train/ema_momentum", torch.tensor(self.jepa.ema_momentum, device=self.device), on_step=True)
            except Exception:
                pass
            self.jepa.momentum_update()

    def on_fit_start(self):
        # Freeze unused modules to avoid DDP unused-param errors
        try:
            if self.lm_weight_final <= 0.0:
                for p in self.model.lm_head.parameters():
                    p.requires_grad = False

            if self.jepa_weight <= 0.0:
                for p in self.jepa.parameters():
                    p.requires_grad = False
                for p in self.model.norm_tap.parameters():
                    p.requires_grad = False
                for p in self.model.lm_bridge.parameters():
                    p.requires_grad = False
                if hasattr(self.model, "lm_bridge_gate") and self.model.lm_bridge_gate is not None:
                    self.model.lm_bridge_gate.requires_grad = False
            else:
                if not self.jepa_tap_norm and hasattr(self.model, "norm_tap"):
                    for p in self.model.norm_tap.parameters():
                        p.requires_grad = False
        except Exception:
            pass

        # Match vocab to datamodule automatically (BPE)
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

                    # Recreate LM head and tie as needed
                    head = nn.Linear(d_model, new_V, bias=False).to(device)
                    if getattr(self.model, "weight_tying", False):
                        head.weight = self.model.token_emb.weight
                    self.model.lm_head = head

                    # Rebuild tied softmax extras
                    if getattr(self.model, "weight_tying", False):
                        import math
                        self.model.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model), device=device))
                        self.model.output_bias = nn.Parameter(torch.zeros(new_V, device=device))
                    else:
                        self.model.logit_scale = None
                        self.model.output_bias = None

                    self.hparams.vocab_size = new_V
                    
        except Exception:
            pass

        # Ensure EMA schedule length sane
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

        # Enable TF32 where available
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # If WandbLogger is active, watch the model
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
        lr_backbone = self.optim_cfg.get("lr", 3e-4)
        lr_head = self.optim_cfg.get("lr_head", lr_backbone)
        warmup = self.optim_cfg.get("warmup_steps", 200)

        head_params: list[torch.nn.Parameter] = []
        head_ids: set[int] = set()

        def add_head_params(params):
            for p in params:
                if p is None or not p.requires_grad:
                    continue
                head_params.append(p)
                head_ids.add(id(p))

        # JEPA heads
        add_head_params(self.jepa.online_proj.parameters())
        add_head_params(self.jepa.predictor.parameters())
        add_head_params(self.jepa.horizon_emb_latent.parameters())

        if self.model.weight_tying:
            add_head_params(self.model.token_emb.parameters())
            add_head_params([getattr(self.model, "logit_scale", None)])
            add_head_params([getattr(self.model, "output_bias", None)])
        else:
            add_head_params(self.model.lm_head.parameters())

        backbone_params = [
            p for p in self.parameters()
            if p is not None and p.requires_grad and id(p) not in head_ids
        ]

        def split_decay(params):
            decay, no_decay = [], []
            for p in params:
                if p.ndim < 2:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        head_decay, head_no_decay = split_decay(head_params)
        bb_decay, bb_no_decay = split_decay(backbone_params)

        param_groups = []
        if head_decay or head_no_decay:
            param_groups.append({
                "params": head_decay + head_no_decay,
                "lr": lr_head,
                "weight_decay": 0.0,
            })
        if bb_decay:
            param_groups.append({
                "params": bb_decay,
                "lr": lr_backbone,
                "weight_decay": wd,
            })
        if bb_no_decay:
            param_groups.append({
                "params": bb_no_decay,
                "lr": lr_backbone,
                "weight_decay": 0.0,
            })

        optimizer = torch.optim.AdamW(param_groups, lr=lr_backbone, betas=betas)

        if self.trainer.max_steps is None:
            return optimizer

        scheduler = WarmupCosineLR(optimizer, warmup_steps=warmup, max_steps=self.trainer.max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
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
        if optimizer_idx is not None and isinstance(optimizer_idx, (float, int)) and gradient_clip_val is None:
            gradient_clip_val = float(optimizer_idx)
            optimizer_idx = None
            gradient_clip_algorithm = kwargs.get("gradient_clip_algorithm", gradient_clip_algorithm)

        try:
            clip_val = float(gradient_clip_val) if gradient_clip_val is not None else 0.0
        except Exception:
            clip_val = 0.0
        if clip_val is None or clip_val <= 0.0:
            return

        try:
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
            lower_params += list(self.model.norm_tap.parameters())

            # Upper: LM-only modules
            upper_params += list(self.model.lm_bridge.parameters())
            upper_params += list(self.model.norm_f.parameters())
            upper_params += list(self.model.lm_head.parameters())

            # JEPA (lower)
            lower_params += list(self.jepa.online_proj.parameters())
            lower_params += list(self.jepa.predictor.parameters())
            lower_params += list(self.jepa.horizon_emb_latent.parameters())
            # target_proj is EMA/no-grad

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

            torch.nn.utils.clip_grad_norm_(lower_params, max_norm=clip_val, norm_type=2.0)
            torch.nn.utils.clip_grad_norm_(upper_params, max_norm=clip_val, norm_type=2.0)

        except Exception:
            try:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val, norm_type=2.0)
            except Exception:
                pass