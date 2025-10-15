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

        # Transformer
        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, ff_multiplier=ff_multiplier, use_rope=use_rope, rope_base=rope_base,
            weight_tying=weight_tying,
        )

        # JEPA
        jepa_cfg = jepa or {}
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

        latent_dim = jepa_cfg.get("latent_dim", d_model)
        self.latent_to_model = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latent_output_bias = nn.Parameter(torch.zeros(vocab_size))

        hv = self.jepa.horizon_values
        try:
            matches = (hv == 1).nonzero(as_tuple=True)
            if matches and matches[0].numel() > 0:
                self.latent_k1_index = int(matches[0][0].item())
            else:
                self.latent_k1_index = -1
        except Exception:
            self.latent_k1_index = -1

        # JEPA/LM weights
        self.jepa_weight = float(jepa_cfg.get("jepa_weight", jepa_cfg.get("lambda_weight", 0.1)))
        self.latent_lm_weight = float(jepa_cfg.get("latent_lm_weight", jepa_cfg.get("latent_ce_weight", 0.0)))
        classic_weight = jepa_cfg.get("classic_lm_weight", jepa_cfg.get("lm_weight", 1.0))
        self.lm_weight_final = float(classic_weight)

        # Baseline toggle (disables JEPA)
        self.run_baseline = bool(jepa_cfg.get("run_baseline", False))
        if self.run_baseline:
            self.jepa_weight = 0.0
            self.latent_lm_weight = 0.0
            self.lm_weight_final = 1.0

        # LM weight scheduling
        self.lm_weight_use_scheduler = bool(jepa_cfg.get("lm_weight_use_scheduler", not self.run_baseline))
        self.lm_warmup_steps = int(jepa_cfg.get("lm_warmup_steps", 0))

        # Gradient decoupling controls
        self.jepa_tap_layer = jepa_cfg.get("tap_layer", -2)
        self.jepa_grad_barrier = bool(jepa_cfg.get("grad_barrier", True))
        self.jepa_tap_norm = bool(jepa_cfg.get("tap_norm", False))

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

    def _latent_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_model(z)
        if getattr(self.model, "weight_tying", False):
            W = self.model.token_emb.weight
        else:
            W = self.model.lm_head.weight
        logits = h @ W.T
        logit_scale = getattr(self.model, "logit_scale", None)
        if logit_scale is not None:
            logits = logits * logit_scale
        logits = logits + self.latent_output_bias
        return logits

    def _latent_lm_loss(self, jepa_out: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self.latent_lm_weight <= 0.0:
            return torch.tensor(0.0, device=device)
        if "z_pred" not in jepa_out or "pairs" not in jepa_out:
            return torch.tensor(0.0, device=device)
        if self.latent_k1_index < 0:
            return torch.tensor(0.0, device=device)

        z_pred = jepa_out["z_pred"]
        b_idx, _, tpos, k_ids = jepa_out["pairs"]
        mask = (k_ids == self.latent_k1_index)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=z_pred.device)

        logits = self._latent_logits(z_pred[mask])
        targets = x[b_idx[mask], tpos[mask]]
        return F.cross_entropy(logits, targets)

    def _build_full_k1_pairs(self, batch: torch.Tensor):
        if self.latent_k1_index < 0:
            return None
        B, T = batch.shape
        if T < 2:
            return None
        device = batch.device
        steps = torch.arange(T - 1, device=device, dtype=torch.long)
        t_idx = steps.unsqueeze(0).expand(B, -1).reshape(-1)
        b_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(-1, T - 1).reshape(-1)
        tpos = t_idx + 1
        k_ids = torch.full((B * (T - 1),), self.latent_k1_index, device=device, dtype=torch.long)
        return b_idx, t_idx, tpos, k_ids

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
        use_jepa_loss = self.jepa_weight > 0.0
        need_latent = self.latent_lm_weight > 0.0
        need_jepa = use_jepa_loss or need_latent

        if need_jepa:
            h_jepa, h_final = self.model(
                x, tap_layer=self.jepa_tap_layer, return_tap=True,
                grad_barrier=self.jepa_grad_barrier, tap_norm=self.jepa_tap_norm
            )
        else:
            h_final = self.model(x, tap_layer=None, return_tap=False)
            h_jepa = None

        lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=x.device)

        if need_jepa:
            jepa_out = self.jepa(h_jepa, return_latents=need_latent)
            jepa_loss = jepa_out["loss"] if use_jepa_loss else torch.tensor(0.0, device=x.device)
        else:
            jepa_out = {}
            jepa_loss = torch.tensor(0.0, device=x.device)

        latent_ce = self._latent_lm_loss(jepa_out, x) if need_latent else torch.tensor(0.0, device=x.device)

        eff_lm_w = self._current_lm_weight(self.global_step)
        total_loss = (
            self.jepa_weight * jepa_loss
            + self.latent_lm_weight * latent_ce
            + eff_lm_w * lm_loss
        )

        # Minimal logging (core metrics only)
        self.log("train/lm_loss", lm_loss, on_step=True)
        if need_latent:
            self.log("train/latent_ce", latent_ce, on_step=True)

        if use_jepa_loss:
            self.log("train/jepa_loss", jepa_out["loss"], on_step=True)
            self.log("train/info_nce_loss", jepa_out.get("info_nce_loss", torch.tensor(0.0, device=self.device)), on_step=True)
            g = torch.sigmoid(self.model.lm_bridge_gate)
            self.log("train/bridge_gate", g, on_step=True)
        else:
            zero = torch.tensor(0.0, device=self.device)
            self.log("train/jepa_loss", zero, on_step=True)
            self.log("train/info_nce_loss", zero, on_step=True)

        return total_loss
    
    def on_validation_epoch_start(self):
        pass
    
    def _get_or_make_val_pairs(self, B: int, T: int, device: torch.device):
        # reuse pairs for the whole epoch to stabilize metrics
        pairs = getattr(self, "_val_pairs", None)
        if pairs is not None:
            b_idx, t_idx, tpos, _ = pairs
            shape_ok = (b_idx.numel() == self.jepa.pairs_per_seq * B) and (t_idx.max().item() < T) and (tpos.max().item() < T)
            device_ok = (b_idx.device == device)
            if shape_ok and device_ok:
                return pairs

        g = torch.Generator(device=device)
        # fixed seed per epoch for determinism; change base if you want a different stream
        g.manual_seed(12345)
        self._val_pairs = self.jepa._sample_pairs(B, T, device, generator=g)
        return self._val_pairs
    
    def validation_step(self, batch, batch_idx):
        x = batch
        use_jepa_loss = self.jepa_weight > 0.0
        need_latent = self.latent_lm_weight > 0.0
        need_jepa = use_jepa_loss or need_latent

        latent_ce_eval = None
        latent_ppl_eval = None

        with torch.no_grad():
            if need_jepa:
                h_jepa, h_final = self.model(
                    x, tap_layer=self.jepa_tap_layer, return_tap=True,
                    grad_barrier=self.jepa_grad_barrier, tap_norm=self.jepa_tap_norm
                )
            else:
                h_final = self.model(x, tap_layer=None, return_tap=False)
                h_jepa = None

            lm_loss = self._lm_loss(h_final, x) if self.lm_weight_final > 0.0 else torch.tensor(0.0, device=x.device)

            if use_jepa_loss:
                B, T = x.shape[0], x.shape[1]
                pairs = self._get_or_make_val_pairs(B, T, x.device)
                jepa_out = self.jepa(h_jepa, pairs=pairs)

                # Compute latents and per-horizon metrics
                z_pred, z_tgt, k_ids = self.jepa.compute_latents(h_jepa, pairs=pairs)
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
                    top1_acc = torch.tensor(0.0, device=x.device)
                    margin = torch.tensor(0.0, device=x.device)

                # Global normalized top-1
                if count_sum > 0:
                    self.log("val/norm_top1", torch.tensor(weighted_norm_sum / count_sum, device=x.device),
                            on_epoch=True, sync_dist=True)

                jepa_loss = jepa_out["loss"]
            else:
                jepa_out, jepa_loss = {}, torch.tensor(0.0, device=x.device)
                top1_acc = torch.tensor(0.0, device=x.device)
                margin = torch.tensor(0.0, device=x.device)

            if need_jepa and self.latent_k1_index >= 0 and h_jepa is not None:
                k1_pairs = self._build_full_k1_pairs(x)
                if k1_pairs is not None:
                    z_pred_k1, _, _ = self.jepa.compute_latents(h_jepa, pairs=k1_pairs)
                    logits_k1 = self._latent_logits(z_pred_k1)
                    b_idx, _, tpos, _ = k1_pairs
                    targets_k1 = x[b_idx, tpos]
                    latent_ce_eval = F.cross_entropy(logits_k1, targets_k1)
                    latent_ppl_eval = torch.exp(latent_ce_eval)

        # Core validation metrics only
        self.log("val/top1_acc", top1_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ppl", torch.exp(lm_loss), on_epoch=True, sync_dist=True)
        
        if latent_ce_eval is not None:
            self.log("val/latent_ce", latent_ce_eval, on_epoch=True, sync_dist=True)
            self.log("val/latent_ppl", latent_ppl_eval, prog_bar=True, on_epoch=True, sync_dist=True)

        if use_jepa_loss:
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
            if self.lm_weight_final <= 0.0 and self.latent_lm_weight <= 0.0:
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

                    self.latent_output_bias = nn.Parameter(torch.zeros(new_V, device=device))

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
        lr_head = self.optim_cfg.get("lr_head", lr_backbone / 10.0)
        warmup = self.optim_cfg.get("warmup_steps", 200)

        # Determine if LM gradients reach the embeddings
        speaker_owns_embeddings = not (self.jepa_weight > 0.0 and self.jepa_grad_barrier)

        # Collect head parameters
        head_params = []
        head_ids = set()
        
        # If tied, also treat logit scale and output bias as head params
        if getattr(self.model, "weight_tying", False):
            extra_head = []
            ls = getattr(self.model, "logit_scale", None)
            if ls is not None:
                extra_head.append(ls)
            ob = getattr(self.model, "output_bias", None)
            if ob is not None:
                extra_head.append(ob)
            head_params += extra_head
            head_ids.update({id(p) for p in extra_head})
        else:
            # Without weight tying, they're separate parameters
            head_params = list(self.model.lm_head.parameters())
            if speaker_owns_embeddings:
                # LM gradients reach embeddings, so train embeddings with fast LR
                head_params += list(self.model.token_emb.parameters())
            head_ids = {id(p) for p in head_params}

        latent_head_params = list(self.latent_to_model.parameters()) + [self.latent_output_bias]
        head_params += latent_head_params
        head_ids.update({id(p) for p in latent_head_params})

        # Backbone: everything else that requires grad and isn't in head_ids
        backbone_params = [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]

        def split_decay(params):
            decay, no_decay = [], []
            for p in params:
                if p is None or not p.requires_grad:
                    continue
                if p.ndim < 2:  # norms, biases, gates
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        head_decay, head_no_decay = split_decay(head_params)
        bb_decay, bb_no_decay = split_decay(backbone_params)

        # No weight decay on "heads"; decay on backbone
        param_groups = [
            {"params": head_decay + head_no_decay, "lr": lr_head, "weight_decay": 0.0},
            {"params": bb_decay, "lr": lr_backbone, "weight_decay": wd},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=betas)

        if self.trainer.max_steps is None:
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