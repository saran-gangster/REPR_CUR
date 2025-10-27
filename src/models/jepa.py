from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.sampling import sample_anchor_target_pairs
from src.utils.ema import update_ema_
from src.utils.vicreg import variance_loss, covariance_loss

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
        dropout: float = 0.0,
        loss_type: str = "soft_nce_psr",
        temperature: float = 0.15,
        teacher_kernel_temp: float = 0.25,
        neighbor_topk: int = 32,
        softnce_uncert_scale: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.0,
        cycle_weight: float = 0.10,
        comp_weight: float = 0.05,
        cycle_stop_grad: bool = True,
        pres_weight: float = 0.02,
        kappa_beta: float = 1.0,
        kappa_temp: float = 1.5,
        normalize_latents: bool = True,
        off_diag_scale: float = 0.01,
        align_scale: float = 1.0,
    ):
        super().__init__()
        assert len(horizons) == len(horizon_probs)
        self.pairs_per_seq = pairs_per_seq
        self.ema_momentum = ema_momentum
        self.loss_type = loss_type
        self.temperature = float(temperature)
        self.teacher_kernel_temp = float(teacher_kernel_temp)
        self.neighbor_topk = int(neighbor_topk) if neighbor_topk is not None else None
        self.softnce_uncert_scale = float(softnce_uncert_scale)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.cycle_weight = float(cycle_weight)
        self.comp_weight = float(comp_weight)
        self.cycle_stop_grad = bool(cycle_stop_grad)
        self.pres_weight = float(pres_weight)
        self.kappa_beta = float(kappa_beta)
        self.kappa_temp = float(kappa_temp)
        self.normalize_latents = bool(normalize_latents)
        self.off_diag_scale = float(off_diag_scale)
        self.align_scale = float(align_scale)

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

        self.latent_to_hidden = MLP(
            in_dim=latent_dim,
            out_dim=d_model,
            hidden_mult=1.5,
            dropout=dropout,
        )

        self._init_target_from_online()

    def _l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latents:
            return x
        return torch.nn.functional.normalize(x, dim=-1)

    def _cosine_mse(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros(())
        a_n = self._l2_norm(a) + eps
        b_n = self._l2_norm(b) + eps
        return (1.0 - (a_n * b_n).sum(dim=-1)).mean()

    def _soft_targets(
        self,
        z_tgt: torch.Tensor,
        k_ids: torch.Tensor,
        teacher_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute soft target distribution Q_i(j) with adaptive kernels and κ_ij."""
        if z_tgt.numel() == 0:
            return z_tgt.new_zeros((0, 0))

        y = self._l2_norm(z_tgt)

        with torch.no_grad():
            logits_t_raw = (y @ y.T)
            mask = (k_ids[:, None] == k_ids[None, :])

            if mask.sum(dim=1).min().item() <= 0:
                mask = torch.eye(y.size(0), dtype=torch.bool, device=y.device)

            neigh_k = min(int(self.neighbor_topk), y.size(0)) if self.neighbor_topk is not None else y.size(0)
            topv, _ = torch.topk(torch.where(mask, logits_t_raw, -1e9), k=neigh_k, dim=1)
            mean_sim = torch.mean(topv, dim=1)
            denom = (mean_sim.max() - mean_sim.min()).clamp(min=1e-8)
            u = torch.relu(1.0 - (mean_sim - mean_sim.min()) / denom)
            row_scale = 1.0 + self.softnce_uncert_scale * u

            logits_t = logits_t_raw / (row_scale[:, None] * max(1e-6, float(self.teacher_kernel_temp)))

            if teacher_logits is None or teacher_logits.numel() == 0:
                kappa = torch.ones_like(logits_t)
            else:
                p_teacher = torch.softmax(teacher_logits / max(1e-6, float(self.kappa_temp)), dim=-1)
                kappa = p_teacher @ p_teacher.T
                if self.kappa_beta != 1.0:
                    kappa = kappa.pow(self.kappa_beta)
                kappa = kappa + 1e-8

            logits_t = logits_t + torch.log(kappa)
            logits_t = logits_t.masked_fill(~mask, float("-inf"))

            if self.neighbor_topk is not None and self.neighbor_topk < y.size(0):
                topk_vals, _ = torch.topk(logits_t, k=neigh_k, dim=1)
                thresh = topk_vals[:, -1].unsqueeze(1)
                logits_t = torch.where(logits_t >= thresh, logits_t, float("-inf"))

        logQ = torch.log_softmax(logits_t, dim=1)
        return logQ

    def _soft_nce(
        self,
        z_pred: torch.Tensor,
        z_tgt: torch.Tensor,
        k_ids: torch.Tensor,
        temperature: float,
        logQ: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_pred.numel() == 0:
            zero = z_pred.new_zeros(())
            return zero, zero, zero

        p = self._l2_norm(z_pred)
        y = self._l2_norm(z_tgt)
        logits = (p @ y.T) / max(1e-6, float(temperature))

        mask = (k_ids[:, None] == k_ids[None, :])
        logits = logits.masked_fill(~mask, float("-inf"))
        logP = torch.log_softmax(logits, dim=1)

        if logQ is None:
            logQ = self._soft_targets(z_tgt, k_ids, teacher_logits)

        loss = -(logQ.exp() * logP).sum(dim=1).mean()

        with torch.no_grad():
            diag_logits = torch.diagonal(logits.masked_fill(~mask, float("-inf")))
            pos_logit = diag_logits.mean() if diag_logits.numel() > 0 else logits.new_zeros(())
            neg = logits.clone()
            neg = neg.masked_fill(~mask, float("-inf"))
            neg.fill_diagonal_(float("-inf"))
            neg_max = neg.max(dim=1).values
            margin = (torch.diagonal(logits) - neg_max).mean()

        return loss, pos_logit.detach(), margin.detach()

    def _preservation_loss(
        self,
        h_anchor: torch.Tensor,
        teacher_anchor_logits: torch.Tensor,
        lm_head_W: torch.Tensor,
        lm_head_b: torch.Tensor | None,
        T0: float = 1.0,
    ) -> torch.Tensor:
        if h_anchor.numel() == 0 or teacher_anchor_logits is None or teacher_anchor_logits.numel() == 0:
            return h_anchor.new_zeros(())

        with torch.no_grad():
            log_qa = torch.log_softmax(teacher_anchor_logits / max(1e-6, float(T0)), dim=-1)
            qa = log_qa.exp()

        z_anchor_roundtrip = self.online_proj(h_anchor)
        h_roundtrip = self.latent_to_hidden(z_anchor_roundtrip)
        logits_roundtrip = F.linear(h_roundtrip, lm_head_W, lm_head_b)
        log_q_tilde_a = torch.log_softmax(logits_roundtrip / max(1e-6, float(T0)), dim=-1)
        q_tilde_a = log_q_tilde_a.exp()

        m = 0.5 * (qa + q_tilde_a)
        m_log = torch.log(m + 1e-8)

        kl_qa_m = (qa * (log_qa - m_log)).sum(dim=-1)
        kl_q_tilde_a_m = (q_tilde_a * (log_q_tilde_a - m_log)).sum(dim=-1)

        js_loss = (0.5 * kl_qa_m + 0.5 * kl_q_tilde_a_m).mean()
        return js_loss

    def _barlow_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute standard Barlow Twins loss with internal batch normalisation."""
        if z_a.numel() == 0:
            zero = z_a.new_zeros(())
            return zero, zero, zero

        N, D = z_a.shape
        if N <= 1:
            zero = z_a.new_zeros(())
            return zero, zero, zero

        a_norm = (z_a - z_a.mean(dim=0, keepdim=True)) / (z_a.std(dim=0, unbiased=False, keepdim=True) + eps)
        b_norm = (z_b - z_b.mean(dim=0, keepdim=True)) / (z_b.std(dim=0, unbiased=False, keepdim=True) + eps)

        c = (a_norm.T @ b_norm) / float(N)

        on_diag = (1.0 - torch.diagonal(c)).pow(2).mean()

        if D <= 1:
            off_diag = c.new_zeros(())
        else:
            off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum()) / (D * (D - 1))

        loss = self.align_scale * on_diag + self.off_diag_scale * off_diag
        return loss, on_diag, off_diag

    def _init_target_from_online(self):
        with torch.no_grad():
            for tp, op in zip(self.target_proj.parameters(), self.online_proj.parameters()):
                if tp.shape == op.shape:
                    tp.data.copy_(op.data)

    @torch.no_grad()
    def momentum_update(self):
        # EMA update for target projector from online projector
        update_ema_(self.target_proj, self.online_proj, self.ema_momentum)
    
    def _sample_pairs(self, B: int, T: int, device, generator: torch.Generator | None = None):
        return sample_anchor_target_pairs(
            batch_size=B,
            seq_len=T,
            pairs_per_seq=self.pairs_per_seq,
            horizon_values=self.horizon_values,
            horizon_probs=self.horizon_probs,
            device=device,
            generator=generator,
        )

    def _resolve_pairs(
        self,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        B: int,
        T: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if pairs is None:
            b_idx, t_idx, tpos, k_ids = self._sample_pairs(B, T, device, generator=generator)
        else:
            b_idx, t_idx, tpos, k_ids = pairs
            if b_idx.device != device:
                b_idx = b_idx.to(device, non_blocking=True)
                t_idx = t_idx.to(device, non_blocking=True)
                tpos = tpos.to(device, non_blocking=True)
                k_ids = k_ids.to(device, non_blocking=True)
        return b_idx, t_idx, tpos, k_ids

    def forward(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        return_latents: bool = False,
        teacher_h: torch.Tensor | None = None,
        teacher_logits_target: torch.Tensor | None = None,
        teacher_logits_anchor: torch.Tensor | None = None,
        detached_lm_head: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute JEPA objective and diagnostics."""
        B, T, C = h.shape
        device = h.device

        # Use fixed pairs if provided; else sample
        b_idx, t_idx, tpos, k_ids = self._resolve_pairs(pairs, B, T, device)

        # gather hidden states
        h_anchor = h[b_idx, t_idx, :]           # (N, C)
        if teacher_h is None:
            h_target = h[b_idx, tpos, :]
        else:
            if teacher_h.device != device:
                teacher_h = teacher_h.to(device, non_blocking=True)
            h_target = teacher_h[b_idx, tpos, :]

        # project to latent space
        z_anchor = self.online_proj(h_anchor)   # (N, D_latent)
        z_tgt = self.target_proj(h_target).detach()  # (N, D_latent), stop-grad teacher

        # horizon embedding in latent space
        z_k = self.horizon_emb_latent(k_ids)    # (N, D_latent)

        # predictor in latent space
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D_latent)

        if z_pred.numel() == 0:
            zero = z_anchor.new_zeros(())
            out = {
                "loss": zero,
                "align_loss": zero,
                "psc_loss": zero,
                "var_loss": zero,
                "cov_loss": zero,
                "cycle_loss": zero,
                "comp_loss": zero,
                "pres_loss": zero,
                "nce_pos": zero,
                "nce_margin": zero,
                "std_anchor": zero,
                "std_pred": zero,
            }
            if return_latents:
                out["z_pred"] = z_pred
                out["pairs"] = (b_idx, t_idx, tpos, k_ids)
            return out

        if teacher_logits_target is not None and teacher_logits_target.device != device:
            teacher_logits_target = teacher_logits_target.to(device, non_blocking=True)
        if teacher_logits_anchor is not None and teacher_logits_anchor.device != device:
            teacher_logits_anchor = teacher_logits_anchor.to(device, non_blocking=True)

        # --- Alignment term ---
        if self.loss_type in ("soft_nce_psr", "soft_nce"):
            psc_loss, pos_logit, margin = self._soft_nce(
                z_pred,
                z_tgt,
                k_ids,
                temperature=self.temperature,
                logQ=None,
                teacher_logits=teacher_logits_target,
            )

            var_reg = variance_loss(z_pred) + variance_loss(z_tgt)
            cov_reg = covariance_loss(z_pred) + covariance_loss(z_tgt)
            align_loss = psc_loss + self.var_weight * var_reg + self.cov_weight * cov_reg
        elif self.loss_type == "barlow":
            align_loss, barlow_diag, barlow_off = self._barlow_loss(z_pred, z_tgt)
            psc_loss = align_loss.new_zeros(())
            pos_logit = align_loss.new_zeros(())
            margin = align_loss.new_zeros(())
            var_reg = align_loss.new_zeros(())
            cov_reg = align_loss.new_zeros(())
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # --- Cycle & composition losses ---
        cycle_loss = z_pred.new_zeros(())
        comp_loss = z_pred.new_zeros(())
        if self.cycle_weight > 0.0 or self.comp_weight > 0.0:
            N = z_pred.size(0)
            probs = self.horizon_probs
            if probs.device != device:
                probs = probs.to(device)
            k2_ids = torch.multinomial(probs, num_samples=N, replacement=True)
            k1_vals = self.horizon_values[k_ids]
            k2_vals = self.horizon_values[k2_ids]

            h_mid = self.latent_to_hidden(z_pred)
            if self.cycle_stop_grad:
                h_mid = h_mid.detach()
            z_anchor_mid = self.online_proj(h_mid)

            z_pred_k1k2 = self.predictor(torch.cat([z_anchor_mid, self.horizon_emb_latent(k2_ids)], dim=-1))

            tpos2 = (tpos + k2_vals).clamp(max=T - 1)
            teacher_source = h if teacher_h is None else teacher_h
            if teacher_source.device != device:
                teacher_source = teacher_source.to(device, non_blocking=True)
            h_tgt2 = teacher_source[b_idx, tpos2, :]
            z_tgt2 = self.target_proj(h_tgt2).detach()

            cycle_loss = self._cosine_mse(z_pred_k1k2, z_tgt2)

            if self.comp_weight > 0.0:
                k12_vals = k1_vals + k2_vals
                horizon_list = self.horizon_values.tolist()
                has_direct_mask = torch.tensor([int(v.item() in horizon_list) for v in k12_vals], device=device, dtype=torch.bool)
                if has_direct_mask.any():
                    idx_map = [horizon_list.index(int(v.item())) if int(v.item()) in horizon_list else 0 for v in k12_vals]
                    k12_ids = torch.tensor(idx_map, device=device, dtype=torch.long)
                    z_pred_k12 = self.predictor(torch.cat([z_anchor, self.horizon_emb_latent(k12_ids)], dim=-1))
                    comp_loss = self._cosine_mse(z_pred_k1k2[has_direct_mask], z_pred_k12[has_direct_mask])

        # --- Preservation loss ---
        pres_loss = z_pred.new_zeros(())
        if self.pres_weight > 0.0 and teacher_logits_anchor is not None and detached_lm_head is not None:
            lm_W, lm_b = detached_lm_head
            if lm_W.device != device:
                lm_W = lm_W.to(device)
            if lm_b is not None and lm_b.device != device:
                lm_b = lm_b.to(device)
            pres_loss = self._preservation_loss(h_anchor, teacher_logits_anchor, lm_W, lm_b, T0=1.0)

        loss = align_loss + self.cycle_weight * cycle_loss + self.comp_weight * comp_loss + self.pres_weight * pres_loss

        std_anchor = z_anchor.std(dim=0, unbiased=False).mean()
        std_pred = z_pred.std(dim=0, unbiased=False).mean()

        out = {
            "loss": loss,
            "align_loss": align_loss.detach(),
            "psc_loss": psc_loss.detach(),
            "var_loss": var_reg.detach(),
            "cov_loss": cov_reg.detach(),
            "cycle_loss": cycle_loss.detach(),
            "comp_loss": comp_loss.detach(),
            "pres_loss": pres_loss.detach(),
            "nce_pos": pos_logit,
            "nce_margin": margin,
            "std_anchor": std_anchor.detach(),
            "std_pred": std_pred.detach(),
        }
        if self.loss_type == "barlow":
            out.update({
                "barlow_on_diag": barlow_diag.detach(),
                "barlow_off_diag": barlow_off.detach(),
            })
        if return_latents:
            out["z_pred"] = z_pred
            out["pairs"] = (b_idx, t_idx, tpos, k_ids)
        return out

    @torch.no_grad()
    def compute_latents(
        self,
        h: torch.Tensor,
        pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        teacher_h: torch.Tensor | None = None,
    ):
        """
        Utility for validation: return prediction/target latents using either:
        - provided fixed pairs (b_idx, t_idx, tpos, k_ids), or
        - freshly sampled pairs if pairs is None.
        Returns:
        z_pred: (N, D)
        z_tgt:  (N, D)
        k_ids:  (N,) horizon category ids
        """
        B, T, C = h.shape
        device = h.device

        b_idx, t_idx, tpos, k_ids = self._resolve_pairs(pairs, B, T, device)

        h_anchor = h[b_idx, t_idx, :]    # (N, C)
        if teacher_h is None:
            h_target = h[b_idx, tpos, :]
        else:
            if teacher_h.device != device:
                teacher_h = teacher_h.to(device, non_blocking=True)
            h_target = teacher_h[b_idx, tpos, :]

        z_anchor = self.online_proj(h_anchor)       # (N, D)
        z_k = self.horizon_emb_latent(k_ids)        # (N, D)
        z_pred = self.predictor(torch.cat([z_anchor, z_k], dim=-1))  # (N, D)
        z_tgt = self.target_proj(h_target)          # (N, D)

        return z_pred, z_tgt, k_ids