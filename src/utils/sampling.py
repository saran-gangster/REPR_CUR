from typing import List, Tuple
import torch

def sample_anchor_target_pairs(
    batch_size: int,
    seq_len: int,
    pairs_per_seq: int,
    horizon_values: List[int],
    horizon_probs: torch.Tensor,
    device=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      b_idx: (N,) batch indices
      t_idx: (N,) anchor positions
      tpos:  (N,) target positions (t + k)
      k_ids: (N,) horizon category ids corresponding to horizon_values
    """
    device = device or torch.device("cpu")
    N = batch_size * pairs_per_seq
    # sample batch index for each pair
    b_idx = torch.arange(batch_size, device=device).repeat_interleave(pairs_per_seq)
    # sample horizons
    k_ids = torch.multinomial(horizon_probs.to(device), num_samples=N, replacement=True)
    k_vals = torch.tensor(horizon_values, device=device, dtype=torch.long)[k_ids]  # (N,)

    # sample anchors uniformly, clamp so t + k < T
    t_idx = torch.randint(low=0, high=max(1, seq_len - 1), size=(N,), device=device)
    # enforce valid targets
    t_idx = torch.minimum(t_idx, (seq_len - 1 - k_vals).clamp(min=0))
    tpos = t_idx + k_vals
    # ensure strictly within sequence
    tpos = torch.clamp(tpos, max=seq_len - 1)

    return b_idx, t_idx, tpos, k_ids