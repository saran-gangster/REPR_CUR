from typing import List, Tuple, Union
import torch

def sample_anchor_target_pairs(
    batch_size: int,
    seq_len: int,
    pairs_per_seq: int,
    horizon_values: Union[torch.Tensor, List[int]],
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

    # ensure tensors are on the right device without doing redundant copies
    if not isinstance(horizon_values, torch.Tensor):
        horizon_values = torch.tensor(horizon_values, dtype=torch.long, device=device)
    elif horizon_values.device != device:
        horizon_values = horizon_values.to(device)

    if horizon_probs.device != device:
        horizon_probs = horizon_probs.to(device)

    # sample batch index for each pair
    b_idx = torch.arange(batch_size, device=device).repeat_interleave(pairs_per_seq)

    # sample horizons
    k_ids = torch.multinomial(horizon_probs, num_samples=N, replacement=True)
    k_vals = horizon_values[k_ids]  # (N,)

    # sample anchors uniformly, clamp so t + k < T
    high = max(1, seq_len - 1)
    t_idx = torch.randint(low=0, high=high, size=(N,), device=device)
    # enforce valid targets
    t_idx = torch.minimum(t_idx, (seq_len - 1 - k_vals).clamp(min=0))
    tpos = t_idx + k_vals
    # ensure strictly within sequence
    tpos = torch.clamp(tpos, max=seq_len - 1)

    return b_idx, t_idx, tpos, k_ids