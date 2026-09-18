"""Coordinate masking from zhiwei-zzz/t2m path commit
4b6caa85183c993fd971d50c135e6640d64e0edb. Preserves its CPU RNG,
batch bypass, overlapping chunks and 20-attempt behavior.
"""

import torch


def mask_paths_no_for_loop(
    paths, mask_rate_range=(0.3, 0.5), max_mask_len=5, max_attempts=20, no_random_prob=0.1
):
    """
    Vectorized contiguous-chunk masking for a batch of paths, with no per-sample for loop.
    """
    randomize = torch.rand(1)
    N, T, C = paths.shape
    time_mask = torch.ones(N, T)
    if randomize <= no_random_prob:
        return paths, time_mask

    # 1. Sample a mask rate for each element in the batch (shape [N]) in [0.3, 0.5]
    mask_rates = torch.empty(N).uniform_(*mask_rate_range)

    # 2. Determine how many steps in each sequence to mask
    need_masked = (mask_rates * T).round().to(dtype=torch.int)

    # time_mask[i, t] = 1.0 means "keep this step"; 0.0 means "mask"

    # Keep track of how many steps are currently masked for each item
    total_masked = torch.zeros(N, dtype=torch.int)

    # We'll create a helper index array for broadcast comparisons
    idxs = torch.arange(T).unsqueeze(0)  # shape [1, T]

    # 3. Perform up to max_attempts of random chunk masking in parallel
    for _ in range(max_attempts):
        # Check if we've already masked enough for all items
        still_need_mask = total_masked < need_masked
        if not still_need_mask.any():
            break  # Everyone reached the target

        # Sample random centers and lengths for the entire batch at once
        centers = torch.randint(0, T, size=(N,))  # shape [N]
        lengths = torch.randint(1, max_mask_len + 1, size=(N,))  # shape [N]

        # Compute left/right boundaries
        lefts = (centers - lengths // 2).clamp_min(0)  # shape [N]
        rights = (lefts + lengths).clamp_max(T)  # shape [N]

        # Create a boolean mask (N, T) that is True where we set time_mask to 0
        chunk_mask = (
            (idxs >= lefts.unsqueeze(1))
            & (idxs < rights.unsqueeze(1))
            & still_need_mask.unsqueeze(1)
        )

        # Zero out the corresponding positions in time_mask
        time_mask[chunk_mask] = 0.0

        # Update total_masked
        total_masked = T - time_mask.sum(dim=1).to(torch.int)

    # 4. Apply the time_mask to the paths along the time dimension
    masked_paths = paths * (time_mask.unsqueeze(-1)).to(paths.device)
    return masked_paths, time_mask
