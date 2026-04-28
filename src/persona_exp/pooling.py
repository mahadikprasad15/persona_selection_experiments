from __future__ import annotations

import torch


def pool_response_mean(hidden: torch.Tensor, response_slice: slice) -> torch.Tensor:
    span = hidden[response_slice]
    if span.numel() == 0:
        raise ValueError("Empty response token span")
    return span.mean(dim=0)


def pool_first_k_generated(hidden: torch.Tensor, response_slice: slice, k: int) -> torch.Tensor:
    start = response_slice.start or 0
    stop = min(response_slice.stop or hidden.shape[0], start + k)
    span = hidden[start:stop]
    if span.numel() == 0:
        raise ValueError(f"Empty generated token span for k={k}")
    return span.mean(dim=0)


def select_token(hidden: torch.Tensor, idx: int) -> torch.Tensor:
    if idx < 0 or idx >= hidden.shape[0]:
        raise IndexError(f"Token index {idx} out of range for sequence length {hidden.shape[0]}")
    return hidden[idx]

