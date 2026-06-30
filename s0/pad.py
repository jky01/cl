"""Tiny batching helper."""
from __future__ import annotations
import torch


def pad_batch(seqs, pad_id, device):
    """List[List[int]] -> (ids [B,Tmax], lengths [B]). Right-padded."""
    T = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    lengths = torch.empty(B, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        lengths[b] = len(s)
    return ids, lengths


def last_token_index(lengths):
    """Index of the final real token (the <ans> position) per row."""
    return lengths - 1
