"""The make-or-break question for "grow small into large": does the GROWTH
PENALTY compound? Grow a model 2->4->6->8 (training at each stage) and compare
each size to a model trained FROM SCRATCH at that size for matched total steps.
If the grown-vs-scratch gap stays small/flat -> growing into a large model is
viable. If the gap widens with depth -> compounding degradation kills the idea.

Task = the depth-sensitive K-hop traversal (deeper = better).

  .venv/bin/python -m s0.diag_growpenalty
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen, acc_by_k, train_core, KMAX

S = 1800          # steps per growth stage
SIZES = [2, 4, 6, 8]
N_SEEDS = 3


def avg_acc(core, world, device):
    a = acc_by_k(core, world, device)
    return sum(a.values()) / len(a)


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    # GROWN trajectory
    torch.manual_seed(seed); world.rng.seed(seed)
    core = ProxyCore(V, d_model=128, n_layers=2, n_heads=4, max_len=64).to(device)
    grown = {}
    for L in SIZES:
        if L > 2:
            grow_deeper(core, 2, trainable=True)
        train_core(core, world, device, S)
        grown[L] = avg_acc(core, world, device)
    # FROM SCRATCH at each size, matched total steps
    scratch = {2: grown[2]}
    for L in SIZES[1:]:
        torch.manual_seed(seed); world.rng.seed(seed)
        c = ProxyCore(V, d_model=128, n_layers=L, n_heads=4, max_len=64).to(device)
        train_core(c, world, device, (L // 2) * S)
        scratch[L] = avg_acc(c, world, device)
    return grown, scratch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = {L: [] for L in SIZES}
    SC = {L: [] for L in SIZES}
    for seed in range(N_SEEDS):
        grown, scratch = run_seed(seed, device)
        for L in SIZES:
            G[L].append(grown[L]); SC[L].append(scratch[L])
        print(f"  seed {seed}: grown " + " ".join(f"{grown[L]:.2f}" for L in SIZES)
              + " | scratch " + " ".join(f"{scratch[L]:.2f}" for L in SIZES), flush=True)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n== GROWTH PENALTY over {N_SEEDS} seeds (mean scratch - grown) ==")
    print(f"  {'L':>3} | {'grown':>7} {'scratch':>8} {'gap':>7}")
    for L in SIZES:
        g, s = mean(G[L]), mean(SC[L])
        print(f"  {L:>3} | {g:7.3f} {s:8.3f} {s-g:7.3f}")
    print("  gap flat/small => growing into a larger model is viable;")
    print("  gap widening with L => compounding penalty (small->super-large degrades).")


if __name__ == "__main__":
    main()
