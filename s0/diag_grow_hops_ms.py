"""MULTI-SEED robustness audit of the load-bearing "growth adds capability" claim.

diag_grow_hops.py showed (single seed) that growing L2->L4 + training breaks
through on high-K in-context traversal where a 2x-trained L2 stays plateaued
("added depth, not compute"). Today's controller work exposed HIGH run-to-run
variance in these toy tasks (same config: fixed-L2 0.16 one run, 0.47 another),
so a single-seed capability claim is suspect. Here we re-run across N seeds with
FOUR arms to separate the effects and test sign-stability:

  A  L2 trained (6k)                    -- shallow baseline
  B  L4 grown from A + trained (6k)     -- growth arm
  C  L2 control, 2x steps (12k)         -- depth vs pure COMPUTE
  D  L4 from scratch (12k)              -- growth vs pure PARAMS / warm-start

Load-bearing question: is B > C at high K, and is that sign STABLE across seeds
(not a lucky seed)? Bonus: B vs D tells whether warm-start growth is a better
route to a working deep model than deep-from-scratch.

  .venv/bin/python -m s0.diag_grow_hops_ms
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen, train_core, acc_by_k, KMAX

N_SEEDS = 5
HIGH_K = [4, 5]   # where depth is expected to matter


def new(world, device, L):
    return ProxyCore(world.vocab_size, d_model=128, n_layers=L, n_heads=4,
                     max_len=64).to(device)


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    # A: L2 trained 6k, then B: grow to L4 and train 6k more (warm-start)
    torch.manual_seed(seed); world.rng.seed(seed)
    core = new(world, device, 2)
    train_core(core, world, device, 6000)
    A = acc_by_k(core, world, device)
    grow_deeper(core, n_new=2, trainable=True)
    train_core(core, world, device, 6000)
    B = acc_by_k(core, world, device)
    # C: fresh L2 trained 12k (2x compute, same depth)
    torch.manual_seed(seed); world.rng.seed(seed)
    c = new(world, device, 2)
    train_core(c, world, device, 12000)
    C = acc_by_k(c, world, device)
    # D: fresh L4 trained 12k (same params/depth as B, no warm-start)
    torch.manual_seed(seed); world.rng.seed(seed)
    d = new(world, device, 4)
    train_core(d, world, device, 12000)
    D = acc_by_k(d, world, device)
    return A, B, C, D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"multi-seed growth-adds-capability audit (N={N_SEEDS}, KMAX={KMAX})")
    arms = {"A_L2_6k": [], "B_L4_grown": [], "C_L2_12k": [], "D_L4_scratch": []}
    bc_gap = {k: [] for k in HIGH_K}   # per-seed B-C gap at high K (sign stability)
    for seed in range(N_SEEDS):
        A, B, C, D = run_seed(seed, device)
        for name, d in zip(arms, (A, B, C, D)):
            arms[name].append(d)
        for k in HIGH_K:
            bc_gap[k].append(B[k] - C[k])
        hi = lambda d: " ".join(f"K{k}:{d[k]:.2f}" for k in HIGH_K)
        print(f"  seed {seed}: B(grown) [{hi(B)}]  C(L2x2) [{hi(C)}]  "
              f"D(L4scr) [{hi(D)}]", flush=True)

    def stat(vals):  # vals: list of per-seed acc floats
        m = sum(vals) / len(vals)
        return m, min(vals), max(vals)

    print(f"\n== mean [min,max] over {N_SEEDS} seeds, by K ==")
    print("  arm            " + " ".join(f"{'K'+str(k):>16}" for k in range(1, KMAX + 1)))
    for name, dlist in arms.items():
        cells = []
        for k in range(1, KMAX + 1):
            m, lo, hi = stat([d[k] for d in dlist])
            cells.append(f"{m:.2f}[{lo:.2f},{hi:.2f}]")
        print(f"  {name:14s} " + " ".join(f"{c:>16}" for c in cells))

    print("\n== load-bearing check: B(grown) - C(L2, 2x compute) at high K, per seed ==")
    for k in HIGH_K:
        g = bc_gap[k]
        pos = sum(1 for x in g if x > 0.02)
        m, lo, hi = stat(g)
        print(f"  K{k}: gap mean {m:+.2f} [{lo:+.2f},{hi:+.2f}]  "
              f"positive in {pos}/{N_SEEDS} seeds  ({[round(x,2) for x in g]})")
    print("\n  STABLE positive gap => 'growth adds DEPTH-capability, not just compute'")
    print("  survives variance. Mixed sign => the single-seed claim was fragile.")


if __name__ == "__main__":
    main()
