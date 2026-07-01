"""How much compute/data budget -> how much depth is optimal? The empirical
basis for a data->params growth controller. For each training BUDGET (steps),
sweep final DEPTH (from scratch), measure K-hop capability. If the best depth
L* grows with the budget, then "more data justifies more params" holds and a
controller has something real to exploit; the off-L* gap = the cost of
mis-sizing growth.

  .venv/bin/python -m s0.diag_budget_depth
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore
from .diag_growpenalty2 import gen, train_core, acc  # K-hop, kmax=7

BUDGETS = [1500, 6000]          # small vs large compute/data budget (steps)
DEPTHS = [2, 4, 6, 8]
N_SEEDS = 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = {(L, B): [] for L in DEPTHS for B in BUDGETS}
    for seed in range(N_SEEDS):
        world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
        for L in DEPTHS:
            for B in BUDGETS:
                torch.manual_seed(seed); world.rng.seed(seed)
                core = ProxyCore(world.vocab_size, d_model=128, n_layers=L,
                                 n_heads=4, max_len=72).to(device)
                train_core(core, world, device, B)
                R[(L, B)].append(acc(core, world, device))
        print(f"  seed {seed} done", flush=True)

    mean = lambda xs: sum(xs) / len(xs)
    print("\n== capability (K-hop avg acc), depth x budget ==")
    print("  depth |" + "".join(f" B={B:<6}" for B in BUDGETS))
    for L in DEPTHS:
        print(f"  L={L:<4} |" + "".join(f" {mean(R[(L,B)]):<7.3f}" for B in BUDGETS))
    print("\n== best depth L* per budget ==")
    for B in BUDGETS:
        col = {L: mean(R[(L, B)]) for L in DEPTHS}
        Lstar = max(col, key=col.get)
        print(f"  budget {B:>6}: L*={Lstar}  (acc {col[Lstar]:.3f}); "
              + " ".join(f"L{L}:{col[L]:.2f}" for L in DEPTHS))
    print("\n  L* increasing with budget => more data/compute justifies more params")
    print("  (the data->params relation a growth controller must track); the spread")
    print("  across depths at each budget = the cost of growing the wrong amount.")


if __name__ == "__main__":
    main()
