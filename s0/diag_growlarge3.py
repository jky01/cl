"""Does growth CADENCE explain the growlarge negative? diag_growlarge grew EVERY
stage (5 grows x 1500 steps) and lost to fixed-small — but our own controller work
showed frequent growth burns budget re-warming, while ONE well-timed grow with enough
post-growth budget was the winning pattern (diag_controller3, diag_grow_hops).

Same escalating curriculum (kmax 3->7), same total budget, three growth CADENCES:
  every-stage   L2->L10, +2 per stage (the growlarge loser)
  once-mid      L2 for stages 1-2, ONE grow to L6 before stage 3, then train
  fixed-small   L2 throughout
  fixed-large   L6 from scratch, same budget (reference)
If once-mid beats every-stage AND fixed-small, the negative was CADENCE, and
"grow AND get smarter" survives with the right (controller-like) timing.

  python3 -m s0.diag_growlarge3      # env: GL_SEEDS
"""
from __future__ import annotations
import os
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_growlarge import train, acc_by_k, STAGES, STEPS, EVAL_K, D

SEEDS = int(os.environ.get("GL_SEEDS", 3))


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    out = {}

    def fresh(L):
        torch.manual_seed(seed); world.rng.seed(seed)
        return ProxyCore(V, d_model=D, n_layers=L, n_heads=4, max_len=72).to(device)

    # every-stage (the growlarge loser): +2 per stage, L2->L10
    m = fresh(2)
    for i, km in enumerate(STAGES):
        if i > 0:
            grow_deeper(m, 2, trainable=True)
        train(m, world, device, km, STEPS)
    out["every-stage(L10)"] = acc_by_k(m, world, device)

    # once-mid: ONE grow (L2 -> L6) before stage 3, then keep training
    m = fresh(2)
    for i, km in enumerate(STAGES):
        if i == 2:
            grow_deeper(m, 4, trainable=True)      # one growth event, then 3 stages of budget
        train(m, world, device, km, STEPS)
    out["once-mid(L6)"] = acc_by_k(m, world, device)

    # fixed-small L2
    m = fresh(2)
    for km in STAGES:
        train(m, world, device, km, STEPS)
    out["fixed-small(L2)"] = acc_by_k(m, world, device)

    # fixed-large L6 from scratch
    m = fresh(6)
    for km in STAGES:
        train(m, world, device, km, STEPS)
    out["fixed-L6-scr"] = acc_by_k(m, world, device)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"growth CADENCE test ({device}) seeds={SEEDS} curriculum kmax={STAGES}")
    names = ["every-stage(L10)", "once-mid(L6)", "fixed-small(L2)", "fixed-L6-scr"]
    agg = {n: {k: [] for k in EVAL_K} for n in names}
    for seed in range(SEEDS):
        res = run_seed(seed, device)
        for n in names:
            for k in EVAL_K:
                agg[n][k].append(res[n][k])
        print(f"  seed {seed}: " + "  ".join(
            f"{n} mean={sum(res[n].values())/len(EVAL_K):.2f}" for n in names), flush=True)

    m = lambda n, k: sum(agg[n][k]) / len(agg[n][k])
    print(f"\n== mean accuracy by K over {SEEDS} seeds ==")
    print("  cadence           " + " ".join(f"K{k}" for k in EVAL_K) + "   mean")
    for n in names:
        mean = sum(m(n, k) for k in EVAL_K) / len(EVAL_K)
        print(f"  {n:17s} " + " ".join(f"{m(n,k):.2f}" for k in EVAL_K) + f"   {mean:.2f}")
    print("\n  once-mid > every-stage AND > fixed-small => the growlarge negative was CADENCE")
    print("  (frequent growth burns budget re-warming); ONE well-timed grow does add capability.")


if __name__ == "__main__":
    main()
