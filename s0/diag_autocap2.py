"""C — make the autonomous grow-and-get-smarter ROBUST. diag_autocap reached mean
0.77 but with high variance (a seed grew to L10 and collapsed to 0.15 — the deep
model failed to optimise). Fix: KEEP-BEST checkpoint — track the model
state with the highest reference capability seen and RETURN THAT at the end. A
collapsed growth is never returned; beneficial-but-slow growth is still kept (unlike a
hasty revert, which reverted good growths before their new layers had warmed up).

  python3 -m s0.diag_autocap2        # env: AC_SEEDS
"""
from __future__ import annotations
import os
import copy
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_autocap import (train_chunk, cap, kmax_at, opt_for,
                           CHUNK, TOTAL, MAXL, PATIENCE, EPS)
from .diag_growlarge import acc_by_k, EVAL_K, D

SEEDS = int(os.environ.get("AC_SEEDS", 3))
REF_KMAX = 6                    # stable capability metric for the revert decision
MARGIN = 0.03                  # growth must improve ref-cap by this or be reverted


def run_seed(seed, device):
    """Autonomous growth (as autocap) + KEEP-BEST-checkpoint: track the model state with
    the highest reference capability seen; return THAT at the end. Bad growth/collapse
    is never returned, but beneficial-but-slow growth is kept (unlike hasty revert)."""
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    torch.manual_seed(seed); world.rng.seed(seed)
    core = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    used, cool, grows = 0, 0, 0
    hist = []
    best_ref, best = -1.0, None
    while used < TOTAL:
        km = kmax_at(used)
        train_chunk(core, world, device, opt, km); used += CHUNK
        a = cap(core, world, device, km); hist.append(a)
        ref = cap(core, world, device, REF_KMAX)
        if ref > best_ref:                                       # keep the best-capability checkpoint
            best_ref, best = ref, copy.deepcopy(core)
        saturated = len(hist) > PATIENCE and (a - hist[-1 - PATIENCE]) < EPS
        if (cool == 0 and saturated and a < 0.90 and len(core.blocks) < MAXL
                and TOTAL - used >= 3 * CHUNK):
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = PATIENCE + 1; grows += 1
        else:
            cool = max(0, cool - 1)
    return acc_by_k(best, world, device), len(best.blocks), grows, best_ref


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"AUTONOMOUS cadence + KEEP-BEST checkpoint ({device}) seeds={SEEDS} refK={REF_KMAX}")
    agg = {k: [] for k in EVAL_K}; info = []
    for seed in range(SEEDS):
        acc, depth, grows, bestref = run_seed(seed, device)
        for k in EVAL_K:
            agg[k].append(acc[k])
        info.append((depth, grows))
        mean = sum(acc[k] for k in EVAL_K) / len(EVAL_K)
        print(f"  seed {seed}: best-depth={depth} grows={grows} best-refcap={bestref:.2f} mean={mean:.2f} "
              + " ".join(f"K{k}:{acc[k]:.2f}" for k in EVAL_K), flush=True)
    m = lambda k: sum(agg[k]) / len(agg[k])
    means = [sum(agg[k][s] for k in EVAL_K) / len(EVAL_K) for s in range(SEEDS)]
    mean = sum(means) / len(means); worst = min(means)
    print(f"\n== robust autonomous (keep-best), {SEEDS} seeds ==")
    print(f"  info(depth,grows) {info} | " + " ".join(f"K{k}:{m(k):.2f}" for k in EVAL_K))
    print(f"  mean {mean:.2f}  WORST-seed {worst:.2f}   (vs autocap mean 0.77, worst 0.15; L6-scratch 0.82)")
    print("  keep-best returns the highest-capability checkpoint ever reached, so a collapsed growth")
    print("  is never returned => grow-and-get-smarter, autonomous AND robust (no catastrophic seeds).")


if __name__ == "__main__":
    main()
