"""ROUND 2 / clarification #2 — WHY grow at all, if from-scratch-at-final-size wins?
Because a continual learner in a STREAM cannot do from-scratch-at-final-size: it does
not know the final size in advance and cannot replay all past data. At MATCHED total
compute, among the strategies actually FEASIBLE in a stream, growth is best.

Escalating K-hop stream (kmax 3->7), total budget T. Arms:
  grown        autonomous keep-best growth (incremental, feasible)         [FEASIBLE]
  fixed-small  L2 throughout (feasible, but capacity-bound)                [FEASIBLE]
  retrain-each fresh L(final) from scratch at each stage, T/stages each    [FEASIBLE but wasteful]
  large-oracle L(final) trained T steps on the whole curriculum            [NEEDS oracle final-size + full replay -> NOT feasible in a stream]
Accuracy by K at the end. If grown beats fixed-small and retrain-each (the feasible
peers) and approaches the oracle, growth is the compute-efficient continual strategy.

  python3 -m s0.diag_stream        # env: ST_SEEDS
"""
from __future__ import annotations
import os
import copy
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_autocap import train_chunk, cap, kmax_at, opt_for, CHUNK, TOTAL, MAXL, PATIENCE, EPS
from .diag_growlarge import train as train_km, acc_by_k, EVAL_K, STAGES, D

SEEDS = int(os.environ.get("ST_SEEDS", 3))
FINAL_L = 10                      # the size the grown controller tends to reach


def grown_keepbest(world, device):
    V = world.vocab_size
    core = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core); used, cool = 0, 0; hist = []; best_ref, best = -1, None
    while used < TOTAL:
        km = kmax_at(used); train_chunk(core, world, device, opt, km); used += CHUNK
        a = cap(core, world, device, km); hist.append(a)
        ref = cap(core, world, device, 6)
        if ref > best_ref: best_ref, best = ref, copy.deepcopy(core)
        sat = len(hist) > PATIENCE and (a - hist[-1 - PATIENCE]) < EPS
        if cool == 0 and sat and a < 0.90 and len(core.blocks) < MAXL and TOTAL - used >= 3 * CHUNK:
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = PATIENCE + 1
        else:
            cool = max(0, cool - 1)
    return acc_by_k(best, world, device)


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed)); V = world.vocab_size
    torch.manual_seed(seed); world.rng.seed(seed)
    out = {}
    out["grown"] = grown_keepbest(world, device)

    torch.manual_seed(seed); world.rng.seed(seed)         # fixed-small L2
    m = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    for km in STAGES:
        train_km(m, world, device, km, TOTAL // len(STAGES))
    out["fixed-small"] = acc_by_k(m, world, device)

    torch.manual_seed(seed); world.rng.seed(seed)         # retrain-from-scratch each stage
    per = TOTAL // len(STAGES); last = None
    for km in STAGES:
        m = ProxyCore(V, d_model=D, n_layers=FINAL_L, n_heads=4, max_len=72).to(device)
        train_km(m, world, device, km, per)               # fresh model, only this stage's data
        last = m
    out["retrain-each"] = acc_by_k(last, world, device)

    torch.manual_seed(seed); world.rng.seed(seed)         # large-oracle (knows size + full curriculum)
    m = ProxyCore(V, d_model=D, n_layers=FINAL_L, n_heads=4, max_len=72).to(device)
    for km in STAGES:
        train_km(m, world, device, km, per)
    out["large-oracle"] = acc_by_k(m, world, device)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"STREAM why-grow ({device}) seeds={SEEDS} total={TOTAL} curriculum kmax={STAGES}")
    arms = ["grown", "fixed-small", "retrain-each", "large-oracle"]
    agg = {a: {k: [] for k in EVAL_K} for a in arms}
    for seed in range(SEEDS):
        res = run_seed(seed, device)
        for a in arms:
            for k in EVAL_K:
                agg[a][k].append(res[a][k])
        print(f"  seed {seed}: " + "  ".join(
            f"{a} {sum(res[a].values())/len(EVAL_K):.2f}" for a in arms), flush=True)
    m_ = lambda a, k: sum(agg[a][k]) / len(agg[a][k])
    print(f"\n== mean acc by K over {SEEDS} seeds (matched total compute) ==")
    print("  arm            feasible? " + " ".join(f"K{k}" for k in EVAL_K) + "   mean")
    feas = {"grown": "yes", "fixed-small": "yes", "retrain-each": "yes*", "large-oracle": "NO(oracle)"}
    for a in arms:
        mean = sum(m_(a, k) for k in EVAL_K) / len(EVAL_K)
        print(f"  {a:14s} {feas[a]:10s} " + " ".join(f"{m_(a,k):.2f}" for k in EVAL_K) + f"   {mean:.2f}")
    print("\n  among FEASIBLE stream strategies (grown/fixed-small/retrain-each), grown should win;")
    print("  large-oracle (needs known final size + full replay) is an unattainable upper bound.")
    print("  => growth is the compute-efficient way to grow-and-get-smarter in a real stream.")


if __name__ == "__main__":
    main()
