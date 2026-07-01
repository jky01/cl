"""CLOUD experiment A2: is A's width-degradation just UNDERTRAINING? Experiment A
scaled width at FIXED budget(12k)/lr(3e-3) and breakthrough rate FELL (3/8 ->
0/8 -> 0/8), but wider nets need more steps + a lower lr (d=512 seed0 even
diverged). Here we scale BUDGET and LOWER LR with width and re-measure: if the
grown breakthrough rate RECOVERS / rises once wider models are properly trained,
then "scale can make growth reliable" survives (A's fall was an optimisation
artifact, not a scale verdict). If it stays ~0, width genuinely doesn't help.

Per width: BUDGET and LR from the tables below. Arms (compute cheaper: 2, not 4):
  grown  L2(width) budget/2 -> grow to L4 -> +budget/2   (warm-start growth)
  L2ctrl L2(width) full budget                            (2x-compute, same depth)

  python3 -m s0.diag_grow_hops_scale2
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import train_core, acc_by_k, KMAX

WIDTHS = [128, 256, 512]
BUDGET = {128: 12000, 256: 24000, 512: 40000}   # more compute for wider
LR = {128: 3e-3, 256: 1.5e-3, 512: 1e-3}         # lower lr for wider (stability)
N_SEEDS = 6


def heads(d):
    return max(4, d // 32)


def new(world, device, L, d):
    return ProxyCore(world.vocab_size, d_model=d, n_layers=L, n_heads=heads(d),
                     max_len=64).to(device)


def broke(a):
    return a[4] >= 0.8 and a[5] >= 0.8


def run_cell(width, seed, device):
    b, lr = BUDGET[width], LR[width]
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    torch.manual_seed(seed); world.rng.seed(seed)
    core = new(world, device, 2, width)
    train_core(core, world, device, b // 2, lr=lr)
    grow_deeper(core, n_new=2, trainable=True)
    train_core(core, world, device, b // 2, lr=lr)
    B = acc_by_k(core, world, device)
    torch.manual_seed(seed); world.rng.seed(seed)
    c = new(world, device, 2, width); train_core(c, world, device, b, lr=lr)
    C = acc_by_k(c, world, device)
    return B, C


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2: growth reliability with COMPUTE+LR scaled to width "
          f"(N={N_SEEDS}, KMAX={KMAX}, dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    print(f"  budget={BUDGET}  lr={LR}")
    summ = {}
    for w in WIDTHS:
        Bs, Cs = [], []
        for seed in range(N_SEEDS):
            B, C = run_cell(w, seed, device)
            Bs.append(B); Cs.append(C)
            print(f"  d={w} seed {seed}: grown[K4:{B[4]:.2f} K5:{B[5]:.2f}]"
                  f"{'BT' if broke(B) else '--'} L2ctrl[K4:{C[4]:.2f} K5:{C[5]:.2f}]"
                  f"{'BT' if broke(C) else '--'}", flush=True)
        bt = lambda L: sum(broke(a) for a in L)
        mk = lambda L, k: sum(a[k] for a in L) / len(L)
        summ[w] = (bt(Bs), bt(Cs), mk(Bs, 4), mk(Bs, 5))
        print(f"  -> d={w}: grown BT {bt(Bs)}/{N_SEEDS}, L2ctrl BT {bt(Cs)}/{N_SEEDS} "
              f"(grown meanK4 {mk(Bs,4):.2f} K5 {mk(Bs,5):.2f})", flush=True)

    print(f"\n== breakthrough rate vs width, COMPUTE+LR SCALED (N={N_SEEDS}) ==")
    print(f"  {'d_model':>8} | {'budget':>7} {'lr':>7} | {'grown':>7} {'L2ctrl':>7} | meanK4/K5")
    for w in WIDTHS:
        gb, cb, m4, m5 = summ[w]
        print(f"  {w:>8} | {BUDGET[w]:>7} {LR[w]:>7.0e} | {gb:>4}/{N_SEEDS} {cb:>4}/{N_SEEDS} | "
              f"{m4:.2f} / {m5:.2f}")
    print("\n  grown BT rate recovering/rising with width (vs A's 3/8->0/8->0/8) => A's fall")
    print("  was UNDERTRAINING; scale + proper lr does help. Still 0 => width truly doesn't.")


if __name__ == "__main__":
    main()
