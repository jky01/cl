"""CLOUD experiment A: does the growth-capability breakthrough become RELIABLE at
larger scale? diag_grow_hops_ms.py found grown-L2->L4 breaks through to high-K in
only ~2/5 seeds at d_model=128 (bimodal lottery, init/basin-determined, not fixed
by curriculum). Here we SWEEP WIDTH (d_model 128->256->512) at N=8 seeds and ask:
does the per-seed BREAKTHROUGH RATE (K4 & K5 >= 0.8) rise as the model gets wider?

Three arms per (width, seed), matched to the audit:
  B  L2(width) 6k -> grow to L4 -> +6k   (warm-start growth)
  C  L2(width) 12k                        (2x compute, same depth control)
  D  L4(width) from scratch 12k           (params/warm-start control)

Success (the thesis): B's breakthrough RATE climbs toward ~N/N with width, while
"grown is the only arm that cracks high-K" (D never breaks) keeps holding. If B
stays ~2/8 even at 512, reliability isn't just a width thing.

  QWEN_MODEL unused (proxy). Run on the GPU box.
  python3 -m s0.diag_grow_hops_scale
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import train_core, acc_by_k, KMAX

WIDTHS = [128, 256, 512]
N_SEEDS = 8


def heads(d):
    return max(4, d // 32)


def new(world, device, L, d):
    return ProxyCore(world.vocab_size, d_model=d, n_layers=L, n_heads=heads(d),
                     max_len=64).to(device)


def broke(a):
    return a[4] >= 0.8 and a[5] >= 0.8


def run_cell(width, seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    torch.manual_seed(seed); world.rng.seed(seed)
    core = new(world, device, 2, width)
    train_core(core, world, device, 6000)
    grow_deeper(core, n_new=2, trainable=True)
    train_core(core, world, device, 6000)
    B = acc_by_k(core, world, device)
    torch.manual_seed(seed); world.rng.seed(seed)
    c = new(world, device, 2, width); train_core(c, world, device, 12000)
    C = acc_by_k(c, world, device)
    torch.manual_seed(seed); world.rng.seed(seed)
    d = new(world, device, 4, width); train_core(d, world, device, 12000)
    D = acc_by_k(d, world, device)
    return B, C, D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"growth-reliability WIDTH sweep (N={N_SEEDS}, KMAX={KMAX}, "
          f"widths={WIDTHS}, dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    summary = {}
    for w in WIDTHS:
        Bs, Cs, Ds = [], [], []
        for seed in range(N_SEEDS):
            B, C, D = run_cell(w, seed, device)
            Bs.append(B); Cs.append(C); Ds.append(D)
            print(f"  d={w} seed {seed}: "
                  f"grown[K4:{B[4]:.2f} K5:{B[5]:.2f}]{'BT' if broke(B) else '--'} "
                  f"L2x2[K4:{C[4]:.2f} K5:{C[5]:.2f}]{'BT' if broke(C) else '--'} "
                  f"L4scr[K4:{D[4]:.2f} K5:{D[5]:.2f}]{'BT' if broke(D) else '--'}",
                  flush=True)
        bt = lambda L: sum(broke(a) for a in L)
        mk = lambda L, k: sum(a[k] for a in L) / len(L)
        summary[w] = (bt(Bs), bt(Cs), bt(Ds), mk(Bs, 4), mk(Bs, 5))
        print(f"  -> d={w}: grown BT {bt(Bs)}/{N_SEEDS}, L2x2 BT {bt(Cs)}/{N_SEEDS}, "
              f"L4scr BT {bt(Ds)}/{N_SEEDS}  (grown meanK4 {mk(Bs,4):.2f} K5 {mk(Bs,5):.2f})",
              flush=True)

    print(f"\n== breakthrough rate (K4&K5>=0.8) vs width, N={N_SEEDS} ==")
    print(f"  {'d_model':>8} | {'grown':>7} {'L2x2':>7} {'L4scr':>7} | grown meanK4/K5")
    for w in WIDTHS:
        gb, cb, db, m4, m5 = summary[w]
        print(f"  {w:>8} | {gb:>4}/{N_SEEDS} {cb:>4}/{N_SEEDS} {db:>4}/{N_SEEDS} | "
              f"{m4:.2f} / {m5:.2f}")
    print("\n  grown BT rate RISING with width => growth-capability reliability is a scale")
    print("  effect (the toy 2/5 was a small-model artifact). Flat => needs more than width.")


if __name__ == "__main__":
    main()
