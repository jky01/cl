"""CLOUD experiment C: a ROBUST growth trigger (held-out slope + patience) fixes
the naive controller. diag_controller2.py showed a single-window training-loss
delta grows PREMATURELY (temporary plateaus before phase transitions look like
saturation) and loses to from-scratch. Fix: decide on HELD-OUT accuracy, and only
after `patience` consecutive chunks of no held-out gain (a real plateau, not a
one-chunk dip). Grow, cool down while the new layers warm, then re-measure.

Task: in-context K-hop (kmax=7) — has a real budget->depth sweet spot (L4 beats
L2 at moderate budget; deep-from-scratch undertrains). A good controller should
auto-land near L4 and beat fixed-L2 (capacity) without over-growing to L8.

Arms at equal budget: robust-controller vs fixed L2 / L4 / L8. N seeds.

  python3 -m s0.diag_controller3
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from . import diag_growpenalty2 as dgp
from .diag_growpenalty2 import gen, acc

import os
CHUNK = 1000
BUDGET = 8000
MAXL = 8
PATIENCE = 2          # chunks of no held-out gain before it counts as saturated
EPS = 0.02            # held-out gain over the patience window below this = plateau
N_SEEDS = int(os.environ.get("C3_SEEDS", 3))


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, steps=CHUNK, B=64):
    core.train()
    for _ in range(steps):
        ids, lengths, ans = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step()
    core.eval()


def controller(world, device, budget, log):
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    used, cool, grows = 0, 0, 0
    hist = []                       # held-out acc after each chunk
    while used < budget:
        train_chunk(core, world, device, opt); used += CHUNK
        a = acc(core, world, device)
        hist.append(a)
        # saturation = no held-out gain over the last PATIENCE chunks (robust)
        saturated = len(hist) > PATIENCE and (a - hist[-1 - PATIENCE]) < EPS
        headroom = a < 0.90
        budget_left = budget - used
        grow = (cool == 0 and saturated and headroom and len(core.blocks) < MAXL
                and budget_left >= 2 * CHUNK)
        log.append(f"    used={used} L={len(core.blocks)} heldout={a:.3f} "
                   f"sat={saturated} head={headroom} -> {'GROW' if grow else 'train'}")
        if grow:
            grow_deeper(core, 2, trainable=True); opt = opt_for(core)
            cool = PATIENCE + 1; grows += 1
        else:
            cool = max(0, cool - 1)
    return len(core.blocks), acc(core, world, device), grows


def fixed(world, device, L, budget, seed):
    torch.manual_seed(seed); world.rng.seed(seed)
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=L, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    for _ in range(budget // CHUNK):
        train_chunk(core, world, device, opt)
    return acc(core, world, device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"robust growth trigger (held-out slope + patience={PATIENCE}) vs fixed, "
          f"kmax={dgp.KMAX}, budget={BUDGET}, N={N_SEEDS}")
    rows = {"ctrl": [], "L2": [], "L4": [], "L8": []}
    depths, grows_all = [], []
    for seed in range(N_SEEDS):
        world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
        torch.manual_seed(seed); world.rng.seed(seed)
        log = []
        depth, a_ctrl, grows = controller(world, device, BUDGET, log)
        a2 = fixed(world, device, 2, BUDGET, seed)
        a4 = fixed(world, device, 4, BUDGET, seed)
        a8 = fixed(world, device, 8, BUDGET, seed)
        rows["ctrl"].append(a_ctrl); rows["L2"].append(a2)
        rows["L4"].append(a4); rows["L8"].append(a8)
        depths.append(depth); grows_all.append(grows)
        for line in log:
            print(line, flush=True)
        print(f"  seed {seed}: controller L={depth} acc={a_ctrl:.3f} ({grows} grows) | "
              f"fixed L2 {a2:.3f}  L4 {a4:.3f}  L8 {a8:.3f}", flush=True)

    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n== mean over {N_SEEDS} seeds ==")
    print(f"  controller {mean(rows['ctrl']):.3f} (final depth {[d for d in depths]}, "
          f"grows {grows_all}) | fixed L2 {mean(rows['L2']):.3f}  "
          f"L4 {mean(rows['L4']):.3f}  L8 {mean(rows['L8']):.3f}")
    print("  robust trigger should land near L4's accuracy, beat fixed-L2, and NOT")
    print("  waste budget over-growing to L8 like the naive loss-delta trigger did.")


if __name__ == "__main__":
    main()
