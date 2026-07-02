"""CLOSE the loop: does an AUTONOMOUS controller pick the capability-adding growth
CADENCE by itself? growlarge3 showed one well-timed grow adds capability while
grow-every-stage loses. Here a robust controller (held-out capability plateau +
patience + budget/cooldown gating — the diag_controller3 signal) decides WHEN to
grow while training through the escalating K-hop curriculum (kmax 3->7). If the
autonomous arm lands near the hand-tuned once-mid (>> fixed-small / grow-every),
the neural controller autonomously realises "grow AND get smarter".

  python3 -m s0.diag_autocap        # env: AC_SEEDS
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen
from .diag_growlarge import acc_by_k, EVAL_K, D

SEEDS = int(os.environ.get("AC_SEEDS", 3))
CHUNK = 300
TOTAL = 7500                       # same total budget as the 5x1500 curriculum
MAXL = 10
PATIENCE = 2
EPS = 0.02


def kmax_at(used):                 # escalating curriculum over the budget
    return 3 + min(4, int(used / (TOTAL / 5)))


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, kmax, steps=CHUNK, B=64):
    core.train()
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step()
    core.eval()


@torch.no_grad()
def cap(core, world, device, kmax, n=1024):
    core.eval()
    ids, lengths, ans, _ = gen(world, device, n, kmax=kmax)
    rows = torch.arange(n, device=device)
    return (core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1) == ans).float().mean().item()


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    torch.manual_seed(seed); world.rng.seed(seed)
    core = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    used, cool, grows = 0, 0, 0
    hist = []
    while used < TOTAL:
        km = kmax_at(used)
        train_chunk(core, world, device, opt, km); used += CHUNK
        a = cap(core, world, device, km); hist.append(a)
        # robust trigger: capability plateaued over PATIENCE chunks, below ceiling, budget left
        saturated = len(hist) > PATIENCE and (a - hist[-1 - PATIENCE]) < EPS
        if (cool == 0 and saturated and a < 0.90 and len(core.blocks) < MAXL
                and TOTAL - used >= 3 * CHUNK):
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = PATIENCE + 1; grows += 1
        else:
            cool = max(0, cool - 1)
    return acc_by_k(core, world, device), len(core.blocks), grows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"AUTONOMOUS cadence ({device}) seeds={SEEDS} total={TOTAL} curriculum kmax 3->7")
    agg = {k: [] for k in EVAL_K}; depths = []; growslist = []
    for seed in range(SEEDS):
        acc, depth, grows = run_seed(seed, device)
        for k in EVAL_K:
            agg[k].append(acc[k])
        depths.append(depth); growslist.append(grows)
        print(f"  seed {seed}: auto-controller depth={depth} grows={grows} "
              + " ".join(f"K{k}:{acc[k]:.2f}" for k in EVAL_K), flush=True)
    m = lambda k: sum(agg[k]) / len(agg[k])
    mean = sum(m(k) for k in EVAL_K) / len(EVAL_K)
    print(f"\n== autonomous controller, mean over {SEEDS} seeds ==")
    print(f"  depth {depths} grows {growslist} | " + " ".join(f"K{k}:{m(k):.2f}" for k in EVAL_K) + f"  mean {mean:.2f}")
    print(f"\n  reference (diag_growlarge3): grow-every 0.47 | fixed-small 0.54 | once-mid 0.72 | L6-scratch 0.82")
    print("  autonomous mean near once-mid (>> grow-every / fixed-small) => the controller")
    print("  autonomously found the capability-adding cadence — grow AND get smarter, on its own.")


if __name__ == "__main__":
    main()
