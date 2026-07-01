"""Can a K-CURRICULUM after growth turn the bimodal lottery into reliable
breakthrough? diag_grow_hops_ms.py found grown-L4 breaks through to high-K in
only 2/5 seeds (else stuck ~0.3). Hypothesis: facing full difficulty (kmax=5) at
once right after growth is a hard optimisation; ramping kmax (solidify K1-2, then
K3, then K4-5) gives the new layers an incremental path to learn one more hop at
a time. Same compute, better ORDERING.

Per seed: warm L2 (6k), grow to L4, then deepcopy the grown core and train two
arms from the SAME start with equal budget (6k):
  U  uniform     -- kmax=5 throughout (the original arm B)
  C  curriculum  -- staged kmax 3 -> 4 -> 5 (2k each)
Count how many of N seeds break through (K4 & K5 >= 0.8) under each.

  .venv/bin/python -m s0.diag_grow_hops_curric
"""
from __future__ import annotations
import copy
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen, acc_by_k, KMAX

N_SEEDS = 5


def train_kmax(core, world, device, steps, kmax, lr=3e-3, B=64):
    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    core.eval()


def train_curric(core, world, device, total, stages=(3, 4, 5), lr=3e-3, B=64):
    per = total // len(stages)
    for km in stages:
        train_kmax(core, world, device, per, km, lr=lr, B=B)


def broke_through(d):
    return d[4] >= 0.8 and d[5] >= 0.8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"post-growth K-curriculum vs uniform (N={N_SEEDS}, KMAX={KMAX})")
    U_res, C_res = [], []
    u_bt, c_bt = 0, 0
    for seed in range(N_SEEDS):
        world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
        torch.manual_seed(seed); world.rng.seed(seed)
        core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4,
                         max_len=64).to(device)
        train_kmax(core, world, device, 6000, KMAX)
        grow_deeper(core, n_new=2, trainable=True)
        # two arms from the identical grown start / identical RNG stream
        u_core = copy.deepcopy(core); c_core = copy.deepcopy(core)
        rng_state = world.rng.getstate()
        world.rng.setstate(rng_state)
        train_kmax(u_core, world, device, 6000, KMAX)
        U = acc_by_k(u_core, world, device)
        world.rng.setstate(rng_state)
        train_curric(c_core, world, device, 6000, stages=(3, 4, 5))
        C = acc_by_k(c_core, world, device)
        U_res.append(U); C_res.append(C)
        u_bt += broke_through(U); c_bt += broke_through(C)
        hi = lambda d: f"K4:{d[4]:.2f} K5:{d[5]:.2f}"
        print(f"  seed {seed}: uniform[{hi(U)}] {'BT' if broke_through(U) else '--'}"
              f"  curric[{hi(C)}] {'BT' if broke_through(C) else '--'}", flush=True)

    mean = lambda res, k: sum(d[k] for d in res) / len(res)
    print(f"\n  breakthrough (K4&K5>=0.8): uniform {u_bt}/{N_SEEDS}  curric {c_bt}/{N_SEEDS}")
    for k in (3, 4, 5):
        print(f"  K{k} mean: uniform {mean(U_res,k):.2f}  curric {mean(C_res,k):.2f}")
    print("\n  curric breakthrough > uniform => ordering (not scale) can lift the lottery;")
    print("  no improvement => reliability is genuinely a scale question, not tuning.")


if __name__ == "__main__":
    main()
