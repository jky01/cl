"""§28 first cut — META-LEARN the growth controller (Omega), replacing the hand-coded
held-out-slope+patience rule (diag_controller3) with a LEARNED neural policy.

Omega is a small MLP that, from per-chunk observables (loss level, relative loss
improvement, held-out acc, acc delta, budget-used fraction, current depth, cooldown),
outputs P(grow now). It is META-TRAINED with REINFORCE across a DISTRIBUTION of tasks
(K-hop with kmax varied) and budgets, rewarded by final accuracy minus a small
per-grow cost. The test of meta-learning: on HELD-OUT tasks (kmax unseen in training),
does the learned Omega match/beat the hand heuristic and fixed depths — WITHOUT any
hand threshold?

  python3 -m s0.diag_omega        # env: OM_EPISODES, OM_LAMBDA
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen

CHUNK = 500
MAXL = 8
D_MODEL = 128
EPISODES = int(os.environ.get("OM_EPISODES", 300))
LAMBDA = float(os.environ.get("OM_LAMBDA", 0.03))     # per-grow cost in the reward
TRAIN_KMAX = [3, 4, 6, 7]
TEST_KMAX = [5, 8]                                    # held-out difficulties
BUDGETS = [4000, 7000]


def new_core(world, device, L):
    return ProxyCore(world.vocab_size, d_model=D_MODEL, n_layers=L, n_heads=4, max_len=72).to(device)


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, kmax, steps=CHUNK, B=64):
    core.train()
    losses = []
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step(); losses.append(loss.item())
    core.eval()
    return sum(losses) / len(losses)


@torch.no_grad()
def acc_km(core, world, device, kmax, n=1024):
    core.eval()
    ids, lengths, ans, _ = gen(world, device, n, kmax=kmax)
    rows = torch.arange(n, device=device)
    pred = core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1)
    return (pred == ans).float().mean().item()


def obs_vec(loss, prev_loss, a, prev_a, used, budget, depth, cool, device):
    rel = 0.0 if prev_loss is None else (prev_loss - loss) / max(prev_loss, 1e-6)
    return torch.tensor([loss / 6.0, rel, a, a - prev_a, used / budget,
                         (depth - 2) / (MAXL - 2), cool / 2.0], device=device)


class Omega(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def episode(omega, world, device, kmax, budget, mode, log=None):
    """mode: 'sample' (train, stochastic), 'greedy' (Omega>0.5), 'heuristic', or fixed int L."""
    if isinstance(mode, int):                                   # fixed-depth baseline
        core = new_core(world, device, mode); opt = opt_for(core)
        for _ in range(budget // CHUNK):
            train_chunk(core, world, device, opt, kmax)
        return acc_km(core, world, device, kmax), 0, mode, []
    core = new_core(world, device, 2); opt = opt_for(core)
    used, prev_loss, prev_a, cool, grows = 0, None, 0.0, 0, 0
    logps = []; hist = []
    while used < budget:
        loss = train_chunk(core, world, device, opt, kmax); used += CHUNK
        a = acc_km(core, world, device, kmax); hist.append(a)
        can = cool == 0 and len(core.blocks) < MAXL and (budget - used) >= 2 * CHUNK
        grow = False
        if can:
            if mode == "heuristic":                            # diag_controller3 rule
                sat = len(hist) > 2 and (a - hist[-3]) < 0.02
                grow = sat and a < 0.90
            else:
                logit = omega(obs_vec(loss, prev_loss, a, prev_a, used, budget, len(core.blocks), cool, device))
                p = torch.sigmoid(logit)
                if mode == "sample":
                    act = torch.bernoulli(p)
                    logps.append(act * torch.log(p + 1e-8) + (1 - act) * torch.log(1 - p + 1e-8))
                    grow = bool(act.item())
                else:                                          # greedy
                    grow = p.item() > 0.5
        if grow:
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = 2; grows += 1
        else:
            cool = max(0, cool - 1)
        prev_loss, prev_a = loss, a
    return acc_km(core, world, device, kmax), grows, len(core.blocks), logps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OMEGA meta-learned growth controller ({device}) episodes={EPISODES} lambda={LAMBDA} "
          f"train_kmax={TRAIN_KMAX} test_kmax={TEST_KMAX}")
    omega = Omega().to(device)
    optO = torch.optim.Adam(omega.parameters(), lr=1e-3)
    baseline = None
    rng = random.Random(0)
    for ep in range(EPISODES):
        kmax = rng.choice(TRAIN_KMAX); budget = rng.choice(BUDGETS)
        world = World(WorldConfig(n_entities=200, n_objects=200, seed=ep))
        torch.manual_seed(ep); world.rng.seed(ep)
        acc, grows, depth, logps = episode(omega, world, device, kmax, budget, "sample")
        reward = acc - LAMBDA * grows
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        if logps:
            loss = -((reward - baseline) * torch.stack(logps).sum())
            optO.zero_grad(); loss.backward(); optO.step()
        if ep % 50 == 0 or ep == EPISODES - 1:
            print(f"  ep {ep:4d} kmax{kmax} B{budget} acc {acc:.2f} grows {grows} depth {depth} "
                  f"reward {reward:.3f} baseline {baseline:.3f}", flush=True)

    # ---- held-out evaluation ----
    print("\n== held-out tasks: Omega (greedy) vs heuristic vs fixed depths ==")
    for kmax in TEST_KMAX:
        row = {}
        for mode, tag in [("greedy", "omega"), ("heuristic", "heur"), (2, "L2"), (4, "L4"), (6, "L6")]:
            accs, gr, dep = [], [], []
            for s in range(3):
                world = World(WorldConfig(n_entities=200, n_objects=200, seed=1000 + s))
                torch.manual_seed(1000 + s); world.rng.seed(1000 + s)
                a, g, d, _ = episode(omega, world, device, kmax, 7000, mode)
                accs.append(a); gr.append(g); dep.append(d)
            row[tag] = (sum(accs) / 3, sum(dep) / 3, sum(gr) / 3)
        print(f"  kmax={kmax}: " + "  ".join(
            f"{t} acc={v[0]:.2f}(L~{v[1]:.1f})" for t, v in row.items()), flush=True)
    print("\n  Omega >= heuristic and > best fixed on HELD-OUT kmax => the growth decision is")
    print("  now a LEARNED neural policy that generalizes, not a hand threshold (§28 first cut).")


if __name__ == "__main__":
    main()
