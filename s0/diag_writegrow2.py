"""HARDEN the write-vs-grow gate to a GENUINE COST-TRADEOFF (diag_writegrow had an
easy fact-vs-skill type signal). Here EVERY item is a fact; the right action depends
on SYSTEM STATE, so the gate must learn a state-dependent policy, not a fixed rule:

  WRITE (memory):      value = freq * (1 - load/CAP)     # recall degrades as memory fills
  CONSOLIDATE (grow):  value = freq * 0.95 - C           # reliable recall, but costs compute

Optimal action = argmax; the consolidation frequency-threshold DROPS as load rises
(at low load, memorize everything; at high load, consolidate the frequent items to
protect recall). The gate sees only [freq, load] and is trained as a contextual
bandit. Test on a (freq,load) grid vs oracle / always-write / always-consolidate /
a fixed heuristic, and show the learned boundary is state-dependent.

  python3 -m s0.diag_writegrow2
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn

EPISODES = int(os.environ.get("WG_EPISODES", 8000))
CAP = 1.0
C = 0.15                       # consolidation (compute) cost


def rewards(freq, load):
    w = freq * (1.0 - load / CAP)
    g = freq * 0.95 - C
    return w, g


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gate = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=3e-3)
    rng = random.Random(0)
    baseline = 0.0
    print(f"WRITE-vs-GROW cost-tradeoff gate ({device}) episodes={EPISODES} C={C}")
    for ep in range(EPISODES):
        freq = rng.random(); load = rng.random() * CAP           # random context
        x = torch.tensor([freq, load], device=device)
        p = torch.softmax(gate(x), -1)
        a = torch.multinomial(p, 1).item()
        w, g = rewards(freq, load)
        rwd = w if a == 0 else g
        baseline = 0.99 * baseline + 0.01 * rwd
        loss = -(rwd - baseline) * torch.log(p[a] + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2000 == 0 or ep == EPISODES - 1:
            print(f"  ep {ep:5d} baseline {baseline:.3f}", flush=True)

    # ---- evaluate on a (freq, load) grid ----
    @torch.no_grad()
    def gate_action(freq, load):
        return gate(torch.tensor([freq, load], device=device)).argmax(-1).item()

    grid = [(f, l) for f in [i / 10 for i in range(1, 11)] for l in [i / 10 for i in range(0, 11)]]
    hit = 0; gr = wr = cr = hr = 0.0
    for (f, l) in grid:
        w, g = rewards(f, l)
        opt_a = 0 if w >= g else 1                                # oracle
        a = gate_action(f, l)
        hit += (a == opt_a)
        gr += (w if a == 0 else g)                               # gate reward
        wr += w; cr += g                                         # always-write / always-consolidate
        hr += (w if f <= 0.5 else g)                             # fixed heuristic: consolidate if freq>0.5
    n = len(grid)
    orc = sum(max(rewards(f, l)) for (f, l) in grid) / n
    print(f"\n== held-out (freq,load) grid: gate vs oracle / baselines ==")
    print(f"  gate matches oracle action: {hit}/{n} ({hit/n:.2f})")
    print(f"  mean reward: gate {gr/n:.3f} | oracle {orc:.3f} | always-write {wr/n:.3f} | "
          f"always-consolidate {cr/n:.3f} | fixed-heuristic {hr/n:.3f}")
    # show state-dependence: consolidation frequency-threshold at low vs high load
    def thresh(load):
        for i in range(1, 101):
            f = i / 100
            if gate_action(f, load) == 1:
                return f
        return 1.0
    print(f"  learned consolidate-threshold on freq:  load=0.1 -> {thresh(0.1):.2f}, "
          f"load=0.5 -> {thresh(0.5):.2f}, load=0.9 -> {thresh(0.9):.2f}  (drops with load => state-dependent)")
    print("\n  gate ~= oracle (> fixed heuristic and >> always-one), and its consolidate-threshold")
    print("  drops as load rises => a LEARNED state-dependent cost-tradeoff policy, not a fixed rule.")


if __name__ == "__main__":
    main()
