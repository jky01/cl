"""A two-signal (saturation + budget) semi-neural growth CONTROLLER.
During online training it auto-senses: (a) is the loss still improving
(saturation?), (b) is there headroom (accuracy not yet at ceiling?), (c) how
much budget remains -> and decides on its own whether to grow (function-
preserving deepening, a warm start) and is budget-gated on the amount. Compared
across budgets to fixed-small (undertrained capacity) and fixed-deep-from-scratch
(over-grown / hard to optimise). The controller should land near the sweet spot
WITHOUT being told the optimal depth, and adapt its final depth to the budget.

  .venv/bin/python -m s0.diag_controller
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from . import diag_growpenalty2 as dgp   # K-hop task (kmax=7); L*=4 was the sweet spot
from .diag_growpenalty2 import gen, acc

CHUNK = 1000
MAXL = 8


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, steps=CHUNK, B=64):
    core.train(); losses = []
    for _ in range(steps):
        ids, lengths, ans = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step(); losses.append(loss.item())
    core.eval(); return sum(losses) / len(losses)


def controller(world, device, budget):
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    used, prev, cool, grows = 0, None, 0, 0
    while used < budget:
        loss = train_chunk(core, world, device, opt); used += CHUNK
        a = acc(core, world, device)
        improved = 1.0 if prev is None else (prev - loss) / max(prev, 1e-6)
        saturated = improved < 0.05          # (a) loss ~stopped improving
        headroom = a < 0.90                  # (b) not yet at the task ceiling
        budget_left = budget - used          # (c) budget signal
        # grow only if saturated AND has headroom AND enough budget to fill new
        # layers -> the AMOUNT of growth is gated by the remaining budget.
        if (cool == 0 and saturated and headroom and len(core.blocks) < MAXL
                and budget_left >= 2 * CHUNK):
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = 2; grows += 1
        else:
            cool = max(0, cool - 1)
        prev = loss
    return len(core.blocks), acc(core, world, device), grows


def fixed(world, device, L, budget):
    torch.manual_seed(0); world.rng.seed(0)
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=L, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    for _ in range(budget // CHUNK):
        train_chunk(core, world, device, opt)
    return acc(core, world, device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task: K-hop kmax={dgp.KMAX} (measured sweet spot L*=4)")
    print(f"  {'budget':>7} | {'controller (depth, acc, #grows)':>34} | {'fixed L2':>9} {'fixed L8':>9}")
    for budget in (4000, 12000):
        torch.manual_seed(0)
        world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
        torch.manual_seed(0); world.rng.seed(0)
        depth, a_ctrl, grows = controller(world, device, budget)
        a2 = fixed(world, device, 2, budget)
        a8 = fixed(world, device, 8, budget)
        print(f"  {budget:>7} | L={depth} acc={a_ctrl:.3f} ({grows} grows){'':>10} | "
              f"{a2:9.3f} {a8:9.3f}", flush=True)
    print("\n  controller should auto-grow toward the sweet spot (beat fixed-L2 on capacity,")
    print("  beat fixed-L8 by not over-growing/undertraining), adapting depth to the budget,")
    print("  WITHOUT being told the optimal depth -> two-signal (saturation+budget) sensing.")


if __name__ == "__main__":
    main()
