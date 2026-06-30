"""Autonomous small->large: a controller that GROWS the core by itself when it
detects saturation (training loss plateaus while accuracy is still poor =
capacity bottleneck). Starts at L=2 and should grow to L=6 on its own, climbing
the K-hop accuracy ladder while preserving low-K -- the closed-loop realisation
of "continually learn from small into large".

(The trigger is a plateau heuristic here -- a placeholder for a learned/neural
growth gate per §27 SleepGate/CapacityNet.)

  .venv/bin/python -m s0.diag_autogrow
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen, acc_by_k, KMAX

CHUNK = 600
MAX_LAYERS = 6


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, steps=CHUNK, B=64):
    core.train(); losses = []
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        logits = core.lm_head(core.hidden(ids)[rows, lengths - 1])
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step(); losses.append(loss.item())
    core.eval(); return sum(losses) / len(losses)


def run(world, device, auto_grow):
    torch.manual_seed(0); world.rng.seed(0)
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=64).to(device)
    opt = opt_for(core)
    prev = None; cooldown = 0
    tag = "AUTO-GROW" if auto_grow else "FIXED L=2"
    print(f"\n== {tag} ==")
    for chunk in range(16):
        mean = train_chunk(core, world, device, opt)
        a = acc_by_k(core, world, device)
        depth = len(core.blocks)
        print(f"  chunk {chunk:2d} L={depth} loss {mean:.3f} | "
              + " ".join(f"K{k}:{a[k]:.2f}" for k in range(1, KMAX + 1)))
        # plateau-triggered growth: loss barely improved & still failing high-K
        improved = 1.0 if prev is None else (prev - mean) / max(prev, 1e-6)
        if (auto_grow and chunk >= 3 and depth < MAX_LAYERS and cooldown == 0
                and improved < 0.03 and a[KMAX] < 0.8):
            grow_deeper(core, 2, trainable=True)
            opt = opt_for(core)            # include the new params
            cooldown = 2
            print(f"     -> saturation detected (Δloss {improved:.1%}); GROW to L={len(core.blocks)}")
        else:
            cooldown = max(0, cooldown - 1)
        prev = mean
    return acc_by_k(core, world, device), len(core.blocks)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
    a_fixed, _ = run(world, device, auto_grow=False)
    a_auto, final_L = run(world, device, auto_grow=True)
    print("\n== final ==")
    print("  fixed L=2:        " + " ".join(f"K{k}:{a_fixed[k]:.2f}" for k in range(1, KMAX + 1)))
    print(f"  auto-grown L={final_L}:   " + " ".join(f"K{k}:{a_auto[k]:.2f}" for k in range(1, KMAX + 1)))
    print("  auto-grow should climb high-K well above fixed L=2 while keeping K1 -- the")
    print("  controller grew capacity by itself, no forgetting = autonomous small->large.")


if __name__ == "__main__":
    main()
