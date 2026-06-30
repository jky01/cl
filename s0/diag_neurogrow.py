"""Neural growth trigger: GrowableCore has learned per-layer gates (extra blocks
start as the identity). The task loss + a capacity penalty open them
differentiably, so the network decides its OWN effective depth -- the neural
replacement for the heuristic plateau trigger. Validate:
  * on the hard K-hop task the gates open (effective depth 2 -> ~6) and accuracy
    climbs, driven purely by loss + capacity cost (no hand-coded trigger);
  * a HIGH capacity penalty keeps it shallow (gates stay shut) = the gate is a
    real cost/benefit controller, not always-grow.

  .venv/bin/python -m s0.diag_neurogrow
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import GrowableCore
from .diag_grow_hops import gen, acc_by_k, KMAX


def run(world, device, lam_cap, steps=10000, B=64, lr=3e-3):
    torch.manual_seed(0); world.rng.seed(0)
    core = GrowableCore(world.vocab_size, d_model=128, n_base=2, n_grow=4,
                        n_heads=4, max_len=64).to(device)
    opt = torch.optim.AdamW(core.parameters(), lr=lr)
    print(f"\n== neural-gated growth, capacity penalty lam={lam_cap} ==")
    for step in range(steps):
        ids, lengths, ans, _ = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        logits = core.lm_head(core.hidden(ids)[rows, lengths - 1])
        loss = F.cross_entropy(logits, ans) + lam_cap * core.grow_gate_penalty()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0 or step == steps - 1:
            a = acc_by_k(core, world, device)
            cv = ",".join(f"{x:.2f}" for x in core.grow_contrib().tolist())
            print(f"  step {step:5d} eff_depth {core.effective_depth():.2f} "
                  f"contrib[{cv}] | " + " ".join(f"K{k}:{a[k]:.2f}" for k in range(1, KMAX + 1)))
    # ABLATION: force grow gates to 0 -> if accuracy drops, the extra blocks WERE
    # being used (the small gate value just hides weight up-scaling).
    a_full = acc_by_k(core, world, device)
    saved = core.gate_logit.data.clone()
    core.gate_logit.data[core.n_base:] = -30.0      # grow gates -> exactly 0
    a_abl = acc_by_k(core, world, device)
    core.gate_logit.data.copy_(saved)
    print("  ablation (grow gates->0): "
          + " ".join(f"K{k}:{a_full[k]:.2f}->{a_abl[k]:.2f}" for k in range(2, KMAX + 1)))
    return core


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
    run(world, device, lam_cap=0.05)   # mild cost: should engage depth it needs
    run(world, device, lam_cap=0.50)   # heavy cost: should stay shallow
    print("\n  With the UNGAMEABLE norm cost, contrib is now a faithful usage meter:")
    print("  mild -> grow blocks contribute, high-K climbs; heavy -> contrib~0, stays shallow.")
    print("  Ablation drop should now MATCH the reported contribution.")


if __name__ == "__main__":
    main()
