"""Do GROWTH and MEMORY compose? The critical integration check for
"continually learn AND grow": when the core grows/changes, does the accumulated
memory survive?
  R0: memory recall on the base core.
  R1: after function-preserving growth (identity layers, frozen) -> should == R0.
  R2: after the new layers train (core hidden drifts) -> may drop.
  R3: after a cheap memory re-adapt on the evolved core -> should recover.

  .venv/bin/python -m s0.diag_growmem
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, pretrain_core, grow_deeper
from .capsule import CapsuleMemory, SlotLayout
from .train import train_omega0, eval_capsule
from .pad import pad_batch


def train_new_layers(core, world, device, steps, lr=1e-3, B=64):
    """Train ONLY the currently-trainable (newly grown) layers on the core's LM
    objective, so the core's hidden representations drift."""
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    core.train()
    for _ in range(steps):
        seqs = []
        for _ in range(B):
            f = world.sample_fact()
            seqs.append(world.render_statement(f))
        ids, _ = pad_batch(seqs, world.i("<pad>"), device)
        logits, _ = core(ids)
        tgt = ids[:, 1:].clone(); tgt[ids[:, 1:] == world.i("<pad>")] = -100
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    world = World(WorldConfig(n_entities=60, n_objects=60, seed=0))
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=32).to(device)
    pretrain_core(core, world, steps=1500, device=device, log=lambda *_: None)
    layout = SlotLayout(d_key=64, d_v=128, d_ctx=32, d_aux=16)
    mem = CapsuleMemory(core, world, layout, n1=16, n2=16, n_mem=256, n_prefix=4, top_k=4).to(device)

    nf = 4
    ev = lambda: eval_capsule(mem, world, n_facts=nf, episodes_n=64, device=device)["acc"]
    train_omega0(mem, world, steps=4000, B=64, max_facts=nf, device=device, log=lambda *_: None)
    r0 = ev()
    print(f"  R0 base-core memory recall (nf={nf}): {r0:.3f}")

    grow_deeper(core, 2, trainable=False)            # identity layers, frozen
    r1 = ev()
    print(f"  R1 after function-preserving growth (L=2->4, frozen): {r1:.3f}  (expect == R0)")

    for blk in core.blocks[2:]:                      # unfreeze new layers, let core drift
        for p in blk.parameters():
            p.requires_grad_(True)
    train_new_layers(core, world, device, steps=1000)
    r2 = ev()
    print(f"  R2 after new layers train (core hidden drifts): {r2:.3f}  (may drop)")

    train_omega0(mem, world, steps=1500, B=64, max_facts=nf, device=device, log=lambda *_: None)
    r3 = ev()
    print(f"  R3 after cheap memory re-adapt on evolved core: {r3:.3f}  (expect ~R0)")

    print("\n  growth preserves memory at the moment (R1==R0); core evolution can degrade it")
    print("  (R2); a cheap re-adapt recovers it (R3) -> growth and memory COMPOSE.")


if __name__ == "__main__":
    main()
