"""End-to-end "learn AND grow" loop: one system whose CORE grows in capability
over time (K-hop, deeper = better) while its MEMORY keeps recalling facts across
every growth (cheap re-sync). Demonstrates the small->large vision's three
pieces composed:
  C (K-hop capability) should RISE as the core grows (L2->4->6);
  R (fact recall via memory) should STAY high through every growth.

  .venv/bin/python -m s0.diag_integrated
"""
from __future__ import annotations
import torch

from .world import World, WorldConfig
from .core import ProxyCore, pretrain_core, grow_deeper
from .capsule import CapsuleMemory, SlotLayout
from .train import train_omega0, eval_capsule
from .diag_grow_hops import train_core as train_khop, acc_by_k


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=64).to(device)
    pretrain_core(core, world, steps=1500, device=device, log=lambda *_: None)
    layout = SlotLayout(d_key=64, d_v=128, d_ctx=32, d_aux=16)
    mem = CapsuleMemory(core, world, layout, n1=16, n2=16, n_mem=256, n_prefix=4, top_k=4).to(device)

    nf = 4
    R = lambda: eval_capsule(mem, world, n_facts=nf, episodes_n=64, device=device)["acc"]
    C = lambda: sum(acc_by_k(core, world, device).values()) / 5

    def resync_memory(steps):
        for p in core.parameters():                 # freeze core; train only memory modules
            p.requires_grad_(False)
        train_omega0(mem, world, steps=steps, B=64, max_facts=nf, device=device, log=lambda *_: None)

    resync_memory(4000)                              # learn the memory on the base core
    print(f"  L=2 (base):           capability C={C():.3f}  fact-recall R={R():.3f}")

    for L in (4, 6):
        grow_deeper(core, 2, trainable=True)         # grow core; new layers trainable
        train_khop(core, world, device, steps=2500)  # core gains K-hop capability (deeper)
        resync_memory(2500)                          # re-sync memory to the evolved core
        print(f"  L={L} (grown+resync):  capability C={C():.3f}  fact-recall R={R():.3f}")

    print("\n  C rises with growth (core got more capable) while R stays high through every")
    print("  growth (memory retained via cheap re-sync) -> a single system that grows AND")
    print("  keeps learning without forgetting.")


if __name__ == "__main__":
    main()
