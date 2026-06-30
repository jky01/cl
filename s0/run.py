"""Step-0 entrypoint: pretrain frozen core, train Omega-0, sweep Accuracy(N_facts).

  python -m s0.run --smoke      # tiny, runs in well under a minute on CPU
  python -m s0.run              # default small config
  python -m s0.run --facts 1 10 100 --core-steps 800 --omega-steps 1500

Success gate for Step 0 (per reference/3.md sec 11.4): at N_facts=1 the capsule
must reach high exact-match accuracy AND keep locality ~1.0. If it cannot store
a single fact, stop -- the deeper layers (dream/rollback/sleep) are moot.
"""

from __future__ import annotations
import argparse
import torch

from .world import World, WorldConfig
from .core import ProxyCore, pretrain_core
from .capsule import CapsuleMemory, SlotLayout
from .train import train_omega0, eval_capsule, eval_baseline, eval_conflict
from .baselines import ALL_BASELINES


def build(args, device):
    world = World(WorldConfig(n_entities=args.entities, n_relations=args.relations,
                              n_objects=args.objects, seed=args.seed))
    core = ProxyCore(world.vocab_size, d_model=args.d_model, n_layers=args.layers,
                     n_heads=args.heads, max_len=args.max_len)
    layout = SlotLayout(d_key=args.d_model // 2, d_v=args.d_model,
                        d_ctx=32, d_aux=16)
    # product codebooks: n1*n2 == n_mem
    n1 = n2 = int(round(args.n_mem ** 0.5))
    n_mem = n1 * n2
    mem = CapsuleMemory(core, world, layout, n1=n1, n2=n2, n_mem=n_mem,
                        n_prefix=args.prefix, top_k=args.topk)
    return world, core, mem, n_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entities", type=int, default=60)
    ap.add_argument("--relations", type=int, default=8)
    ap.add_argument("--objects", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--n-mem", type=int, default=256)
    ap.add_argument("--prefix", type=int, default=4)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--core-steps", type=int, default=1500)
    ap.add_argument("--omega-steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--omega-batch", type=int, default=32)
    # Multi-fact episodes up to this size (Step 1). The curriculum in
    # train_omega0 ramps episode size 1 -> max_facts so large values bootstrap
    # cleanly (without it, big max_facts collapses to chance).
    ap.add_argument("--max-facts", type=int, default=16)
    ap.add_argument("--facts", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--eval-episodes", type=int, default=64)
    args = ap.parse_args()

    if args.smoke:
        args.entities, args.objects = 60, 60
        args.d_model, args.layers, args.max_len = 64, 2, 32
        args.n_mem = 64
        # core needs ~1k steps before the copy-from-prompt readout works.
        # Multi-fact + locality (gate selectivity) need ~5-6k steps to converge.
        args.core_steps, args.omega_steps, args.omega_batch = 1000, 6000, 64
        args.max_facts = 8
        args.facts = [1, 4, 8, 16]
        args.eval_episodes = 32

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    world, core, mem, n_mem = build(args, device)
    mem.to(device)            # moves core + all Omega-0 modules (and buffers)
    print(f"device={device} vocab={world.vocab_size} d_model={args.d_model} "
          f"layers={args.layers} n_mem={n_mem} prefix={args.prefix} topk={args.topk}")

    print("\n== pretrain frozen proxy core (syntax only) ==")
    pretrain_core(core, world, steps=args.core_steps, device=device)

    print("\n== train Omega-0 (multi-fact episodes w/ hard negatives) ==")
    train_omega0(mem, world, steps=args.omega_steps, B=args.omega_batch,
                 max_facts=args.max_facts, lr=args.lr, device=device)

    print("\n== Accuracy(N_facts):  exact-match object accuracy ==")
    header = f"{'N_facts':>8} | {'capsule':>8} {'(loc)':>6} | " + " ".join(
        f"{b.name:>13}" for b in ALL_BASELINES)
    print(header); print("-" * len(header))
    for nf in args.facts:
        if nf > n_mem:
            continue
        cap = eval_capsule(mem, world, n_facts=nf, episodes_n=args.eval_episodes, device=device)
        cells = []
        for B in ALL_BASELINES:
            acc = eval_baseline(B, core, world, n_facts=nf,
                                episodes_n=max(8, args.eval_episodes // 4), device=device)
            cells.append(f"{acc:13.3f}")
        print(f"{nf:>8} | {cap['acc']:8.3f} {cap['locality']:6.3f} | " + " ".join(cells))

    print("\nReading: capsule should beat A:no-mem and approach D:oracle-slot at "
          "N_facts=1; watch how it degrades vs N_facts and vs B/C/E.")

    conf = eval_conflict(mem, world, episodes_n=args.eval_episodes, device=device)
    print("\n== Conflict versioning (write o_before@t=0 then o_now@t=1) ==")
    print(f"  now-acc {conf['now']:.3f}  before-acc {conf['before']:.3f}  "
          f"routing-fail {conf['routing_fail']:.3f}")
    print("  (non-destructive update: both versions recalled, context-routed)")


if __name__ == "__main__":
    main()
