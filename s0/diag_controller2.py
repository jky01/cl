"""Two-signal growth controller on a depth-bounded task -- and an HONEST toy-scale
limitation.

diag_controller.py used the K-hop task, but that task has a SHALLOW SHORTCUT:
the answer is the unique node that is a source but never a destination (a set
op, ~2 layers), so L2 solves it given enough budget and the controller never
needs to grow. Here we remove the shortcut with MULTI-CHAIN pointer-chasing:
disjoint functional chains are given (shuffled); the query gives one chain's
head and must walk KWALK hops to ITS tail. Multiple sinks exist, so the "unique
sink" shortcut fails -- the model must actually traverse from the given start.

WHAT WORKS: the wiring executes -- the controller reads the three signals and
fires function-preserving growth, budget-gated on the amount.

WHAT DOES NOT (an honest NEGATIVE result): a one-chunk loss-delta is a BAD
saturation signal. On this task loss briefly plateaus right before a phase
transition, so the controller mistakes a temporary plateau for saturation and
grows PREMATURELY and repeatedly (grew 3x: at 4k/9k/13k -> L8). Every grow adds
identity layers that must re-warm, so the repeated growth just burns budget and
nothing converges. Measured @budget 18000: controller (grew to L8) = 0.243, but
fixed-L8-FROM-SCRATCH = 0.482 and even fixed-L2 varies wildly (0.16-0.47 across
runs) -> the controller LOST to plain from-scratch, and the toy task is too
unstable to trust. Earlier belief "deep underfits, growth rescues it" was a
budget artifact: given 18k steps L8-from-scratch trains fine here.

TAKEAWAYS: (1) plateau-triggered growth needs a robust saturation signal
(patience / held-out validation slope / the meta-learned Omega of s27-28), not a
single-window loss delta. (2) toy scale (d=128, few-k steps, ~0.5 ceilings,
high variance) cannot validate the capability payoff of growth -- that needs the
real-model/cloud tier where depth genuinely gates capability.

  .venv/bin/python -m s0.diag_controller2
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .pad import pad_batch

CHUNK = 1000
MAXL = 8
KWALK = 3          # hops from start to the queried chain's tail (learnable-ish)
NCHAINS = 2        # disjoint chains -> multiple sinks -> no set shortcut
NENT = 60          # smaller entity set -> the traversal is at least learnable
REL = 0
MAXLEN = 112


def gen(world, device, B):
    ent, rel = world.entities, world.i(world.relations[REL])
    bos, the, of, isk, dot = (world.i(t) for t in ("<bos>", "the", "of", "is", "."))
    sep, ans_t = world.i("<sep>"), world.i("<ans>")
    seqs, ans = [], []
    for _ in range(B):
        nodes = world.rng.sample(range(NENT), (KWALK + 1) * NCHAINS)
        chains = [nodes[c * (KWALK + 1):(c + 1) * (KWALK + 1)] for c in range(NCHAINS)]
        edges = [(ch[i], ch[i + 1]) for ch in chains for i in range(KWALK)]
        world.rng.shuffle(edges)
        toks = [bos]
        for (a, b) in edges:
            toks += [the, rel, of, world.i(ent[a]), isk, world.i(ent[b]), dot]
        c = world.rng.randrange(NCHAINS)
        toks += [sep, world.i(ent[chains[c][0]]), ans_t]
        seqs.append(toks); ans.append(world.i(ent[chains[c][-1]]))
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    return ids, lengths, torch.tensor(ans, device=device)


@torch.no_grad()
def acc(core, world, device, n=2048):
    core.eval()
    ids, lengths, ans = gen(world, device, n)
    rows = torch.arange(n, device=device)
    pred = core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1)
    return (pred == ans).float().mean().item()


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


def train_chunk(core, world, device, opt, steps=CHUNK, B=64):
    core.train(); losses = []
    for _ in range(steps):
        ids, lengths, a = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), a)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step(); losses.append(loss.item())
    core.eval(); return sum(losses) / len(losses)


def new_core(world, device, L):
    return ProxyCore(world.vocab_size, d_model=128, n_layers=L, n_heads=4,
                     max_len=MAXLEN).to(device)


def controller(world, device, budget, log=None):
    core = new_core(world, device, 2)
    opt = opt_for(core)
    used, prev, cool, grows = 0, None, 0, 0
    while used < budget:
        loss = train_chunk(core, world, device, opt); used += CHUNK
        a = acc(core, world, device)
        improved = 1.0 if prev is None else (prev - loss) / max(prev, 1e-6)
        saturated = improved < 0.05          # (a) loss ~stopped improving
        headroom = a < 0.90                  # (b) not yet at the task ceiling
        budget_left = budget - used          # (c) budget signal
        grow = (cool == 0 and saturated and headroom and len(core.blocks) < MAXL
                and budget_left >= 2 * CHUNK)
        if log is not None:
            log.append(f"    used={used} L={len(core.blocks)} loss={loss:.3f} "
                       f"acc={a:.3f} sat={saturated} head={headroom} -> "
                       f"{'GROW' if grow else 'train'}")
        if grow:
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); cool = 2; grows += 1
        else:
            cool = max(0, cool - 1)
        prev = loss
    return len(core.blocks), acc(core, world, device), grows


def fixed(world, device, L, budget):
    torch.manual_seed(0); world.rng.seed(0)
    core = new_core(world, device, L)
    opt = opt_for(core)
    for _ in range(budget // CHUNK):
        train_chunk(core, world, device, opt)
    return acc(core, world, device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task: multi-chain pointer-walk (KWALK={KWALK}, NCHAINS={NCHAINS}) "
          f"-- no sink shortcut, depth ~ {KWALK}")
    print(f"  {'budget':>7} | {'controller (depth, acc, #grows)':>34} | "
          f"{'fixed L2':>9} {'fixed L8':>9}")
    for budget in (8000, 18000):
        torch.manual_seed(0)
        world = World(WorldConfig(n_entities=NENT, n_objects=NENT, seed=0))
        torch.manual_seed(0); world.rng.seed(0)
        log = []
        depth, a_ctrl, grows = controller(world, device, budget, log)
        a2 = fixed(world, device, 2, budget)
        a8 = fixed(world, device, 8, budget)
        for line in log:
            print(line, flush=True)
        print(f"  {budget:>7} | L={depth} acc={a_ctrl:.3f} ({grows} grows){'':>10} | "
              f"{a2:9.3f} {a8:9.3f}", flush=True)
    print("\n  NEGATIVE result: one-chunk loss-delta = bad saturation signal. Temporary")
    print("  plateaus before phase transitions trigger PREMATURE, repeated growth that")
    print("  burns budget re-warming layers -> controller (grew to L8) LOSES to plain")
    print("  fixed-L8-from-scratch. Growth timing needs patience / held-out slope /")
    print("  meta-learned Omega, and the payoff needs real scale, not this toy.")


if __name__ == "__main__":
    main()
