"""Does GROWTH improve capability? A depth-sensitive task: in-context K-hop
chain traversal. Edges x_i --r--> x_{i+1} are shown SHUFFLED; the model must
follow r from the start to the chain's end. Each hop needs ~one more layer, so a
shallow core saturates at larger K; deepening (function-preserving) + continued
training should break through WITHOUT forgetting small-K.

  .venv/bin/python -m s0.diag_grow_hops
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .pad import pad_batch

KMAX = 5
REL = 0  # single relation for the chain


def gen(world, device, B, kmax=KMAX):
    """Batch of K-hop traversal problems (K uniform in 1..kmax)."""
    seqs, ans, ks = [], [], []
    ent = world.entities
    for _ in range(B):
        K = world.rng.randint(1, kmax)
        nodes = world.rng.sample(range(world.cfg.n_entities), K + 1)
        edges = [(nodes[i], nodes[i + 1]) for i in range(K)]
        shuf = edges[:]; world.rng.shuffle(shuf)
        toks = [world.i("<bos>")]
        for (a, b) in shuf:                       # "the r of A is B ."
            toks += [world.i("the"), world.i(world.relations[REL]), world.i("of"),
                     world.i(ent[a]), world.i("is"), world.i(ent[b]), world.i(".")]
        toks += [world.i("<sep>"), world.i(ent[nodes[0]]), world.i("<ans>")]
        seqs.append(toks); ans.append(world.i(ent[nodes[-1]])); ks.append(K)
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    return ids, lengths, torch.tensor(ans, device=device), torch.tensor(ks)


def train_core(core, world, device, steps, lr=3e-3, B=64):
    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        logits = core.lm_head(core.hidden(ids)[rows, lengths - 1])
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    core.eval()


@torch.no_grad()
def acc_by_k(core, world, device, n=2048):
    core.eval()
    ids, lengths, ans, ks = gen(world, device, n)
    rows = torch.arange(n, device=device)
    pred = core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1)
    correct = (pred == ans).cpu()
    return {k: correct[(ks == k)].float().mean().item() for k in range(1, KMAX + 1)}


def show(tag, d):
    print(f"  {tag:22s} " + " ".join(f"K{k}:{d[k]:.2f}" for k in range(1, KMAX + 1)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))

    core = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=64).to(device)
    print(f"== train L=2 core on K-hop traversal ({device}) ==")
    train_core(core, world, device, steps=6000)
    a_small = acc_by_k(core, world, device)
    show("L=2 (trained):", a_small)

    # function-preserving grow to L=4 (trainable), verify no change at growth
    before = acc_by_k(core, world, device)
    grow_deeper(core, n_new=2, trainable=True)
    after = acc_by_k(core, world, device)
    show("L=4 @growth:", after)
    print(f"  [growth preserved] max|Δacc| = "
          f"{max(abs(after[k]-before[k]) for k in range(1,KMAX+1)):.3f}")

    # continue training the grown core
    train_core(core, world, device, steps=6000)
    a_grown = acc_by_k(core, world, device)
    show("L=4 (grown+trained):", a_grown)

    # control: a fresh L=2 trained for the SAME extra steps (12000 total)
    torch.manual_seed(0); world.rng.seed(0)
    ctrl = ProxyCore(world.vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=64).to(device)
    train_core(ctrl, world, device, steps=12000)
    show("L=2 control (2x steps):", acc_by_k(ctrl, world, device))
    print("  growth wins if L=4(grown) beats both L=2(trained) and L=2-control at high K,"
          " while keeping low-K -> added depth = added capability, no forgetting.")


if __name__ == "__main__":
    main()
