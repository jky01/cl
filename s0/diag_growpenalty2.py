"""Push the growth-penalty test to LARGER scale: deeper (2->12) on a HARDER task
(K-hop with kmax=7, so deep models have headroom). Does the grown-vs-scratch gap
widen at L=10/12 (penalty emerging) or stay flat (growth keeps scaling)?

  .venv/bin/python -m s0.diag_growpenalty2
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .pad import pad_batch

KMAX = 7
REL = 0
S = 1200
GROWN = [2, 4, 6, 8, 10, 12]
COMPARE = [4, 8, 12]
N_SEEDS = 2


def gen(world, device, B):
    seqs, ans = [], []
    ent = world.entities
    for _ in range(B):
        K = world.rng.randint(1, KMAX)
        nodes = world.rng.sample(range(world.cfg.n_entities), K + 1)
        edges = [(nodes[i], nodes[i + 1]) for i in range(K)]
        world.rng.shuffle(edges)
        toks = [world.i("<bos>")]
        for (a, b) in edges:
            toks += [world.i("the"), world.i(world.relations[REL]), world.i("of"),
                     world.i(ent[a]), world.i("is"), world.i(ent[b]), world.i(".")]
        toks += [world.i("<sep>"), world.i(ent[nodes[0]]), world.i("<ans>")]
        seqs.append(toks); ans.append(world.i(ent[nodes[-1]]))
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    return ids, lengths, torch.tensor(ans, device=device)


def train_core(core, world, device, steps, lr=3e-3, B=64):
    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        ids, lengths, ans = gen(world, device, B)
        rows = torch.arange(ids.size(0), device=device)
        logits = core.lm_head(core.hidden(ids)[rows, lengths - 1])
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    core.eval()


@torch.no_grad()
def acc(core, world, device, n=2048):
    core.eval()
    ids, lengths, ans = gen(world, device, n)
    rows = torch.arange(n, device=device)
    pred = core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1)
    return (pred == ans).float().mean().item()


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    torch.manual_seed(seed); world.rng.seed(seed)
    core = ProxyCore(V, d_model=128, n_layers=2, n_heads=4, max_len=72).to(device)
    grown = {}
    prev = 2
    for L in GROWN:
        if L > prev:
            grow_deeper(core, L - prev, trainable=True); prev = L
        train_core(core, world, device, S)
        grown[L] = acc(core, world, device)
    scratch = {}
    for L in COMPARE:
        torch.manual_seed(seed); world.rng.seed(seed)
        c = ProxyCore(V, d_model=128, n_layers=L, n_heads=4, max_len=72).to(device)
        train_core(c, world, device, (L // 2) * S)
        scratch[L] = acc(c, world, device)
    return grown, scratch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = {L: [] for L in GROWN}; SC = {L: [] for L in COMPARE}
    for seed in range(N_SEEDS):
        grown, scratch = run_seed(seed, device)
        for L in GROWN: G[L].append(grown[L])
        for L in COMPARE: SC[L].append(scratch[L])
        print(f"  seed {seed}: grown " + " ".join(f"L{L}:{grown[L]:.2f}" for L in GROWN)
              + " | scratch " + " ".join(f"L{L}:{scratch[L]:.2f}" for L in COMPARE), flush=True)
    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n== penalty over {N_SEEDS} seeds (kmax={KMAX}, deeper) ==")
    print(f"  {'L':>3} | {'grown':>7} {'scratch':>8} {'gap':>7}")
    for L in COMPARE:
        g, s = mean(G[L]), mean(SC[L])
        print(f"  {L:>3} | {g:7.3f} {s:8.3f} {s - g:7.3f}")
    print("  gap staying flat as L grows => growth keeps scaling; widening => penalty emerges.")


if __name__ == "__main__":
    main()
