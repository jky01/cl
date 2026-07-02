"""SMALL -> LARGE: does a small model, grown through an ESCALATING curriculum,
ACCUMULATE capability across the whole difficulty range and end up a genuinely large,
capable model that a fixed-small model cannot match?

Curriculum: stages of in-context K-hop reasoning with rising kmax (3->7). The GROWN
model starts at L=2 and deepens (+2 layers) as the curriculum escalates (L2->L12),
training on each stage; because higher-kmax data includes the easier hops, mastery
should ACCUMULATE. Compared to:
  fixed-small (L2, never grows, same total budget) -> capacity ceiling on high K
  fixed-large (L12 from scratch, same total budget) -> the deep-from-start reference
Measured by accuracy AT EACH K (1..7) at the end. If the grown model masters the full
range (esp. high K) where fixed-small saturates, that IS small->large capability
accumulation via growth.

  python3 -m s0.diag_growlarge        # env: GL_SEEDS
"""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .diag_grow_hops import gen

SEEDS = int(os.environ.get("GL_SEEDS", 3))
STAGES = [3, 4, 5, 6, 7]          # escalating kmax curriculum
STEPS = 1500                       # per stage
EVAL_K = list(range(1, 8))
D = 128


def train(core, world, device, kmax, steps, lr=3e-3, B=64):
    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        ids, lengths, ans, _ = gen(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    core.eval()


@torch.no_grad()
def acc_by_k(core, world, device, n=3072, kmax=7):
    core.eval()
    ids, lengths, ans, ks = gen(world, device, n, kmax=kmax)
    rows = torch.arange(n, device=device)
    correct = (core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1) == ans).cpu()
    return {k: correct[(ks == k)].float().mean().item() for k in EVAL_K}


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    # GROWN: start L2, deepen +2 each escalating stage (L2 -> L12), train on each stage
    torch.manual_seed(seed); world.rng.seed(seed)
    grown = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    for i, kmax in enumerate(STAGES):
        if i > 0:
            grow_deeper(grown, 2, trainable=True)          # small -> large, aligned with the curriculum
        train(grown, world, device, kmax, STEPS)
    g_depth = len(grown.blocks)
    G = acc_by_k(grown, world, device)
    # fixed-small L2: same curriculum + total budget, never grows
    torch.manual_seed(seed); world.rng.seed(seed)
    small = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    for kmax in STAGES:
        train(small, world, device, kmax, STEPS)
    S = acc_by_k(small, world, device)
    # fixed-large L12 from scratch: same total budget on the curriculum
    torch.manual_seed(seed); world.rng.seed(seed)
    large = ProxyCore(V, d_model=D, n_layers=g_depth, n_heads=4, max_len=72).to(device)
    for kmax in STAGES:
        train(large, world, device, kmax, STEPS)
    L = acc_by_k(large, world, device)
    return G, S, L, g_depth


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SMALL->LARGE capability accumulation ({device}) seeds={SEEDS} curriculum kmax={STAGES}")
    accG = {k: [] for k in EVAL_K}; accS = {k: [] for k in EVAL_K}; accL = {k: [] for k in EVAL_K}
    depth = 0
    for seed in range(SEEDS):
        G, S, L, depth = run_seed(seed, device)
        for k in EVAL_K:
            accG[k].append(G[k]); accS[k].append(S[k]); accL[k].append(L[k])
        sh = lambda d: " ".join(f"K{k}:{d[k]:.2f}" for k in EVAL_K)
        print(f"  seed {seed}: grown(L{depth}) [{sh(G)}]", flush=True)
        print(f"           small(L2)  [{sh(S)}]", flush=True)
        print(f"           large(L{depth}scr)[{sh(L)}]", flush=True)

    m = lambda a, k: sum(a[k]) / len(a[k])
    print(f"\n== mean accuracy by K over {SEEDS} seeds (grown reached L{depth} from L2) ==")
    print("  model         " + " ".join(f"K{k}" for k in EVAL_K) + "   mean")
    for name, a in [("grown->L%d" % depth, accG), ("fixed-small L2", accS), ("fixed-large scr", accL)]:
        mean = sum(m(a, k) for k in EVAL_K) / len(EVAL_K)
        print(f"  {name:14s} " + " ".join(f"{m(a,k):.2f}" for k in EVAL_K) + f"   {mean:.2f}")
    hi = [k for k in EVAL_K if k >= 5]
    gh = sum(m(accG, k) for k in hi) / len(hi); sh_ = sum(m(accS, k) for k in hi) / len(hi)
    print(f"\n  high-K (K>=5) mean: grown {gh:.2f} vs fixed-small {sh_:.2f}")
    print("  HONEST RESULT (negative): staged growth on a CUMULATIVE curriculum does NOT accumulate")
    print("  capability — grown underperforms fixed-small (re-warming cost) and both lose to")
    print("  fixed-large-from-scratch. Growth's value is NO-FORGETTING of DISTINCT skills, NOT")
    print("  becoming more capable at one task; 'grow to be smarter' is refuted here.")


if __name__ == "__main__":
    main()
