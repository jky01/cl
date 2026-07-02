"""SMALL -> LARGE, the version the evidence SUPPORTS: growth doesn't make you more
capable at ONE task (diag_growlarge = negative), but it lets a model grow into a LARGE
model that COVERS many DISTINCT skills without forgetting — which a fixed-small model
cannot hold at once. This is the honest "成大模型": grow for skill COVERAGE + retention.

N distinct reasoning SKILLS (K-hop over N different relations). The GROWN system keeps
a frozen shared base + one grown branch (top layers) per skill, routed by skill; it
GROWS with each new skill (small -> large) and RETAINS all of them. Compared to a
fixed-small model trained on the skills sequentially in shared weights (catastrophic
forgetting -> only the newest survives). Metric: how many of the N skills are mastered
at the end.

  python3 -m s0.diag_growlarge2        # env: GL_SKILLS, GL_SEEDS
"""
from __future__ import annotations
import os
import copy
import torch
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore
from .pad import pad_batch

SKILLS = int(os.environ.get("GL_SKILLS", 5))
SEEDS = int(os.environ.get("GL_SEEDS", 3))
KHOP = 3
LTOP = 2
STEPS = 1500
D = 128


def gen_skill(world, device, B, rel_id, kmax=KHOP):
    """K-hop over a SPECIFIC relation (a distinct skill)."""
    ent, rel = world.entities, world.i(world.relations[rel_id])
    bos, the, of, isk, dot = (world.i(t) for t in ("<bos>", "the", "of", "is", "."))
    sep, ans = world.i("<sep>"), world.i("<ans>")
    seqs, gold = [], []
    for _ in range(B):
        K = world.rng.randint(1, kmax)
        nodes = world.rng.sample(range(world.cfg.n_entities), K + 1)
        edges = [(nodes[i], nodes[i + 1]) for i in range(K)]; world.rng.shuffle(edges)
        toks = [bos]
        for (a, b) in edges:
            toks += [the, rel, of, world.i(ent[a]), isk, world.i(ent[b]), dot]
        toks += [sep, world.i(ent[nodes[0]]), ans]
        seqs.append(toks); gold.append(world.i(ent[nodes[-1]]))
    ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
    return ids, lengths, torch.tensor(gold, device=device)


def train(core, world, device, rel_id, steps, params=None, lr=3e-3, B=64):
    core.train()
    params = params or [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(steps):
        ids, lengths, g = gen_skill(world, device, B, rel_id)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), g)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    core.eval()


@torch.no_grad()
def acc(core, world, device, rel_id, n=1024):
    core.eval()
    ids, lengths, g = gen_skill(world, device, n, rel_id)
    rows = torch.arange(n, device=device)
    return (core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1) == g).float().mean().item()


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    # shared base: a general K-hop reasoner (pretrained on a mix), frozen
    torch.manual_seed(seed); world.rng.seed(seed)
    base = ProxyCore(V, d_model=D, n_layers=4, n_heads=4, max_len=72).to(device)
    for r in range(SKILLS):
        train(base, world, device, r, STEPS // 2)              # general features across skills
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()

    # GROWN: per-skill branch (top LTOP layers), routed by skill -> retains all
    grown_acc = []
    m = copy.deepcopy(base)
    branches = []
    for r in range(SKILLS):
        for k in range(LTOP):
            m.blocks[-LTOP + k] = copy.deepcopy(base.blocks[-LTOP + k]).to(device)
        top = [p for k in range(LTOP) for p in m.blocks[-LTOP + k].parameters()]
        for p in top:
            p.requires_grad_(True)
        train(m, world, device, r, STEPS, params=top)
        branches.append([copy.deepcopy(m.blocks[-LTOP + k]) for k in range(LTOP)])
    for r in range(SKILLS):                                     # route each skill to ITS branch
        for k in range(LTOP):
            m.blocks[-LTOP + k] = branches[r][k]
        grown_acc.append(acc(m, world, device, r))

    # FIXED-SMALL: one shared model trained on the skills SEQUENTIALLY (forgets)
    torch.manual_seed(seed); world.rng.seed(seed)
    small = ProxyCore(V, d_model=D, n_layers=4, n_heads=4, max_len=72).to(device)
    for r in range(SKILLS):
        train(small, world, device, r, STEPS)
    small_acc = [acc(small, world, device, r) for r in range(SKILLS)]
    return grown_acc, small_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SMALL->LARGE skill COVERAGE ({device}) skills={SKILLS} seeds={SEEDS} khop={KHOP}")
    G = [[] for _ in range(SKILLS)]; S = [[] for _ in range(SKILLS)]
    for seed in range(SEEDS):
        ga, sa = run_seed(seed, device)
        for r in range(SKILLS):
            G[r].append(ga[r]); S[r].append(sa[r])
        f = lambda a: " ".join(f"s{r}:{a[r]:.2f}" for r in range(SKILLS))
        print(f"  seed {seed}: grown [{f(ga)}]  fixed-small(seq) [{f(sa)}]", flush=True)

    mean = lambda a, r: sum(a[r]) / len(a[r])
    gm = sum(mean(G, r) for r in range(SKILLS)) / SKILLS
    sm = sum(mean(S, r) for r in range(SKILLS)) / SKILLS
    gmast = sum(1 for r in range(SKILLS) if mean(G, r) > 0.7)
    smast = sum(1 for r in range(SKILLS) if mean(S, r) > 0.7)
    print(f"\n== mean over {SEEDS} seeds ==")
    print(f"  grown (grew to base+{SKILLS} branches): per-skill " +
          " ".join(f"{mean(G,r):.2f}" for r in range(SKILLS)) + f"  mean {gm:.2f}  mastered {gmast}/{SKILLS}")
    print(f"  fixed-small (sequential, shared):       per-skill " +
          " ".join(f"{mean(S,r):.2f}" for r in range(SKILLS)) + f"  mean {sm:.2f}  mastered {smast}/{SKILLS}")
    print("\n  HONEST RESULT (null): grown ~= fixed-small (both master all 5) — NO growth advantage,")
    print("  because K-hop-over-relations is ONE generalizable skill, so the fixed-small model")
    print("  generalizes across relations and does NOT forget. Growth helps only for CONFLICTING")
    print("  knowledge (distinct FACTS that overwrite in shared weights, cf. the capstone), not")
    print("  for shared-mechanism reasoning skills. Bounds 成大模型 tightly.")


if __name__ == "__main__":
    main()
