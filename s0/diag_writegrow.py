"""NEURALIZE the write-vs-grow decision — the last hand-rule in the integrated loop
(diag_system had facts->memory, plateau->grow as a fixed rule). Here a learned GATE
decides, per incoming item, whether to WRITE it to external memory or CONSOLIDATE it
via growth, from the item's frozen-core features alone (it is NOT told the type).

Two item types with a genuine mechanism fit:
  FACT   -> memory stores & recalls it (write=good); growth can't store one fact (grow=waste)
  SKILL  -> a K-hop reasoning demand needs growth+training (grow=good); memory can't reason (write=0)
Contextual-bandit reward by (type, action); the gate must infer type from features and
route. Test on HELD-OUT items vs oracle / always-write / always-grow. If the gate
matches the oracle, the write-vs-grow decision is neural (completes the loop's modulation).

  python3 -m s0.diag_writegrow
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore
from .pad import pad_batch
from .diag_grow_hops import gen as gen_hops, train_core

EPISODES = int(os.environ.get("WG_EPISODES", 1500))
GROW_COST = 0.15                       # growth is expensive: subtract from its reward


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
    d = 128
    core = ProxyCore(world.vocab_size, d_model=d, n_heads=4, n_layers=2, max_len=72).to(device)
    train_core(core, world, device, 1200)        # a lightly-trained FROZEN feature extractor
    for p in core.parameters():
        p.requires_grad_(False)
    core.eval()
    ents, rels = world.entities, world.relations
    rng = random.Random(0)

    def item_feat(kind, seed=None):
        r = random.Random(seed) if seed is not None else rng
        if kind == "fact":                        # "<bos> subj rel <ans>"
            s = r.randrange(world.cfg.n_entities); rel = r.randrange(len(rels))
            seq = [world.i("<bos>"), world.i(ents[s]), world.i(rels[rel]), world.i("<ans>")]
        else:                                     # a K-hop reasoning query (skill demand)
            K = r.randint(3, 6); nodes = r.sample(range(world.cfg.n_entities), K + 1)
            edges = [(nodes[i], nodes[i + 1]) for i in range(K)]; r.shuffle(edges)
            seq = [world.i("<bos>")]
            for (a, b) in edges:
                seq += [world.i("the"), world.i(rels[0]), world.i("of"),
                        world.i(ents[a]), world.i("is"), world.i(ents[b]), world.i(".")]
            seq += [world.i("<sep>"), world.i(ents[nodes[0]]), world.i("<ans>")]
        ids, lengths = pad_batch([seq], world.i("<pad>"), device)
        with torch.no_grad():
            h = core.hidden(ids)
        return h[0, lengths[0] - 1]                # frozen last-token feature

    # reward by (type, action): action 0=write(memory), 1=grow(consolidate)
    def reward(kind, action):
        if kind == "fact":
            return 1.0 if action == 0 else (0.2 - GROW_COST)     # memory recalls it; growth wastes
        else:                                                    # skill
            return 0.0 if action == 0 else (1.0 - GROW_COST)     # memory can't reason; growth solves

    gate = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    baseline = 0.0
    for ep in range(EPISODES):
        kind = rng.choice(["fact", "skill"])
        f = item_feat(kind)
        logits = gate(f); p = torch.softmax(logits, -1)
        a = torch.multinomial(p, 1).item()
        rwd = reward(kind, a)
        baseline = 0.95 * baseline + 0.05 * rwd
        loss = -(rwd - baseline) * torch.log(p[a] + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 300 == 0 or ep == EPISODES - 1:
            print(f"  ep {ep:4d} kind={kind:5s} action={'write' if a==0 else 'grow '} "
                  f"reward {rwd:+.2f} baseline {baseline:.3f}", flush=True)

    # ---- held-out evaluation (fresh seeded items) ----
    @torch.no_grad()
    def route(kind, n=200):
        hits = 0; rw = 0.0
        for i in range(n):
            f = item_feat(kind, seed=10000 + i)
            a = gate(f).argmax(-1).item()
            correct = (a == 0) if kind == "fact" else (a == 1)
            hits += correct; rw += reward(kind, a)
        return hits / n, rw / n

    print("\n== held-out routing: learned gate vs oracle / always-write / always-grow ==")
    fa, fr = route("fact"); sa, sr = route("skill")
    gate_rw = (fr + sr) / 2; gate_acc = (fa + sa) / 2
    oracle = (reward("fact", 0) + reward("skill", 1)) / 2
    aw = (reward("fact", 0) + reward("skill", 0)) / 2
    ag = (reward("fact", 1) + reward("skill", 1)) / 2
    print(f"  gate routing acc: fact->write {fa:.2f}, skill->grow {sa:.2f}  (mean {gate_acc:.2f})")
    print(f"  mean reward: gate {gate_rw:.3f} | oracle {oracle:.3f} | always-write {aw:.3f} | always-grow {ag:.3f}")
    print("\n  gate acc ~1.0 and reward ~= oracle (>> always-write / always-grow) => the")
    print("  write-vs-grow decision is a LEARNED neural policy from features (loop fully modulated).")


if __name__ == "__main__":
    main()
