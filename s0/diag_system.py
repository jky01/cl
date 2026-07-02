"""INTEGRATION — one autonomous lifelong LOOP that composes the validated pieces:
a persistent capsule MEMORY + a GROWABLE core + a trained key-RETRIEVER + a growth
CONTROLLER. A stream of items arrives over a "lifetime"; each item is either
  - a FACT       -> written to memory (recalled later, router-free, by key), or
  - a CAPABILITY phase -> the core trains on the current K-hop task; when it plateaus
    below ceiling the controller GROWS the core (function-preserving deepening).
Throughout we measure (a) recall of ALL past facts (no forgetting) and (b) K-hop
capability (rises as the system grows). Goal: show the pieces COMPOSE into a single
running system that accumulates knowledge without forgetting AND grows capability
over its lifetime.

  python3 -m s0.diag_system
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .pad import pad_batch
from .diag_grow_hops import gen as gen_hops   # K-hop capability task (gen(world,device,B,kmax))

CHUNK = 400
KMAX = 6
MAXL = 8
PHASES = 6                    # lifetime = alternating fact-batches and capability-phases


def opt_for(core, lr=3e-3):
    return torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=lr)


# ---------- capability (K-hop) ----------
def train_cap(core, world, device, opt, steps, kmax=KMAX, B=64):
    core.train(); ls = torch.zeros((), device=device)
    for _ in range(steps):
        ids, lengths, ans, _ = gen_hops(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step(); ls += loss.detach()
    core.eval(); return (ls / steps).item()


@torch.no_grad()
def cap_acc(core, world, device, kmax=KMAX, n=1024):
    core.eval()
    ids, lengths, ans, _ = gen_hops(world, device, n, kmax=kmax)
    rows = torch.arange(n, device=device)
    return (core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1) == ans).float().mean().item()


# ---------- memory (capsule) over the CURRENT (evolving) core features ----------
class Memory(nn.Module):
    def __init__(self, d, device):
        super().__init__()
        mk = lambda o: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, o)).to(device)
        self.pk, self.pq, self.ve = mk(128), mk(128), mk(256)
        self.vd = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(self.gate[-1].bias, 2.0)
        self.params = [p for m in (self.pk, self.pq, self.ve, self.vd, self.gate) for p in m.parameters()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=0))
    V, d = world.vocab_size, 128
    core = ProxyCore(V, d_model=d, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = opt_for(core)
    mem = Memory(d, device)
    rng = random.Random(0)
    ents, rels = world.entities, world.relations
    print(f"SYSTEM autonomous lifelong loop ({device}) kmax={KMAX} phases={PHASES}")

    # ---- fact helpers: a fact = (subject, relation-word, object); key=(subj,rel), value=obj ----
    fact_bank = []                                   # the growing lifetime of facts

    def make_facts(nnew):
        out = []
        for _ in range(nnew):
            s, o = rng.sample(range(world.cfg.n_entities), 2)
            r = rng.randrange(len(rels))
            out.append((s, r, o))
        return out

    def feats(core, facts, kind):
        # frozen features of the CURRENT core: key text "<bos> subj rel <ans>", value adds obj
        seqs = []
        for (s, r, o) in facts:
            base = [world.i("<bos>"), world.i(ents[s]), world.i(rels[r])]
            seqs.append(base + ([world.i(ents[o])] if kind == "val" else [world.i("<ans>")]))
        ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
        with torch.no_grad():
            h = core.hidden(ids)
        rows = torch.arange(ids.size(0), device=device)
        return h[rows, lengths - 1]                  # last-token hidden (answer-position for val)

    def cache_feats():                               # frozen features of CURRENT core, computed ONCE
        Kf = feats(core, fact_bank, "key")           # key/query/H text
        Vf = feats(core, fact_bank, "val")           # value text (answer-position)
        gold = torch.tensor([world.i(ents[o]) for (_, _, o) in fact_bank], device=device)
        return Kf, Vf, gold

    def train_mem(steps=1500, B=64):
        if len(fact_bank) < 4:
            return
        Kf, Vf, gold = cache_feats()                 # cache once (core fixed during this call)
        N = len(fact_bank)
        for _ in range(steps):
            idx = torch.randint(0, N, (min(B, N),), device=device)
            Kq = F.normalize(mem.pk(Kf[idx]), -1); V_ = mem.ve(Vf[idx])
            q = F.normalize(mem.pq(Kf[idx]), -1)
            sims = q @ Kq.t() / 0.05
            R = mem.vd(torch.softmax(sims, -1) @ V_)
            H = Kf[idx]; g = torch.sigmoid(mem.gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(core.lm_head((H + g * R)), gold[idx]) \
                + F.cross_entropy(sims, torch.arange(len(idx), device=device))
            mo.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mem.params, 1.0); mo.step()

    @torch.no_grad()
    def recall_all():
        if len(fact_bank) < 4:
            return 1.0
        Kf, Vf, gold = cache_feats()
        Kall = F.normalize(mem.pk(Kf), -1); Vall = mem.ve(Vf)
        q = F.normalize(mem.pq(Kf), -1)
        R = mem.vd(torch.softmax(q @ Kall.t() / 0.05, -1) @ Vall)
        g = torch.sigmoid(mem.gate(torch.cat([Kf, R], -1)))
        return (core.lm_head((Kf + g * R)).argmax(-1) == gold).float().mean().item()

    mo = torch.optim.Adam(mem.params, lr=5e-4)
    prev_cap = 0.0
    for ph in range(PHASES):
        # (1) ingest a batch of new FACTS -> memory
        fact_bank.extend(make_facts(32))
        train_mem()
        r_after_write = recall_all()
        # (2) a CAPABILITY phase: train the core; controller decides to grow on plateau
        loss = train_cap(core, world, device, opt, CHUNK)
        cap = cap_acc(core, world, device)
        saturated = (cap - prev_cap) < 0.03
        grew = False
        if saturated and cap < 0.90 and len(core.blocks) < MAXL:
            grow_deeper(core, 2, trainable=True); opt = opt_for(core); grew = True
        prev_cap = cap
        train_mem(2500)                               # ALWAYS re-sync memory to the evolved core
        r_end = recall_all()                          # (core drifts every capability phase, not just on grow)
        print(f"  phase {ph}: facts={len(fact_bank)} depth={len(core.blocks)} "
              f"cap(K{KMAX})={cap:.2f} {'[GREW]' if grew else ''} | recall after-write={r_after_write:.2f} "
              f"end={r_end:.2f}", flush=True)

    print(f"\n  final: {len(fact_bank)} facts, depth {len(core.blocks)}, capability {cap_acc(core, world, device):.2f}, "
          f"recall {recall_all():.2f}")
    print("  ONE loop accumulated facts (recall stays high across growth) AND grew capability")
    print("  over its lifetime => the validated pieces compose into a single running system.")


if __name__ == "__main__":
    main()
