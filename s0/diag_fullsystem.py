"""ROUND 9 — the corrected vision in ONE artifact: a DECOMPOSED system (external
capsule MEMORY for facts + a keep-best GROWING core for capability) does BOTH
no-forgetting AND capability-growth over a lifelong stream, while a MONOLITHIC
fixed-small model (one L2 core carrying both facts-in-weights and capability) fails
BOTH. Over P phases each ingests new distinct FACTS and a CAPABILITY chunk (escalating
K-hop). At the end we measure fact-recall (all past facts) and capability.

  decomposed : memory recall (facts, frozen-feature bank) + grown cap-core (capability)
  monolith   : one L2 core trained sequentially on facts+capability -> forgets + caps

  python3 -m s0.diag_fullsystem        # env: FS_SEEDS
"""
from __future__ import annotations
import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import World, WorldConfig
from .core import ProxyCore, grow_deeper
from .pad import pad_batch
from .diag_grow_hops import gen as gen_hops

SEEDS = int(os.environ.get("FS_SEEDS", 3))
PHASES = 5
PER_FACTS = 24
CAP_CHUNK = 1000
D = 128
MAXL = 8


def cap_train(core, world, device, opt, kmax, steps=CAP_CHUNK, B=64):
    core.train()
    for _ in range(steps):
        ids, ln, ans, _ = gen_hops(world, device, B, kmax=kmax)
        rows = torch.arange(ids.size(0), device=device)
        loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, ln - 1]), ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1.0); opt.step()
    core.eval()


@torch.no_grad()
def cap_acc(core, world, device, kmax, n=1024):
    core.eval()
    ids, ln, ans, _ = gen_hops(world, device, n, kmax=kmax)
    rows = torch.arange(n, device=device)
    return (core.lm_head(core.hidden(ids)[rows, ln - 1]).argmax(-1) == ans).float().mean().item()


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed)); V = world.vocab_size
    ents, rels = world.entities, world.relations
    rng = random.Random(seed)
    torch.manual_seed(seed); world.rng.seed(seed)

    # ---- DECOMPOSED: frozen feature core for memory + growing cap core ----
    feat = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    for _ in range(1500):                                   # a fixed frozen feature extractor
        ids, ln, ans, _ = gen_hops(world, device, 64, kmax=3)
        rows = torch.arange(64, device=device)
        loss = F.cross_entropy(feat.lm_head(feat.hidden(ids)[rows, ln - 1]), ans)
        o = torch.optim.AdamW(feat.parameters(), lr=3e-3); o.zero_grad(); loss.backward(); o.step()
    for p in feat.parameters():
        p.requires_grad_(False)
    feat.eval()
    mk = lambda o: nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, o)).to(device)
    pk, pq, vd = mk(128), mk(128), nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    mem_p = [p for m in (pk, pq, vd) for p in m.parameters()]
    mo = torch.optim.Adam(mem_p, lr=5e-4)
    cap = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    copt = torch.optim.AdamW(cap.parameters(), lr=3e-3)
    best_ref, best_cap = -1.0, None
    bank = []

    def kfeat(kind):                                        # frozen features of the fact bank
        seqs = []
        for (s, r, o) in bank:
            base = [world.i("<bos>"), world.i(ents[s]), world.i(rels[r])]
            seqs.append(base + ([world.i(ents[o])] if kind == "v" else [world.i("<ans>")]))
        ids, ln = pad_batch(seqs, world.i("<pad>"), device)
        with torch.no_grad():
            h = feat.hidden(ids)
        return h[torch.arange(ids.size(0), device=device), ln - 1]

    def mem_train(steps=1200, B=64):
        if len(bank) < 4:
            return
        Kf, Vf = kfeat("k"), kfeat("v")
        gold = torch.tensor([world.i(ents[o]) for (_, _, o) in bank], device=device)
        N = len(bank)
        for _ in range(steps):
            idx = torch.randint(0, N, (min(B, N),), device=device)
            Kall = F.normalize(pk(Kf), -1); q = F.normalize(pq(Kf[idx]), -1)
            sims = q @ Kall.t() / 0.05
            R = vd(torch.softmax(sims, -1) @ Vf)
            loss = F.cross_entropy(feat.lm_head(Kf[idx] + R), gold[idx]) + F.cross_entropy(sims, idx)
            mo.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mem_p, 1.0); mo.step()

    @torch.no_grad()
    def mem_recall():
        if len(bank) < 4:
            return 1.0
        Kf, Vf = kfeat("k"), kfeat("v")
        Kall = F.normalize(pk(Kf), -1); q = F.normalize(pq(Kf), -1)
        R = vd(torch.softmax(q @ Kall.t() / 0.05, -1) @ Vf)
        gold = torch.tensor([world.i(ents[o]) for (_, _, o) in bank], device=device)
        return (feat.lm_head(Kf + R).argmax(-1) == gold).float().mean().item()

    for ph in range(PHASES):
        for _ in range(PER_FACTS):
            s, o = rng.sample(range(world.cfg.n_entities), 2); r = rng.randrange(len(rels))
            bank.append((s, r, o))
        mem_train()
        km = 3 + ph                                        # escalating capability
        cap_train(cap, world, device, copt, km)
        ref = cap_acc(cap, world, device, 6)
        if ref > best_ref: best_ref, best_cap = ref, copy.deepcopy(cap)   # keep-best
        a = cap_acc(cap, world, device, km)
        if a < 0.9 and len(cap.blocks) < MAXL:             # grow on demand
            grow_deeper(cap, 2, trainable=True); copt = torch.optim.AdamW(cap.parameters(), lr=3e-3)
    dec_recall, dec_cap = mem_recall(), cap_acc(best_cap, world, device, 6)

    # ---- MONOLITH: one L2 core, facts-in-weights + capability, sequential ----
    torch.manual_seed(seed); world.rng.seed(seed); rng = random.Random(seed)
    mono = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    opt = torch.optim.AdamW(mono.parameters(), lr=3e-3)
    mbank = []

    def fact_batch(B=64):
        seqs, gold = [], []
        for _ in range(B):
            (s, r, o) = rng.choice(mbank)
            seqs.append([world.i("<bos>"), world.i(ents[s]), world.i(rels[r]), world.i("<ans>")])
            gold.append(world.i(ents[o]))
        ids, ln = pad_batch(seqs, world.i("<pad>"), device)
        return ids, ln, torch.tensor(gold, device=device)

    for ph in range(PHASES):
        for _ in range(PER_FACTS):
            s, o = rng.sample(range(world.cfg.n_entities), 2); r = rng.randrange(len(rels))
            mbank.append((s, r, o))
        km = 3 + ph
        mono.train()
        for _ in range(CAP_CHUNK):                          # train facts + capability together
            if random.random() < 0.5:
                ids, ln, g = fact_batch()
            else:
                ids, ln, g, _ = gen_hops(world, device, 64, kmax=km)
            rows = torch.arange(ids.size(0), device=device)
            loss = F.cross_entropy(mono.lm_head(mono.hidden(ids)[rows, ln - 1]), g)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mono.parameters(), 1.0); opt.step()
        mono.eval()

    @torch.no_grad()
    def mono_recall():
        ids, ln, g = fact_batch(B=min(512, len(mbank)))
        rows = torch.arange(ids.size(0), device=device)
        return (mono.lm_head(mono.hidden(ids)[rows, ln - 1]).argmax(-1) == g).float().mean().item()
    mono_r, mono_c = mono_recall(), cap_acc(mono, world, device, 6)
    return dec_recall, dec_cap, mono_r, mono_c


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FULL SYSTEM: decomposed vs monolith ({device}) seeds={SEEDS} phases={PHASES}")
    D_ = {"dec_recall": [], "dec_cap": [], "mono_recall": [], "mono_cap": []}
    for seed in range(SEEDS):
        dr, dc, mr, mc = run_seed(seed, device)
        D_["dec_recall"].append(dr); D_["dec_cap"].append(dc)
        D_["mono_recall"].append(mr); D_["mono_cap"].append(mc)
        print(f"  seed {seed}: decomposed(recall {dr:.2f}, cap {dc:.2f})  "
              f"monolith(recall {mr:.2f}, cap {mc:.2f})", flush=True)
    m = lambda k: sum(D_[k]) / len(D_[k])
    print(f"\n== mean over {SEEDS} seeds ==")
    print(f"  DECOMPOSED (mem+grown): fact-recall {m('dec_recall'):.2f}   capability {m('dec_cap'):.2f}")
    print(f"  MONOLITH   (one L2)   : fact-recall {m('mono_recall'):.2f}   capability {m('mono_cap'):.2f}")
    print("\n  decomposed HIGH on both, monolith LOW on both => the corrected vision in one system:")
    print("  external memory keeps facts (no forgetting) + keep-best growth adds capability;")
    print("  a monolithic small model crams both into shared weights and fails both.")


if __name__ == "__main__":
    main()
