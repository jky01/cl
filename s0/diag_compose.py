"""THE missing "grow AND get smarter" mechanism — §27.12 distill-into-core enables
COMPOSITION. Retrieval memory is single-shot: it can recall fact A and fact B
separately but cannot CHAIN them. If accumulated facts are DISTILLED into (grown)
core weights, the core can compose them internally — answering 2-hop queries that
span facts learned in DIFFERENT sessions. That is a genuine new capability
("smarter"), not just retention.

Setup: r1 maps X->Y (learned in session 1), r2 maps Y->Z (learned in session 2).
Composition query: "the r2 of the r1 of X" -> Z, requiring both facts. Train on all
atomic facts + a SUBSET of compositions; evaluate HELD-OUT compositions (both facts
known, never composed together in training).

  A  memory (single-shot retrieval over atomic facts)  -> 1-hop recall high, 2-hop ~0
  B  distill-into-grown-core (facts+some compositions trained into weights,
     after a function-preserving grow)                  -> held-out 2-hop should work
  C  fixed-small sequential fine-tune (sess1 then sess2 then compositions)
     -> forgets session 1 -> compositions crippled

  python3 -m s0.diag_compose      # env: CP_SEEDS
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

SEEDS = int(os.environ.get("CP_SEEDS", 2))
NX = 60                        # X entities; r1: X->Y, r2: Y->Z
STEPS_FACT = 3000
STEPS_COMP = int(os.environ.get("CP_STEPS_COMP", 4000))
D = 128
HOLD = float(os.environ.get("CP_HOLD", 0.3))                     # fraction of compositions held out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"COMPOSITION via distill-into-core ({device}) seeds={SEEDS} NX={NX}")
    for seed in range(SEEDS):
        run_seed(seed, device)


def run_seed(seed, device):
    world = World(WorldConfig(n_entities=200, n_objects=200, seed=seed))
    V = world.vocab_size
    rng = random.Random(seed)
    ents = world.entities
    r1, r2 = world.i(world.relations[0]), world.i(world.relations[1])
    bos, the, of, isk, dot, ans = (world.i(t) for t in ("<bos>", "the", "of", "is", ".", "<ans>"))

    # ground truth: bijections X->Y->Z over disjoint entity pools
    ids_all = rng.sample(range(world.cfg.n_entities), 3 * NX)
    X, Y, Z = ids_all[:NX], ids_all[NX:2 * NX], ids_all[2 * NX:]
    perm1 = rng.sample(range(NX), NX); perm2 = rng.sample(range(NX), NX)
    f1 = {X[i]: Y[perm1[i]] for i in range(NX)}                   # r1
    f2 = {Y[i]: Z[perm2[i]] for i in range(NX)}                   # r2

    def fact_seq(rel, a, b):                                       # "the r of A is B ."
        return [bos, the, rel, of, world.i(ents[a]), isk, world.i(ents[b]), dot]

    def fact_q(rel, a):                                            # "the r of A is <ans>"
        return [bos, the, rel, of, world.i(ents[a]), isk, ans]

    def comp_q(x):                                                 # "the r2 of the r1 of X is <ans>"
        return [bos, the, r2, of, the, r1, of, world.i(ents[x]), isk, ans]

    comp_pairs = list(range(NX)); rng.shuffle(comp_pairs)
    n_hold = int(NX * HOLD)
    held, shown = comp_pairs[:n_hold], comp_pairs[n_hold:]

    def batch(kind, idxs, B=64):
        seqs, gold = [], []
        for _ in range(B):
            i = rng.choice(idxs)
            x = X[i]
            if kind == "f1":
                seqs.append(fact_q(r1, x)); gold.append(world.i(ents[f1[x]]))
            elif kind == "f2":
                y = Y[i]; seqs.append(fact_q(r2, y)); gold.append(world.i(ents[f2[y]]))
            else:
                seqs.append(comp_q(x)); gold.append(world.i(ents[f2[f1[x]]]))
        ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
        return ids, lengths, torch.tensor(gold, device=device)

    def train_mix(core, mix, steps, lr=3e-3):
        core.train()
        params = [p for p in core.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
        for _ in range(steps):
            kind, idxs = mix[rng.randrange(len(mix))]
            ids, lengths, g = batch(kind, idxs)
            rows = torch.arange(ids.size(0), device=device)
            loss = F.cross_entropy(core.lm_head(core.hidden(ids)[rows, lengths - 1]), g)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        core.eval()

    @torch.no_grad()
    def evl(core, kind, idxs, n=512):
        ids, lengths, g = batch(kind, idxs, B=n)
        rows = torch.arange(n, device=device)
        return (core.lm_head(core.hidden(ids)[rows, lengths - 1]).argmax(-1) == g).float().mean().item()

    all_i = list(range(NX))

    # ---- B: distill-into-GROWN-core: sess1 facts -> grow -> sess2 facts + shown compositions ----
    torch.manual_seed(seed)
    core = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    train_mix(core, [("f1", all_i)], STEPS_FACT)                  # session 1: r1 facts into weights
    grow_deeper(core, 2, trainable=True)                          # grow (small -> larger)
    train_mix(core, [("f1", all_i), ("f2", all_i), ("comp", shown)], STEPS_COMP)  # consolidate + compose
    B_f1, B_f2 = evl(core, "f1", all_i), evl(core, "f2", all_i)
    B_comp_shown, B_comp_held = evl(core, "comp", shown), evl(core, "comp", held)

    # ---- C: fixed-small SEQUENTIAL (no replay of f1): forgets -> compositions crippled ----
    torch.manual_seed(seed)
    small = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    train_mix(small, [("f1", all_i)], STEPS_FACT)
    train_mix(small, [("f2", all_i), ("comp", shown)], STEPS_COMP)   # sess2 WITHOUT f1 replay
    C_f1 = evl(small, "f1", all_i)
    C_comp_held = evl(small, "comp", held)

    # ---- A: memory single-shot retrieval (atomic facts as bank; frozen feature core) ----
    fro = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    fro.load_state_dict(small.state_dict(), strict=False) if False else None
    torch.manual_seed(seed)
    fro = ProxyCore(V, d_model=D, n_layers=2, n_heads=4, max_len=72).to(device)
    train_mix(fro, [("f1", all_i), ("f2", all_i)], 1500)          # feature extractor only
    for p in fro.parameters():
        p.requires_grad_(False)
    fro.eval()

    def feats(seqs):
        ids, lengths = pad_batch(seqs, world.i("<pad>"), device)
        with torch.no_grad():
            h = fro.hidden(ids)
        rows = torch.arange(ids.size(0), device=device)
        return h[rows, lengths - 1]

    bank_seqs = [fact_seq(r1, x, f1[x]) for x in X] + [fact_seq(r2, y, f2[y]) for y in Y]
    bank_keys = [fact_q(r1, x) for x in X] + [fact_q(r2, y) for y in Y]
    bank_gold = torch.tensor([world.i(ents[f1[x]]) for x in X] +
                             [world.i(ents[f2[y]]) for y in Y], device=device)
    mk = lambda o: nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, o)).to(device)
    pk, pq = mk(128), mk(128)
    vd = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    prm = [p for m in (pk, pq, vd) for p in m.parameters()]
    opt = torch.optim.Adam(prm, lr=5e-4)
    Kf = feats(bank_keys); Vf = feats(bank_seqs)
    N = len(bank_keys)
    for _ in range(2500):                                          # train retrieval+inject on ATOMIC queries
        idx = torch.randint(0, N, (64,), device=device)
        K_ = F.normalize(pk(Kf), -1); q = F.normalize(pq(Kf[idx]), -1)
        sims = q @ K_.t() / 0.05
        R = vd(torch.softmax(sims, -1) @ Vf)
        logits = fro.lm_head(Kf[idx] + R)
        loss = F.cross_entropy(logits, bank_gold[idx]) + F.cross_entropy(sims, idx)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(prm, 1.0); opt.step()

    @torch.no_grad()
    def mem_eval(qseqs, gold):
        qf = feats(qseqs)
        K_ = F.normalize(pk(Kf), -1); q = F.normalize(pq(qf), -1)
        R = vd(torch.softmax(q @ K_.t() / 0.05, -1) @ Vf)
        pred = fro.lm_head(qf + R).argmax(-1)
        return (pred == torch.tensor(gold, device=device)).float().mean().item()

    A_1hop = mem_eval([fact_q(r1, x) for x in X], [world.i(ents[f1[x]]) for x in X])
    A_comp = mem_eval([comp_q(x) for x in [X[i] for i in held]],
                      [world.i(ents[f2[f1[X[i]]]]) for i in held])

    print(f"  seed {seed}:")
    print(f"    A memory(single-shot): 1-hop {A_1hop:.2f}  held-out 2-hop {A_comp:.2f}")
    print(f"    B distill+grown core : f1 {B_f1:.2f} f2 {B_f2:.2f} comp(shown) {B_comp_shown:.2f} "
          f"comp(HELD-OUT) {B_comp_held:.2f}")
    print(f"    C fixed-small seq    : f1-after {C_f1:.2f}  comp(HELD-OUT) {C_comp_held:.2f}", flush=True)


if __name__ == "__main__":
    main()
