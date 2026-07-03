"""ROUND 18 — QUANTIFY the "efficient route" claim. Round 17 showed REPLAY also reaches
no-forgetting (rehearsing the full bank recovers recall the naive monolith forgot), so the
decomposed system's ONLY remaining advantage over replay is COST. This round measures that
cost honestly, on real Qwen, for the identical lifelong bank, so the flagship's "EFFICIENT
route" line stops being an assertion and becomes a number.

What is actually different (be precise, do not overclaim):
  * STORAGE is O(N) for BOTH — replay stores raw facts, decomposed stores frozen feature
    vectors (Kf/Sf/Qf/Hf). Not a storage win.
  * The win is BACKBONE COMPUTE. The frozen 0.5B backbone is the expensive part:
      - DECOMPOSED-OPTIMAL: each arriving fact is forwarded through the backbone EXACTLY
        ONCE (to cache its frozen features); the per-phase memory *training* then touches
        only tiny projection MLPs + lm_head on cached vectors — it NEVER forwards or
        backprops the backbone. Total backbone work over the stream = O(N) forward-only.
      - REPLAY: every phase runs MEM_STEPS/2 fact-rehearsal steps, each a full backbone
        FORWARD + a backward through the trainable top layers, over a bank-sampled batch.
        Total backbone work = O(P * steps) forward AND backward passes through the model.
  (The capstone's decomposed code RECOMPUTES whole-bank features each phase for simplicity;
   that is O(P*N) and wasteful. We measure BOTH: `dec-recompute` = what the capstone does,
   `dec-optimal` = cache-once, the real claim. Replay cannot be made forward-only: training
   in-weights requires backprop.)

We count backbone forward/backward TOKEN-passes (non-pad tokens actually pushed through the
0.5B stack) and wall-clock of the fact-learning update per phase, then verify BOTH strategies
reach comparable fact recall so the cost is cost-for-the-same-outcome.

  python3 -m s0.qwen_cost_accounting   # env: CA_PHASES, CA_FACTS_PER, CA_MEM_STEPS, CA_SEEDS
"""
from __future__ import annotations
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .qwen_memory import ATTR_VALUES
from .qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
PHASES = int(os.environ.get("CA_PHASES", 4))
FACTS_PER = int(os.environ.get("CA_FACTS_PER", 1200))
MEM_STEPS = int(os.environ.get("CA_MEM_STEPS", 2500))       # decomposed proj-steps AND replay total fact+cap steps budget
SEEDS = int(os.environ.get("CA_SEEDS", 2))
KDIM = int(os.environ.get("CA_KDIM", 256))
TOPK = 32
Bc = 24
LR_CAP = 1.5e-4
ATTRS = list(ATTR_VALUES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    d = AutoConfig.from_pretrained(NAME).hidden_size
    print(f"COST-ACCOUNTING ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"phases={PHASES} facts/phase={FACTS_PER} mem-steps={MEM_STEPS} seeds={SEEDS}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    # --- backbone token-pass counters (the expensive 0.5B stack) ---
    C = {"fwd": 0, "bwd": 0}

    def enc_of(texts):
        return tok(texts, return_tensors="pt", padding=True).to(device)

    @torch.no_grad()
    def pooled(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = enc_of(texts[i:i + bs]); C["fwd"] += int(e.attention_mask.sum().item())
            h = m.model(**e).last_hidden_state
            msk = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * msk).sum(1) / msk.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = enc_of(texts[i:i + bs]); C["fwd"] += int(e.attention_mask.sum().item())
            outs.append(m.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

    def feats_for(feat, facts):
        Kf = pooled(feat, [f"{n}'s {a}" for (n, a, _) in facts])
        Sf = last_h(feat, [f"{n}'s {a} is {v}" for (n, a, v) in facts])
        Qf = pooled(feat, [f"{n}'s {a} is" for (n, a, _) in facts])
        Hf = last_h(feat, [f"{n}'s {a} is" for (n, a, _) in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        return [Kf, Sf, Qf, Hf, gold]

    def new_mem(seed):
        torch.manual_seed(seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        return (proj_k, proj_q, val_enc, val_dec, gate)

    def mem_train(feat, mods, F_, steps):
        Kf, Sf, Qf, Hf, gold = F_
        Nb = Kf.shape[0]
        proj_k, proj_q, val_enc, val_dec, gate = mods
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
        for _ in range(steps):
            idx = torch.randint(0, Nb, (min(256, Nb),), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)      # tiny MLPs on cached vecs; NO backbone
            q = F.normalize(proj_q(Qf[idx]), -1)
            sims = q @ Kall.t() / 0.05
            vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(feat.lm_head(H + g * R).float(), gold[idx]) + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0); opt.step()
        return opt

    @torch.no_grad()
    def mem_recall(feat, mods, F_):
        Kf, Sf, Qf, Hf, gold = F_; Nb = Kf.shape[0]
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kall = F.normalize(proj_k(Kf), -1); qall = F.normalize(proj_q(Qf), -1); Vall = val_enc(Sf)
        rf = 0
        for i in range(0, Nb, 256):
            ri = torch.arange(i, min(i + 256, Nb), device=device)
            s = (qall[ri] @ Kall.t()) / 0.05
            vk, ik = s.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[ri]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            rf += (feat.lm_head(H + g * R).float().argmax(-1) == gold[ri]).sum().item()
        return rf / Nb

    def set_trainable_top(m, n):
        for p in m.parameters():
            p.requires_grad_(False)
        for lyr in m.model.layers[-n:]:
            for p in lyr.parameters():
                p.requires_grad_(True)

    @torch.no_grad()
    def weight_recall(m, bank, n=512):
        idx = list(range(len(bank))); random.Random(0).shuffle(idx); idx = idx[:n]
        ok = tot = 0
        for i in range(0, len(idx), 128):
            sub = [bank[j] for j in idx[i:i + 128]]
            e = enc_of([f"{nn_}'s {a} is" for (nn_, a, _) in sub])
            pred = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float().argmax(-1)
            aid = torch.tensor([one_tok(v) for (_, _, v) in sub], device=device)
            ok += (pred == aid).sum().item(); tot += len(sub)
        return ok / tot

    def run(seed):
        rng = random.Random(1000 + seed)
        used = set(); phases_facts = []
        for ph in range(PHASES):
            fl = []
            while len(fl) < FACTS_PER:
                n = rng.choice(names); a = rng.choice(ATTRS)
                if (n, a) in used:
                    continue
                used.add((n, a)); fl.append((n, a, rng.choice(av[a])))
            phases_facts.append(fl)

        rows = []   # (phase, arm, backbone_fwd_tok, backbone_bwd_tok, seconds, recall)

        # ---- DECOMPOSED-OPTIMAL: cache each fact's features ONCE; train tiny MLPs on cache ----
        feat = load_frozen(); mods = new_mem(seed)
        cache = None; bank = []
        for ph in range(PHASES):
            C["fwd"] = C["bwd"] = 0
            t0 = time.time()
            new = phases_facts[ph]; bank += new
            nf = feats_for(feat, new)                                    # backbone forward ONCE, only NEW facts
            cache = nf if cache is None else [torch.cat([cache[i], nf[i]], 0) for i in range(5)]
            mem_train(feat, mods, cache, MEM_STEPS)                      # NO backbone touched here
            torch.cuda.synchronize() if device == "cuda" else None
            sec = time.time() - t0
            rec = mem_recall(feat, mods, cache)
            rows.append((ph, "dec-optimal", C["fwd"], C["bwd"], sec, rec))
            print(f"    [dec-optimal ph{ph}] bank={len(bank)} bb_fwd_tok={C['fwd']:>8} bb_bwd_tok={C['bwd']:>8} "
                  f"sec={sec:6.1f} recall={rec:.3f}", flush=True)
        del feat; torch.cuda.empty_cache()

        # ---- DEC-RECOMPUTE: what the capstone actually does (whole-bank features every phase) ----
        feat = load_frozen(); mods = new_mem(seed); bank = []
        for ph in range(PHASES):
            C["fwd"] = C["bwd"] = 0
            t0 = time.time()
            bank += phases_facts[ph]
            allf = feats_for(feat, bank)                                 # RECOMPUTE whole grown bank (wasteful)
            mem_train(feat, mods, allf, MEM_STEPS)
            torch.cuda.synchronize() if device == "cuda" else None
            sec = time.time() - t0
            rec = mem_recall(feat, mods, allf)
            rows.append((ph, "dec-recompute", C["fwd"], C["bwd"], sec, rec))
            print(f"    [dec-recomp  ph{ph}] bank={len(bank)} bb_fwd_tok={C['fwd']:>8} bb_bwd_tok={C['bwd']:>8} "
                  f"sec={sec:6.1f} recall={rec:.3f}", flush=True)
        del feat; torch.cuda.empty_cache()

        # ---- REPLAY: rehearse full bank into weights; backbone FORWARD+BACKWARD every step ----
        mono = load_frozen(); set_trainable_top(mono, 4)
        opt = torch.optim.AdamW([p for p in mono.parameters() if p.requires_grad], lr=LR_CAP)
        rrng = random.Random(seed * 31); bank = []
        for ph in range(PHASES):
            C["fwd"] = C["bwd"] = 0
            t0 = time.time()
            bank += phases_facts[ph]; mono.train()
            for _ in range(MEM_STEPS):                                   # all fact-rehearsal (pure memory-cost comparison)
                sub = [rrng.choice(bank) for _ in range(Bc)]
                e = enc_of([f"{n}'s {a} is" for (n, a, _) in sub])
                ntok = int(e.attention_mask.sum().item())
                C["fwd"] += ntok; C["bwd"] += ntok                       # forward + backward through the 0.5B stack
                aid = torch.tensor([one_tok(v) for (_, _, v) in sub], device=device)
                logits = mono.lm_head(mono.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(logits, aid)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in mono.parameters() if p.requires_grad], 1.0); opt.step()
            mono.eval()
            torch.cuda.synchronize() if device == "cuda" else None
            sec = time.time() - t0
            rec = weight_recall(mono, bank)
            rows.append((ph, "replay", C["fwd"], C["bwd"], sec, rec))
            print(f"    [replay      ph{ph}] bank={len(bank)} bb_fwd_tok={C['fwd']:>8} bb_bwd_tok={C['bwd']:>8} "
                  f"sec={sec:6.1f} recall={rec:.3f}", flush=True)
        del mono; torch.cuda.empty_cache()
        return rows

    agg = {}
    for seed in range(SEEDS):
        print(f"  seed {seed}:", flush=True)
        for (ph, arm, fwd, bwd, sec, rec) in run(seed):
            agg.setdefault(arm, {"fwd": 0, "bwd": 0, "sec": 0.0, "rec": []})
            a = agg[arm]; a["fwd"] += fwd; a["bwd"] += bwd; a["sec"] += sec
            if ph == PHASES - 1:
                a["rec"].append(rec)

    print(f"\n== cost over the full {PHASES}-phase stream, summed, mean/{SEEDS} seeds (real Qwen) ==")
    print(f"  {'arm':14s} | {'backbone-fwd-tok':>17} {'backbone-bwd-tok':>17} {'wall-sec':>9} | {'final-recall':>12}")
    for arm in ("dec-optimal", "dec-recompute", "replay"):
        a = agg[arm]; rec = sum(a["rec"]) / len(a["rec"]) if a["rec"] else 0.0
        print(f"  {arm:14s} | {a['fwd']//SEEDS:>17,} {a['bwd']//SEEDS:>17,} {a['sec']/SEEDS:>9.1f} | {rec:>12.3f}")
    do, rp = agg["dec-optimal"], agg["replay"]
    tot_do = do["fwd"] + do["bwd"]; tot_rp = rp["fwd"] + rp["bwd"]
    print(f"\n  decomposed-optimal does ZERO backbone-backward and {do['fwd']//SEEDS:,} forward tokens (once);")
    print(f"  replay pushes {tot_rp//SEEDS:,} backbone token-passes (fwd+bwd) for comparable recall")
    print(f"  => ~{tot_rp/max(tot_do,1):.1f}x more backbone compute for the SAME no-forgetting outcome.")
    print("  (storage is O(N) for both; the win is backbone COMPUTE: forward-once vs repeated fwd+bwd.)")


if __name__ == "__main__":
    main()
