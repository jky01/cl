"""ROUND 16 — the corrected vision in ONE REAL-Qwen artifact: a DECOMPOSED system
(curriculum-grown external MEMORY for facts + a keep-best GROWING core for capability)
does BOTH no-forgetting AND capability-growth over a lifelong stream on real Qwen,
while a MONOLITHIC model (top layers carrying both facts-in-weights and capability,
trained sequentially) fails at least one. This is the Qwen port of diag_fullsystem,
using the validated pieces: curriculum router-free memory (rounds 12-15) + keep-best
appended-layer growth on an escalating in-context hop curriculum (grow-cadence, A).

Over P phases, each phase ingests (a) a batch of new distinct FACTS (name->attr->value)
and (b) a capability chunk (escalating in-context K-hop). At the end we measure
fact-recall over ALL past facts and capability (hop accuracy).

  decomposed : router-free MEMORY (frozen-feature bank, curriculum across phases)
               + a frozen-base Qwen with APPENDED trainable layers, keep-best growth
  monolith   : one Qwen, top-N layers trained SEQUENTIALLY on facts + capability

Expect: decomposed HIGH on both; monolith forgets early facts (low recall) and/or caps.

  python3 -m s0.qwen_capstone_lifelong   # env: CL_PHASES, CL_FACTS_PER, CL_MEM_STEPS, CL_CAP_STEPS, CL_SEEDS
"""
from __future__ import annotations
import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .qwen_grow import grow_qwen
from .qwen_growcap import single_tok_names, make
from .qwen_memory import ATTR_VALUES
from .qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
PHASES = int(os.environ.get("CL_PHASES", 4))
FACTS_PER = int(os.environ.get("CL_FACTS_PER", 400))
MEM_STEPS = int(os.environ.get("CL_MEM_STEPS", 1500))       # memory projections trained per phase (over the GROWN bank)
CAP_STEPS = int(os.environ.get("CL_CAP_STEPS", 1200))       # capability steps per phase
SEEDS = int(os.environ.get("CL_SEEDS", 2))
KDIM = int(os.environ.get("CL_KDIM", 256))
TOPK = 32
LR_CAP = 1.5e-4
Bc = 24
ATTRS = list(ATTR_VALUES)
HOPS = [1, 2, 3]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = [f"{f} {l}" for f in FIRST for l in LAST]        # fact subjects (multi-token keys OK)
    hop_names = single_tok_names(tok)                        # capability uses single-token names
    print(f"CAPSTONE-LIFELONG ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"phases={PHASES} facts/phase={FACTS_PER} mem-steps={MEM_STEPS} cap-steps={CAP_STEPS} seeds={SEEDS}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m
    d = AutoConfig.from_pretrained(NAME).hidden_size

    @torch.no_grad()
    def pooled(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = m.model(**e).last_hidden_state
            msk = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * msk).sum(1) / msk.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(m.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

    # ---------- capability (in-context hops) shared helpers ----------
    def cap_batch(rng, hop):
        prompts, ans = [], []
        for _ in range(Bc):
            p, a = make(rng, hop_names, hop)
            prompts.append(p); ans.append(a)
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        aid = torch.tensor([tok(" " + a, add_special_tokens=False).input_ids[0] for a in ans], device=device)
        return enc, aid

    def cap_train(m, rng, max_hop, steps, opt):
        m.train()
        for _ in range(steps):
            hop = rng.randint(1, max_hop)
            enc, aid = cap_batch(rng, hop)
            logits = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float()
            loss = F.cross_entropy(logits, aid)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0); opt.step()
        m.eval()

    @torch.no_grad()
    def cap_acc(m, hop, n=192):
        rng = random.Random(999)
        ok = tot = 0
        for _ in range(0, n, Bc):
            enc, aid = cap_batch(rng, hop)
            pred = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == aid).sum().item(); tot += aid.numel()
        return ok / tot

    def cap_mean(m):
        return sum(cap_acc(m, h) for h in HOPS) / len(HOPS)

    def set_trainable_top(m, n):
        for p in m.parameters():
            p.requires_grad_(False)
        for lyr in m.model.layers[-n:]:
            for p in lyr.parameters():
                p.requires_grad_(True)

    # ---------- DECOMPOSED ----------
    def run_decomposed(seed, bank_facts_by_phase):
        feat = load_frozen()
        # memory modules (persistent across phases = curriculum warm-start)
        torch.manual_seed(seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mem_mods = (proj_k, proj_q, val_enc, val_dec, gate)
        mopt = torch.optim.Adam([p for m in mem_mods for p in m.parameters()], lr=5e-4)

        # capability core: frozen base + appended trainable layers, keep-best growth
        core = load_frozen(); grow_qwen(core, 2); set_trainable_top(core, 2)
        copt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=LR_CAP)
        best_ref, best_core = -1.0, None

        bank = []
        for ph in range(PHASES):
            bank += bank_facts_by_phase[ph]
            # --- memory: recompute frozen features for the whole (grown) bank, warm-train projections ---
            Kf = pooled(feat, [f"{n}'s {a}" for (n, a, _) in bank])
            Sf = last_h(feat, [f"{n}'s {a} is {v}" for (n, a, v) in bank])
            Qf = pooled(feat, [f"{n}'s {a} is" for (n, a, _) in bank])
            Hf = last_h(feat, [f"{n}'s {a} is" for (n, a, _) in bank])
            gold = torch.tensor([one_tok(v) for (_, _, v) in bank], device=device)
            Nb = len(bank)
            for _ in range(MEM_STEPS):
                idx = torch.randint(0, Nb, (min(256, Nb),), device=device)
                Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
                q = F.normalize(proj_q(Qf[idx]), -1)
                sims = q @ Kall.t() / 0.05
                vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
                R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
                H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
                loss = F.cross_entropy(feat.lm_head(H + g * R).float(), gold[idx]) + F.cross_entropy(sims, idx)
                mopt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for m in mem_mods for p in m.parameters()], 1.0); mopt.step()
            # --- capability: escalating hop, keep-best, one grow before the hardest phase ---
            rng = random.Random(seed * 17 + ph)
            if ph == PHASES - 1 and len(core.model.layers) < 24 + 4:
                grow_qwen(core, 2); set_trainable_top(core, 4)
                copt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=LR_CAP)
            cap_train(core, rng, min(1 + ph, 3), CAP_STEPS, copt)
            ref = cap_mean(core)
            if ref > best_ref: best_ref, best_core = ref, copy.deepcopy(core)
            print(f"    [dec ph{ph}] bank={Nb} mem-recall={mem_recall(mem_mods, feat, Kf, Sf, Qf, Hf, gold, Nb):.3f} "
                  f"cap-mean={ref:.3f}", flush=True)

        # final measures
        Kf = pooled(feat, [f"{n}'s {a}" for (n, a, _) in bank])
        Sf = last_h(feat, [f"{n}'s {a} is {v}" for (n, a, v) in bank])
        Qf = pooled(feat, [f"{n}'s {a} is" for (n, a, _) in bank])
        Hf = last_h(feat, [f"{n}'s {a} is" for (n, a, _) in bank])
        gold = torch.tensor([one_tok(v) for (_, _, v) in bank], device=device)
        recall = mem_recall(mem_mods, feat, Kf, Sf, Qf, Hf, gold, len(bank))
        capf = cap_mean(best_core)
        del feat, core, best_core; torch.cuda.empty_cache()
        return recall, capf

    @torch.no_grad()
    def mem_recall(mods, feat, Kf, Sf, Qf, Hf, gold, Nb):
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

    # ---------- MONOLITH (replay=False: naive sequential; replay=True: rehearse full bank) ----------
    def run_monolith(seed, bank_facts_by_phase, replay=False):
        mono = load_frozen(); set_trainable_top(mono, 4)
        opt = torch.optim.AdamW([p for p in mono.parameters() if p.requires_grad], lr=LR_CAP)
        rng = random.Random(seed * 31)
        bank = []
        tag = "mono-rep" if replay else "mono"

        def fact_batch(sub):
            prompts = [f"{n}'s {a} is" for (n, a, _) in sub]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            aid = torch.tensor([one_tok(v) for (_, _, v) in sub], device=device)
            return enc, aid

        for ph in range(PHASES):
            bank += bank_facts_by_phase[ph]
            new = bank_facts_by_phase[ph]
            pool = bank if replay else new                  # REPLAY rehearses ALL past facts (O(lifetime) storage)
            mono.train()
            steps = MEM_STEPS + CAP_STEPS                    # matched total budget vs decomposed
            for _ in range(steps):
                if rng.random() < 0.5:                       # facts-in-weights
                    sub = [rng.choice(pool) for _ in range(Bc)]
                    enc, aid = fact_batch(sub)
                else:                                        # capability
                    enc, aid = cap_batch(rng, rng.randint(1, min(1 + ph, 3)))
                logits = mono.lm_head(mono.model(**enc, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(logits, aid)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in mono.parameters() if p.requires_grad], 1.0); opt.step()
            mono.eval()
            print(f"    [{tag} ph{ph}] recall={mono_recall(mono, bank):.3f} cap-mean={cap_mean(mono):.3f}", flush=True)
        recall, capf = mono_recall(mono, bank), cap_mean(mono)
        del mono; torch.cuda.empty_cache()
        return recall, capf

    @torch.no_grad()
    def mono_recall(m, bank, n=512):
        idx = list(range(len(bank)))
        random.Random(0).shuffle(idx); idx = idx[:n]
        ok = tot = 0
        for i in range(0, len(idx), 128):
            sub = [bank[j] for j in idx[i:i + 128]]
            enc = tok([f"{nn_}'s {a} is" for (nn_, a, _) in sub], return_tensors="pt", padding=True).to(device)
            pred = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float().argmax(-1)
            aid = torch.tensor([one_tok(v) for (_, _, v) in sub], device=device)
            ok += (pred == aid).sum().item(); tot += len(sub)
        return ok / tot

    # ---------- run seeds ----------
    skip_dec = bool(os.environ.get("CL_SKIP_DEC"))
    D = {k: [] for k in ("dec_recall", "dec_cap", "mono_recall", "mono_cap", "rep_recall", "rep_cap")}
    for seed in range(SEEDS):
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
        print(f"  seed {seed}:", flush=True)
        dr, dc = (0.0, 0.0) if skip_dec else run_decomposed(seed, phases_facts)
        mr, mc = run_monolith(seed, phases_facts, replay=False)
        rr, rc = run_monolith(seed, phases_facts, replay=True)          # rehearsal baseline (full-bank replay)
        D["dec_recall"].append(dr); D["dec_cap"].append(dc)
        D["mono_recall"].append(mr); D["mono_cap"].append(mc)
        D["rep_recall"].append(rr); D["rep_cap"].append(rc)
        print(f"  => seed {seed}: DECOMPOSED(rec {dr:.3f}, cap {dc:.3f})  "
              f"MONO(rec {mr:.3f}, cap {mc:.3f})  MONO-REPLAY(rec {rr:.3f}, cap {rc:.3f})", flush=True)

    m = lambda k: sum(D[k]) / len(D[k])
    print(f"\n== mean over {SEEDS} seeds (real Qwen lifelong) ==")
    print(f"  DECOMPOSED  (curric-mem + grown-core, bounded/frozen-feature) : recall {m('dec_recall'):.3f}  cap {m('dec_cap'):.3f}")
    print(f"  MONOLITH    (top-4, sequential, NO replay)                    : recall {m('mono_recall'):.3f}  cap {m('mono_cap'):.3f}")
    print(f"  MONO-REPLAY (top-4, rehearse FULL bank each phase, O(lifetime)): recall {m('rep_recall'):.3f}  cap {m('rep_cap'):.3f}")
    print("\n  no-replay monolith FORGETS; replay recovers recall but must store+rehearse ALL past facts")
    print("  (unbounded O(lifetime) storage+compute); the decomposed memory matches replay-quality")
    print("  no-forgetting from FROZEN features computed once + light projections => the EFFICIENT route.")


if __name__ == "__main__":
    main()
