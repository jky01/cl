"""ROUND 18 — the decomposed external memory COLLAPSES on some seeds. Round 17's clean
2-seed capstone exposed that the decomposed fact-recall is not just high-variance but
SEED-FRAGILE: seed 1 collapsed to ~0.03 (retrieval ≈ random) from phase 0 and never
recovered, while seed 0 reached 0.63 and round-16 got 0.86/0.96. So the round-16 "0.909"
flagship recall was a favorable-seed artifact of a collapse-prone optimizer. Before the
decomposed system's cost advantage (round 19) means anything, the memory must train
RELIABLY. This round isolates the memory (no capability, no growth — same 4-phase 1200/
phase pair-name bank + curriculum warm-start as the capstone) and tests a stabilizer.

Collapse mechanism (from rounds 10-11 + this): full-bank InfoNCE with a SHARP temperature
(0.05) from a cold start can fall into a degenerate basin where proj_q/proj_k map every
query to one region -> sims uninformative -> retrieval stuck near random, unrecoverable.
Some inits avoid it, some don't -> seed fragility.

Arms (SEEDS seeds each; report per-seed final recall + collapse count):
  plain  : current recipe (temp 0.05, lr 5e-4, no guard)                 <- reproduces collapse
  stable : temperature warmup (0.2->0.05) + lr warmup + RESTART-ON-COLLAPSE
           (after a short probe, if retrieval@1 < thresh, reinit modules & retry the stage)

If `stable` removes the collapses (all seeds high) while `plain` collapses on a fraction,
the fix is a robust-recipe guard, not more compute -> the decomposed memory becomes reliable.

  python3 -m s0.qwen_mem_stability   # env: MS_PHASES, MS_FACTS_PER, MS_STEPS, MS_SEEDS
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .qwen_memory import ATTR_VALUES
from .qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
PHASES = int(os.environ.get("MS_PHASES", 4))
FACTS_PER = int(os.environ.get("MS_FACTS_PER", 1200))
STEPS = int(os.environ.get("MS_STEPS", 2500))           # per-phase memory steps (matches capstone MEM_STEPS)
SEEDS = int(os.environ.get("MS_SEEDS", 6))
KDIM = int(os.environ.get("MS_KDIM", 256))
TOPK = 32
COLLAPSE_THR = float(os.environ.get("MS_THR", 0.30))    # end-of-phase0 retrieval@1 below this = collapsed
MAX_RESTART = int(os.environ.get("MS_MAX_RESTART", 4))
ATTRS = list(ATTR_VALUES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    d = AutoConfig.from_pretrained(NAME).hidden_size
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    print(f"MEM-STABILITY ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"phases={PHASES} facts/phase={FACTS_PER} steps/phase={STEPS} seeds={SEEDS} "
          f"probe@end-phase0 thr={COLLAPSE_THR} max_restart={MAX_RESTART}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = lm.model(**e).last_hidden_state
            m = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * m).sum(1) / m.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(lm.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

    def feats(facts):
        Kf = pooled([f"{n}'s {a}" for (n, a, _) in facts])
        Sf = last_h([f"{n}'s {a} is {v}" for (n, a, v) in facts])
        Qf = pooled([f"{n}'s {a} is" for (n, a, _) in facts])
        Hf = last_h([f"{n}'s {a} is" for (n, a, _) in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        return [Kf, Sf, Qf, Hf, gold]

    def new_mem(iseed):
        torch.manual_seed(iseed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        return (proj_k, proj_q, val_enc, val_dec, gate)

    @torch.no_grad()
    def retr_at1(mods, F_, sample=256):
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kf, Sf, Qf, Hf, gold = F_; Nb = Kf.shape[0]
        Kall = F.normalize(proj_k(Kf), -1)
        idx = torch.randint(0, Nb, (min(sample, Nb),), device=device)
        q = F.normalize(proj_q(Qf[idx]), -1)
        return ((q @ Kall.t()).argmax(1) == idx).float().mean().item()

    @torch.no_grad()
    def recall(mods, F_):
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kf, Sf, Qf, Hf, gold = F_; Nb = Kf.shape[0]
        Kall = F.normalize(proj_k(Kf), -1); qall = F.normalize(proj_q(Qf), -1); Vall = val_enc(Sf)
        rf = 0
        for i in range(0, Nb, 256):
            ri = torch.arange(i, min(i + 256, Nb), device=device)
            s = (qall[ri] @ Kall.t()) / 0.05
            vk, ik = s.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[ri]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            rf += (lm.lm_head(H + g * R).float().argmax(-1) == gold[ri]).sum().item()
        return rf / Nb

    def train_phase(mods, opt, F_, steps, stable, global_step):
        """One phase of memory training. Returns updated global_step. `stable` toggles
        temperature+lr warmup. Restart handling is done by the caller via retr probe."""
        Kf, Sf, Qf, Hf, gold = F_; Nb = Kf.shape[0]
        proj_k, proj_q, val_enc, val_dec, gate = mods
        base_lr = 5e-4
        for s in range(steps):
            gstep = global_step + s
            if stable:
                temp = 0.05 + (0.20 - 0.05) * max(0.0, 1 - gstep / 500.0)      # 0.20 -> 0.05 over 500 steps
                lr = base_lr * min(1.0, (gstep + 1) / 200.0)                    # linear warmup 200 steps
                for pg in opt.param_groups:
                    pg["lr"] = lr
            else:
                temp = 0.05
            idx = torch.randint(0, Nb, (min(256, Nb),), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
            q = F.normalize(proj_q(Qf[idx]), -1)
            sims = q @ Kall.t() / temp
            vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(lm.lm_head(H + g * R).float(), gold[idx]) + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0); opt.step()
        return global_step + steps

    def run_arm(seed, phases_facts, stable):
        # curriculum: grow bank across phases, warm-start same modules
        for attempt in range(MAX_RESTART + 1):
            mods = new_mem(seed * 100 + 1 + attempt * 9973)          # new init each restart
            opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
            bank = []; gstep = 0
            # ---- train phase 0 FULLY, THEN probe: healthy retr@1 ~0.9 vs collapsed ~0.0 is a
            #      clean margin, and probing after (not during) warmup avoids false positives ----
            bank += phases_facts[0]; cache = feats(bank)
            gstep = train_phase(mods, opt, cache, STEPS, stable, gstep)
            probe = retr_at1(mods, cache)
            if stable and probe < COLLAPSE_THR and attempt < MAX_RESTART:
                print(f"      [seed {seed} stable] phase0 retr@1={probe:.3f} < {COLLAPSE_THR} "
                      f"-> RESTART {attempt+1}/{MAX_RESTART}", flush=True)
                continue
            for ph in range(1, PHASES):
                bank += phases_facts[ph]; cache = feats(bank)
                gstep = train_phase(mods, opt, cache, STEPS, stable, gstep)
            return recall(mods, cache), (probe < COLLAPSE_THR)
        return recall(mods, cache), (probe < COLLAPSE_THR)

    def make_bank(seed):
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
        return phases_facts

    res = {"plain": [], "stable": []}
    for seed in range(SEEDS):
        pf = make_bank(seed)
        rp, cp = run_arm(seed, pf, stable=False)
        rs, cs = run_arm(seed, pf, stable=True)
        res["plain"].append(rp); res["stable"].append(rs)
        print(f"  seed {seed}: plain recall={rp:.3f}{'  <COLLAPSE>' if rp < COLLAPSE_THR else ''}   "
              f"stable recall={rs:.3f}{'  <COLLAPSE>' if rs < COLLAPSE_THR else ''}", flush=True)

    print(f"\n== over {SEEDS} seeds (real Qwen, isolated decomposed memory) ==")
    for arm in ("plain", "stable"):
        rs = res[arm]; nc = sum(1 for r in rs if r < COLLAPSE_THR)
        mean = sum(rs) / len(rs)
        good = [r for r in rs if r >= COLLAPSE_THR]
        print(f"  {arm:6s} | mean-recall {mean:.3f}  collapses {nc}/{SEEDS}  "
              f"mean(non-collapsed) {sum(good)/len(good) if good else 0:.3f}")
    pc = sum(1 for r in res["plain"] if r < COLLAPSE_THR)
    sc = sum(1 for r in res["stable"] if r < COLLAPSE_THR)
    print(f"\n  plain collapses {pc}/{SEEDS}, stable collapses {sc}/{SEEDS}  => "
          + ("RESTART-ON-COLLAPSE + warmup makes the decomposed memory RELIABLE (robust-recipe fix)."
             if sc < pc else "stabilizer did NOT remove collapses -> needs a deeper retriever change."))


if __name__ == "__main__":
    main()
