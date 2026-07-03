"""ROUND 12 — is the 20k retrieval collapse fixable by an ALGORITHMIC change?
Rounds 10-11 found router-free learned-key retrieval COLLAPSES at 20k (retr@1 0.000)
and that MORE compute (2.5x steps, 2x key-dim) does NOT rescue it -> the full-bank
InfoNCE (20000-way CE from a COLD start) is an optimization wall, not a budget issue.
Cheapest algorithmic fix to test: a BANK-GROWTH CURRICULUM. Start the retrieval on a
small sub-bank (1k), where the softmax target-count is easy, then GROW the bank
(1k->3k->8k->20k) warm-starting the SAME projections, so the discrimination problem
escalates gradually instead of cold-starting at 20000-way. Compare at matched TOTAL
steps against the cold-start-at-20k baseline.

  curric : grow the bank 1k->3k->8k->20k, warm-start projections, split steps by stage
  cold   : train over the full 20k bank for the same TOTAL steps (the round-11 baseline)

Final eval over ALL N facts: retrieval@1 / answer-recall / top32. If curric >> cold,
the collapse is an optimization-path artifact curable by curriculum (a real fix); if
curric also collapses, 20k needs a structurally different retriever (hierarchical/ANN).

  python3 -m s0.qwen_memscale_curric   # env: MC_N, MC_STAGES, MC_STEPS, MC_SEEDS, MC_KDIM
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_memory import ATTR_VALUES
from .qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
N = int(os.environ.get("MC_N", 20000))
STAGES = [int(x) for x in os.environ.get("MC_STAGES", "1000,3000,8000,20000").split(",")]
STEPS = int(os.environ.get("MC_STEPS", 8000))           # TOTAL steps (matched across arms)
SEEDS = int(os.environ.get("MC_SEEDS", 2))
KDIM = int(os.environ.get("MC_KDIM", 256))
TOPK = int(os.environ.get("MC_TOPK", 32))
ATTRS = list(ATTR_VALUES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    d = lm.config.hidden_size
    if os.environ.get("MC_BIGPOOL"):                        # large pool from REAL-name TRIPLES (first x middle x last)
        need = N // len(ATTRS) + 200                        # -> discriminable keys (avoids near-duplicate gibberish)
        names = []                                          # 68 x 68 x 64 ~= 296k distinct real-name triples
        for f in FIRST:
            for m in FIRST:
                for l in LAST:
                    names.append(f"{f} {m} {l}")
                    if len(names) >= need:
                        break
                if len(names) >= need:
                    break
            if len(names) >= need:
                break
    else:
        names = [f"{f} {l}" for f in FIRST for l in LAST]
    print(f"MEMSCALE-CURRIC ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"N={N} stages={STAGES} total-steps={STEPS} seeds={SEEDS} kdim={KDIM}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = lm.model(**e).last_hidden_state
            m = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * m).sum(1) / m.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(texts, bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(lm.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

    def build(N, seed):
        rng = random.Random(seed)
        pairs = rng.sample([(n, a) for n in names for a in ATTRS], N)
        facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
        Kf = pooled([f"{n}'s {a}" for (n, a, _) in facts])
        Sf = last_h([f"{n}'s {a} is {v}" for (n, a, v) in facts])
        Qf = pooled([f"{n}'s {a} is" for (n, a, _) in facts])
        Hf = last_h([f"{n}'s {a} is" for (n, a, _) in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        return Kf, Sf, Qf, Hf, gold

    def new_mods(seed):
        torch.manual_seed(seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        return (proj_k, proj_q, val_enc, val_dec, gate)

    def train_prefix(mods, opt, F_, sz, steps, Bq=256):
        """Train retrieval+injection over the first `sz` facts (the current sub-bank)."""
        Kf, Sf, Qf, Hf, gold = (t[:sz] for t in F_)
        proj_k, proj_q, val_enc, val_dec, gate = mods
        for _ in range(steps):
            idx = torch.randint(0, sz, (min(Bq, sz),), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
            q = F.normalize(proj_q(Qf[idx]), -1)
            sims = q @ Kall.t() / 0.05
            vk, ik = sims.topk(min(TOPK, sz), dim=1)
            w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(lm.lm_head((H + g * R)).float(), gold[idx]) \
                + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0); opt.step()

    @torch.no_grad()
    def evaluate(mods, F_, sz):
        Kf, Sf, Qf, Hf, gold = (t[:sz] for t in F_)
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kall = F.normalize(proj_k(Kf), -1); qall = F.normalize(proj_q(Qf), -1); Vall = val_enc(Sf)
        r1 = rf = rk = 0
        for i in range(0, sz, 256):
            ri = torch.arange(i, min(i + 256, sz), device=device)
            s = qall[ri] @ Kall.t()
            r1 += (s.argmax(1) == ri).sum().item()
            vk, ik = (s / 0.05).topk(min(TOPK, sz), 1)
            rk += (ik == ri[:, None]).any(1).sum().item()
            w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[ri]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            rf += (lm.lm_head((H + g * R)).float().argmax(-1) == gold[ri]).sum().item()
        return r1 / sz, rf / sz, rk / sz

    def run_curric(F_, seed):
        mods = new_mods(seed * 100 + 1)
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
        per = STEPS // len(STAGES)
        for si, sz in enumerate(STAGES):
            steps = per + (STEPS - per * len(STAGES) if si == len(STAGES) - 1 else 0)
            train_prefix(mods, opt, F_, sz, steps)
        return evaluate(mods, F_, N)

    def run_cold(F_, seed):
        mods = new_mods(seed * 100 + 1)
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
        train_prefix(mods, opt, F_, N, STEPS)
        return evaluate(mods, F_, N)

    skip_cold = bool(os.environ.get("MC_SKIP_COLD"))        # cold collapses (see rounds 10-14); skip to save wall-time at large N
    print(f"\n  {'arm':7s} | {'retr@1':>7} {'ans-recall':>11} {'top32':>7}   (mean/{SEEDS})")
    agg = {"curric": [], "cold": []}
    for seed in range(SEEDS):
        F_ = build(N, seed)
        agg["curric"].append(run_curric(F_, seed))
        agg["cold"].append((0.0, 0.0, 0.0) if skip_cold else run_cold(F_, seed))
        print(f"  seed {seed}: curric {tuple(round(x,3) for x in agg['curric'][-1])}  "
              f"cold {tuple(round(x,3) for x in agg['cold'][-1])}{' (skipped)' if skip_cold else ''}", flush=True)
    for arm in ("curric", "cold"):
        rs = agg[arm]; m = lambda j: sum(r[j] for r in rs) / len(rs)
        print(f"  {arm:7s} | {m(0):>7.3f} {m(1):>11.3f} {m(2):>7.3f}")
    cm, km = sum(r[0] for r in agg["curric"]) / SEEDS, sum(r[0] for r in agg["cold"]) / SEEDS
    print(f"\n  curric retr@1 {cm:.3f} vs cold {km:.3f}  => "
          + ("CURRICULUM RESCUES 20k (collapse was an optimization-path artifact)."
             if cm > 0.5 and cm > km + 0.2 else
             "curriculum does NOT rescue 20k -> needs a structurally different retriever."))


if __name__ == "__main__":
    main()
