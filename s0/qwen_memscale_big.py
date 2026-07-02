"""EXPERIMENT D — push router-free memory to a REAL lifelong scale (1k -> 10k facts).
Exp A closed at ~1500 (name-pool limited). The key insight to scale further: only
the ANSWER value must be single-token; the KEY text ("X's attr") can tokenize to
anything, so a large first x last name space gives >>10k distinct facts. Uses the
same closed-form fixes (cache frozen features once; train retrieval+injection over
the FULL bank; restart-on-collapse) so 10k is cheap. Bigger key dim (256) for
headroom. Reports retrieval@1 (exact argmax = the accuracy ANN must preserve),
answer-recall, and top-32 (ANN shortlist) hit-rate, mean over seeds.

  python3 -m s0.qwen_memscale_big    # env: MB_SIZES="1000,4000,10000", MB_STEPS, MB_SEEDS, MB_KDIM
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_memory import ATTR_VALUES

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
SIZES = [int(x) for x in os.environ.get("MB_SIZES", "1000,4000,10000").split(",")]
STEPS = int(os.environ.get("MB_STEPS", 5000))
SEEDS = int(os.environ.get("MB_SEEDS", 2))
KDIM = int(os.environ.get("MB_KDIM", 256))
RESTARTS = int(os.environ.get("MB_RESTARTS", 3))
TOPK = int(os.environ.get("MB_TOPK", 32))   # ANN-style shortlist for injection readout
ATTRS = list(ATTR_VALUES)

FIRST = ("Alice Bob Carol David Emma Frank Grace Henry Iris Jack Karen Leo Mia Nina "
         "Oscar Paula Quinn Rosa Sam Tina Uma Victor Wendy Xavier Yara Zack Adam Beth "
         "Caleb Dana Eli Fiona Gabe Hana Ian Julia Kyle Lena Mike Nora Omar Ravi Sara "
         "Theo Vince Will Yuri Aaron Bella Cody Diana Ethan Faith Greg Holly Isaac Jane "
         "Kevin Laura Mark Naomi Owen Priya Ryan Sofia Tom Vera Wade Zoe").split()
LAST = ("Smith Jones Lee Brown Garcia Miller Davis Chen Kim Patel Nguyen Khan Silva "
        "Rossi Muller Wang Kumar Singh Lopez Young Hall Allen Scott Green Baker Adams "
        "Nelson Hill Reed Cook Ward Gray Cole Wood Bell Ross Ward Price Long Foster "
        "Bennett Barnes Coleman Hayes Jenkins Perry Powell Russell Bryant Butler Simmons "
        "Fisher Murphy Kelly Sanders Price Bishop Chapman Fox Grant Hunt Knight Lane").split()


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
    names = [f"{f} {l}" for f in FIRST for l in LAST]        # first x last (multi-token OK for keys)
    print(f"MEMSCALE-BIG ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"sizes={SIZES} steps={STEPS} seeds={SEEDS} kdim={KDIM}")
    print(f"  name space: {len(names)} x {len(ATTRS)} attrs = {len(names)*len(ATTRS)} distinct facts")

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
        Sf = pooled([f"{n}'s {a} is {v}." for (n, a, v) in facts])
        Qf = pooled([f"{n}'s {a} is" for (n, a, _) in facts])
        Hf = last_h([f"{n}'s {a} is" for (n, a, _) in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        return Kf, Sf, Qf, Hf, gold

    def train_once(Kf, Sf, Qf, Hf, gold, N, init_seed):
        torch.manual_seed(init_seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mods = (proj_k, proj_q, val_enc, val_dec, gate)
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        Bq = 256
        for _ in range(STEPS):
            idx = torch.randint(0, N, (Bq,), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
            q = F.normalize(proj_q(Qf[idx]), -1)
            sims = q @ Kall.t() / 0.05
            vk, ik = sims.topk(min(TOPK, N), dim=1)               # ANN shortlist
            w = torch.softmax(vk, -1)                             # softmax over top-k only
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))      # inject top-k mixture
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(lm.lm_head((H + g * R)).float(), gold[idx]) \
                + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        return mods

    @torch.no_grad()
    def evaluate(mods, Kf, Sf, Qf, Hf, gold, N, rows=None):
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kall = F.normalize(proj_k(Kf), -1); qall = F.normalize(proj_q(Qf), -1); Vall = val_enc(Sf)
        rows = torch.arange(N, device=device) if rows is None else rows
        r1 = rf = rk = 0; n = len(rows)
        for i in range(0, n, 256):
            ri = rows[i:i + 256]
            s = qall[ri] @ Kall.t()
            r1 += (s.argmax(1) == ri).sum().item()
            vk, ik = (s / 0.05).topk(min(TOPK, N), 1)             # ANN shortlist
            rk += (ik == ri[:, None]).any(1).sum().item()
            w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))      # top-k injection readout
            H = Hf[ri]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            rf += (lm.lm_head((H + g * R)).float().argmax(-1) == gold[ri]).sum().item()
        return r1 / n, rf / n, rk / n

    def run(N, seed):
        Kf, Sf, Qf, Hf, gold = build(N, seed)
        sub = torch.randperm(N, device=device)[:256]
        used = 0
        for attempt in range(RESTARTS + 1):
            used = attempt
            mods = train_once(Kf, Sf, Qf, Hf, gold, N, init_seed=seed * 100 + attempt)
            if evaluate(mods, Kf, Sf, Qf, Hf, gold, N, rows=sub)[1] >= 0.5:
                break
        r1, rf, rk = evaluate(mods, Kf, Sf, Qf, Hf, gold, N)
        return r1, rf, rk, used

    print(f"\n  {'N':>7} | {'retr@1':>7} {'ans-recall':>11} {'top32':>7} {'restarts':>9}  (mean/{SEEDS})")
    for N in SIZES:
        rs = [run(N, s) for s in range(SEEDS)]
        m = lambda j: sum(r[j] for r in rs) / len(rs)
        lo = lambda j: min(r[j] for r in rs)
        print(f"  {N:>7} | {m(0):>7.3f} {m(1):>11.3f} {m(2):>7.3f} {sum(r[3] for r in rs):>9}   "
              f"(recall min {lo(1):.3f})", flush=True)
    print("\n  retrieval@1 & recall holding at 10k => router-free lifelong memory scales to a")
    print("  real lifetime of facts; top32 ~1.0 => an ANN index preserves it at speed.")


if __name__ == "__main__":
    main()
