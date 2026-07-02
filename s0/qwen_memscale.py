"""EXPERIMENT A: does the router-free memory SCALE? capstone-2 retained ~0.99 but
over only 96 facts -- retrieval was easy. Here we grow the bank (128 -> ~1500
DISTINCT facts) and ask whether the LEARNED-KEY retrieval stays discriminative.
The bottleneck for scale is retrieval@1 (does the query's argmax key = the right
fact); ANN only makes that FAST, it can't fix accuracy -- so retrieval@1 vs bank
size is the real scaling test. answer-recall = retrieval + injection.

Key efficiency: Qwen is frozen, so pooled(text) is FIXED -> we precompute the
bank's features ONCE and train the small memory modules on cached tensors (no
repeated Qwen forwards) -> large banks + many steps become cheap.

Reports, per bank size N: retrieval@1 (argmax over all N keys), answer-recall
(full-softmax injection), and top-k=8 retrieval@1 (an ANN proxy: does restricting
to the 8 nearest keep the right fact in the shortlist).

  python3 -m s0.qwen_memscale      # env: MS_SIZES="128,512,1024", MS_STEPS, MS_KDIM
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
SIZES = [int(x) for x in os.environ.get("MS_SIZES", "128,512,1024").split(",")]
STEPS = int(os.environ.get("MS_STEPS", 6000))
KDIM = int(os.environ.get("MS_KDIM", 128))
ATTRS = list(ATTR_VALUES)

# a large pool of (mostly) single-token first names; filtered to 1-token at runtime
NAME_POOL = (
    "Alice Bob Carol David Emma Frank Grace Henry Iris Jack Karen Leo Mia Nina Oscar "
    "Paula Quinn Rosa Sam Tina Uma Victor Wendy Xavier Yara Zack Adam Beth Caleb Dana "
    "Eli Fiona Gabe Hana Ian Julia Kyle Lena Mike Nora Omar Page Ravi Sara Theo Ula "
    "Vince Will Xena Yuri Aaron Bella Cody Diana Ethan Faith Greg Holly Isaac Jane "
    "Kevin Laura Mark Naomi Owen Priya Ryan Sofia Tom Vera Wade Zoe Abel Bianca Carl "
    "Delia Evan Freya Gary Hazel Igor Jenna Kurt Liam Molly Neil Olga Pete Rita Seth "
    "Tara Umar Wren Yusuf Alan Britt Cole Daisy Emil Gina Hugo Ivy Jonah Kara Luke "
    "Maya Noel Opal Reed Ruby Saul Faye Blake Cara Dean Elsa Finn Gwen Hank Isla Jade "
    "Kirk Lois Milo Nova Otis Pearl Rory Skye Troy Vlad Wyatt Zane Anya Bruno Clara "
    "Dirk Enzo Gail Hugh Inez Jules Knox Lila Mace Nell Orla Piper Roan Suki Toby"
).split()


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
    print(f"MEMSCALE ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"sizes={SIZES} steps={STEPS} kdim={KDIM}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}
    names = [n for n in NAME_POOL if len(tok(" " + n, add_special_tokens=False).input_ids) == 1]
    names = list(dict.fromkeys(names))                       # dedupe, keep order
    max_pairs = len(names) * len(ATTRS)
    print(f"  usable single-token names: {len(names)}  -> up to {max_pairs} distinct facts")

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

    def run(N):
        rng = random.Random(0)
        pairs = rng.sample([(n, a) for n in names for a in ATTRS], N)
        facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
        kt = [f"{n}'s {a}" for (n, a, _) in facts]
        st = [f"{n}'s {a} is {v}." for (n, a, v) in facts]
        qt = [f"{n}'s {a} is" for (n, a, _) in facts]
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        # ---- precompute frozen features ONCE ----
        Kf, Sf, Qf, Hf = pooled(kt), pooled(st), pooled(qt), last_h(qt)   # [N,d]...
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mods = [proj_k, proj_q, val_enc, val_dec, gate]
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        Bsz = min(128, N);
        for step in range(STEPS):
            idx = torch.randint(0, N, (Bsz,), device=device)
            K = F.normalize(proj_k(Kf[idx]), -1); V = val_enc(Sf[idx])
            q = F.normalize(proj_q(Qf[idx]), -1)
            R = val_dec(torch.softmax(q @ K.t() / 0.05, -1) @ V)
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            logits = lm.lm_head((H + g * R)).float()
            tgt = torch.arange(Bsz, device=device)
            loss = F.cross_entropy(logits, gold[idx]) + F.cross_entropy(q @ K.t() / 0.05, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        # ---- eval over the FULL bank ----
        with torch.no_grad():
            Kall = F.normalize(proj_k(Kf), -1)                      # [N,kdim]
            qall = F.normalize(proj_q(Qf), -1)
            Vall = val_enc(Sf)
            sims = qall @ Kall.t()                                  # [N,N]
            r1 = (sims.argmax(1) == torch.arange(N, device=device)).float().mean().item()
            topk = sims.topk(min(8, N), dim=1).indices
            rk = (topk == torch.arange(N, device=device)[:, None]).any(1).float().mean().item()
            # answer recall (full softmax injection), batched
            rec = 0
            for i in range(0, N, 256):
                s = sims[i:i + 256]
                R = val_dec(torch.softmax(s / 0.05, -1) @ Vall)
                H = Hf[i:i + 256]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
                pred = lm.lm_head((H + g * R)).float().argmax(-1)
                rec += (pred == gold[i:i + 256]).sum().item()
            rec /= N
        return r1, rk, rec

    print(f"\n  {'N':>6} | {'retrieval@1':>12} {'top8@ANN':>9} {'answer-recall':>14}")
    results = []
    for N in SIZES:
        if N > max_pairs:
            print(f"  {N:>6} | skip (only {max_pairs} distinct pairs available)"); continue
        r1, rk, rec = run(N)
        results.append((N, r1, rk, rec))
        print(f"  {N:>6} | {r1:>12.3f} {rk:>9.3f} {rec:>14.3f}", flush=True)
    print("\n  retrieval@1 holding as N grows => router-free memory SCALES (ANN keeps it fast);")
    print("  collapsing => learned keys saturate -> need bigger key dim / harder-negative training.")


if __name__ == "__main__":
    main()
