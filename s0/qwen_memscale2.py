"""EXPERIMENT A' — clean up the memory-scaling result. qwen_memscale showed
router-free RETRIEVAL scales (retrieval@1 ~0.97 to 1200) but answer/injection
recall was unstable at large N (collapsed at 1200; a 512 collapse recovered on
rerun). Diagnosis: TRAIN/EVAL MISMATCH — training retrieved/injected over only the
in-batch Bsz keys, but eval is over ALL N keys, so at large N the eval softmax is
far more diffuse than anything seen in training. FIX: train retrieval AND injection
over the FULL bank every step (cheap with cached frozen features). Plus fixed
seeds + multi-seed for a trustworthy number.

Reports mean over seeds, per bank size: retrieval@1, answer-recall (full-bank
softmax injection = the eval-matched objective), answer-recall (hard top-1).

  python3 -m s0.qwen_memscale2     # env: MS_SIZES, MS_STEPS, MS_SEEDS, MS_KDIM
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
SIZES = [int(x) for x in os.environ.get("MS_SIZES", "512,1024,1500").split(",")]
STEPS = int(os.environ.get("MS_STEPS", 5000))
SEEDS = int(os.environ.get("MS_SEEDS", 3))
KDIM = int(os.environ.get("MS_KDIM", 128))
ATTRS = list(ATTR_VALUES)

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
    "Dirk Enzo Gail Hugh Inez Jules Knox Lila Mace Nell Orla Piper Roan Suki Toby "
    "Amos Bree Chip Dale Erin Fred Gwyn Hope Igna Joel Kai Lars Nate Odin Poppy Quin "
    "Ross Suze Trey Vic Wes Xander Yael Zed Ada Ben Cleo Drew Ella Gus Hugo Ivan Jo "
    "Kim Lou Max Ned Ola Rex Sky Tod Val Wil Yan Zoltan Bo Cy Deb Ed Flo Hal Jed Mo"
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
    print(f"MEMSCALE-2 full-bank training ({NAME}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"sizes={SIZES} steps={STEPS} seeds={SEEDS} kdim={KDIM}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}
    names = list(dict.fromkeys(n for n in NAME_POOL
                               if len(tok(" " + n, add_special_tokens=False).input_ids) == 1))
    max_pairs = len(names) * len(ATTRS)
    print(f"  usable single-token names: {len(names)} -> up to {max_pairs} distinct facts")

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

    def build(N, seed):
        rng = random.Random(seed)
        pairs = rng.sample([(n, a) for n in names for a in ATTRS], N)
        facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
        kt = [f"{n}'s {a}" for (n, a, _) in facts]
        st = [f"{n}'s {a} is {v}." for (n, a, v) in facts]
        qt = [f"{n}'s {a} is" for (n, a, _) in facts]
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        return pooled(kt), pooled(st), pooled(qt), last_h(qt), gold

    def run(N, seed):
        Kf, Sf, Qf, Hf, gold = build(N, seed)
        torch.manual_seed(seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        params = [p for m in (proj_k, proj_q, val_enc, val_dec, gate) for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        Bq = min(256, N)
        for step in range(STEPS):
            idx = torch.randint(0, N, (Bq,), device=device)
            Kall = F.normalize(proj_k(Kf), -1)            # FULL bank keys [N,kd]
            Vall = val_enc(Sf)                            # FULL bank values [N,256]
            q = F.normalize(proj_q(Qf[idx]), -1)          # [Bq,kd]
            sims = q @ Kall.t() / 0.05                    # [Bq,N] retrieve over WHOLE bank
            R = val_dec(torch.softmax(sims, -1) @ Vall)   # inject over WHOLE bank (eval-matched)
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(lm.lm_head((H + g * R)).float(), gold[idx]) \
                + F.cross_entropy(sims, idx)              # target = own index in full bank
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            Kall = F.normalize(proj_k(Kf), -1); qall = F.normalize(proj_q(Qf), -1); Vall = val_enc(Sf)
            sims = qall @ Kall.t()
            r1 = (sims.argmax(1) == torch.arange(N, device=device)).float().mean().item()
            rf = rt1 = 0
            for i in range(0, N, 256):
                s = sims[i:i + 256] / 0.05
                Rf = val_dec(torch.softmax(s, -1) @ Vall)
                Rt = val_dec(Vall[s.argmax(1)])
                for R, acc in ((Rf, "f"), (Rt, "t")):
                    H = Hf[i:i + 256]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
                    pred = lm.lm_head((H + g * R)).float().argmax(-1)
                    n_ok = (pred == gold[i:i + 256]).sum().item()
                    if acc == "f": rf += n_ok
                    else: rt1 += n_ok
            return r1, rf / N, rt1 / N

    print(f"\n  {'N':>6} | {'retr@1':>7} {'ans:full':>9} {'ans:top1':>9}  (mean over {SEEDS} seeds)")
    for N in SIZES:
        if N > max_pairs:
            print(f"  {N:>6} | skip (max {max_pairs} pairs)"); continue
        rs = [run(N, s) for s in range(SEEDS)]
        m = lambda j: sum(r[j] for r in rs) / len(rs)
        lo = lambda j: min(r[j] for r in rs)
        print(f"  {N:>6} | {m(0):>7.3f} {m(1):>9.3f} {m(2):>9.3f}   (full min {lo(1):.3f})", flush=True)
    print("\n  full-bank training should keep ans:full high (no train/eval mismatch) and stable")
    print("  across seeds at N>=1024 -> reliable router-free 1000+fact recall.")


if __name__ == "__main__":
    main()
