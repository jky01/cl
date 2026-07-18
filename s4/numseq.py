#!/usr/bin/env python3
"""From-scratch small model that EXTRAPOLATES integer sequences by identifying a rule from a prefix and
executing its local recurrence into the UNTRAINED range (user goal 2026-07-18; codex-hardened prereg
2026-07-18.08.30.24).

Honest claim bound (codex): a finite prefix cannot determine a unique generating function; what this shows
is IN-CLASS identification + local-recurrence extrapolation within a PRE-DECLARED hypothesis class
(arithmetic / quadratic) and within a PRE-SELECTED integer width. odd/even are just d=2 arithmetic.

Design:
 - numbers = fixed-width base-10 digits, LSB-first by default (--order); MSB-first is an ablation.
 - NO semantic EOS: sequences are truncated infinite streams, no terminator, loss on digit tokens only
   (an EOS at the training truncation would teach "stop at trained length" and contaminate extrapolation).
 - serialize: BOS d(a0) SEP d(a1) SEP ... (truncated).  next-token LM loss on digit positions only.
 - EVAL = free-running greedy continuation (main metric). 2x2 generalization matrix:
       {seen coef, held-out coef} x {within trained horizon, beyond trained horizon},
   plus a digit-length-boundary crossing test and a value-beyond-trained-max flag; coefficient pools are
   ENUMERATED so we report correct/total, and the train/test split is by GENERATING PARAMETERS.
 - inductive-bias ablations on the SAME task: global vs local-window attention (--W), LSB vs MSB (--order).
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

SEP, BOS, PAD = 10, 11, 12                                # no EOS by design
NTOK = 13


def enc_num(v, wd, order):
    ds = []
    for _ in range(wd):
        ds.append(v % 10); v //= 10                       # LSB-first
    return ds if order == "lsb" else ds[::-1]


def dec_num(ds, order):
    d = ds if order == "lsb" else ds[::-1]
    return sum(x * (10 ** i) for i, x in enumerate(d))


def seq_terms(fam, coef, L):
    if fam == "arith":
        a0, d = coef; return [a0 + d * n for n in range(L)]
    a0, b, c = coef; return [a0 + b * n + c * n * n for n in range(L)]


# ---- coefficient pools, split by generating parameters ----
def all_coefs(fam):
    if fam == "arith":
        return [(a0, d) for a0 in range(10) for d in range(1, 10)]
    return [(a0, b, c) for a0 in range(6) for b in range(6) for c in range(1, 5)]


def is_heldout(fam, coef):
    h = (hash((fam,) + tuple(coef)) % 5)                  # deterministic ~20% held-out by parameters
    return h == 0


def pool(fam, split):
    return [c for c in all_coefs(fam) if (is_heldout(fam, c) == (split == "held"))]


# ---- difference-scaffold (self-generated): make an order-S polynomial a LOCAL step ----
# stream = SEP-separated numbers; term i's row = [D^min(i,S)_i ... D^1_i, a_i]. Model generates the diffs
# itself (integrity: NOT fed by evaluator). arith fixed by S=1 (D^1 constant->copy); quad by S=2.
def diff_table(terms, S):
    T = [list(terms)]
    for k in range(1, S + 1):
        prev = T[k - 1]
        T.append([prev[i] - prev[i - 1] if i >= k else 0 for i in range(len(terms))])
    return T


def num_stream(fam, coef, L, S):
    terms = seq_terms(fam, coef, L); T = diff_table(terms, S)
    out = []                                              # (value, is_term)
    for i in range(L):
        for k in range(min(i, S), 0, -1):
            out.append((T[k][i], False))
        out.append((terms[i], True))
    return out


def serialize_scaf(fam, coef, L, S, wd, order):
    toks = [BOS]; ypos = []
    for j, (v, _) in enumerate(num_stream(fam, coef, L, S)):
        if j > 0:
            toks.append(SEP)
        for dd in enc_num(v, wd, order):
            toks.append(dd); ypos.append(len(toks) - 1)
    return toks, ypos


@torch.no_grad()
def continue_scaf(model, fam, coef, kterm, L, S, wd, order, maxlen, device):
    full = num_stream(fam, coef, L, S)
    npre = sum(min(i, S) + 1 for i in range(kterm))       # numbers in the first kterm term-rows
    toks = [BOS]
    for j in range(npre):
        if j > 0:
            toks.append(SEP)
        for dd in enc_num(full[j][0], wd, order):
            toks.append(dd)
    pred = []
    for j in range(npre, len(full)):
        toks.append(SEP); got = []
        for _ in range(wd):
            idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
            nd = int(model(idx)[0, len(toks) - 1].argmax()); toks.append(nd); got.append(nd)
        if full[j][1]:
            pred.append(dec_num(got, order))
    return pred, [v for v, t in full[npre:] if t]


def serialize(terms, wd, order):
    toks = [BOS]; ypos = []
    for i, t in enumerate(terms):
        for dd in enc_num(t, wd, order):
            toks.append(dd); ypos.append(len(toks) - 1)
        if i != len(terms) - 1:
            toks.append(SEP)
    return toks, ypos


def gen_batch(fams, n, Lrange, wd, order, rng, maxlen, device, vmax, S=0):
    idx, msk = [], []; got = 0
    while got < n:
        fam = rng.choice(fams); coef = rng.choice(pool(fam, "seen"))
        L = rng.randint(*Lrange); terms = seq_terms(fam, coef, L)
        if terms[-1] > vmax:
            continue
        toks, ypos = serialize_scaf(fam, coef, L, S, wd, order) if S else serialize(terms, wd, order)
        if len(toks) > maxlen:
            continue
        a = toks + [PAD] * (maxlen - len(toks)); m = [0] * maxlen
        for p in ypos:
            m[p] = 1
        idx.append(a[:maxlen]); msk.append(m[:maxlen]); got += 1
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool)


# ---- small rotary transformer, optional local window ----
def rope(x, pos):
    B, H, T, Dh = x.shape; half = Dh // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    ang = pos[:, None].float() * freq[None, :]
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(s, d, h, ff, W):
        super().__init__()
        s.h, s.W = h, W
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)

    def forward(s, x, pos, mask):
        B, T, Dd = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(Dd, 2)
        q = q.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dd // s.h)
        i = torch.arange(T, device=x.device); m = mask
        if s.W:
            m = m | (i[None, :] < i[:, None] - s.W + 1)[None, None]
        att = att.masked_fill(m, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, Dd)
        x = x + s.proj(o)
        return x + s.f2(F.gelu(s.f1(s.ln2(x))))


class TM(nn.Module):
    def __init__(s, d=128, h=4, nl=4, ff=512, W=0):
        super().__init__()
        s.emb = nn.Embedding(NTOK, d)
        s.blocks = nn.ModuleList([Block(d, h, ff, W) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NTOK)

    def forward(s, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, pos, mask)
        return s.head(s.lnf(x))


def train(model, fams, steps, bs, lr, Lrange, wd, order, train_maxlen, device, seed, vmax, S=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    for _ in range(steps):
        idx, msk = gen_batch(fams, bs, Lrange, wd, order, rng, train_maxlen, device, vmax, S)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def continue_seq(model, fam, coef, k, L, wd, order, maxlen, device):
    terms = seq_terms(fam, coef, L); toks = [BOS]
    for i in range(k):
        toks += enc_num(terms[i], wd, order)
        toks.append(SEP)
    pred = []
    for i in range(k, L):
        got = []
        for _ in range(wd):
            idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
            nd = int(model(idx)[0, len(toks) - 1].argmax()); toks.append(nd); got.append(nd)
        pred.append(dec_num(got, order))
        if i != L - 1:
            toks.append(SEP)
    return pred, terms[k:L]


@torch.no_grad()
def eval_pool(model, fam, split, k, L, wd, order, maxlen, device, vmax, Ltrain, S=0):
    """free-running continuation over the ENUMERATED coef pool; correct/total + horizon diagnostics."""
    full_ok = tot = far_ok = 0; first_err = []; crossed = crossed_ok = 0
    for coef in pool(fam, split):
        terms = seq_terms(fam, coef, L)
        if terms[-1] > vmax:
            continue
        if S:
            pred, true = continue_scaf(model, fam, coef, k, L, S, wd, order, maxlen, device)
        else:
            pred, true = continue_seq(model, fam, coef, k, L, wd, order, maxlen, device)
        tot += 1
        full_ok += int(pred == true)
        far_ok += int(pred[-1] == true[-1])
        fe = next((j for j in range(len(true)) if pred[j] != true[j]), len(true))
        first_err.append(fe)
        # digit-length boundary: did any trained-range term have fewer digits than a beyond-range term?
        if len(str(true[-1])) > len(str(seq_terms(fam, coef, Ltrain)[-1])):
            crossed += 1; crossed_ok += int(pred[-1] == true[-1])
    mfe = sum(first_err) / max(len(first_err), 1)
    return dict(full=full_ok / max(tot, 1), far=far_ok / max(tot, 1), tot=tot, mfe=mfe,
                cross=crossed, cross_ok=crossed_ok / max(crossed, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", type=str, default="arith")
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--W", type=int, default=0)
    ap.add_argument("--order", type=str, default="lsb")          # lsb | msb (ablation)
    ap.add_argument("--wd", type=int, default=4)
    ap.add_argument("--Ltrain", type=int, default=8); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=512)           # EVAL context (rollout is long)
    ap.add_argument("--train_maxlen", type=int, default=128)     # TRAIN seqs are short -> keep attn O(T^2) small
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scaffold", type=int, default=0)           # 0=plain; 1=first-diff; 2=second-diff
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    fams = args.fam.split(","); vmax = 10 ** args.wd - 1; S = args.scaffold
    model = TM(W=args.W).to(device)
    train(model, fams, args.steps, args.bs, args.lr, (args.Ltrain // 2, args.Ltrain),
          args.wd, args.order, args.train_maxlen, device, args.seed, vmax, S)
    npar = sum(p.numel() for p in model.parameters())
    print(f"PREREG NUMSEQ fam={fams} W={args.W} order={args.order} wd={args.wd} vmax={vmax} scaffold={S} "
          f"Ltrain={args.Ltrain} k={args.k} steps={args.steps} seed={args.seed} params={npar/1e6:.2f}M "
          f"device={device}")
    print("success bar (arith): held-out coef, beyond horizon, value>train-max, cross>=1 digit boundary, "
          "free-running exact.  main cell = [held x beyond]. metric=correct/total over enumerated pool.")
    horizons = [args.Ltrain, 2 * args.Ltrain, 3 * args.Ltrain]
    for fam in fams:
        print(f"[{fam}]  (pool sizes: seen={len(pool(fam,'seen'))} held={len(pool(fam,'held'))})")
        for split in ["seen", "held"]:
            for L in horizons:
                tag = "within" if L <= args.Ltrain else "BEYOND"
                r = eval_pool(model, fam, split, args.k, L, args.wd, args.order, args.maxlen, device, vmax, args.Ltrain, S)
                print(f"   {split:>4} x {tag:>6} L={L:>2} (n={r['tot']:>3}): full-exact {r['full']:.2f}  "
                      f"farthest {r['far']:.2f}  first-err-term {r['mfe']:.1f}  "
                      f"digit-cross {r['cross_ok']:.2f}(n={r['cross']})")
    print("\nread: [held x BEYOND] farthest & digit-cross high => identified the rule from prefix and "
          "executed its recurrence past trained horizon/value/width => genuine in-class extrapolation.")


if __name__ == "__main__":
    main()
