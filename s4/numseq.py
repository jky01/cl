#!/usr/bin/env python3
"""From-scratch small model that EXTRAPOLATES integer sequences -- genuinely learning the generating
function, verified by continuation into the UNTRAINED range (user goal 2026-07-18).

Families: arithmetic  a_n = a0 + d*n     (odd/even = a0 in {1,0}, d=2; general d)   -- order-1 recurrence
          quadratic   a_n = a0 + b*n + c*n^2 (c>=1)                                  -- order-2 recurrence

Numbers are fixed-width base-10 digits, LSB-FIRST (units first) so carry is local (the digit analogue of
the interleaving that let csum_reset extrapolate). Serialize a sequence as
  BOS  d(a0) SEP d(a1) SEP ... d(a_{L-1})  EOS
and train next-token LM loss on all digit positions. EXTRAPOLATION test: feed the first k terms, generate
the rest autoregressively, exact-match the terms that lie BEYOND the trained term-index / value range and
on HELD-OUT coefficients. A local attention window (~3 terms) forces position-invariance so an order-<=2
recurrence rolled forward extrapolates. Honest control: a closed-form-from-index view (given n predict a_n)
is expected to OOD-fail -- reported to show the difference is inductive bias, not capacity.
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- vocab: 0..9 digits, SEP, BOS, EOS, PAD ----
D0 = 0                       # digit tokens are their own value 0..9
SEP, BOS, EOS, PAD = 10, 11, 12, 13
NTOK = 14
WD = 4                       # digits per number (LSB first) -> values 0..9999


def enc_num(v):
    ds = []
    for _ in range(WD):
        ds.append(v % 10); v //= 10
    return ds                                            # LSB-first, fixed width


def dec_num(ds):
    return sum(d * (10 ** i) for i, d in enumerate(ds))


def seq_terms(fam, coef, L):
    if fam == "arith":
        a0, d = coef
        return [a0 + d * n for n in range(L)]
    a0, b, c = coef
    return [a0 + b * n + c * n * n for n in range(L)]


def serialize(terms):
    toks = [BOS]
    ypos = []                                            # positions we score (all digit tokens)
    for i, t in enumerate(terms):
        for d in enc_num(t):
            toks.append(d); ypos.append(len(toks) - 1)
        if i != len(terms) - 1:
            toks.append(SEP)
    toks.append(EOS)
    return toks


def sample_coef(fam, rng, held_out=False):
    if fam == "arith":
        a0 = rng.randint(0, 9); d = rng.randint(1, 9)
        return (a0, d)
    a0 = rng.randint(0, 5); b = rng.randint(0, 5); c = rng.randint(1, 4)
    return (a0, b, c)


def gen_batch(fams, n, Lrange, rng, maxlen, device, vmax=9999):
    idx, msk = [], []
    got = 0
    while got < n:
        fam = rng.choice(fams)
        coef = sample_coef(fam, rng)
        L = rng.randint(*Lrange)
        terms = seq_terms(fam, coef, L)
        if terms[-1] > vmax:                             # keep within digit width
            continue
        toks = serialize(terms)
        if len(toks) > maxlen:
            continue
        a = toks + [PAD] * (maxlen - len(toks))
        m = [0] * maxlen
        for p in range(len(toks)):
            if a[p] < SEP:                               # score only digit tokens
                m[p] = 1
        idx.append(a); msk.append(m); got += 1
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool)


# ---- small rotary transformer, optional local window ----
def rope(x, pos):
    B, H, T, Dh = x.shape
    half = Dh // 2
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
        i = torch.arange(T, device=x.device)
        m = mask
        if s.W:
            win = (i[None, :] < i[:, None] - s.W + 1)
            m = m | win[None, None]
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


def train(model, fams, steps, bs, lr, Lrange, maxlen, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    for _ in range(steps):
        idx, msk = gen_batch(fams, bs, Lrange, rng, maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def continue_seq(model, fam, coef, k, L, maxlen, device):
    """feed first k terms, autoregressively generate through term L-1; return predicted terms[k:L]."""
    terms = seq_terms(fam, coef, L)
    toks = [BOS]
    for i in range(k):
        toks += enc_num(terms[i]); toks.append(SEP)
    pred = []
    for i in range(k, L):
        got = []
        for _ in range(WD):
            idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
            nd = int(model(idx)[0, len(toks) - 1].argmax()); toks.append(nd); got.append(nd)
        pred.append(dec_num(got))
        if i != L - 1:
            toks.append(SEP)
    return pred, terms[k:L]


@torch.no_grad()
def eval_extrap(model, fam, k, L, maxlen, device, n, seed, vmax=9999):
    rng = random.Random(seed); ok_all = 0; ok_last = 0; tot = 0
    for _ in range(n):
        coef = sample_coef(fam, rng)
        if seq_terms(fam, coef, L)[-1] > vmax:
            continue
        pred, true = continue_seq(model, fam, coef, k, L, maxlen, device)
        ok_all += int(pred == true)                     # every extrapolated term exact
        ok_last += int(pred[-1] == true[-1])            # the FARTHEST term exact
        tot += 1
    return ok_all / max(tot, 1), ok_last / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", type=str, default="arith")          # arith | quad | arith,quad
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--W", type=int, default=0)
    ap.add_argument("--Ltrain", type=int, default=8); ap.add_argument("--Ltest", type=int, default=16)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    fams = args.fam.split(",")
    model = TM(W=args.W).to(device)
    train(model, fams, args.steps, args.bs, args.lr, (args.Ltrain // 2, args.Ltrain), args.maxlen, device, args.seed)
    npar = sum(p.numel() for p in model.parameters())
    print(f"device={device} NUMSEQ fam={fams} W={args.W} params={npar/1e6:.2f}M "
          f"train_terms<= {args.Ltrain} (val<=9999); test continue k={args.k}->L")
    for fam in fams:
        print(f"[{fam}]")
        for L in [args.Ltrain, 12, args.Ltest, 24]:
            if L < args.k + 1:
                continue
            a, last, tot = eval_extrap(model, fam, args.k, L, args.maxlen, device, args.eval_n, 500 + L)
            tag = "in-range" if L <= args.Ltrain else "EXTRAP"
            print(f"   continue to L={L:>2} ({tag:>8}, n={tot:>3}): all-terms-exact {a:.2f}  farthest-term {last:.2f}")
    print("\nread: farthest-term exact at L>Ltrain = genuine forward extrapolation of the generating "
          "recurrence (learned the function). all-terms-exact is the strict bar. arithmetic (order-1) "
          "should extrapolate; quadratic (order-2) is the harder probe.")


if __name__ == "__main__":
    main()
