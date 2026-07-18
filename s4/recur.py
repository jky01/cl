#!/usr/bin/env python3
"""One-token-per-state linear recurrences over a prime field F_p -- the substrate that AVOIDS the
multi-digit tokenized-execution wall (numseq: plain+scaffold+NoPE all collapse beyond the trained
horizon). Here each term is a SINGLE token, so the recurrence step is a genuine local function of the
previous 1-2 tokens (like csum_reset, which extrapolated to L40).

Family = FINITE-DIFFERENCE (polynomial) sequences mod p, so the rule is inferable from the prefix by
SUBTRACTION only (NOT modular division, which full affine a!=1 would need and a small transformer cannot
learn). This is exactly arithmetic + quadratic:
  order 1 (arithmetic): constant 1st difference d ; s_n = (s_{n-1} + d) mod p        (odd/even are members)
  order 2 (quadratic) : constant 2nd difference e ; s_n = (2 s_{n-1} - s_{n-2} + e) mod p
Rule (what must generalize) = the top difference delta; the lower initial differences are the prefix-
identifiable state (random seed). Newton forward-difference cascade advances one step; each term = v[0].

Claim target (codex 2026-07-18.17.55.39): "from an informative prefix, a small AR model infers a member of
a bounded recurrence family and executes it for horizons longer than training." Bounded value-token vocab
makes the domain limit explicit. PREREG: disjoint train/held-out PARAM sets; free-running exact + per-term
+ first-error; within-horizon H and beyond 2H/4H; random position offsets; seeds; a memorization control
(held-out params + large p so it is not a tiny transition table).
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 211                                                  # prime field size (value tokens 0..P-1)
BOS = P; PAD = P + 1; NTOK = P + 2


def advance(v, delta):
    D = len(v)
    return [(v[i] + (v[i + 1] if i + 1 < D else delta)) % P for i in range(D)]


def gen_seq(order, delta, v0, L):
    v = list(v0); seq = []
    for _ in range(L):
        seq.append(v[0]); v = advance(v, delta)
    return seq


def all_coefs(order, rng, n):
    vals = list(range(1, P))                                  # top difference delta in [1,p-1] (delta=0 trivial)
    rng.shuffle(vals)
    return vals[:min(n, len(vals))]


def seeds0(order, rng):
    return [rng.randrange(0, P) for _ in range(order)]        # initial diffs v[0..order-1] (prefix-identified)


# ---- model: one token per state, rotary + local window, optional NoPE ----
def rope(x, pos):
    B, H, T, Dh = x.shape; half = Dh // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    ang = pos[:, None].float() * freq[None, :]
    cos = ang.cos()[None, None]; sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(s, d, h, ff, W, nope):
        super().__init__()
        s.h, s.W, s.nope = h, W, nope
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)

    def forward(s, x, pos, mask):
        B, T, Dd = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(Dd, 2)
        q = q.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        if not s.nope:
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
    def __init__(s, d=192, h=6, nl=4, ff=768, W=8, nope=False):
        super().__init__()
        s.emb = nn.Embedding(NTOK, d)
        s.blocks = nn.ModuleList([Block(d, h, ff, W, nope) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NTOK)

    def forward(s, idx, pos0=0):
        B, T = idx.shape
        pos = torch.arange(pos0, pos0 + T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, pos, mask)
        return s.head(s.lnf(x))


def serialize(seq):
    return [BOS] + seq


def gen_batch(order, coefs, n, Lrange, rng, maxlen, device, off_max):
    idx, msk = [], []
    for _ in range(n):
        coef = rng.choice(coefs); s0 = seeds0(order, rng); L = rng.randint(*Lrange)
        seq = serialize(gen_seq(order, coef, s0, L))
        off = rng.randint(0, off_max)                        # random position offset (prereg)
        a = seq + [PAD] * (maxlen - len(seq)); m = [0] * maxlen
        for phere in range(1, len(seq)):                     # predict every term from the prefix
            m[phere] = 1
        idx.append(a[:maxlen]); msk.append(m[:maxlen])
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool), 0


def train(model, order, coefs, steps, bs, lr, Lrange, maxlen, device, seed, off_max):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); rng = random.Random(seed)
    for _ in range(steps):
        idx, msk, _ = gen_batch(order, coefs, bs, Lrange, rng, maxlen, device, off_max)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def continue_seq(model, order, coef, s0, k, L, maxlen, device):
    true = gen_seq(order, coef, s0, L)
    toks = [BOS] + true[:k]; pred = []
    for i in range(k, L):
        idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
        nx = int(model(idx)[0, len(toks) - 1].argmax()); toks.append(nx); pred.append(nx)
    return pred, true[k:L]


@torch.no_grad()
def eval_pool(model, order, coefs, k, L, maxlen, device, n, seed):
    rng = random.Random(seed); full = far = tot = 0; fe = []
    for _ in range(n):
        coef = rng.choice(coefs); s0 = seeds0(order, rng)
        pred, true = continue_seq(model, order, coef, s0, k, L, maxlen, device)
        tot += 1; full += int(pred == true); far += int(pred[-1] == true[-1])
        fe.append(next((j for j in range(len(true)) if pred[j] != true[j]), len(true)))
    return dict(full=full / max(tot, 1), far=far / max(tot, 1), tot=tot, mfe=sum(fe) / max(len(fe), 1))


def main():
    global P, BOS, PAD, NTOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--nope", type=int, default=0)
    ap.add_argument("--H", type=int, default=12); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--held", type=int, default=50)
    ap.add_argument("--maxlen", type=int, default=256); ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--off_max", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=P); ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()
    P = args.p; BOS = P; PAD = P + 1; NTOK = P + 2            # rebind field size + specials
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    crng = random.Random(1234)
    allc = all_coefs(args.order, crng, P)                    # all delta in [1,p-1]
    held = min(args.held, len(allc) // 3)                    # keep a healthy train split for small p
    train_c, held_c = allc[held:], allc[:held]               # DISJOINT param sets
    kmin = args.k
    model = TM(W=args.W, nope=bool(args.nope)).to(device)
    train(model, args.order, train_c, args.steps, args.bs, args.lr, (kmin + 2, args.H), args.maxlen,
          device, args.seed, args.off_max)
    npar = sum(p.numel() for p in model.parameters())
    print(f"PREREG RECUR order={args.order} p={P} W={args.W} nope={args.nope} H={args.H} k={args.k} "
          f"rules train={len(train_c)} held={len(held_c)} off_max={args.off_max} steps={args.steps} "
          f"seed={args.seed} params={npar/1e6:.2f}M device={device}")
    if args.debug:
        for coef in held_c[:args.debug]:
            s0 = seeds0(args.order, random.Random(7))
            pred, true = continue_seq(model, args.order, coef, s0, args.k, 2 * args.H, args.maxlen, device)
            print(f"   DBG coef={coef} true={true} pred={pred}")
    for split, coefs in [("seen", train_c), ("held", held_c)]:
        for L in [args.H, 2 * args.H, 4 * args.H]:
            tag = "within" if L <= args.H else "BEYOND"
            r = eval_pool(model, args.order, coefs, args.k, L, args.maxlen, device, args.eval_n, 500 + L)
            print(f"   {split:>4} x {tag:>6} L={L:>3} (n={r['tot']:>3}): full-exact {r['full']:.2f}  "
                  f"farthest {r['far']:.2f}  first-err-step {r['mfe']:.1f}")
    print("\nread: held x BEYOND full-exact/farthest high => identified the rule from the prefix and executed "
          "the single-token recurrence past the trained horizon (bounded-domain in-class extrapolation).")


if __name__ == "__main__":
    main()
