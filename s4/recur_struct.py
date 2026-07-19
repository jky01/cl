#!/usr/bin/env python3
"""Structural continual learning: arithmetic (order-1) -> quadratic (order-2) over F_p, single-token
(codex 2026-07-19.22.32.33). Unlike the arbitrary same-family delta partition (which is associative
OWNERSHIP memory, not rule acquisition), the two phases differ by RECURRENCE ORDER, which is
prefix-computable (2nd difference == 0 ?) -> a legitimate, generalizable, analytic routing gate.

Arms (all single-token, prefix-only, no task ID at inference):
  fixed A->B      : train arith, then quad (naive)              -> establish structural interference
  joint A|B       : train arith+quad jointly (ORACLE, diagnostic)-> can this fixed arch hold both at all?
  consolidation   : phase B = quad + pseudo-rehearsal distilling the FROZEN arith teacher on generated
                    arith prefixes (memory-free at deploy; teacher = training-time memory) -> shared-weight
                    integration (the primary scientific arm)
  grown+gate      : freeze arith trunk, add a quad adapter, route by ANALYTIC order detector
                    (adapter ON iff 2nd-diff != 0) -> modular quarantine with exact structural gate
Report A retention / B acquisition / held (unseen deltas of each order) at H/2H/4H, worst-case not mean,
plus detector routing accuracy and parameter/inference-cost accounting.
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recur as R                                             # gen_seq, all_coefs, seeds0, rope


# ---- analytic order detector from a prefix (prefix-computable, exact) ----
def detect_order(prefix, p):
    d1 = [(prefix[i] - prefix[i - 1]) % p for i in range(1, len(prefix))]
    d2 = [(d1[i] - d1[i - 1]) % p for i in range(1, len(d1))]
    return 1 if all(x == 0 for x in d2) else 2                # all 2nd diffs zero => arithmetic


# ---- model: single-token, local window, per-block adapter routed by detected order ----
class Adapter(nn.Module):
    def __init__(s, d, r):
        super().__init__(); s.dn = nn.Linear(d, r); s.up = nn.Linear(r, d)
        nn.init.zeros_(s.up.weight); nn.init.zeros_(s.up.bias)

    def forward(s, x):
        return s.up(F.gelu(s.dn(x)))


class Block(nn.Module):
    def __init__(s, d, h, ff, W, r):
        super().__init__()
        s.h, s.W = h, W
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)
        s.ad = Adapter(d, r); s.use_ad = False

    def forward(s, x, pos, mask, route):
        B, T, Dd = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(Dd, 2)
        q = q.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, Dd // s.h).transpose(1, 2)
        q, k = R.rope(q, pos), R.rope(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dd // s.h)
        i = torch.arange(T, device=x.device); m = mask | (i[None, :] < i[:, None] - s.W + 1)[None, None]
        att = att.masked_fill(m, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, Dd)
        x = x + s.proj(o)
        h = s.f2(F.gelu(s.f1(s.ln2(x))))
        if s.use_ad and route is not None and route.max() > 0:
            h = h + route * s.ad(s.ln2(x))
        return x + h


class Net(nn.Module):
    def __init__(s, ntok, d=192, h=6, nl=4, ff=768, W=8, r=16):
        super().__init__()
        s.emb = nn.Embedding(ntok, d)
        s.blocks = nn.ModuleList([Block(d, h, ff, W, r) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, ntok)

    def forward(s, idx, route=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, pos, mask, route)
        return s.head(s.lnf(x))

    def trunk_params(s):
        return [p for n, p in s.named_parameters() if ".ad." not in n]

    def adapter_params(s):
        return [p for n, p in s.named_parameters() if ".ad." in n]


# ---- data (order-1 arith / order-2 quad), unified single-token serialization ----
def gen_batch(orders_coefs, n, Lrange, p, rng, maxlen, device):
    BOS, PAD = p, p + 1; idx, msk, ords = [], [], []
    for _ in range(n):
        order, coefs = rng.choice(orders_coefs); delta = rng.choice(coefs)
        s0 = [rng.randrange(0, p) for _ in range(order)]
        L = rng.randint(*Lrange); seq = [BOS] + R.gen_seq(order, delta, s0, L)
        a = seq + [PAD] * (maxlen - len(seq)); m = [0] * maxlen
        for q in range(1, len(seq)):
            m[q] = 1
        idx.append(a[:maxlen]); msk.append(m[:maxlen]); ords.append(order)
    return (torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool),
            torch.tensor(ords, device=device))


def route_for(idx, p, device):
    """analytic gate: route=1 for order-2 sequences (2nd diff != 0), 0 for order-1. from the token seq."""
    B = idx.shape[0]; r = torch.zeros(B, 1, 1, device=device)
    seq = idx.tolist()
    for b in range(B):
        row = [t for t in seq[b] if t < p]                    # strip BOS/PAD
        r[b, 0, 0] = 1.0 if (len(row) >= 3 and detect_order(row, p) == 2) else 0.0
    return r


@torch.no_grad()
def self_pool(teacher, order, npool, L, p, maxlen, device, seed=0):
    """teacher SELF-GENERATES old-family support from RANDOM short seeds (no analytic generator, no known
    deltas). Aux state = 'sample order+1 random tokens' = O(1) per family -> recursive-consolidation ready."""
    BOS, PAD = p, p + 1; g = torch.Generator(device=device).manual_seed(seed); kseed = order + 1
    toks = torch.cat([torch.full((npool, 1), BOS, device=device),
                      torch.randint(0, p, (npool, kseed), generator=g, device=device)], 1)
    for _ in range(L - kseed):
        pad = torch.full((npool, maxlen - toks.shape[1]), PAD, device=device)
        nx = teacher(torch.cat([toks, pad], 1)[:, :maxlen])[torch.arange(npool), toks.shape[1] - 1].argmax(-1)
        toks = torch.cat([toks, nx[:, None]], 1)
    idx = torch.cat([toks, torch.full((npool, maxlen - toks.shape[1]), PAD, device=device)], 1)[:, :maxlen]
    msk = torch.zeros(npool, maxlen, dtype=torch.bool, device=device); msk[:, 1:1 + L] = True
    return idx, msk


def train(model, orders_coefs, steps, bs, lr, Lrange, p, maxlen, device, seed, params=None,
          routed=False, ewc=None, teacher=None, distill_batch=None):
    params = params if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01); rng = random.Random(seed)
    for _ in range(steps):
        idx, msk, ords = gen_batch(orders_coefs, bs, Lrange, p, rng, maxlen, device)
        route = (ords == 2).float().view(-1, 1, 1) if routed else None   # train route from known order
        logits = model(idx[:, :-1], route=route); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        if teacher is not None:                               # generative functional replay: distill teacher
            tidx, tmsk = distill_batch(rng, bs)               # closure: analytic-gen / self-pool / B-only
            with torch.no_grad():
                tlog = teacher(tidx[:, :-1])
            slog = model(tidx[:, :-1])
            kl = F.kl_div(F.log_softmax(slog, -1), F.softmax(tlog, -1), reduction="none").sum(-1)
            loss = loss + (kl * tmsk[:, 1:]).sum() / tmsk[:, 1:].sum().clamp(min=1)
        if ewc is not None:
            Fd, star, lam = ewc
            loss = loss + lam * sum((Fd[n] * (pp - star[n]) ** 2).sum() for n, pp in model.named_parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def cont(model, order, delta, s0, k, L, p, maxlen, device, gate=False):
    BOS, PAD = p, p + 1; true = R.gen_seq(order, delta, s0, L); toks = [BOS] + true[:k]; pred = []
    for i in range(k, L):
        route = route_for(torch.tensor([toks], device=device), p, device) if gate else None
        idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
        nx = int(model(idx, route=route)[0, len(toks) - 1].argmax()); toks.append(nx); pred.append(nx)
    return pred, true[k:L]


@torch.no_grad()
def acc(model, order, coefs, k, L, p, maxlen, device, n, seed, gate=False):
    rng = random.Random(seed); full = tot = 0
    for _ in range(n):
        delta = rng.choice(coefs); s0 = [rng.randrange(0, p) for _ in range(order)]
        pred, true = cont(model, order, delta, s0, k, L, p, maxlen, device, gate)
        tot += 1; full += int(pred == true)
    return full / max(tot, 1)


def report(model, tag, A_seen, A_held, B_seen, B_held, k, H, p, maxlen, device, en, gate=False):
    def row(order, coefs):
        return "/".join(f"{acc(model, order, coefs, k, L, p, maxlen, device, en, 500+L, gate):.2f}" for L in [H, 2*H, 4*H])
    print(f"[{tag:>16}] arithA {row(1,A_seen)} (held {row(1,A_held)}) | quadB {row(2,B_seen)} "
          f"(held {row(2,B_held)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=23); ap.add_argument("--stepsA", type=int, default=12000)
    ap.add_argument("--stepsB", type=int, default=12000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--arms", type=str, default="fixed,joint,consol,grown"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = args.p; ntok = p + 2; device = "cuda" if torch.cuda.is_available() else "cpu"
    R.P = p; R.BOS = p; R.PAD = p + 1; R.NTOK = p + 2         # gen_seq/all_coefs read recur's globals
    torch.manual_seed(args.seed); random.seed(args.seed)
    crng = random.Random(1234); deltas = R.all_coefs(1, crng, p)
    nb = len(deltas) // 4
    A_seen, A_held = deltas[nb:], deltas[:nb]                 # arithmetic deltas (order 1)
    crng2 = random.Random(4321); dq = R.all_coefs(2, crng2, p)
    B_seen, B_held = dq[nb:], dq[:nb]                         # quadratic 2nd-diffs (order 2)
    A_oc = [(1, A_seen)]; B_oc = [(2, B_seen)]; Lr = (args.k + 2, args.H); maxlen = 4 * args.H + 16
    print(f"RECUR-STRUCT p={p} arith|seen={len(A_seen)}/held={len(A_held)}| quad|seen={len(B_seen)}/held="
          f"{len(B_held)}| H={args.H} k={args.k} r={args.r} arms={args.arms} device={device}")

    # phase A: arithmetic trunk (shared by the sequential arms)
    base = Net(ntok, W=args.W, r=args.r).to(device)
    train(base, A_oc, args.stepsA, args.bs, args.lr, Lr, p, maxlen, device, args.seed, params=base.trunk_params())
    report(base, "after A (arith)", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n)
    arms = args.arms.split(",")
    def freegpu():
        import gc; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

    if "joint" in arms:                                       # ORACLE capacity diagnostic
        m = Net(ntok, W=args.W, r=args.r).to(device)
        train(m, A_oc + B_oc, args.stepsA + args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed,
              params=m.trunk_params())
        report(m, "joint A|B ORACLE", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m; freegpu()

    if "fixed" in arms:                                       # naive sequential
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        train(m, B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1, params=m.trunk_params())
        report(m, "fixed A->B naive", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m; freegpu()

    def dgen(oc):                                             # analytic-generator distill batch (known deltas)
        return lambda rng, bs: gen_batch(oc, bs, Lr, p, rng, maxlen, device)[:2]

    if "consol" in arms:                                      # teacher + analytic-A-gen prefixes (KL)
        tea = Net(ntok, W=args.W, r=args.r).to(device); tea.load_state_dict(base.state_dict()); tea.eval()
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        train(m, B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1,
              params=m.trunk_params(), teacher=tea, distill_batch=dgen(A_oc))
        report(m, "consol(tea+genA)", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m, tea; freegpu()

    if "consol_Bonly" in arms:                                # ABLATION (2): teacher queried on B inputs only
        tea = Net(ntok, W=args.W, r=args.r).to(device); tea.load_state_dict(base.state_dict()); tea.eval()
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        train(m, B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1,
              params=m.trunk_params(), teacher=tea, distill_batch=dgen(B_oc))
        report(m, "consol(tea+Bonly)", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m, tea; freegpu()

    if "consol_analytic" in arms:                             # ABLATION (3): synthetic A replay + TRUE labels, NO teacher
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        train(m, A_oc + B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1, params=m.trunk_params())
        report(m, "consol(genA labels)", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m; freegpu()

    if "consol_self" in arms:                                 # SELF-generated support (recursive-ready): teacher rolls random seeds
        tea = Net(ntok, W=args.W, r=args.r).to(device); tea.load_state_dict(base.state_dict()); tea.eval()
        pidx, pmsk = self_pool(tea, 1, 4000, args.H, p, maxlen, device, seed=args.seed)   # A=order1 self-support
        def dself(rng, bs):
            j = torch.randint(0, pidx.shape[0], (bs,), device=device); return pidx[j], pmsk[j]
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        train(m, B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1,
              params=m.trunk_params(), teacher=tea, distill_batch=dself)
        report(m, "consol(SELF-gen)", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n); del m, tea; freegpu()

    if "grown" in arms:                                       # frozen arith trunk + quad adapter + analytic gate
        m = Net(ntok, W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        for b in m.blocks:
            b.use_ad = True
        train(m, B_oc, args.stepsB, args.bs, args.lr, Lr, p, maxlen, device, args.seed + 1,
              params=m.adapter_params(), routed=True)
        # exact non-interference invariant on arithmetic (adapter routed off):
        with torch.no_grad():
            ex, _, _ = gen_batch(A_oc, 64, Lr, p, random.Random(7), maxlen, device)
            d0 = base(ex[:, :-1]); d1 = m(ex[:, :-1], route=route_for(ex[:, :-1], p, device))
            inv = (d0 - d1).abs().max().item()
        radd = sum(x.numel() for x in m.adapter_params())
        print(f"   [grown] adapter params={radd/1e3:.1f}K  arith non-interference max|Δ|={inv:.2e}")
        report(m, "grown+gate", A_seen, A_held, B_seen, B_held, args.k, args.H, p, maxlen, device, args.eval_n, gate=True)
    print("\nread: fixed A->B forgets arith; joint ORACLE = is capacity even the issue; consolidation = "
          "shared-weight retention; grown+gate = modular quarantine (arith Δ≈0 exact) — necessity needs "
          "joint-oracle-fails-while-grown-succeeds.")


if __name__ == "__main__":
    main()
