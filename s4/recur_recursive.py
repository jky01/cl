#!/usr/bin/env python3
"""RECURSIVE continual consolidation (codex 2026-07-20.04.39.14, centerpiece): can a single small model keep
accumulating recurrence FAMILIES into shared weights while reconstructing its OWN past support, so the only
growing state is one integer per structural family (the order), not a per-instance catalogue or a stored
generator?

Families over F_p, single token: order-1 arith / order-2 quad / order-3 cubic (const k-th difference),
disjoint delta sets. Protocol: phase k learns family k AND rehearses all PAST families by SELF-GENERATION
-- sample (order+1) random tokens, roll the CURRENT model forward to produce support, train (hard-label
self-distillation) on it. Only the current model M_k is kept between phases; past teachers are discarded.
Aux state = the set {orders seen} (one int per family). Report each family's exact free-run retain/acquire
at H/2H/4H after each phase, plus SELF-GEN SUPPORT fidelity (does the model's rolled support match the true
recurrence) so compounding drift is visible.
"""
import argparse, sys, os, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recur as R


def gen_batch(order, coefs, n, Lrange, p, rng, maxlen, device):
    BOS, PAD = p, p + 1; idx, msk = [], []
    for _ in range(n):
        delta = rng.choice(coefs); s0 = [rng.randrange(0, p) for _ in range(order)]
        L = rng.randint(*Lrange); seq = [BOS] + R.gen_seq(order, delta, s0, L)
        a = seq + [PAD] * (maxlen - len(seq)); m = [0] * maxlen
        for q in range(1, len(seq)):
            m[q] = 1
        idx.append(a[:maxlen]); msk.append(m[:maxlen])
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool)


@torch.no_grad()
def self_pool(model, order, npool, L, p, maxlen, device, seed):
    """model self-generates order-k support from RANDOM (order+1)-token seeds (no analytic gen, no deltas)."""
    BOS, PAD = p, p + 1; g = torch.Generator(device=device).manual_seed(seed); ks = order + 1
    toks = torch.cat([torch.full((npool, 1), BOS, device=device),
                      torch.randint(0, p, (npool, ks), generator=g, device=device)], 1)
    for _ in range(L - ks):
        pad = torch.full((npool, maxlen - toks.shape[1]), PAD, device=device)
        nx = model(torch.cat([toks, pad], 1)[:, :maxlen])[torch.arange(npool), toks.shape[1] - 1].argmax(-1)
        toks = torch.cat([toks, nx[:, None]], 1)
    idx = torch.cat([toks, torch.full((npool, maxlen - toks.shape[1]), PAD, device=device)], 1)[:, :maxlen]
    msk = torch.zeros(npool, maxlen, dtype=torch.bool, device=device); msk[:, 1:1 + L] = True
    return idx, msk


def train_phase(model, new_order, new_coefs, past_pools, steps, bs, lr, Lrange, p, maxlen, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); rng = random.Random(seed)
    for _ in range(steps):
        idx, msk = gen_batch(new_order, new_coefs, bs, Lrange, p, rng, maxlen, device)   # new family (true)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        if past_pools:                                        # self-generative rehearsal of past families
            pidx, pmsk = past_pools; j = torch.randint(0, pidx.shape[0], (bs,), device=device)
            ri, rm = pidx[j], pmsk[j]
            rl = model(ri[:, :-1]); rt = ri[:, 1:]; rmt = rm[:, 1:]
            rlt = F.cross_entropy(rl.reshape(-1, p + 2), rt.reshape(-1), reduction="none").view(rt.shape)
            loss = loss + (rlt * rmt).sum() / rmt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def acc(model, order, coefs, k, L, p, maxlen, device, n, seed):
    BOS, PAD = p, p + 1; rng = random.Random(seed); full = tot = 0
    for _ in range(n):
        delta = rng.choice(coefs); s0 = [rng.randrange(0, p) for _ in range(order)]
        true = R.gen_seq(order, delta, s0, L); toks = [BOS] + true[:k]; pred = []
        for i in range(k, L):
            idx = torch.tensor([toks + [PAD] * (maxlen - len(toks))], device=device)[:, :maxlen]
            nx = int(model(idx)[0, len(toks) - 1].argmax()); toks.append(nx); pred.append(nx)
        tot += 1; full += int(pred == true[k:L])
    return full / max(tot, 1)


@torch.no_grad()
def selfgen_fidelity(model, order, npool, L, p, maxlen, device, seed):
    """fraction of self-generated support sequences that ARE valid order-k recurrences (drift probe)."""
    idx, _ = self_pool(model, order, npool, L, p, maxlen, device, seed); ok = 0
    for row in idx.tolist():
        s = [t for t in row if t < p]
        if len(s) < order + 3:
            continue
        d = [s]                                              # difference table up to (order+1)
        for _o in range(order + 1):
            d.append([(d[-1][i + 1] - d[-1][i]) % p for i in range(len(d[-1]) - 1)])
        # valid order-k: (order+1)-th diff all zero AND order-th diff genuinely nonzero-constant
        ok += int(all(x == 0 for x in d[order + 1]) and any(x != 0 for x in d[order]))
    return ok / npool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=23); ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--W", type=int, default=8); ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--npool", type=int, default=4000)
    ap.add_argument("--orders", type=str, default="1,2,3"); ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--rehearsal", type=str, default="self")  # self | oracle | none
    ap.add_argument("--joint", type=int, default=0)           # 1 = train ALL orders jointly (capacity oracle)
    ap.add_argument("--d", type=int, default=192)             # width, for capacity-frontier sweep
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = args.p; R.P = p; R.BOS = p; R.PAD = p + 1; R.NTOK = p + 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    orders = [int(x) for x in args.orders.split(",")]; Lr = (args.k + 2, args.H); maxlen = 4 * args.H + 16
    ff = 4 * args.d; h = max(2, args.d // 32)
    # disjoint delta sets per order
    crng = random.Random(1234); coefs = {o: R.all_coefs(o, random.Random(100 + o), p) for o in orders}
    seen = {o: coefs[o][len(coefs[o]) // 4:] for o in orders}
    held = {o: coefs[o][:len(coefs[o]) // 4] for o in orders}

    if args.joint:                                            # ADJUDICATOR: is fixed width sufficient at all?
        model = R.TM(d=args.d, h=h, ff=ff, W=args.W).to(device)
        npar = sum(x.numel() for x in model.parameters())
        rng = random.Random(args.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        for _ in range(args.steps * len(orders)):             # match total sequential compute
            o = rng.choice(orders); idx, msk = gen_batch(o, seen[o], args.bs, Lr, p, rng, maxlen, device)
            logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
            lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
            loss = (lt * mt).sum() / mt.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        row = [f"o{po} " + "/".join(f"{acc(model, po, seen[po], args.k, L, p, maxlen, device, args.eval_n, 500+L):.2f}"
                                    for L in [args.H, 2*args.H, 4*args.H]) for po in orders]
        print(f"JOINT-ORACLE p={p} d={args.d} ff={ff} h={h} params={npar/1e6:.2f}M orders={orders} "
              f"steps={args.steps*len(orders)} seed={args.seed}: " + " | ".join(row), flush=True)
        return

    print(f"RECUR-RECURSIVE p={p} orders={orders} rehearsal={args.rehearsal} H={args.H} k={args.k} "
          f"steps={args.steps} npool={args.npool} seed={args.seed} device={device}")

    model = R.TM(W=args.W).to(device)
    learned = []
    for oi, o in enumerate(orders):
        past_pools = None
        if learned and args.rehearsal != "none":             # rehearse ALL past families
            model.eval(); pis, pms = [], []
            for po in learned:
                if args.rehearsal == "self":                 # CURRENT model self-generates past support
                    pi, pm = self_pool(model, po, args.npool, args.H, p, maxlen, device, args.seed + po)
                    fid = selfgen_fidelity(model, po, 400, args.H, p, maxlen, device, args.seed + po)
                    print(f"   [phase {oi+1}] self-gen order-{po} support fidelity vs TRUE recurrence = {fid:.3f}")
                else:                                        # oracle: analytic true support + true labels
                    pi, pm = gen_batch(po, seen[po], args.npool, Lr, p, random.Random(7 + po), maxlen, device)
                pis.append(pi); pms.append(pm)
            past_pools = (torch.cat(pis), torch.cat(pms)); model.train()
        train_phase(model, o, seen[o], past_pools, args.steps, args.bs, args.lr, Lr, p, maxlen, device, args.seed + oi)
        learned.append(o)
        model.eval()
        row = []
        for po in orders:
            tag = "learned" if po in learned else "future"
            rr = "/".join(f"{acc(model, po, seen[po], args.k, L, p, maxlen, device, args.eval_n, 500+L):.2f}"
                          for L in [args.H, 2*args.H, 4*args.H])
            row.append(f"o{po}({tag[:3]}) {rr}")
        print(f"[after phase {oi+1} (order-{o}); aux-state=orders{learned}] " + " | ".join(row), flush=True)
    print("\nread: each learned order stays high (H/2H/4H) after later phases, using only the current model to "
          "regenerate past support (aux state = the small set of orders). Compounding drift shows up as "
          "declining self-gen fidelity and/or declining old-order retention across phases.")


if __name__ == "__main__":
    main()
