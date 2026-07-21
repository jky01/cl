#!/usr/bin/env python3
"""o2 sequential-interference DIAGNOSIS (codex 2026-07-21.17.40.29): o2 is representable at d=192 (joint
1.0) yet the sequential+oracle-replay protocol leaves it at ~0.3-0.5. MEASURE why before trying fixes.

Protocol: train o1 (phase A) -> then phase B = learn o2 + oracle-replay o1, WHILE logging every K steps:
  - o2 free-run exact H/2H/4H and o1 free-run exact (retention trajectory);
  - new(o2) vs old(o1-replay) gradient COSINE and norm ratio (global) — are they opposed / imbalanced?
  - the o2 loss trajectory (gradual forgetting vs abrupt boundary damage vs failed recovery).
Then a RECOVERY control: from the failed sequential endpoint, run a short BALANCED-oracle recovery (equal
o1+o2 true support) and see if o2 returns fast to ~1.0 (knowledge nearby/suppressed => optimizer/path) or
slowly (representation drift / basin separation). Joint/oracle data are adjudicators, never the method.
"""
import argparse, sys, os, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recur as R
from recur_recursive import gen_batch, acc


def grads(model, idx, msk, p):
    logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
    lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
    loss = (lt * mt).sum() / mt.sum().clamp(min=1)
    g = torch.autograd.grad(loss, [q for q in model.parameters() if q.requires_grad], retain_graph=False)
    return torch.cat([x.flatten() for x in g]), loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=23); ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--stepsA", type=int, default=12000); ap.add_argument("--stepsB", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--W", type=int, default=8); ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--recover", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = args.p; R.P = p; R.BOS = p; R.PAD = p + 1; R.NTOK = p + 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    ff = 4 * args.d; h = max(2, args.d // 32); Lr = (args.k + 2, args.H); maxlen = 4 * args.H + 16
    co = {o: R.all_coefs(o, random.Random(100 + o), p) for o in (1, 2)}
    seen = {o: co[o][len(co[o]) // 4:] for o in (1, 2)}
    print(f"RECUR-DIAG o1->o2 p={p} d={args.d} params-scale H={args.H} stepsA/B={args.stepsA}/{args.stepsB} "
          f"seed={args.seed} device={device}")

    model = R.TM(d=args.d, h=h, ff=ff, W=args.W).to(device)
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for _ in range(args.stepsA):                              # phase A: o1
        idx, msk = gen_batch(1, seen[1], args.bs, Lr, p, rng, maxlen, device)
        _, _ = None, None
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    a1 = acc(model, 1, seen[1], args.k, 2*args.H, p, maxlen, device, 100, 700)
    print(f"   after A: o1(2H) exact {a1:.2f}"); model.train()

    # phase B: o2 + oracle-replay o1, logging gradient conflict + retention trajectory
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for it in range(args.stepsB):
        ni, nm = gen_batch(2, seen[2], args.bs, Lr, p, rng, maxlen, device)   # new o2
        oi, om = gen_batch(1, seen[1], args.bs, Lr, p, rng, maxlen, device)   # oracle-replay o1
        if (it + 1) % args.log_every == 0:
            gnew, lnew = grads(model, ni, nm, p); gold, lold = grads(model, oi, om, p)
            cos = F.cosine_similarity(gnew[None], gold[None]).item()
            nr = (gnew.norm() / gold.norm().clamp(min=1e-9)).item()
            model.eval()
            o2 = "/".join(f"{acc(model,2,seen[2],args.k,L,p,maxlen,device,80,500+L):.2f}" for L in [args.H,2*args.H,4*args.H])
            o1 = "/".join(f"{acc(model,1,seen[1],args.k,L,p,maxlen,device,80,500+L):.2f}" for L in [args.H,2*args.H,4*args.H])
            print(f"   [B it {it+1:>6}] o2 {o2} | o1 {o1} | grad cos {cos:+.3f} norm(new/old) {nr:.2f} "
                  f"| loss new {lnew:.4f} old {lold:.4f}", flush=True); model.train()
        # combined step (equal weight = the current consolidation objective)
        for bi, bm in ((ni, nm), (oi, om)):
            logits = model(bi[:, :-1]); tgt = bi[:, 1:]; mt = bm[:, 1:]
            lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
            loss = (lt * mt).sum() / mt.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()
    o2e = "/".join(f"{acc(model,2,seen[2],args.k,L,p,maxlen,device,150,500+L):.2f}" for L in [args.H,2*args.H,4*args.H])
    print(f"   sequential endpoint: o2 {o2e}"); model.train()

    # RECOVERY control: short balanced-oracle recovery from the failed endpoint
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for it in range(args.recover):
        for o in (1, 2):
            bi, bm = gen_batch(o, seen[o], args.bs, Lr, p, rng, maxlen, device)
            logits = model(bi[:, :-1]); tgt = bi[:, 1:]; mt = bm[:, 1:]
            lt = F.cross_entropy(logits.reshape(-1, p + 2), tgt.reshape(-1), reduction="none").view(tgt.shape)
            loss = (lt * mt).sum() / mt.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (it + 1) % (args.recover // 4) == 0:
            model.eval()
            o2 = "/".join(f"{acc(model,2,seen[2],args.k,L,p,maxlen,device,80,500+L):.2f}" for L in [args.H,2*args.H,4*args.H])
            print(f"   [recover it {it+1:>5}] o2 {o2}", flush=True); model.train()
    print("\nread: grad cos<<0 => opposed (need conflict projection); norm ratio skewed => rebalance loss; "
          "FAST recovery to ~1.0 => knowledge suppressed/optimizer-conditioning (path), SLOW => representation "
          "drift/basin separation. oracle already fails o2 => raising self-replay fidelity won't fix it.")


if __name__ == "__main__":
    main()
