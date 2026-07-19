#!/usr/bin/env python3
"""Continual acquire-retain on the single-token recurrence substrate (codex 2026-07-19.07.49.58, Q3).

The substrate (s4/recur.py) free-runs TRAINED arithmetic deltas to 4x horizon. Here we test whether new
rule-instances can enter the weights WITHOUT erasing old ones, under prefix-only selection, no replay, no
joint retrain. TWO scoreboards kept separate (codex): (1) INSTANCE retention -- exact rollouts on trained
delta sets A and B from unseen seeds; (2) FAMILY generalization -- held-out deltas (never trained) before
and after B (expected ~0; kept so catalogue retention is never silently renamed rule retention).

phase A: train delta set A.  phase B: train DISJOINT set B, no A examples/replay.  Arms: naive (all params)
vs ewc (model-sampled diagonal Fisher penalty to phase-A weights = fixed-capacity retention).  Growth arm
is deferred until a fixed-capacity stability-plasticity failure is established.
"""
import argparse, sys, os, random, copy
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recur as R


def set_field(p):
    R.P = p; R.BOS = p; R.PAD = p + 1; R.NTOK = p + 2


def train_phase(model, coefs, steps, bs, lr, Lrange, maxlen, device, seed, wd, ewc=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); rng = random.Random(seed)
    for _ in range(steps):
        idx, msk, _ = R.gen_batch(1, coefs, bs, Lrange, rng, maxlen, device, 0)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, R.NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        if ewc is not None:
            Fd, star, lam = ewc
            loss = loss + lam * sum((Fd[n] * (p - star[n]) ** 2).sum() for n, p in model.named_parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def fisher(model, coefs, bs, Lrange, maxlen, device, nb=60):
    Fd = {n: torch.zeros_like(p) for n, p in model.named_parameters()}; rng = random.Random(999)
    for _ in range(nb):
        idx, msk, _ = R.gen_batch(1, coefs, bs, Lrange, rng, maxlen, device, 0)
        logits = model(idx[:, :-1]); mt = msk[:, 1:]
        logp = F.log_softmax(logits, -1)
        with torch.no_grad():
            samp = torch.multinomial(logp.exp().reshape(-1, R.NTOK), 1).view(logits.shape[:2])
        nll = F.nll_loss(logp.reshape(-1, R.NTOK), samp.reshape(-1), reduction="none").view(mt.shape)
        loss = (nll * mt).sum() / mt.sum().clamp(min=1)
        model.zero_grad(); loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                Fd[n] += p.grad.detach() ** 2 / nb
    return Fd


def report(model, tag, sets, k, H, maxlen, device, en):
    row = []
    for name, coefs in sets:
        rr = [R.eval_pool(model, 1, coefs, k, L, maxlen, device, en, 500 + L)["full"] for L in [H, 2 * H, 4 * H]]
        row.append(f"{name}(n={len(coefs)}) H/2H/4H " + "/".join(f"{a:.2f}" for a in rr))
    print(f"[{tag:>16}] " + "  |  ".join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=23); ap.add_argument("--nA", type=int, default=8)
    ap.add_argument("--nB", type=int, default=8); ap.add_argument("--stepsA", type=int, default=12000)
    ap.add_argument("--stepsB", type=int, default=12000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--H", type=int, default=12); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lam", type=float, default=2000.0); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval_n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_field(args.p)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    crng = random.Random(1234); allc = R.all_coefs(1, crng, R.P)
    A, B, held = allc[:args.nA], allc[args.nA:args.nA + args.nB], allc[args.nA + args.nB:]
    Lr = (args.k + 2, args.H); maxlen = 256; sets = [("A", A), ("B", B), ("held", held)]
    print(f"RECUR-CONTINUAL p={args.p} |A|={len(A)} |B|={len(B)} |held|={len(held)} H={args.H} k={args.k} "
          f"lam={args.lam} stepsA/B={args.stepsA}/{args.stepsB} device={device}")

    base = R.TM(W=args.W).to(device)
    train_phase(base, A, args.stepsA, args.bs, args.lr, Lr, maxlen, device, args.seed, args.wd)
    report(base, "after A", sets, args.k, args.H, maxlen, device, args.eval_n)
    star = {n: p.detach().clone() for n, p in base.named_parameters()}
    Fd = fisher(base, A, args.bs, Lr, maxlen, device)
    fv = torch.cat([v.flatten() for v in Fd.values()])
    print(f"   phase-A Fisher: median={fv.median():.2e} frac~0={(fv<1e-12).float().mean():.2f} norm={fv.sum().sqrt():.2e}")

    for arm in ["naive", "ewc"]:
        m = R.TM(W=args.W).to(device); m.load_state_dict(base.state_dict())
        train_phase(m, B, args.stepsB, args.bs, args.lr, Lr, maxlen, device, args.seed + 1, args.wd,
                    ewc=(Fd, star, args.lam) if arm == "ewc" else None)
        report(m, f"after B/{arm}", sets, args.k, args.H, maxlen, device, args.eval_n)
    print("\nread: INSTANCE retention = A stays high after B. naive likely forgets A (A->0, B->1); ewc is the "
          "fixed-capacity test. FAMILY generalization = held column (expected ~0 both = catalogue not rule).")


if __name__ == "__main__":
    main()
