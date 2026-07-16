#!/usr/bin/env python3
"""Consolidation (fold) experiment at a tight-but-feasible width (torch/GPU).

Regime (from s3/toy_capacity_sweep.py): at h=128, the joint oracle = 1.0, so a shared
substrate that holds A U B provably EXISTS, yet naive sequential forgets A (~0.92).
So forgetting here is a path/optimization pathology, not infeasibility.

Question: can we CONSOLIDATE the `local` pre-fold state (A in shared W1/W2, B in private
additive hidden slots) into the shared weights ALONE (slots deleted) while keeping A?

Stages:
  1. Learn A into shared W1/W2 (frozen random address).
  2. Learn B into private per-key additive hidden slots, shared weights frozen -> pre-fold
     upper bound (A~1, B~1) but with unbounded per-key memory (illegal at inference).
  3. FOLD: train slot-free shared student to match B while preserving A, under a trust
     region. Two variants of the referee/search:
       - full-A logit/margin preservation penalty (oracle diagnostic), A-init.
       - same objective but from several FRESH inits (path-vs-representation control).
     Sweep the A-preservation weight -> Pareto (accA vs accB), slots deleted.

All A keys are used as the oracle referee here (finite synthetic universe); this is a
FEASIBILITY diagnostic, not a deployable CL method. Later rounds reduce the referee budget.
"""
import argparse
import torch
import torch.nn.functional as F


def make(seed, N, C, d, h, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = torch.randn(N, d, generator=g).to(device)                       # frozen address
    P = {
        "W1": (torch.randn(d, h, generator=g) / d ** 0.5).to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }
    y = torch.randint(0, C, (N,), generator=g).to(device)
    return E, P, y


def fwd(E, P, keys, slot=None):
    pre = E[keys] @ P["W1"] + P["b1"]
    if slot is not None:
        pre = pre + slot[keys]
    return torch.relu(pre) @ P["W2"] + P["b2"]


def fit(E, P, keys, y, epochs, lr, params=None):
    params = params or [P[n] for n in P]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(E, P, keys), y[keys]).backward()
        opt.step()


@torch.no_grad()
def acc(E, P, keys, y, slot=None):
    if keys.numel() == 0:
        return float("nan")
    return (fwd(E, P, keys, slot).argmax(1) == y[keys]).float().mean().item()


def clonePW(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def fold(E, P_init, A, B, y, teacherB_logits, A_ref_logits, alpha, epochs, lr, from_init=None):
    """Train slot-free shared student: fit B (to teacher logits) + preserve A logits (weight alpha)."""
    P = clonePW(P_init) if from_init is None else from_init
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        lossB = F.cross_entropy(fwd(E, P, B), y[B])
        # A preservation: match stored A logits (distillation / trust region)
        lossA = F.mse_loss(fwd(E, P, A), A_ref_logits)
        (lossB + alpha * lossA).backward()
        opt.step()
    return P


def run(seed, args, device):
    N, C, d, h = args.N, args.C, args.d, args.h
    E, P0, y = make(seed, N, C, d, h, device)
    keys = torch.arange(N, device=device)
    A = keys[keys % 2 == 1]
    B = keys[keys % 2 == 0]

    # stage 1: A into shared weights
    P = clonePW(P0)
    fit(E, P, A, y, args.epochs, args.lr)
    accA_afterA = acc(E, P, A, y)
    with torch.no_grad():
        A_ref_logits = fwd(E, P, A).detach()          # A trust-region target

    # stage 2: B into private slots, shared weights frozen (pre-fold upper bound)
    slot = torch.zeros(N, h, device=device, requires_grad=True)
    fit_params = [slot]
    opt = torch.optim.Adam(fit_params, lr=args.lr)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(E, P, B, slot), y[B]).backward()
        opt.step()
    pre_accA = acc(E, P, A, y, slot)                   # A still ~1 (shared frozen)
    pre_accB = acc(E, P, B, y, slot)
    with torch.no_grad():
        teacherB = fwd(E, P, B, slot).detach()

    # stage 3: fold -> slot-free shared student, sweep alpha, A-init
    results = []
    for alpha in args.alphas:
        Pf = fold(E, P, A, B, y, teacherB, A_ref_logits, alpha, args.fold_epochs, args.lr)
        results.append(("Ainit", alpha, acc(E, Pf, A, y), acc(E, Pf, B, y)))

    # path-vs-representation control: fold from FRESH inits (no A-solution warm start)
    best_fresh = None
    for r in range(args.fresh_restarts):
        Ef, Pfresh, _ = make(seed + 104729 * (r + 1), N, C, d, h, device)
        Pfresh = {k: v for k, v in Pfresh.items()}       # fresh weights, SAME frozen address E? use E
        Pf = fold(E, P, A, B, y, teacherB, A_ref_logits, args.alpha_fresh,
                  args.fold_epochs, args.lr, from_init=clonePW(Pfresh))
        m = min(acc(E, Pf, A, y), acc(E, Pf, B, y))
        cand = ("fresh", args.alpha_fresh, acc(E, Pf, A, y), acc(E, Pf, B, y))
        if best_fresh is None or m > min(best_fresh[2], best_fresh[3]):
            best_fresh = cand

    return accA_afterA, pre_accA, pre_accB, results, best_fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--fold_epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--alpha_fresh", type=float, default=1.0)
    ap.add_argument("--fresh_restarts", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} N={args.N} C={args.C} d={args.d} h={args.h} epochs={args.epochs} "
          f"fold_epochs={args.fold_epochs} lr={args.lr} seeds={args.seeds}")

    import collections
    ainit = collections.defaultdict(list)
    fresh = []
    pre = []
    for s in range(args.seeds):
        accA_afterA, pre_accA, pre_accB, results, best_fresh = run(s, args, device)
        pre.append((accA_afterA, pre_accA, pre_accB))
        for tag, alpha, aA, aB in results:
            ainit[alpha].append((aA, aB))
        fresh.append(best_fresh)

    pre = torch.tensor(pre).mean(0).tolist()
    print(f"\npre-fold (slots present, illegal):  accA_afterA={pre[0]:.3f}  "
          f"accA_withB={pre[1]:.3f}  accB={pre[2]:.3f}")
    print(f"\nFOLD (slots deleted, shared-only) — A-init, Pareto over alpha:")
    print(f"{'alpha':>8} {'accA':>8} {'accB':>8} {'min':>8}")
    for alpha in args.alphas:
        rows = torch.tensor(ainit[alpha])
        aA, aB = rows.mean(0).tolist()
        print(f"{alpha:>8.2f} {aA:>8.3f} {aB:>8.3f} {min(aA,aB):>8.3f}")
    fa = torch.tensor([[f[2], f[3]] for f in fresh]).mean(0).tolist()
    print(f"\nfresh-init control (alpha={args.alpha_fresh}, best of {args.fresh_restarts}): "
          f"accA={fa[0]:.3f} accB={fa[1]:.3f} min={min(fa[0],fa[1]):.3f}")
    print("interpretation: fresh-init good but A-init stuck => path/optimization wall; "
          "both stuck => representational (contradicted by joint=1.0); A-init good => fold exists.")


if __name__ == "__main__":
    main()
