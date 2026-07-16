#!/usr/bin/env python3
"""Fold v2: is the fold wall real, or an artifact of over-rigid A-logit preservation? (torch/GPU)

Fold v1 (s3/toy_fold.py) at h=128 gave a hard A-xor-B Pareto (best min ~0.43) even though
joint=1.0 proves a shared union solution EXISTS. codex's diagnosis: exact-logit-MSE preservation
may be strictly stronger than "keep A's knowledge", and joint-representable != A-teacher-logits+B
simultaneously satisfiable. This script runs the decisive controls.

Preservation modes for the fold (slot-free shared student, private slots deleted):
  labels : rehearse A's true labels + B teacher  -> path-reachability ORACLE (upper bound)
  mse    : alpha * MSE(A logits, A-only-teacher logits) + B teacher   (v1's rigid penalty)
  ce     : alpha * CE(A, A pseudo-labels)          + B teacher   (soft: keep argmax, allow drift)

Inits: A-init (warm start at A solution) and fresh-init (separate optimization / path control).

Key control: JOINT WITNESS compatibility. Train a fresh model on A∪B TRUE labels (should hit ~1.0),
then measure its MSE to the A-only teacher logits. If the witness fits A labels but has large A-logit
MSE, the mse-fold target is INCOMPATIBLE with a B-fitting solution -> wall is objective, not capacity.

Metrics per fold: accA, accB, A-logit drift (MSE to teacher), A mean top1-top2 margin.
"""
import argparse
import collections
import torch
import torch.nn.functional as F


def make(seed, N, C, d, h, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = torch.randn(N, d, generator=g).to(device)
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


def fit_labels(E, P, keys, y, epochs, lr, params=None):
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


@torch.no_grad()
def margin(E, P, keys):
    lg = fwd(E, P, keys)
    top2 = lg.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).mean().item()


def cloneP(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def fold(E, Pinit, A, B, y, teacherB, A_logits, A_pseudo, mode, alpha, epochs, lr):
    P = cloneP(Pinit)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        lossB = F.cross_entropy(fwd(E, P, B), y[B])          # B via true labels == teacher argmax
        if mode == "labels":
            lossA = F.cross_entropy(fwd(E, P, A), y[A])
        elif mode == "mse":
            lossA = F.mse_loss(fwd(E, P, A), A_logits)
        elif mode == "ce":
            lossA = F.cross_entropy(fwd(E, P, A), A_pseudo)
        else:
            lossA = torch.zeros((), device=E.device)
        (lossB + alpha * lossA).backward()
        opt.step()
    return P


def run(seed, args, device):
    N, C, d, h = args.N, args.C, args.d, args.h
    E, P0, y = make(seed, N, C, d, h, device)
    keys = torch.arange(N, device=device)
    A = keys[keys % 2 == 1]
    B = keys[keys % 2 == 0]

    # A into shared weights
    P = cloneP(P0)
    fit_labels(E, P, A, y, args.epochs, args.lr)
    with torch.no_grad():
        A_logits = fwd(E, P, A).detach()
        A_pseudo = A_logits.argmax(1)
    A_margin0 = margin(E, P, A)

    # B into private slots (pre-fold upper bound)
    slot = torch.zeros(N, h, device=device, requires_grad=True)
    opt = torch.optim.Adam([slot], lr=args.lr)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(E, P, B, slot), y[B]).backward()
        opt.step()
    with torch.no_grad():
        teacherB = fwd(E, P, B, slot).detach()
    pre = (acc(E, P, A, y, slot), acc(E, P, B, y, slot))

    # joint witness on A∪B TRUE labels (fresh) + its compatibility with the A teacher logits
    Ej, Pj, _ = make(seed, N, C, d, h, device)   # same E,y (same seed) fresh weights
    Pj = cloneP(Pj)
    fit_labels(E, Pj, keys, y, args.epochs, args.lr)
    with torch.no_grad():
        witness_accA = acc(E, Pj, A, y)
        witness_accB = acc(E, Pj, B, y)
        witness_Amse = F.mse_loss(fwd(E, Pj, A), A_logits).item()   # KEY compatibility number

    out = {"pre": pre, "A_margin0": A_margin0,
           "witness": (witness_accA, witness_accB, witness_Amse)}

    # folds
    fresh0 = cloneP(make(seed + 999, N, C, d, h, device)[1])
    for init_tag, Pinit in [("Ainit", P), ("fresh", fresh0)]:
        for mode in ["labels", "mse", "ce"]:
            alphas = [1.0] if mode == "labels" else args.alphas
            for alpha in alphas:
                Pf = fold(E, Pinit, A, B, y, teacherB, A_logits, A_pseudo,
                          mode, alpha, args.fold_epochs, args.lr)
                out[(init_tag, mode, alpha)] = (
                    acc(E, Pf, A, y), acc(E, Pf, B, y),
                    F.mse_loss(fwd(E, Pf, A), A_logits).item(), margin(E, Pf, A))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--fold_epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} N={args.N} C={args.C} d={args.d} h={args.h} "
          f"epochs={args.epochs} fold_epochs={args.fold_epochs} seeds={args.seeds}")

    agg = collections.defaultdict(list)
    for s in range(args.seeds):
        r = run(s, args, device)
        for k, v in r.items():
            agg[k].append(v)

    pre = torch.tensor(agg["pre"]).mean(0).tolist()
    wit = torch.tensor(agg["witness"]).mean(0).tolist()
    m0 = sum(agg["A_margin0"]) / len(agg["A_margin0"])
    print(f"\npre-fold (slots, illegal):  accA={pre[0]:.3f} accB={pre[1]:.3f}  (A margin0={m0:.3f})")
    print(f"joint witness (A∪B true labels): accA={wit[0]:.3f} accB={wit[1]:.3f}  "
          f"A-logit-MSE-to-teacher={wit[2]:.3f}  <- compatibility (large => mse-fold target off-manifold)")
    print(f"\n{'init':>6} {'mode':>7} {'alpha':>6} {'accA':>7} {'accB':>7} {'min':>7} "
          f"{'A_logMSE':>9} {'A_margin':>9}")
    for init_tag in ["Ainit", "fresh"]:
        for mode in ["labels", "mse", "ce"]:
            alphas = [1.0] if mode == "labels" else args.alphas
            for alpha in alphas:
                rows = torch.tensor(agg[(init_tag, mode, alpha)])
                aA, aB, mse, mg = rows.mean(0).tolist()
                print(f"{init_tag:>6} {mode:>7} {alpha:>6.2f} {aA:>7.3f} {aB:>7.3f} "
                      f"{min(aA,aB):>7.3f} {mse:>9.3f} {mg:>9.3f}")


if __name__ == "__main__":
    main()
