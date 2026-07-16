#!/usr/bin/env python3
"""Referee-budget sweep: how much OLD data must the fold retain to keep A? (torch/GPU)

Fold v2 (s3/toy_fold2.py) showed a decision-preserving CE self-distillation fold recovers 1.0/1.0
at h=128 -- but using ALL of A's inputs as the preservation referee (oracle). This measures the
DEPLOYABLE gap: shrink the retained A-input budget and watch A retention.

For each budget fraction we CE-distill on only that subset of A's inputs (A's own argmax as targets,
label-free for old data) while fitting B labels, slots deleted. accA is measured on ALL of A, so a
subset that fails to generalize to the rest of A shows up as loss.

Selection strategies for the retained subset:
  random    : uniform subset of A.
  lowmargin : the A inputs with the smallest A-teacher margin (nearest the decision boundary =
              most fragile under B) -- connects the referee to a surprise/fragility criterion.

On incompressible random-label facts we expect accA ~ budget (each fact independent -> no
generalization from the subset). That floor is the point: it shows when a small referee CANNOT work,
motivating a structured/compressible task where surprise-selected retention should beat random.
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


def fit_labels(E, P, keys, y, epochs, lr):
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
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
def margins(E, P, keys):
    lg = fwd(E, P, keys)
    top2 = lg.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def cloneP(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def ce_fold(E, Pinit, ref, A_pseudo_ref, B, y, alpha, epochs, lr):
    """CE self-distill on referee inputs `ref` (targets A_pseudo_ref) + fit B labels."""
    P = cloneP(Pinit)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fwd(E, P, B), y[B])
        if ref.numel() > 0:
            loss = loss + alpha * F.cross_entropy(fwd(E, P, ref), A_pseudo_ref)
        loss.backward()
        opt.step()
    return P


def run(seed, args, device):
    N, C, d, h = args.N, args.C, args.d, args.h
    E, P0, y = make(seed, N, C, d, h, device)
    keys = torch.arange(N, device=device)
    A = keys[keys % 2 == 1]
    B = keys[keys % 2 == 0]

    P = cloneP(P0)
    fit_labels(E, P, A, y, args.epochs, args.lr)
    with torch.no_grad():
        A_pseudo_all = fwd(E, P, A).argmax(1)          # A's own predictions (== labels here, acc~1)
    A_marg = margins(E, P, A)                          # per-A-key fragility
    order_lowmarg = A[torch.argsort(A_marg)]           # ascending margin

    out = {}
    for budget in args.budgets:
        k = int(round(budget * A.numel()))
        sels = ["random", "lowmargin"] if 0 < k < A.numel() else ["all"]
        for sel in sels:
            if sel == "random":
                idx = torch.randperm(A.numel(), device=device)[:k]
                ref = A[idx]
                pseudo = A_pseudo_all[idx]
            elif sel == "lowmargin":
                ref = order_lowmarg[:k]
                # map ref back to positions in A to fetch pseudo
                pos = torch.argsort(A_marg)[:k]
                pseudo = A_pseudo_all[pos]
            else:  # all (budget 1.0) or none (budget 0.0)
                if k == 0:
                    ref = A[:0]; pseudo = A_pseudo_all[:0]
                else:
                    ref = A; pseudo = A_pseudo_all
            Pf = ce_fold(E, P, ref, pseudo, B, y, args.alpha, args.fold_epochs, args.lr)
            out[(budget, sel)] = (acc(E, Pf, A, y), acc(E, Pf, B, y),
                                  margins(E, Pf, A).mean().item())
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
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.02, 0.0])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} N={args.N} C={args.C} h={args.h} alpha={args.alpha} "
          f"fold_epochs={args.fold_epochs} seeds={args.seeds} task=random-labels")
    print(f"{'budget':>7} {'sel':>10} {'accA_all':>9} {'accB':>7} {'min':>7} {'A_margin':>9}")
    agg = collections.defaultdict(list)
    for s in range(args.seeds):
        for k, v in run(s, args, device).items():
            agg[k].append(v)
    for budget in args.budgets:
        k = int(round(budget * (args.N // 2)))
        sels = ["random", "lowmargin"] if 0 < k < (args.N // 2) else ["all"]
        for sel in sels:
            rows = torch.tensor(agg[(budget, sel)])
            aA, aB, mg = rows.mean(0).tolist()
            print(f"{budget:>7.2f} {sel:>10} {aA:>9.3f} {aB:>7.3f} {min(aA,aB):>7.3f} {mg:>9.3f}")


if __name__ == "__main__":
    main()
