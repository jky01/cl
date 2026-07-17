#!/usr/bin/env python3
"""The decoy frontier (codex): when a decoy coordinate makes two mechanisms equally short and equally
good on the training environment, description length CANNOT choose. Only evidence that breaks the
observational equivalence -- multiple environments where the decoy correlation changes while the true
rule stays invariant -- can identify the real mechanism. (torch/GPU, pure lstsq)

Coordinates: x = n/K (grows unboundedly), d = (n mod K)/K (wraps into [0,1)), plus bits.
On the base environment n in [0,K): d == x exactly.  On n >= K: d != x (d wraps, x keeps growing).
Target y = (n/K)^2 = x*x.  So:
  x*x  -> = y everywhere, extrapolates.
  d*d  -> = y on the base env (d=x there) but wrong once d!=x -> does NOT extrapolate.
x*x and d*d have identical base-env predictions, identical support size, identical code length.

Preregistered (codex):
  single-env symmetric MDL      -> NON-identifiable (x*x and d*d tie); tie/chance is the CORRECT result.
  multi-env (d-x correlation broken while y=x^2 invariant) -> x*x becomes identifiable.
Selector = exhaustive singleton-support enumeration, refit by lstsq per environment, ranked by
worst-environment (invariant) fit + support size. OOD sealed until after selection.
"""
import argparse
import itertools
import torch


def build(K, M, device):
    n = torch.arange(0, M, device=device)
    x = (n.float() / K)
    d = ((n % K).float() / K)
    nbits = M.bit_length()
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    feat = torch.cat([x.unsqueeze(1), d.unsqueeze(1), bits], 1)   # [M, 2+nbits]
    y = (n.float() / K) ** 2
    names = ["x", "d"] + [f"b{i}" for i in range(nbits)]
    return feat, y, names


def fit_score(feat, y, idx, i, j):
    f = feat[idx, i] * feat[idx, j]
    if f.std() < 1e-8:                     # degenerate (constant/all-zero) feature: cannot fit x^2
        return torch.tensor([0.0, y[idx].mean()], device=f.device), y[idx].std().item()
    A = torch.stack([f, torch.ones_like(f)], 1)
    ATA = A.T @ A + 1e-8 * torch.eye(2, device=f.device)      # ridge -> well-defined for any rank
    coef = torch.linalg.solve(ATA, A.T @ y[idx].unsqueeze(1)).squeeze(1)
    pred = coef[0] * f + coef[1]
    rmse = ((pred - y[idx]) ** 2).mean().sqrt().item()
    return coef, rmse


def ood_w15(feat, y, coef, i, j, idx):
    f = feat[idx, i] * feat[idx, j]
    pred = coef[0] * f + coef[1]
    return ((pred - y[idx]).abs() <= 0.15 * y[idx].clamp(min=1e-3)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512); ap.add_argument("--M", type=int, default=2048)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, M = args.K, args.M
    feat, y, names = build(K, M, device)
    n = torch.arange(0, M, device=device)
    envA = (n < K)                                  # base: d == x
    envB = (n >= K) & (n < 2 * K)                   # second env: d wraps, x in [1,2) -> d != x
    ood = (n >= 2 * K) & (n < 3 * K)                # sealed OOD
    idxA, idxB, idxO = torch.where(envA)[0], torch.where(envB)[0], torch.where(ood)[0]

    pairs = list(itertools.combinations_with_replacement(range(len(names)), 2))

    def rank(train_idxs, label):
        rows = []
        for (i, j) in pairs:
            # worst-environment (invariant) RMSE across the given training environments
            worst = 0.0; coef_last = None
            for idx in train_idxs:
                coef, r = fit_score(feat, y, idx, i, j)
                worst = max(worst, r); coef_last = coef
            # refit on pooled for the committed coefficients
            pooled = torch.cat(train_idxs)
            coef, _ = fit_score(feat, y, pooled, i, j)
            rows.append(((i, j), worst, coef))
        rows.sort(key=lambda r: r[1])
        print(f"\n[{label}]  top candidates by worst-env RMSE (support size all =1 coord/factor):")
        print(f"  {'support':>8} {'worstRMSE':>10} {'OOD[2,3)_w15':>12}")
        for (i, j), worst, coef in rows[:6]:
            oo = ood_w15(feat, y, coef, i, j, idxO)
            tag = "  <- x*x" if (i, j) == (0, 0) else ("  <- d*d (decoy)" if (i, j) == (1, 1) else "")
            print(f"  {names[i]+'*'+names[j]:>8} {worst:>10.4f} {oo:>12.3f}{tag}")
        # explicit x*x vs d*d
        def get(pi, pj):
            return next(r for r in rows if r[0] == (pi, pj))
        xx, dd = get(0, 0), get(1, 1)
        print(f"  x*x worstRMSE={xx[1]:.4f}  d*d worstRMSE={dd[1]:.4f}  "
              f"-> {'TIE (non-identifiable)' if abs(xx[1]-dd[1])<1e-4 else ('x*x wins' if xx[1]<dd[1] else 'd*d wins')}")

    print(f"device={device} DECOY frontier  K={K} M={M}; x=n/K, d=(n mod K)/K (d==x on base env)")
    rank([idxA], "single-env (base only)")
    rank([idxA, idxB], "multi-env (base + shifted, breaks d=x)")
    print("\nPreregistered: single-env -> x*x and d*d TIE (description length cannot choose; correct). "
          "multi-env -> x*x uniquely wins (invariant across environments) and extrapolates, d*d fails. "
          "=> what identifies the rule beyond MDL is EVIDENCE THAT BREAKS OBSERVATIONAL EQUIVALENCE "
          "(multi-environment invariance), exactly as codex framed.")


if __name__ == "__main__":
    main()
