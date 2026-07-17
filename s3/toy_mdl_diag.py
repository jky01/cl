#!/usr/bin/env python3
"""Diagnostic (codex-prioritized): does the intended objective PREFER x*x, or does SGD/gates miss it?
Enumerate every singleton-support product candidate on the tiny graph and compare ID fit + extrapolation
BEFORE any stochastic-gate machinery. (torch/GPU, pure least-squares -> fast)

Inputs (no q): input = [n/K, bit_0..bit_{B-1}]. A "singleton-support product" candidate is
feat = input_i * input_j for a pair (i<=j). Fit y = a*feat + b on ID (n in [0,K)) by least squares;
report ID-RMSE and OOD within-15% on shell n/K in [2,3). The intended rule is (i,j)=(0,0) = (n/K)^2.

Reads (codex's four failure modes):
  - (0,0) has the lowest ID loss AND extrapolates -> the objective prefers x*x; if SGD/gates missed it,
    the fix is the gate optimizer, not the code.
  - some bit-supported pair has lower ID loss (and 0 extrap) -> the code/prior is what's wrong; length
    alone lets a shortcut win -> need identifying environments/invariance, not just edge count.
Also a counterfactual: for the best-ID candidate, report its extrap (does the ID winner generalize?).
"""
import argparse
import itertools
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M, K = args.M, args.K
    nbits = M.bit_length()
    names = ["x"] + [f"b{i}" for i in range(nbits)]

    print(f"device={device} MDL-DIAG enumerate singleton-support products; K={K} M={M} seeds={args.seeds}")
    print("candidate feat = input_i * input_j ; fit a*feat+b on ID (n<K) by lstsq; "
          "report ID-RMSE (norm) and OOD[2,3) within15%")

    agg = {}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        n_id = torch.arange(0, K, device=device)
        n_ood = torch.arange(2 * K, 3 * K, device=device)

        def inp(n):
            x = (n.float() / K).unsqueeze(1)
            bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
            return torch.cat([x, bits], 1)                       # [N, 1+nbits]

        Xid, Xood = inp(n_id), inp(n_ood)
        yid = (n_id.float() / K) ** 2
        yood = (n_ood.float() / K) ** 2

        for i, j in itertools.combinations_with_replacement(range(1 + nbits), 2):
            fid = Xid[:, i] * Xid[:, j]
            food = Xood[:, i] * Xood[:, j]
            A = torch.stack([fid, torch.ones_like(fid)], 1)
            coef = torch.linalg.lstsq(A, yid.unsqueeze(1)).solution.squeeze(1)
            pid = coef[0] * fid + coef[1]
            pood = coef[0] * food + coef[1]
            rmse = ((pid - yid) ** 2).mean().sqrt().item()
            w15 = ((pood - yood).abs() <= 0.15 * yood.clamp(min=1e-3)).float().mean().item()
            agg.setdefault((i, j), [0.0, 0.0])
            agg[(i, j)][0] += rmse; agg[(i, j)][1] += w15

    rows = [((i, j), v[0] / args.seeds, v[1] / args.seeds) for (i, j), v in agg.items()]
    rows.sort(key=lambda r: r[1])                                # by ID-RMSE ascending
    print(f"\n{'support':>10} {'ID_RMSE':>9} {'OOD[2,3)_w15':>12}")
    for (i, j), rmse, w15 in rows[:10]:
        tag = "  <- x*x (intended)" if (i, j) == (0, 0) else ""
        print(f"{names[i]+'*'+names[j]:>10} {rmse:>9.4f} {w15:>12.3f}{tag}")
    best = rows[0]
    xx = next(r for r in rows if r[0] == (0, 0))
    print(f"\nID-best support = {names[best[0][0]]}*{names[best[0][1]]} (RMSE {best[1]:.4f}, "
          f"OOD {best[2]:.3f});  x*x rank by ID-RMSE = {[r[0] for r in rows].index((0,0))+1}/{len(rows)}"
          f" (RMSE {xx[1]:.4f}, OOD {xx[2]:.3f})")
    print("If x*x is ID-best AND OOD~1.0 -> objective prefers the rule; SGD/gates missed it (fix "
          "optimizer). If a bit-pair is ID-best with OOD~0 -> code/length alone picks a shortcut "
          "(need identifying evidence, not just edge count).")


if __name__ == "__main__":
    main()
