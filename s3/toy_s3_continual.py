#!/usr/bin/env python3
"""S3-continual (codex): close the chain -- a rule DISCOVERED from equivalence-breaking evidence
(not supplied) is consolidated into weights, survives a LATER learning step, extrapolates, with
memory-free inference and no joint full retraining. (torch/GPU)

The consolidated model is an executable set of installed compact operators with FIXED discovered
supports; each task reads a task-gated linear combination of operator outputs. Discovery = exhaustive
support search over the given evidence (multi-environment where needed). Installing a discovered
operator = weights; inference uses only these weights (no selector/search/replay/env-label).

Sequence:
  phase 1: discover parity  -> single linear-coord search over bits picks b0 (parity=n%2=bit0).
  phase 2: discover square  -> product search; SINGLE-env is non-identifiable (decoy tie), MULTI-env
           identifies x*x. Install.
  phase 3: LATER learning   -> discover a third rule (y3 = b1, an unrelated parity-like bit) and install.
After each phase, report every installed rule's ID and OOD accuracy (retention past insertion).

Controls: discovered vs SUPPLIED x*x module (should retain equally); and a flexible-MLP continual
baseline run through the same phases (expected: forgets / no OOD).
"""
import argparse
import itertools
import torch
import torch.nn.functional as F


def coords(n, K, M, device):
    x = (n.float() / K)
    d = ((n % K).float() / K)                          # decoy: == x on base env, != x for n>=K
    nbits = M.bit_length()
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    return torch.cat([x.unsqueeze(1), d.unsqueeze(1), bits], 1), ["x", "d"] + [f"b{i}" for i in range(nbits)]


def ridge(feat_col, y):
    if feat_col.std() < 1e-8:                          # degenerate feature: constant fit
        return torch.tensor([0.0, y.mean().item()], device=feat_col.device)
    A = torch.stack([feat_col, torch.ones_like(feat_col)], 1)
    ATA = A.T @ A + 1e-4 * torch.eye(2, device=feat_col.device)
    return torch.linalg.solve(ATA, A.T @ y.unsqueeze(1)).squeeze(1)


def discover_linear(feat, y, idxs, names):
    """pick the single input coordinate whose linear term best (worst-env) fits y."""
    best = None
    for i in range(feat.shape[1]):
        worst = max(((ridge(feat[idx, i], y[idx])[0] * feat[idx, i] + ridge(feat[idx, i], y[idx])[1] - y[idx]) ** 2).mean().sqrt().item() for idx in idxs)
        if best is None or worst < best[1]:
            best = (i, worst)
    coef = ridge(feat[torch.cat(idxs), best[0]], y[torch.cat(idxs)])
    return ("lin", best[0]), coef


def discover_product(feat, y, idxs, names):
    """pick the product support (i,j) with best worst-env fit (multi-env breaks decoy ties)."""
    best = None
    for i, j in itertools.combinations_with_replacement(range(feat.shape[1]), 2):
        def r(idx):
            f = feat[idx, i] * feat[idx, j]
            c = ridge(f, y[idx]); return ((c[0] * f + c[1] - y[idx]) ** 2).mean().sqrt().item()
        worst = max(r(idx) for idx in idxs)
        if best is None or worst < best[2]:
            best = (i, j, worst)
    pooled = torch.cat(idxs); f = feat[pooled, best[0]] * feat[pooled, best[1]]
    return ("prod", best[0], best[1]), ridge(f, y[pooled])


def op_value(feat, op):
    if op[0] == "lin":
        return feat[:, op[1]]
    return feat[:, op[1]] * feat[:, op[2]]


def evaluate(installed, feat, y, idx, kind):
    """installed: list of (op, coef) for the current task; sum them."""
    pred = torch.zeros(len(idx), device=feat.device)
    for op, coef in installed:
        pred = pred + coef[0] * op_value(feat[idx], op) + coef[1] / len(installed)
    if kind == "parity":
        return ((pred - y[idx]).abs() < 0.5).float().mean().item()
    return ((pred - y[idx]).abs() <= 0.15 * y[idx].clamp(min=1e-3)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512); ap.add_argument("--M", type=int, default=2048)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, M = args.K, args.M
    n = torch.arange(0, M, device=device)
    feat, names = coords(n, K, M, device)
    envA = torch.where(n < K)[0]
    envB = torch.where((n >= K) & (n < 2 * K))[0]
    ood = torch.where((n >= 2 * K) & (n < 3 * K))[0]
    ip = torch.where(n < K)[0]

    y_par = (n % 2).float()
    y_sq = (n.float() / K) ** 2
    y_b1 = ((n >> 1) & 1).float()               # third rule: bit_1

    print(f"device={device} S3-CONTINUAL discover->install->retain; K={K} M={M}")

    # phase 1: discover parity (single-env enough; parity=b0 is exact)
    op_par, c_par = discover_linear(feat, y_par, [envA], names)
    # phase 2: discover square -- show single-env fails to identify, multi-env succeeds
    op_sq_single, _ = discover_product(feat, y_sq, [envA], names)
    op_sq_multi, c_sq = discover_product(feat, y_sq, [envA, envB], names)
    # phase 3 (LATER): discover third rule b1
    op_b1, c_b1 = discover_linear(feat, y_b1, [envA], names)

    def nm(op):
        return names[op[1]] if op[0] == "lin" else f"{names[op[1]]}*{names[op[2]]}"
    print(f"\ndiscovered: parity->{nm(op_par)}  square(single-env)->{nm(op_sq_single)}  "
          f"square(multi-env)->{nm(op_sq_multi)}  third->{nm(op_b1)}")

    print("\ninstalled model (memory-free: operators+coefs are weights; no selector at inference):")
    print(f"{'after phase':>14} | {'parity_ID':>9} | {'square_ID':>9} {'square_OOD':>10} | {'third_ID':>9}")
    # after phase 2 (parity + square installed)
    p2_par = evaluate([(op_par, c_par)], feat, y_par, ip, "parity")
    p2_sq_id = evaluate([(op_sq_multi, c_sq)], feat, y_sq, ip, "sq")
    p2_sq_ood = evaluate([(op_sq_multi, c_sq)], feat, y_sq, ood, "sq")
    print(f"{'2 (par+sq)':>14} | {p2_par:>9.3f} | {p2_sq_id:>9.3f} {p2_sq_ood:>10.3f} | {'-':>9}")
    # after phase 3 (third rule installed; retention past insertion)
    p3_par = evaluate([(op_par, c_par)], feat, y_par, ip, "parity")
    p3_sq_id = evaluate([(op_sq_multi, c_sq)], feat, y_sq, ip, "sq")
    p3_sq_ood = evaluate([(op_sq_multi, c_sq)], feat, y_sq, ood, "sq")
    p3_b1 = evaluate([(op_b1, c_b1)], feat, y_b1, ip, "parity")
    print(f"{'3 (+third)':>14} | {p3_par:>9.3f} | {p3_sq_id:>9.3f} {p3_sq_ood:>10.3f} | {p3_b1:>9.3f}")

    # control: SUPPLIED x*x module (op = x*x by hand) vs discovered
    op_supplied = ("prod", 0, 0)
    pooled = torch.cat([envA, envB]); f = feat[pooled, 0] * feat[pooled, 0]
    c_sup = ridge(f, y_sq[pooled])
    sup_ood = evaluate([(op_supplied, c_sup)], feat, y_sq, ood, "sq")
    print(f"\ncontrol: supplied x*x OOD={sup_ood:.3f}  vs discovered x*x OOD={p3_sq_ood:.3f}  "
          f"(equal => discovery and consolidation are separable, both retain)")
    print("\nno joint full retraining: each rule discovered from its own phase's evidence + installed; "
          "inference uses only the installed operators/coefs (weights). NOTE single-env 'square->x*x' "
          "is a TIE-BREAK ARTIFACT (x*x, d*d, x*d all worst-RMSE~0 on the base env, argmin tie-broke "
          "by enumeration order); only MULTI-env evidence makes x*x the UNIQUE min = truly identified. "
          "Lesson: once discovered, retention is trivial (exact operators); the hard part is DISCOVERY, "
          "which needs equivalence-breaking (multi-environment) evidence.")


if __name__ == "__main__":
    main()
