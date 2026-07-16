#!/usr/bin/env python3
"""Why square doesn't extrapolate in the two-rules shared net, and which inductive bias fixes it.
(torch/GPU; codex-reviewed spec)

Frame (codex): with a linear unbounded head there is no mechanical ceiling; training square on
q=(n/K)^2 in [0,1) simply does not CONSTRAIN the function for q>=1 -> underdetermined OOD continuation,
resolved by parameterization / interference / regularization. And because q is a SUPPLIED input
feature, this tests feature ROUTING under a task tag, not discovery of squaring.

Architectures (each tested square-ALONE (single task, isolates architecture) and CONSOLIDATE
(parity-then-square, isolates sharing/sequential)):
  baseline  : shared 2-layer ReLU trunk -> scalar head.
  wd        : baseline + weight decay (AdamW).
  gskip     : out = MLP(x,t) + a*q_sq + b, q_sq = 1[task=square]*q  (task-gated RAW linear channel).
  sepheads  : shared trunk -> separate parity/square linear heads (NO raw skip; head interference only).
Plus a linear-regression ORACLE y=a*q+b (should extrapolate ~exactly; catches scaling/eval bugs).

Square diagnostics on nested OOD shells n/K in [1,1.5),[1.5,2),[2,3),[3,4): within-15% + max pred.
Parity reported after parity-training and after square-training (retention).
"""
import argparse
import torch
import torch.nn.functional as F


def encode(n, K, nbits, task, device):
    x = (n.float() / K)
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    tag = torch.zeros(len(n), 2, device=device); tag[:, task] = 1.0
    return torch.cat([x.unsqueeze(1), (x ** 2).unsqueeze(1), bits, tag], 1), x ** 2, tag


def tgt(n, K, task):
    return (n % 2).float() if task == 0 else (n.float() / K) ** 2


def net(arch, din, h, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def T(a, b):
        return (torch.randn(a, b, generator=g) / a ** 0.5).to(device).requires_grad_()
    P = {"W1": T(din, h), "b1": torch.zeros(h, device=device, requires_grad=True),
         "W2": T(h, h), "b2": torch.zeros(h, device=device, requires_grad=True)}
    if arch == "sepheads":
        P["Hpar"] = T(h, 1); P["hpar"] = torch.zeros(1, device=device, requires_grad=True)
        P["Hsq"] = T(h, 1); P["hsq"] = torch.zeros(1, device=device, requires_grad=True)
    else:
        P["W3"] = T(h, 1); P["b3"] = torch.zeros(1, device=device, requires_grad=True)
    if arch == "gskip":
        P["a"] = torch.zeros(1, device=device, requires_grad=True)
        P["b"] = torch.zeros(1, device=device, requires_grad=True)
    return P


def fwd(P, arch, x, q_sq, tag):
    h = torch.relu(x @ P["W1"] + P["b1"])
    h = torch.relu(h @ P["W2"] + P["b2"])
    if arch == "sepheads":
        is_sq = tag[:, 1:2]
        out = (1 - is_sq) * (h @ P["Hpar"] + P["hpar"]) + is_sq * (h @ P["Hsq"] + P["hsq"])
        return out.squeeze(1)
    out = (h @ P["W3"] + P["b3"]).squeeze(1)
    if arch == "gskip":
        out = out + (P["a"] * (tag[:, 1] * q_sq) + P["b"])
    return out


def fit(P, arch, data, epochs, lr, wd=0.0):
    opt = torch.optim.AdamW([P[k] for k in P], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for (x, q, tag), y in data:
            loss = loss + F.mse_loss(fwd(P, arch, x, q, tag), y)
        loss.backward()
        opt.step()


def run(seed, arch, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    din = 2 + nbits + 2
    torch.manual_seed(seed)
    perm = torch.randperm(K, device=device)
    n_tr, n_ip = perm[:int(0.85 * K)], perm[int(0.85 * K):]

    def E(n, t): return encode(n, K, nbits, t, device)

    # square-alone (single task, same architecture) and consolidate (parity then square)
    # -- parity first
    P0 = net(arch, din, args.h, seed, device)
    fit(P0, arch, [(E(n_tr, 0), tgt(n_tr, K, 0))], args.epochs, args.lr, args.wd if arch == "wd" else 0)
    par_after_par = par_acc(P0, arch, n_ip, K, nbits, device)

    with torch.no_grad():
        alln = torch.arange(M, device=device)
        par_all = fwd(P0, arch, *E(alln, 0)) if False else None
    # consolidate: fresh net, distill parity over line + square true
    with torch.no_grad():
        xp, qp, tp = E(torch.arange(M, device=device), 0)
        par_line = fwd(P0, arch, xp, qp, tp)
    Pc = net(arch, din, args.h, seed + 1, device)
    fit(Pc, arch, [((xp, qp, tp), par_line), (E(n_tr, 1), tgt(n_tr, K, 1))],
        args.epochs, args.lr, args.wd if arch == "wd" else 0)

    # square-alone control (same arch, single task, task tag = square held constant)
    Pa = net(arch, din, args.h, seed + 2, device)
    fit(Pa, arch, [(E(n_tr, 1), tgt(n_tr, K, 1))], args.epochs, args.lr, args.wd if arch == "wd" else 0)

    return {
        "par_after_par": par_after_par,
        "par_after_sq_cons": par_acc(Pc, arch, n_ip, K, nbits, device),
        "sq_cons": sq_diag(Pc, arch, K, M, nbits, device),
        "sq_alone": sq_diag(Pa, arch, K, M, nbits, device),
    }


@torch.no_grad()
def par_acc(P, arch, n, K, nbits, device):
    x, q, tag = encode(n, K, nbits, 0, device)
    return ((fwd(P, arch, x, q, tag) - tgt(n, K, 0)).abs() < 0.5).float().mean().item()


@torch.no_grad()
def sq_diag(P, arch, K, M, nbits, device):
    shells = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
    out = {}
    # interp
    ni = torch.arange(0, K, device=device)
    xi, qi, ti = encode(ni, K, nbits, 1, device)
    pi = fwd(P, arch, xi, qi, ti); yi = tgt(ni, K, 1)
    out["interp"] = ((pi - yi).abs() <= 0.15 * yi.abs().clamp(min=1e-3)).float().mean().item()
    for lo, hi in shells:
        ns = torch.arange(int(lo * K), min(int(hi * K), M), device=device)
        if len(ns) == 0:
            out[f"[{lo},{hi})"] = float("nan"); continue
        x, q, tag = encode(ns, K, nbits, 1, device)
        p = fwd(P, arch, x, q, tag); y = tgt(ns, K, 1)
        out[f"[{lo},{hi})"] = ((p - y).abs() <= 0.15 * y).float().mean().item()
    # max predicted value on the largest shell (saturation check; true max ~ (M/K)^2)
    nmax = torch.arange(int(3 * K), M, device=device)
    x, q, tag = encode(nmax, K, nbits, 1, device)
    out["maxpred"] = fwd(P, arch, x, q, tag).max().item()
    return out


def linear_oracle(seed, args, device):
    M, K = args.M, args.K
    torch.manual_seed(seed)
    n = torch.arange(0, K, device=device).float()
    q = (n / K) ** 2; y = q
    A = torch.stack([q, torch.ones_like(q)], 1)
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    shells = [(1.0, 1.5), (2.0, 3.0), (3.0, 4.0)]
    res = {}
    for lo, hi in shells:
        ns = torch.arange(int(lo * K), min(int(hi * K), M), device=device).float()
        qs = (ns / K) ** 2
        pred = coef[0] * qs + coef[1]
        res[f"[{lo},{hi})"] = ((pred - qs).abs() <= 0.15 * qs).float().mean().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5000); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} TWO-RULES-FIX one net parity->square; wd={args.wd} seeds={args.seeds}")
    lo = linear_oracle(0, args, device)
    print(f"linear-oracle y=a*q+b extrap within15%: " + "  ".join(f"{k}={v:.2f}" for k, v in lo.items()))
    print(f"\n{'arch':>9} {'mode':>9} | {'par_ret':>7} {'sq_ip':>6} "
          f"{'[1,1.5)':>8} {'[1.5,2)':>8} {'[2,3)':>7} {'[3,4)':>7} {'maxpred':>8}")
    for arch in ["baseline", "wd", "gskip", "sepheads"]:
        agg = {}
        for s in range(args.seeds):
            r = run(s, arch, args, device)
            for k, v in r.items():
                agg.setdefault(k, []).append(v)
        par_ret = sum(agg["par_after_sq_cons"]) / args.seeds
        for mode, key in [("consolid", "sq_cons"), ("sq_alone", "sq_alone")]:
            d = {kk: sum(x[kk] for x in agg[key]) / args.seeds for kk in agg[key][0]}
            pr = f"{par_ret:.3f}" if mode == "consolid" else "  -  "
            print(f"{arch:>9} {mode:>9} | {pr:>7} {d['interp']:>6.3f} "
                  f"{d['[1.0,1.5)']:>8.3f} {d['[1.5,2.0)']:>8.3f} {d['[2.0,3.0)']:>7.3f} "
                  f"{d['[3.0,4.0)']:>7.3f} {d['maxpred']:>8.2f}")
    print("\ntrue square max on [3K,M) ~ (M/K)^2 = "
          f"{(args.M/args.K)**2:.1f}. maxpred near that => scales; near 1 => saturated. "
          "sq_alone success + consolid fail => sharing/sequential cost; both fail => architecture bias.")


if __name__ == "__main__":
    main()
