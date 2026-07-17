#!/usr/bin/env python3
"""Whole-graph structural MDL selector on the discovery toy: can a target-independent accounting rule
make the reachable short mechanism (square = x*x) WIN and extrapolate, without hand-restricting the
hypothesis class? (torch/GPU; codex's first probe)

Discovery architecture (no q supplied): x=[n/K, bits]; product units z_k=(A_k.x)*(B_k.x) (x*x
reachable); residual units h_j=relu(W_j.x+b_j); out = sum gp_k vp_k z_k + sum gh_j vh_j h_j + b.
Both routes are gated (hard-concrete L0) and charged under ONE accounting, so the dense residual
cannot implement a free expensive local fit.

Selector arms (beta/gamma FIXED, never tuned on OOD shells):
  none     : no gates, task loss only                         (negative baseline)
  wl1      : L1 on all weights, no gates                       (optimization regularization)
  prodL0   : L0 gate on PRODUCT units only, residual ungated   (narrow; residual loophole remains)
  graphL0  : L0 gate on products AND residual, one accounting  (whole-graph MDL: should select x*x)

Report OOD extrapolation shells AND graph recovery: expected active product / residual units. Success
(codex) = the selected graph collapses to the short multiplicative mechanism AND extrapolates.
"""
import argparse
import torch
import torch.nn.functional as F

# hard-concrete constants (Louizos et al.)
GAMMA, ZETA, BETA_T = -0.1, 1.1, 0.5
LOG_RATIO = BETA_T * torch.log(torch.tensor(-GAMMA / ZETA))


def hc_sample(log_a):
    u = torch.rand_like(log_a).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_a) / BETA_T)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


def hc_eval(log_a):
    s = torch.sigmoid(log_a)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


def hc_l0(log_a):                        # expected number of active gates
    return torch.sigmoid(log_a - LOG_RATIO.to(log_a.device)).sum()


def feats(n, K, nbits, device):
    x = (n.float() / K).unsqueeze(1)
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    return torch.cat([x, bits], 1)


def model(din, P, H, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def T(a, b, s=1.0):
        return (torch.randn(a, b, generator=g) * s / a ** 0.5).to(device).requires_grad_()
    return {"A": T(din, P), "B": T(din, P), "vp": T(P, 1),
            "W": T(din, H), "bh": torch.zeros(H, device=device, requires_grad=True), "vh": T(H, 1),
            "bo": torch.zeros(1, device=device, requires_grad=True),
            "la_p": (torch.ones(P, device=device) * 1.0).requires_grad_(),     # product gates
            "la_h": (torch.ones(H, device=device) * 1.0).requires_grad_()}     # residual gates


def fwd(P, x, arm, train):
    z = (x @ P["A"]) * (x @ P["B"])                     # [N, P]
    h = torch.relu(x @ P["W"] + P["bh"])                # [N, H]
    if arm in ("prodL0", "graphL0"):
        gp = hc_sample(P["la_p"]) if train else hc_eval(P["la_p"])
        z = z * gp
    if arm == "graphL0":
        gh = hc_sample(P["la_h"]) if train else hc_eval(P["la_h"])
        h = h * gh
    return (z @ P["vp"] + h @ P["vh"] + P["bo"]).squeeze(1)


def struct_cost(P, arm):
    if arm == "prodL0":
        return hc_l0(P["la_p"])
    if arm == "graphL0":
        return hc_l0(P["la_p"]) + hc_l0(P["la_h"])
    return torch.zeros((), device=P["A"].device)


def weight_cost(P, arm):
    if arm == "wl1":
        return sum(P[k].abs().sum() for k in ["A", "B", "vp", "W", "vh"])
    # coefficient precision proxy (L2) for the L0 arms so magnitude can't be free
    if arm in ("prodL0", "graphL0"):
        return sum((P[k] ** 2).sum() for k in ["A", "B", "vp", "W", "vh"])
    return torch.zeros((), device=P["A"].device)


def fit(P, x, y, arm, epochs, lr, beta, gam):
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(fwd(P, x, arm, True), y) + beta * struct_cost(P, arm) + gam * weight_cost(P, arm)
        loss.backward()
        opt.step()


def tgt(n, K):
    return (n.float() / K) ** 2


def run(seed, arm, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    din = 1 + nbits
    torch.manual_seed(seed)
    perm = torch.randperm(K, device=device)
    n_tr = perm[:int(0.85 * K)]
    P = model(din, args.P, args.H, seed, device)
    fit(P, feats(n_tr, K, nbits, device), tgt(n_tr, K), arm, args.epochs, args.lr, args.beta, args.gam)

    @torch.no_grad()
    def shell(lo, hi):
        n = torch.arange(int(lo * K), min(int(hi * K), M), device=device)
        p = fwd(P, feats(n, K, nbits, device), arm, False); y = tgt(n, K)
        return ((p - y).abs() <= 0.15 * y.clamp(min=1e-3)).float().mean().item()

    @torch.no_grad()
    def active():
        ap = (hc_eval(P["la_p"]) > 0.01).sum().item() if arm in ("prodL0", "graphL0") else args.P
        ah = (hc_eval(P["la_h"]) > 0.01).sum().item() if arm == "graphL0" else args.H
        return ap, ah

    ap, ah = active()
    return {"ip": shell(0, 1), "s15": shell(1, 1.5), "s23": shell(2, 3), "s34": shell(3, 4),
            "act_prod": ap, "act_res": ah}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--P", type=int, default=8); ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6000); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--beta", type=float, default=0.02)   # L0 structure cost (PREREGISTERED, not OOD-tuned)
    ap.add_argument("--gam", type=float, default=1e-4)    # weight/precision cost
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} MDL discovery (square=x*x reachable, no q). P={args.P} H={args.H} "
          f"beta={args.beta} gam={args.gam} seeds={args.seeds}  (beta/gam fixed, NOT OOD-tuned)")
    print(f"{'selector':>9} | {'sq_ip':>6} {'[1,1.5)':>8} {'[2,3)':>7} {'[3,4)':>7} | "
          f"{'act_prod':>8} {'act_res':>8}   note")
    notes = {"none": "negative baseline", "wl1": "opt regularization only",
             "prodL0": "product-only (residual loophole)", "graphL0": "WHOLE-GRAPH MDL"}
    for arm in ["none", "wl1", "prodL0", "graphL0"]:
        agg = {}
        for s in range(args.seeds):
            for k, v in run(s, arm, args, device).items():
                agg.setdefault(k, []).append(v)
        m = {k: sum(v) / len(v) for k, v in agg.items()}
        print(f"{arm:>9} | {m['ip']:>6.3f} {m['s15']:>8.3f} {m['s23']:>7.3f} {m['s34']:>7.3f} | "
              f"{m['act_prod']:>8.1f} {m['act_res']:>8.1f}   {notes[arm]}")
    print("\nSUCCESS (graphL0): OOD shells ~1.0 AND active graph collapses to ~1 product unit + ~0 "
          "residual (the short x*x mechanism). If graphL0 extrapolates while prodL0 does not, the "
          "whole-graph accounting -- not mere product sparsity -- is what selects the reusable rule.")


if __name__ == "__main__":
    main()
