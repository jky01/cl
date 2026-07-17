#!/usr/bin/env python3
"""Discovery test: remove the gift. Do NOT supply q=(n/K)^2. Give the net raw inputs [n/K, bits] and a
MULTIPLICATIVE capability (so x*x is REACHABLE), and ask whether it DISCOVERS square = x*x, preserves
it, and extrapolates -- standalone and continually. (torch/GPU; codex-suggested sharp factorization)

Architecture with a product layer:
  feat = [n/K, bits]
  z_k  = (A_k . feat) * (B_k . feat)     for k=1..P products   (x^2 reachable via one product of x with x)
  out  = MLP/linear on [feat, z] (task-gated)
If it cannot extrapolate square even with x*x reachable -> the failure is RULE DISCOVERY (selecting the
compact mechanism), not availability. If standalone works but continual doesn't -> CL implicated again.

Arms: square-alone (isolates discovery) and consolidate (parity then square). Also report parity.
Compare to the positive control (q supplied) as the ceiling.
"""
import argparse
import torch
import torch.nn.functional as F


def feats(n, K, nbits, device):
    x = (n.float() / K).unsqueeze(1)
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    return torch.cat([x, bits], 1)                 # NOTE: no q=x^2 supplied


def net(din, nprod, h, seed, device, tagdim=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def T(a, b, s=1.0):
        return (torch.randn(a, b, generator=g) * s / a ** 0.5).to(device).requires_grad_()
    d_in = din + tagdim
    P = {"A": T(d_in, nprod), "B": T(d_in, nprod)}     # product layer
    d_h = d_in + nprod
    P["W1"] = T(d_h, h); P["b1"] = torch.zeros(h, device=device, requires_grad=True)
    P["W2"] = T(h, 1); P["b2"] = torch.zeros(1, device=device, requires_grad=True)
    return P


def fwd(P, feat, tag):
    x = torch.cat([feat, tag], 1)
    z = (x @ P["A"]) * (x @ P["B"])                    # multiplicative units (x*x reachable)
    h = torch.relu(torch.cat([x, z], 1) @ P["W1"] + P["b1"])
    return (h @ P["W2"] + P["b2"]).squeeze(1)


def tgt(n, K, task):
    return (n % 2).float() if task == 0 else (n.float() / K) ** 2


def tagvec(nn, task, device):
    t = torch.zeros(nn, 2, device=device); t[:, task] = 1.0
    return t


def fit(P, data, epochs, lr, wd=0.0):
    opt = torch.optim.AdamW([P[k] for k in P], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = sum(F.mse_loss(fwd(P, f, t), y) for (f, t), y in data)
        loss.backward()
        opt.step()


def run(seed, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    din = 1 + nbits
    torch.manual_seed(seed)
    perm = torch.randperm(K, device=device)
    n_tr, n_ip = perm[:int(0.85 * K)], perm[int(0.85 * K):]

    def FE(n): return feats(n, K, nbits, device)
    def tv(n, t): return tagvec(len(n), t, device)

    # square-alone (isolates discovery)
    Pa = net(din, args.nprod, args.h, seed, device)
    fit(Pa, [((FE(n_tr), tv(n_tr, 1)), tgt(n_tr, K, 1))], args.epochs, args.lr, args.wd)

    # continual: parity then consolidate square
    P0 = net(din, args.nprod, args.h, seed + 1, device)
    fit(P0, [((FE(n_tr), tv(n_tr, 0)), tgt(n_tr, K, 0))], args.epochs, args.lr, args.wd)
    alln = torch.arange(M, device=device)
    with torch.no_grad():
        par_line = fwd(P0, FE(alln), tv(alln, 0))
    Pc = net(din, args.nprod, args.h, seed + 2, device)
    fit(Pc, [((FE(alln), tv(alln, 0)), par_line), ((FE(n_tr), tv(n_tr, 1)), tgt(n_tr, K, 1))],
        args.epochs, args.lr, args.wd)

    @torch.no_grad()
    def par_acc(P, n):
        return ((fwd(P, FE(n), tv(n, 0)) - tgt(n, K, 0)).abs() < 0.5).float().mean().item()

    @torch.no_grad()
    def sq_shell(P, lo, hi):
        n = torch.arange(int(lo * K), min(int(hi * K), M), device=device)
        p = fwd(P, FE(n), tv(n, 1)); y = tgt(n, K, 1)
        return ((p - y).abs() <= 0.15 * y.clamp(min=1e-3)).float().mean().item()

    return {
        "alone_ip": sq_shell(Pa, 0, 1), "alone_15": sq_shell(Pa, 1, 1.5),
        "alone_23": sq_shell(Pa, 2, 3), "alone_34": sq_shell(Pa, 3, 4),
        "cons_par": par_acc(Pc, n_ip), "cons_ip": sq_shell(Pc, 0, 1),
        "cons_15": sq_shell(Pc, 1, 1.5), "cons_23": sq_shell(Pc, 2, 3), "cons_34": sq_shell(Pc, 3, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--nprod", type=int, default=8); ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6000); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} DISCOVERY no-q, multiplicative arch (nprod={args.nprod}); x*x reachable. "
          f"seeds={args.seeds}")
    agg = {}
    for s in range(args.seeds):
        for k, v in run(s, args, device).items():
            agg.setdefault(k, []).append(v)
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"\nsquare-alone (discovery): interp={m['alone_ip']:.3f}  [1,1.5)={m['alone_15']:.3f}  "
          f"[2,3)={m['alone_23']:.3f}  [3,4)={m['alone_34']:.3f}")
    print(f"consolidate: parity_ret={m['cons_par']:.3f}  square interp={m['cons_ip']:.3f}  "
          f"[1,1.5)={m['cons_15']:.3f}  [2,3)={m['cons_23']:.3f}  [3,4)={m['cons_34']:.3f}")
    print("\nalone extrapolates => the net DISCOVERED square=x*x from raw x (selection of the compact "
          "mechanism worked). alone fails => rule DISCOVERY is the wall even with x*x reachable. "
          "alone works but consolidate fails => CL mechanism implicated in preserving the discovered rule.")


if __name__ == "__main__":
    main()
