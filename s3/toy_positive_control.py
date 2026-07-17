#!/usr/bin/env python3
"""Positive control: with the CORRECT restricted hypothesis class, does the continual learner both
RETAIN parity and EXTRAPOLATE the square? (torch/GPU; codex-suggested)

Establishes the clean conjunction the flexible-net experiments couldn't: consolidation can retain an
old rule WHILE a suitable hypothesis class acquires an extrapolable new rule. Does NOT show rule
discovery -- q and the linear form are supplied (that's the next, discovery experiment).

Restricted model (task-gated, NO flexible ReLU trunk for square):
  parity (task 0): y = w_p . bits + c_p      (parity = bit_0, linear in bits -> extrapolates)
  square (task 1): y = a * q + b             (q=(n/K)^2 supplied; linear -> extrapolates)
Continual protocol: learn parity, then consolidate square (self-distill parity over the number line
+ square true labels) into a fresh restricted model. Report parity retention + square OOD shells.
"""
import argparse
import torch
import torch.nn.functional as F


def feats(n, K, nbits, device):
    x = n.float() / K
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    q = x ** 2
    return bits, q


def model(nbits, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {"w_p": (torch.randn(nbits, 1, generator=g) * 0.1).to(device).requires_grad_(),
            "c_p": torch.zeros(1, device=device, requires_grad=True),
            "a": torch.zeros(1, device=device, requires_grad=True),
            "b": torch.zeros(1, device=device, requires_grad=True)}


def out(P, bits, q, task):
    par = (bits @ P["w_p"] + P["c_p"]).squeeze(1)
    sq = P["a"] * q + P["b"]
    return par if task == 0 else sq


def tgt(n, K, task):
    return (n % 2).float() if task == 0 else (n.float() / K) ** 2


def fit(P, data, epochs, lr):
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = sum(F.mse_loss(out(P, b, q, t), y) for (b, q, t), y in data)
        loss.backward()
        opt.step()


def run(seed, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    torch.manual_seed(seed)
    perm = torch.randperm(K, device=device)
    n_tr = perm[:int(0.85 * K)]

    def F_(n): return feats(n, K, nbits, device)

    # phase 1: parity
    P0 = model(nbits, seed, device)
    b_tr, q_tr = F_(n_tr)
    fit(P0, [((b_tr, q_tr, 0), tgt(n_tr, K, 0))], args.epochs, args.lr)

    # consolidate: fresh restricted model, self-distill parity over all n + square true
    alln = torch.arange(M, device=device)
    b_all, q_all = F_(alln)
    with torch.no_grad():
        par_line = out(P0, b_all, q_all, 0)
    Pc = model(nbits, seed + 1, device)
    fit(Pc, [((b_all, q_all, 0), par_line), ((b_tr, q_tr, 1), tgt(n_tr, K, 1))], args.epochs, args.lr)

    @torch.no_grad()
    def par_acc(P, n):
        b, q = F_(n)
        return ((out(P, b, q, 0) - tgt(n, K, 0)).abs() < 0.5).float().mean().item()

    @torch.no_grad()
    def sq_shell(P, lo, hi):
        n = torch.arange(int(lo * K), min(int(hi * K), M), device=device)
        b, q = F_(n)
        p = out(P, b, q, 1); y = tgt(n, K, 1)
        return ((p - y).abs() <= 0.15 * y.clamp(min=1e-3)).float().mean().item()

    n_ip = perm[int(0.85 * K):]
    return {
        "par_ret": par_acc(Pc, n_ip),
        "sq_ip": sq_shell(Pc, 0.0, 1.0),
        "sq_15": sq_shell(Pc, 1.0, 1.5), "sq_23": sq_shell(Pc, 2.0, 3.0), "sq_34": sq_shell(Pc, 3.0, 4.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4000); ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} POSITIVE-CONTROL restricted class (parity=linear(bits), square=a*q+b); "
          f"continual parity->square; seeds={args.seeds}")
    agg = {}
    for s in range(args.seeds):
        for k, v in run(s, args, device).items():
            agg.setdefault(k, []).append(v)
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"\nparity_retention={m['par_ret']:.3f}   square: interp={m['sq_ip']:.3f}  "
          f"[1,1.5)={m['sq_15']:.3f}  [2,3)={m['sq_23']:.3f}  [3,4)={m['sq_34']:.3f}")
    print("expected: parity retained ~1.0 AND square extrapolates ~1.0 on all shells -> the right "
          "hypothesis class + consolidation gives BOTH retention and extrapolable acquisition (the "
          "conjunction flexible ReLU nets could not achieve). q supplied => NOT rule discovery.")


if __name__ == "__main__":
    main()
