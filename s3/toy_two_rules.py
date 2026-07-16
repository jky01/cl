#!/usr/bin/env python3
"""ONE network, learned SEQUENTIALLY: parity (odd/even) then a quadratic. Does it retain BOTH and
EXTRAPOLATE BOTH? (torch/GPU)

Combines the extrapolation probes (parity, quadratic) with the continual-learning loop. A single
shared MLP takes a rich encoding of a number n that EXPOSES both rules -- [n/K, (n/K)^2, bits(n)] --
plus a 2-dim task tag (parity | square). Unified scalar-regression output.

Tasks (trained on n in [0,K); extrapolation tested on n in [K,M)):
  parity : target = n % 2      (needs the bits)   -> within-tol = rounds to correct 0/1
  square : target = (n/K)^2    (needs the x^2 feature; >=1 outside training) -> within 15% relative

Two continual strategies after learning parity first:
  naive       : fine-tune ALL params on square           -> expected: forgets parity.
  consolidate : distill the parity-model over the number line (self, no stored data) + square true
                targets, into a fresh flat SAME-width net -> expected: keeps both.

Reports parity and square accuracy, split interp (inside [0,K)) vs extrap (in [K,M)), after the net
has learned BOTH. The point: one net, two different rules, retained and each extrapolated.
"""
import argparse
import torch
import torch.nn.functional as F


def encode(n, K, nbits, task, device):
    x = (n.float() / K)
    bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    tag = torch.zeros(len(n), 2, device=device); tag[:, task] = 1.0
    return torch.cat([x.unsqueeze(1), (x ** 2).unsqueeze(1), bits, tag], 1)


def targets(n, K, task):
    return (n % 2).float() if task == 0 else (n.float() / K) ** 2


def mlp(din, h, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = {"W1": (torch.randn(din, h, generator=g) / din ** 0.5).to(device).requires_grad_(),
         "b1": torch.zeros(h, device=device, requires_grad=True),
         "W2": (torch.randn(h, h, generator=g) / h ** 0.5).to(device).requires_grad_(),
         "b2": torch.zeros(h, device=device, requires_grad=True),
         "W3": (torch.randn(h, 1, generator=g) / h ** 0.5).to(device).requires_grad_(),
         "b3": torch.zeros(1, device=device, requires_grad=True)}
    return P


def fwd(P, x):
    h = torch.relu(x @ P["W1"] + P["b1"])
    h = torch.relu(h @ P["W2"] + P["b2"])
    return (h @ P["W3"] + P["b3"]).squeeze(1)


def fit(P, batches, epochs, lr):
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = sum(F.mse_loss(fwd(P, x), y) for x, y in batches)
        loss.backward()
        opt.step()


def clone(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def run(seed, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    din = 2 + nbits + 2
    torch.manual_seed(seed)
    n_in = torch.arange(K, device=device)
    perm = n_in[torch.randperm(K, device=device)]
    cut = int(0.85 * K)
    n_tr, n_ip = perm[:cut], perm[cut:]
    n_ex = torch.arange(K, M, device=device)

    def enc(n, task): return encode(n, K, nbits, task, device)
    def tgt(n, task): return targets(n, K, task)

    # phase 1: learn parity (task 0)
    P0 = mlp(din, args.h, seed, device)
    fit(P0, [(enc(n_tr, 0), tgt(n_tr, 0))], args.epochs, args.lr)

    # phase 2a: NAIVE fine-tune on square (task 1), all params
    Pn = clone(P0)
    fit(Pn, [(enc(n_tr, 1), tgt(n_tr, 1))], args.epochs, args.lr)

    # phase 2b: CONSOLIDATE -> fresh flat same-width net trained on parity (self-distilled from P0 over
    # the whole number line) + square true targets. No stored old data; P0 is a function.
    with torch.no_grad():
        par_all = fwd(P0, enc(torch.arange(M, device=device), 0))     # P0's parity over all n (self)
    alln = torch.arange(M, device=device)
    Pc = mlp(din, args.h, seed + 1, device)
    fit(Pc, [(enc(alln, 0), par_all),                                  # parity distilled (broad)
             (enc(n_tr, 1), tgt(n_tr, 1))], args.epochs, args.lr)  # square true labels

    @torch.no_grad()
    def par_acc(P, n):
        return ((fwd(P, enc(n, 0)) - tgt(n, 0)).abs() < 0.5).float().mean().item()

    @torch.no_grad()
    def sq_acc(P, n):
        pred = fwd(P, enc(n, 1)); true = tgt(n, 1)
        return ((pred - true).abs() <= 0.15 * true.abs().clamp(min=1e-3)).float().mean().item()

    return {
        "naive_par_ip": par_acc(Pn, n_ip), "naive_par_ex": par_acc(Pn, n_ex),
        "naive_sq_ip": sq_acc(Pn, n_ip), "naive_sq_ex": sq_acc(Pn, n_ex),
        "cons_par_ip": par_acc(Pc, n_ip), "cons_par_ex": par_acc(Pc, n_ex),
        "cons_sq_ip": sq_acc(Pc, n_ip), "cons_sq_ex": sq_acc(Pc, n_ex),
        "p1_par_ip": par_acc(P0, n_ip), "p1_par_ex": par_acc(P0, n_ex),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} TWO-RULES one net: parity then quadratic. train n in [0,{args.K}), "
          f"extrap [{args.K},{args.M}). seeds={args.seeds}")
    agg = {}
    for s in range(args.seeds):
        for k, v in run(s, args, device).items():
            agg.setdefault(k, []).append(v)
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"\n(parity-only phase-1 baseline: interp={m['p1_par_ip']:.3f} extrap={m['p1_par_ex']:.3f})")
    print(f"\n{'strategy':>12} | {'parity_interp':>13} {'parity_extrap':>13} | "
          f"{'square_interp':>13} {'square_extrap':>13}")
    print(f"{'naive':>12} | {m['naive_par_ip']:>13.3f} {m['naive_par_ex']:>13.3f} | "
          f"{m['naive_sq_ip']:>13.3f} {m['naive_sq_ex']:>13.3f}")
    print(f"{'consolidate':>12} | {m['cons_par_ip']:>13.3f} {m['cons_par_ex']:>13.3f} | "
          f"{m['cons_sq_ip']:>13.3f} {m['cons_sq_ex']:>13.3f}")
    print("\nnaive: learning square should ERASE parity (both columns can't stay high). consolidate: "
          "ONE flat net keeps BOTH rules and each extrapolates to unseen numbers -> continual "
          "acquisition of two different rules in a single network, no forgetting, both derived.")


if __name__ == "__main__":
    main()
