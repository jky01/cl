#!/usr/bin/env python3
"""Does a fresh net LEARN the odd/even rule and EXTRAPOLATE it to numbers it never saw? (torch/GPU)

The session's north-star ("find the function -> don't need to store; can extrapolate") on the user's
original odd/even example. Train parity (n%2) on n in [0,K), then test:
  interp : held-out numbers inside [0,K)   (did it fit the rule in-range?)
  extrap : numbers in [K,M)  -- NEVER SEEN  (did it DERIVE the rule forward?)

The decisive variable is the INPUT REPRESENTATION -- whether it exposes the rule:
  scalar : x = n/M (one real).  Parity oscillates every 1/M -> a finite MLP cannot even fit it well,
           let alone extrapolate. "Rule not accessible in this representation."
  binary : x = bits of n.  Parity = the least-significant bit -> trivially found, extrapolates exactly.
  embed  : a learned vector per n.  Fits train by MEMORIZING; unseen n has no vector -> extrap = chance.

Lesson: "finding the function" is representation-dependent. Same rule, same net; extrapolation happens
only when the representation makes the rule a simple function of the input.
"""
import argparse
import torch
import torch.nn.functional as F


def encode(n, mode, nbits, M, emb):
    if mode == "scalar":
        return (n.float() / M).unsqueeze(1)
    if mode == "binary":
        bits = ((n.unsqueeze(1) >> torch.arange(nbits, device=n.device)) & 1).float()
        return bits
    return emb[n]                                   # embed


def mlp(din, h, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {"W1": (torch.randn(din, h, generator=g) / din ** 0.5).to(device).requires_grad_(),
            "b1": torch.zeros(h, device=device, requires_grad=True),
            "W2": (torch.randn(h, 2, generator=g) / h ** 0.5).to(device).requires_grad_(),
            "b2": torch.zeros(2, device=device, requires_grad=True)}


def fwd(P, x):
    return torch.relu(x @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


def run(seed, mode, args, device):
    M, K = args.M, args.K
    nbits = M.bit_length()
    torch.manual_seed(seed)
    alln = torch.arange(M, device=device)
    y = (alln % 2).long()
    # split: train/interp inside [0,K), extrap = [K,M)
    inside = alln[alln < K]
    perm = inside[torch.randperm(len(inside), device=device)]
    cut = int(0.8 * len(perm))
    train_n, interp_n = perm[:cut], perm[cut:]
    extrap_n = alln[alln >= K]

    din = {"scalar": 1, "binary": nbits, "embed": args.edim}[mode]
    emb = None
    P = mlp(din, args.h, seed, device)
    params = [P[k] for k in P]
    if mode == "embed":
        g = torch.Generator(device="cpu").manual_seed(seed + 3)
        emb = (torch.randn(M, args.edim, generator=g) * 0.5).to(device).requires_grad_()
        params = params + [emb]

    opt = torch.optim.Adam(params, lr=args.lr)
    yt = y[train_n]
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        xt = encode(train_n, mode, nbits, M, emb)   # recompute (embed is a trainable leaf)
        F.cross_entropy(fwd(P, xt), yt).backward()
        opt.step()

    @torch.no_grad()
    def acc(ns):
        if len(ns) == 0:
            return float("nan")
        return (fwd(P, encode(ns, mode, nbits, M, emb)).argmax(1) == y[ns]).float().mean().item()
    return acc(train_n), acc(interp_n), acc(extrap_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=2048)     # numbers 0..M-1
    ap.add_argument("--K", type=int, default=512)      # train/interp inside [0,K), extrap [K,M)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--edim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} PARITY M={args.M} train_range=[0,{args.K}) extrap=[{args.K},{args.M}) "
          f"h={args.h} seeds={args.seeds}")
    print(f"{'encoding':>9} {'train':>7} {'interp':>7} {'extrap':>7}   verdict")
    for mode in ["scalar", "binary", "embed"]:
        tr = ip = ex = 0.0
        for s in range(args.seeds):
            a, b, c = run(s, mode, args, device)
            tr += a; ip += b; ex += c
        tr, ip, ex = tr / args.seeds, ip / args.seeds, ex / args.seeds
        v = ("DERIVES the rule (extrapolates to unseen numbers)" if ex > 0.9 else
             "fits in-range but CANNOT extrapolate" if ip > 0.9 else
             "MEMORIZES train, zero generalization (interp=chance)" if tr > 0.9 else
             "cannot even fit the rule in this representation")
        print(f"{mode:>9} {tr:>7.3f} {ip:>7.3f} {ex:>7.3f}   {v}")
    print("\nSame rule, same net: extrapolation ('deriving further numbers') happens ONLY when the "
          "input representation exposes the rule (binary: parity=LSB). scalar can't fit; embed "
          "memorizes but has nothing for unseen numbers -> chance 0.5.")


if __name__ == "__main__":
    main()
