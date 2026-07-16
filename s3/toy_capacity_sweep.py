#!/usr/bin/env python3
"""Capacity frontier for the frozen-substrate toy (torch/GPU).

Resolves the confound codex flagged: at h=64 the joint oracle only reaches ~0.62,
so naive forgetting there mixes INTERFERENCE with plain INFEASIBILITY. Sweep the
shared width h and locate three regimes:
  infeasible      : joint itself misses the target,
  tight-feasible  : joint ~perfect but naive still forgets  <- the clean fold-test regime,
  roomy           : both easy.

For each h we report the JOINT oracle (best of R restarts, long training -> is 0.62
really a ceiling?) and NAIVE sequential forgetting. Random-label task = worst case
(no compressibility); frozen key rep forces knowledge into shared W1/W2.
"""
import argparse
import torch
import torch.nn.functional as F


def make_targets(seed, N, C):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, C, (N,), generator=g)


def make_params(init_seed, N, C, d, h, device):
    g = torch.Generator(device="cpu").manual_seed(init_seed)
    E = (torch.randn(N, d, generator=g)).to(device)                # frozen random address
    P = {
        "W1": (torch.randn(d, h, generator=g) / d ** 0.5).to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }
    return E, P


def fwd(E, P, keys):
    z = torch.relu(E[keys] @ P["W1"] + P["b1"])
    return z @ P["W2"] + P["b2"]


def fit(E, P, keys, y, epochs, lr):
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(E, P, keys), y[keys]).backward()
        opt.step()


@torch.no_grad()
def acc(E, P, keys, y):
    if keys.numel() == 0:
        return float("nan")
    return (fwd(E, P, keys).argmax(1) == y[keys]).float().mean().item()


def joint_best(E0, seed, N, C, d, h, device, y, keys, epochs, lr, restarts):
    best = None
    for r in range(restarts):
        E, P = make_params(seed + 7919 * (r + 1), N, C, d, h, device)
        E = E0  # keep the SAME frozen address across restarts; reinit only trainable weights
        fit(E, P, keys, y, epochs, lr)
        A = keys[keys % 2 == 1]
        B = keys[keys % 2 == 0]
        m = min(acc(E, P, A, y), acc(E, P, B, y))
        if best is None or m > best[0]:
            best = (m, acc(E, P, A, y), acc(E, P, B, y))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hs", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} N={args.N} C={args.C} d={args.d} epochs={args.epochs} "
          f"lr={args.lr} restarts={args.restarts} seeds={args.seeds}")
    print(f"{'h':>6} {'sharedW':>8} {'joint_min':>10} {'joint_A':>8} {'joint_B':>8} "
          f"{'naive_A0':>9} {'naive_Apost':>11} {'forget':>7}")

    for h in args.hs:
        sharedW = args.d * h + h * args.C
        jm = ja = jb = 0.0
        n0 = npost = 0.0
        for s in range(args.seeds):
            y = make_targets(s, args.N, args.C).to(device)
            keys = torch.arange(args.N, device=device)
            A = keys[keys % 2 == 1]
            B = keys[keys % 2 == 0]
            E0, _ = make_params(s, args.N, args.C, args.d, h, device)

            bm, ba, bb = joint_best(E0, s, args.N, args.C, args.d, h, device,
                                    y, keys, args.epochs, args.lr, args.restarts)
            jm += bm; ja += ba; jb += bb

            # naive sequential from a fresh init sharing the same frozen address
            _, P = make_params(s, args.N, args.C, args.d, h, device)
            fit(E0, P, A, y, args.epochs, args.lr)
            a0 = acc(E0, P, A, y)
            fit(E0, P, B, y, args.epochs, args.lr)
            n0 += a0; npost += acc(E0, P, A, y)

        sd = args.seeds
        jm, ja, jb, n0, npost = jm/sd, ja/sd, jb/sd, n0/sd, npost/sd
        print(f"{h:>6} {sharedW:>8} {jm:>10.3f} {ja:>8.3f} {jb:>8.3f} "
              f"{n0:>9.3f} {npost:>11.3f} {n0-npost:>7.3f}")


if __name__ == "__main__":
    main()
