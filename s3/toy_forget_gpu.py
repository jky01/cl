#!/usr/bin/env python3
"""Toy continual-learning testbed (torch / GPU).

Minimal model of "read facts, retain them": memorize key -> value associations.
Key -> representation -> SHARED hidden -> SHARED output head. Sequential tasks
A (odd keys) then B (even keys); measure whether A survives B.

Two representation regimes (the addressing contrast):
  --embed trainable : each key has its OWN trainable embedding row (private address).
  --embed frozen    : key rep is a FIXED random vector; ALL knowledge must live in
                      the shared W1/W2 -> forces interference (the realistic case).

Arms:
  naive  : train A (all trainable params), then B.                 -> forgetting?
  joint  : train A U B together.                                   -> oracle
  replay : train B with a fraction of A rehearsed.                 -> rehearsal
  ewc    : train B with a quadratic penalty anchoring params to their
           post-A values, weighted by the (diagonal) Fisher of task A. -> rehearsal-free reg
  local  : freeze shared substrate after A; only per-key params of B move
           (embed mode: B's embedding rows; frozen mode: a private low-rank
            slot per B key added to the hidden pre-activation).      -> concept-local write

Capacity is controlled by N (facts) vs h (shared width): to see catastrophic
forgetting, make the shared substrate the bottleneck (large N, small h, frozen embed).
"""
import argparse
import torch
import torch.nn.functional as F


def make(seed, N, C, d, h, device, frozen_embed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = torch.randn(N, d, generator=g) * (1.0 if frozen_embed else 0.1)
    W1 = torch.randn(d, h, generator=g) / d ** 0.5
    W2 = torch.randn(h, C, generator=g) / h ** 0.5
    targets = torch.randint(0, C, (N,), generator=g)
    P = {
        "E": E.to(device).requires_grad_(not frozen_embed),
        "W1": W1.to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": W2.to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }
    return P, targets.to(device)


def forward(P, keys):
    e = P["E"][keys]
    z = torch.relu(e @ P["W1"] + P["b1"])
    return z @ P["W2"] + P["b2"]


def trainable(P, names):
    return [P[n] for n in names]


def fit(P, keys, y, epochs, lr, names, penalty=None):
    """penalty: optional callable(P)->scalar added to loss (EWC)."""
    params = trainable(P, names)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(P, keys), y[keys])
        if penalty is not None:
            loss = loss + penalty(P)
        loss.backward()
        opt.step()


@torch.no_grad()
def acc(P, keys, y):
    if keys.numel() == 0:
        return float("nan")
    return (forward(P, keys).argmax(1) == y[keys]).float().mean().item()


def clone(P):
    return {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in P.items()}


def fisher_diag(P, keys, y, names, max_samples=256):
    """Diagonal empirical Fisher of task A (subsampled) for EWC."""
    if keys.numel() > max_samples:
        sub = keys[torch.randperm(keys.numel(), device=keys.device)[:max_samples]]
    else:
        sub = keys
    fish = {n: torch.zeros_like(P[n]) for n in names}
    for i in range(sub.numel()):
        for n in names:
            P[n].grad = None
        logits = forward(P, sub[i:i + 1])          # single-sample graph
        logp = F.log_softmax(logits, 1)
        logp[0, y[sub[i]]].backward()
        for n in names:
            if P[n].grad is not None:
                fish[n] += P[n].grad.detach() ** 2
    for n in names:
        fish[n] /= max(1, sub.numel())
    return fish


def run(seed, args, device):
    N, C, d, h = args.N, args.C, args.d, args.h
    frozen = args.embed == "frozen"
    all_names = ["W1", "b1", "W2", "b2"] + ([] if frozen else ["E"])
    P0, y = make(seed, N, C, d, h, device, frozen)
    keys = torch.arange(N, device=device)
    A = keys[keys % 2 == 1]
    B = keys[keys % 2 == 0]
    e, lr = args.epochs, args.lr
    out = {}

    # naive
    P = clone(P0)
    fit(P, A, y, e, lr, all_names)
    aA = acc(P, A, y)
    fit(P, B, y, e, lr, all_names)
    out["naive"] = (aA, acc(P, A, y), acc(P, B, y))

    # joint
    P = clone(P0)
    fit(P, keys, y, e, lr, all_names)
    out["joint"] = (float("nan"), acc(P, A, y), acc(P, B, y))

    # replay
    P = clone(P0)
    fit(P, A, y, e, lr, all_names)
    aA = acc(P, A, y)
    nrep = int(A.numel() * args.replay_frac)
    rep = A[torch.randperm(A.numel(), device=device)[:nrep]] if nrep else A[:0]
    mix = torch.cat([B, rep])
    fit(P, mix, y, e, lr, all_names)
    out["replay"] = (aA, acc(P, A, y), acc(P, B, y))

    # ewc (rehearsal-free regularizer)
    P = clone(P0)
    fit(P, A, y, e, lr, all_names)
    aA = acc(P, A, y)
    fish = fisher_diag(P, A, y, all_names)
    star = {n: P[n].detach().clone() for n in all_names}
    lam = args.ewc_lambda

    def penalty(P):
        s = 0.0
        for n in all_names:
            s = s + (fish[n] * (P[n] - star[n]) ** 2).sum()
        return 0.5 * lam * s

    fit(P, B, y, e, lr, all_names, penalty=penalty)
    out["ewc"] = (aA, acc(P, A, y), acc(P, B, y))

    # local: freeze shared substrate after A; only per-key params of B move
    P = clone(P0)
    fit(P, A, y, e, lr, all_names)
    aA = acc(P, A, y)
    if not frozen:
        # only B's embedding rows learn (private address)
        Efull = P["E"].detach()
        Emask = torch.zeros(N, 1, device=device)
        Emask[B] = 1.0
        Evar = (Efull * (1 - Emask)).clone()  # frozen part
        Btrain = (Efull * Emask).clone().requires_grad_()
        opt = torch.optim.Adam([Btrain], lr=lr)
        for _ in range(e):
            opt.zero_grad(set_to_none=True)
            P["E"] = Evar + Btrain * Emask
            F.cross_entropy(forward(P, B), y[B]).backward()
            opt.step()
        P["E"] = (Evar + Btrain * Emask).detach()
        out["local"] = (aA, acc(P, A, y), acc(P, B, y))
    else:
        # frozen-embed: give each B key a private additive hidden slot (rank-1 write),
        # shared W1/W2 frozen -> A untouched by construction.
        slot = torch.zeros(N, h, device=device, requires_grad=True)
        opt = torch.optim.Adam([slot], lr=lr)
        base_e = P["E"]
        for _ in range(e):
            opt.zero_grad(set_to_none=True)
            z = torch.relu(base_e[B] @ P["W1"] + P["b1"] + slot[B])
            F.cross_entropy(z @ P["W2"] + P["b2"], y[B]).backward()
            opt.step()

        def fwd_local(keys):
            z = torch.relu(base_e[keys] @ P["W1"] + P["b1"] + slot[keys])
            return z @ P["W2"] + P["b2"]
        with torch.no_grad():
            accA = (fwd_local(A).argmax(1) == y[A]).float().mean().item()
            accB = (fwd_local(B).argmax(1) == y[B]).float().mean().item()
        out["local"] = (aA, accA, accB)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--replay_frac", type=float, default=0.25)
    ap.add_argument("--ewc_lambda", type=float, default=1e3)
    ap.add_argument("--embed", choices=["trainable", "frozen"], default="frozen")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} N={args.N} C={args.C} d={args.d} h={args.h} "
          f"epochs={args.epochs} lr={args.lr} embed={args.embed} "
          f"replay_frac={args.replay_frac} ewc_lambda={args.ewc_lambda} seeds={args.seeds}")
    print(f"{'arm':8} {'accA_postA':>11} {'accA_postB':>11} {'accB_postB':>11} {'forget(A)':>10}")
    agg = {}
    for s in range(args.seeds):
        r = run(s, args, device)
        for arm, v in r.items():
            agg.setdefault(arm, []).append(v)
    for arm in ["naive", "replay", "ewc", "local", "joint"]:
        rows = torch.tensor(agg[arm])
        aa, ab, bb = rows.nanmean(0).tolist()
        forget = (aa - ab) if aa == aa else float("nan")
        print(f"{arm:8} {aa:11.3f} {ab:11.3f} {bb:11.3f} {forget:10.3f}")


if __name__ == "__main__":
    main()
