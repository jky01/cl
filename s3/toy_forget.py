#!/usr/bin/env python3
"""Toy continual-learning testbed (pure numpy, CPU, seconds).

Minimal model of "read facts, retain them": memorize key -> value associations
in a tiny MLP with a PER-KEY embedding (the local "address") feeding a SHARED
hidden layer + output head (the shared substrate through which interference
happens). Sequential tasks A (odd keys) then B (even keys); measure whether A
is still remembered after B.

Arms:
  naive   : train A (all params), then B (all params).            -> expect forgetting
  joint   : train A U B together.                                 -> oracle upper bound
  replay  : train B with a fraction of A rehearsed alongside.     -> rehearsal control
  local   : train A (all params); for B freeze shared W1/b1/W2/b2 and update
            only the embeddings of B's keys.                       -> concept-local write

This does NOT test our real open problems (label-free write signal; transfer via
shared coordinates) -- odd/even share no exploitable structure. It is a harness
shakedown + a forgetting/stability-plasticity baseline we can iterate on in seconds.
"""
import argparse
import numpy as np


def build(seed, N, C, d, h):
    rng = np.random.default_rng(seed)
    P = {
        "E": rng.normal(0, 0.1, (N, d)),
        "W1": rng.normal(0, 1 / np.sqrt(d), (d, h)),
        "b1": np.zeros(h),
        "W2": rng.normal(0, 1 / np.sqrt(h), (h, C)),
        "b2": np.zeros(C),
    }
    targets = rng.integers(0, C, size=N)          # key -> gold class
    return P, targets, rng


def forward(P, keys):
    e = P["E"][keys]                               # [B, d]
    pre = e @ P["W1"] + P["b1"]
    z = np.maximum(pre, 0.0)                        # relu
    logits = z @ P["W2"] + P["b2"]
    cache = (keys, e, pre, z, logits)
    return logits, cache


def softmax_ce_grad(logits, y):
    m = logits.max(1, keepdims=True)
    ex = np.exp(logits - m)
    sm = ex / ex.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(sm[np.arange(n), y] + 1e-12).mean()
    d_logits = sm.copy()
    d_logits[np.arange(n), y] -= 1.0
    d_logits /= n
    return loss, d_logits


def backward(P, cache, d_logits, train_keys_mask=None, frozen=()):
    keys, e, pre, z, logits = cache
    g = {}
    g["W2"] = z.T @ d_logits
    g["b2"] = d_logits.sum(0)
    dz = d_logits @ P["W2"].T
    dpre = dz * (pre > 0)
    g["W1"] = e.T @ dpre
    g["b1"] = dpre.sum(0)
    de = dpre @ P["W1"].T                           # [B, d] grad wrt looked-up embeddings
    gE = np.zeros_like(P["E"])
    gE[keys] = de                                   # keys unique within a batch -> direct assign
    g["E"] = gE
    for k in frozen:
        g[k] = np.zeros_like(P[k])
    return g


class Adam:
    def __init__(self, P, lr):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    def step(self, P, g):
        self.t += 1
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * (g[k] ** 2)
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(P, keys, targets, epochs, lr, frozen=(), embed_only_keys=None):
    opt = Adam(P, lr)
    y = targets[keys]
    for _ in range(epochs):
        logits, cache = forward(P, keys)
        _, dl = softmax_ce_grad(logits, y)
        g = backward(P, cache, dl, frozen=tuple(frozen))
        if embed_only_keys is not None:
            # only allow embedding rows of the given keys to move; freeze everything else
            gE = np.zeros_like(P["E"])
            gE[embed_only_keys] = g["E"][embed_only_keys]
            g = {k: (gE if k == "E" else np.zeros_like(P[k])) for k in P}
        opt.step(P, g)


def acc(P, keys, targets):
    if len(keys) == 0:
        return float("nan")
    logits, _ = forward(P, keys)
    return (logits.argmax(1) == targets[keys]).mean()


def run(seed, N, C, d, h, epochs, lr, replay_frac):
    P0, targets, rng = build(seed, N, C, d, h)
    keys = np.arange(N)
    A = keys[keys % 2 == 1]     # odd
    B = keys[keys % 2 == 0]     # even
    out = {}

    def clone(P):
        return {k: v.copy() for k, v in P.items()}

    # naive sequential
    P = clone(P0)
    train(P, A, targets, epochs, lr)
    accA_postA = acc(P, A, targets)
    train(P, B, targets, epochs, lr)
    out["naive"] = (accA_postA, acc(P, A, targets), acc(P, B, targets))

    # joint oracle
    P = clone(P0)
    train(P, keys, targets, epochs, lr)
    out["joint"] = (float("nan"), acc(P, A, targets), acc(P, B, targets))

    # replay
    P = clone(P0)
    train(P, A, targets, epochs, lr)
    aA = acc(P, A, targets)
    nrep = int(len(A) * replay_frac)
    rep = rng.choice(A, size=nrep, replace=False) if nrep else np.array([], int)
    mix = np.concatenate([B, rep]).astype(int)
    train(P, mix, targets, epochs, lr)
    out["replay"] = (aA, acc(P, A, targets), acc(P, B, targets))

    # localized write: freeze shared substrate after A, only B embeddings move
    P = clone(P0)
    train(P, A, targets, epochs, lr)
    aA = acc(P, A, targets)
    train(P, B, targets, epochs, lr, embed_only_keys=B)
    out["local"] = (aA, acc(P, A, targets), acc(P, B, targets))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--C", type=int, default=100)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--replay_frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"N={args.N} C={args.C} d={args.d} h={args.h} epochs={args.epochs} "
          f"lr={args.lr} replay_frac={args.replay_frac} seeds={args.seeds}")
    print(f"{'arm':8} {'accA_postA':>11} {'accA_postB':>11} {'accB_postB':>11} {'forget(A)':>10}")
    agg = {}
    for s in range(args.seeds):
        r = run(s, args.N, args.C, args.d, args.h, args.epochs, args.lr, args.replay_frac)
        for arm, (aa, ab, bb) in r.items():
            agg.setdefault(arm, []).append((aa, ab, bb))
    for arm in ["naive", "replay", "local", "joint"]:
        rows = np.array(agg[arm])
        aa, ab, bb = np.nanmean(rows, 0)
        forget = (aa - ab) if not np.isnan(aa) else float("nan")
        print(f"{arm:8} {aa:11.3f} {ab:11.3f} {bb:11.3f} {forget:10.3f}")


if __name__ == "__main__":
    main()
