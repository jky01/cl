#!/usr/bin/env python3
"""Algorithmic micro-language generator + environment interventions (LM port; see docs/LM_PORT_PREREG.md).

Typed sequence transduction. An example is  CMD_op  x_1..x_L  SEP  y_1..y_L  with y = op(x). Operators:
old primitives {copy, inc, shift}; new stateful {csum_reset}; interference {rmax_reset}; plus
compositions B(A(x)). Environments are TARGETED interventions (length / reset / alphabet / template)
that each break >=1 shortcut while preserving the latent program. Environment labels are returned for
training-objective use but must NOT be fed to the model at inference.

This file is the data layer only (no model). Deterministic given (seed, env, split).
"""
import argparse

# ---- vocabulary -------------------------------------------------------------
V = 16                                   # value tokens 0..V-1
RESET = V                                # reserved special: reset symbol (appears IN the value stream)
BOS, EOS, SEP = V + 1, V + 2, V + 3
CMD = {"copy": V + 4, "inc": V + 5, "shift": V + 6, "csum_reset": V + 7, "rmax_reset": V + 8}
# composition commands use a pair token per ordered pair we allow
VOCAB_SIZE = V + 9


def op_copy(x):
    return [t for t in x]


def op_inc(x):
    return [(t + 1) % V if t != RESET else RESET for t in x]


def op_shift(x):
    return [0] + x[:-1]


def op_csum_reset(x):
    s, y = 0, []
    for t in x:
        if t == RESET:
            s = 0; y.append(0)
        else:
            s = (s + t) % V; y.append(s)
    return y


def op_rmax_reset(x):
    s, y = 0, []
    for t in x:
        if t == RESET:
            s = 0; y.append(0)
        else:
            s = max(s, t); y.append(s)
    return y


OPS = {"copy": op_copy, "inc": op_inc, "shift": op_shift,
       "csum_reset": op_csum_reset, "rmax_reset": op_rmax_reset}


def apply_op(name, x):
    if "+" in name:                       # composition "A+B" means B(A(x))
        a, b = name.split("+")
        return OPS[b](OPS[a](x))
    return OPS[name](x)


# ---- environments (interventions) ------------------------------------------
# each returns a sampler for input sequences x under its intervention.
import random


def sample_input(rng, env, Lrange):
    L = rng.randint(*Lrange) if env in ("E_len", "E_reset", "E_alpha", "E_tmpl", "mix") else 8
    # alphabet subset (E_alpha uses two disjoint halves across sub-envs)
    if env == "E_alpha":
        half = rng.choice([0, 1])
        lo, hi = (0, V // 2) if half == 0 else (V // 2, V)
        vals = lambda: rng.randrange(lo, hi)
    else:
        vals = lambda: rng.randrange(0, V)
    x = [vals() for _ in range(L)]
    # reset injection
    if env == "E_reset":
        nres = rng.choice([0, 1, 2])
        for _ in range(nres):
            x[rng.randrange(0, L)] = RESET
    elif env in ("E_len", "E_alpha", "E_tmpl", "mix"):
        if rng.random() < 0.5:            # sometimes a single reset, position varied
            x[rng.randrange(0, L)] = RESET
    else:  # E0 base: at most one reset near the start
        if rng.random() < 0.5:
            x[rng.randrange(0, min(3, L))] = RESET
    return x


def render(cmd_name, x, y, env):
    """token sequence: BOS CMD x.. SEP y.. EOS. E_tmpl varies the delimiter (SEP vs BOS-repeat)."""
    sep = SEP if env != "E_tmpl" else BOS   # template intervention changes the delimiter token
    cmd = CMD[cmd_name] if "+" not in cmd_name else CMD[cmd_name.split("+")[1]]  # comp uses final-op cmd
    return [BOS, cmd] + x + [sep] + y + [EOS]


def make_examples(n, op_name, env, seed, Lrange=(3, 12)):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = sample_input(rng, env, Lrange)
        y = apply_op(op_name, x)
        out.append({"tokens": render(op_name, x, y, env), "x": x, "y": y,
                    "op": op_name, "env": env})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    print(f"VOCAB_SIZE={VOCAB_SIZE} V={V} RESET={RESET} SEP={SEP} CMD={CMD}")
    # smoke: verify each op on a hand example incl. reset, and show rendered tokens per env
    xtest = [3, 5, RESET, 2, 4, 7]
    for name in ["copy", "inc", "shift", "csum_reset", "rmax_reset", "inc+csum_reset", "csum_reset+shift"]:
        print(f"{name:>18}  x={xtest}  y={apply_op(name, xtest)}")
    print()
    for env in ["E0", "E_len", "E_reset", "E_alpha", "E_tmpl"]:
        ex = make_examples(args.n, "csum_reset", env, seed=0)
        Ls = [len(e['x']) for e in ex]
        nres = [e['x'].count(RESET) for e in ex]
        print(f"env={env:>8}  lens={Ls}  resets={nres}  sample_tokens={ex[0]['tokens']}")
    # verify csum_reset extrapolates by construction to long L (the OOD axis)
    rng = random.Random(1)
    xlong = [rng.randrange(0, V) for _ in range(40)]
    xlong[10] = RESET; xlong[25] = RESET
    y = op_csum_reset(xlong)
    ok = (y[11] == xlong[11] % V) and (y[26] == xlong[26] % V)   # first after each reset == that token
    print(f"\nOOD length-40 csum_reset post-reset check ok={ok} (y resets correctly at 40-len)")


if __name__ == "__main__":
    main()
