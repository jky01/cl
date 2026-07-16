#!/usr/bin/env python3
"""Rule + atomic exceptions: does retention decompose into free rule-transfer vs paid exceptions? (torch/GPU)

Structured sweep (s3/toy_structured.py) showed retention = reconstruction (rule found from later data)
+ retained state + loss, mixed by compressibility. This tests codex's decomposed storage law: put a
low-rank RULE plus a fraction rho of pure ATOMIC EXCEPTIONS (a pair's label replaced by a uniformly
sampled DIFFERENT class), and ask where retention comes from.

Predictions (codex):
  rule-following A pairs      -> reconstructed from B, budget-insensitive (found the function)
  exception A pairs in referee-> retained (directly supported)
  exception A pairs NOT in ref-> collapse to B-only/guessing (atomic, unreconstructible)
So the referee's value localizes onto exceptions; rule pairs are ~free.

Selectors: stratified-random, low-margin, and an EXCEPTION-ORACLE (knows the mask; illegal upper
bound giving the clean storage frontier). Reports the five-way A split + B-only reconstruction.
"""
import argparse
import collections
import torch
import torch.nn.functional as F


def make_generator(seed, n_ent, n_rel, C, de, dr, k, rho, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_e = torch.randn(n_ent, de, generator=g)
    x_r = torch.randn(n_rel, dr, generator=g)
    A = torch.randn(k, de, generator=g) / de ** 0.5
    Bm = torch.randn(k, dr, generator=g) / dr ** 0.5
    U = torch.randn(C, k, generator=g)
    le = x_e @ A.T
    lr = x_r @ Bm.T
    scores = (le[:, None, :] * lr[None, :, :]) @ U.T
    scores = scores - scores.mean(dim=(0, 1), keepdim=True)
    y_rule = scores.argmax(-1)                                   # [n_ent, n_rel]
    # atomic exceptions: replace rho fraction with a uniformly sampled DIFFERENT class
    npair = n_ent * n_rel
    exc_flat = torch.zeros(npair, dtype=torch.bool)
    nexc = int(round(rho * npair))
    sel = torch.randperm(npair, generator=g)[:nexc]
    exc_flat[sel] = True
    y = y_rule.clone().flatten()
    if nexc:
        off = torch.randint(1, C, (nexc,), generator=g)          # 1..C-1 shift => different class
        y[sel] = (y[sel] + off) % C
    y = y.reshape(n_ent, n_rel)
    exc = exc_flat.reshape(n_ent, n_rel)
    return x_e.to(device), x_r.to(device), y.to(device), exc.to(device)


def make_student(seed, de, dr, h, C, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 555)
    return {
        "W1": (torch.randn(de + dr, h, generator=g) / (de + dr) ** 0.5).to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }


def fwd(x_e, x_r, P, pairs):
    feat = torch.cat([x_e[pairs[:, 0]], x_r[pairs[:, 1]]], dim=1)
    return torch.relu(feat @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


def labels_of(y, pairs):
    return y[pairs[:, 0], pairs[:, 1]]


def fit(x_e, x_r, P, pairs, y, epochs, lr):
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    yl = labels_of(y, pairs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(x_e, x_r, P, pairs), yl).backward()
        opt.step()


@torch.no_grad()
def acc(x_e, x_r, P, pairs, y):
    if len(pairs) == 0:
        return float("nan")
    return (fwd(x_e, x_r, P, pairs).argmax(1) == labels_of(y, pairs)).float().mean().item()


@torch.no_grad()
def margins_of(x_e, x_r, P, pairs):
    lg = fwd(x_e, x_r, P, pairs)
    t2 = lg.topk(2, 1).values
    return t2[:, 0] - t2[:, 1]


def cloneP(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def ce_fold(x_e, x_r, Pinit, refA, refpseudo, B, y, alpha, epochs, lr):
    P = cloneP(Pinit)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    yB = labels_of(y, B)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fwd(x_e, x_r, P, B), yB)
        if len(refA) > 0:
            loss = loss + alpha * F.cross_entropy(fwd(x_e, x_r, P, refA), refpseudo)
        loss.backward()
        opt.step()
    return P


def run(seed, rho, args, device):
    n_ent, n_rel, C = args.n_ent, args.n_rel, args.C
    x_e, x_r, y, exc = make_generator(seed, n_ent, n_rel, C, args.de, args.dr, args.k, rho, device)
    ee, rr = torch.meshgrid(torch.arange(n_ent), torch.arange(n_rel), indexing="ij")
    pairs = torch.stack([ee.flatten(), rr.flatten()], 1)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    pairs = pairs[torch.randperm(pairs.shape[0], generator=g)].to(device)
    half = pairs.shape[0] // 2
    A, B = pairs[:half], pairs[half:2 * half]
    A_is_exc = exc[A[:, 0], A[:, 1]]                         # bool over A
    A_rule = A[~A_is_exc]
    A_exc = A[A_is_exc]

    # learn A (rule + its exceptions)
    P = make_student(seed, args.de, args.dr, args.h, C, device)
    fit(x_e, x_r, P, A, y, args.epochs, args.lr)
    with torch.no_grad():
        A_pseudo = fwd(x_e, x_r, P, A).argmax(1)
    A_marg = margins_of(x_e, x_r, P, A)

    # B-only reconstruction control
    Pbo = make_student(seed + 33, args.de, args.dr, args.h, C, device)
    fit(x_e, x_r, Pbo, B, y, args.epochs, args.lr)
    bo = (acc(x_e, x_r, Pbo, A_rule, y), acc(x_e, x_r, Pbo, A_exc, y))

    out = {"bonly": bo, "n_exc": A_exc.shape[0], "n_rule": A_rule.shape[0], "nA": A.shape[0]}

    order_low = torch.argsort(A_marg)                       # ascending margin (positions in A)
    exc_pos = torch.nonzero(A_is_exc, as_tuple=False).flatten()   # positions of exceptions in A

    for budget in args.budgets:
        m = int(round(budget * A.shape[0]))
        for sel in ["random", "lowmargin", "excoracle"]:
            if m == 0:
                idx = torch.arange(0, device=device)
            elif sel == "random":
                idx = torch.randperm(A.shape[0], device=device)[:m]
            elif sel == "lowmargin":
                idx = order_low[:m]
            else:  # excoracle: fill with exceptions first, then random rule pairs
                exc_sh = exc_pos[torch.randperm(exc_pos.shape[0], device=device)]
                if exc_sh.shape[0] >= m:
                    idx = exc_sh[:m]
                else:
                    rest = torch.tensor([i for i in range(A.shape[0]) if not A_is_exc[i]],
                                        device=device)
                    rest = rest[torch.randperm(rest.shape[0], device=device)[:m - exc_sh.shape[0]]]
                    idx = torch.cat([exc_sh, rest])
            refA = A[idx] if m else A[:0]
            pseudo = A_pseudo[idx] if m else A_pseudo[:0]
            in_ref = torch.zeros(A.shape[0], dtype=torch.bool, device=device)
            if m:
                in_ref[idx] = True
            exc_in = A[A_is_exc & in_ref]
            exc_out = A[A_is_exc & ~in_ref]
            Pf = ce_fold(x_e, x_r, P, refA, pseudo, B, y, args.alpha, args.fold_epochs, args.lr)
            out[(budget, sel)] = (
                acc(x_e, x_r, Pf, A_rule, y),      # rule-following A
                acc(x_e, x_r, Pf, exc_in, y),      # exceptions in referee
                acc(x_e, x_r, Pf, exc_out, y),     # exceptions out of referee
                acc(x_e, x_r, Pf, A, y),           # all A
                acc(x_e, x_r, Pf, B, y),           # B
                exc_in.shape[0], exc_out.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ent", type=int, default=80)
    ap.add_argument("--n_rel", type=int, default=80)
    ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--de", type=int, default=32)
    ap.add_argument("--dr", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--fold_epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.1, 0.5])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} RULE+EXCEPTIONS k={args.k} n_ent={args.n_ent} n_rel={args.n_rel} "
          f"C={args.C} h={args.h} epochs={args.epochs} seeds={args.seeds}")

    for rho in args.rhos:
        agg = collections.defaultdict(list)
        for s in range(args.seeds):
            for kk, v in run(s, rho, args, device).items():
                agg[kk].append(v)
        bo = torch.tensor(agg["bonly"]).mean(0).tolist()
        nrule = sum(agg["n_rule"]) / len(agg["n_rule"])
        nexc = sum(agg["n_exc"]) / len(agg["n_exc"])
        print(f"\n=== rho={rho}  (A: ~{nrule:.0f} rule, ~{nexc:.0f} exc)  "
              f"B-only recon: rule={bo[0]:.3f} exc={bo[1]:.3f} ===")
        print(f"{'budget':>7} {'sel':>10} {'ruleA':>7} {'excIn':>7} {'excOut':>7} "
              f"{'allA':>7} {'B':>7}")
        for budget in args.budgets:
            for sel in ["random", "lowmargin", "excoracle"]:
                rows = torch.tensor(agg[(budget, sel)])
                rA, eIn, eOut, aA, b = rows[:, :5].nanmean(0).tolist()
                print(f"{budget:>7.2f} {sel:>10} {rA:>7.3f} {eIn:>7.3f} {eOut:>7.3f} "
                      f"{aA:>7.3f} {b:>7.3f}")
    print("\nprediction: ruleA ~ budget-insensitive (reconstructed); excOut ~ B-only floor "
          "(atomic, unreconstructible); excIn ~ retained; excoracle maximizes allA at low budget.")


if __name__ == "__main__":
    main()
