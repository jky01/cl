#!/usr/bin/env python3
"""Structured (compressible) continual-learning toy (torch/GPU).

The random-label sweep (s3/toy_referee_sweep.py) established the incompressible storage law
R(b) = b + (1-b)R(0): a referee protects exactly its own members, no more. This asks whether a
COMPRESSIBLE rule changes that -- i.e. whether a small referee gives ABOVE-CHORD retention
R(b) > R(0) + b[R(1)-R(0)], meaning a few anchors constrain many old decisions through shared weights.

Generator (codex's low-rank bilinear, description length controlled by rank k):
    frozen descriptors x_e in R^de, x_r in R^dr;  fixed A[k,de], Bm[k,dr], U[C,k]
    s_o(e,r) = U[o] . ((A x_e) (elementwise*) (Bm x_r));   y(e,r) = argmax_o s_o
Facts = all (e,r) pairs. A/B tasks share all entities & relations but have DISJOINT pairs.

Student: generic MLP on concat[frozen emb_e, frozen emb_r] -> shared hidden(h) -> C. Knowledge (the
rule) must live in the shared weights (the honest "can the learner DISCOVER the rule" case; a matched
factorized student is a later arm).

Fold = learn A into shared weights, then CE-distill on a referee subset of A (A's own argmax) while
fitting B labels; slots-free. Referee-budget sweep with random / low-margin selection.
Decomposition (codex): R(0)=zero-referee = A recovered by B-training transfer; R(1)=full-A ceiling.
Report full-A acc (primary), selected/unselected separately, B acc, and above-chord lift.
"""
import argparse
import collections
import torch
import torch.nn.functional as F


def make_generator(seed, n_ent, n_rel, C, de, dr, k, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_e = torch.randn(n_ent, de, generator=g)
    x_r = torch.randn(n_rel, dr, generator=g)
    A = torch.randn(k, de, generator=g) / de ** 0.5
    Bm = torch.randn(k, dr, generator=g) / dr ** 0.5
    U = torch.randn(C, k, generator=g)
    le = x_e @ A.T                     # [n_ent, k]
    lr = x_r @ Bm.T                    # [n_rel, k]
    inter = le[:, None, :] * lr[None, :, :]        # [n_ent, n_rel, k]
    scores = inter @ U.T                            # [n_ent, n_rel, C]
    scores = scores - scores.mean(dim=(0, 1), keepdim=True)   # remove per-class bias -> balanced argmax
    y = scores.argmax(-1)                           # [n_ent, n_rel]
    return x_e.to(device), x_r.to(device), y.to(device)


def make_student(seed, de, dr, h, C, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 555)
    P = {
        "W1": (torch.randn(de + dr, h, generator=g) / (de + dr) ** 0.5).to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }
    return P


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
def margins(x_e, x_r, P, pairs):
    lg = fwd(x_e, x_r, P, pairs)
    t2 = lg.topk(2, 1).values
    return t2[:, 0] - t2[:, 1]


def cloneP(P):
    return {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}


def ce_fold(x_e, x_r, Pinit, refA, refA_pseudo, B, y, alpha, epochs, lr):
    P = cloneP(Pinit)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    yB = labels_of(y, B)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fwd(x_e, x_r, P, B), yB)
        if len(refA) > 0:
            loss = loss + alpha * F.cross_entropy(fwd(x_e, x_r, P, refA), refA_pseudo)
        loss.backward()
        opt.step()
    return P


def run(seed, args, device):
    n_ent, n_rel, C = args.n_ent, args.n_rel, args.C
    x_e, x_r, y = make_generator(seed, n_ent, n_rel, C, args.de, args.dr, args.k, device)

    # class balance check
    counts = torch.bincount(y.flatten(), minlength=C).float()
    balance = (counts.max() / counts.clamp(min=1).float().mean()).item()

    # all pairs, split into disjoint A/B (shared entities & relations on both sides)
    gg = torch.Generator(device="cpu").manual_seed(seed + 1)
    ee, rr = torch.meshgrid(torch.arange(n_ent), torch.arange(n_rel), indexing="ij")
    pairs = torch.stack([ee.flatten(), rr.flatten()], 1)
    perm = torch.randperm(pairs.shape[0], generator=gg)
    pairs = pairs[perm].to(device)
    half = pairs.shape[0] // 2
    A = pairs[:half]
    B = pairs[half:2 * half]

    # joint capacity witness
    Pj = make_student(seed + 77, args.de, args.dr, args.h, C, device)
    fit(x_e, x_r, Pj, torch.cat([A, B]), y, args.epochs, args.lr)
    joint = (acc(x_e, x_r, Pj, A, y), acc(x_e, x_r, Pj, B, y))

    # B-only reconstruction control (codex): fresh init, train on B alone, eval A.
    # A-perf here = what the shared rule + B identifies with NO persistent A-specific state.
    Pbo = make_student(seed + 33, args.de, args.dr, args.h, C, device)
    fit(x_e, x_r, Pbo, B, y, args.epochs, args.lr)
    bonly = (acc(x_e, x_r, Pbo, A, y), acc(x_e, x_r, Pbo, B, y))

    # learn A into shared weights
    P = make_student(seed, args.de, args.dr, args.h, C, device)
    fit(x_e, x_r, P, A, y, args.epochs, args.lr)
    accA0 = acc(x_e, x_r, P, A, y)
    with torch.no_grad():
        A_pseudo = fwd(x_e, x_r, P, A).argmax(1)
    A_marg = margins(x_e, x_r, P, A)
    order_low = torch.argsort(A_marg)      # ascending margin (positions into A)

    out = {"joint": joint, "accA0": accA0, "balance": balance, "bonly": bonly}
    for budget in args.budgets:
        kk = int(round(budget * A.shape[0]))
        sels = ["random", "lowmargin"] if 0 < kk < A.shape[0] else ["all"]
        for sel in sels:
            if sel == "random":
                idx = torch.randperm(A.shape[0], device=device)[:kk]
            elif sel == "lowmargin":
                idx = order_low[:kk]
            else:
                idx = torch.arange(A.shape[0], device=device) if kk else torch.arange(0, device=device)
            refA = A[idx]
            pseudo = A_pseudo[idx]
            unsel = torch.ones(A.shape[0], dtype=torch.bool, device=device)
            unsel[idx] = False
            Pf = ce_fold(x_e, x_r, P, refA, pseudo, B, y, args.alpha, args.fold_epochs, args.lr)
            out[(budget, sel)] = (
                acc(x_e, x_r, Pf, A, y),                       # full-A (primary)
                acc(x_e, x_r, Pf, refA, y) if kk else float("nan"),   # selected
                acc(x_e, x_r, Pf, A[unsel], y),               # unselected
                acc(x_e, x_r, Pf, B, y))                      # B
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
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--fold_epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.02, 0.0])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} STRUCTURED rank_k={args.k} n_ent={args.n_ent} n_rel={args.n_rel} "
          f"C={args.C} h={args.h} alpha={args.alpha} epochs={args.epochs} seeds={args.seeds}")
    agg = collections.defaultdict(list)
    for s in range(args.seeds):
        for kk, v in run(s, args, device).items():
            agg[kk].append(v)

    joint = torch.tensor(agg["joint"]).mean(0).tolist()
    accA0 = sum(agg["accA0"]) / len(agg["accA0"])
    bal = sum(agg["balance"]) / len(agg["balance"])
    print(f"joint witness: accA={joint[0]:.3f} accB={joint[1]:.3f} | A-only accA0={accA0:.3f} "
          f"| class-balance(max/mean)={bal:.2f}")

    # gather full-A curve for chord
    def fullA(budget, sel):
        return torch.tensor(agg[(budget, sel)]).mean(0)[0].item()
    R1 = fullA(1.0, "all")
    R0 = fullA(0.0, "all")               # A->B, zero referee (warm from A solution)
    bonly = torch.tensor(agg["bonly"]).mean(0).tolist()
    retained = R0 - bonly[0]             # A-specific state left in weights beyond B-only reconstruction
    print(f"R(1)=full-A ceiling={R1:.3f}  R(0)=A->B zero-referee={R0:.3f}  "
          f"chord slope={(R1-R0):.3f}")
    print(f"DECOMP: B-only reconstruction(accA)={bonly[0]:.3f}  "
          f"retained-A-state(R0 - Bonly)={retained:+.3f}  "
          f"(B-only accB={bonly[1]:.3f})")
    print(f"\n{'budget':>7} {'sel':>10} {'fullA':>7} {'chord':>7} {'lift':>7} "
          f"{'selA':>7} {'unselA':>7} {'B':>7}")
    for budget in args.budgets:
        kk = int(round(budget * ((args.n_ent * args.n_rel) // 2)))
        sels = ["random", "lowmargin"] if 0 < kk < ((args.n_ent * args.n_rel) // 2) else ["all"]
        for sel in sels:
            fa, sa, ua, b = torch.tensor(agg[(budget, sel)]).mean(0).tolist()
            chord = R0 + budget * (R1 - R0)
            print(f"{budget:>7.2f} {sel:>10} {fa:>7.3f} {chord:>7.3f} {fa-chord:>+7.3f} "
                  f"{sa:>7.3f} {ua:>7.3f} {b:>7.3f}")
    print("above-chord lift > 0 => small referee protects MORE old decisions than its membership "
          "(compression via shared weights). ~0 => behaves like incompressible facts.")


if __name__ == "__main__":
    main()
