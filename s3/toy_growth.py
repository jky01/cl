#!/usr/bin/env python3
"""Growth intervention: does width-growth convert a STRUCTURED residual into unseen generalization,
while giving only linear storage for an ATOMIC residual? (torch/GPU)

Decisive contrast (codex): capacity growth is warranted only when a residual is persistently
surprising AND jointly compressible, and successful only when it turns the residual into MODEL-ONLY
generalization on UNSEEN inputs without sacrificing the old rule.

Phase 1: learn rule-1 (low-rank bilinear, k) on clean rule-1 pairs into shared weights (h0).
Phase 2 (residual): train a residual set + a matched rule-1 referee, all plastic. Residual is either
  - atomic  : random-different-class labels on residual pairs (incompressible), or
  - rule2   : a SECOND independent low-rank rule on residual pairs (compressible).
Residual pairs are split into residual-TRAIN (supported) and residual-UNSEEN (held-out inputs, never
trained) so we can measure generalization vs mere storage.

Capacity arms:
  fixed  : stay at h0 through phase 2.
  grow   : function-preserving expand h0 -> H at the phase boundary (new units: random W1 cols,
           ZERO W2 rows -> zero immediate loss jump), then train phase 2.
  wide   : width H from the start (controls for 'benefit of capacity' vs 'benefit of when added').

Eval is MODEL-ONLY (no referee): rule1-unseen retention, residual-train acc, residual-UNSEEN acc.
Prediction:  grow/wide x rule2 -> residual-UNSEEN rises (discovered 2nd rule), rule1 kept.
             any capacity x atomic -> residual-UNSEEN ~ chance (no generalization).
"""
import argparse
import collections
import torch
import torch.nn.functional as F


def bilinear_labels(g, n_ent, n_rel, C, de, dr, k, x_e, x_r):
    A = torch.randn(k, de, generator=g) / de ** 0.5
    Bm = torch.randn(k, dr, generator=g) / dr ** 0.5
    U = torch.randn(C, k, generator=g)
    sc = (x_e @ A.T)[:, None, :] * (x_r @ Bm.T)[None, :, :]
    sc = sc @ U.T
    sc = sc - sc.mean(dim=(0, 1), keepdim=True)
    return sc.argmax(-1)


def make_world(seed, args, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_ent, n_rel, C = args.n_ent, args.n_rel, args.C
    x_e = torch.randn(n_ent, args.de, generator=g)
    x_r = torch.randn(n_rel, args.dr, generator=g)
    y1 = bilinear_labels(g, n_ent, n_rel, C, args.de, args.dr, args.k, x_e, x_r)   # rule 1
    y2 = bilinear_labels(g, n_ent, n_rel, C, args.de, args.dr, args.k, x_e, x_r)   # rule 2 (indep)
    return x_e.to(device), x_r.to(device), y1.to(device), y2.to(device)


def make_student(seed, din, h, C, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 555)
    return {
        "W1": (torch.randn(din, h, generator=g) / din ** 0.5).to(device).requires_grad_(),
        "b1": torch.zeros(h, device=device, requires_grad=True),
        "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
        "b2": torch.zeros(C, device=device, requires_grad=True),
    }


def grow_width(P, add, seed, device):
    """Function-preserving width expansion: new hidden units get random W1 cols, ZERO W2 rows."""
    g = torch.Generator(device="cpu").manual_seed(seed + 999)
    din, h = P["W1"].shape
    newW1 = (torch.randn(din, add, generator=g) / din ** 0.5).to(device)
    W1 = torch.cat([P["W1"].detach(), newW1], dim=1)
    b1 = torch.cat([P["b1"].detach(), torch.zeros(add, device=device)])
    W2 = torch.cat([P["W2"].detach(), torch.zeros(add, P["W2"].shape[1], device=device)], dim=0)
    return {"W1": W1.requires_grad_(), "b1": b1.requires_grad_(),
            "W2": W2.requires_grad_(), "b2": P["b2"].detach().requires_grad_()}


def fwd(x_e, x_r, P, pairs):
    feat = torch.cat([x_e[pairs[:, 0]], x_r[pairs[:, 1]]], dim=1)
    return torch.relu(feat @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


def fit(x_e, x_r, P, pairs, labels, epochs, lr):
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(x_e, x_r, P, pairs), labels).backward()
        opt.step()


@torch.no_grad()
def acc(x_e, x_r, P, pairs, labels):
    if len(pairs) == 0:
        return float("nan")
    return (fwd(x_e, x_r, P, pairs).argmax(1) == labels).float().mean().item()


def run(seed, residual_type, capacity, args, device):
    x_e, x_r, y1, y2 = make_world(seed, args, device)
    n_ent, n_rel = args.n_ent, args.n_rel
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ee, rr = torch.meshgrid(torch.arange(n_ent), torch.arange(n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1)
    allp = allp[torch.randperm(allp.shape[0], generator=g)].to(device)

    n = allp.shape[0]
    n_res = int(args.residual_frac * n)
    res_pairs = allp[:n_res]
    rule1_pairs = allp[n_res:]
    # residual train / unseen split
    n_res_tr = n_res // 2
    res_tr, res_un = res_pairs[:n_res_tr], res_pairs[n_res_tr:]
    # rule1 train / unseen split (unseen = generalization retention probe)
    n_r1 = rule1_pairs.shape[0]
    r1_tr, r1_un = rule1_pairs[:n_r1 * 3 // 4], rule1_pairs[n_r1 * 3 // 4:]

    din = args.de + args.dr
    h0, H = args.h0, args.H
    start_h = H if capacity == "wide" else h0

    def lab(pairs, which):
        return (y1 if which == 1 else y2)[pairs[:, 0], pairs[:, 1]]

    def res_lab(pairs):
        return atomic_lab(pairs) if residual_type == "atomic" else lab(pairs, 2)

    # atomic labels: fixed per seed, different-from-rule1 class
    gg = torch.Generator(device="cpu").manual_seed(seed + 55)
    off_all = torch.randint(1, args.C, (n_res,), generator=gg).to(device)
    off_map = {}
    for i in range(n_res):
        off_map[(res_pairs[i, 0].item(), res_pairs[i, 1].item())] = off_all[i].item()

    def atomic_lab(pairs):
        base = lab(pairs, 1)
        offs = torch.tensor([off_map[(p[0].item(), p[1].item())] for p in pairs], device=device)
        return (base + offs) % args.C

    # phase 1: learn rule 1 on r1_tr (clean)
    P = make_student(seed, din, start_h, args.C, device)
    fit(x_e, x_r, P, r1_tr, lab(r1_tr, 1), args.epochs, args.lr)
    r1_before = acc(x_e, x_r, P, r1_un, lab(r1_un, 1))

    # phase boundary: grow if requested
    if capacity == "grow":
        P = grow_width(P, H - h0, seed, device)

    # phase 2: residual train + matched rule1 referee (to retain rule 1), all plastic
    ref_n = int(args.ref_frac * r1_tr.shape[0])
    ref = r1_tr[torch.randperm(r1_tr.shape[0], device=device)[:ref_n]]
    p2_pairs = torch.cat([res_tr, ref])
    p2_labels = torch.cat([res_lab(res_tr), lab(ref, 1)])
    fit(x_e, x_r, P, p2_pairs, p2_labels, args.fold_epochs, args.lr)

    # model-only eval (no referee at inference)
    return {
        "r1_before": r1_before,
        "r1_retain": acc(x_e, x_r, P, r1_un, lab(r1_un, 1)),       # rule1 generalization kept
        "res_train": acc(x_e, x_r, P, res_tr, res_lab(res_tr)),    # supported residual
        "res_unseen": acc(x_e, x_r, P, res_un, res_lab(res_un)),   # residual GENERALIZATION (money)
        "params": P["W1"].numel() + P["W2"].numel(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ent", type=int, default=90)
    ap.add_argument("--n_rel", type=int, default=90)
    ap.add_argument("--C", type=int, default=40)
    ap.add_argument("--de", type=int, default=32)
    ap.add_argument("--dr", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--h0", type=int, default=128)
    ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--residual_frac", type=float, default=0.34)
    ap.add_argument("--ref_frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--fold_epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} GROWTH k={args.k} h0={args.h0} H={args.H} "
          f"residual_frac={args.residual_frac} ref_frac={args.ref_frac} seeds={args.seeds}")
    print(f"{'residual':>9} {'capacity':>9} {'r1_retain':>10} {'res_train':>10} "
          f"{'res_unseen':>11} {'params':>8}")
    for rtype in ["atomic", "rule2"]:
        for cap in ["fixed", "grow", "wide"]:
            agg = collections.defaultdict(list)
            for s in range(args.seeds):
                for kk, v in run(s, rtype, cap, args, device).items():
                    agg[kk].append(v)
            r1 = sum(agg["r1_retain"]) / args.seeds
            rt = sum(agg["res_train"]) / args.seeds
            ru = sum(agg["res_unseen"]) / args.seeds
            pr = int(sum(agg["params"]) / args.seeds)
            print(f"{rtype:>9} {cap:>9} {r1:>10.3f} {rt:>10.3f} {ru:>11.3f} {pr:>8}")
    print("\nmoney metric = res_unseen (model-only generalization to held-out residual inputs).")
    print("expect: rule2 x {grow,wide} >> rule2 x fixed on res_unseen; atomic res_unseen ~ chance "
          f"(~{1.0/args.C:.3f}) for all capacities.")


if __name__ == "__main__":
    main()
