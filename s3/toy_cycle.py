#!/usr/bin/env python3
"""Multi-acquisition cycle: does grow->consolidate repeated over several residual rounds keep the
width FLAT (bounded params) and NOT erode earlier rules? (torch/GPU)

Entities are partitioned into G groups by a descriptor quantile (input-identifiable regions). Each
group g has its own independent bilinear rule y_g; a pair (e,r)'s label is y_{group(e)}(e,r).

Round 0: train flat M_0 (width h0) on group-0.
Round k: grow M_{k-1} (h0->H, function-preserving); train a gated centered correction branch + learned
  router to ADD group-k (residual=group-k true labels; referee=old groups self-distilled from M_{k-1},
  no stored old data); teacher=gated combined; CONSOLIDATE by distilling the teacher over the input
  manifold + group-k true labels into a fresh flat M_k of width h0. Delete branch+router.

After each round report width and held-out accuracy on EVERY group seen so far. Claim: width stays h0
(params bounded), group-0..k-1 accuracy does not erode, group-k is acquired.
"""
import argparse
import sys
import os
import collections
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_route as tr


def flat_logits(P, x_e, x_r, pairs):
    f = tr.feats(x_e, x_r, pairs)
    return torch.relu(f @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


@torch.no_grad()
def flat_acc(P, x_e, x_r, pairs, labels):
    if len(pairs) == 0:
        return float("nan")
    return (flat_logits(P, x_e, x_r, pairs).argmax(1) == labels).float().mean().item()


def distill_flat(x_e, x_r, pairs, labels, W, din, C, epochs, lr, seed, device):
    P = tr.init_params(seed, din, W, C, device)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(flat_logits(P, x_e, x_r, pairs), labels).backward()
        opt.step()
    return P


def run(seed, args, device):
    G, C = args.groups, args.C
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_e = torch.randn(args.n_ent, args.de, generator=g)
    x_r = torch.randn(args.n_rel, args.dr, generator=g)
    ys = [tr.bilinear(g, C, args.de, args.dr, args.k, x_e, x_r) for _ in range(G)]  # per-group rules
    x_e, x_r = x_e.to(device), x_r.to(device)
    ys = [y.to(device) for y in ys]
    # group of each entity by quantile band of descriptor 0
    q = x_e[:, 0]
    edges = torch.quantile(q, torch.linspace(0, 1, G + 1, device=device))
    grp_of_ent = torch.bucketize(q, edges[1:-1].contiguous())          # 0..G-1

    ee, rr = torch.meshgrid(torch.arange(args.n_ent), torch.arange(args.n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1).to(device)
    pair_grp = grp_of_ent[allp[:, 0]]

    def true_lab(pairs):
        gp = grp_of_ent[pairs[:, 0]]
        out = torch.empty(len(pairs), dtype=torch.long, device=device)
        for gi in range(G):
            m = gp == gi
            if m.any():
                out[m] = ys[gi][pairs[m][:, 0], pairs[m][:, 1]]
        return out

    # per-group train/held-out pairs
    gpairs, gtrain, gtest = [], [], []
    for gi in range(G):
        p = allp[pair_grp == gi]
        p = p[torch.randperm(p.shape[0], device=device)]
        cut = p.shape[0] * 3 // 4
        gpairs.append(p); gtrain.append(p[:cut]); gtest.append(p[cut:])

    din, h0, H = args.de + args.dr, args.h0, args.H
    # round 0
    M = tr.init_params(seed + 1, din, h0, C, device)
    tr.fit_old(x_e, x_r, M, gtrain[0], true_lab(gtrain[0]), args.epochs, args.lr)

    rows = []  # (round, width, [acc per group so far])
    rows.append((0, M["W1"].shape[1], [flat_acc(M, x_e, x_r, gtest[0], true_lab(gtest[0]))]))

    for k in range(1, G):
        # grow M_{k-1} -> H
        P = tr.grow(M, H - h0, seed + k, device)
        res_tr = gtrain[k]
        # referee: sample of old groups, self-distilled from M_{k-1} (no stored old labels)
        old_pool = torch.cat([gtrain[j] for j in range(k)])
        ref = old_pool[torch.randperm(old_pool.shape[0], device=device)[:int(args.ref_frac * old_pool.shape[0])]]
        with torch.no_grad():
            ref_lab = flat_logits(M, x_e, x_r, ref).argmax(1)
        tr.fit_correction(x_e, x_r, P, h0, res_tr, true_lab(res_tr), ref, ref_lab,
                          args.fold_epochs, args.lr, args.lam_s, args.lam_c)
        router = tr.train_router(x_e, x_r, res_tr, ref, din, args.router_epochs, args.lr, device, seed + k)

        @torch.no_grad()
        def teacher_pred(pairs):
            zo = tr.center(tr.z_old(x_e, x_r, P, pairs, h0))
            pd = tr.center(tr.delta(x_e, x_r, P, pairs, h0))
            return (zo + router(pairs).view(-1, 1) * pd).argmax(1)

        # consolidate: self-distill teacher on the OLD region (groups 0..k-1, strong) + group-k under
        # its TRUE labels (don't let the gated branch's weak group-k prediction cap residual quality).
        old_region = allp[pair_grp < k]
        cons_pairs = torch.cat([old_region, res_tr])
        cons_labels = torch.cat([teacher_pred(old_region), true_lab(res_tr)])
        M = distill_flat(x_e, x_r, cons_pairs, cons_labels, h0, din, C,
                         args.distill_epochs, args.lr, seed + 100 + k, device)
        accs = [flat_acc(M, x_e, x_r, gtest[j], true_lab(gtest[j])) for j in range(k + 1)]
        rows.append((k, M["W1"].shape[1], accs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ent", type=int, default=120); ap.add_argument("--n_rel", type=int, default=120)
    ap.add_argument("--C", type=int, default=40)
    ap.add_argument("--de", type=int, default=32); ap.add_argument("--dr", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--h0", type=int, default=160); ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--ref_frac", type=float, default=0.3)
    ap.add_argument("--lam_s", type=float, default=0.03); ap.add_argument("--lam_c", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=2500); ap.add_argument("--fold_epochs", type=int, default=2500)
    ap.add_argument("--router_epochs", type=int, default=1200)
    ap.add_argument("--distill_epochs", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} CYCLE groups={args.groups} h0={args.h0} H(grow)={args.H} "
          f"seeds={args.seeds} chance={1.0/args.C:.3f}")

    # aggregate: per round, width and per-group acc averaged over seeds
    accum = collections.defaultdict(lambda: collections.defaultdict(list))
    widths = {}
    for s in range(args.seeds):
        for (rnd, width, accs) in run(s, args, device):
            widths[rnd] = width
            for gi, a in enumerate(accs):
                accum[rnd][gi].append(a)

    hdr = "round  width  " + "  ".join([f"grp{g}" for g in range(args.groups)])
    print(hdr)
    for rnd in range(args.groups):
        cells = []
        for gi in range(args.groups):
            if gi in accum[rnd]:
                cells.append(f"{sum(accum[rnd][gi]) / len(accum[rnd][gi]):.3f}")
            else:
                cells.append("  -  ")
        print(f"{rnd:>5}  {widths[rnd]:>5}  " + "  ".join(cells))
    print("\nwidth stays h0 across rounds => params bounded (grow is temporary). "
          "group-0..k-1 columns not eroding down the rows => no catastrophic forgetting across "
          "acquisitions. diagonal (newly added group) high => acquisition works each round.")


if __name__ == "__main__":
    main()
