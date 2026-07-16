#!/usr/bin/env python3
"""Protected-growth matrix: can reserved/grown capacity absorb a compressible residual WITHOUT
overwriting the old rule? (torch/GPU)

toy_growth.py showed structured residual -> unseen generalization, atomic -> storage only, but all
capacities CRASHED rule-1 (all-plastic overwrite). codex: width alone is not continual retention;
test hard-freeze protected growth, and report OLD-BRANCH vs COMBINED separately (freezing the old
block does not guarantee the combined model's old output, because the new branch can override it).

Capacity arms (residual phase):
  fixed   : h0, all plastic.                         (baseline: overwrite)
  wide    : H from start, all plastic.               (capacity, no protection)
  protect : phase-1 at h0 (cheap); at boundary grow h0->H function-preserving (new W2 rows=0);
            phase-2 HARD-FREEZE old h0 block (+b2), train ONLY the new units on residual+referee.
  partition: H from start but old block=h0 (reserved block frozen at 0 in phase-1); phase-2 same
            freeze/train split as protect. Same final width & split; differs only in WHEN capacity
            was added (protect was small in phase-1). Isolates capacity-timing from capacity.

Residual: atomic exceptions vs a second low-rank rule2 (held-out UNSEEN inputs).

Five-way report (codex): (1) phase-1 rule1, (2) post old-branch-only rule1, (3) post COMBINED rule1
[the retention verdict], (4) new-branch logit norm on rule1 inputs [interference], (5) rule2 train/unseen.
Success (protect): combined rule1 ~= per-seed phase-1; rule2 unseen >> chance & > fixed; atomic unseen ~ chance.
"""
import argparse
import collections
import torch
import torch.nn.functional as F


def bilinear(g, C, de, dr, k, x_e, x_r):
    A = torch.randn(k, de, generator=g) / de ** 0.5
    Bm = torch.randn(k, dr, generator=g) / dr ** 0.5
    U = torch.randn(C, k, generator=g)
    sc = ((x_e @ A.T)[:, None, :] * (x_r @ Bm.T)[None, :, :]) @ U.T
    sc = sc - sc.mean(dim=(0, 1), keepdim=True)
    return sc.argmax(-1)


def make_world(seed, args, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_e = torch.randn(args.n_ent, args.de, generator=g)
    x_r = torch.randn(args.n_rel, args.dr, generator=g)
    y1 = bilinear(g, args.C, args.de, args.dr, args.k, x_e, x_r)
    y2 = bilinear(g, args.C, args.de, args.dr, args.k, x_e, x_r)
    return x_e.to(device), x_r.to(device), y1.to(device), y2.to(device)


def init_params(seed, din, h, C, device, zero_from=None):
    g = torch.Generator(device="cpu").manual_seed(seed + 555)
    W1 = (torch.randn(din, h, generator=g) / din ** 0.5).to(device)
    W2 = (torch.randn(h, C, generator=g) / h ** 0.5).to(device)
    b1 = torch.zeros(h, device=device)
    if zero_from is not None:                      # reserved block contributes nothing initially
        W2[zero_from:] = 0.0
    return {"W1": W1.requires_grad_(), "b1": b1.requires_grad_(),
            "W2": W2.requires_grad_(), "b2": torch.zeros(C, device=device).requires_grad_()}


def grow(P, add, seed, device):
    """function-preserving: append random W1 cols, ZERO W2 rows -> zero immediate loss jump."""
    g = torch.Generator(device="cpu").manual_seed(seed + 999)
    din, h = P["W1"].shape
    nW1 = (torch.randn(din, add, generator=g) / din ** 0.5).to(device)
    W1 = torch.cat([P["W1"].detach(), nW1], 1)
    b1 = torch.cat([P["b1"].detach(), torch.zeros(add, device=device)])
    W2 = torch.cat([P["W2"].detach(), torch.zeros(add, P["W2"].shape[1], device=device)], 0)
    return {"W1": W1.requires_grad_(), "b1": b1.requires_grad_(),
            "W2": W2.requires_grad_(), "b2": P["b2"].detach().requires_grad_()}


def fwd(x_e, x_r, P, pairs, unit_lo=None, unit_hi=None):
    feat = torch.cat([x_e[pairs[:, 0]], x_r[pairs[:, 1]]], 1)
    W1, b1, W2 = P["W1"], P["b1"], P["W2"]
    if unit_lo is not None or unit_hi is not None:
        lo = unit_lo or 0
        hi = unit_hi if unit_hi is not None else W1.shape[1]
        h = torch.relu(feat @ W1[:, lo:hi] + b1[lo:hi])
        return h @ W2[lo:hi, :] + P["b2"]
    return torch.relu(feat @ W1 + b1) @ W2 + P["b2"]


def fit(x_e, x_r, P, pairs, labels, epochs, lr, freeze_below=None, freeze_above=None):
    """freeze_below=h0: freeze old block units [:h0]. freeze_above=h0: freeze reserved units [h0:]."""
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(x_e, x_r, P, pairs), labels).backward()
        if freeze_below is not None:
            h0 = freeze_below
            P["W1"].grad[:, :h0] = 0
            P["b1"].grad[:h0] = 0
            P["W2"].grad[:h0, :] = 0
            P["b2"].grad.zero_()
        if freeze_above is not None:
            h0 = freeze_above
            P["W1"].grad[:, h0:] = 0
            P["b1"].grad[h0:] = 0
            P["W2"].grad[h0:, :] = 0
        opt.step()


@torch.no_grad()
def acc(x_e, x_r, P, pairs, labels, **kw):
    if len(pairs) == 0:
        return float("nan")
    return (fwd(x_e, x_r, P, pairs, **kw).argmax(1) == labels).float().mean().item()


@torch.no_grad()
def newbranch_norm(x_e, x_r, P, pairs, h0):
    if P["W1"].shape[1] <= h0 or len(pairs) == 0:
        return 0.0
    feat = torch.cat([x_e[pairs[:, 0]], x_r[pairs[:, 1]]], 1)
    hn = torch.relu(feat @ P["W1"][:, h0:] + P["b1"][h0:])
    return (hn @ P["W2"][h0:, :]).norm(dim=1).mean().item()


def run(seed, rtype, cap, args, device):
    x_e, x_r, y1, y2 = make_world(seed, args, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ee, rr = torch.meshgrid(torch.arange(args.n_ent), torch.arange(args.n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1)
    allp = allp[torch.randperm(allp.shape[0], generator=g)].to(device)
    n = allp.shape[0]
    n_res = int(args.residual_frac * n)
    res, rule1 = allp[:n_res], allp[n_res:]
    res_tr, res_un = res[:n_res // 2], res[n_res // 2:]
    r1_tr, r1_un = rule1[:rule1.shape[0] * 3 // 4], rule1[rule1.shape[0] * 3 // 4:]

    def L1(p):
        return y1[p[:, 0], p[:, 1]]

    gg = torch.Generator(device="cpu").manual_seed(seed + 55)
    offs = {(res[i, 0].item(), res[i, 1].item()): torch.randint(1, args.C, (1,), generator=gg).item()
            for i in range(n_res)}

    def resL(p):
        if rtype == "rule2":
            return y2[p[:, 0], p[:, 1]]
        base = L1(p)
        o = torch.tensor([offs[(x[0].item(), x[1].item())] for x in p], device=device)
        return (base + o) % args.C

    din, h0, H = args.de + args.dr, args.h0, args.H

    # phase 1
    if cap == "fixed":
        P = init_params(seed, din, h0, args.C, device); fb1 = None
    elif cap == "wide":
        P = init_params(seed, din, H, args.C, device); fb1 = None
    elif cap == "protect":
        P = init_params(seed, din, h0, args.C, device); fb1 = None
    else:  # partition: H wide, reserved block [h0:] zeroed & frozen in phase 1 (train old [:h0])
        P = init_params(seed, din, H, args.C, device, zero_from=h0); fb1 = h0
    # partition freezes the RESERVED block [h0:] during phase 1 (trains the old block [:h0])
    fit(x_e, x_r, P, r1_tr, L1(r1_tr), args.epochs, args.lr,
        freeze_above=fb1 if cap == "partition" else None)
    r1_phase1 = acc(x_e, x_r, P, r1_un, L1(r1_un))

    # phase boundary
    if cap == "protect":
        P = grow(P, H - h0, seed, device)

    # phase 2: residual + matched rule1 referee
    ref = r1_tr[torch.randperm(r1_tr.shape[0], device=device)[:int(args.ref_frac * r1_tr.shape[0])]]
    p2 = torch.cat([res_tr, ref]); p2l = torch.cat([resL(res_tr), L1(ref)])
    fb2 = h0 if cap in ("protect", "partition") else None
    fit(x_e, x_r, P, p2, p2l, args.fold_epochs, args.lr, freeze_below=fb2)

    split = h0 if cap in ("protect", "partition") else None
    return {
        "r1_phase1": r1_phase1,
        "r1_oldbranch": acc(x_e, x_r, P, r1_un, L1(r1_un), unit_hi=split) if split else
                        acc(x_e, x_r, P, r1_un, L1(r1_un)),
        "r1_combined": acc(x_e, x_r, P, r1_un, L1(r1_un)),
        "newbr_norm": newbranch_norm(x_e, x_r, P, r1_un, h0) if split else 0.0,
        "res_train": acc(x_e, x_r, P, res_tr, resL(res_tr)),
        "res_unseen": acc(x_e, x_r, P, res_un, resL(res_un)),
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
    print(f"device={device} GROWTH2 k={args.k} h0={args.h0} H={args.H} "
          f"residual_frac={args.residual_frac} ref_frac={args.ref_frac} seeds={args.seeds} "
          f"chance={1.0/args.C:.3f}")
    print(f"{'resid':>7} {'cap':>10} {'r1_ph1':>7} {'r1_old':>7} {'r1_comb':>8} "
          f"{'newbr':>6} {'res_tr':>7} {'res_un':>7}")
    for rtype in ["atomic", "rule2"]:
        for cap in ["fixed", "wide", "protect", "partition"]:
            agg = collections.defaultdict(list)
            for s in range(args.seeds):
                for kk, v in run(s, rtype, cap, args, device).items():
                    agg[kk].append(v)
            m = {kk: sum(vs) / len(vs) for kk, vs in agg.items()}
            print(f"{rtype:>7} {cap:>10} {m['r1_phase1']:>7.3f} {m['r1_oldbranch']:>7.3f} "
                  f"{m['r1_combined']:>8.3f} {m['newbr_norm']:>6.2f} {m['res_train']:>7.3f} "
                  f"{m['res_unseen']:>7.3f}")
    print("\nverdict: protect/partition r1_old should == r1_ph1 (freeze OK); r1_comb is the real "
          "retention; rule2 res_un>>chance & >fixed = growth absorbs compressible residual w/o "
          "overwrite; atomic res_un~chance.")


if __name__ == "__main__":
    main()
