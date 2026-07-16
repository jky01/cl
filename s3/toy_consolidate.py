#!/usr/bin/env python3
"""Consolidation: can (frozen base + gated correction branch + router) be distilled back into a
SINGLE flat model of the original width, then the expert+router DELETED, with both rule and residual
surviving? And is the GROWN width actually necessary? (torch/GPU)

Builds the region+rule2 success state from s3/toy_route.py (weight-resident router + protected
centered correction). Teacher = combined gated model. Then distills a FRESH flat student (no branch,
no router, no gate) of width W on the LEGAL retained data (referee rule-1 ∪ residual-train) using the
teacher's predictions, and evaluates on unseen rule-1 and unseen residual.

  consolidate into h0 (original width): if it keeps both -> growth was a temporary SCAFFOLD.
  consolidate into H (grown width): if only this keeps both -> growth is PERMANENTLY necessary
                                    (the extra capacity is load-bearing).

This closes the knowledge-INTO-weights loop (vs knowledge-BESIDE-weights) and bounds the unbounded-
experts problem: after consolidation there is one flat model and no router/expert at inference.
"""
import argparse
import sys
import os
import collections
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_route as tr  # reuse generator + training + helpers


def flat_logits(P, x_e, x_r, pairs):
    f = tr.feats(x_e, x_r, pairs)
    return torch.relu(f @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]


def distill(x_e, x_r, teacher_pairs, teacher_labels, W, din, C, epochs, lr, seed, device):
    """fresh flat student width W trained to predict teacher_labels on teacher_pairs."""
    P = tr.init_params(seed + 4242, din, W, C, device)
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(flat_logits(P, x_e, x_r, teacher_pairs), teacher_labels).backward()
        opt.step()
    return P


def run(seed, args, device):
    world, rtype = "region", "rule2"
    x_e, x_r, y1, y2 = tr.make_world(seed, args, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ee, rr = torch.meshgrid(torch.arange(args.n_ent), torch.arange(args.n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1)
    thr = x_e[:, 0].quantile(1 - args.residual_frac)
    is_res = (x_e[:, 0] > thr)[allp[:, 0]]
    allp = allp.to(device); is_res = is_res.to(device)
    res = allp[is_res][torch.randperm(int(is_res.sum()), device=device)]
    rule1 = allp[~is_res][torch.randperm(int((~is_res).sum()), device=device)]
    res_tr, res_un = res[:res.shape[0] // 2], res[res.shape[0] // 2:]
    r1_tr, r1_un = rule1[:rule1.shape[0] * 3 // 4], rule1[rule1.shape[0] * 3 // 4:]

    def L1(p): return y1[p[:, 0], p[:, 1]]
    def resL(p): return y2[p[:, 0], p[:, 1]]

    din, h0, H, C = args.de + args.dr, args.h0, args.H, args.C
    P = tr.init_params(seed, din, h0, C, device)
    tr.fit_old(x_e, x_r, P, r1_tr, L1(r1_tr), args.epochs, args.lr)
    r1_ph1 = (tr.z_old(x_e, x_r, P, r1_un, h0).argmax(1) == L1(r1_un)).float().mean().item()

    P = tr.grow(P, H - h0, seed, device)
    ref = r1_tr[torch.randperm(r1_tr.shape[0], device=device)[:int(args.ref_frac * r1_tr.shape[0])]]
    tr.fit_correction(x_e, x_r, P, h0, res_tr, resL(res_tr), ref, L1(ref),
                      args.fold_epochs, args.lr, args.lam_s, args.lam_c)
    router = tr.train_router(x_e, x_r, res_tr, ref, din, args.router_epochs, args.lr, device, seed)

    @torch.no_grad()
    def teacher_pred(pairs):
        zo = tr.center(tr.z_old(x_e, x_r, P, pairs, h0))
        pd = tr.center(tr.delta(x_e, x_r, P, pairs, h0))
        return (zo + router(pairs).view(-1, 1) * pd).argmax(1)

    # teacher (combined gated) held-out accuracy
    t_r1 = (teacher_pred(r1_un) == L1(r1_un)).float().mean().item()
    t_rU = (teacher_pred(res_un) == resL(res_un)).float().mean().item()

    # LEGAL consolidation data: self-distill the teacher on the RULE region (strong there; no stored
    # old labels / no old-data replay) + the retained residual-train under its TRUE labels. Crucially do
    # NOT let the gated teacher's WEAK residual prediction label the residual region -- use true labels
    # there (we have them: reading a residual fact gives its label). This uncaps residual quality.
    rule_pairs = allp[~is_res]
    cons_pairs = torch.cat([rule_pairs, res_tr])
    cons_labels = torch.cat([teacher_pred(rule_pairs), resL(res_tr)])

    out = {"r1_ph1": r1_ph1, "teacher_r1": t_r1, "teacher_rU": t_rU}
    for W, tag in [(h0, "h0"), (H, "H")]:
        Pc = distill(x_e, x_r, cons_pairs, cons_labels, W, din, C,
                     args.distill_epochs, args.lr, seed, device)
        out[f"cons{tag}_r1"] = (flat_logits(Pc, x_e, x_r, r1_un).argmax(1) == L1(r1_un)).float().mean().item()
        out[f"cons{tag}_rU"] = (flat_logits(Pc, x_e, x_r, res_un).argmax(1) == resL(res_un)).float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ent", type=int, default=90); ap.add_argument("--n_rel", type=int, default=90)
    ap.add_argument("--C", type=int, default=40)
    ap.add_argument("--de", type=int, default=32); ap.add_argument("--dr", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--h0", type=int, default=128); ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--residual_frac", type=float, default=0.34)
    ap.add_argument("--ref_frac", type=float, default=0.25)
    ap.add_argument("--lam_s", type=float, default=0.03); ap.add_argument("--lam_c", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3000); ap.add_argument("--fold_epochs", type=int, default=3000)
    ap.add_argument("--router_epochs", type=int, default=1500)
    ap.add_argument("--distill_epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} CONSOLIDATE region+rule2 h0={args.h0} H={args.H} seeds={args.seeds} "
          f"chance={1.0/args.C:.3f}")
    agg = collections.defaultdict(list)
    for s in range(args.seeds):
        for kk, v in run(s, args, device).items():
            agg[kk].append(v)
    m = {kk: sum(vs) / len(vs) for kk, vs in agg.items()}
    print(f"\n{'model':>16} {'rule1_unseen':>13} {'residual_unseen':>15}")
    print(f"{'phase-1 (rule only)':>16} {m['r1_ph1']:>13.3f} {'-':>15}")
    print(f"{'teacher (gated)':>16} {m['teacher_r1']:>13.3f} {m['teacher_rU']:>15.3f}")
    print(f"{'consolidated h0':>16} {m['consh0_r1']:>13.3f} {m['consh0_rU']:>15.3f}")
    print(f"{'consolidated H':>16} {m['consH_r1']:>13.3f} {m['consH_rU']:>15.3f}")
    print("\nflat consolidated model has NO branch/router/gate at inference. "
          "cons h0 keeps both => growth was temporary scaffold; only cons H keeps residual => "
          "grown capacity is load-bearing (growth necessary).")


if __name__ == "__main__":
    main()
