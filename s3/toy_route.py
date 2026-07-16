#!/usr/bin/env python3
"""Routability: is the residual addressable at inference, and does a localized correction branch
+ gate recover the old rule while keeping the residual? (torch/GPU)

toy_growth2.py localized the barrier to BRANCH INTERFERENCE: a frozen old block is preserved, but the
new branch's large logits override it on old inputs (combined rule-1 collapses). codex: separate
COMPRESSIBILITY (can the branch learn a generalizing rule) from ROUTABILITY (can a label-free signal
say when to use it). If old-rule and residual share the same x-support and differ only by label, no
g(x) can route -> a label-free router does not exist.

Two worlds:
  scatter : residual = random subset of pairs (identity NOT encoded in x) -> impossibility control.
  region  : residual = all pairs whose entity is in a fixed subset (identity IS in x) -> routable.
Residual type: rule2 (compressible) | atomic (incompressible).

Fix for interference: train the new units as a CORRECTION delta with a SILENCE penalty on referee
rule inputs (learn ~0 on rule inputs). Compose z = z_old + g * delta_new.
Gates: none (g=1), oracle (true identity; train-part is deployable via write-time surprise, unseen-part
is a true oracle), margin (g=1 if old-branch top1-top2 margin < tau).

Reports per (world,residual,gate): combined rule1 (vs phase-1 target), res_train, res_unseen,
||g*delta||/||z_old||; plus AUROC of the old-branch margin for residual detection (routability).
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


def init_params(seed, din, h, C, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 555)
    return {"W1": (torch.randn(din, h, generator=g) / din ** 0.5).to(device).requires_grad_(),
            "b1": torch.zeros(h, device=device, requires_grad=True),
            "W2": (torch.randn(h, C, generator=g) / h ** 0.5).to(device).requires_grad_(),
            "b2": torch.zeros(C, device=device, requires_grad=True)}


def grow(P, add, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed + 999)
    din, h = P["W1"].shape
    nW1 = (torch.randn(din, add, generator=g) / din ** 0.5).to(device)
    return {"W1": torch.cat([P["W1"].detach(), nW1], 1).requires_grad_(),
            "b1": torch.cat([P["b1"].detach(), torch.zeros(add, device=device)]).requires_grad_(),
            "W2": torch.cat([P["W2"].detach(), torch.zeros(add, P["W2"].shape[1], device=device)],
                            0).requires_grad_(),
            "b2": P["b2"].detach().requires_grad_()}


def feats(x_e, x_r, pairs):
    return torch.cat([x_e[pairs[:, 0]], x_r[pairs[:, 1]]], 1)


def old_logits(x_e, x_r, P, pairs, h0):
    f = feats(x_e, x_r, pairs)
    return torch.relu(f @ P["W1"][:, :h0] + P["b1"][:h0]) @ P["W2"][:h0, :] + P["b2"]


def new_logits(x_e, x_r, P, pairs, h0):
    f = feats(x_e, x_r, pairs)
    return torch.relu(f @ P["W1"][:, h0:] + P["b1"][h0:]) @ P["W2"][h0:, :]


def fit_old(x_e, x_r, P, pairs, labels, epochs, lr):
    opt = torch.optim.Adam([P[n] for n in P], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        f = feats(x_e, x_r, pairs)
        lg = torch.relu(f @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
        F.cross_entropy(lg, labels).backward()
        opt.step()


def fit_correction(x_e, x_r, P, h0, res_tr, res_lab, ref, ref_lab, epochs, lr, lam):
    """train ONLY new units [h0:]; combined fits residual+referee; silence penalty on referee delta."""
    opt = torch.optim.Adam([P["W1"], P["b1"], P["W2"], P["b2"]], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_res = old_logits(x_e, x_r, P, res_tr, h0) + new_logits(x_e, x_r, P, res_tr, h0)
        z_ref = old_logits(x_e, x_r, P, ref, h0) + new_logits(x_e, x_r, P, ref, h0)
        loss = F.cross_entropy(z_res, res_lab) + F.cross_entropy(z_ref, ref_lab)
        if lam > 0:
            loss = loss + lam * new_logits(x_e, x_r, P, ref, h0).pow(2).sum(1).mean()
        loss.backward()
        P["W1"].grad[:, :h0] = 0; P["b1"].grad[:h0] = 0
        P["W2"].grad[:h0, :] = 0; P["b2"].grad.zero_()      # freeze old block + shared bias
        opt.step()


@torch.no_grad()
def margin_of(logits):
    t2 = logits.topk(2, 1).values
    return t2[:, 0] - t2[:, 1]


@torch.no_grad()
def auroc(pos, neg):
    # P(score_pos > score_neg); pos=residual margins (want LOW), so score = -margin
    s = torch.cat([-pos, -neg]); y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, device=s.device).float()
    n1 = pos.numel(); n0 = neg.numel()
    return ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


@torch.no_grad()
def gated_acc(x_e, x_r, P, h0, pairs, labels, gate, tau=None, is_res=None):
    if len(pairs) == 0:
        return float("nan")
    zo = old_logits(x_e, x_r, P, pairs, h0)
    zn = new_logits(x_e, x_r, P, pairs, h0)
    if gate == "none":
        g = torch.ones(len(pairs), 1, device=zo.device)
    elif gate == "oracle":
        g = is_res.float().view(-1, 1)
    else:  # margin
        g = (margin_of(zo) < tau).float().view(-1, 1)
    pred = (zo + g * zn).argmax(1)
    return (pred == labels).float().mean().item()


def run(seed, world, rtype, args, device):
    x_e, x_r, y1, y2 = make_world(seed, args, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ee, rr = torch.meshgrid(torch.arange(args.n_ent), torch.arange(args.n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1)
    # residual mask
    if world == "scatter":
        perm = torch.randperm(allp.shape[0], generator=g)
        res_idx = perm[:int(args.residual_frac * allp.shape[0])]
        is_res_full = torch.zeros(allp.shape[0], dtype=torch.bool)
        is_res_full[res_idx] = True
    else:  # region: entities in a fixed subset -> residual
        n_res_ent = int(args.residual_frac * args.n_ent)
        res_ents = torch.randperm(args.n_ent, generator=g)[:n_res_ent]
        is_res_full = torch.isin(allp[:, 0], res_ents)
    allp = allp.to(device); is_res_full = is_res_full.to(device)
    res = allp[is_res_full]; rule1 = allp[~is_res_full]
    res = res[torch.randperm(res.shape[0], device=device)]
    rule1 = rule1[torch.randperm(rule1.shape[0], device=device)]
    res_tr, res_un = res[:res.shape[0] // 2], res[res.shape[0] // 2:]
    r1_tr, r1_un = rule1[:rule1.shape[0] * 3 // 4], rule1[rule1.shape[0] * 3 // 4:]

    def L1(p): return y1[p[:, 0], p[:, 1]]
    gg = torch.Generator(device="cpu").manual_seed(seed + 55)
    off = {(res[i, 0].item(), res[i, 1].item()): torch.randint(1, args.C, (1,), generator=gg).item()
           for i in range(res.shape[0])}

    def resL(p):
        if rtype == "rule2": return y2[p[:, 0], p[:, 1]]
        o = torch.tensor([off[(x[0].item(), x[1].item())] for x in p], device=device)
        return (L1(p) + o) % args.C

    din, h0, H = args.de + args.dr, args.h0, args.H
    P = init_params(seed, din, h0, args.C, device)
    fit_old(x_e, x_r, P, r1_tr, L1(r1_tr), args.epochs, args.lr)
    r1_phase1 = (old_logits(x_e, x_r, P, r1_un, h0).argmax(1) == L1(r1_un)).float().mean().item()

    P = grow(P, H - h0, seed, device)
    ref = r1_tr[torch.randperm(r1_tr.shape[0], device=device)[:int(args.ref_frac * r1_tr.shape[0])]]
    fit_correction(x_e, x_r, P, h0, res_tr, resL(res_tr), ref, L1(ref),
                   args.fold_epochs, args.lr, args.lam)

    # routability: AUROC of old-branch margin for residual detection (unseen residual vs unseen rule)
    with torch.no_grad():
        m_res = margin_of(old_logits(x_e, x_r, P, res_un, h0))
        m_rule = margin_of(old_logits(x_e, x_r, P, r1_un, h0))
    au = auroc(m_res, m_rule)

    # choose tau on referee+res_tr (deployable selection): median margin of res_tr
    with torch.no_grad():
        tau = margin_of(old_logits(x_e, x_r, P, res_tr, h0)).median().item()
        nr = new_logits(x_e, x_r, P, r1_un, h0).norm(dim=1).mean().item()
        zr = old_logits(x_e, x_r, P, r1_un, h0).norm(dim=1).mean().item()

    out = {"r1_phase1": r1_phase1, "auroc": au, "newnorm_on_rule": nr / (zr + 1e-9)}
    for gate in ["none", "oracle", "margin"]:
        out[f"r1_{gate}"] = gated_acc(x_e, x_r, P, h0, r1_un, L1(r1_un), gate, tau,
                                      torch.zeros(len(r1_un), dtype=torch.bool, device=device))
        out[f"resU_{gate}"] = gated_acc(x_e, x_r, P, h0, res_un, resL(res_un), gate, tau,
                                        torch.ones(len(res_un), dtype=torch.bool, device=device))
    return out


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
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--fold_epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ROUTE k={args.k} h0={args.h0} H={args.H} lam={args.lam} "
          f"seeds={args.seeds} chance={1.0/args.C:.3f}")
    print(f"{'world':>8} {'resid':>7} {'r1_ph1':>7} {'auroc':>6} {'nrm/rule':>8} | "
          f"{'r1_none':>7} {'r1_orac':>7} {'r1_marg':>7} | "
          f"{'rU_none':>7} {'rU_orac':>7} {'rU_marg':>7}")
    for world in ["scatter", "region"]:
        for rtype in ["rule2", "atomic"]:
            agg = collections.defaultdict(list)
            for s in range(args.seeds):
                for kk, v in run(s, world, rtype, args, device).items():
                    agg[kk].append(v)
            m = {kk: sum(vs) / len(vs) for kk, vs in agg.items()}
            print(f"{world:>8} {rtype:>7} {m['r1_phase1']:>7.3f} {m['auroc']:>6.2f} "
                  f"{m['newnorm_on_rule']:>8.2f} | "
                  f"{m['r1_none']:>7.3f} {m['r1_oracle']:>7.3f} {m['r1_margin']:>7.3f} | "
                  f"{m['resU_none']:>7.3f} {m['resU_oracle']:>7.3f} {m['resU_margin']:>7.3f}")
    print("\nauroc~0.5 => residual NOT addressable from old-branch margin (scatter expected). "
          "r1_orac recovering + rU_orac high (rule2) = routing is the only barrier. "
          "region rule2: r1_marg~=r1_ph1 & rU_marg high = autonomous protected growth works.")


if __name__ == "__main__":
    main()
