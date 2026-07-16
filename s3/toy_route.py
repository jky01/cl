#!/usr/bin/env python3
"""Routability (codex-reviewed): is a residual addressable at inference by a WEIGHT-RESIDENT gate,
and does a scale-controlled correction branch recover the old rule while keeping the residual? (GPU)

Separates COMPRESSIBILITY (branch learns a generalizing rule) from ROUTABILITY (a label-free signal
says when to use it). Three gates, honestly named (codex):
  seen-key      : exact write-time identity for residual-TRAIN only -> memory-backed baseline.
  unseen-oracle : true identity on residual-UNSEEN -> privileged upper bound.
  model-learned : a small x->g router trained on referee/res-train identities, no key lookup ->
                  the only arm that satisfies no-memory inference; tested on DISJOINT held-out pairs.
Also report the unsupervised old-branch margin AUROC.

Worlds:
  scatter : residual = random pair subset (identity has NO learnable transferable structure).
  region  : residual = pairs whose entity descriptor bit x_e[:,0]>0 (STRUCTURAL predicate visible in
            x, generalizes to unseen entities/pairs).
Residual type: rule2 (compressible) | atomic (incompressible).

Correction branch: frozen old block; new units trained as a CENTERED min-norm correction
  z = z_old + g * Pdelta,  Pz = z - mean_class(z),
  loss = CE(z_old+Pdelta, y)_res + CE(...)_ref + lam_s||Pdelta||^2_ref + lam_c||Pdelta||^2_res.
Reports r1 (vs phase-1) and residual-unseen under each gate, and correction norm ratio per population.
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


def center(z):
    return z - z.mean(1, keepdim=True)


def z_old(x_e, x_r, P, pairs, h0):
    f = feats(x_e, x_r, pairs)
    return torch.relu(f @ P["W1"][:, :h0] + P["b1"][:h0]) @ P["W2"][:h0, :] + P["b2"]


def delta(x_e, x_r, P, pairs, h0):
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


def fit_correction(x_e, x_r, P, h0, res_tr, res_lab, ref, ref_lab, epochs, lr, lam_s, lam_c):
    opt = torch.optim.Adam([P["W1"], P["b1"], P["W2"], P["b2"]], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        Pd_res = center(delta(x_e, x_r, P, res_tr, h0))
        Pd_ref = center(delta(x_e, x_r, P, ref, h0))
        loss = (F.cross_entropy(center(z_old(x_e, x_r, P, res_tr, h0)) + Pd_res, res_lab)
                + F.cross_entropy(center(z_old(x_e, x_r, P, ref, h0)) + Pd_ref, ref_lab)
                + lam_s * Pd_ref.pow(2).sum(1).mean()
                + lam_c * Pd_res.pow(2).sum(1).mean())
        loss.backward()
        P["W1"].grad[:, :h0] = 0; P["b1"].grad[:h0] = 0
        P["W2"].grad[:h0, :] = 0; P["b2"].grad.zero_()
        opt.step()


def train_router(x_e, x_r, res_tr, ref, din, epochs, lr, device, seed):
    """small supervised x->g router: residual(1) vs rule(0), BCE. returns fn pairs->prob."""
    g = torch.Generator(device="cpu").manual_seed(seed + 222)
    W1 = (torch.randn(din, 64, generator=g) / din ** 0.5).to(device).requires_grad_()
    b1 = torch.zeros(64, device=device, requires_grad=True)
    w2 = torch.zeros(64, 1, device=device, requires_grad=True)
    b2 = torch.zeros(1, device=device, requires_grad=True)
    Xp = feats(x_e, x_r, torch.cat([res_tr, ref]))
    yb = torch.cat([torch.ones(len(res_tr)), torch.zeros(len(ref))]).to(device)
    opt = torch.optim.Adam([W1, b1, w2, b2], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logit = (torch.relu(Xp @ W1 + b1) @ w2 + b2).squeeze(1)
        F.binary_cross_entropy_with_logits(logit, yb).backward()
        opt.step()

    @torch.no_grad()
    def score(pairs):
        X = feats(x_e, x_r, pairs)
        return torch.sigmoid((torch.relu(X @ W1 + b1) @ w2 + b2).squeeze(1))
    return score


@torch.no_grad()
def margin_of(logits):
    t2 = logits.topk(2, 1).values
    return t2[:, 0] - t2[:, 1]


@torch.no_grad()
def auroc(pos_score, neg_score):
    s = torch.cat([pos_score, neg_score])
    y = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    order = torch.argsort(s); ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, device=s.device).float()
    n1 = pos_score.numel(); n0 = neg_score.numel()
    return ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


@torch.no_grad()
def comb_acc(x_e, x_r, P, h0, pairs, labels, g):
    zo = center(z_old(x_e, x_r, P, pairs, h0))
    pd = center(delta(x_e, x_r, P, pairs, h0))
    return ((zo + g.view(-1, 1) * pd).argmax(1) == labels).float().mean().item()


def run(seed, world, rtype, args, device):
    x_e, x_r, y1, y2 = make_world(seed, args, device)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    ee, rr = torch.meshgrid(torch.arange(args.n_ent), torch.arange(args.n_rel), indexing="ij")
    allp = torch.stack([ee.flatten(), rr.flatten()], 1)
    if world == "scatter":
        perm = torch.randperm(allp.shape[0], generator=g)
        is_res = torch.zeros(allp.shape[0], dtype=torch.bool)
        is_res[perm[:int(args.residual_frac * allp.shape[0])]] = True
    else:  # region: structural predicate on entity descriptor bit (visible in x, generalizes)
        thr = x_e[:, 0].quantile(1 - args.residual_frac)
        res_ent = (x_e[:, 0] > thr)
        is_res = res_ent[allp[:, 0]]
    allp = allp.to(device); is_res = is_res.to(device)
    res = allp[is_res][torch.randperm(int(is_res.sum()), device=device)]
    rule1 = allp[~is_res][torch.randperm(int((~is_res).sum()), device=device)]
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
    r1_ph1 = (z_old(x_e, x_r, P, r1_un, h0).argmax(1) == L1(r1_un)).float().mean().item()

    P = grow(P, H - h0, seed, device)
    ref = r1_tr[torch.randperm(r1_tr.shape[0], device=device)[:int(args.ref_frac * r1_tr.shape[0])]]
    fit_correction(x_e, x_r, P, h0, res_tr, resL(res_tr), ref, L1(ref),
                   args.fold_epochs, args.lr, args.lam_s, args.lam_c)

    # routability signals
    m_resU = margin_of(z_old(x_e, x_r, P, res_un, h0))
    m_r1U = margin_of(z_old(x_e, x_r, P, r1_un, h0))
    au_margin = auroc(-m_resU, -m_r1U)                 # low old-margin => residual
    router = train_router(x_e, x_r, res_tr, ref, din, args.router_epochs, args.lr, device, seed)
    au_router_tr = auroc(router(res_tr), router(ref))
    au_router_te = auroc(router(res_un), router(r1_un))

    # correction norm ratio (median) per population
    def nrm(pairs):
        pd = center(delta(x_e, x_r, P, pairs, h0)).norm(dim=1)
        zo = center(z_old(x_e, x_r, P, pairs, h0)).norm(dim=1)
        return (pd / (zo + 1e-9)).median().item()

    o = {"r1_ph1": r1_ph1, "au_margin": au_margin, "au_router_tr": au_router_tr,
         "au_router_te": au_router_te, "nrm_rule": nrm(r1_un), "nrm_resU": nrm(res_un)}
    ones = lambda p: torch.ones(len(p), device=device)
    zeros = lambda p: torch.zeros(len(p), device=device)
    # gates on r1_un (want g=0) and res_un (want g=1)
    o["r1_none"] = comb_acc(x_e, x_r, P, h0, r1_un, L1(r1_un), ones(r1_un))
    o["r1_oracle"] = comb_acc(x_e, x_r, P, h0, r1_un, L1(r1_un), zeros(r1_un))
    o["r1_learned"] = comb_acc(x_e, x_r, P, h0, r1_un, L1(r1_un), router(r1_un))
    o["rU_none"] = comb_acc(x_e, x_r, P, h0, res_un, resL(res_un), ones(res_un))
    o["rU_oracle"] = comb_acc(x_e, x_r, P, h0, res_un, resL(res_un), ones(res_un))
    o["rU_learned"] = comb_acc(x_e, x_r, P, h0, res_un, resL(res_un), router(res_un))
    return o


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
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ROUTE2 k={args.k} h0={args.h0} H={args.H} lam_s={args.lam_s} "
          f"lam_c={args.lam_c} seeds={args.seeds} chance={1.0/args.C:.3f}")
    print(f"{'world':>8} {'resid':>6} {'r1ph1':>6} {'auMrg':>6} {'auRtr':>6} {'auRte':>6} "
          f"{'nrmRl':>6} {'nrmRs':>6} | {'r1_no':>6} {'r1_or':>6} {'r1_lr':>6} | "
          f"{'rU_no':>6} {'rU_or':>6} {'rU_lr':>6}")
    for world in ["scatter", "region"]:
        for rtype in ["rule2", "atomic"]:
            agg = collections.defaultdict(list)
            for s in range(args.seeds):
                for kk, v in run(s, world, rtype, args, device).items():
                    agg[kk].append(v)
            m = {kk: sum(vs) / len(vs) for kk, vs in agg.items()}
            print(f"{world:>8} {rtype:>6} {m['r1_ph1']:>6.3f} {m['au_margin']:>6.2f} "
                  f"{m['au_router_tr']:>6.2f} {m['au_router_te']:>6.2f} {m['nrm_rule']:>6.2f} "
                  f"{m['nrm_resU']:>6.2f} | {m['r1_none']:>6.3f} {m['r1_oracle']:>6.3f} "
                  f"{m['r1_learned']:>6.3f} | {m['rU_none']:>6.3f} {m['rU_oracle']:>6.3f} "
                  f"{m['rU_learned']:>6.3f}")
    print("\nauRte = held-out router AUROC (routability). scatter~0.5 (memorizable not generalizable), "
          "region>>0.5 (structural). SUCCESS region+rule2: r1_lr~=r1_ph1 AND rU_lr high => autonomous "
          "weight-resident routing works. atomic rU stays chance regardless (routable != compressible).")


if __name__ == "__main__":
    main()
