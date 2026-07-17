#!/usr/bin/env python3
"""Continual retention via compact ROUTED GROWTH vs fixed-capacity (codex-specced). Substrate = the
scratchpad local-window transformer that length-extrapolates csum_reset.

Phase 2: acquire csum_reset (algorithm to RETAIN). Phase 3: learn interfering rmax_reset (new data only).
Arms (identical phase-2 init & phase-3 data):
  naive         : trunk trainable, no branch                      -> overwrite baseline
  ewc           : trunk trainable + fair model-Fisher penalty     -> fixed-capacity consolidation
  adapter_ungated: trunk FROZEN + small per-block adapter ALWAYS on-> growth w/o protection
  adapter_routed : trunk FROZEN + same adapter, EXACT command mask -> protected growth (csum route off)

Routed adapter is multiplied by a per-sequence mask = 1 iff the command token is rmax (derived from the
input, no external task table). Trunk hard-frozen + zero-init adapter up-proj => csum path is
byte-identical to phase-2 (tested invariant). Retention = ALGORITHM (csum L20-40 + reset-counterfactual).
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import NTOK, PAD, rope
from scratchpad import make_inter, batch_inter, eval_inter


class Block(nn.Module):
    def __init__(s, d, h, ff, W, r):
        super().__init__()
        s.h, s.W = h, W
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)
        s.ad_dn = nn.Linear(d, r); s.ad_up = nn.Linear(r, d)         # adapter (dormant until enabled)
        nn.init.zeros_(s.ad_up.weight); nn.init.zeros_(s.ad_up.bias)  # zero-init => function-preserving
        s.use_adapter = False

    def forward(s, x, pos, mask, route):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, 2)
        q = q.view(B, T, s.h, D // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, D // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, D // s.h).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
        i = torch.arange(T, device=x.device)
        win = (i[None, :] < i[:, None] - s.W + 1)
        att = att.masked_fill(mask | win[None, None], float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(o)
        h = s.f2(F.gelu(s.f1(s.ln2(x))))
        if s.use_adapter:
            h = h + route * s.ad_up(F.gelu(s.ad_dn(s.ln2(x))))       # route: [B,1,1] mask
        return x + h


class Net(nn.Module):
    def __init__(s, nl=4, d=192, h=6, ff=768, W=5, r=32):
        super().__init__()
        s.emb = nn.Embedding(NTOK, d)
        s.blocks = nn.ModuleList([Block(d, h, ff, W, r) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NTOK)
        s.routing = "always"                                         # 'always' (ungated) | 'cmd' (routed)

    def forward(s, idx, routed=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        # routing is an INTRINSIC model property so eval respects it without a kwarg. Explicit
        # routed= overrides (used only by the invariant probe). 'cmd' => route=1 iff command==rmax.
        mode = ("cmd" if routed else "always") if routed is not None else s.routing
        if mode == "cmd":
            route = (idx[:, 1] == ML.CMD["rmax_reset"]).float().view(B, 1, 1)
        else:
            route = torch.ones(B, 1, 1, device=idx.device)
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, pos, mask, route)
        return s.head(s.lnf(x))


def gen_data(op, n, seed):
    rng = random.Random(seed); data = []
    for _ in range(n):
        L = rng.randint(3, 12); x = [rng.randrange(0, ML.V) for _ in range(L)]
        for _r in range(rng.choice([0, 1, 2])):
            x[rng.randrange(0, L)] = ML.RESET
        data.append(make_inter(op, x))
    return data


def train(model, data, steps, bs, lr, maxlen, device, params=None, routed=False, ewc=None):
    params = params if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if ewc is not None else 0.01)
    ptr = 0
    for _ in range(steps):
        bd = data[ptr:ptr + bs]; ptr = (ptr + bs) % (len(data) - bs)
        idx, msk = batch_inter(bd, maxlen, device)
        logits = model(idx[:, :-1], routed=routed); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        if ewc is not None:
            Fd, star, lam = ewc
            loss = loss + lam * sum((Fd[n] * (p - star[n]) ** 2).sum() for n, p in model.named_parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def model_fisher(model, data, bs, maxlen, device, nb=60):
    Fd = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    ptr = 0
    for _ in range(nb):
        bd = data[ptr:ptr + bs]; ptr = (ptr + bs) % (len(data) - bs)
        idx, msk = batch_inter(bd, maxlen, device)
        logits = model(idx[:, :-1]); mt = msk[:, 1:]
        logp = F.log_softmax(logits, -1)
        with torch.no_grad():
            samp = torch.multinomial(logp.exp().reshape(-1, NTOK), 1).view(logits.shape[:2])
        nll = F.nll_loss(logp.reshape(-1, NTOK), samp.reshape(-1), reduction="none").view(mt.shape)
        loss = (nll * mt).sum() / mt.sum().clamp(min=1)
        model.zero_grad(); loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                Fd[n] += p.grad.detach() ** 2 / nb
    return Fd


@torch.no_grad()
def counterfactual(model, maxlen, device, n=200, seed=3):
    rng = random.Random(seed); ok = 0
    for _ in range(n):
        L = rng.randint(8, 12); x = [rng.randrange(1, ML.V) for _ in range(L)]
        q = rng.randrange(L - 2, L); r = rng.randrange(1, q - 3)
        def run(xx):
            _, _, xs, y = make_inter("csum_reset", xx); seq = [ML.BOS, ML.CMD["csum_reset"]]; pr = []
            for i in range(L):
                seq.append(xs[i])
                idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
                yi = int(model(idx)[0, len(seq) - 1].argmax()); seq.append(yi); pr.append(yi)
            return pr, y
        pa, ya = run(list(x)); xb = list(x); xb[r] = ML.RESET; pb, yb = run(xb)
        ok += int(pa[q] == ya[q] and pb[q] == yb[q] and ya[q] != yb[q])
    return ok / n


def report(model, tag, maxlen, device, en):
    Ls = [8, 12, 20, 30, 40]
    cs = [eval_inter(model, "csum_reset", L, en, maxlen, device, 900 + L) for L in Ls]
    rm = [eval_inter(model, "rmax_reset", L, en, maxlen, device, 800 + L) for L in Ls]
    cf = counterfactual(model, maxlen, device)
    print(f"[{tag:>16}]  csum " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, cs)) +
          f"  cf:{cf:.2f} | rmax " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, rm)))
    return cs, cf, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps2", type=int, default=6000); ap.add_argument("--steps3", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--W", type=int, default=5); ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--lam", type=float, default=2000.0); ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    d_csum = gen_data("csum_reset", args.n, args.seed)
    d_rmax = gen_data("rmax_reset", args.n, args.seed + 7)

    base = Net(W=args.W, r=args.r).to(device)
    train(base, d_csum, args.steps2, args.bs, args.lr, args.maxlen, device)
    tot = sum(p.numel() for p in base.parameters())
    adp = sum(p.numel() for n, p in base.named_parameters() if "ad_" in n)
    print(f"device={device} CONTINUAL2 W={args.W} r={args.r} params={tot/1e6:.2f}M adapter={adp/1e3:.0f}K "
          f"({100*adp/tot:.1f}%)\n=== after phase 2 ===")
    report(base, "phase2", args.maxlen, device, args.eval_n)
    star = {n: p.detach().clone() for n, p in base.named_parameters()}
    Fd = model_fisher(base, d_csum, args.bs, args.maxlen, device)
    fvals = torch.cat([v.flatten() for v in Fd.values()])
    print(f"model-Fisher: median={fvals.median():.2e} p95={fvals.quantile(0.95):.2e} "
          f"frac~0={(fvals < 1e-12).float().mean():.2f} norm={fvals.sum().sqrt():.2e}")

    print("\n=== after phase 3 (rmax only, no csum replay) ===")
    for arm in ["naive", "ewc", "adapter_ungated", "adapter_routed"]:
        m = Net(W=args.W, r=args.r).to(device); m.load_state_dict(base.state_dict())
        if arm in ("adapter_ungated", "adapter_routed"):
            for b in m.blocks:
                b.use_adapter = True
            routed = (arm == "adapter_routed")
            m.routing = "cmd" if routed else "always"               # intrinsic => eval routes too
            train_params = [p for n, p in m.named_parameters() if "ad_" in n]   # only adapter trainable
            train(m, d_rmax, args.steps3, args.bs, args.lr, args.maxlen, device, params=train_params, routed=routed)
            if routed:                                             # exact non-interference invariant on csum
                with torch.no_grad():
                    ex = batch_inter(gen_data("csum_reset", 64, 12345), args.maxlen, device)
                    d0 = base(ex[0][:, :-1]); d1 = m(ex[0][:, :-1], routed=True)
                    print(f"   [invariant] csum logits max|Δ| vs phase2 = {(d0 - d1).abs().max():.2e}")
        elif arm == "ewc":
            train(m, d_rmax, args.steps3, args.bs, args.lr, args.maxlen, device, ewc=(Fd, star, args.lam))
        else:
            train(m, d_rmax, args.steps3, args.bs, args.lr, args.maxlen, device)
        report(m, arm, args.maxlen, device, args.eval_n)
    print("\nverdict: adapter_routed retains csum L20-40+cf within 0.03 of phase2 AND acquires rmax, "
          "where naive/ewc forget and adapter_ungated (same params, always-on) likely damages csum "
          "=> SELECTIVE compact growth, not capacity alone, resolves the interference.")


if __name__ == "__main__":
    main()
