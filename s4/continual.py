#!/usr/bin/env python3
"""Sequential continual retention of an EXECUTABLE, extrapolating algorithm (codex). Substrate = the
scratchpad model (interleaved I/O + local-window attn) that length-extrapolates csum_reset to L40.

Phase 2: acquire csum_reset (the algorithm to RETAIN).
Phase 3: learn the interference-capable rmax_reset from NEW data only (shared params), two arms:
  naive : fine-tune all params on rmax_reset          -> exposes natural interference.
  ewc   : fine-tune + diagonal-Fisher penalty to phase-2 weights (bounded param summary, no old replay).

codex's key point: extrapolation (L40) is PART OF retained knowledge and the SENSITIVE probe -- a model
that keeps L3-12 but loses L20-40 has algorithmically forgotten. Report old csum_reset retention at
L3-12 (local) AND L14-40 (algorithmic) + reset-counterfactual, and new rmax_reset acquisition, after
phase 3. No external memory / no joint full retraining.
"""
import argparse, sys, os, random, copy
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import NTOK, PAD
from scratchpad import LocalTM, make_inter, batch_inter, eval_inter


def gen_data(op, n, seed, Lrange=(3, 12)):
    rng = random.Random(seed); data = []
    for _ in range(n):
        L = rng.randint(*Lrange)
        x = [rng.randrange(0, ML.V) for _ in range(L)]
        for _r in range(rng.choice([0, 1, 2])):
            x[rng.randrange(0, L)] = ML.RESET
        data.append(make_inter(op, x))
    return data


def train(model, data, steps, bs, lr, maxlen, device, ewc=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ptr = 0
    for _ in range(steps):
        bd = data[ptr:ptr + bs]; ptr = (ptr + bs) % (len(data) - bs)
        idx, msk = batch_inter(bd, maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        if ewc is not None:
            F_, star, lam = ewc
            loss = loss + lam * sum((F_[n] * (p - star[n]) ** 2).sum()
                                    for n, p in model.named_parameters())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def fisher(model, data, bs, maxlen, device, nb=40):
    F_ = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    ptr = 0
    for _ in range(nb):
        bd = data[ptr:ptr + bs]; ptr = (ptr + bs) % (len(data) - bs)
        idx, msk = batch_inter(bd, maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        model.zero_grad(); loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                F_[n] += p.grad.detach() ** 2 / nb
    return F_


@torch.no_grad()
def counterfactual(model, op, maxlen, device, n=200, seed=3):
    rng = random.Random(seed); ok = 0
    for _ in range(n):
        L = rng.randint(8, 12)
        x = [rng.randrange(1, ML.V) for _ in range(L)]
        q = rng.randrange(L - 2, L); r = rng.randrange(1, q - 3)
        def run(xx):
            _, _, xs, y = make_inter(op, xx); seq = [ML.BOS, ML.CMD[op]]; pr = []
            for i in range(L):
                seq.append(xs[i])
                idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
                yi = int(model(idx)[0, len(seq) - 1].argmax()); seq.append(yi); pr.append(yi)
            return pr, y
        pa, ya = run(list(x)); xb = list(x); xb[r] = ML.RESET; pb, yb = run(xb)
        if pa[q] == ya[q] and pb[q] == yb[q] and ya[q] != yb[q]:
            ok += 1
    return ok / n


def report(model, tag, maxlen, device, en):
    Ls = [8, 12, 16, 20, 30, 40]
    cs = [eval_inter(model, "csum_reset", L, en, maxlen, device, 900 + L) for L in Ls]
    rm = [eval_inter(model, "rmax_reset", L, en, maxlen, device, 800 + L) for L in Ls]
    cf = counterfactual(model, "csum_reset", maxlen, device)
    print(f"[{tag}]")
    print(f"   csum_reset  " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, cs)) + f"   cf:{cf:.2f}")
    print(f"   rmax_reset  " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, rm)))
    return cs, cf, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps2", type=int, default=6000); ap.add_argument("--steps3", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--window", type=int, default=5); ap.add_argument("--lam", type=float, default=3000.0)
    ap.add_argument("--eval_n", type=int, default=150); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)

    d_csum = gen_data("csum_reset", args.n, args.seed)
    d_rmax = gen_data("rmax_reset", args.n, args.seed + 7)

    # phase 2: acquire csum_reset
    m2 = LocalTM(W=args.window).to(device)
    train(m2, d_csum, args.steps2, args.bs, args.lr, args.maxlen, device)
    print(f"device={device} CONTINUAL window={args.window} lam={args.lam}\n=== after phase 2 (csum acquired) ===")
    report(m2, "phase2", args.maxlen, device, args.eval_n)

    star = {n: p.detach().clone() for n, p in m2.named_parameters()}
    F_ = fisher(m2, d_csum, args.bs, args.maxlen, device)

    print("\n=== after phase 3 (learn interfering rmax_reset, NO csum replay) ===")
    for arm in ["naive", "ewc"]:
        m3 = LocalTM(W=args.window).to(device)
        m3.load_state_dict(m2.state_dict())
        train(m3, d_rmax, args.steps3, args.bs, args.lr, args.maxlen, device,
              ewc=(F_, star, args.lam) if arm == "ewc" else None)
        report(m3, arm, args.maxlen, device, args.eval_n)
    print("\nverdict: RETENTION of the ALGORITHM = csum_reset L20-40 + cf staying high after phase 3. "
          "naive should show algorithmic forgetting (L3-12 may stay but L40 falls); ewc should protect "
          "the extrapolating operator. rmax acquisition should be high in both.")


if __name__ == "__main__":
    main()
