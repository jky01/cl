#!/usr/bin/env python3
"""Reset-stratified WITHIN-LENGTH acquisition gate (codex): isolates recurrence acquisition from the
length-generalization wall. All sequences stay in [3,12]; we make the RECURRENCE HORIZON OOD instead.

Sub-gate 2 (distance-since-reset extrapolation): TRAIN caps the maximum reset-free run at D_train
(resets injected so no gap > D_train); TEST uses longer reset-free runs (distance > D_train) that still
fit inside length<=12. csum_reset accuracy binned by distance-since-reset: a true recurrence stays flat
past D_train; a bounded-horizon/template shortcut drops.

Paired counterfactual: same values/length, one reset moved far BEFORE a queried position so the correct
suffix changes while the local window at that position is identical -> positional/finite-window rule
fails the pair; the recurrence responds exactly.

Conditions insuf_erm / suf_erm / suf_dro evaluated on the SAME split. copy is the distance-independent
control.
"""
import argparse
import sys
import os
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import TM, batch, gen_out, NTOK, PAD


def capped_input(rng, L, Dtrain, alpha_env):
    """values in [0,V) with a reset every <=Dtrain positions (caps reset-free run)."""
    if alpha_env:
        lo, hi = (0, ML.V // 2) if rng.random() < 0.5 else (ML.V // 2, ML.V)
    else:
        lo, hi = 0, ML.V
    x = [rng.randrange(lo, hi) for _ in range(L)]
    i = rng.randrange(0, min(Dtrain, L))          # first reset within Dtrain of start
    while i < L:
        x[i] = ML.RESET
        i += 1 + rng.randrange(1, Dtrain + 1)
    return x


def long_run_input(rng, L):
    """single reset near start (or none) -> long reset-free run up to L-1 (the OOD horizon)."""
    x = [rng.randrange(0, ML.V) for _ in range(L)]
    if rng.random() < 0.7:
        x[rng.randrange(0, min(2, L))] = ML.RESET
    return x


def dist_since_reset(x):
    d, out = 0, []
    for t in x:
        if t == ML.RESET:
            d = 0
        else:
            d += 1
        out.append(d)
    return out                                    # 0 at reset, else run length


def make(op, x, env="mix"):
    y = ML.apply_op(op, x)
    return {"tokens": ML.render(op, x, y, env), "x": x, "y": y, "op": op}


def gen_train(ops, cond, n, seed, Dtrain, Lrange=(3, 12)):
    rng = random.Random(seed)
    data = []
    envs = ["E0"] if cond == "insuf_erm" else ["mix", "mix", "mix", "alpha"]
    for gi in range(len(envs)):
        for _ in range(n):
            L = 8 if cond == "insuf_erm" else rng.randint(*Lrange)
            x = capped_input(rng, L, (3 if cond == "insuf_erm" else Dtrain), envs[gi] == "alpha")
            op = rng.choice(ops)
            e = make(op, x); e["group"] = gi
            data.append(e)
    rng.shuffle(data)
    return data, len(envs)


@torch.no_grad()
def eval_distance(model, maxlen, device, n=400, seed=7):
    """csum_reset token accuracy binned by distance-since-reset (test uses long reset-free runs)."""
    rng = random.Random(seed)
    bins = {}
    for _ in range(n):
        L = rng.randint(3, 12)
        x = long_run_input(rng, L)
        e = make("csum_reset", x)
        pred = gen_out(model, e, maxlen, device)
        ds = dist_since_reset(x)
        for i, (p, g, d) in enumerate(zip(pred, e["y"], ds)):
            bins.setdefault(d, [0, 0])
            bins[d][0] += int(p == g); bins[d][1] += 1
    return {d: (c / n_, n_) for d, (c, n_) in sorted(bins.items())}


@torch.no_grad()
def eval_counterfactual(model, maxlen, device, n=300, seed=11):
    """pair: same input; second copy adds a reset early so suffix changes; local window at query same.
    recurrence -> both correct & differ; finite-window -> identical local => same (wrong) answer."""
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        L = rng.randint(8, 12)
        x = [rng.randrange(1, ML.V) for _ in range(L)]      # avoid 0 so states differ
        qpos = rng.randrange(L - 2, L)                      # query near end
        rpos = rng.randrange(1, qpos - 3)                   # reset far before query
        xa = list(x)
        xb = list(x); xb[rpos] = ML.RESET
        ea, eb = make("csum_reset", xa), make("csum_reset", xb)
        pa = gen_out(model, ea, maxlen, device)
        pb = gen_out(model, eb, maxlen, device)
        # correct iff both match gold at the query position AND they differ there (suffix changed)
        if pa[qpos] == ea["y"][qpos] and pb[qpos] == eb["y"][qpos] and ea["y"][qpos] != eb["y"][qpos]:
            ok += 1
    return ok / n


def run(cond, args, device):
    torch.manual_seed(args.seed); random.seed(args.seed)
    ops = ["copy", "csum_reset"]
    data, ng = gen_train(ops, cond, args.n_per, args.seed, args.Dtrain)
    model = TM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gw = torch.ones(ng, device=device) / ng
    ptr = 0
    for step in range(args.steps):
        bd = data[ptr:ptr + args.bs]; ptr = (ptr + args.bs) % (len(data) - args.bs)
        idx, msk = batch(bd, args.maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        if cond == "suf_dro":
            grp = torch.tensor([e["group"] for e in bd], device=device)
            gl = torch.stack([(lt[grp == g] * mt[grp == g]).sum() / mt[grp == g].sum().clamp(min=1)
                              for g in range(ng)])
            gw = (gw * torch.exp(0.01 * gl.detach())); gw = gw / gw.sum()
            loss = (gw * gl).sum()
        else:
            loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return eval_distance(model, args.maxlen, device), eval_counterfactual(model, args.maxlen, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per", type=int, default=6000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=48); ap.add_argument("--Dtrain", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} RESET-GATE Dtrain(cap reset-free run)={args.Dtrain}; test distances up to 11 "
          f"(within length<=12, NO length-OOD). steps={args.steps}")
    for cond in ["insuf_erm", "suf_erm", "suf_dro"]:
        dist, cf = run(cond, args, device)
        trained = " ".join(f"{d}:{a:.2f}" for d, (a, _) in dist.items() if d <= args.Dtrain)
        ood = " ".join(f"{d}:{a:.2f}(n{n})" for d, (a, n) in dist.items() if d > args.Dtrain)
        print(f"\n[{cond}] csum_reset token-acc by distance-since-reset:")
        print(f"   trained (d<= {args.Dtrain}): {trained}")
        print(f"   OOD-horizon (d> {args.Dtrain}): {ood}")
        print(f"   paired-counterfactual consistency: {cf:.3f}")
    print("\ngate: recurrence acquired => OOD-horizon distance bins stay HIGH (flat past Dtrain) AND "
          "counterfactual consistency high. bounded-horizon/template shortcut => OOD-horizon bins drop.")


if __name__ == "__main__":
    main()
