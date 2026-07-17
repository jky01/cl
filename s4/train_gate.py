#!/usr/bin/env python3
"""First cheap gate (LM port): does MULTI-ENVIRONMENT evidence let a shared transformer ACQUIRE the
csum_reset recurrence (length-OOD + reset-stratified), while an INSUFFICIENT single-environment
baseline shortcuts (fails OOD)? Conditions: insufficient-ERM / sufficient-ERM / sufficient-group-DRO.
(decoder transformer w/ rotary, local RTX 2070; see docs/LM_PORT_PREREG.md)

Not the full protocol -- the gate that must pass before the expensive phase-3 retention comparison.
Reserves "acquisition" for input/output supervision only (no state labels / traces). Loss on output
tokens only; eval by autoregressive greedy exact-match, plus reset-distance-stratified accuracy.
"""
import argparse
import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML

PAD = ML.VOCAB_SIZE                       # extra pad id
NTOK = ML.VOCAB_SIZE + 1


# ---------------- model: pre-LN decoder w/ rotary ----------------
def rope(x, pos):                         # x:[B,H,T,d]  pos:[T]
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos[:, None].float() * inv[None, :]           # [T, d/2]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    xr = torch.cat([-x2, x1], -1)
    return x * cos + xr * sin


class Block(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__()
        s.h = h; s.d = d
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)
        s.growth = None                    # dormant growth hook (activatable later)

    def forward(s, x, pos, mask):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, 2)
        q = q.view(B, T, s.h, D // s.h).transpose(1, 2)
        k = k.view(B, T, s.h, D // s.h).transpose(1, 2)
        v = v.view(B, T, s.h, D // s.h).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(o)
        h = s.f2(F.gelu(s.f1(s.ln2(x))))
        if s.growth is not None:
            h = h + s.growth(s.ln2(x))
        return x + h


class TM(nn.Module):
    def __init__(s, nl=4, d=192, h=6, ff=768):
        super().__init__()
        s.emb = nn.Embedding(NTOK, d)
        s.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, NTOK)

    def forward(s, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        x = s.emb(idx)
        for b in s.blocks:
            x = b(x, pos, mask)
        return s.head(s.lnf(x))


# ---------------- data ----------------
def encode(ex, maxlen):
    t = ex["tokens"]
    sep_pos = t.index(ML.SEP) if ML.SEP in t else t.index(ML.BOS, 1)   # E_tmpl uses BOS as delim
    idx = t + [PAD] * (maxlen - len(t))
    # loss mask: only output tokens (after the delimiter, up to EOS)
    m = [0] * maxlen
    for i in range(sep_pos + 1, len(t)):
        m[i] = 1
    return idx[:maxlen], m[:maxlen], sep_pos


def batch(examples, maxlen, device):
    idx, msk = [], []
    for ex in examples:
        a, b, _ = encode(ex, maxlen)
        idx.append(a); msk.append(b)
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool)


def gen_train(ops, envs, n_per, seed, Lrange=(3, 12)):
    data = []
    for gi, env in enumerate(envs):
        for op in ops:
            exs = ML.make_examples(n_per, op, env, seed + gi * 131 + hash(op) % 100, Lrange)
            for e in exs:
                e["group"] = gi
            data += exs
    random.Random(seed).shuffle(data)
    return data


# ---------------- eval (autoregressive greedy) ----------------
@torch.no_grad()
def gen_out(model, ex, maxlen, device):
    t = ex["tokens"]
    sep_pos = t.index(ML.SEP) if ML.SEP in t else t.index(ML.BOS, 1)
    prefix = t[:sep_pos + 1]
    ylen = len(ex["y"])
    seq = list(prefix)
    for _ in range(ylen):
        idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
        logits = model(idx)[0, len(seq) - 1]
        seq.append(int(logits.argmax()))
    return seq[sep_pos + 1:sep_pos + 1 + ylen]


@torch.no_grad()
def eval_op(model, op, env, n, seed, Lrange, maxlen, device):
    exs = ML.make_examples(n, op, env, seed, Lrange)
    correct = 0
    for e in exs:
        pred = gen_out(model, e, maxlen, device)
        if pred == e["y"]:
            correct += 1
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["insuf_erm", "suf_erm", "suf_dro"], default="suf_erm")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)

    ops = ["copy", "inc", "shift", "csum_reset"]
    envs = ["E0"] if args.cond == "insuf_erm" else ["E_len", "E_reset", "E_alpha", "E_tmpl"]
    data = gen_train(ops, envs, args.n_per, args.seed)
    print(f"device={device} cond={args.cond} envs={envs} ops={ops} n={len(data)} maxlen={args.maxlen}")

    model = TM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    nparam = sum(p.numel() for p in model.parameters())
    ng = len(envs); gw = torch.ones(ng, device=device) / ng            # group-DRO weights
    ptr = 0
    for step in range(args.steps):
        bd = data[ptr:ptr + args.bs]; ptr = (ptr + args.bs) % (len(data) - args.bs)
        idx, msk = batch(bd, args.maxlen, device)
        logits = model(idx[:, :-1])
        tgt = idx[:, 1:].clone(); mtgt = msk[:, 1:]
        loss_tok = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        if args.cond == "suf_dro":
            grp = torch.tensor([e["group"] for e in bd], device=device)
            gl = torch.zeros(ng, device=device)
            for g in range(ng):
                sel = (grp == g)
                if sel.any():
                    gl[g] = (loss_tok[sel] * mtgt[sel]).sum() / mtgt[sel].sum().clamp(min=1)
            gw = (gw * torch.exp(0.01 * gl.detach())); gw = gw / gw.sum()
            loss = (gw * gl).sum()
        else:
            loss = (loss_tok * mtgt).sum() / mtgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    model.eval()
    ev_env = "E_reset"                    # eval under a reset-varying env so recurrence is exercised
    print(f"\n[{args.cond}]  params={nparam/1e6:.2f}M")
    print(f"{'op':>12} {'ID[3,12]':>9} {'OOD[16,40]':>11}")
    for op in ops:
        idacc = eval_op(model, op, ev_env, args.eval_n, 9991, (3, 12), args.maxlen, device)
        oodacc = eval_op(model, op, ev_env, args.eval_n, 9992, (16, 40), args.maxlen, device)
        print(f"{op:>12} {idacc:>9.3f} {oodacc:>11.3f}")
    print("gate: csum_reset OOD[16,40] high => recurrence acquired (extrapolates); low => shortcut. "
          "expect suf_* >> insuf on csum_reset OOD.")


if __name__ == "__main__":
    main()
