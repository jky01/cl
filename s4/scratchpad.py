#!/usr/bin/env python3
"""Scratchpad / step-execution arm (codex fix arm 3): force the LOCAL step so length is emergent.
The diagnostic ruled out absolute-position (posrand) and length-coverage (wider curriculum): a plain
transformer with separated I/O learns length-bounded templates. Here we INTERLEAVE input and output
(x1 y1 x2 y2 ... xL yL) so csum_reset's recurrence y_i=(y_{i-1}+x_i) mod V is a LOCAL step (y_i depends
on the adjacent x_i and the previous y_{i-1}), and optionally restrict attention to a LOCAL WINDOW so the
computation is provably position/length-invariant. Then length generalization should follow.

Honest label (codex): scratchpad SUPPLIES the step decomposition; the model learns to execute it. Not
autonomous discovery. Inference uses a growing generated trace (not constant-compute / memory-free in
the strong sense), but no external retrieval. Train len [3,12]; test to 40.
"""
import argparse, sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import TM, NTOK, PAD, rope


# interleaved serialization: BOS CMD x1 y1 x2 y2 ... xL yL EOS ; loss on the y positions.
def make_inter(op, x):
    y = ML.apply_op(op, x)
    toks = [ML.BOS, ML.CMD[op]]
    ypos = []
    for xi, yi in zip(x, y):
        toks.append(xi); toks.append(yi); ypos.append(len(toks) - 1)
    toks.append(ML.EOS)
    return toks, ypos, x, y


def batch_inter(exs, maxlen, device):
    idx, msk = [], []
    for toks, ypos, _, _ in exs:
        a = toks + [PAD] * (maxlen - len(toks))
        m = [0] * maxlen
        for p in ypos:
            if p < maxlen:
                m[p] = 1
        idx.append(a[:maxlen]); msk.append(m[:maxlen])
    return torch.tensor(idx, device=device), torch.tensor(msk, device=device, dtype=torch.bool)


class LocalTM(TM):
    """TM but with optional sliding-window causal attention (window W) to force locality."""
    def __init__(s, W=0, **kw):
        super().__init__(**kw); s.W = W
        if W:
            for b in s.blocks:
                b.W = W
                b.forward = _local_forward.__get__(b)


def _local_forward(s, x, pos, mask):
    B, T, D = x.shape
    q, k, v = s.qkv(s.ln1(x)).split(D, 2)
    q = q.view(B, T, s.h, D // s.h).transpose(1, 2)
    k = k.view(B, T, s.h, D // s.h).transpose(1, 2)
    v = v.view(B, T, s.h, D // s.h).transpose(1, 2)
    q, k = rope(q, pos), rope(k, pos)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
    i = torch.arange(T, device=x.device)
    win = (i[None, :] < i[:, None] - s.W + 1)                # too-far-back -> masked (local window)
    att = att.masked_fill(mask | win[None, None], float("-inf")).softmax(-1)
    o = (att @ v).transpose(1, 2).reshape(B, T, D)
    x = x + s.proj(o)
    return x + s.f2(F.gelu(s.f1(s.ln2(x))))


@torch.no_grad()
def eval_inter(model, op, L, n, maxlen, device, seed):
    rng = random.Random(seed); em = 0
    for _ in range(n):
        x = [rng.randrange(0, ML.V) for _ in range(L)]
        if rng.random() < 0.6:
            x[rng.randrange(0, L)] = ML.RESET
        _, _, xx, y = make_inter(op, x)
        seq = [ML.BOS, ML.CMD[op]]; pred = []
        ok = True
        for i in range(L):
            seq.append(xx[i])                                 # feed x_i
            idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
            yi = int(model(idx)[0, len(seq) - 1].argmax())    # predict y_i
            seq.append(yi); pred.append(yi)
        em += int(pred == y)
    return em / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per", type=int, default=4000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=0)          # 0 = full attention; >0 = local window
    ap.add_argument("--eval_n", type=int, default=150)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    ops = ["copy", "csum_reset"]
    rng = random.Random(args.seed)
    data = []
    for _ in range(args.n_per):
        for op in ops:
            for env in ["E_len", "E_reset"]:
                L = rng.randint(3, 12)
                x = [rng.randrange(0, ML.V) for _ in range(L)]
                nres = rng.choice([0, 1, 2]) if env == "E_reset" else (1 if rng.random() < 0.5 else 0)
                for _r in range(nres):
                    x[rng.randrange(0, L)] = ML.RESET
                data.append(make_inter(op, x))
    random.shuffle(data)
    model = (LocalTM(W=args.window) if args.window else TM()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ptr = 0
    for step in range(args.steps):
        bd = data[ptr:ptr + args.bs]; ptr = (ptr + args.bs) % (len(data) - args.bs)
        idx, msk = batch_inter(bd, args.maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    tag = f"window={args.window}" if args.window else "full-attn"
    print(f"device={device} SCRATCHPAD interleaved I/O ({tag}); trained len[3,12]; test to 40")
    print(f"{'op':>11} " + " ".join(f"L{L:>2}" for L in [12, 14, 16, 20, 30, 40]))
    for op in ops:
        accs = [eval_inter(model, op, L, args.eval_n, args.maxlen, device, 5000 + L) for L in [12, 14, 16, 20, 30, 40]]
        print(f"{op:>11} " + " ".join(f"{a:>4.2f}" for a in accs))
    print("\ninterleaved local step: exact-match STAYS HIGH at L=14..40 => the model executes the "
          "recurrence step-by-step and length-extrapolates (forward derivation). still cliffs => even "
          "the local-step format doesn't generalize -> try local window attention.")


if __name__ == "__main__":
    main()
