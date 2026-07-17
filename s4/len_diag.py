#!/usr/bin/env python3
"""Graded length diagnostic (codex): characterize the length-extrapolation FAILURE MODE before choosing
a fix. Train suf_erm (multi-env, len [3,12]); eval copy & csum_reset at lengths 12,14,16,20,30,40.
Report seq exact-match, token acc, and OUTPUT-LENGTH correctness (right # tokens?). Distinguishes an
absolute-position cliff from a stop/format failure from graceful decay. (reuses s4/train_gate.py)
"""
import argparse, sys, os, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import TM, batch, NTOK, PAD


@torch.no_grad()
def gen_full(model, ex, maxlen, device, max_out):
    """generate output until EOS or max_out tokens; return the produced output list (excl specials)."""
    t = ex["tokens"]
    sep = t.index(ML.SEP) if ML.SEP in t else t.index(ML.BOS, 1)
    seq = list(t[:sep + 1])
    out = []
    for _ in range(max_out):
        idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
        nxt = int(model(idx)[0, len(seq) - 1].argmax())
        seq.append(nxt)
        if nxt == ML.EOS:
            break
        out.append(nxt)
    return out


def evalL(model, op, L, n, maxlen, device, seed):
    exs = ML.make_examples(n, op, "E_reset", seed, (L, L))
    em = tok = lenok = 0
    for e in exs:
        pred = gen_full(model, e, maxlen, device, L + 5)
        y = e["y"]
        if len(pred) == len(y):
            lenok += 1
        if pred == y:
            em += 1
        m = min(len(pred), len(y))
        tok += (sum(int(pred[i] == y[i]) for i in range(m)) / len(y)) if len(y) else 0
    return em / n, tok / n, lenok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=150)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    ops = ["copy", "inc", "shift", "csum_reset"]
    envs = ["E_len", "E_reset", "E_alpha", "E_tmpl"]
    data = []
    for gi, env in enumerate(envs):
        for op in ops:
            data += ML.make_examples(args.n_per, op, env, args.seed + gi * 131, (3, 12))
    random.Random(args.seed).shuffle(data)
    model = TM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ptr = 0
    for step in range(args.steps):
        bd = data[ptr:ptr + args.bs]; ptr = (ptr + args.bs) % (len(data) - args.bs)
        idx, msk = batch(bd, args.maxlen, device)
        logits = model(idx[:, :-1]); tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    print(f"device={device} LEN-DIAG suf_erm trained len[3,12]; graded length sweep")
    print(f"{'op':>11} {'L':>4} {'exact':>6} {'tokacc':>7} {'len_ok':>7}")
    for op in ["copy", "csum_reset"]:
        for L in [12, 14, 16, 20, 30, 40]:
            em, tk, lo = evalL(model, op, L, args.eval_n, args.maxlen, device, 4000 + L)
            print(f"{op:>11} {L:>4} {em:>6.3f} {tk:>7.3f} {lo:>7.3f}")
    print("\nlen_ok = fraction emitting the correct OUTPUT LENGTH. immediate cliff at L=14 => "
          "absolute-position artifact; len_ok low => stop/format failure; graceful tokacc decay => "
          "partial length-gen.")


if __name__ == "__main__":
    main()
