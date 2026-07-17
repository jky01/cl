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
def gen_full(model, ex, maxlen, device, max_out, force_len=None):
    """generate output; if force_len set, emit exactly force_len tokens (EOS suppressed) = fixed-horizon."""
    t = ex["tokens"]
    sep = t.index(ML.SEP) if ML.SEP in t else t.index(ML.BOS, 1)
    seq = list(t[:sep + 1]); out = []
    steps = force_len if force_len else max_out
    for _ in range(steps):
        idx = torch.tensor([seq + [PAD] * (maxlen - len(seq))], device=device)[:, :maxlen]
        logits = model(idx)[0, len(seq) - 1]
        if force_len is not None:
            logits[ML.EOS] = -1e9                      # suppress EOS -> isolate the transition/length ctrl
        nxt = int(logits.argmax())
        seq.append(nxt)
        if force_len is None and nxt == ML.EOS:
            break
        out.append(nxt)
    return out


@torch.no_grad()
def teacher_forced_tokacc(model, op, L, n, maxlen, device, seed):
    """feed GOLD prefix; next-token acc at each output position (no rollout error). isolates transition."""
    exs = ML.make_examples(n, op, "E_reset", seed, (L, L))
    corr = tot = 0
    for e in exs:
        t = e["tokens"]
        sep = t.index(ML.SEP) if ML.SEP in t else t.index(ML.BOS, 1)
        idx = torch.tensor([t + [PAD] * (maxlen - len(t))], device=device)[:, :maxlen]
        logits = model(idx)[0]
        for p in range(sep + 1, len(t) - 1):          # predict token at p+1 from gold prefix
            corr += int(int(logits[p].argmax()) == t[p + 1]); tot += 1
    return corr / max(tot, 1)


def evalL(model, op, L, n, maxlen, device, seed):
    exs = ML.make_examples(n, op, "E_reset", seed, (L, L))
    em = tok = lenok = fh_em = 0
    for e in exs:
        pred = gen_full(model, e, maxlen, device, L + 6)
        y = e["y"]
        lenok += int(len(pred) == len(y))
        em += int(pred == y)
        m = min(len(pred), len(y)); tok += (sum(int(pred[i] == y[i]) for i in range(m)) / len(y)) if y else 0
        fh = gen_full(model, e, maxlen, device, L + 6, force_len=len(y))    # fixed-horizon decode
        fh_em += int(fh == y)
    tf = teacher_forced_tokacc(model, op, L, min(n, 60), maxlen, device, seed + 1)
    return em / n, tok / n, lenok / n, fh_em / n, tf


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
    print(f"device={device} LEN-DIAG suf_erm trained len[3,12]; graded sweep + TF + fixed-horizon")
    print(f"{'op':>11} {'L':>4} {'free_em':>7} {'tokacc':>7} {'len_ok':>7} {'fixedH_em':>9} {'TF_tokacc':>9}")
    for op in ["copy", "csum_reset"]:
        for L in [12, 13, 14, 16, 20, 30, 40]:
            em, tk, lo, fh, tf = evalL(model, op, L, args.eval_n, args.maxlen, device, 4000 + L)
            print(f"{op:>11} {L:>4} {em:>7.3f} {tk:>7.3f} {lo:>7.3f} {fh:>9.3f} {tf:>9.3f}")
    print("\nCAUSAL SPLIT: fixedH_em high while free_em low => EOS/length-control failure (cheap fix). "
          "TF_tokacc high while free/fixedH low => exposure-bias rollout amplification. TF_tokacc cliffs "
          "at L>12 => the transition itself is position-broken (position probe justified).")


if __name__ == "__main__":
    main()
