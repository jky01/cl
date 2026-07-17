#!/usr/bin/env python3
"""Position-randomization arm (codex fix arm 1): the length-diag causal split showed the transition is
position-broken past L=13 (teacher-forced collapses). Train with a random absolute-position OFFSET per
batch so high positions are exposed WITHOUT long sequences; test whether the transition then generalizes
to real length-40. Labeled a position-DISTRIBUTION probe (it exposes positions 13-40 via offset), not
proof of emergent rollout. Compares to the zero-offset baseline; reuses s4/len_diag eval.
"""
import argparse, sys, os, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from train_gate import TM, batch, NTOK
from len_diag import evalL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n_per", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--max_offset", type=int, default=90); ap.add_argument("--randpos", type=int, default=1)
    ap.add_argument("--train_hi", type=int, default=12)      # wider-curriculum: train len [3, train_hi]
    ap.add_argument("--eval_lens", type=int, nargs="+", default=[12, 13, 14, 16, 20, 30, 40])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    ops = ["copy", "inc", "shift", "csum_reset"]
    data = []
    for gi, env in enumerate(["E_len", "E_reset", "E_alpha", "E_tmpl"]):
        for op in ops:
            data += ML.make_examples(args.n_per, op, env, args.seed + gi * 131, (3, args.train_hi))
    random.Random(args.seed).shuffle(data)
    model = TM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(args.seed); ptr = 0
    for step in range(args.steps):
        bd = data[ptr:ptr + args.bs]; ptr = (ptr + args.bs) % (len(data) - args.bs)
        idx, msk = batch(bd, args.maxlen, device)
        off = rng.randint(0, args.max_offset) if args.randpos else 0     # random absolute-position offset
        logits = model(idx[:, :-1], pos_offset=off)
        tgt = idx[:, 1:]; mt = msk[:, 1:]
        lt = F.cross_entropy(logits.reshape(-1, NTOK), tgt.reshape(-1), reduction="none").view(tgt.shape)
        loss = (lt * mt).sum() / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    tag = f"randpos(max_offset={args.max_offset})" if args.randpos else "zero-offset baseline"
    print(f"device={device} POSRAND {tag}; trained len[3,%d]; eval offset 0")
    print(f"{'op':>11} {'L':>4} {'free_em':>7} {'tokacc':>7} {'fixedH_em':>9} {'TF_tokacc':>9}")
    for op in ["copy", "csum_reset"]:
        for L in args.eval_lens:
            em, tk, lo, fh, tf = evalL(model, op, L, args.eval_n, args.maxlen, device, 4000 + L)
            print(f"{op:>11} {L:>4} {em:>7.3f} {tk:>7.3f} {fh:>9.3f} {tf:>9.3f}")
    print("\nif TF_tokacc & free_em now stay HIGH at L=14..40 => exposing high absolute positions (via "
          "offset) fixed the transition = it was a position-coverage artifact. If still cliffs => rotary "
          "phase coverage alone is insufficient; the transition isn't shift-invariant -> escalate to "
          "wider-curriculum / scratchpad.")


if __name__ == "__main__":
    main()
