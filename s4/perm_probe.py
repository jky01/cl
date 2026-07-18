#!/usr/bin/env python3
"""Interpret the stream negative result: a frozen csum-trunk + r=8 FFN slot could NOT acquire a permutation
recurrence s_i=(s_{i-1}+P(x_i)) mod V (op1 = 0.00 at every length), though it base-learned csum perfectly.

Before calling this a representation frontier ("primitive needs retentive trunk rewrite"), rule out mundane
causes. Probes on ONE fixed permutation op:
  A. scratch-full : fresh model, ALL params trainable        -> is the recurrence learnable by the substrate?
  B. frozen r=8/32/64 FFN slot over the csum-trunk           -> does more isolated capacity acquire it?
  C. frozen ATTN-side slot (r=32) over the csum-trunk        -> is it a PLACEMENT issue (relabel must precede
                                                                the frozen aggregation)?
If A succeeds but B/C fail across capacity, the frozen trunk is genuinely insufficient for input-relabel
primitives -> the codex adapter-frontier / retentive-rewrite question. If B or C succeed, it was capacity/
placement and the stream family is fine (just needs that config).
"""
import argparse, sys, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from stream import Net, Adapter, gen_data, train, ev, perm_for, run_perm


def acc(m, c, P, slot, maxlen, device, en=150):
    return [ev(m, c, P, slot, L, en, maxlen, device, 700 + L) for L in [8, 12, 20, 40]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=1)                 # which permutation op to probe
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n", type=int, default=40000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxlen", type=int, default=96); ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    c, P = args.c, perm_for(args.c)
    Ls = [8, 12, 20, 40]
    dc = gen_data(c, P, args.n, args.seed + c)
    print(f"device={device} PERM_PROBE c={c} P={P}\nLs={Ls}")

    # ---- base csum-trunk (op0 identity), frozen for B/C ----
    base = Net(W=args.W, r=64, R=1).to(device)
    d0 = gen_data(0, perm_for(0), args.n, args.seed)
    train(base, d0, args.steps, args.bs, args.lr, args.maxlen, device, base.trunk_params())
    trunk_sd = {k: v.detach().clone() for k, v in base.state_dict().items() if ".slots." not in k}
    print("base csum(op0): " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, acc(base, 0, perm_for(0), None, args.maxlen, device))))

    # ---- A. scratch-full: fresh model, everything trainable, on op c ----
    m = Net(W=args.W, r=8, R=1).to(device)
    m.active = 0
    train(m, dc, args.steps, args.bs, args.lr, args.maxlen, device, list(m.parameters()))
    print("A scratch-full   : " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, acc(m, c, P, 0, args.maxlen, device))))

    # ---- B. frozen trunk + FFN slot, r in {8,32,64} ----
    for r in [8, 32, 64]:
        m = Net(W=args.W, r=r, R=1).to(device); m.load_state_dict(trunk_sd, strict=False)
        train(m, dc, args.steps, args.bs, args.lr, args.maxlen, device, m.slot_params(0))
        print(f"B frozen ffn r={r:>2}: " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, acc(m, c, P, 0, args.maxlen, device))))

    # ---- C. frozen trunk + ATTN-input slot r=32 (relabel BEFORE aggregation) ----
    m = AttnSlotNet(W=args.W, r=32).to(device); m.load_state_dict(trunk_sd, strict=False)
    train(m, dc, args.steps, args.bs, args.lr, args.maxlen, device,
          [p for n, p in m.named_parameters() if "aslot" in n])
    print(f"C frozen attn r=32: " + " ".join(f"L{L}:{a:.2f}" for L, a in zip(Ls, acc(m, c, P, 0, args.maxlen, device))))
    print("\nread: A high & B/C low across r => frozen trunk insufficient for input-relabel (retentive "
          "rewrite frontier). B or C high => capacity/placement, stream family is fine with that config.")


class AttnSlotNet(Net):
    """slot applied to the token stream BEFORE each block's attention (pre-aggregation relabel site)."""
    def __init__(s, **kw):
        r = kw.pop("r", 32); super().__init__(r=8, R=1, **kw)
        d = s.emb.embedding_dim
        s.aslot = nn.ModuleList([Adapter(d, r) for _ in range(len(s.blocks))])

    def forward(s, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)[None, None]
        x = s.emb(idx)
        for b, a in zip(s.blocks, s.aslot):
            x = x + a(x)                                        # pre-attention relabel residual
            x = b(x, pos, mask, None)
        return s.head(s.lnf(x))


if __name__ == "__main__":
    main()
