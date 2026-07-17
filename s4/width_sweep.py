#!/usr/bin/env python3
"""Adapter-WIDTH frontier for routed compact growth (codex directive: find the SMALLEST capacity that
retains the old algorithm AND acquires the new one, rather than showing one oversized branch works).

Reuses s4/continual2.py's routed architecture. Phase-2 (acquire csum_reset) is trained ONCE and cached:
the adapter is zero-init and OFF during phase-2, so the trunk is r-independent. For each r we then build a
fresh routed adapter over a hard-frozen copy of that trunk and train phase-3 on rmax_reset only (no csum
replay). Predefined criterion (codex): csum L8-40 + cf within 0.03 of phase-2  AND  high rmax.

Reports per r: retain (csum L8-40 + cf), acquire (rmax L8-40), added params, adapter per-token FLOPs,
and the exact non-interference invariant (csum logits max|Δ| vs phase-2, must be 0).
"""
import argparse, sys, os, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microlang as ML
from scratchpad import batch_inter
from continual2 import Net, gen_data, train, report, counterfactual  # reuse routed arch + metrics


def trunk_state(sd):
    return {k: v for k, v in sd.items() if "ad_" not in k}          # everything except the adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps2", type=int, default=6000); ap.add_argument("--steps3", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--W", type=int, default=5); ap.add_argument("--eval_n", type=int, default=150)
    ap.add_argument("--rs", type=str, default="1,2,4,8,16,32"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)
    rs = [int(x) for x in args.rs.split(",")]
    d_csum = gen_data("csum_reset", args.n, args.seed)
    d_rmax = gen_data("rmax_reset", args.n, args.seed + 7)

    # ---- phase 2 once (trunk is r-independent: adapter is zero-init and OFF here) ----
    base = Net(W=args.W, r=max(rs)).to(device)
    train(base, d_csum, args.steps2, args.bs, args.lr, args.maxlen, device)
    d = base.emb.embedding_dim if hasattr(base.emb, "embedding_dim") else base.emb.weight.shape[1]
    nl = len(base.blocks)
    tot = sum(p.numel() for p in base.parameters()) - sum(p.numel() for n, p in base.named_parameters() if "ad_" in n)
    print(f"device={device} WIDTH_SWEEP W={args.W} d={d} nl={nl} trunk={tot/1e6:.2f}M rs={rs}\n=== phase 2 ===")
    cs2, cf2, _ = report(base, "phase2", args.maxlen, device, args.eval_n)
    trunk = trunk_state(base.state_dict())
    ex = batch_inter(gen_data("csum_reset", 64, 12345), args.maxlen, device)   # fixed invariant probe
    with torch.no_grad():
        base_csum_logits = base(ex[0][:, :-1])

    print("\n=== phase 3 per adapter width (rmax only, hard-frozen trunk, command-routed) ===")
    rows = []
    for r in rs:
        m = Net(W=args.W, r=r).to(device)
        m.load_state_dict(trunk, strict=False)          # copy trunk; adapter stays fresh zero-init
        for b in m.blocks:
            b.use_adapter = True
        m.routing = "cmd"
        params = [p for n, p in m.named_parameters() if "ad_" in n]   # train ONLY the adapter
        train(m, d_rmax, args.steps3, args.bs, args.lr, args.maxlen, device, params=params, routed=True)
        with torch.no_grad():
            inv = (base_csum_logits - m(ex[0][:, :-1], routed=True)).abs().max().item()
        add = sum(p.numel() for p in params)
        flop = 4 * d * r * nl                            # ad_dn+ad_up matmuls, per token (fwd, ~2*2*d*r*nl)
        cs, cf, rm = report(m, f"r={r}", args.maxlen, device, args.eval_n)
        retain_ok = all(abs(a - b) <= 0.03 for a, b in zip(cs, cs2)) and abs(cf - cf2) <= 0.03
        acquire = min(rm)
        rows.append((r, add, flop, retain_ok, acquire, inv))
        print(f"   r={r:>2} add={add/1e3:.1f}K flop/tok={flop} inv={inv:.1e} "
              f"retain<=0.03:{retain_ok} rmax_min:{acquire:.2f}")

    ok = [r for r, _, _, ret, acq, _ in rows if ret and acq >= 0.99]
    print(f"\nFRONTIER r* (retain within 0.03 AND rmax_min>=0.99) = {min(ok) if ok else 'NONE'} ; rows={rows}")


if __name__ == "__main__":
    main()
