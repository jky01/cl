"""A — does 'grow AND get smarter with the right cadence' transfer to a REAL model?
Toy autocap/growlarge3 showed ONE well-timed grow beats fixed-small on an escalating
curriculum. Here the faithful Qwen version: frozen Qwen-0.5B + APPENDED trainable
decoder layers, an escalating in-context multi-hop curriculum (hop 1->3), three
cadences at matched budget:
  fixed-small   append 2 layers, train them across the whole curriculum
  grown         append 2 (train easy stages) -> append 2 MORE mid-curriculum (train hard)
  fixed-large   append 4 layers, train them across the whole curriculum
Accuracy per hop at the end. If grown >= fixed-small (esp. hop3) the cadence result
transfers; if fixed-small/large win, it does not (Qwen isn't capacity-bound like toy-L2).

  python3 -m s0.qwen_growcap_curric      # env: QWEN_MODEL, GC_STEPS
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_grow import grow_qwen
from .qwen_growcap import single_tok_names, make

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
STAGES = [1, 2, 3]                          # escalating max-hop curriculum
STEPS = int(os.environ.get("GC_STEPS", 2000))   # per stage
SEEDS = int(os.environ.get("GC_SEEDS", 3))
LR = 1.5e-4
B = 24
HOPS_EVAL = [1, 2, 3]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = single_tok_names(tok)
    print(f"A: grow-cadence on real Qwen ({NAME}, {device}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"stages={STAGES} steps/stage={STEPS} usable-names={len(names)}")

    def load():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def set_trainable(m, n):                # train the top-n (appended) layers only
        for p in m.parameters():
            p.requires_grad_(False)
        for lyr in m.model.layers[-n:]:
            for p in lyr.parameters():
                p.requires_grad_(True)

    def batch(m, rng, hop):
        prompts, ans = [], []
        for _ in range(B):
            p, a = make(rng, names, hop)
            prompts.append(p); ans.append(a)
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        aid = torch.tensor([tok(" " + a, add_special_tokens=False).input_ids[0] for a in ans], device=device)
        return enc, aid

    def train_stage(m, rng, max_hop, steps, opt):
        m.train()
        for _ in range(steps):
            hop = rng.randint(1, max_hop)
            enc, aid = batch(m, rng, hop)
            logits = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float()
            loss = F.cross_entropy(logits, aid)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
        m.eval()

    @torch.no_grad()
    def evalh(m, hop, n=256):
        rng = random.Random(999)
        ok = tot = 0
        for i in range(0, n, B):
            enc, aid = batch(m, rng, hop)
            pred = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == aid).sum().item(); tot += aid.numel()
        return ok / tot

    def opt_for(m):
        return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)

    def run_arm(arm, seed):
        torch.manual_seed(seed); rng = random.Random(seed)
        m = load()
        if arm == "fixed-large":
            grow_qwen(m, 4); set_trainable(m, 4); opt = opt_for(m)
            for mh in STAGES:
                train_stage(m, rng, mh, STEPS, opt)
        elif arm == "grown":
            grow_qwen(m, 2); set_trainable(m, 2); opt = opt_for(m)
            for i, mh in enumerate(STAGES):
                if i == len(STAGES) - 1:              # one well-timed grow before the hard stage
                    grow_qwen(m, 2); set_trainable(m, 4); opt = opt_for(m)
                train_stage(m, rng, mh, STEPS, opt)
        else:                                         # fixed-small
            grow_qwen(m, 2); set_trainable(m, 2); opt = opt_for(m)
            for mh in STAGES:
                train_stage(m, rng, mh, STEPS, opt)
        r = {h: evalh(m, h) for h in HOPS_EVAL}
        del m; torch.cuda.empty_cache()
        return r

    arms = ["fixed-small", "grown", "fixed-large"]
    agg = {a: {h: [] for h in HOPS_EVAL} for a in arms}
    margins = []
    for seed in range(SEEDS):
        res = {a: run_arm(a, seed) for a in arms}
        for a in arms:
            for h in HOPS_EVAL:
                agg[a][h].append(res[a][h])
        gmean = sum(res["grown"].values()) / 3; smean = sum(res["fixed-small"].values()) / 3
        margins.append(gmean - smean)
        print(f"  seed {seed}: grown mean {gmean:.3f}  fixed-small mean {smean:.3f}  "
              f"margin {gmean - smean:+.3f}  (hop3 {res['grown'][3]:.2f} vs {res['fixed-small'][3]:.2f})", flush=True)

    m_ = lambda a, h: sum(agg[a][h]) / len(agg[a][h])
    print(f"\n== mean accuracy by hop over {SEEDS} seeds (real Qwen) ==")
    print(f"  {'arm':13s} " + " ".join(f"hop{h}" for h in HOPS_EVAL) + "   mean")
    for a in arms:
        mean = sum(m_(a, h) for h in HOPS_EVAL) / 3
        print(f"  {a:13s} " + " ".join(f"{m_(a,h):.2f}" for h in HOPS_EVAL) + f"   {mean:.2f}")
    pos = sum(1 for x in margins if x > 0)
    mm = sum(margins) / len(margins)
    print(f"\n  grown-minus-fixed-small margin: mean {mm:+.3f}, positive in {pos}/{SEEDS} seeds  {[round(x,3) for x in margins]}")
    print("  consistently positive => grow-cadence-smarter robustly transfers to Qwen; mixed/near-0")
    print("  => the single-seed +0.04 was within noise (honest down-adjust).")


if __name__ == "__main__":
    main()
