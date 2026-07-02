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
STEPS = int(os.environ.get("GC_STEPS", 2500))   # per stage
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

    results = {}

    # fixed-small: append 2, train across all stages
    m = load(); grow_qwen(m, 2); set_trainable(m, 2); opt = opt_for(m)
    rng = random.Random(0)
    for mh in STAGES:
        train_stage(m, rng, mh, STEPS, opt)
    results["fixed-small(+2)"] = {h: evalh(m, h) for h in HOPS_EVAL}
    del m; torch.cuda.empty_cache()

    # grown: append 2 (easy stages) -> append 2 MORE (hard stage)
    m = load(); grow_qwen(m, 2); set_trainable(m, 2); opt = opt_for(m)
    rng = random.Random(0)
    for i, mh in enumerate(STAGES):
        if i == len(STAGES) - 1:                     # one well-timed grow before the hard stage
            grow_qwen(m, 2); set_trainable(m, 4); opt = opt_for(m)
        train_stage(m, rng, mh, STEPS, opt)
    results["grown(2->4)"] = {h: evalh(m, h) for h in HOPS_EVAL}
    del m; torch.cuda.empty_cache()

    # fixed-large: append 4, train across all stages
    m = load(); grow_qwen(m, 4); set_trainable(m, 4); opt = opt_for(m)
    rng = random.Random(0)
    for mh in STAGES:
        train_stage(m, rng, mh, STEPS, opt)
    results["fixed-large(+4)"] = {h: evalh(m, h) for h in HOPS_EVAL}
    del m; torch.cuda.empty_cache()

    print(f"\n== accuracy by hop (real Qwen, curriculum) ==")
    print(f"  {'arm':16s} " + " ".join(f"hop{h}" for h in HOPS_EVAL) + "   mean")
    for name, r in results.items():
        mean = sum(r.values()) / len(r)
        print(f"  {name:16s} " + " ".join(f"{r[h]:.2f}" for h in HOPS_EVAL) + f"   {mean:.2f}")
    g, s = results["grown(2->4)"], results["fixed-small(+2)"]
    print(f"\n  grown-vs-fixed-small hop3: {g[3]:.2f} vs {s[3]:.2f}")
    print("  grown >= fixed-small (esp hop3) => cadence-timed grow-smarter TRANSFERS to real Qwen;")
    print("  fixed >= grown => it does not (pretrained Qwen not capacity-bound like toy-L2).")


if __name__ == "__main__":
    main()
