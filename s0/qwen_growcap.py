"""CLOUD experiment B: does GROWTH add capability a real model can't get by
IN-PLACE adaptation? The cleanest, EXACTLY param-matched control:

  BASE     frozen Qwen (floor)
  INPLACE  unfreeze the TOP N existing decoder layers, train them
  GROW     append N identity layers, train ONLY those N

Both trained arms optimise exactly N Qwen decoder layers for the same steps on
the same data; the ONLY difference is DEPTH (24 vs 24+N). If GROW > INPLACE at
high hop-count, added depth-via-growth buys capability that adapting existing
layers cannot. If they tie, the benefit is just trainable capacity (honest null;
expected-possible since Qwen is already deep -- +N is a small relative increase).

Task: in-context multi-hop over single-token names ("A's friend is B." shuffled,
with distractors; query "X's friend's friend is" -> the hop-th name). Base fails
at high hop; training should lift it. fp32 for unattended stability.

  python3 -m s0.qwen_growcap
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_grow import grow_qwen
from .qwen_retrieval import NAMES

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
NLAYER = int(os.environ.get("GC_NLAYER", 4))
STEPS = int(os.environ.get("GC_STEPS", 4000))
LR = float(os.environ.get("GC_LR", 1.5e-4))
B = int(os.environ.get("GC_B", 24))
HOPS_TRAIN = [1, 2, 3]
HOPS_EVAL = [1, 2, 3]
N_DISTRACT_PAIRS = 4


def single_tok_names(tok):
    out = []
    for n in NAMES:
        if len(tok(" " + n, add_special_tokens=False).input_ids) == 1:
            out.append(n)
    return out


def make(rng, names, hop):
    chain = rng.sample(names, hop + 1)
    edges = [(chain[i], chain[i + 1]) for i in range(hop)]
    others = [n for n in names if n not in chain]
    ds = rng.sample(others, min(2 * N_DISTRACT_PAIRS, len(others) - len(others) % 2))
    for i in range(0, len(ds) - 1, 2):
        edges.append((ds[i], ds[i + 1]))
    rng.shuffle(edges)
    facts = " ".join(f"{a}'s friend is {b}." for (a, b) in edges)
    q = f" {chain[0]}" + "'s friend" * hop + " is"
    return facts + q, chain[-1]


def batch(rng, names, tok, device, n, hops):
    prompts, ans = [], []
    for _ in range(n):
        p, a = make(rng, names, rng.choice(hops))
        prompts.append(p); ans.append(a)
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    aid = torch.tensor([tok(" " + a, add_special_tokens=False).input_ids[0] for a in ans],
                       device=device)
    return enc, aid


@torch.no_grad()
def eval_hop(model, tok, names, device, rng, hop, n=256):
    enc, aid = batch(rng, names, tok, device, n, [hop])
    logits = model(**enc, use_cache=False).logits[:, -1].float()
    return (logits.argmax(-1) == aid).float().mean().item()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def set_top_n_trainable(model, n):
    for p in model.parameters():
        p.requires_grad_(False)
    for lyr in model.model.layers[-n:]:
        for p in lyr.parameters():
            p.requires_grad_(True)


def load(device):
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
    return m


def run_arm(tag, grow, names, tok, device, seed):
    rng = random.Random(seed)
    model = load(device)
    if grow:
        grow_qwen(model, NLAYER)
    set_top_n_trainable(model, NLAYER)
    n_train = sum(p.numel() for p in trainable(model))
    depth = len(model.model.layers)
    opt = torch.optim.AdamW(trainable(model), lr=LR)
    model.train()
    for step in range(STEPS):
        enc, aid = batch(rng, names, tok, device, B, HOPS_TRAIN)
        logits = model(**enc, use_cache=False).logits[:, -1].float()
        loss = F.cross_entropy(logits, aid)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()
        if step % 800 == 0 or step == STEPS - 1:
            model.eval()
            ev = {h: eval_hop(model, tok, names, device, random.Random(999), h) for h in HOPS_EVAL}
            model.train()
            print(f"  [{tag}] step {step:4d} loss {loss.item():.3f} depth {depth} "
                  f"trainable {n_train/1e6:.1f}M | " +
                  " ".join(f"h{h}:{ev[h]:.2f}" for h in HOPS_EVAL), flush=True)
    model.eval()
    final = {h: eval_hop(model, tok, names, device, random.Random(1234), h, n=512) for h in HOPS_EVAL}
    del model, opt
    torch.cuda.empty_cache()
    return final, n_train, depth


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = single_tok_names(tok)
    print(f"growth-adds-capability vs in-place ({NAME}, {device}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    print(f"  usable single-token names: {len(names)}; NLAYER={NLAYER} STEPS={STEPS}")

    # BASE floor (frozen, no training)
    base = load(device); base.eval()
    b_ev = {h: eval_hop(base, tok, names, device, random.Random(1234), h, n=512) for h in HOPS_EVAL}
    print("  [BASE] frozen floor | " + " ".join(f"h{h}:{b_ev[h]:.2f}" for h in HOPS_EVAL), flush=True)
    del base; torch.cuda.empty_cache()

    ip, ip_n, ip_d = run_arm("INPLACE", False, names, tok, device, seed=0)
    gr, gr_n, gr_d = run_arm("GROW", True, names, tok, device, seed=0)

    print(f"\n== final (n=512), NLAYER={NLAYER}, param-matched trainable ==")
    print(f"  arm       depth  trainable |  " + " ".join(f"hop{h}" for h in HOPS_EVAL))
    print(f"  BASE      {24:>5}     0.0M  |  " + " ".join(f"{b_ev[h]:.2f}" for h in HOPS_EVAL))
    print(f"  INPLACE   {ip_d:>5}  {ip_n/1e6:>5.1f}M  |  " + " ".join(f"{ip[h]:.2f}" for h in HOPS_EVAL))
    print(f"  GROW      {gr_d:>5}  {gr_n/1e6:>5.1f}M  |  " + " ".join(f"{gr[h]:.2f}" for h in HOPS_EVAL))
    diff = {h: gr[h] - ip[h] for h in HOPS_EVAL}
    print(f"  GROW-INPLACE gap:            |  " + " ".join(f"{diff[h]:+.2f}" for h in HOPS_EVAL))
    print("\n  GROW > INPLACE at high hop => depth-via-growth adds capability in-place")
    print("  adaptation can't (param-matched). Tie => benefit is capacity, not depth (honest).")


if __name__ == "__main__":
    main()
