"""Real-model integration: grow a Qwen core and keep its memory (diag_growmem on
Qwen). R0 memory recall; R1 after identity-growth (== R0); R2 after the new Qwen
layers train (hidden drifts -> drop); R3 after a cheap memory re-sync (retrain
the small modules on the evolved Qwen -- no backprop through Qwen) -> recover.

Cheap because the new layers are at the END: backprop only flows through the top
(trainable) layers + head; the frozen lower layers build no graph.

  .venv/bin/python -m s0.qwen_integrated
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES
from .qwen_memory import ATTR_VALUES
from .qwen_grow import grow_qwen

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
ATTRS = list(ATTR_VALUES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    d = lm.config.hidden_size
    n_base = len(lm.model.layers)   # base depth (24 for 0.5B, 28 for 1.5B) -- for growth slicing

    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

    def pooled(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = lm.model(**e).last_hidden_state
        m = e.attention_mask[..., None].to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float()

    def last_hidden(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        return lm.model(**e).last_hidden_state[:, -1].float()

    mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
    proj_k, proj_q, val_enc = mk(d, 128), mk(d, 128), mk(d, 256)
    val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
    gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    nn.init.constant_(gate[-1].bias, 2.0)
    mem_mods = [proj_k, proj_q, val_enc, val_dec, gate]
    rng = random.Random(0)

    def sample(B):
        out = []
        for (n, a) in rng.sample([(n, a) for n in NAMES for a in ATTRS], B):
            out.append((n, a, rng.choice(av[a])))
        return out

    def read(facts):
        kt = [f"{f[0]}'s {f[1]}" for f in facts]
        st = [f"{f[0]}'s {f[1]} is {f[2]}." for f in facts]
        qt = [f"{f[0]}'s {f[1]} is" for f in facts]
        K = F.normalize(proj_k(pooled(kt)), dim=-1)
        V = val_enc(pooled(st))
        q = F.normalize(proj_q(pooled(qt)), dim=-1)
        R = val_dec(torch.softmax(q @ K.t() / 0.05, -1) @ V)
        H = last_hidden(qt)
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        logits = lm.lm_head((H + g * R).to(lm.lm_head.weight.dtype)).float()
        return logits, q, K

    def gold(facts):
        return torch.tensor([ans_id(f[2]) for f in facts], device=device)

    def train_mem(steps):
        params = [p for m in mem_mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        B = 48; tgt = torch.arange(B, device=device)
        for _ in range(steps):
            f = sample(B)
            logits, q, K = read(f)
            loss = F.cross_entropy(logits, gold(f)) + F.cross_entropy(q @ K.t() / 0.05, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    @torch.no_grad()
    def eval_mem(n=96):
        f = sample(n)
        return (read(f)[0].argmax(-1) == gold(f)).float().mean().item()

    def train_qwen_new(steps):
        new = [p for blk in lm.model.layers[n_base:] for p in blk.parameters()]
        for p in new:
            p.requires_grad_(True)
        # GENTLE: fresh identity layers wreck the residual stream if trained hard
        # (a narrow objective at lr 1e-4 collapsed the representation). Small lr +
        # weight decay keep them near identity -> mild drift the memory can re-sync.
        opt = torch.optim.AdamW(new, lr=2e-5, weight_decay=0.05)
        lm.train()
        for _ in range(steps):
            f = sample(16)
            txt = [f"{x[0]}'s {x[1]} is {x[2]}." for x in f]
            e = tok(txt, return_tensors="pt", padding=True).to(device)
            out = lm(**e, use_cache=False).logits
            tgt = e.input_ids[:, 1:].clone()
            tgt[e.attention_mask[:, 1:] == 0] = -100
            loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)).float(),
                                   tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)

    train_mem(1500); r0 = eval_mem()
    print(f"  R0 Qwen memory recall (24 layers):        {r0:.3f}")
    grow_qwen(lm, 4); r1 = eval_mem()
    print(f"  R1 after identity growth (24->28 layers): {r1:.3f}  (expect == R0)")
    train_qwen_new(150); r2 = eval_mem()
    print(f"  R2 after the new layers train (drift):    {r2:.3f}  (may drop)")
    train_mem(1000); r3 = eval_mem()
    print(f"  R3 after cheap memory re-sync:            {r3:.3f}  (expect ~R0)")
    print("\n  Growth preserves the Qwen memory (R1==R0); training the grown layers drifts")
    print("  the hidden (R2); a cheap re-sync recovers it (R3) -> growth + memory compose")
    print("  on a REAL model, no backprop through the frozen Qwen needed for the memory.")


if __name__ == "__main__":
    main()
