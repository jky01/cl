"""Qwen integration step 3 — full write/read memory on a real frozen LM.
A free-text fact is written (key from Qwen features, value encodes the answer);
a cloze query retrieves it and INJECTS a signal at the query's final hidden so
Qwen emits the answer token -- WITHOUT the fact in context. Single-token answers
keep the injection a clean single-position hook (the multi-token / KV-prefix
version is the next step). Compares to no-memory and in-context (=RAG).

  .venv/bin/python -m s0.qwen_memory
"""
from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES

NAME = "Qwen/Qwen2.5-0.5B"
# typed value pools so each fact is SENSIBLE (random attr/value pairing makes
# nonsensical facts that even RAG fights). The model still can't know WHICH
# sensible value a given person has -> no-mem fails, memory/RAG succeed.
ATTR_VALUES = {
    "favorite color": "blue red green violet amber pink gray gold".split(),
    "hometown": "Paris London Tokyo Berlin Rome Cairo Lima Madrid".split(),
    "job": "doctor teacher pilot nurse baker dancer lawyer chef".split(),
    "favorite food": "pizza sushi ramen tacos curry pasta soup bread".split(),
    "pet": "cat dog parrot rabbit hamster turtle goldfish snake".split(),
    "favorite sport": "tennis chess soccer boxing rowing golf rugby hockey".split(),
    "favorite drink": "coffee tea juice cocoa water soda wine cider".split(),
    "hobby": "painting hiking reading pottery gardening singing fishing sewing".split(),
    "favorite season": "summer winter autumn spring".split(),
    "lucky number": "seven three nine four eight five six two".split(),
}
ATTRS = list(ATTR_VALUES)
S_TMPL = ["{n}'s {a} is {v}.", "The {a} of {n} is {v}.", "{n} has {v} as a {a}."]
Q_TMPL = ["{n}'s {a} is", "The {a} of {n} is"]   # cloze: next token = answer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    enc = lm.model
    d = lm.config.hidden_size

    # keep only single-token answers (with leading space, as Qwen BPE sees them)
    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    attr_values = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(texts):                       # mean-pooled last hidden [B,d]
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        m = e.attention_mask[..., None].to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float()

    @torch.no_grad()
    def last_hidden(texts):                  # final hidden at the last real token [B,d]
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        idx = e.attention_mask.sum(1) - 1
        return h[torch.arange(h.size(0), device=device), idx].float()

    proj_k = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)).to(device)
    proj_q = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)).to(device)
    val_enc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 256)).to(device)
    val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
    gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    nn.init.constant_(gate[-1].bias, 2.0)
    mods = [proj_k, proj_q, val_enc, val_dec, gate]
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=5e-4)   # lower lr + grad clip = stable
    rng = random.Random(0)

    def sample(B):
        facts, ss, qs, ans = [], [], [], []
        pairs = rng.sample([(n, a) for n in NAMES for a in ATTRS], B)
        for (n, a) in pairs:
            v = rng.choice(attr_values[a])
            ss.append(rng.choice(S_TMPL).format(n=n, a=a, v=v))
            qs.append(rng.choice(Q_TMPL).format(n=n, a=a))
            ans.append(ans_id(v))
        return ss, qs, torch.tensor(ans, device=device)

    def read_logits(ss, qs):
        k = F.normalize(proj_k(pooled(ss)), dim=-1)          # [B,128] bank keys
        v = val_enc(pooled(ss))                              # [B,256] bank values
        q = F.normalize(proj_q(pooled(qs)), dim=-1)          # [B,128]
        H = last_hidden(qs)                                  # [B,d]
        w = torch.softmax(q @ k.t() / 0.05, dim=-1)          # [B,B] retrieve over bank
        R = val_dec(w @ v)                                   # [B,d]
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))       # [B,1]
        # direct additive injection: lm_head is linear (tied emb), so adding R
        # cleanly boosts the answer-token logit (R learns to point at emb[answer]).
        logits = lm.lm_head((H + g * R).to(lm.lm_head.weight.dtype)).float()
        return logits, q, k

    for step in range(2000):
        ss, qs, ans = sample(64)
        logits, q, k = read_logits(ss, qs)
        ce = F.cross_entropy(logits, ans)
        retr = F.cross_entropy(q @ k.t() / 0.05, torch.arange(64, device=device))
        loss = ce + retr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0 or step == 1999:
            with torch.no_grad():
                ss, qs, ans = sample(64)
                logits, _, _ = read_logits(ss, qs)
                acc = (logits.argmax(-1) == ans).float().mean().item()
            print(f"  step {step:4d} loss {loss.item():.3f} capsule_acc {acc:.3f}")

    # ---- final comparison vs no-memory and in-context (RAG) ----
    with torch.no_grad():
        ss, qs, ans = sample(128)
        cap = (read_logits(ss, qs)[0].argmax(-1) == ans).float().mean().item()
        # no memory: Qwen predicts the cloze next token with no fact available
        H = last_hidden(qs)
        nomem = (lm.lm_head(H.to(lm.lm_head.weight.dtype)).float().argmax(-1) == ans).float().mean().item()
        # in-context RAG, zero-shot: statement before the cloze query
        rag_q = [s + " " + q for s, q in zip(ss, qs)]
        rag0 = (lm.lm_head(last_hidden(rag_q).to(lm.lm_head.weight.dtype)).float().argmax(-1) == ans).float().mean().item()
        # FAIR RAG: few-shot demos prime the small base model to copy
        demo = ("Anna's hometown is Rome. Anna's hometown is Rome. "
                "Ben's pet is cat. Ben's pet is cat. "
                "Cara's job is chef. Cara's job is chef. ")
        rag_fs = [demo + s + " " + q for s, q in zip(ss, qs)]
        ragf = (lm.lm_head(last_hidden(rag_fs).to(lm.lm_head.weight.dtype)).float().argmax(-1) == ans).float().mean().item()
    print(f"\n  capsule (mem, NO context) {cap:.3f} | no-mem {nomem:.3f} | "
          f"RAG zero-shot {rag0:.3f} | RAG few-shot {ragf:.3f}")
    print("  HONEST read: the memory recalls facts on a real frozen LM with NO context")
    print("  (vs no-mem ~0). The capsule is TRAINED; RAG is zero/few-shot -- different")
    print("  paradigms. Few-shot RAG is the fair in-context baseline.")


if __name__ == "__main__":
    main()
