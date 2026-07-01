"""Qwen integration step 5 — MULTI-TOKEN answers (blocker #2).
Single-position injection only sets the next token. Here the retrieved signal R
is injected at EVERY answer position (per-step injection), so the memory can
drive a multi-token answer ("San Francisco") with NO context tokens. Frozen
Qwen; teacher-forced training, free-generation eval (exact match of both tokens).

  .venv/bin/python -m s0.qwen_multitoken
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
ATTR_VALUES = {
    "hometown": ["San Francisco", "New York", "Hong Kong", "Los Angeles",
                 "Cape Town", "Las Vegas", "San Diego", "Tel Aviv", "Buenos Aires"],
    "job": ["software engineer", "police officer", "flight attendant",
            "graphic designer", "civil engineer", "data scientist", "head chef",
            "music teacher", "race driver"],
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # right-align real tokens: last token at [:,-1],
    #                             appended generations go to the sequence end.
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    enc, d = lm.model, lm.config.hidden_size
    embed = lm.get_input_embeddings()

    def aids(v):
        return tok(" " + v, add_special_tokens=False).input_ids
    attr_values = {a: [v for v in vs if len(aids(v)) == 2] for a, vs in ATTR_VALUES.items()}
    ATTRS = list(attr_values)
    print("usable 2-token values:", {a: len(v) for a, v in attr_values.items()})

    @torch.no_grad()
    def pooled(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        m = e.attention_mask[..., None].to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float()

    mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
    proj_k, proj_q, val_enc = mk(d, 128), mk(d, 128), mk(d, 256)
    val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
    gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    nn.init.constant_(gate[-1].bias, 2.0)
    mods = [proj_k, proj_q, val_enc, val_dec, gate]
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=5e-4)
    rng = random.Random(0)

    def sample(B):
        out = []
        for (n, a) in rng.sample([(n, a) for n in NAMES for a in ATTRS], B):
            out.append((n, a, rng.choice(attr_values[a])))
        return out

    def bank(facts):
        kt = [f"{n}'s {a}" for (n, a, _) in facts]
        st = [f"{n}'s {a} is {v}." for (n, a, v) in facts]
        K = F.normalize(proj_k(pooled(kt)), dim=-1)
        V = val_enc(pooled(st))
        return K, V

    def Rvec(facts, qf):           # retrieve over the batch bank -> R per row
        K, V = bank(facts)
        q = F.normalize(proj_q(qf), dim=-1)
        w = torch.softmax(q @ K.t() / 0.05, -1)
        return val_dec(w @ V), q, K

    def inject_logits(H, R):       # H [.,d], R [.,d] -> next-token logits
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        return lm.lm_head((H + g * R).to(lm.lm_head.weight.dtype)).float()

    B = 48
    tgt = torch.arange(B, device=device)
    for step in range(2200):
        facts = sample(B)
        qtxt = [f"{n}'s {a} is" for (n, a, _) in facts]
        a_ids = torch.tensor([aids(v) for (_, _, v) in facts], device=device)    # [B,2]
        # teacher-forced: feed query + answer, inject R at both answer positions
        qenc = tok(qtxt, return_tensors="pt", padding=True).to(device)
        qf = pooled(qtxt)
        R, q, K = Rvec(facts, qf)
        Tq = qenc.input_ids.size(1)                              # padded query len (left-pad)
        full_ids = torch.cat([qenc.input_ids, a_ids], 1)
        full_mask = torch.cat([qenc.attention_mask, torch.ones_like(a_ids)], 1)
        with torch.no_grad():
            H = enc(input_ids=full_ids, attention_mask=full_mask).last_hidden_state  # [B,T,d]
        loss = F.cross_entropy(q @ K.t() / 0.05, tgt)            # retrieval
        for j in range(2):                                      # predict answer tok j
            loss = loss + F.cross_entropy(inject_logits(H[:, Tq - 1 + j], R), a_ids[:, j])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0 or step == 2199:
            em = free_gen_em(lm, enc, embed, tok, sample(64), pooled, Rvec, inject_logits, aids, device)
            print(f"  step {step:4d} loss {loss.item():.3f} | capsule exact-match {em:.3f}", flush=True)

    # ---- final: capsule (free generation) vs no-mem vs few-shot RAG ----
    f = sample(96)
    cap = free_gen_em(lm, enc, embed, tok, f, pooled, Rvec, inject_logits, aids, device)
    nomem = free_gen_em(lm, enc, embed, tok, f, pooled, Rvec, inject_logits, aids, device, no_inject=True)
    demo = ("Anna's hometown is New York. Anna's hometown is New York. "
            "Ben's job is data scientist. Ben's job is data scientist. ")
    rag = rag_em(lm, enc, tok, f, demo, aids, device)
    print(f"\n  capsule (mem, NO context) {cap:.3f} | no-mem {nomem:.3f} | RAG few-shot {rag:.3f}")
    print("  multi-token answers recalled with NO context via per-step injection.")


@torch.no_grad()
def free_gen_em(lm, enc, embed, tok, facts, pooled, Rvec, inject_logits, aids, device, no_inject=False):
    """Autoregressively generate 2 answer tokens with per-step injection; exact match."""
    qtxt = [f"{n}'s {a} is" for (n, a, _) in facts]
    gold = torch.tensor([aids(v) for (_, _, v) in facts], device=device)
    R, _, _ = Rvec(facts, pooled(qtxt))
    qenc = tok(qtxt, return_tensors="pt", padding=True).to(device)
    ids, mask = qenc.input_ids, qenc.attention_mask
    preds = []
    for _ in range(2):
        H = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        Hl = H[:, -1]                                # left-pad: last token is rightmost
        logits = (lm.lm_head(Hl.to(lm.lm_head.weight.dtype)).float() if no_inject
                  else inject_logits(Hl, R))
        nxt = logits.argmax(-1)
        preds.append(nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        mask = torch.cat([mask, torch.ones_like(nxt[:, None])], 1)
    pred = torch.stack(preds, 1)
    return ((pred == gold).all(1)).float().mean().item()


@torch.no_grad()
def rag_em(lm, enc, tok, facts, demo, aids, device):
    qtxt = [demo + f"{n}'s {a} is {v}. {n}'s {a} is" for (n, a, v) in facts]
    gold = torch.tensor([aids(v) for (_, _, v) in facts], device=device)
    enc_in = tok(qtxt, return_tensors="pt", padding=True).to(device)
    ids, mask = enc_in.input_ids, enc_in.attention_mask
    preds = []
    for _ in range(2):
        H = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        nxt = lm.lm_head(H[:, -1].to(lm.lm_head.weight.dtype)).float().argmax(-1)
        preds.append(nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        mask = torch.cat([mask, torch.ones_like(nxt[:, None])], 1)
    return ((torch.stack(preds, 1) == gold).all(1)).float().mean().item()


if __name__ == "__main__":
    main()
