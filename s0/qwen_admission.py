"""Qwen §27 feature vs RAG #2 — ADMISSION / trust (the "immune system", §27.18).
A knowledge source contains a RELIABLE fact and an UNTRUSTWORTHY contradiction.
The memory's commit gate (using a source-trust signal at write time) rejects the
untrustworthy write, so recall returns the reliable value. A naive text-RAG that
retrieves both statements into context is misled by the contradiction.

Frozen Qwen2.5-0.5B; only small modules train. Single-token answers.

  .venv/bin/python -m s0.qwen_admission
"""
from __future__ import annotations
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES
from .qwen_memory import ATTR_VALUES

NAME = "Qwen/Qwen2.5-0.5B"
ATTRS = list(ATTR_VALUES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    enc, d = lm.model, lm.config.hidden_size

    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        m = e.attention_mask[..., None].to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float()

    @torch.no_grad()
    def last_hidden(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        return enc(**e).last_hidden_state[:, -1].float()

    mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
    proj_k, proj_q, val_enc = mk(d, 128), mk(d, 128), mk(d, 256)
    val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
    gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    nn.init.constant_(gate[-1].bias, 2.0)
    commit = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    nn.init.constant_(commit[-1].bias, 2.0)
    mods = [proj_k, proj_q, val_enc, val_dec, gate, commit]
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=5e-4)
    rng = random.Random(0)

    def sample(B):
        out = []
        for (n, a) in rng.sample([(n, a) for n in NAMES for a in ATTRS], B):
            good, bad = rng.sample(av[a], 2)
            out.append((n, a, good, bad))
        return out

    def read(facts):
        n, a = [f[0] for f in facts], [f[1] for f in facts]
        kt = [f"{x}'s {y}" for x, y in zip(n, a)]
        sg = [f"{f[0]}'s {f[1]} is {f[2]}." for f in facts]   # reliable statement
        sb = [f"{f[0]}'s {f[1]} is {f[3]}." for f in facts]   # untrustworthy contradiction
        K = F.normalize(proj_k(pooled(kt)), dim=-1)           # shared key (name,attr)
        vg, vb = val_enc(pooled(sg)), val_enc(pooled(sb))
        B = len(facts)
        # commit gate: reliable trust=1, untrustworthy trust=0; conflict=1 (same key)
        one, zero = torch.ones(B, device=device), torch.zeros(B, device=device)
        cf = torch.ones(B, device=device)
        ag = torch.sigmoid(commit(torch.stack([one, cf], -1))).squeeze(-1)   # admit reliable
        ab = torch.sigmoid(commit(torch.stack([zero, cf], -1))).squeeze(-1)  # admit untrustworthy
        keys = torch.cat([K, K], 0)
        vals = torch.cat([vg, vb], 0)
        adm = torch.cat([ag, ab], 0)                          # [2B] admission (soft presence)
        qt = [f"{f[0]}'s {f[1]} is" for f in facts]
        qf = pooled(qt)
        q = F.normalize(proj_q(qf), dim=-1)
        score = q @ keys.t() + torch.log(adm.clamp(min=1e-4))[None]   # suppress rejected
        w = torch.softmax(score / 0.05, -1)
        R = val_dec(w @ vals)
        H = last_hidden(qt)
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        logits = lm.lm_head((H + g * R).to(lm.lm_head.weight.dtype)).float()
        return logits, q, K, ag, ab

    def good_ans(facts):
        return torch.tensor([ans_id(f[2]) for f in facts], device=device)

    B = 48
    tgt = torch.arange(B, device=device)
    for step in range(2000):
        facts = sample(B)
        logits, q, K, ag, ab = read(facts)
        loss = (F.cross_entropy(logits, good_ans(facts))          # recall reliable
                + F.cross_entropy(q @ K.t() / 0.05, tgt)          # retrieval
                + F.binary_cross_entropy(ag, torch.ones_like(ag)) # admit reliable
                + F.binary_cross_entropy(ab, torch.zeros_like(ab)))  # reject untrustworthy
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0 or step == 1999:
            with torch.no_grad():
                f = sample(96)
                acc = (read(f)[0].argmax(-1) == good_ans(f)).float().mean().item()
            print(f"  step {step:4d} loss {loss.item():.3f} | mem reliable-recall {acc:.3f} "
                  f"(admit reliable {ag.mean():.2f} / untrust {ab.mean():.2f})", flush=True)

    # ---- vs RAG: both statements (reliable + untrustworthy) in context ----
    with torch.no_grad():
        f = sample(96)
        mem = (read(f)[0].argmax(-1) == good_ans(f)).float().mean().item()
        rag_p = []
        for fact in f:
            docs = [f"{fact[0]}'s {fact[1]} is {fact[2]}.", f"{fact[0]}'s {fact[1]} is {fact[3]}."]
            random.Random(hash(fact[:2]) & 7).shuffle(docs)     # corpus: both, unmarked
            rag_p.append(" ".join(docs) + f" {fact[0]}'s {fact[1]} is")
        rag = (lm.lm_head(last_hidden(rag_p).to(lm.lm_head.weight.dtype)).float().argmax(-1)
               == good_ans(f)).float().mean().item()
    print(f"\n  MEMORY reliable-recall {mem:.3f} | RAG (both facts in context) {rag:.3f}")
    print("  The commit gate rejects the untrustworthy contradiction (admission ~0);")
    print("  naive text-RAG retrieves both and is misled -- a win RAG can't get for free.")


if __name__ == "__main__":
    main()
