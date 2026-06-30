"""Where structured memory beats raw-text RAG: TEMPORAL UPDATES.
Each fact has two versions written over time (v1@t=0 then v2@t=1, sharing a
key). A query asks for the current OR the original value; the memory routes by
explicit time stamp. A naive text-RAG that retrieves both (undated, unordered)
statements cannot tell which is current vs original.

Key design: the retrieval key is built from (name, attribute) ONLY (both
versions share it); the value is encoded from the full statement (with the
answer). Frozen Qwen2.5-0.5B; only small modules train.

  .venv/bin/python -m s0.qwen_conflict
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
    lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    enc, d = lm.model, lm.config.hidden_size

    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    attr_values = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        m = e.attention_mask[..., None].to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float()

    @torch.no_grad()
    def last_hidden(texts):
        e = tok(texts, return_tensors="pt", padding=True).to(device)
        h = enc(**e).last_hidden_state
        idx = e.attention_mask.sum(1) - 1
        return h[torch.arange(h.size(0), device=device), idx].float()

    mk = lambda o: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, o)).to(device)
    proj_k, proj_q, val_enc = mk(128), mk(128), mk(256)
    val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
    ctx_enc = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)).to(device)
    gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    nn.init.constant_(gate[-1].bias, 2.0)
    mods = [proj_k, proj_q, val_enc, val_dec, ctx_enc, gate]
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=5e-4)
    rng = random.Random(0)

    def sample(B):
        facts = []
        for (n, a) in rng.sample([(n, a) for n in NAMES for a in ATTRS], B):
            v1, v2 = rng.sample(attr_values[a], 2)
            facts.append((n, a, v1, v2))
        return facts

    def bank(facts):
        kt = [f"{n}'s {a}" for (n, a, _, _) in facts]                 # key text (name,attr)
        s1 = [f"{n}'s {a} is {v1}." for (n, a, v1, _) in facts]
        s2 = [f"{n}'s {a} is {v2}." for (n, a, _, v2) in facts]
        K = F.normalize(proj_k(pooled(kt)), dim=-1)                   # [B,128]
        v1e, v2e = val_enc(pooled(s1)), val_enc(pooled(s2))           # [B,256]
        B = len(facts)
        keys = torch.cat([K, K], 0)                                  # [2B,128] versions share key
        vals = torch.cat([v1e, v2e], 0)                              # [2B,256]
        times = torch.cat([torch.zeros(B), torch.ones(B)]).to(device)  # v1=0, v2=1
        return keys, vals, times, K

    def qtexts(facts, which):
        return [(f"Currently, {n}'s {a} is" if which == "current"
                 else f"Originally, {n}'s {a} was") for (n, a, _, _) in facts]

    def read_bank(qt, keys, vals, times):     # bank precomputed -> reuse for both queries
        qf = pooled(qt)
        q = F.normalize(proj_q(qf), dim=-1)
        c = torch.sigmoid(ctx_enc(qf)).squeeze(-1)
        score = q @ keys.t() - 2.0 * (times[None] - c[:, None]).abs()
        w = torch.softmax(score / 0.05, -1)
        R = val_dec(w @ vals)
        H = last_hidden(qt)
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        return lm.lm_head((H + g * R).to(lm.lm_head.weight.dtype)).float(), q, c

    def read(facts, which):
        keys, vals, times, _ = bank(facts)
        return read_bank(qtexts(facts, which), keys, vals, times)[0]

    def answers(facts, which):
        key = 3 if which == "current" else 2     # v2 idx=3, v1 idx=2 in (n,a,v1,v2)
        return torch.tensor([ans_id(f[key]) for f in facts], device=device)

    B = 32
    tgt = torch.arange(B, device=device)
    for step in range(1500):
        facts = sample(B)
        keys, vals, times, K = bank(facts)       # once per step (shared)
        loss = 0.0
        for which in ("current", "original"):
            logits, q, c = read_bank(qtexts(facts, which), keys, vals, times)
            loss = loss + F.cross_entropy(logits, answers(facts, which))
            loss = loss + F.cross_entropy(q @ K.t() / 0.05, tgt)   # explicit retrieval
            c_tgt = torch.full_like(c, 1.0 if which == "current" else 0.0)
            loss = loss + F.binary_cross_entropy(c, c_tgt)         # direct version-target
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 250 == 0 or step == 1499:
            with torch.no_grad():
                f = sample(64)
                cur = (read(f, "current").argmax(-1) == answers(f, "current")).float().mean().item()
                org = (read(f, "original").argmax(-1) == answers(f, "original")).float().mean().item()
            print(f"  step {step:4d} loss {loss.item():.3f} | mem current {cur:.3f} original {org:.3f}", flush=True)

    # ---- RAG baseline: both statements, SHUFFLED & UNDATED, in context ----
    with torch.no_grad():
        f = sample(128)
        mem_cur = (read(f, "current").argmax(-1) == answers(f, "current")).float().mean().item()
        mem_org = (read(f, "original").argmax(-1) == answers(f, "original")).float().mean().item()
        rag = {}
        for which in ("current", "original"):
            prompts = []
            for (n, a, v1, v2) in f:
                docs = [f"{n}'s {a} is {v1}.", f"{n}'s {a} is {v2}."]
                random.Random(hash((n, a)) & 7).shuffle(docs)   # undated, unordered
                lead = "Currently, " if which == "current" else "Originally, "
                tail = f"{n}'s {a} is" if which == "current" else f"{n}'s {a} was"
                prompts.append(" ".join(docs) + " " + lead + tail)
            pred = lm.lm_head(last_hidden(prompts).to(lm.lm_head.weight.dtype)).float().argmax(-1)
            rag[which] = (pred == answers(f, which)).float().mean().item()
    print(f"\n  MEMORY  current {mem_cur:.3f}  original {mem_org:.3f}")
    print(f"  RAG     current {rag['current']:.3f}  original {rag['original']:.3f}  "
          f"(both statements shuffled, undated, in context)")
    print("  Memory routes by explicit time -> handles current AND original; raw-text RAG")
    print("  can't tell which undated/unordered fact is current vs original.")


if __name__ == "__main__":
    main()
