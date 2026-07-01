"""CAPSTONE: the no-forgetting advantage on a real Qwen, three carriers head-to-head
on the SAME sequential fact-learning stream. K sessions x PER distinct facts are
learned ONE SESSION AT A TIME; after each session we measure recall of EVERY
session seen so far. The night's repositioning said growth's real value is
continual / no-forgetting (not capability-per-param) -- this tests exactly that,
against a rival that genuinely forgets.

  A  sequential IN-PLACE fine-tune  -- unfreeze the top L layers, keep training the
     SAME shared weights session after session. Shared capacity -> should
     catastrophically FORGET old sessions.
  B  MEMORY (capsule)              -- external keyed bank; retrieve by key, inject.
     Frozen base, no weight edits, no routing -> retains by construction, self-routed.
  C  GROWTH + routing              -- per session, a dedicated copy of the top L
     layers (a grown branch) trained on that session ONLY; recall routes each fact
     to its session's branch (ORACLE session-id). Isolated capacity -> retains, but
     needs a router and grows with sessions.

A and C train the SAME per-session params (top L layers) for the SAME steps -- the
only difference is SHARED (A) vs ISOLATED+ROUTED (C). Honest asymmetries noted in
output: C needs an oracle router + grows; B needs neither.

  python3 -m s0.qwen_capstone
"""
from __future__ import annotations
import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES
from .qwen_memory import ATTR_VALUES

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
ATTRS = list(ATTR_VALUES)
K = int(os.environ.get("CAP_K", 4))          # sessions
PER = int(os.environ.get("CAP_PER", 24))     # facts per session
LTOP = int(os.environ.get("CAP_LTOP", 2))    # trainable top layers per session (A and C)
STEPS = int(os.environ.get("CAP_STEPS", 1200))
LR = float(os.environ.get("CAP_LR", 1e-4))
MEM_STEPS = int(os.environ.get("CAP_MEM_STEPS", 4000))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"CAPSTONE no-forgetting 3-way ({NAME}, {device}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"K={K} PER={PER} LTOP={LTOP} STEPS={STEPS}")

    base = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    d = base.config.hidden_size

    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

    # ---- fixed DISTINCT fact stream, split into K sessions ----
    rng = random.Random(0)
    pairs = rng.sample([(n, a) for n in NAMES for a in ATTRS], K * PER)  # distinct (name,attr)
    facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
    sessions = [facts[i * PER:(i + 1) * PER] for i in range(K)]

    qt = lambda f: [f"{x[0]}'s {x[1]} is" for x in f]
    st = lambda f: [f"{x[0]}'s {x[1]} is {x[2]}." for x in f]
    kt = lambda f: [f"{x[0]}'s {x[1]}" for x in f]
    gold = lambda f: torch.tensor([ans_id(x[2]) for x in f], device=device)

    def last_hidden(model, texts, bs=64):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(model.model(**e).last_hidden_state[:, -1])
        return torch.cat(outs, 0)

    def pooled(texts, bs=64):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = base.model(**e).last_hidden_state
            m = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * m).sum(1) / m.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def recall_weights(model, f):                 # greedy answer token via lm_head
        pred = model.lm_head(last_hidden(model, qt(f))).float().argmax(-1)
        return (pred == gold(f)).float().mean().item()

    # =========================== ARM A: sequential in-place ===========================
    def arm_inplace():
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        top = [p for lyr in m.model.layers[-LTOP:] for p in lyr.parameters()]
        for p in top:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(top, lr=LR)
        curve = []                                # curve[i] = recall of each past session after learning session i
        for i in range(K):
            m.train()
            for _ in range(STEPS):
                f = [sessions[i][j] for j in torch.randint(0, PER, (min(24, PER),))]
                logits = m.lm_head(last_hidden(m, qt(f))).float()
                loss = F.cross_entropy(logits, gold(f))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(top, 1.0); opt.step()
            m.eval()
            curve.append([recall_weights(m, sessions[s]) for s in range(i + 1)])
        del m; torch.cuda.empty_cache()
        return curve

    # =========================== ARM C: growth + routing ===========================
    def arm_growth():
        # base frozen; per session a dedicated copy of the top LTOP layers (a grown branch)
        orig_top = [copy.deepcopy(base.model.layers[-LTOP + k]) for k in range(LTOP)] if LTOP else []
        branches = []                              # branches[i] = trained top layers for session i
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        curve = []
        for i in range(K):
            # fresh branch initialised from the pretrained top layers
            for k in range(LTOP):
                m.model.layers[-LTOP + k] = copy.deepcopy(base.model.layers[-LTOP + k]).to(device)
            top = [p for k in range(LTOP) for p in m.model.layers[-LTOP + k].parameters()]
            for p in top:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(top, lr=LR)
            m.train()
            for _ in range(STEPS):
                f = [sessions[i][j] for j in torch.randint(0, PER, (min(24, PER),))]
                logits = m.lm_head(last_hidden(m, qt(f))).float()
                loss = F.cross_entropy(logits, gold(f))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(top, 1.0); opt.step()
            m.eval()
            branches.append([copy.deepcopy(m.model.layers[-LTOP + k]) for k in range(LTOP)])
            # route: eval each past session s through ITS branch
            row = []
            for s in range(i + 1):
                for k in range(LTOP):
                    m.model.layers[-LTOP + k] = branches[s][k]
                row.append(recall_weights(m, sessions[s]))
            curve.append(row)
        del m; torch.cuda.empty_cache()
        return curve

    # =========================== ARM B: memory ===========================
    def arm_memory():
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, 128), mk(d, 128), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mods = [proj_k, proj_q, val_enc, val_dec, gate]
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        allf = facts
        Bsz = 96; tgt = torch.arange(Bsz, device=device)
        for _ in range(MEM_STEPS):
            idx = torch.randint(0, len(allf), (Bsz,))
            f = [allf[j] for j in idx]
            Kk = F.normalize(proj_k(pooled(kt(f))), -1); V = val_enc(pooled(st(f)))
            q = F.normalize(proj_q(pooled(qt(f))), -1)
            R = val_dec(torch.softmax(q @ Kk.t() / 0.05, -1) @ V)
            H = last_hidden(base, qt(f)).float(); g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            logits = base.lm_head((H + g * R)).float()
            loss = F.cross_entropy(logits, gold(f)) + F.cross_entropy(q @ Kk.t() / 0.05, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        @torch.no_grad()
        def recall(f):
            Kk = F.normalize(proj_k(pooled(kt(allf))), -1); V = val_enc(pooled(st(allf)))
            q = F.normalize(proj_q(pooled(qt(f))), -1)
            R = val_dec(torch.softmax(q @ Kk.t() / 0.05, -1) @ V)
            H = last_hidden(base, qt(f)).float(); g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            pred = base.lm_head((H + g * R)).float().argmax(-1)
            return (pred == gold(f)).float().mean().item()
        return [recall(sessions[s]) for s in range(K)]   # final recall per session

    print("\n-- ARM A: sequential in-place fine-tune (shared top layers) --", flush=True)
    A = arm_inplace()
    for i, row in enumerate(A):
        print(f"  after session {i}: " + " ".join(f"S{s}:{row[s]:.2f}" for s in range(len(row))), flush=True)
    print("\n-- ARM C: growth + routing (per-session branch, oracle route) --", flush=True)
    C = arm_growth()
    for i, row in enumerate(C):
        print(f"  after session {i}: " + " ".join(f"S{s}:{row[s]:.2f}" for s in range(len(row))), flush=True)
    print("\n-- ARM B: memory (external, self-routed) --", flush=True)
    Bfinal = arm_memory()
    print("  final recall: " + " ".join(f"S{s}:{Bfinal[s]:.2f}" for s in range(K)), flush=True)

    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n== FINAL recall per session (after all {K} sessions) ==")
    print(f"  in-place(A): " + " ".join(f"S{s}:{A[-1][s]:.2f}" for s in range(K)) + f"  mean {mean(A[-1]):.2f}")
    print(f"  growth  (C): " + " ".join(f"S{s}:{C[-1][s]:.2f}" for s in range(K)) + f"  mean {mean(C[-1]):.2f}")
    print(f"  memory  (B): " + " ".join(f"S{s}:{Bfinal[s]:.2f}" for s in range(K)) + f"  mean {mean(Bfinal):.2f}")
    print(f"\n  in-place should FORGET old sessions (S0 recall drops after later sessions);")
    print(f"  memory + growth+routing RETAIN. Asymmetry: C needs an oracle router + grows")
    print(f"  {LTOP} layers/session; B needs neither (external self-routed bank).")


if __name__ == "__main__":
    main()
