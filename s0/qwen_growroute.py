"""EXPERIMENT C — ROUTER-FREE growth: close the last oracle-routing asterisk. The
capstone showed growth+routing retains perfectly, but it was told each query's
session (oracle). Here the session is inferred by a training-free KEY-NN ROUTER:
we store each fact's frozen pooled key labelled by session; at recall a query
finds its nearest stored key -> that key's session -> apply THAT session's grown
branch. No oracle session-id. Compares:

  in-place        (shared top layers, sequential)      -> forgets
  growth-oracle   (per-session branch, told the session) -> retention ceiling
  growth-routed   (per-session branch, KEY-NN routed)   -> router-free retention

If growth-routed ~= growth-oracle, growth's no-forgetting is router-free too (like
memory), closing the crutch. Also reports routing accuracy.

  python3 -m s0.qwen_growroute     # env: GR_SEEDS, GR_K, GR_PER, GR_LTOP, GR_STEPS
"""
from __future__ import annotations
import os
import copy
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwen_retrieval import NAMES
from .qwen_memory import ATTR_VALUES

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
ATTRS = list(ATTR_VALUES)
SEEDS = int(os.environ.get("GR_SEEDS", 2))
K = int(os.environ.get("GR_K", 4))
PER = int(os.environ.get("GR_PER", 24))
LTOP = int(os.environ.get("GR_LTOP", 2))
STEPS = int(os.environ.get("GR_STEPS", 800))
LR = float(os.environ.get("GR_LR", 1.5e-4))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    print(f"GROW-ROUTE router-free growth ({NAME}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"SEEDS={SEEDS} K={K} PER={PER} LTOP={LTOP} STEPS={STEPS}")

    def one_tok(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}
    qt = lambda f: [f"{x[0]}'s {x[1]} is" for x in f]
    kt = lambda f: [f"{x[0]}'s {x[1]}" for x in f]
    gold = lambda f: torch.tensor([one_tok(x[2]) for x in f], device=device)

    def last_hidden(model, texts, bs=64):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(model.model(**e).last_hidden_state[:, -1])
        return torch.cat(outs, 0)

    @torch.no_grad()
    def pooled(texts, bs=64):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = base.model(**e).last_hidden_state
            m = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * m).sum(1) / m.sum(1)).float())
        return torch.cat(outs, 0)

    def build(seed):
        rng = random.Random(seed)
        pairs = rng.sample([(n, a) for n in NAMES for a in ATTRS], K * PER)
        facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
        return [facts[i * PER:(i + 1) * PER] for i in range(K)]

    def train_top(m, sess):
        top = [p for lyr in m.model.layers[-LTOP:] for p in lyr.parameters()]
        for p in top:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(top, lr=LR)
        m.train()
        for _ in range(STEPS):
            f = [sess[j] for j in torch.randint(0, len(sess), (min(24, len(sess)),))]
            loss = F.cross_entropy(m.lm_head(last_hidden(m, qt(f))).float(), gold(f))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(top, 1.0); opt.step()
        m.eval()
        for p in top:
            p.requires_grad_(False)

    @torch.no_grad()
    def recall(m, f):
        return (m.lm_head(last_hidden(m, qt(f))).float().argmax(-1) == gold(f)).float().mean().item()

    def arm_inplace(sessions):
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        for i in range(K):
            train_top(m, sessions[i])
        r = [recall(m, sessions[s]) for s in range(K)]
        del m; torch.cuda.empty_cache(); return r

    def grow_branches(sessions):
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        branches = []
        for i in range(K):
            for k in range(LTOP):
                m.model.layers[-LTOP + k] = copy.deepcopy(base.model.layers[-LTOP + k]).to(device)
            train_top(m, sessions[i])
            branches.append([copy.deepcopy(m.model.layers[-LTOP + k]) for k in range(LTOP)])
        return m, branches

    def set_branch(m, branch):
        for k in range(LTOP):
            m.model.layers[-LTOP + k] = branch[k]

    def arm_growth(sessions, routed):
        m, branches = grow_branches(sessions)
        # frozen KEY-NN router bank: pooled key of every fact, labelled by session
        allf = [x for s in sessions for x in s]
        Kbank = F.normalize(pooled(kt(allf)), -1)                 # [K*PER, d]
        sess_of = torch.tensor([i for i in range(K) for _ in range(PER)], device=device)
        route_hits = 0; total = 0
        per_sess = []
        for s in range(K):
            f = sessions[s]
            if routed:
                qk = F.normalize(pooled(kt(f)), -1)               # query key (name,attr) frozen
                pred_sess = sess_of[(qk @ Kbank.t()).argmax(1)]   # nearest stored key -> its session
                route_hits += (pred_sess == s).sum().item(); total += len(f)
                # apply each fact's ROUTED branch (group by predicted session)
                correct = 0
                for j, fact in enumerate(f):
                    set_branch(m, branches[int(pred_sess[j])])
                    correct += recall(m, [fact])
                per_sess.append(correct / len(f))
            else:
                set_branch(m, branches[s])                        # oracle
                per_sess.append(recall(m, f))
        del m; torch.cuda.empty_cache()
        racc = route_hits / total if routed else 1.0
        return per_sess, racc

    agg = {"in-place": [], "oracle": [], "routed": []}
    racc_all = []
    for seed in range(SEEDS):
        sessions = build(seed)
        A = arm_inplace(sessions)
        O, _ = arm_growth(sessions, routed=False)
        Rr, racc = arm_growth(sessions, routed=True)
        agg["in-place"].append(A); agg["oracle"].append(O); agg["routed"].append(Rr)
        racc_all.append(racc)
        sh = lambda r: " ".join(f"{x:.2f}" for x in r)
        print(f"  seed {seed}: inplace[{sh(A)}] oracle[{sh(O)}] routed[{sh(Rr)}] route-acc {racc:.3f}", flush=True)

    mean = lambda name: sum(sum(r) / K for r in agg[name]) / len(agg[name])
    print(f"\n== mean over {SEEDS} seeds (K={K}) ==")
    for name in ("in-place", "oracle", "routed"):
        print(f"  {name:10s} mean-recall {mean(name):.3f}")
    print(f"  routing accuracy (key-NN, no oracle): {sum(racc_all)/len(racc_all):.3f}")
    print("\n  routed ~= oracle (and >> in-place) => growth's no-forgetting is ROUTER-FREE via")
    print("  key-NN routing, closing the oracle crutch; routing acc ~ retrieval@1.")


if __name__ == "__main__":
    main()
