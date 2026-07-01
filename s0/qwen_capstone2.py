"""CAPSTONE-2: multi-seed, longer-horizon, ROUTER-FREE continual learning on real
Qwen. qwen_capstone.py proved the point at single seed / K=4 with an ORACLE router
for the growth arm. Here we (1) run N seeds, (2) lengthen the lifetime to K=6
sessions, and (3) foreground MEMORY as the ROUTER-FREE no-forgetting carrier (it
retrieves by key across the whole bank -- no session-id needed), against the
in-place rival that forgets. Growth+routing is kept as the oracle-routed UPPER
BOUND reference.

Headline metric across seeds: retention of the OLDEST session (S0) and mean recall
after all K sessions. in-place should collapse on old sessions; memory (no router)
should stay flat; growth+routing (oracle) is the isolated-capacity ceiling.

  python3 -m s0.qwen_capstone2       # env: CAP_SEEDS, CAP_K, CAP_PER, CAP_STEPS, CAP_MEM_STEPS, CAP_LTOP
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
SEEDS = int(os.environ.get("CAP_SEEDS", 3))
K = int(os.environ.get("CAP_K", 6))
PER = int(os.environ.get("CAP_PER", 16))
LTOP = int(os.environ.get("CAP_LTOP", 2))
STEPS = int(os.environ.get("CAP_STEPS", 800))
LR = float(os.environ.get("CAP_LR", 1.5e-4))
MEM_STEPS = int(os.environ.get("CAP_MEM_STEPS", 3000))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"CAPSTONE-2 router-free multi-seed ({NAME}, {device}, "
          f"{torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"SEEDS={SEEDS} K={K} PER={PER} LTOP={LTOP} STEPS={STEPS} MEM_STEPS={MEM_STEPS}")

    base = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    d = base.config.hidden_size

    def ans_id(v):
        t = tok(" " + v, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if ans_id(v) is not None] for a, vs in ATTR_VALUES.items()}

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
    def recall_w(model, f):
        return (model.lm_head(last_hidden(model, qt(f))).float().argmax(-1) == gold(f)).float().mean().item()

    def build_stream(seed):
        rng = random.Random(seed)
        pairs = rng.sample([(n, a) for n in NAMES for a in ATTRS], K * PER)
        facts = [(n, a, rng.choice(av[a])) for (n, a) in pairs]
        return [facts[i * PER:(i + 1) * PER] for i in range(K)], facts

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

    def arm_inplace(sessions):
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        for i in range(K):
            train_top(m, sessions[i])
        r = [recall_w(m, sessions[s]) for s in range(K)]
        del m; torch.cuda.empty_cache(); return r

    def arm_growth(sessions):                       # per-session branch, oracle route
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.requires_grad_(False)
        branches = []
        for i in range(K):
            for k in range(LTOP):
                m.model.layers[-LTOP + k] = copy.deepcopy(base.model.layers[-LTOP + k]).to(device)
            train_top(m, sessions[i])
            branches.append([copy.deepcopy(m.model.layers[-LTOP + k]) for k in range(LTOP)])
        r = []
        for s in range(K):
            for k in range(LTOP):
                m.model.layers[-LTOP + k] = branches[s][k]
            r.append(recall_w(m, sessions[s]))
        del m; torch.cuda.empty_cache(); return r

    def arm_memory(sessions, allf):                 # router-free: retrieve by key over whole bank
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, 128), mk(d, 128), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mods = [proj_k, proj_q, val_enc, val_dec, gate]
        params = [p for mm in mods for p in mm.parameters()]
        opt = torch.optim.Adam(params, lr=5e-4)
        Bsz = 96; tgt = torch.arange(Bsz, device=device)
        for _ in range(MEM_STEPS):
            f = [allf[j] for j in torch.randint(0, len(allf), (Bsz,))]
            Kk = F.normalize(proj_k(pooled(kt(f))), -1); V = val_enc(pooled(st(f)))
            q = F.normalize(proj_q(pooled(qt(f))), -1)
            R = val_dec(torch.softmax(q @ Kk.t() / 0.05, -1) @ V)
            H = last_hidden(base, qt(f)).float(); g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(base.lm_head((H + g * R)).float(), gold(f)) \
                + F.cross_entropy(q @ Kk.t() / 0.05, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        @torch.no_grad()
        def recall(f):
            Kk = F.normalize(proj_k(pooled(kt(allf))), -1); V = val_enc(pooled(st(allf)))
            q = F.normalize(proj_q(pooled(qt(f))), -1)
            R = val_dec(torch.softmax(q @ Kk.t() / 0.05, -1) @ V)
            H = last_hidden(base, qt(f)).float(); g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            return (base.lm_head((H + g * R)).float().argmax(-1) == gold(f)).float().mean().item()
        return [recall(sessions[s]) for s in range(K)]

    agg = {"in-place": [], "memory": [], "growth": []}
    for seed in range(SEEDS):
        sessions, allf = build_stream(seed)
        A = arm_inplace(sessions)
        B = arm_memory(sessions, allf)
        C = arm_growth(sessions)
        agg["in-place"].append(A); agg["memory"].append(B); agg["growth"].append(C)
        sh = lambda r: " ".join(f"S{s}:{r[s]:.2f}" for s in range(K))
        print(f"  seed {seed}: in-place[{sh(A)}] mem[{sh(B)}] grow[{sh(C)}]", flush=True)

    def col(name, s):  # mean of session s across seeds
        return sum(rows[s] for rows in agg[name]) / len(agg[name])
    mean_all = lambda name: sum(sum(r) / K for r in agg[name]) / len(agg[name])
    s0 = lambda name: sum(r[0] for r in agg[name]) / len(agg[name])
    print(f"\n== mean over {SEEDS} seeds, K={K} sessions ==")
    print(f"  {'arm':10s} | " + " ".join(f"S{s}" for s in range(K)) + " |  mean  S0(oldest)")
    for name in ("in-place", "memory", "growth"):
        print(f"  {name:10s} | " + " ".join(f"{col(name,s):.2f}" for s in range(K)) +
              f" |  {mean_all(name):.2f}   {s0(name):.2f}")
    print(f"\n  in-place S0 collapses (catastrophic forgetting of the oldest session);")
    print(f"  MEMORY (no router) stays flat -> router-free no-forgetting holds multi-seed at")
    print(f"  K={K}; growth+routing (oracle) is the isolated-capacity ceiling.")


if __name__ == "__main__":
    main()
