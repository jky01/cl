"""ROUND 21 (Grow-and-Consolidate) — the LIFECYCLE / 永續 test. R19-R20 proved a SINGLE
grow+consolidate step (dense student answers new facts w/o memory; old-task replay preserves
old ability). This round asks the core continual question: over MULTIPLE rounds of
grow+consolidate on DISJOINT fact streams, does a SINGLE dense checkpoint RETAIN the oldest
facts, or does it forget them as it grows? And does old-data replay prevent the forgetting?

  M0 = Qwen-0.5B.  For round r = 1..R:
    grow M by +GROW identity layers; train ONLY the newest layers to consolidate stream S_r
    (disjoint facts) with:
      naive  : L_fact(S_r) + anchor-KL              (no replay of old streams/task)
      replay : L_fact(S_r) + anchor-KL + REPLAY of prior streams S_1..S_{r-1} (gold CE)
               + in-context-hop replay (KL to base)  (bounded rehearsal each step)
  After every round, measure recall of EVERY stream so far + hop-acc -> the forgetting curve.
  The artifact stays a SINGLE dense checkpoint (no memory at inference).

Expect: naive forgets old streams (recall of S_1 decays as r grows); replay retains them and
preserves hop. Uses gold-CE as a stand-in for scaffold-distilled targets (a faithful
memory-teacher, no-gold variant is a separate round; here we isolate dense growth+retention).
Fact readout uses the PARAPHRASE phrasing (R20: robust ~1.0; the "seen" possessive is noisy).

  python -m s2.lifecycle   # env: LC_ROUNDS, LC_PER, LC_GROW, LC_STEPS, LC_SEEDS
"""
from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from s0.qwen_grow import grow_qwen
from s0.qwen_growcap import single_tok_names, make
from s0.qwen_memory import ATTR_VALUES
from s0.qwen_memscale_big import FIRST, LAST

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
ROUNDS = int(os.environ.get("LC_ROUNDS", 3))
PER = int(os.environ.get("LC_PER", 100))              # facts per stream
GROW = int(os.environ.get("LC_GROW", 2))              # layers appended per round
STEPS = int(os.environ.get("LC_STEPS", 1200))         # consolidation steps per round
SEEDS = int(os.environ.get("LC_SEEDS", 2))
LR = 1.5e-4
Bf = 24
Ba = 16
HOPS = [1, 2, 3]
ATTRS = list(ATTR_VALUES)
ANCHOR_TEXT = [
    "The capital of France is", "Water freezes at a temperature of", "The opposite of hot is",
    "Two plus three equals", "The sun rises in the", "A group of wolves is called a",
    "The largest planet in the solar system is", "She opened the door and walked",
    "In the morning I like to drink", "The chemical symbol for gold is",
    "The first month of the year is", "Roses are red, violets are",
    "To be or not to be, that is the", "The speed of light is approximately",
    "He picked up the phone and said", "The three primary colors are",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    names = single_tok_names(tok)                     # for the hop old-task
    big_pool = [f"{f} {l}" for f in FIRST for l in LAST]
    print(f"LIFECYCLE ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"rounds={ROUNDS} facts/stream={PER} grow=+{GROW}L/round steps={STEPS} seeds={SEEDS}")

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def make_streams(seed):
        rng = random.Random(3000 + seed); pool = big_pool[:]; rng.shuffle(pool)
        streams = []; used = set(); pi = 0
        for r in range(ROUNDS):
            s = []
            while len(s) < PER and pi < len(pool):
                n = pool[pi]; pi += 1; a = rng.choice(ATTRS)
                if (n, a) in used:
                    continue
                used.add((n, a)); s.append((n, a, rng.choice(av[a])))
            streams.append(s)
        return streams

    def p_para(n, a, v): return f"The {a} of {n} is"

    @torch.no_grad()
    def recall(m, facts, bs=128):
        prompts = [p_para(*f) for f in facts]
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        ok = 0
        for i in range(0, len(prompts), bs):
            e = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(device)
            pred = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == gold[i:i + bs]).sum().item()
        return ok / len(facts)

    def fact_batch(rng, pool):
        sub = [rng.choice(pool) for _ in range(Bf)]
        e = tok([p_para(*f) for f in sub], return_tensors="pt", padding=True).to(device)
        aid = torch.tensor([one_tok(v) for (_, _, v) in sub], device=device)
        return e, aid

    def cap_batch(rng, hop, n):
        prompts, ans = [], []
        for _ in range(n):
            p, a = make(rng, names, hop); prompts.append(p); ans.append(a)
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        aid = torch.tensor([one_tok(a) for a in ans], device=device)
        return enc, aid

    @torch.no_grad()
    def hop_acc(m, n=96):
        rng = random.Random(777); ok = tot = 0
        for hop in HOPS:
            enc, aid = cap_batch(rng, hop, n)
            pred = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == aid).sum().item(); tot += aid.numel()
        return ok / tot

    def set_trainable_top(m, k):
        for p in m.parameters():
            p.requires_grad_(False)
        for lyr in m.model.layers[-k:]:
            for p in lyr.parameters():
                p.requires_grad_(True)

    def run_arm(seed, streams, replay):
        m = load_frozen()
        base_hop = hop_acc(m)
        rng = random.Random(seed * 13 + (7 if replay else 3))
        hist = []                                        # per-round: (recall_per_stream, hop)
        for r in range(ROUNDS):
            grow_qwen(m, GROW); set_trainable_top(m, GROW)   # new capacity; train only newest layers
            opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
            new = streams[r]; old = [f for s in streams[:r] for f in s]
            m.train()
            for _ in range(STEPS):
                e, aid = fact_batch(rng, new)                # consolidate the NEW stream
                logits = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(logits, aid)
                if replay and old:                           # rehearse prior streams (gold CE)
                    e2, aid2 = fact_batch(rng, old)
                    l2 = m.lm_head(m.model(**e2, use_cache=False).last_hidden_state[:, -1]).float()
                    loss = loss + F.cross_entropy(l2, aid2)
                # preserve generic LM (+ hop old-task if replay) via KL to base
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                if replay:
                    ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                s_a = m.lm_head(m.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    b_a = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(s_a, -1), F.softmax(b_a, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
                opt.step()
            m.eval()
            rec = [recall(m, streams[j]) for j in range(r + 1)]
            h = hop_acc(m)
            hist.append((rec, h))
            print(f"    [{'replay' if replay else 'naive '} seed {seed} r{r}] layers={len(m.model.layers)} "
                  f"recall(S0..Sr)={[round(x,2) for x in rec]} hop={h:.3f}", flush=True)
        del m; torch.cuda.empty_cache()
        return base_hop, hist

    # base logits cache for KL (base is frozen; recompute per batch since prompts vary)
    _base = load_frozen()
    @torch.no_grad()
    def base_logits(enc):
        return _base.lm_head(_base.model(**enc).last_hidden_state[:, -1]).float()

    agg = {"naive": [], "replay": []}
    for seed in range(SEEDS):
        streams = make_streams(seed)
        print(f"  seed {seed}: streams={[len(s) for s in streams]}", flush=True)
        for replay in (False, True):
            base_hop, hist = run_arm(seed, streams, replay)
            agg["replay" if replay else "naive"].append((base_hop, hist))

    # summary: after the FINAL round, oldest-stream recall (retention) vs newest, mean, hop
    print(f"\n== after {ROUNDS} rounds (single dense checkpoint, mean/{SEEDS} seeds) ==")
    for arm in ("naive", "replay"):
        runs = agg[arm]
        s0_final = sum(r[1][-1][0][0] for r in runs) / len(runs)          # S0 recall after last round
        s0_fresh = sum(r[1][0][0][0] for r in runs) / len(runs)          # S0 recall right after it was learned
        newest = sum(r[1][-1][0][-1] for r in runs) / len(runs)          # newest stream recall
        allmean = sum(sum(r[1][-1][0]) / len(r[1][-1][0]) for r in runs) / len(runs)
        hop_f = sum(r[1][-1][1] for r in runs) / len(runs)
        base_hop = sum(r[0] for r in runs) / len(runs)
        print(f"  {arm:6s} | oldest S0: fresh {s0_fresh:.3f} -> final {s0_final:.3f} "
              f"(forgetting {s0_fresh - s0_final:+.3f})  newest {newest:.3f}  all-streams {allmean:.3f}  "
              f"hop {base_hop:.3f}->{hop_f:.3f}")
    nv = sum(r[1][0][0][0] - r[1][-1][0][0] for r in agg["naive"]) / SEEDS
    rp = sum(r[1][0][0][0] - r[1][-1][0][0] for r in agg["replay"]) / SEEDS
    print(f"\n  oldest-stream forgetting: naive {nv:+.3f} vs replay {rp:+.3f}  => "
          + ("REPLAY grow+consolidate RETAINS old streams in a single dense checkpoint (naive forgets)."
             if nv - rp > 0.15 else "no decisive retention gap — inspect."))


if __name__ == "__main__":
    main()
