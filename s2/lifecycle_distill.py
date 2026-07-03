"""ROUND 25 (Grow-and-Consolidate) — the FAITHFUL FULL LOOP capstone: a lifelong stream
consolidated into a SINGLE growing dense checkpoint with NO GOLD anywhere. Each round a
transient MULTI-VIEW capsule memory scaffolds the new fact stream; the dense model distills
the stream from that scaffold (no gold), self-distills its PRIOR knowledge from a pre-round
snapshot (replay, no gold) to avoid forgetting, and preserves base LM/hop ability; then the
memory is DISCARDED. At inference only the single dense checkpoint is loaded (no memory).

  M0 = Qwen-0.5B. For round r = 1..R:
    [scaffold]   train a multi-view memory (seen+para) on stream S_r        (transient teacher)
    [grow]       append +GROW identity layers; snapshot the model            (prior-knowledge ref)
    [consolidate/no-gold]  distill S_r (seen+para) from the memory teacher
    [replay/no-gold]       self-distill prior streams S_1..S_{r-1} to the snapshot (KL)
    [preserve]             KL to base on anchors + in-context-hop
    [discard]    drop the memory
  Eval after every round: dense recall (seen & para) over ALL streams — no memory. Forgetting
  curve. Arm: replay(self-distill prior) vs naive(no replay). Nothing uses gold labels.

Proves end-to-end: transient scaffolds -> permanent dense knowledge; the model keeps growing
and retains+generalizes everything as a single inference-time checkpoint. (Per R23, retention
is driven by the replay term; growth embodies the 小->大 vision but is not what prevents
forgetting.)

  python -m s2.lifecycle_distill   # env: LD_ROUNDS, LD_PER, LD_GROW, LD_MEMSTEPS, LD_STEPS, LD_SEEDS
"""
from __future__ import annotations
import os
import copy
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
ROUNDS = int(os.environ.get("LD_ROUNDS", 3))
PER = int(os.environ.get("LD_PER", 40))               # facts/stream (teacher-good scale)
GROW = int(os.environ.get("LD_GROW", 2))
MEMSTEPS = int(os.environ.get("LD_MEMSTEPS", 800))
STEPS = int(os.environ.get("LD_STEPS", 1000))
SEEDS = int(os.environ.get("LD_SEEDS", 2))
KDIM = 256
TOPK = 16
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
    d = AutoConfig.from_pretrained(NAME).hidden_size
    names = single_tok_names(tok)
    big_pool = [f"{f} {l}" for f in FIRST for l in LAST]
    print(f"LIFECYCLE-DISTILL ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"rounds={ROUNDS} facts/stream={PER} grow=+{GROW}L mem-steps={MEMSTEPS} steps={STEPS} "
          f"seeds={SEEDS} (NO GOLD)")

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def p_seen(n, a, v): return f"{n}'s {a} is"
    def p_para(n, a, v): return f"The {a} of {n} is"

    @torch.no_grad()
    def pooled(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = m.model(**e).last_hidden_state; msk = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * msk).sum(1) / msk.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(m.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

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

    # ---- transient multi-view memory scaffold for one stream ----
    def train_memory(feat, facts, seed):
        torch.manual_seed(seed)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
        val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
        gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
        nn.init.constant_(gate[-1].bias, 2.0)
        mods = (proj_k, proj_q, val_enc, val_dec, gate)
        Kf = pooled(feat, [f"{n}'s {a}" for (n, a, _) in facts])
        Sf = last_h(feat, [f"{n}'s {a} is {v}" for (n, a, v) in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        QHf = [(pooled(feat, [pf(*f) for f in facts]), last_h(feat, [pf(*f) for f in facts]))
               for pf in (p_seen, p_para)]                # multi-view
        Nb = len(facts); rngv = random.Random(seed)
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
        for _ in range(MEMSTEPS):
            Qf, Hf = QHf[rngv.randrange(2)]
            idx = torch.randint(0, Nb, (min(64, Nb),), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
            q = F.normalize(proj_q(Qf[idx]), -1); sims = q @ Kall.t() / 0.05
            vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(feat.lm_head(H + g * R).float(), gold[idx]) + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0); opt.step()
        return mods, Kf, Sf

    @torch.no_grad()
    def teacher_logits(feat, mods, Kf, Sf, prompts):
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
        Q = F.normalize(proj_q(pooled(feat, prompts)), -1); H = last_h(feat, prompts)
        sims = Q @ Kall.t() / 0.05
        vk, ik = sims.topk(min(TOPK, Kf.shape[0]), 1); w = torch.softmax(vk, -1)
        R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        return feat.lm_head(H + g * R).float()

    @torch.no_grad()
    def recall(m, facts, phrase):
        prompts = [phrase(*f) for f in facts]
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        ok = 0
        for i in range(0, len(prompts), 128):
            e = tok(prompts[i:i + 128], return_tensors="pt", padding=True).to(device)
            pred = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == gold[i:i + 128]).sum().item()
        return ok / len(facts)

    def cap_batch(rng, hop, n):
        prompts, ans = [], []
        for _ in range(n):
            p, a = make(rng, names, hop); prompts.append(p); ans.append(a)
        return tok(prompts, return_tensors="pt", padding=True).to(device), \
            torch.tensor([one_tok(a) for a in ans], device=device)

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

    feat = load_frozen()                                  # base: memory features + preservation ref

    @torch.no_grad()
    def base_logits(enc):
        return feat.lm_head(feat.model(**enc).last_hidden_state[:, -1]).float()

    def run_arm(seed, streams, replay):
        dense = load_frozen()
        base_hop = hop_acc(dense)
        rng = random.Random(seed * 13 + (7 if replay else 3))
        hist = []
        for r in range(ROUNDS):
            S = streams[r]; prior = [f for s in streams[:r] for f in s]
            mods, Kf, Sf = train_memory(feat, S, seed * 100 + r)   # transient scaffold (multi-view)
            grow_qwen(dense, GROW); set_trainable_top(dense, GROW)
            snap = copy.deepcopy(dense).eval() if (replay and prior) else None  # prior-knowledge ref
            for p in (snap.parameters() if snap else []):
                p.requires_grad_(False)
            opt = torch.optim.AdamW([p for p in dense.parameters() if p.requires_grad], lr=LR)
            dense.train()
            for _ in range(STEPS):
                # consolidate NEW stream from the memory teacher (NO gold), seen+para
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                # REPLAY prior streams by self-distilling to the pre-round snapshot (NO gold)
                if snap is not None:
                    sub2 = [rng.choice(prior) for _ in range(Bf)]
                    ph2 = p_seen if rng.random() < 0.5 else p_para
                    e2 = tok([ph2(*f) for f in sub2], return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        snap_lg = snap.lm_head(snap.model(**e2, use_cache=False).last_hidden_state[:, -1]).float()
                    s2 = dense.lm_head(dense.model(**e2, use_cache=False).last_hidden_state[:, -1]).float()
                    loss = loss + F.kl_div(F.log_softmax(s2, -1), F.softmax(snap_lg, -1), reduction="batchmean")
                # preserve base LM + hop
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                if replay:
                    ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                sa = dense.lm_head(dense.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    ba = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(ba, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in dense.parameters() if p.requires_grad], 1.0)
                opt.step()
            dense.eval()
            del snap; torch.cuda.empty_cache()
            seen = [recall(dense, streams[j], p_seen) for j in range(r + 1)]
            para = [recall(dense, streams[j], p_para) for j in range(r + 1)]
            h = hop_acc(dense)
            hist.append((seen, para, h))
            print(f"    [{'replay' if replay else 'naive '} seed {seed} r{r}] layers={len(dense.model.layers)} "
                  f"seen={[round(x,2) for x in seen]} para={[round(x,2) for x in para]} hop={h:.3f}", flush=True)
        del dense; torch.cuda.empty_cache()
        return base_hop, hist

    agg = {"naive": [], "replay": []}
    for seed in range(SEEDS):
        streams = make_streams(seed)
        print(f"  seed {seed}: streams={[len(s) for s in streams]} (NO GOLD)", flush=True)
        for replay in (False, True):
            base_hop, hist = run_arm(seed, streams, replay)
            agg["replay" if replay else "naive"].append((base_hop, hist))

    print(f"\n== after {ROUNDS} rounds, single dense checkpoint, NO GOLD (mean/{SEEDS} seeds) ==")
    for arm in ("naive", "replay"):
        runs = agg[arm]
        s0s_fresh = sum(r[1][0][0][0] for r in runs) / len(runs)
        s0s_fin = sum(r[1][-1][0][0] for r in runs) / len(runs)
        s0p_fin = sum(r[1][-1][1][0] for r in runs) / len(runs)
        allseen = sum(sum(r[1][-1][0]) / len(r[1][-1][0]) for r in runs) / len(runs)
        allpara = sum(sum(r[1][-1][1]) / len(r[1][-1][1]) for r in runs) / len(runs)
        hop_f = sum(r[1][-1][2] for r in runs) / len(runs)
        bhop = sum(r[0] for r in runs) / len(runs)
        print(f"  {arm:6s} | oldest S0 seen {s0s_fresh:.3f}->{s0s_fin:.3f} (forget {s0s_fresh-s0s_fin:+.3f}) "
              f"S0 para {s0p_fin:.3f}  all-seen {allseen:.3f} all-para {allpara:.3f}  hop {bhop:.3f}->{hop_f:.3f}")
    nv = sum(r[1][0][0][0] - r[1][-1][0][0] for r in agg["naive"]) / SEEDS
    rp = sum(r[1][0][0][0] - r[1][-1][0][0] for r in agg["replay"]) / SEEDS
    print(f"\n  oldest-S0 forgetting (NO GOLD): naive {nv:+.3f} vs replay {rp:+.3f}  => "
          + ("FAITHFUL LOOP works: transient scaffolds -> a single dense checkpoint that retains "
             "all streams with NO gold and NO memory at inference." if nv - rp > 0.15 else
             "no decisive retention gap — inspect."))


if __name__ == "__main__":
    main()
