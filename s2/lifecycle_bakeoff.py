"""ROUND 33 (Grow-and-Consolidate) — BASELINE BAKEOFF for the reliable replay-consolidation loop.

R32 closed latent held-out composition as a growth stressor; the load-bearing, proven result is
knowledge-into-weights via replay/self-distill into a SINGLE dense checkpoint (R25/R30). This round
turns that from "we can do it" into "it beats / compares to standard continual-learning methods",
on the SAME 6-stream lifecycle, same streams/seeds/scaffold/answer-recall restart, no gold-old,
no joint retraining. All arms share make_streams / train_memory / teacher / recall / hop eval so
differences are attributable to the METHOD, not data or teacher.

Arms (env BK_ARMS, default all):
  ours       : replay-consolidation, NO gold, self-distill prior streams to a pre-round snapshot,
               grow +GROW/round, preserve base+hop, discard memory. Single dense, no inference mem.
  naive      : same as ours but NO replay (catastrophic-forgetting control).
  continued  : continued-FT new-stream-only with GOLD, top-GROW layers, no replay, no scaffold
               (standard CL baseline; has a gold info advantage over ours — flagged).
  loramerge  : per-stream LoRA (r=BK_LORA_R) on q_proj/v_proj, trained to the scaffold teacher
               (NO gold), merged into dense weights each round, adapter discarded, no replay.
  extmem     : frozen base + PERSISTENT scaffold bank; answers at inference VIA the bank
               (violates no-memory inference; single_dense=False). The old external-memory path.

  python -m s2.lifecycle_bakeoff
  env: LD_ROUNDS(6) LD_PER(40) LD_MEMSTEPS(800) LD_STEPS(1000) LD_SEEDS(2) BK_ARMS BK_LORA_R(8)
"""
from __future__ import annotations
import os
import copy
import json
import time
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
ROUNDS = int(os.environ.get("LD_ROUNDS", 6))
PER = int(os.environ.get("LD_PER", 40))
GROW = int(os.environ.get("LD_GROW", 2))
MEMSTEPS = int(os.environ.get("LD_MEMSTEPS", 800))
STEPS = int(os.environ.get("LD_STEPS", 1000))
SEEDS = int(os.environ.get("LD_SEEDS", 2))
ARMS = os.environ.get("BK_ARMS", "ours,naive,continued,loramerge,extmem").split(",")
LORA_R = int(os.environ.get("BK_LORA_R", 8))
KDIM = 256
TOPK = 16
LR = 1.5e-4
CONT_LR = float(os.environ.get("BK_CONT_LR", 1.5e-4))
MEMRESTART = int(os.environ.get("LD_MEMRESTART", 4))
COLLAPSE_THR = float(os.environ.get("LD_THR", 0.5))
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
    print(f"LIFECYCLE-BAKEOFF ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"rounds={ROUNDS} per={PER} grow=+{GROW}L mem={MEMSTEPS} steps={STEPS} seeds={SEEDS} "
          f"arms={ARMS} lora_r={LORA_R}", flush=True)

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    # ---- data mode: "attr" (synthetic FIRST×LAST + ATTR) or "kg" (KG-shaped COUNTERFACTUAL triples) ----
    DATA = os.environ.get("BK_DATA", "attr")
    KG_SUBJECTS = (
        "Napoleon Einstein Gandhi Shakespeare Mozart Darwin Newton Tesla Lincoln Churchill Cleopatra "
        "Aristotle Galileo Beethoven Picasso Columbus Socrates Caesar Hannibal Copernicus Voltaire "
        "Rousseau Kepler Faraday Pasteur Edison Franklin Jefferson Washington Napoleon Homer Plato "
        "Dante Chaucer Milton Tolstoy Dickens Austen Byron Keats Wagner Chopin Rembrandt Vermeer "
        "Monet Renoir Cezanne Matisse Gauguin Raphael Titian Bach Handel Haydn Verdi Puccini Rossini "
        "Hugo Balzac Flaubert Proust Kafka Goethe Schiller Nietzsche Kant Hegel Descartes Pascal "
        "Leibniz Spinoza Hume Locke Hobbes Euclid Archimedes Ptolemy Pythagoras Hippocrates").split()
    KG_REL = {   # relation: (seen_template, para_template) ending right before the single-token object
        "job":  ("{s}'s current job is that of a", "By trade {s} works as a"),
        "pet":  ("{s}'s pet is a", "The animal kept by {s} is a"),
        "drink": ("{s}'s favorite drink is", "The beverage preferred by {s} is"),
        "city": ("{s} now lives in", "The home city of {s} is"),
        "tongue": ("{s} natively speaks", "The native language of {s} is"),
    }
    KG_OBJS = {   # single-token object candidates per relation (filtered to single-token at runtime)
        "job":  "chef nurse pilot actor singer farmer judge poet monk clerk baker tutor sailor tailor".split(),
        "pet":  "dog cat fish horse goat frog mouse snake duck rabbit sheep pig hen owl".split(),
        "drink": "coffee tea water juice wine beer milk cola cider cocoa lemonade whiskey".split(),
        "city": "Paris Rome Tokyo Cairo Lima Oslo Bern Doha Nice Vienna Madrid Boston Dublin Prague".split(),
        "tongue": "French German Spanish Latin Greek Dutch Arabic Thai Hindi Polish Czech Danish".split(),
    }
    KG_RELS = list(KG_REL)
    kg_objs = {r: [o for o in KG_OBJS[r] if one_tok(o) is not None] for r in KG_RELS}  # single-token only

    def p_seen(n, a, v):
        return KG_REL[a][0].format(s=n) if DATA == "kg" else f"{n}'s {a} is"

    def p_para(n, a, v):
        return KG_REL[a][1].format(s=n) if DATA == "kg" else f"The {a} of {n} is"

    def kstem(f):   # retrieval-key stem
        return p_seen(*f) if DATA == "kg" else f"{f[0]}'s {f[1]}"

    def vfull(f):   # full prompt including the answer token (for value features)
        return p_seen(*f) + " " + f[2] if DATA == "kg" else f"{f[0]}'s {f[1]} is {f[2]}"

    @torch.no_grad()
    def pooled(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = m.model(**e).last_hidden_state[..., :, :]; msk = e.attention_mask[..., None].to(h.dtype)
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
        rng = random.Random(3000 + seed)
        if DATA == "kg":                              # KG-shaped COUNTERFACTUAL triples
            pairs = [(s, r) for s in KG_SUBJECTS for r in KG_RELS]
            rng.shuffle(pairs)
            streams = []; pi = 0
            for _ in range(ROUNDS):
                s = []
                while len(s) < PER and pi < len(pairs):
                    subj, rel = pairs[pi]; pi += 1
                    s.append((subj, rel, rng.choice(kg_objs[rel])))   # counterfactual object
                streams.append(s)
            return streams
        pool = big_pool[:]; rng.shuffle(pool)
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

    def train_memory(feat, facts, seed):
        Kf = pooled(feat, [kstem(f) for f in facts])
        Sf = last_h(feat, [vfull(f) for f in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        QHf = [(pooled(feat, [pf(*f) for f in facts]), last_h(feat, [pf(*f) for f in facts]))
               for pf in (p_seen, p_para)]
        Nb = len(facts)
        mk = lambda i, o: nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Linear(i, o)).to(device)
        best = None
        for attempt in range(MEMRESTART + 1):
            torch.manual_seed(seed + attempt * 911)
            proj_k, proj_q, val_enc = mk(d, KDIM), mk(d, KDIM), mk(d, 256)
            val_dec = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d)).to(device)
            gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
            nn.init.constant_(gate[-1].bias, 2.0)
            mods = (proj_k, proj_q, val_enc, val_dec, gate)
            rngv = random.Random(seed + attempt)
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
            with torch.no_grad():
                Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
                Qs, Hs = QHf[0]
                q = F.normalize(proj_q(Qs), -1); sims = q @ Kall.t() / 0.05
                vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
                Rr = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
                gg = torch.sigmoid(gate(torch.cat([Hs, Rr], -1)))
                ans = feat.lm_head(Hs + gg * Rr).float().argmax(-1).eq(gold).float().mean().item()
            if best is None or ans > best[3]:
                best = (mods, Kf, Sf, ans, attempt)
            if ans >= COLLAPSE_THR:
                return mods, Kf, Sf, ans, attempt
        return best

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

    @torch.no_grad()
    def recall_bank(feat, bank, facts, phrase):
        # external-memory arm: answer via the persistent scaffold bank
        mods, Kf, Sf = bank
        prompts = [phrase(*f) for f in facts]
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        ok = 0
        for i in range(0, len(prompts), 128):
            lg = teacher_logits(feat, mods, Kf, Sf, prompts[i:i + 128])
            ok += (lg.argmax(-1) == gold[i:i + 128]).sum().item()
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

    feat = load_frozen()

    @torch.no_grad()
    def base_logits(enc):
        return feat.lm_head(feat.model(**enc).last_hidden_state[:, -1]).float()

    def eval_all(m, streams, r):
        seen = [recall(m, streams[j], p_seen) for j in range(r + 1)]
        para = [recall(m, streams[j], p_para) for j in range(r + 1)]
        return seen, para, hop_acc(m)

    def n_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # ---------------- consolidation-style arms (ours / naive / continued / loramerge / oracle) ---
    # scaffolds: shared per-(seed,stream) teacher list (train_memory once, reuse across arms).
    def run_consolidate(seed, streams, kind, scaffolds):
        dense = load_frozen()
        base_hop = hop_acc(dense)
        rng = random.Random(seed * 13 + hash(kind) % 97)
        hist = []; opt_steps = 0; tparams = []
        replay = kind in ("ours", "oracle")            # ours=self-distill(no gold); oracle=gold-old CE
        for r in range(ROUNDS):
            S = streams[r]; prior = [f for s in streams[:r] for f in s]
            mods = Kf = Sf = None
            if kind in ("ours", "naive", "loramerge", "oracle"):
                mods, Kf, Sf = scaffolds[r][0], scaffolds[r][1], scaffolds[r][2]
            grow_qwen(dense, GROW); set_trainable_top(dense, GROW)
            tparams.append(n_trainable(dense))
            snap = None
            if kind == "ours" and prior:               # self-distill needs a pre-round snapshot
                snap = copy.deepcopy(dense).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
            opt = torch.optim.AdamW([p for p in dense.parameters() if p.requires_grad],
                                    lr=(CONT_LR if kind == "continued" else LR))
            dense.train()
            for _ in range(STEPS):
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                if kind == "continued":                              # GOLD new-stream signal
                    gold = torch.tensor([one_tok(f[2]) for f in sub], device=device)
                    loss = F.cross_entropy(s_lg, gold)
                else:                                                # scaffold teacher (NO gold)
                    with torch.no_grad():
                        t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                    loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                        F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                if replay and prior:                                 # REPLAY prior streams
                    sub2 = [rng.choice(prior) for _ in range(Bf)]
                    ph2 = p_seen if rng.random() < 0.5 else p_para
                    e2 = tok([ph2(*f) for f in sub2], return_tensors="pt", padding=True).to(device)
                    s2 = dense.lm_head(dense.model(**e2, use_cache=False).last_hidden_state[:, -1]).float()
                    if kind == "oracle":                             # ORACLE: old-stream GOLD CE
                        gold2 = torch.tensor([one_tok(f[2]) for f in sub2], device=device)
                        loss = loss + F.cross_entropy(s2, gold2)
                    else:                                            # ours: self-distill to snapshot (NO gold)
                        with torch.no_grad():
                            snap_lg = snap.lm_head(snap.model(**e2, use_cache=False).last_hidden_state[:, -1]).float()
                        loss = loss + F.kl_div(F.log_softmax(s2, -1), F.softmax(snap_lg, -1), reduction="batchmean")
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                sa = dense.lm_head(dense.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    ba = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(ba, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in dense.parameters() if p.requires_grad], 1.0)
                opt.step(); opt_steps += 1
            dense.eval(); del snap; torch.cuda.empty_cache()
            seen, para, h = eval_all(dense, streams, r)
            hist.append((seen, para, h))
            print(f"    [{kind:9s} seed {seed} r{r}] L={len(dense.model.layers)} "
                  f"seen={[round(x,2) for x in seen]} hop={h:.3f}", flush=True)
        del dense; torch.cuda.empty_cache()
        return dict(base_hop=base_hop, hist=hist, opt_steps=opt_steps,
                    tparams_per_round=tparams, updated_params_total=sum(tparams))

    # ---------------- LoRA-merge arm (fixed size, per-stream adapter merged in) -----------------
    def run_loramerge(seed, streams, scaffolds):
        from peft import LoraConfig, get_peft_model
        dense = load_frozen()
        base_hop = hop_acc(dense)
        rng = random.Random(seed * 13 + 41)
        hist = []; opt_steps = 0; tparams = []
        for r in range(ROUNDS):
            S = streams[r]
            mods, Kf, Sf = scaffolds[r][0], scaffolds[r][1], scaffolds[r][2]
            cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0,
                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            peft_m = get_peft_model(dense, cfg)
            tparams.append(sum(p.numel() for p in peft_m.parameters() if p.requires_grad))
            opt = torch.optim.AdamW([p for p in peft_m.parameters() if p.requires_grad], lr=2e-4)
            peft_m.train()
            for _ in range(STEPS):
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = peft_m(**e, use_cache=False).logits[:, -1].float()
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in peft_m.parameters() if p.requires_grad], 1.0)
                opt.step(); opt_steps += 1
            dense = peft_m.merge_and_unload()                        # fold LoRA into dense; drop adapter
            for p in dense.parameters():
                p.requires_grad_(False)
            dense.eval(); torch.cuda.empty_cache()
            seen, para, h = eval_all(dense, streams, r)
            hist.append((seen, para, h))
            print(f"    [loramerge seed {seed} r{r}] seen={[round(x,2) for x in seen]} hop={h:.3f}", flush=True)
        del dense; torch.cuda.empty_cache()
        return dict(base_hop=base_hop, hist=hist, opt_steps=opt_steps,
                    tparams_per_round=tparams, updated_params_total=sum(tparams))

    # ---------------- external-memory arm (persistent bank; inference USES memory) --------------
    def run_extmem(seed, streams):
        base_hop = hop_acc(feat)
        hist = []
        bankK = []; bankS = []; bankmods = None
        # a single scaffold trained on the CUMULATIVE facts each round (persistent memory)
        for r in range(ROUNDS):
            cumulative = [f for s in streams[:r + 1] for f in s]
            mods, Kf, Sf, t_r1, n_rs = train_memory(feat, cumulative, seed * 100 + r)
            bank = (mods, Kf, Sf)
            seen = [recall_bank(feat, bank, streams[j], p_seen) for j in range(r + 1)]
            para = [recall_bank(feat, bank, streams[j], p_para) for j in range(r + 1)]
            hist.append((seen, para, base_hop))
            print(f"    [extmem    seed {seed} r{r}] seen={[round(x,2) for x in seen]}", flush=True)
        return dict(base_hop=base_hop, hist=hist, opt_steps=0,
                    tparams_per_round=[0], updated_params_total=0)

    # ---------------- ewc_fixed arm (canonical online-EWC, NO growth, NO replay) -----------------
    # Codex caveat: with per-round growth the prior-round Fisher params freeze next round -> EWC is
    # vacuous. So EWC uses a FIXED top-GROW base layer set across all rounds (standard CL setup):
    # can a regularization-only baseline replace replay? online-EWC penalty anchors to the running
    # params weighted by accumulated diagonal Fisher; new stream distilled from the scaffold teacher.
    def run_ewc(seed, streams, scaffolds):
        lam = float(os.environ.get("BK_EWC_LAMBDA", 100.0))
        fisher_n = int(os.environ.get("BK_EWC_FISHERN", 8))
        dense = load_frozen()
        base_hop = hop_acc(dense)
        set_trainable_top(dense, GROW)                    # FIXED top-GROW base layers (no growth)
        trainable = [p for p in dense.parameters() if p.requires_grad]
        rng = random.Random(seed * 13 + 71)
        hist = []; opt_steps = 0
        f_accum = [torch.zeros_like(p) for p in trainable]   # accumulated diagonal Fisher
        theta_star = None                                    # anchor (running params)
        for r in range(ROUNDS):
            S = streams[r]; mods, Kf, Sf = scaffolds[r][0], scaffolds[r][1], scaffolds[r][2]
            opt = torch.optim.AdamW(trainable, lr=LR)
            dense.train()
            for _ in range(STEPS):
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                sa = dense.lm_head(dense.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    ba = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(ba, -1), reduction="batchmean")
                if theta_star is not None:                   # online-EWC quadratic penalty
                    pen = sum((fa * (p - ts) ** 2).sum() for fa, p, ts in zip(f_accum, trainable, theta_star))
                    loss = loss + lam * pen
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt_steps += 1
            dense.eval()
            # estimate diagonal Fisher on this stream (squared grads of the teacher-target loss)
            fish = [torch.zeros_like(p) for p in trainable]
            for _ in range(fisher_n):
                sub = [rng.choice(S) for _ in range(Bf)]
                pr = [(p_seen if rng.random() < 0.5 else p_para)(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    tgt = teacher_logits(feat, mods, Kf, Sf, pr).argmax(-1)
                fl = F.cross_entropy(s_lg, tgt)
                opt.zero_grad(); fl.backward()
                for i, p in enumerate(trainable):
                    if p.grad is not None:
                        fish[i] += p.grad.detach() ** 2
            for i in range(len(f_accum)):
                f_accum[i] = f_accum[i] + fish[i] / fisher_n
            theta_star = [p.detach().clone() for p in trainable]
            seen, para, h = eval_all(dense, streams, r)
            hist.append((seen, para, h))
            print(f"    [ewc       seed {seed} r{r}] lam={lam} seen={[round(x,2) for x in seen]} hop={h:.3f}", flush=True)
        del dense; torch.cuda.empty_cache()
        tp = n_trainable_count(trainable)
        return dict(base_hop=base_hop, hist=hist, opt_steps=opt_steps,
                    tparams_per_round=[tp] * ROUNDS, updated_params_total=tp * ROUNDS)

    def n_trainable_count(ps):
        return sum(p.numel() for p in ps)

    # ---------------- nswrite arm (REHEARSAL-FORBIDDEN activation null-space writer) --------------
    # The frontier probe (R36-I): write new-stream knowledge WITHOUT replaying old inputs, by
    # projecting each trainable Linear's weight-gradient off the input directions that old streams
    # read from (null space of old-stream last-token input activations). Only a low-rank basis U is
    # kept as TRAINING state (no old prompts/answers/Kf/Sf); inference uses only the dense checkpoint.
    # mode="act": U from old activations (the method). mode="rand": random-basis control of matched
    # rank (attributes any gain to interference-awareness, not mere gradient shrinkage).
    # Reports gradient OCCUPANCY ||G·UUᵀ||/||G|| — the future growth-saturation signal.
    # margin modes ("margin"/"marginrandv") also protect the OUTPUT directions the old answer-margin is
    # sensitive to: bilinear projection V_out V_outᵀ·G·U_in U_inᵀ removes only gradient that both acts on
    # old inputs AND moves old answer-margin directions (targets the token, not just the hidden response).
    # cap_policy: "old_first" (current QR-truncation) or "noevict" (keep full union rank).
    def run_nswrite(seed, streams, scaffolds, mode):
        kper = int(os.environ.get("BK_NS_KPER", 40))
        rankcap = int(os.environ.get("BK_NS_RANK", 256))
        cap_policy = os.environ.get("BK_NS_CAP_POLICY", "old_first")
        reproject = int(os.environ.get("BK_NS_REPROJECT", 0))  # enforce accumulated ΔW ⊥ U each step
        margin_mode = mode in ("margin", "marginrandv")
        dense = load_frozen()
        base_hop = hop_acc(dense)
        set_trainable_top(dense, GROW)                    # FIXED top-GROW base layers (no growth)
        targets = []
        for lyr in dense.model.layers[-GROW:]:
            for mod in lyr.modules():
                if isinstance(mod, nn.Linear):
                    targets.append(mod)
                    if mod.bias is not None:
                        mod.bias.requires_grad_(False)     # freeze bias (shared shift over all inputs)
        trainable = [p for p in dense.parameters() if p.requires_grad]
        Umap = {id(m): None for m in targets}             # input-side basis
        Vmap = {id(m): None for m in targets}             # output-side answer-margin basis (margin modes)
        rng = random.Random(seed * 13 + 53)
        hist = []; occ_hist = []; eff_hist = []; leak_hist = []; opt_steps = 0

        def collect_inputs(prompts):
            store = {id(m): [] for m in targets}
            def mk(mid):
                def hook(module, inp):
                    store[mid].append(inp[0][:, -1, :].detach().float())
                return hook
            handles = [m.register_forward_pre_hook(mk(id(m))) for m in targets]
            with torch.no_grad():
                for i in range(0, len(prompts), 128):
                    e = tok(prompts[i:i + 128], return_tensors="pt", padding=True).to(device)
                    dense.model(**e)
            for h in handles:
                h.remove()
            return {mid: torch.cat(v, 0) for mid, v in store.items()}

        def collect_Vout(facts, k):
            # per-module d(margin)/d(y_last) over COMMITTED-CORRECT old facts (eval-only training state);
            # SVD of the stacked (not averaged) sensitivity vectors -> low-rank output basis V_out.
            items = []
            for f in facts:
                gt = one_tok(f[2])
                items.append((p_seen(*f), gt)); items.append((p_para(*f), gt))
            store = {id(m): [] for m in targets}
            for i in range(0, len(items), 64):
                chunk = items[i:i + 64]
                e = tok([p for p, _ in chunk], return_tensors="pt", padding=True).to(device)
                gold = torch.tensor([g for _, g in chunk], device=device)
                cap = {}
                def mk(mid):
                    def hook(module, inp, output):
                        output.retain_grad(); cap[mid] = output
                    return hook
                handles = [m.register_forward_hook(mk(id(m))) for m in targets]
                logits = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                gl = logits.gather(1, gold[:, None]).squeeze(1)
                masked = logits.clone(); masked.scatter_(1, gold[:, None], float("-inf"))
                correct = (logits.argmax(1) == gold)
                margin = ((gl - masked.max(1).values) * correct.float()).sum()
                dense.zero_grad()
                if margin.requires_grad:
                    margin.backward()
                for m in targets:
                    y = cap.get(id(m))
                    if y is not None and y.grad is not None:
                        gy = y.grad[:, -1, :].detach().float()[correct]
                        if gy.shape[0] > 0:
                            store[id(m)].append(gy)
                for hh in handles:
                    hh.remove()
            out = {}
            for mid, lst in store.items():
                if not lst:
                    out[mid] = None; continue
                A = torch.cat(lst, 0)
                if mode == "marginrandv":
                    q, _ = torch.linalg.qr(torch.randn(A.shape[1], min(k, A.shape[1]), device=device))
                    out[mid] = q
                else:
                    try:
                        _u, _s, Vh = torch.linalg.svd(A, full_matrices=False)
                        out[mid] = Vh[:min(k, Vh.shape[0])].t().contiguous()
                    except Exception:
                        out[mid] = None
            return out

        def basis_from(A, k):
            if mode == "none":                            # naive_fixed: no basis, no projection
                return None
            if mode == "rand":
                q, _ = torch.linalg.qr(torch.randn(A.shape[1], min(k, A.shape[1]), device=device))
                return q
            try:
                _u, _s, Vh = torch.linalg.svd(A, full_matrices=False)
            except Exception:
                return None
            return Vh[:min(k, Vh.shape[0])].t().contiguous()

        def update_U(Uold, newB):
            if newB is None:
                return Uold
            M = newB if Uold is None else torch.cat([Uold, newB], 1)
            q, _ = torch.linalg.qr(M)
            cap = M.shape[1] if cap_policy == "noevict" else rankcap
            return q[:, :cap].contiguous()

        ns_lr = float(os.environ.get("BK_NS_LR", LR))     # drift sweep: lower LR = less per-step drift
        ns_wd = float(os.environ.get("BK_NS_WD", 0.01))   # AdamW decoupled decay is an UNPROJECTED update
        ns_sgd = int(os.environ.get("BK_NS_SGD", 0))      # SGD update stays ⊥U (Adam preconditioner rotates it)
        if int(os.environ.get("BK_NS_LINEAR_ONLY", 0)):   # freeze non-Linear top params (norms leak, unprotected)
            tw = {id(m.weight) for m in targets}
            for p in list(trainable):
                if id(p) not in tw:
                    p.requires_grad_(False)
            trainable = [p for p in dense.parameters() if p.requires_grad]
        for r in range(ROUNDS):
            S = streams[r]; mods, Kf, Sf = scaffolds[r][0], scaffolds[r][1], scaffolds[r][2]
            opt = (torch.optim.SGD(trainable, lr=ns_lr, momentum=0.9) if ns_sgd
                   else torch.optim.AdamW(trainable, lr=ns_lr, weight_decay=ns_wd))
            dense.train()
            occ_acc = 0.0; occ_n = 0
            Wstart = {id(m): m.weight.detach().clone() for m in targets}   # for realized-ΔW leak + reproject
            for _ in range(STEPS):
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                sa = dense.lm_head(dense.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    ba = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(ba, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                gn = gpn = 0.0
                for m in targets:
                    U = Umap[id(m)]
                    if m.weight.grad is None or U is None:
                        continue
                    g = m.weight.grad
                    V = Vmap[id(m)]
                    if margin_mode and V is not None:         # bilinear: only old-input × old-margin-output
                        proj = V @ ((V.t() @ g) @ U) @ U.t()
                    else:                                     # input-only null space
                        proj = (g @ U) @ U.t()
                    gpn += proj.pow(2).sum().item(); gn += g.pow(2).sum().item()
                    m.weight.grad = g - proj
                if gn > 0:
                    occ_acc += gpn / gn; occ_n += 1
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt_steps += 1
                if reproject:                             # enforce accumulated ΔW ⊥ U (immune to Adam rotation)
                    with torch.no_grad():
                        for m in targets:
                            U = Umap[id(m)]
                            if U is None:
                                continue
                            dW = m.weight.data - Wstart[id(m)]
                            m.weight.data.sub_((dW @ U) @ U.t())
            dense.eval()
            if mode != "none":                            # commit: update protected basis U (and V_out)
                acts = collect_inputs([p_seen(*f) for f in S] + [p_para(*f) for f in S])
                for m in targets:
                    Umap[id(m)] = update_U(Umap[id(m)], basis_from(acts[id(m)], kper))
                if margin_mode:
                    vout = collect_Vout(S, kper)
                    for m in targets:
                        Vmap[id(m)] = update_U(Vmap[id(m)], vout[id(m)])
            energy = occ_acc / max(occ_n, 1)              # ||G·UUᵀ||²/||G||² (gradient ENERGY fraction)
            occ_hist.append(round(energy, 3))
            eff_hist.append(round((max(0.0, 1.0 - energy)) ** 0.5, 3))  # ||G(I-UUᵀ)||/||G|| norm ratio
            # realized-ΔW leak: fraction of the ACTUAL weight change that landed in old-input dirs U
            with torch.no_grad():
                dn = ln = 0.0
                for m in targets:
                    U = Umap[id(m)]
                    if U is None:
                        continue
                    dW = m.weight.data - Wstart[id(m)]
                    ln += ((dW @ U) @ U.t()).pow(2).sum().item(); dn += dW.pow(2).sum().item()
                leak_hist.append(round(ln / dn, 3) if dn > 0 else 0.0)
            seen, para, h = eval_all(dense, streams, r)
            hist.append((seen, para, h))
            print(f"    [nswrite-{mode:4s} seed {seed} r{r}] energy-occ={occ_hist[-1]} "
                  f"eff-grad={eff_hist[-1]} dW-leak={leak_hist[-1]} fresh={round(seen[r],2)} "
                  f"seen={[round(x,2) for x in seen]} hop={h:.3f}", flush=True)
        del dense; torch.cuda.empty_cache()
        tp = n_trainable_count(trainable)
        return dict(base_hop=base_hop, hist=hist, opt_steps=opt_steps,
                    tparams_per_round=[tp] * ROUNDS, updated_params_total=tp * ROUNDS,
                    occupancy_curve=occ_hist, effective_grad_curve=eff_hist,
                    realized_leak_curve=leak_hist)

    # ---------------- ogd arm (answer-level Orthogonal Gradient Descent, rehearsal-free) ----------
    # R36-C: the STRONGEST rehearsal-free primitive, distinct from nswrite. Instead of protecting each
    # Linear's LOCAL input response (ΔW·U≈0, per-module), store a low-rank basis Q of the old-stream
    # ANSWER-MARGIN gradients w.r.t. ALL trainable params jointly, and project the flattened update off Q
    # (g <- g - Q Qᵀ g). This is a FIRST-ORDER protection of the old ANSWER objective itself: the stored
    # directions route through the later nonlinearity, norms, residual stream and lm_head — coupling that
    # the per-module V⊗U factorization (nswrite/margin) cannot represent. NOTE (design decision, see
    # qa/claude 2026-07-06): blockwise per-tensor OGD is NOT run — a linear layer's per-sample margin grad
    # is outer(g_y, x), so a per-tensor basis spans a SUBSET of margin-mode's V_out⊗U_in, which R36-I
    # already showed retains WORSE than blunt input-only nswrite. Only the JOINT flattened basis is a
    # genuinely different (stronger) test. Memory: Q is [P≈29.8M, R] fp16 on GPU (R=64 → 3.8GB); per-stream
    # merge uses a Gram/eigh trick to avoid any giant SVD/QR. Only Q survives across rounds (training state,
    # no old prompts/answers/gold); inference is the single dense checkpoint.
    def run_ogd(seed, streams, scaffolds):
        kper = int(os.environ.get("BK_OGD_KPER", 24))     # new basis dirs kept per stream
        rankcap = int(os.environ.get("BK_OGD_RANK", 64))  # total Q rank cap (memory: 119MB/rank fp32)
        ncollect = int(os.environ.get("BK_OGD_NCOL", 80)) # committed-correct margin samples per stream
        ogd_lr = float(os.environ.get("BK_OGD_LR", LR))
        ogd_sgd = int(os.environ.get("BK_OGD_SGD", 0))    # cross-check: SGD-mom keeps update ~⊥Q natively
        ogd_wd = float(os.environ.get("BK_OGD_WD", 0.0))  # wd=0: decoupled decay is an unprojected update
        # exact-OGD: reproject the REALIZED ΔΘ ⊥ Q each step (AdamW precond/decay do NOT commute with the
        # gradient projection, so Qᵀg=0 does NOT imply Qᵀ ΔΘ=0 — same realized-leak issue as nswrite).
        reproject = int(os.environ.get("BK_OGD_REPROJECT", 1))
        obj = os.environ.get("BK_OGD_OBJ", "ce_gold")     # ce_gold=canonical OGD (old task loss); margin=thin
        dense = load_frozen()
        base_hop = hop_acc(dense)
        set_trainable_top(dense, GROW)                    # FIXED top-GROW base layers (all params protected)
        trainable = [p for p in dense.parameters() if p.requires_grad]
        shapes = [p.shape for p in trainable]
        sizes = [p.numel() for p in trainable]
        P = sum(sizes); offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        rng = random.Random(seed * 13 + 71)
        hist = []; occ_hist = []; eff_hist = []; opt_steps = 0
        Q = None                                          # [P, R] fp16 orthonormal answer-grad basis

        def flat_grad():
            return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                              for p in trainable]).float()

        def write_grad(g):
            for p, a, b in zip(trainable, offs[:-1], offs[1:]):
                p.grad = g[a:b].view_as(p).clone()

        def flat_params():
            return torch.cat([p.data.reshape(-1) for p in trainable]).float()

        def add_flat(delta):                              # theta += delta (flattened)
            for p, a, b in zip(trainable, offs[:-1], offs[1:]):
                p.data.add_(delta[a:b].view_as(p))

        def old_obj_grad(prompt, gold):
            # d(old_answer_objective)/d(theta_trainable) for ONE committed-correct fact -> flattened [P] fp16.
            # obj="ce_gold": CE(-logP(gold)) = the canonical OGD old-TASK-LOSS gradient (shares structure
            #   with the new write CE-to-teacher-argmax term). obj="margin": logit_gold - runnerup (thin,
            #   ranking-only; diagnostic contrast). NOTE: KL-to-committed is NOT usable here — at commit
            #   p_theta==p_committed so its gradient is 0 (needs a Fisher/Jacobian basis, deferred).
            e = tok([prompt], return_tensors="pt", padding=True).to(device)
            logits = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()[0]
            if logits.argmax().item() != gold:            # only protect what the model actually knows
                return None
            if obj == "margin":
                masked = logits.clone(); masked[gold] = float("-inf")
                old = logits[gold] - masked.max()
            else:                                          # ce_gold (default)
                old = F.cross_entropy(logits[None, :], torch.tensor([gold], device=device))
            dense.zero_grad()
            old.backward()
            v = flat_grad()
            dense.zero_grad()
            n = v.norm()
            return (v / n).half() if n > 0 else None

        def collect_Q(S):
            items = []
            for f in S:
                gt = one_tok(f[2])
                items += [(p_seen(*f), gt), (p_para(*f), gt)]
            rng.shuffle(items)
            A = torch.empty(P, ncollect, dtype=torch.float16, device=device)  # write columns in-place (no 2x peak)
            n = 0
            for pr, gt in items:
                if n >= ncollect:
                    break
                v = old_obj_grad(pr, gt)
                if v is not None:
                    A[:, n] = v; n += 1
            if n == 0:
                return None, 0
            A = A[:, :n]
            G = (A.transpose(0, 1) @ A).float()            # [n, n] (fp32-accumulated)
            evals, evecs = torch.linalg.eigh(G)
            k = min(kper, n)
            E = evecs[:, -k:]; s = evals[-k:].clamp_min(1e-8).sqrt()
            return ((A @ E.half()) / s.half()).contiguous(), n  # [P, k] fp16, ~orthonormal

        def merge_Q(Qold, Qnew):
            if Qnew is None:
                return Qold
            C = Qnew if Qold is None else torch.cat([Qold, Qnew], 1)   # [P, m] fp16
            G = (C.transpose(0, 1) @ C).float()
            evals, evecs = torch.linalg.eigh(G)
            k = min(rankcap, C.shape[1])
            E = evecs[:, -k:]; s = evals[-k:].clamp_min(1e-8).sqrt()
            return ((C @ E.half()) / s.half()).contiguous()

        leak_hist = []; ncol_hist = []
        for r in range(ROUNDS):
            S = streams[r]; mods, Kf, Sf = scaffolds[r][0], scaffolds[r][1], scaffolds[r][2]
            opt = (torch.optim.SGD(trainable, lr=ogd_lr, momentum=0.9) if ogd_sgd
                   else torch.optim.AdamW(trainable, lr=ogd_lr, weight_decay=ogd_wd))
            dense.train()
            occ_acc = 0.0; occ_n = 0; uleak_acc = 0.0; uleak_n = 0
            for _ in range(STEPS):
                sub = [rng.choice(S) for _ in range(Bf)]
                phrase = p_seen if rng.random() < 0.5 else p_para
                pr = [phrase(*f) for f in sub]
                e = tok(pr, return_tensors="pt", padding=True).to(device)
                s_lg = dense.lm_head(dense.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, pr)
                loss = F.cross_entropy(s_lg, t_lg.argmax(-1)) + F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                sa = dense.lm_head(dense.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    ba = base_logits(ea)
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(ba, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                if Q is not None:                          # OGD: project joint gradient off old-answer basis
                    g = flat_grad()
                    gn = g.pow(2).sum().item()
                    coef = Q.transpose(0, 1) @ g.half()    # [R]
                    proj = (Q @ coef).float()
                    write_grad(g - proj)
                    if gn > 0:
                        occ_acc += proj.pow(2).sum().item() / gn; occ_n += 1
                th0 = flat_params() if (Q is not None and reproject) else None
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt_steps += 1
                if th0 is not None:                        # exact-OGD: force realized ΔΘ ⊥ Q (kills Adam leak)
                    dth = flat_params() - th0
                    dn = dth.pow(2).sum().item()
                    lk = (Q @ (Q.transpose(0, 1) @ dth.half())).float()
                    add_flat(-lk)                          # theta <- theta - Q Qᵀ ΔΘ
                    if dn > 0:
                        uleak_acc += lk.pow(2).sum().item() / dn; uleak_n += 1
            dense.eval()
            Qnew, ncol = collect_Q(S)                       # commit: fold this stream's answer dirs into Q
            Q = merge_Q(Q, Qnew); ncol_hist.append(ncol)
            energy = occ_acc / max(occ_n, 1)
            occ_hist.append(round(energy, 3))
            eff_hist.append(round((max(0.0, 1.0 - energy)) ** 0.5, 3))
            leak_hist.append(round(uleak_acc / uleak_n, 4) if uleak_n else 0.0)  # realized ΔΘ leak (~0 = clean)
            seen, para, h = eval_all(dense, streams, r)
            hist.append((seen, para, h))
            print(f"    [ogd-{obj:7s} seed {seed} r{r}] rank={0 if Q is None else Q.shape[1]} ncol={ncol} "
                  f"energy-occ={occ_hist[-1]} eff-grad={eff_hist[-1]} upd-leak={leak_hist[-1]} "
                  f"fresh={round(seen[r],2)} seen={[round(x,2) for x in seen]} hop={h:.3f}", flush=True)
        del dense, Q; torch.cuda.empty_cache()
        tp = n_trainable_count(trainable)
        return dict(base_hop=base_hop, hist=hist, opt_steps=opt_steps,
                    tparams_per_round=[tp] * ROUNDS, updated_params_total=tp * ROUNDS,
                    occupancy_curve=occ_hist, effective_grad_curve=eff_hist,
                    realized_leak_curve=leak_hist, ncollect_curve=ncol_hist)

    flags = {
        "ours":      dict(uses_gold_new=False, uses_gold_old=False, uses_replay=True,  uses_inference_memory=False, single_dense=True),
        "naive":     dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True),
        "continued": dict(uses_gold_new=True,  uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True),
        "loramerge": dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True),
        "oracle":    dict(uses_gold_new=False, uses_gold_old=True,  uses_replay=True,  uses_inference_memory=False, single_dense=True),
        "ewc":       dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True),
        "naive_fixed": dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=False),
        "nswrite":   dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=True),
        "nswrite_rand": dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=True),
        "margin":    dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=True),
        "marginrandv": dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=True),
        "ogd":       dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=False, single_dense=True, uses_training_state=True, uses_commit_answers=True),
        "extmem":    dict(uses_gold_new=False, uses_gold_old=False, uses_replay=False, uses_inference_memory=True,  single_dense=False),
    }

    results = {a: [] for a in ARMS}
    need_scaffold = any(a in ARMS for a in ("ours", "naive", "loramerge", "oracle", "ewc", "nswrite", "nswrite_rand", "naive_fixed", "margin", "marginrandv", "ogd"))

    def agg(arm):
        runs = results[arm]
        s0_fresh = sum(r["hist"][0][0][0] for r in runs) / len(runs)
        s0_fin = sum(r["hist"][-1][0][0] for r in runs) / len(runs)
        s0_para = sum(r["hist"][-1][1][0] for r in runs) / len(runs)
        allseen = sum(sum(r["hist"][-1][0]) / len(r["hist"][-1][0]) for r in runs) / len(runs)
        allpara = sum(sum(r["hist"][-1][1]) / len(r["hist"][-1][1]) for r in runs) / len(runs)
        newest = sum(r["hist"][-1][0][-1] for r in runs) / len(runs)
        bhop = sum(r["base_hop"] for r in runs) / len(runs)
        hop_f = sum(r["hist"][-1][2] for r in runs) / len(runs)
        # mean forgetting across ALL streams: fresh (stream j at round j) -> final
        def mf(run):
            h = run["hist"]
            return sum(h[j][0][j] - h[-1][0][j] for j in range(len(h))) / len(h)
        mean_forget = sum(mf(r) for r in runs) / len(runs)
        fvs = [r["hist"][-1][0] for r in runs]            # final per-stream seen vectors
        age = [round(sum(v[j] for v in fvs) / len(fvs), 3) for j in range(len(fvs[0]))]
        nst = len(runs[0]["hist"])
        fresh_diag = [round(sum(r["hist"][j][0][j] for r in runs) / len(runs), 3) for j in range(nst)]
        has_occ = all("occupancy_curve" in r for r in runs)
        has_eff = all("effective_grad_curve" in r for r in runs)
        occ = sum(r["occupancy_curve"][-1] for r in runs) / len(runs) if has_occ else None
        eff = sum(r["effective_grad_curve"][-1] for r in runs) / len(runs) if has_eff else None
        occ_curve = ([round(sum(r["occupancy_curve"][j] for r in runs) / len(runs), 3)
                      for j in range(nst)] if has_occ else None)
        eff_curve = ([round(sum(r["effective_grad_curve"][j] for r in runs) / len(runs), 3)
                      for j in range(nst)] if has_eff else None)
        return dict(
            gradient_occupancy_final=occ, effective_grad_final=eff,
            occupancy_curve=occ_curve, effective_grad_curve=eff_curve, fresh_diagonal=fresh_diag,
            oldest_S0_fresh=s0_fresh, oldest_S0_final=s0_fin, oldest_S0_forgetting=s0_fresh - s0_fin,
            oldest_S0_para_final=s0_para, all_seen_final=allseen, all_para_final=allpara,
            mean_forgetting_all_streams=mean_forget, age_curve_final_seen=age,
            newest_stream_final=newest, base_hop_before=bhop, base_hop_after=hop_f,
            trainable_params_per_round=runs[0]["tparams_per_round"],
            updated_params_total=sum(r["updated_params_total"] for r in runs) / len(runs),
            optimizer_steps=sum(r["opt_steps"] for r in runs) / len(runs),
            wall_clock_seconds=sum(r["wall"] for r in runs) / len(runs),
            peak_vram_mb=max(r["peak_vram_mb"] for r in runs), **flags[arm])

    out = os.environ.get("BK_OUT", "lifecycle_bakeoff_result.json")

    def dump(nseeds):
        print(f"\n== BAKEOFF (mean/{nseeds} seeds done, {ROUNDS} rounds x {PER} facts) ==")
        print(f"  {'arm':10s} {'S0 fresh->final(forget)':26s} {'all-seen':9s} {'all-para':9s} "
              f"{'newest':7s} {'hop':13s} {'gold':5s} {'replay':7s} {'infmem':7s}")
        summary = {}
        for arm in ARMS:
            if not results[arm]:
                continue
            a = agg(arm); summary[arm] = a
            print(f"  {arm:10s} {a['oldest_S0_fresh']:.2f}->{a['oldest_S0_final']:.2f} "
                  f"({a['oldest_S0_forgetting']:+.3f})            {a['all_seen_final']:.3f}     "
                  f"{a['all_para_final']:.3f}     {a['newest_stream_final']:.2f}    "
                  f"{a['base_hop_before']:.3f}->{a['base_hop_after']:.3f}  "
                  f"{str(a['uses_gold_new']):5s} {str(a['uses_replay']):7s} {str(a['uses_inference_memory'])}")
            print(f"             mean-forget(all-streams)={a['mean_forgetting_all_streams']:+.3f} "
                  f"age-curve(final seen S0..Sn)={a['age_curve_final_seen']}")
            if a.get("gradient_occupancy_final") is not None:
                print(f"             energy-occ={a['occupancy_curve']} eff-grad={a['effective_grad_curve']} "
                      f"fresh-diag={a['fresh_diagonal']}")
        if "ours" in summary and "oracle" in summary:
            og_s = summary["oracle"]["all_seen_final"] - summary["ours"]["all_seen_final"]
            og_p = summary["oracle"]["all_para_final"] - summary["ours"]["all_para_final"]
            print(f"  ORACLE_GAP (oracle-ours): seen {og_s:+.3f}  para {og_p:+.3f}  "
                  f"(small gap => no-gold self-distill ~ gold-old upper bound)")
        with open(out, "w") as f:
            json.dump(dict(config=dict(model=NAME, rounds=ROUNDS, per=PER, grow=GROW, steps=STEPS,
                                       seeds_done=nseeds, arms=ARMS, lora_r=LORA_R), summary=summary), f, indent=2)
        print(f"RESULT_JSON_{nseeds}SEED " + json.dumps(summary), flush=True)

    for seed in range(SEEDS):
        streams = make_streams(seed)
        print(f"  seed {seed}: streams={[len(s) for s in streams]}", flush=True)
        if DATA == "kg":                              # base-knowledge screen (codex: require <=0.15)
            allf = [f for s in streams for f in s]
            bseen = recall(feat, allf, p_seen); bpar = recall(feat, allf, p_para)
            print(f"    [KG base-recall screen] frozen-base seen={bseen:.3f} para={bpar:.3f} "
                  f"({'OK <=0.15' if bseen <= 0.15 else 'WARNING: base already knows facts'})", flush=True)
        # train each stream's scaffold ONCE (deterministic in seed*100+r); share across arms
        scaffolds = None
        if need_scaffold:
            scaffolds = [train_memory(feat, streams[r], seed * 100 + r) for r in range(ROUNDS)]
            print(f"    scaffolds answer-recall={[round(s[3],2) for s in scaffolds]} "
                  f"restarts={[s[4] for s in scaffolds]}", flush=True)
        for arm in ARMS:
            torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
            t0 = time.time()
            if arm == "loramerge":
                out_a = run_loramerge(seed, streams, scaffolds)
            elif arm == "extmem":
                out_a = run_extmem(seed, streams)
            elif arm == "ewc":
                out_a = run_ewc(seed, streams, scaffolds)
            elif arm == "nswrite":
                out_a = run_nswrite(seed, streams, scaffolds, "act")
            elif arm == "nswrite_rand":
                out_a = run_nswrite(seed, streams, scaffolds, "rand")
            elif arm == "naive_fixed":
                out_a = run_nswrite(seed, streams, scaffolds, "none")
            elif arm == "margin":
                out_a = run_nswrite(seed, streams, scaffolds, "margin")
            elif arm == "marginrandv":
                out_a = run_nswrite(seed, streams, scaffolds, "marginrandv")
            elif arm == "ogd":
                out_a = run_ogd(seed, streams, scaffolds)
            else:
                out_a = run_consolidate(seed, streams, arm, scaffolds)
            out_a["wall"] = time.time() - t0
            out_a["peak_vram_mb"] = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0
            results[arm].append(out_a)
        dump(seed + 1)                                   # partial-safe: summary+JSON after EVERY seed
    print("\n[all seeds done]", flush=True)


if __name__ == "__main__":
    main()
