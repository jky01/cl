"""ROUND 19 (PIVOT: Grow-and-Consolidate) — the CORE BRIDGE experiment. Stop treating the
external memory as a product; use it as a short-term TEACHER SCAFFOLD, then CONSOLIDATE its
knowledge into a grown DENSE model so the final artifact is a SINGLE checkpoint that answers
new facts WITHOUT any memory at inference.

  base    : Qwen-0.5B, frozen (this is M_t and the preservation reference)
  student : base + grow_qwen(k) appended identity-init layers; ONLY the appended layers train
            (function-preserving: student == base at init, new capacity absorbs the facts)
  teacher : base + capsule memory (memory trained on the new facts -> answers them via
            retrieval+injection at the last hidden state). The scaffold. Discarded after.

Consolidation: train the appended layers so the student, WITH NO MEMORY, reproduces the
teacher's fact-answering and predicts the gold answers, WITHOUT drifting on old ability.
  L = L_fact(gold CE, seen+paraphrase+reverse) + λd·L_distill(KL to teacher on seen)
      + λp·L_preserve(KL to base on anchor prompts)     [old-ability guard / locality]

The thesis number is STUDENT-WITHOUT-MEMORY recall, reported in FOUR ways so we can't fake
consolidation with surface pattern-matching (per design guidance):
  seen        : "{n}'s {a} is"            -> value
  paraphrase  : "The {a} of {n} is"       -> value
  reverse     : "The person whose {a} is {v} is" -> name   (single-token names)
  interference: full-set recall over all facts (disambiguation is implicit)
Also: teacher recall (with memory), base recall (should be ~0, facts are novel), old hop-acc
drop (base vs student), and Consolidation Gap G = teacher_seen − student_seen.

Tiered by CO_FACTS: run 50 (R19a) -> 200 (R19b) -> 1000 (R19c). R19a success bar:
teacher seen >0.95, student-no-mem seen >0.85, old hop-acc drop <2%.

  python -m s2.consolidate_memory_to_weights   # env: CO_FACTS, CO_GROW, CO_MEM_STEPS,
                                               #      CO_STU_STEPS, CO_SEEDS, CO_LD, CO_LP
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
FACTS = int(os.environ.get("CO_FACTS", 50))
GROW = int(os.environ.get("CO_GROW", 4))              # appended identity layers (new capacity)
MEM_STEPS = int(os.environ.get("CO_MEM_STEPS", 800))  # teacher-memory training (few facts -> cheap)
STU_STEPS = int(os.environ.get("CO_STU_STEPS", 1500)) # student consolidation steps
SEEDS = int(os.environ.get("CO_SEEDS", 2))
LD = float(os.environ.get("CO_LD", 1.0))              # distill weight
LP = float(os.environ.get("CO_LP", 1.0))             # preserve weight
# preservation set: "anchor" = generic completions only (R19a); "hopreplay" adds in-context
# hop prompts (old-task replay) so KL-to-base protects the reasoning ability, not just LM text.
PRESERVE = os.environ.get("CO_PRESERVE", "anchor")
# distill-only: the student learns facts SOLELY from the teacher (base+memory) outputs on the
# SEEN phrasing — NO gold labels, and para/reverse are NOT trained (held out to test whether the
# student GENERALIZES from what the scaffold taught). Proves the memory scaffold (not gold) is
# what carries the knowledge into dense weights. Use at a teacher-good scale (few facts).
DISTILL_ONLY = bool(os.environ.get("CO_DISTILL_ONLY"))
KDIM = 256
TOPK = 16
Bf = 24                                               # fact-batch
Ba = 16                                               # anchor-batch
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
    tok.padding_side = "left"                          # [:, -1] = real last token
    d = AutoConfig.from_pretrained(NAME).hidden_size
    print(f"CONSOLIDATE ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"facts={FACTS} grow=+{GROW}L mem-steps={MEM_STEPS} stu-steps={STU_STEPS} "
          f"seeds={SEEDS} λd={LD} λp={LP} preserve={PRESERVE} distill_only={DISTILL_ONLY}")

    def load_frozen():
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None
    av = {a: [v for v in vs if one_tok(v) is not None] for a, vs in ATTR_VALUES.items()}

    @torch.no_grad()
    def pooled(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            h = m.model(**e).last_hidden_state
            msk = e.attention_mask[..., None].to(h.dtype)
            outs.append(((h * msk).sum(1) / msk.sum(1)).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def last_h(m, texts, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(device)
            outs.append(m.model(**e).last_hidden_state[:, -1].float())
        return torch.cat(outs, 0)

    # ---- QA phrasings for a fact (n, a, v); all answers single-token ----
    def p_seen(n, a, v): return f"{n}'s {a} is"
    def p_para(n, a, v): return f"The {a} of {n} is"
    def p_rev(n, a, v):  return f"The person whose {a} is {v} is"

    def build_facts(seed):
        """Returns (facts, rev_idx). Scales to FACTS via a large multi-token name pool
        (seen/para only need the name in the PROMPT; the single-token VALUE is the answer).
        rev_idx marks a subset with SINGLE-TOKEN names AND globally-unique (attr,value) so the
        reverse-query answer (the name) is a single token and unambiguous."""
        rng = random.Random(2000 + seed)
        st_names = single_tok_names(tok); rng.shuffle(st_names)   # reverse-eligible subjects
        big_pool = [f"{f} {l}" for f in FIRST for l in LAST]; rng.shuffle(big_pool)  # bulk (multi-tok ok)
        facts = []; used_na = set(); av_ct = {}
        for n in st_names:                                # first: reverse-eligible facts (unique a,v)
            a = rng.choice(ATTRS); v = rng.choice(av[a])
            if (a, v) in av_ct or (n, a) in used_na:
                continue
            used_na.add((n, a)); av_ct[(a, v)] = 1; facts.append((n, a, v))
            if len(facts) >= FACTS:
                break
        n_rev = len(facts)                                # facts[:n_rev] are reverse-eligible
        pi = 0                                            # then: bulk facts to reach FACTS
        while len(facts) < FACTS and pi < len(big_pool):
            n = big_pool[pi]; pi += 1
            a = rng.choice(ATTRS); v = rng.choice(av[a])
            if (n, a) in used_na:
                continue
            used_na.add((n, a)); facts.append((n, a, v))
        return facts, list(range(n_rev))

    # ---- teacher memory (scaffold) ----
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
        Qf = pooled(feat, [p_seen(*f) for f in facts])
        Hf = last_h(feat, [p_seen(*f) for f in facts])
        gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        Nb = len(facts)
        opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
        for _ in range(MEM_STEPS):
            idx = torch.randint(0, Nb, (min(64, Nb),), device=device)
            Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
            q = F.normalize(proj_q(Qf[idx]), -1)
            sims = q @ Kall.t() / 0.05
            vk, ik = sims.topk(min(TOPK, Nb), 1); w = torch.softmax(vk, -1)
            R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
            H = Hf[idx]; g = torch.sigmoid(gate(torch.cat([H, R], -1)))
            loss = F.cross_entropy(feat.lm_head(H + g * R).float(), gold[idx]) + F.cross_entropy(sims, idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0); opt.step()
        return mods, (Kf, Sf)

    @torch.no_grad()
    def teacher_logits(feat, mods, Kf, Sf, prompts):
        """base last-hidden + memory injection -> logits (the scaffold's answer)."""
        proj_k, proj_q, val_enc, val_dec, gate = mods
        Kall = F.normalize(proj_k(Kf), -1); Vall = val_enc(Sf)
        Q = F.normalize(proj_q(pooled(feat, prompts)), -1)
        H = last_h(feat, prompts)
        sims = Q @ Kall.t() / 0.05
        vk, ik = sims.topk(min(TOPK, Kf.shape[0]), 1); w = torch.softmax(vk, -1)
        R = val_dec((w.unsqueeze(-1) * Vall[ik]).sum(1))
        g = torch.sigmoid(gate(torch.cat([H, R], -1)))
        return feat.lm_head(H + g * R).float()

    # ---- recall helpers ----
    @torch.no_grad()
    def acc_logits(logits, gold_ids):
        return (logits.argmax(-1) == gold_ids).float().mean().item()

    @torch.no_grad()
    def model_recall(m, prompts, gold_ids, bs=128):
        ok = tot = 0
        for i in range(0, len(prompts), bs):
            e = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(device)
            pred = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == gold_ids[i:i + bs]).sum().item(); tot += pred.numel()
        return ok / tot

    @torch.no_grad()
    def teacher_recall(feat, mods, Kf, Sf, prompts, gold_ids, bs=128):
        ok = tot = 0
        for i in range(0, len(prompts), bs):
            lg = teacher_logits(feat, mods, Kf, Sf, prompts[i:i + bs])
            ok += (lg.argmax(-1) == gold_ids[i:i + bs]).sum().item(); tot += lg.shape[0]
        return ok / tot

    # ---- old-ability (in-context hops) ----
    def cap_batch(rng, names, hop, n):
        prompts, ans = [], []
        for _ in range(n):
            p, a = make(rng, names, hop); prompts.append(p); ans.append(a)
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        aid = torch.tensor([one_tok(a) for a in ans], device=device)
        return enc, aid

    @torch.no_grad()
    def hop_acc(m, names, n=96):
        rng = random.Random(777); ok = tot = 0
        for hop in HOPS:
            enc, aid = cap_batch(rng, names, hop, n)
            pred = m.lm_head(m.model(**enc, use_cache=False).last_hidden_state[:, -1]).float().argmax(-1)
            ok += (pred == aid).sum().item(); tot += aid.numel()
        return ok / tot

    @torch.no_grad()
    def anchor_agreement(base, stu):
        """fraction of anchor prompts where student's next-token == base's (headroom ~1.0)."""
        e = tok(ANCHOR_TEXT, return_tensors="pt", padding=True).to(device)
        b = base.lm_head(base.model(**e).last_hidden_state[:, -1]).argmax(-1)
        s = stu.lm_head(stu.model(**e).last_hidden_state[:, -1]).argmax(-1)
        return (b == s).float().mean().item()

    def set_trainable_top(m, k):
        for p in m.parameters():
            p.requires_grad_(False)
        for lyr in m.model.layers[-k:]:
            for p in lyr.parameters():
                p.requires_grad_(True)

    def run(seed):
        feat = load_frozen()                              # base / teacher backbone / preservation ref
        facts, rev_idx = build_facts(seed)
        names = single_tok_names(tok)
        rev_facts = [facts[i] for i in rev_idx]           # reverse-eligible subset (single-tok names)
        # gold ids
        seen_p = [p_seen(*f) for f in facts]; para_p = [p_para(*f) for f in facts]
        rev_p = [p_rev(*f) for f in rev_facts]
        val_gold = torch.tensor([one_tok(v) for (_, _, v) in facts], device=device)
        name_gold = torch.tensor([one_tok(n) for (n, _, _) in rev_facts], device=device)

        # ---- teacher scaffold ----
        mods, (Kf, Sf) = train_memory(feat, facts, seed)
        t_seen = teacher_recall(feat, mods, Kf, Sf, seen_p, val_gold)
        t_para = teacher_recall(feat, mods, Kf, Sf, para_p, val_gold)
        t_rev = teacher_recall(feat, mods, Kf, Sf, rev_p, name_gold) if rev_facts else 0.0

        # ---- student: base + grown identity layers, train only appended ----
        student = load_frozen()
        n0 = len(student.model.layers)
        grow_qwen(student, GROW); set_trainable_top(student, GROW)
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1.5e-4)

        base_hop = hop_acc(feat, names)
        rng = random.Random(seed * 7 + 1)
        # precompute teacher soft targets for seen prompts (distill source)
        for step in range(STU_STEPS):
            student.train()
            # --- fact objective: gold CE on seen+para+reverse, distill KL to teacher on seen ---
            if DISTILL_ONLY:                              # student supervised ONLY by the teacher (no gold)
                fi = [rng.randrange(len(facts)) for _ in range(Bf)]
                prompts = [seen_p[i] for i in fi]         # teacher is strong on the SEEN phrasing
                e = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    t_lg = teacher_logits(feat, mods, Kf, Sf, prompts)
                pseudo = t_lg.argmax(-1)                  # the memory's answer = the only supervision
                s_lg = student.lm_head(student.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(s_lg, pseudo) + LD * F.kl_div(
                    F.log_softmax(s_lg, -1), F.softmax(t_lg, -1), reduction="batchmean")
            else:
                phr = rng.choice(["seen", "para", "rev"]) if rev_facts else rng.choice(["seen", "para"])
                if phr == "rev":
                    ri = [rng.randrange(len(rev_facts)) for _ in range(Bf)]
                    prompts = [rev_p[i] for i in ri]; gold = name_gold[torch.tensor(ri, device=device)]
                else:
                    fi = [rng.randrange(len(facts)) for _ in range(Bf)]
                    src = seen_p if phr == "seen" else para_p
                    prompts = [src[i] for i in fi]; gold = val_gold[torch.tensor(fi, device=device)]
                e = tok(prompts, return_tensors="pt", padding=True).to(device)
                s_lg = student.lm_head(student.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
                loss = F.cross_entropy(s_lg, gold)
                if LD > 0 and phr == "seen":
                    with torch.no_grad():
                        t_lg = teacher_logits(feat, mods, Kf, Sf, prompts)
                    loss = loss + LD * F.kl_div(F.log_softmax(s_lg, -1), F.softmax(t_lg, -1),
                                                reduction="batchmean")
            # --- preservation: KL to base on anchor prompts (old ability / locality) ---
            # hopreplay ALSO puts old-task (in-context hop) prompts in the preserve batch so
            # the KL-to-base protects the reasoning computation, not only generic LM text.
            if LP > 0:
                ap = [ANCHOR_TEXT[rng.randrange(len(ANCHOR_TEXT))] for _ in range(Ba)]
                if PRESERVE == "hopreplay":
                    ap += [make(rng, names, rng.choice(HOPS))[0] for _ in range(Ba)]
                ea = tok(ap, return_tensors="pt", padding=True).to(device)
                s_a = student.lm_head(student.model(**ea, use_cache=False).last_hidden_state[:, -1]).float()
                with torch.no_grad():
                    b_a = feat.lm_head(feat.model(**ea).last_hidden_state[:, -1]).float()
                loss = loss + LP * F.kl_div(F.log_softmax(s_a, -1), F.softmax(b_a, -1),
                                            reduction="batchmean")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
        student.eval()

        # ---- eval: student WITHOUT memory (the thesis number) ----
        s_seen = model_recall(student, seen_p, val_gold)
        s_para = model_recall(student, para_p, val_gold)
        s_rev = model_recall(student, rev_p, name_gold) if rev_facts else 0.0
        b_seen = model_recall(feat, seen_p, val_gold)     # base has never seen the facts (~0 expected)
        stu_hop = hop_acc(student, names)
        anc = anchor_agreement(feat, student)             # cleaner preservation probe (headroom ~1.0)
        gap = t_seen - s_seen
        print(f"    [seed {seed}] facts={len(facts)} rev={len(rev_facts)} layers {n0}->{len(student.model.layers)}  "
              f"TEACHER(seen {t_seen:.3f} para {t_para:.3f} rev {t_rev:.3f})  "
              f"STUDENT-noMem(seen {s_seen:.3f} para {s_para:.3f} rev {s_rev:.3f})  "
              f"base-seen {b_seen:.3f}  hop base {base_hop:.3f}->stu {stu_hop:.3f}  "
              f"anchor-agree {anc:.3f}  gap {gap:.3f}", flush=True)
        del feat, student; torch.cuda.empty_cache()
        return dict(t_seen=t_seen, t_para=t_para, t_rev=t_rev, s_seen=s_seen, s_para=s_para,
                    s_rev=s_rev, b_seen=b_seen, base_hop=base_hop, stu_hop=stu_hop, anc=anc, gap=gap)

    R = [run(s) for s in range(SEEDS)]
    m = lambda k: sum(r[k] for r in R) / len(R)
    print(f"\n== mean over {SEEDS} seeds ({FACTS} facts, real Qwen, preserve={PRESERVE}) ==")
    print(f"  TEACHER (w/ memory)   : seen {m('t_seen'):.3f}  para {m('t_para'):.3f}  rev {m('t_rev'):.3f}")
    print(f"  STUDENT (NO memory)   : seen {m('s_seen'):.3f}  para {m('s_para'):.3f}  rev {m('s_rev'):.3f}   <- thesis")
    print(f"  base (novel facts)    : seen {m('b_seen'):.3f}   (sanity floor = attr value-set chance)")
    print(f"  old hop-acc           : base {m('base_hop'):.3f} -> student {m('stu_hop'):.3f}  "
          f"(drop {m('base_hop')-m('stu_hop'):+.3f})")
    print(f"  anchor-agreement      : {m('anc'):.3f}  (student==base next-token on anchors; 1.0=fully preserved)")
    print(f"  Consolidation Gap G   : {m('gap'):.3f}  (teacher_seen - student_seen; smaller = better)")
    ok = m('t_seen') > 0.95 and m('s_seen') > 0.85 and (m('base_hop') - m('stu_hop')) < 0.02
    print(f"\n  bar (teacher_seen>0.95, student_seen>0.85, hop-drop<0.02): "
          + ("PASS — consolidation preserves old ability; scale facts / add lifecycle." if ok else
             "NOT MET — inspect which term fails."))


if __name__ == "__main__":
    main()
