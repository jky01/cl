"""R44-schema_comp — MINER + AUDIT (inference-only, no training yet).

Question (codex-converged 2026-07-09): does consolidating N same-RELATION facts into weights make the
(N+1)th same-relation base-hard binding CHEAPER to write / more robust to retain? I.e. can lifetime cost
move from item-ledger O(#facts) toward reusable structure? This is "existing-schema ACTIVATION" (the
pretrained base already has latent relation structure) — the case R32/R34 do NOT kill (unlike automatic
NEW-schema acquisition, which they do). R43-ladder killed surprise-as-ROUTER on real text; the open
cost-curve question is schema COMPRESSION.

THIS FILE = stage 1 only: build a relation-labeled, base-hard, RAG-answerable, self-contained, confound-
logged dataset and AUDIT whether it can form MATCHED blocks. Training GO (codex 2026-07-09.19.35.55) only if:
>=5 relations at >=WANT usable after original+manual-paraphrase+3B-paraphrase base-hard screening, >=2
same-kind buckets where BOTH relations are >=WANT, and explicit matchable-block feasibility counts.

Source (real): `relbert/t_rex_relation_similarity` (721 relations, bulk head/tail pairs; the LAMA/T-REx
script datasets are deprecated on HF). Evidence is TEMPLATED (real subjects/relations/objects, synthetic
carrier sentence) = codex's "real-KB / templated-evidence" middle rung. `SC_SOURCE`: trex | controlled | auto.
Relations chosen to be reliably base-hard for a 0.5B (obscure creative-work->creator, product->maker);
geography/language relations are NOT base-hard (the base memorized them) — verified in the first audit.

Model: Qwen2.5-0.5B (base scoring), Qwen2.5-3B-Instruct (held-out paraphrase). Reuses WikiBridge eval contract.
"""
import os, json, re, random, collections, math, copy, time
import torch
from s3.wikibridge import (normalize, em, QT, RT, gen, qa_answer_bits, qa_ce, base_anchor_logits,
                           NEUTRAL, load_model, tok, device)
import torch.nn.functional as F
from s3.census import Instruct, RSYS

SEED = int(os.environ.get("SC_SEED", 0))
PER_REL = int(os.environ.get("SC_PER_REL", 400))     # candidate pairs pulled per relation before filtering
WANT = int(os.environ.get("SC_WANT", 24))            # target usable base-hard items per relation
SOURCE = os.environ.get("SC_SOURCE", "auto")
OUT = os.environ.get("SC_OUT", "schema_audit.json")
JOUT = re.sub(r"\.json$", "", OUT) + ".jsonl"
PARA_NAME = os.environ.get("WB_PARA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# match tolerances for the feasibility count (same-kind, other-relation neighbor within these)
TOL = dict(ans_ntok=1, subj_ntok=2, bpt=2.5, subj_bits=3.0)
MATCH_MIN = 3                                         # an item is "matchable" if >= this many same-kind neighbors

# ------------------------- relation registry: base-hard-able, 2 answer-kind buckets -------------------------
# q=closed-book question, p=manual held-out paraphrase, st=templated evidence, kind=answer bucket.
# {s}=subject(work/product) {o}=object(creator/maker). head=subject, tail=object in relbert positives.
RELATIONS = {
    "P50":  dict(kind="person", q="Who wrote {s}?",                         p="Who is the author of {s}?",            st="{s} was written by {o}."),
    "P57":  dict(kind="person", q="Who directed {s}?",                      p="Who is the director of {s}?",          st="{s} was directed by {o}."),
    "P86":  dict(kind="person", q="Who composed the music for {s}?",         p="Who is the composer of {s}?",          st="The music for {s} was composed by {o}."),
    "P58":  dict(kind="person", q="Who wrote the screenplay for {s}?",       p="Who is the screenwriter of {s}?",      st="The screenplay for {s} was written by {o}."),
    "P178": dict(kind="org",    q="Which company developed {s}?",            p="Who is the developer of {s}?",         st="{s} was developed by {o}."),
    "P176": dict(kind="org",    q="Which company manufactures {s}?",         p="Who is the manufacturer of {s}?",      st="{s} is manufactured by {o}."),
}

# ------------------------- source loaders -------------------------
def load_trex(rng):
    """real T-REx triples from relbert/t_rex_relation_similarity: rows = {relation_type, positives:[[h,t],..]}."""
    from datasets import load_dataset
    try:
        ds = load_dataset("relbert/t_rex_relation_similarity", split="train")
    except Exception as e:
        print(f"  trex FAIL: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None, None
    buf = collections.defaultdict(list)
    hits = collections.Counter()
    for r in ds:
        pid = r["relation_type"]
        if pid not in RELATIONS:
            continue
        for pair in r["positives"]:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            sub, obj = str(pair[0]).strip(), str(pair[1]).strip()
            hits[pid] += 1
            if not sub or not obj or normalize(obj) in normalize(sub):
                continue
            if not (1 <= len(tok(obj, add_special_tokens=False).input_ids) <= 6):
                continue
            buf[pid].append(dict(sub=sub, obj=obj, evi=None))
    for p in buf:
        rng.shuffle(buf[p]); buf[p] = buf[p][:PER_REL]
    print(f"  SOURCE=trex predicate-hits={dict(hits)} kept={ {p: len(buf[p]) for p in buf} }", flush=True)
    return (buf, "trex") if sum(len(v) for v in buf.values()) >= 3 * len(RELATIONS) else (None, None)

_CONS = "bcdfghjklmnpqrstvwz"; _VOW = "aeiou"
def _coin(rng, nsyl):
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) + (rng.choice(_CONS) if rng.random() < 0.4 else "")
                   for _ in range(nsyl)).capitalize()
def load_controlled(rng):
    """controlled-fake smoke: real templates, invented subject+object (guaranteed base-hard). SECONDARY only."""
    buf = collections.defaultdict(list)
    for p in RELATIONS:
        for _ in range(PER_REL):
            s = _coin(rng, rng.choice([2, 3])) + " " + _coin(rng, 2)
            o = _coin(rng, 2) + " " + _coin(rng, rng.choice([2, 3]))
            buf[p].append(dict(sub=s, obj=o, evi=None))
    print(f"  SOURCE=controlled { {p: len(buf[p]) for p in buf} }", flush=True)
    return buf, "controlled"

# ------------------------- subject familiarity proxy: NLL in a NEUTRAL carrier -------------------------
@torch.no_grad()
def subject_bits(base, subs):
    res = []
    pre = "Here is some information about"
    for i in range(0, len(subs), 16):
        chunk = subs[i:i + 16]
        full = [pre + " " + s + "." for s in chunk]
        tok.padding_side = "right"
        e = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        pl = len(tok(pre).input_ids)
        logp = torch.log_softmax(base(**e, use_cache=False).logits[:, :-1].float(), -1)
        labels = e["input_ids"][:, 1:].clone()
        labels[e["attention_mask"][:, 1:] == 0] = -100
        for r in range(len(chunk)):
            lab = labels[r].clone(); lab[:max(0, pl - 1)] = -100
            m = lab != -100
            if bool(m.any()):
                nll = -logp[r][m].gather(1, lab[m][:, None]).squeeze(1)
                res.append(round((nll.sum() / math.log(2)).item() / int(m.sum().item()), 3))
            else:
                res.append(0.0)
    tok.padding_side = "left"
    return res

def _emlist(base, prompts, items):
    return [em(x, q["answers"]) for x, q in zip(gen(base, prompts), items)]

def audit_main():
    rng = random.Random(4400 + SEED)
    print(f"SCHEMA_COMP miner (base=Qwen2.5-0.5B, {device}) rels={list(RELATIONS)} want={WANT} src={SOURCE}", flush=True)
    base = load_model()
    used = None; buf = None
    if SOURCE in ("auto", "trex"):
        buf, used = load_trex(rng)
        if buf is None and SOURCE == "trex":
            print("  SC_SOURCE=trex but T-REx unavailable -> ABORT (no silent fallback)", flush=True)
            json.dump(dict(meta=dict(source="trex", GO_for_training=False, error="trex_unavailable")), open(OUT, "w"))
            print("[done]", flush=True); return
    if buf is None:
        buf, used = (load_controlled(rng) if SOURCE in ("auto", "controlled") else (None, None))

    items = []
    for pid, rows in buf.items():
        R = RELATIONS[pid]
        for r in rows:
            s, o = r["sub"], r["obj"]
            evi = r.get("evi") or R["st"].format(s=s, o=o)     # templated evidence (real-KB middle rung)
            items.append(dict(pid=pid, kind=R["kind"], sub=s, answers=[o],
                              question=R["q"].format(s=s), eval_question=R["p"].format(s=s),
                              context=evi, src=used))
    print(f"  candidate items: {len(items)}", flush=True)

    # base-hard screen on ORIGINAL + MANUAL paraphrase; RAG-answerable with evidence
    e1 = _emlist(base, [QT.format(q=q["question"]) for q in items], items)
    ep = _emlist(base, [QT.format(q=q["eval_question"]) for q in items], items)
    rg = _emlist(base, [RT.format(c=q["context"], q=q["question"]) for q in items], items)
    for q, a, b, c in zip(items, e1, ep, rg):
        q["base_em_orig"], q["base_em_para"], q["rag_em"] = a, b, c
    hard = [q for q in items if q["base_em_orig"] == 0 and q["base_em_para"] == 0 and q["rag_em"] == 1]
    print(f"  base-hard(orig+manual-para) & RAG-answerable: {len(hard)}/{len(items)}", flush=True)

    # 3B held-out paraphrase, then SCREEN it too (codex: the held-out surface must itself be base-hard)
    inst = Instruct()
    paras = inst.chat([(RSYS, q["question"]) for q in hard], max_new=40, bs=16)
    for q, pp in zip(hard, paras):
        q["eval_question_3b"] = (pp.split("\n")[0].strip() or q["eval_question"])
    inst.free()
    b3 = _emlist(base, [QT.format(q=q["eval_question_3b"]) for q in hard], hard)
    r3 = _emlist(base, [RT.format(c=q["context"], q=q["eval_question_3b"]) for q in hard], hard)
    for q, a, b in zip(hard, b3, r3):
        q["base_em_3b"], q["rag_em_3b"] = a, b

    # confounds: answer bpt (orig+manual-para), subject familiarity, token counts
    bo = qa_answer_bits(base, hard, "question"); bp = qa_answer_bits(base, hard, "eval_question")
    sb = subject_bits(base, [q["sub"] for q in hard])
    for q, o_, p_, s_ in zip(hard, bo, bp, sb):
        q["ans_ntok"] = o_[1]; q["bpt_orig"] = round(o_[0] / max(o_[1], 1), 3)
        q["bpt_para"] = round(p_[0] / max(p_[1], 1), 3); q["subj_bits"] = s_
        q["subj_ntok"] = len(tok(q["sub"], add_special_tokens=False).input_ids)

    # leakage + dedup; usable requires ALL held-out surfaces base-hard
    seen = set()
    for q in hard:
        na = normalize(q["answers"][0])
        q["ans_in_q"] = int(na in normalize(q["question"]) or na in normalize(q["eval_question_3b"]))
        key = (q["pid"], normalize(q["sub"]))
        q["dup"] = int(key in seen); seen.add(key)
    usable = [q for q in hard if not q["dup"] and not q["ans_in_q"]
              and q["base_em_3b"] == 0 and q["rag_em"] == 1]
    byrel = collections.defaultdict(list)
    for q in usable:
        byrel[q["pid"]].append(q)

    # ------- matchable-block feasibility (codex #3): same-kind, OTHER-relation neighbor within TOL -------
    for pid, qs in byrel.items():
        others = [x for x in usable if x["kind"] == RELATIONS[pid]["kind"] and x["pid"] != pid]
        for q in qs:
            nb = sum(1 for x in others
                     if abs(x["ans_ntok"] - q["ans_ntok"]) <= TOL["ans_ntok"]
                     and abs(x["subj_ntok"] - q["subj_ntok"]) <= TOL["subj_ntok"]
                     and abs(x["bpt_para"] - q["bpt_para"]) <= TOL["bpt"]
                     and abs(x["subj_bits"] - q["subj_bits"]) <= TOL["subj_bits"])
            q["match_nb"] = nb; q["matchable"] = int(nb >= MATCH_MIN)

    # ------- AUDIT -------
    def dist(qs, k):
        v = sorted(q[k] for q in qs)
        return None if not v else dict(n=len(v), mean=round(sum(v) / len(v), 2),
                                       p10=v[len(v) // 10], p50=v[len(v) // 2], p90=v[min(len(v) - 1, 9 * len(v) // 10)])
    audit = {}
    for pid in RELATIONS:
        qs = byrel.get(pid, [])
        audit[pid] = dict(kind=RELATIONS[pid]["kind"], n_usable=len(qs),
                          n_matchable=sum(q["matchable"] for q in qs),
                          ans_bpt=dist(qs, "bpt_para"), subj_bits=dist(qs, "subj_bits"),
                          ans_ntok=dist(qs, "ans_ntok"), subj_ntok=dist(qs, "subj_ntok"))
    # GO: relation counts use MATCHABLE usable (codex bug #1: don't count half-filled neighbors)
    ok_rels = [p for p in RELATIONS if audit[p]["n_matchable"] >= WANT]
    kinds = collections.defaultdict(list)
    for p in ok_rels:
        kinds[RELATIONS[p]["kind"]].append(p)
    ok_buckets = {k: v for k, v in kinds.items() if len(v) >= 2}
    GO = len(ok_rels) >= 5 and len(ok_buckets) >= 2
    meta = dict(source=used, source_mode=SOURCE, n_candidate=len(items), n_hard=len(hard),
                n_usable=len(usable), want=WANT, tol=TOL, match_min=MATCH_MIN,
                relations_ge_want_matchable=ok_rels, same_kind_buckets_ge2=ok_buckets,
                GO_for_training=GO,
                note="real-KB/templated-evidence" if used == "trex" else used)
    json.dump(dict(meta=meta, per_relation=audit,
                   items={q["pid"] + ":" + q["sub"]: {k: v for k, v in q.items() if k != "context"} for q in usable}),
              open(OUT, "w"), indent=1)
    with open(JOUT, "w") as f:
        for q in usable:
            f.write(json.dumps({k: v for k, v in q.items() if k != "context"}) + "\n")
    print(f"  AUDIT src={used} cand={len(items)} hard={len(hard)} usable={len(usable)}", flush=True)
    for pid in RELATIONS:
        a = audit[pid]
        print(f"    {pid}[{a['kind']:6s}] usable={a['n_usable']:3d} matchable={a['n_matchable']:3d} "
              f"ans_bpt={a['ans_bpt']} subj_bits={a['subj_bits']}", flush=True)
    print(f"  relations>=WANT(matchable): {ok_rels}  buckets>=2: {ok_buckets}  GO={GO}", flush=True)
    print("[done]", flush=True)

# ============================= STAGE 2: PRIME-then-BIND training ladder =============================
# codex-approved 2026-07-09.19.49: isolate the dose-response — prior same-relation density N vs marginal
# cost to BIND a held-out same-relation target at fixed budget. Arms differ ONLY in prime relation-coherence
# (matched count + confounds). Eval surface = MANUAL held-out paraphrase (primary; 3B = secondary robustness).
AUDIT_IN = os.environ.get("SC_AUDIT_IN", "docs/cloud_results/r44_audit.jsonl")
N_LEVELS = [int(x) for x in os.environ.get("SC_N", "0,4,8,16").split(",")]
B_SCHED = sorted(int(x) for x in os.environ.get("SC_B", "1,2,4").split(","))   # cumulative bind steps to measure at
T_TARGET = int(os.environ.get("SC_T", 8))            # held-out targets per relation
PRIME_STEPS = int(os.environ.get("SC_PRIME_STEPS", 12))  # per-prime-fact consolidation steps (calibrated in pilot)
LR = float(os.environ.get("SC_LR", 1e-5))
TRAIN_SEEDS = int(os.environ.get("SC_TRAIN_SEEDS", 2))
ARMS = os.environ.get("SC_ARMS", "schema_commit,item_k_matched,schema_shuffle_same_kind").split(",")

def load_audit(rng):
    """read frozen audit jsonl; per relation split disjoint TARGET (held-out) and PRIME pools."""
    rows = [json.loads(l) for l in open(AUDIT_IN)]
    byrel = collections.defaultdict(list)
    for r in rows:
        byrel[r["pid"]].append(r)
    targets, primes = {}, {}
    for pid, qs in byrel.items():
        qs = [q for q in qs if q.get("matchable")]
        rng.shuffle(qs)
        targets[pid] = qs[:T_TARGET]
        primes[pid] = qs[T_TARGET:]                  # disjoint prime pool
    return targets, primes

def _match(cands, ref, k, rng):
    """pick k prime facts from cands whose confounds match the ref target set's distribution (nearest)."""
    out = []
    pool = list(cands); rng.shuffle(pool)
    for _ in range(k):
        t = rng.choice(ref)
        pool.sort(key=lambda q: (abs(q["ans_ntok"] - t["ans_ntok"]) + abs(q["bpt_para"] - t["bpt_para"]) / 3
                                 + abs(q["subj_bits"] - t["subj_bits"]) / 3))
        if pool:
            out.append(pool.pop(0))
    return out

def select_primes(pid, arm, N, targets, primes, rng):
    if N <= 0:
        return []
    ref = targets[pid]
    if arm == "schema_commit":
        pool = list(primes[pid])                     # same relation
    elif arm == "item_k_matched":
        pool = [q for p, qs in primes.items() if p != pid for q in qs]   # any other relation
    elif arm == "schema_shuffle_same_kind":
        kind = RELATIONS[pid]["kind"]
        pool = [q for p, qs in primes.items() if p != pid and RELATIONS[p]["kind"] == kind for q in qs]
    else:
        raise ValueError(arm)
    return _match(pool, ref, min(N, len(pool)), rng)

def consolidate(M, facts, base, steps, rng):
    """train QA-CE on facts (original question->answer) + neutral base anchor. This IS the prime write."""
    if not facts:
        return
    M.train(); opt = torch.optim.AdamW(M.parameters(), lr=LR)
    for _ in range(steps):
        loss = qa_ce(M, [rng.choice(facts) for _ in range(min(8, len(facts)))])
        with torch.no_grad():                        # frozen base anchor target (codex: no backward graph thru base)
            ne, nb = base_anchor_logits(base, [rng.choice(NEUTRAL) for _ in range(8)])
        sa = M.lm_head(M.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
        loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
    M.eval()

@torch.no_grad()
def para_em(M, qs, key):
    if not qs:
        return None
    preds = gen(M, [QT.format(q=q[key]) for q in qs])
    return round(sum(em(p, q["answers"]) for p, q in zip(preds, qs)) / len(qs), 3)

def bind_progressive(M, targets, base, rng):
    """BIND the T targets with QA-CE, measuring held-out paraphrase EM at each cumulative budget in B_SCHED."""
    M.train(); opt = torch.optim.AdamW(M.parameters(), lr=LR)
    out = {}; step = 0
    for b in B_SCHED:
        while step < b:
            loss = qa_ce(M, targets)                 # one full-batch pass over the T targets = 1 step
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
            step += 1
        M.eval()
        out[b] = dict(tgt_manual=para_em(M, targets, "eval_question"),
                      tgt_orig=para_em(M, targets, "question"),
                      tgt_3b=para_em(M, [q for q in targets if q.get("rag_em_3b")], "eval_question_3b"))
        M.train()
    M.eval()
    return out

def train_main():
    t0 = time.time()
    print(f"SCHEMA_COMP train (prime-then-bind) N={N_LEVELS} B={B_SCHED} T={T_TARGET} "
          f"prime_steps={PRIME_STEPS} arms={ARMS} seeds={TRAIN_SEEDS}", flush=True)
    base = load_model(); base.requires_grad_(False)   # anchor teacher is frozen
    results = []
    for seed in range(TRAIN_SEEDS):
        rng = random.Random(9400 + seed)
        targets, primes = load_audit(rng)
        rels = [p for p in RELATIONS if len(targets.get(p, [])) >= T_TARGET and primes.get(p)]
        print(f"  seed {seed}: relations usable={rels}", flush=True)
        for pid in rels:
            R = RELATIONS[pid]
            # commit-check budget: PRIME must actually write the primes before we trust the density effect
            for arm in ARMS:
                for N in N_LEVELS:
                    if N == 0 and arm != ARMS[0]:
                        continue                      # N=0 floor is arm-independent; run once
                    M = load_model()
                    pr = select_primes(pid, arm, N, targets, primes, rng)
                    consolidate(M, pr, base, PRIME_STEPS * len(pr), rng)   # steps scale with N so all primes commit
                    prime_committed = para_em(M, pr, "question") if pr else None
                    prime_para_pre = para_em(M, pr, "eval_question") if pr else None
                    binds = bind_progressive(M, targets[pid], base, rng)
                    prime_para_post = para_em(M, pr, "eval_question") if pr else None
                    del M; torch.cuda.empty_cache()
                    tg = targets[pid]
                    tgt_ntok_sum = sum(q["ans_ntok"] + 1 for q in tg)   # +1 stop token (answer-target-token unit)
                    def mdelta(k):                                     # nearest-neighbor match quality (codex diag)
                        return None if not pr else round(sum(min(abs(p[k] - t[k]) for t in tg) for p in pr) / len(pr), 3)
                    row = dict(seed=seed, pid=pid, kind=R["kind"], arm=(arm if N > 0 else "floor"),
                               N=N, n_prime=len(pr), prime_committed=prime_committed,
                               prime_para_pre=prime_para_pre, prime_para_post=prime_para_post,
                               tgt_ans_ntok=round(sum(q["ans_ntok"] for q in tg) / T_TARGET, 2),
                               n_tgt_3b=sum(1 for q in tg if q.get("rag_em_3b")),
                               tok_updates={b: b * tgt_ntok_sum for b in B_SCHED},   # answer-target-token updates
                               match_delta=dict(ans_ntok=mdelta("ans_ntok"), bpt_para=mdelta("bpt_para"),
                                                subj_bits=mdelta("subj_bits"), subj_ntok=mdelta("subj_ntok")),
                               tgt_subs=[q["sub"] for q in tg], prime_subs=[p["sub"] for p in pr],
                               binds=binds)
                    results.append(row)
                    b_last = binds[B_SCHED[-1]]
                    print(f"    [s{seed} {pid} {row['arm']:22s} N={N:2d}] prime_commit={prime_committed} "
                          f"tgt_manual@B={ {b: binds[b]['tgt_manual'] for b in B_SCHED} } "
                          f"prime_post={prime_para_post}", flush=True)
        json.dump(dict(config=dict(N=N_LEVELS, B=B_SCHED, T=T_TARGET, prime_steps=PRIME_STEPS,
                                   arms=ARMS, seeds=seed + 1, relations=list(RELATIONS)),
                       rows=results), open(OUT, "w"), indent=1)
    # slope summary: mean target manual-para EM vs N per arm (at max B), per bucket
    print("  -- slope: mean tgt manual-para EM vs N (at B={}) per arm/bucket --".format(B_SCHED[-1]), flush=True)
    for kind in sorted(set(R["kind"] for R in RELATIONS.values())):
        for arm in ["floor"] + ARMS:
            for N in N_LEVELS:
                vals = [r["binds"][B_SCHED[-1]]["tgt_manual"] for r in results
                        if r["kind"] == kind and r["arm"] == arm and r["N"] == N
                        and r["binds"][B_SCHED[-1]]["tgt_manual"] is not None]
                if vals:
                    print(f"     {kind:6s} {arm:22s} N={N:2d}: {round(sum(vals)/len(vals),3)} (n={len(vals)})", flush=True)
    print(f"[done] wall={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    (train_main if os.environ.get("SC_STAGE", "audit") == "train" else audit_main)()
