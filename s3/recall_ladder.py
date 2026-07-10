"""R49a — addressability (cued-recall) ladder. NO generated-target training / NO consolidation of dreams.

Question (codex qa 2026-07-10 15:58 + review 16:13): a checkpoint that ANSWERS an old fact when fully cued
does NOT surface it under free generation (R48 coverage_bound). HOW MUCH cue must be supplied before a
written fact enters the model's generated QA support — AND is it recovered CORRECTLY? Decides whether a
write-conditioned generator (R49b) is worth building. codex frozen rules on cov_CORRECT_threatened (per seed):
  * availability_fail : availability_rate (n_answerable/n_old) < 0.2 -> back to acquisition.
  * fixed_family_win  : L1 - L0 >= +0.20 in EACH seed, no age hole, target_err<=0.10 -> fixed prompt mixture.
  * search_warranted  : L0/L1 < 0.05 AND L3_entity recovers >= 0.5 of reachable ceiling AND shadow proxy passes
                        -> build R49b real-shadow contrastive decoding.
  * narrow_basin_fail : L3 < 0.5 ceiling but L4 (near-complete, answer-redacted) >= 0.5 -> do NOT build.
  * underpowered      : threatened n < 20 in a seed -> report intervals, NO binary branch.

Two SEPARATE outputs: addressability_result (what cue reaches CORRECT old facts) and shadow_proxy_result
(does the 60-step shadow predict the 400-step real damage: pooled/per-seed Spearman>=0.30 AND top-quartile
>=1.5x enrichment). search_warranted for R49b needs BOTH. Oracle rungs (L2/L3/L4) DELIBERATELY use historical
info, are AUDIT-ONLY (O(#answerable), gold answers/aliases/numbers redacted), and are provenance-locked (a
per-fact candidate may only recover ITS OWN target — no Jaccard remap). Model: Qwen2.5-0.5B, census real text.
"""
import os, sys, json, re, copy, time, math, random, collections
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr

NAME       = os.environ.get("RL_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS    = int(os.environ.get("RL_STREAMS", 6))         # codex: 6 for a bigger old universe
ARTS       = int(os.environ.get("RL_ARTS", 5))
QA_PER     = int(os.environ.get("RL_QA", 5))
CPT_STEPS  = int(os.environ.get("RL_CPT_STEPS", 300))
CONS_STEPS = int(os.environ.get("RL_CONS_STEPS", 400))
SHADOW_STEPS = int(os.environ.get("RL_SHADOW_STEPS", 60))
SEEDS      = int(os.environ.get("RL_SEEDS", 1))
LR         = float(os.environ.get("RL_LR", 1e-5))
GEN_N      = int(os.environ.get("RL_GEN_N", 400))         # exact attempts for L0/L1/L2 global pools
GEN_PER    = int(os.environ.get("RL_GEN_PER", 4))         # attempts per fact for L3/L4 (O(answerable), audit)
GEN_TEMP   = float(os.environ.get("RL_GEN_TEMP", 0.9))
JACC_THR   = float(os.environ.get("RL_JACC_THR", 0.34))
DAMAGE_MIN = float(os.environ.get("RL_DAMAGE_MIN", 0.5))  # FROZEN: bits/tok rise under real write => threatened
POWER_MIN  = int(os.environ.get("RL_POWER_MIN", 20))
OUT        = os.environ.get("RL_OUT", "recall_ladder_result.json")
SOURCE     = os.environ.get("RL_SOURCE", "census")
device = wb.device

wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CPT_STEPS, wb.CONS_STEPS, wb.LR = CPT_STEPS, CONS_STEPS, LR
wb.SOURCE = SOURCE
sr.LR = LR; sr.GEN_TEMP = GEN_TEMP
tok = wb.tok
QT, em, normalize, gen, score = wb.QT, wb.em, wb.normalize, wb.gen, wb.score
_sig = sr._sig

_L1_BANK = [
    sr._FEWSHOT,
    ("Trivia. Give a factual question and its short answer.\n"
     "Question: In what year did World War II end?\nAnswer: 1945\nQuestion:"),
    ("Here are questions about people and who did what.\n"
     "Question: Who painted the Mona Lisa?\nAnswer: Leonardo da Vinci\nQuestion:"),
    ("Here are questions about places and organizations.\n"
     "Question: Where are the headquarters of the United Nations?\nAnswer: New York City\nQuestion:"),
    ("Here are questions about dates and quantities.\n"
     "Question: How many players are on a soccer team?\nAnswer: 11\nQuestion:"),
    ("Here are questions about titles, works, and events.\n"
     "Question: Who wrote the novel 1984?\nAnswer: George Orwell\nQuestion:"),
]
_QWORDS = set("what which who whom whose when where why how is are was were do does did".split())

def _redact(text, answers):
    out = text
    for a in answers:
        out = re.sub(re.escape(a), " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d+\b", " ", out)
    return " ".join(out.split())

def _entity_cue(q, answers):
    """best-effort entity = capitalized proper-noun tokens (drop leading Q-word), answer/number redacted."""
    red = _redact(q, answers)
    toks = red.split()
    ents = [w.strip(".,?;:") for i, w in enumerate(toks)
            if w[:1].isupper() and w.lower() not in _QWORDS and len(w) > 2]
    return " ".join(dict.fromkeys(ents))            # dedup, keep order; "" if none found

def _relation_cue(q, answers):
    """L4 near-complete cue = the whole question with the gold answer/aliases/numbers redacted."""
    return _redact(q, answers)

def _count_tokens(strs):
    return int(sum(len(tok(s, add_special_tokens=False).input_ids) for s in strs))

@torch.no_grad()
def gen_pool(M, prompts, per_prompt, seed, qids=None, cues=None, total=None):
    """sample continuations per prompt; parse (question, answer). qids -> provenance target_qid; cues ->
    the cue TEXT (for L3 relation-add guard). `total` (global pools) forces EXACTLY that many attempts by
    distributing the remainder across prompts. Returns (cands, ledger)."""
    counts = [per_prompt] * len(prompts)
    if total is not None:                            # distribute remainder so sum(counts)==total (equal budgets)
        base_c, rem = divmod(total, len(prompts))
        counts = [base_c + (1 if k < rem else 0) for k in range(len(prompts))]
    cands = []
    flat, flat_qid, flat_cue = [], [], []
    for k, p in enumerate(prompts):
        flat += [p] * counts[k]
        flat_qid += [qids[k] if qids else None] * counts[k]
        flat_cue += [cues[k] if cues else None] * counts[k]
    gen_toks = 0
    gstate = torch.get_rng_state(); tok.padding_side = "left"
    try:
        for i in range(0, len(flat), 32):
            chunk = flat[i:i + 32]; cq = flat_qid[i:i + 32]; cc = flat_cue[i:i + 32]
            e = tok(chunk, return_tensors="pt", padding=True).to(device)
            torch.manual_seed(seed * 100003 + i)
            g = M.generate(**e, max_new_tokens=40, do_sample=True, temperature=GEN_TEMP, top_p=0.95,
                           pad_token_id=tok.pad_token_id)
            gen_toks += int((g[:, e["input_ids"].shape[1]:] != tok.pad_token_id).sum().item())
            for j in range(g.shape[0]):
                txt = tok.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True)
                block = txt.split("Question:")[0]
                if "Answer:" not in block:
                    continue
                qp, ap = block.split("Answer:", 1)
                q = qp.strip().split("\n")[0].strip(); a = ap.strip().split("\n")[0].strip()
                if len(q) >= 8 and q.endswith("?") and 0 < len(a) <= 60:
                    cands.append({"question": q, "sampled_answer": a, "target_qid": cq[j], "cue": cc[j]})
    finally:
        torch.set_rng_state(gstate)
    ledger = dict(n_attempted=len(flat), n_parsed=len(cands),
                  prompt_tokens=_count_tokens(flat), gen_tokens=gen_toks)
    return cands, ledger

def _match_qid(cq_sig, target):
    return len(cq_sig & _sig(target["question"])) / max(len(cq_sig | _sig(target["question"])), 1)

def surfaced_global(M, cands, ans_by_qid):
    """global pool: best-Jaccard match to ANY answerable fact (raw), correct if M's answer to the generated Q
    == that fact's gold. Returns (raw_qids, correct_qids)."""
    raw, hits = set(), []
    for c in cands:
        s = _sig(c["question"]); best, bj = None, JACC_THR
        for g in ans_by_qid.values():
            j = _match_qid(s, g)
            if j >= bj:
                best, bj = g, j
        if best is not None:
            raw.add(best["qid"]); hits.append((c, best))
    correct = set()
    if hits:
        preds = gen(M, [QT.format(q=c["question"]) for c, _ in hits])
        for (c, g), p in zip(hits, preds):
            if em(p, g["answers"]) == 1:
                correct.add(g["qid"])
    return raw, correct

_BOILER = set("give short answer factual question involving complete this into full name person who".split())
def surfaced_perfact(M, cands, ans_by_qid, relation_guard=False):
    """provenance-locked: a candidate may ONLY recover its OWN target_qid (no remap). raw if its generated Q
    matches its target's signature. If relation_guard (L3): the generated question must ADD a target-relation
    token beyond the entity cue — A_i=sig(gen)-E_i must intersect R_i=sig(target)-E_i (codex). correct if
    additionally M's answer to the generated Q == target gold. Returns (raw, correct, audit_rows)."""
    raw, hits, audit = set(), [], []
    for c in cands:
        g = ans_by_qid.get(c["target_qid"])
        if g is None:
            continue
        gs, ts = _sig(c["question"]), _sig(g["question"])
        if _match_qid(gs, g) < JACC_THR:
            continue
        if relation_guard:
            E = _sig(c.get("cue") or "")
            R = (ts - E) - _BOILER
            A = (gs - E) - _BOILER
            if not (A & R):                          # generated Q added no target-relation token -> reject
                continue
        raw.add(g["qid"]); hits.append((c, g))
    correct = set()
    if hits:
        preds = gen(M, [QT.format(q=c["question"]) for c, _ in hits])
        for (c, g), p in zip(hits, preds):
            ok = em(p, g["answers"]) == 1
            if ok:
                correct.add(g["qid"])
            audit.append(dict(qid=g["qid"], cue=c.get("cue"), gen_q=c["question"], gold_q=g["question"],
                              pred=p, gold=g["answers"][0], correct=int(ok)))
    return raw, correct, audit

def bare_write(M, passages, new_qa, base, steps, seed):
    S = copy.deepcopy(M); S.train()
    opt = torch.optim.AdamW(S.parameters(), lr=LR); r = random.Random(seed)
    for _ in range(steps):
        loss = wb.lm_step(S, [r.choice(passages) for _ in range(4)])
        loss = loss + wb.qa_ce(S, [r.choice(new_qa) for _ in range(8)])
        ne, nb = wb.base_anchor_logits(base, [r.choice(wb.NEUTRAL) for _ in range(8)])
        sa = S.lm_head(S.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
        loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step()
    S.eval(); return S

def _ranks(v):
    n = len(v); order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
    while i < n:                                     # average ranks for ties
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r

def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return round(num / den, 3) if den else None

def top_quartile_enrichment(damage_shadow, damaged_flags):
    """fraction of real-damaged facts in the shadow's top damage quartile / overall damaged rate."""
    n = len(damage_shadow)
    if n < 4 or not any(damaged_flags):
        return None
    order = sorted(range(n), key=lambda i: -damage_shadow[i]); q = max(1, n // 4)
    topq = order[:q]
    top_rate = sum(damaged_flags[i] for i in topq) / len(topq)
    overall = sum(damaged_flags) / n
    return round(top_rate / overall, 3) if overall else None

def run_seed(base, seed):
    streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
    for t, s in enumerate(streams):
        for ai, a in enumerate(s):
            for j, q in enumerate(a["qas"]):
                q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
    if len(streams) < 2:
        return None
    final_t = len(streams) - 1
    old = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"]]
    new_qa = [q for a in streams[final_t] for q in a["qas"]]
    passages = [a["context"] for a in streams[final_t]]

    M = wb.load_model()
    for t in range(final_t):
        M = bare_write(M, [a["context"] for a in streams[t]],
                       [q for a in streams[t] for q in a["qas"]], base, CONS_STEPS, seed * 991 + t)
    M.eval()

    # availability + threat, ALL on the eval_question (paraphrase) surface
    def bits(model, qas):
        return [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(model, qas, key="eval_question")]
    def ems(model, qas):
        return [em(p, q["answers"]) for p, q in zip(gen(model, [QT.format(q=q["eval_question"]) for q in qas]), qas)]
    avail_bits = bits(M, old); avail_em = ems(M, old)
    M_real = bare_write(M, passages, new_qa, base, CONS_STEPS, seed * 13 + 777)
    M_shadow = bare_write(M, passages, new_qa, base, SHADOW_STEPS, seed * 13 + 777)
    real_bits, shadow_bits = bits(M_real, old), bits(M_shadow, old)
    real_em, shadow_em = ems(M_real, old), ems(M_shadow, old)
    new_real = round(sum(ems(M_real, new_qa)) / max(len(new_qa), 1), 3)     # shadow-vs-real sanity on new stream
    new_shadow = round(sum(ems(M_shadow, new_qa)) / max(len(new_qa), 1), 3)
    del M_real, M_shadow; torch.cuda.empty_cache()
    damage_real = [real_bits[i] - avail_bits[i] for i in range(len(old))]
    damage_shadow = [shadow_bits[i] - avail_bits[i] for i in range(len(old))]

    answerable = [i for i in range(len(old)) if avail_em[i] == 1]
    threatened = [i for i in answerable if (damage_real[i] >= DAMAGE_MIN or real_em[i] == 0)]
    thr_bits = sum(1 for i in answerable if damage_real[i] >= DAMAGE_MIN)
    thr_flip = sum(1 for i in answerable if real_em[i] == 0)
    ans_by_qid = {old[i]["qid"]: old[i] for i in answerable}
    thr_ids = {old[i]["qid"] for i in threatened}
    ages_ans = collections.Counter(final_t - old[i]["stream_t"] for i in answerable)
    ages_thr = collections.Counter(final_t - old[i]["stream_t"] for i in threatened)

    # shadow proxy (on the answerable audit universe). enrichment gates on the OR-THREATENED label (codex).
    ds = [damage_shadow[i] for i in answerable]; dr = [damage_real[i] for i in answerable]
    flag_or = [1 if (damage_real[i] >= DAMAGE_MIN or real_em[i] == 0) else 0 for i in answerable]
    flag_bits = [1 if damage_real[i] >= DAMAGE_MIN else 0 for i in answerable]
    flag_flip = [1 if real_em[i] == 0 else 0 for i in answerable]
    shadow_proxy = dict(spearman=spearman(ds, dr), topq_enrichment=top_quartile_enrichment(ds, flag_or),
                        enrich_bits=top_quartile_enrichment(ds, flag_bits),
                        enrich_flip=top_quartile_enrichment(ds, flag_flip),
                        new_shadow=new_shadow, new_real=new_real, n=len(answerable),
                        damage_pairs=[[round(ds[k], 4), round(dr[k], 4)] for k in range(len(answerable))])

    # ---- cue ladder ----
    levels, ledgers, audits = {}, {}, {}
    def record(name, cands, ledger, perfact, relation_guard=False, eligible_ids=None):
        if perfact:
            raw, cor, audit = surfaced_perfact(M, cands, ans_by_qid, relation_guard=relation_guard)
            audits[name] = audit
        else:
            raw, cor = surfaced_global(M, cands, ans_by_qid)
        rat = lambda S, ids: len(S & ids)
        terr = round(1 - len(cor) / max(len(raw), 1), 3) if (perfact and raw) else None
        # eligible = facts this rung could address (entity-having for L3); threatened denominators
        elig = set(eligible_ids) if eligible_ids is not None else set(ans_by_qid)
        elig_thr = thr_ids & elig
        per_age = {a: dict(thr=ages_thr.get(a, 0),
                           cor=len({q for q in (cor & thr_ids) if (final_t - ans_by_qid[q]["stream_t"]) == a}))
                   for a in sorted(ages_ans)}
        levels[name] = dict(
            n_cands=len(cands), U_raw_ans=len(raw), U_correct_ans=len(cor),
            U_raw_thr=rat(raw, thr_ids), U_correct_thr=rat(cor, thr_ids),
            cov_correct_ans=round(len(cor) / max(len(answerable), 1), 3),
            cov_correct_thr=round(len(cor & thr_ids) / max(len(threatened), 1), 3),         # BRANCH statistic
            cov_correct_thr_eligible=round(len(cor & elig_thr) / max(len(elig_thr), 1), 3), # model-vs-extractor
            cov_raw_thr=round(rat(raw, thr_ids) / max(len(threatened), 1), 3),
            target_err=terr, per_age=per_age)
        ledgers[name] = ledger
        print(f"    [{name}] cands={len(cands)} corr_ans={levels[name]['cov_correct_ans']} "
              f"corr_thr={levels[name]['cov_correct_thr']} raw_thr={levels[name]['cov_raw_thr']} terr={terr}", flush=True)

    c0, l0 = gen_pool(M, [sr._FEWSHOT], 1, seed, total=GEN_N); record("L0_free", c0, l0, False)
    c1, l1 = gen_pool(M, _L1_BANK, 0, seed + 1, total=GEN_N); record("L1_fixed_family", c1, l1, False)
    doms = sorted({q.get("src", SOURCE) for q in ans_by_qid.values()})
    dom_prompts = [f"Here are trivia questions about {d.replace('_', ' ')} topics.\n"
                   f"Question: What is a well-known fact?\nAnswer: unknown\nQuestion:" for d in doms]
    c2, l2 = gen_pool(M, dom_prompts, 0, seed + 2, total=GEN_N)
    record("L2_oracle_domain", c2, l2, False)
    # L3 entity-only (relation-guarded) + L4 relation (answer-redacted question); per-fact w/ provenance, O(answerable)
    aq = list(ans_by_qid.values())
    ent_cues = [_entity_cue(q["question"], q["answers"]) for q in aq]
    l3_idx = [k for k, e in enumerate(ent_cues) if e]                       # facts with a usable entity cue
    l3_prompts = [f"Write a factual question about {ent_cues[k]}.\nQuestion:" for k in l3_idx]
    c3, l3 = gen_pool(M, l3_prompts, GEN_PER, seed + 3, qids=[aq[k]["qid"] for k in l3_idx],
                      cues=[ent_cues[k] for k in l3_idx])
    l3["n_facts_with_entity"] = len(l3_idx); l3["n_answerable"] = len(aq)
    record("L3_oracle_entity", c3, l3, True, relation_guard=True,
           eligible_ids=[aq[k]["qid"] for k in l3_idx])
    l4_prompts = [f"Complete this into a full factual question: {_relation_cue(q['question'], q['answers'])}\n"
                  f"Question:" for q in aq]
    c4, l4 = gen_pool(M, l4_prompts, GEN_PER, seed + 4, qids=[q["qid"] for q in aq],
                      cues=[_relation_cue(q["question"], q["answers"]) for q in aq])
    record("L4_relation_redacted", c4, l4, True)
    # L5 ceiling = 1.0 relative to reachable (ask paraphrase) — cache, don't regen
    l5_ids = {aq[k]["qid"] for k in range(len(aq)) if avail_em[answerable[k]] == 1}
    levels["L5_full_question"] = dict(n_cands=len(aq), U_correct_ans=len(l5_ids),
                                      cov_correct_ans=round(len(l5_ids) / max(len(answerable), 1), 3),
                                      cov_correct_thr=round(len(l5_ids & thr_ids) / max(len(threatened), 1), 3))
    del M; torch.cuda.empty_cache()
    return dict(n_old=len(old), n_answerable=len(answerable), n_threatened=len(threatened),
                threatened_by_bits=thr_bits, threatened_by_flip=thr_flip,
                availability_rate=round(len(answerable) / max(len(old), 1), 3),
                ages_answerable=dict(ages_ans), ages_threatened=dict(ages_thr),
                shadow_proxy=shadow_proxy, levels=levels, ledgers=ledgers, audits=audits)

def decide(seed_results):
    """codex frozen rules on cov_correct_thr, PER SEED. phase (shakedown_only|two_seed) is separate from
    power (underpowered|adequate). shadow proxy (per-seed + pooled Spearman>=0.30, per-seed enrich>=1.5,
    new-stream sanity) gates build_r49b independently of addressability."""
    n = len(seed_results)
    power = "adequate" if all(r["n_threatened"] >= POWER_MIN for r in seed_results) else "underpowered"
    phase = "two_seed" if n >= 2 else "shakedown_only"
    out = dict(n_seeds=n, phase=phase, power=power,
               threatened_n=[r["n_threatened"] for r in seed_results])
    def lv(name, key):
        return [r["levels"].get(name, {}).get(key) for r in seed_results]
    L0, L1 = lv("L0_free", "cov_correct_thr"), lv("L1_fixed_family", "cov_correct_thr")
    L3, L4 = lv("L3_oracle_entity", "cov_correct_thr"), lv("L4_relation_redacted", "cov_correct_thr")
    L5, avail = lv("L5_full_question", "cov_correct_thr"), [r["availability_rate"] for r in seed_results]
    L1terr = lv("L1_fixed_family", "target_err")
    # shadow proxy: per-seed Spearman>=0.30 AND enrich(OR)>=1.5; new-stream EM must be comparable (|d|<=0.2)
    def sp_ok(r):
        s = r["shadow_proxy"]
        return ((s.get("spearman") or 0) >= 0.30 and (s.get("topq_enrichment") or 0) >= 1.5
                and abs((s.get("new_shadow") or 0) - (s.get("new_real") or 0)) <= 0.2)
    pooled_ds = [p[0] for r in seed_results for p in r["shadow_proxy"].get("damage_pairs", [])]
    pooled_dr = [p[1] for r in seed_results for p in r["shadow_proxy"].get("damage_pairs", [])]
    pooled_rho = spearman(pooled_ds, pooled_dr)
    shadow_ok = all(sp_ok(r) for r in seed_results) and (pooled_rho or 0) >= 0.30
    out.update(L0=L0, L1=L1, L3=L3, L3_eligible=lv("L3_oracle_entity", "cov_correct_thr_eligible"),
               L4=L4, L5=L5, availability_rate=avail, pooled_shadow_spearman=pooled_rho,
               shadow_proxy_pass=shadow_ok)
    if n < 2:
        out["addressability"] = "shakedown_only"; return out
    fin = lambda xs: [x for x in xs if x is not None]
    both = lambda xs, f: len(fin(xs)) == n and all(f(x) for x in fin(xs))     # require ALL n finite
    L1gate = (both([L1[s] - (L0[s] or 0) for s in range(n)], lambda d: d >= 0.20)
              and all((t is None or t <= 0.10) for t in L1terr))
    if any(a < 0.20 for a in avail):                                          # EVERY seed must meet floor
        out["addressability"] = "availability_fail" if all(a < 0.20 for a in avail) else "availability_unstable"
    elif L1gate:
        out["addressability"] = "fixed_family_win"
    elif both(L0, lambda x: x < 0.05) and both(L1, lambda x: x < 0.05) and both(L3, lambda x: x >= 0.5):
        out["addressability"] = "search_warranted"
    elif both(L3, lambda x: x < 0.5) and both(L4, lambda x: x >= 0.5):
        out["addressability"] = "narrow_basin_fail"
    else:
        out["addressability"] = "inconclusive"
    # power gates any POSITIVE branch; negatives (availability/narrow) can stand
    if power == "underpowered" and out["addressability"] in ("fixed_family_win", "search_warranted"):
        out["addressability"] = "underpowered"
    out["build_r49b"] = bool(out["addressability"] == "search_warranted" and shadow_ok and power == "adequate")
    return out

def main():
    print(f"RECALL_LADDER ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cons={CONS_STEPS} shadow={SHADOW_STEPS} gen_N={GEN_N} gen_per={GEN_PER} seeds={SEEDS} "
          f"damage_min={DAMAGE_MIN}", flush=True)
    base = wb.load_model(); seed_results = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(base, seed)
        if r is None:
            print("  <2 streams — abort seed", flush=True); continue
        print(f"  seed {seed}: n_old={r['n_old']} answerable={r['n_answerable']} threatened={r['n_threatened']} "
              f"(bits={r['threatened_by_bits']} flip={r['threatened_by_flip']}) avail={r['availability_rate']} "
              f"shadow_rho={r['shadow_proxy']['spearman']} enrich={r['shadow_proxy']['topq_enrichment']}", flush=True)
        seed_results.append(r)
        d = decide(seed_results)
        json.dump(dict(config=dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, cons=CONS_STEPS,
                                   shadow=SHADOW_STEPS, gen_N=GEN_N, gen_per=GEN_PER, damage_min=DAMAGE_MIN,
                                   seeds=seed + 1),
                       seeds=seed_results, decision=d), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(d)}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
