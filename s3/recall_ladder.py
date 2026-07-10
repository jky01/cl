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
def gen_pool(M, prompts, per_prompt, seed, qids=None):
    """sample per_prompt continuations per prompt; parse (question, answer). If qids given, tag each candidate
    with its prompt's target_qid (provenance). Returns (cands, ledger)."""
    cands = []
    flat, flat_qid = [], []
    for k, p in enumerate(prompts):
        flat += [p] * per_prompt
        flat_qid += [qids[k] if qids else None] * per_prompt
    gen_toks = 0
    gstate = torch.get_rng_state(); tok.padding_side = "left"
    try:
        for i in range(0, len(flat), 32):
            chunk = flat[i:i + 32]; cq = flat_qid[i:i + 32]
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
                    cands.append({"question": q, "sampled_answer": a, "target_qid": cq[j]})
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

def surfaced_perfact(M, cands, ans_by_qid):
    """provenance-locked: a candidate may ONLY recover its OWN target_qid (no remap). raw if its generated Q
    matches its target's signature; correct if additionally M's answer to the generated Q == target gold."""
    raw, hits = set(), []
    for c in cands:
        g = ans_by_qid.get(c["target_qid"])
        if g is None:
            continue
        if _match_qid(_sig(c["question"]), g) >= JACC_THR:
            raw.add(g["qid"]); hits.append((c, g))
    correct = set()
    if hits:
        preds = gen(M, [QT.format(q=c["question"]) for c, _ in hits])
        for (c, g), p in zip(hits, preds):
            if em(p, g["answers"]) == 1:
                correct.add(g["qid"])
    return raw, correct

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

    # shadow proxy (on the answerable audit universe)
    ds = [damage_shadow[i] for i in answerable]; dr = [damage_real[i] for i in answerable]
    dflag = [1 if damage_real[i] >= DAMAGE_MIN else 0 for i in answerable]
    shadow_proxy = dict(spearman=spearman(ds, dr), topq_enrichment=top_quartile_enrichment(ds, dflag),
                        new_shadow=new_shadow, new_real=new_real, n=len(answerable))

    # ---- cue ladder ----
    levels, ledgers = {}, {}
    def record(name, cands, ledger, perfact):
        raw, cor = (surfaced_perfact if perfact else surfaced_global)(M, cands, ans_by_qid)
        # target error (poison) among per-fact raw hits: generated Q matched but M's answer != gold
        cov = lambda S, denom: round(len(S) / max(denom, 1), 3)
        rat = lambda S, ids: len(S & ids)
        terr = None
        if perfact and raw:
            terr = round(1 - len(cor) / max(len(raw), 1), 3)
        per_age = {a: dict(ans=ages_ans.get(a, 0),
                           cor=len({q for q in cor if (final_t - ans_by_qid[q]["stream_t"]) == a}))
                   for a in sorted(ages_ans)}
        levels[name] = dict(
            n_cands=len(cands), U_raw_ans=len(raw), U_correct_ans=len(cor),
            U_raw_thr=rat(raw, thr_ids), U_correct_thr=rat(cor, thr_ids),
            cov_correct_ans=cov(cor, len(answerable)),
            cov_correct_thr=round(len(cor & thr_ids) / max(len(threatened), 1), 3),
            cov_raw_thr=round(rat(raw, thr_ids) / max(len(threatened), 1), 3),
            target_err=terr, per_age=per_age)
        ledgers[name] = ledger
        print(f"    [{name}] cands={len(cands)} corr_ans={levels[name]['cov_correct_ans']} "
              f"corr_thr={levels[name]['cov_correct_thr']} raw_thr={levels[name]['cov_raw_thr']} terr={terr}", flush=True)

    c0, l0 = gen_pool(M, [sr._FEWSHOT], GEN_N, seed); record("L0_free", c0, l0, False)
    per1 = GEN_N // len(_L1_BANK)
    c1, l1 = gen_pool(M, _L1_BANK, per1, seed + 1); record("L1_fixed_family", c1, l1, False)
    doms = sorted({q.get("src", SOURCE) for q in ans_by_qid.values()})
    dom_prompts = [f"Here are trivia questions about {d.replace('_', ' ')} topics.\n"
                   f"Question: What is a well-known fact?\nAnswer: unknown\nQuestion:" for d in doms]
    c2, l2 = gen_pool(M, dom_prompts, max(1, GEN_N // max(len(doms), 1)), seed + 2)
    record("L2_oracle_domain", c2, l2, False)
    # L3 entity-only + L4 relation (answer-redacted question); per-fact w/ provenance, O(answerable)
    aq = list(ans_by_qid.values())
    ent_cues = [_entity_cue(q["question"], q["answers"]) for q in aq]
    l3_idx = [k for k, e in enumerate(ent_cues) if e]                       # facts with a usable entity cue
    l3_prompts = [f"Write a factual question about {ent_cues[k]}.\nQuestion:" for k in l3_idx]
    c3, l3 = gen_pool(M, l3_prompts, GEN_PER, seed + 3, qids=[aq[k]["qid"] for k in l3_idx])
    l3["n_facts_with_entity"] = len(l3_idx); l3["n_answerable"] = len(aq)
    record("L3_oracle_entity", c3, l3, True)
    l4_prompts = [f"Complete this into a full factual question: {_relation_cue(q['question'], q['answers'])}\n"
                  f"Question:" for q in aq]
    c4, l4 = gen_pool(M, l4_prompts, GEN_PER, seed + 4, qids=[q["qid"] for q in aq])
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
                shadow_proxy=shadow_proxy, levels=levels, ledgers=ledgers)

def decide(seed_results):
    """apply codex frozen rules on cov_correct_thr, PER SEED (needs 2 seeds); shadow proxy gates search."""
    out = dict(n_seeds=len(seed_results))
    if any(r["n_threatened"] < POWER_MIN for r in seed_results):
        out["addressability"] = "underpowered"
        out["note"] = f"threatened n = {[r['n_threatened'] for r in seed_results]} < {POWER_MIN}"
    if len(seed_results) < 2:
        out["addressability"] = out.get("addressability", "shakedown_only")
        return out
    def lv(name, key):
        return [r["levels"].get(name, {}).get(key) for r in seed_results]
    L0, L1 = lv("L0_free", "cov_correct_thr"), lv("L1_fixed_family", "cov_correct_thr")
    L3, L4 = lv("L3_oracle_entity", "cov_correct_thr"), lv("L4_relation_redacted", "cov_correct_thr")
    L5 = lv("L5_full_question", "cov_correct_thr")
    avail = [r["availability_rate"] for r in seed_results]
    shadow_ok = all((r["shadow_proxy"].get("spearman") or 0) >= 0.30 and
                    (r["shadow_proxy"].get("topq_enrichment") or 0) >= 1.5 for r in seed_results)
    out.update(L0=L0, L1=L1, L3=L3, L4=L4, L5=L5, availability_rate=avail, shadow_proxy_pass=shadow_ok)
    if out.get("addressability") == "underpowered":
        return out
    both = lambda xs, f: all(f(x) for x in xs if x is not None) and len(xs) == 2
    if all(a < 0.2 for a in avail):
        out["addressability"] = "availability_fail"
    elif both([L1[s] - (L0[s] or 0) for s in range(2)], lambda d: d >= 0.20):
        out["addressability"] = "fixed_family_win"
    elif both(L0, lambda x: x < 0.05) and both(L1, lambda x: x < 0.05) and both(L3, lambda x: x >= 0.5):
        out["addressability"] = "search_warranted"      # needs shadow_proxy too (below)
    elif both(L3, lambda x: x < 0.5) and both(L4, lambda x: x >= 0.5):
        out["addressability"] = "narrow_basin_fail"
    else:
        out["addressability"] = "inconclusive"
    out["build_r49b"] = bool(out["addressability"] == "search_warranted" and shadow_ok)
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
