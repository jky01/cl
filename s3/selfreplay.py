"""R48-SelfReplay — checkpoint-only self-replay-into-weights on REAL TEXT.

Question (codex-converged, qa 2026-07-10 10:31..11:27): can a checkpoint replace STORED old QA replay
targets with its OWN training-time recall — generated with NO retained item ledger (qids/titles/subjects) —
and still protect old closed-book QA in one dense checkpoint? And does selecting WHICH self-generated facts
to protect (by write-susceptibility) beat protecting a uniform random subset?

Contract (STRICT; incorporates codex review qa/codex/2026-07-10.11.27.14.md):
  * Generation sees ONLY M_{t-1} + a fixed GENERIC prompt + N + seed. Never old qids/titles/subjects/QAs.
  * Two-stage targets: (1) admit iff M answers q and >=2 independently-worded views with the SAME answer
    (consensus) AND mean gold-answer logprob >= ADMIT_MIN_LOGP -> FREEZE that pre-update canonical answer as
    the CE target; (2) fragility = gold-answer bits INCREASE under ONE shared frozen shadow write (same
    current loss+anchor+LR+fixed batches as the intended write, minus old replay).
  * COMMON old-eval universe + EXOGENOUS common B_t across all arms (no survivorship confound).
  * OFFLINE auditor: match candidate->old fact by QUESTION signature (NOT the answer), THEN score frozen
    target vs gold => matched-target error (poison). Coverage is a co-primary executable GATE.
  * verdict() returns an ordered STATE (uninformative|coverage_bound|reliability_bound|self_fail|self_pass
    + fragility) — not a bare boolean. Self arms store ZERO persistent old items.

Arms (SR_ARMS): no_replay_compute_matched | stored_random_B | self_passive_random_B | self_passive_fragile_B.
Source (SR_SOURCE): cf (mechanical shakedown) | census (real-density scientific surface). Model: Qwen2.5-0.5B.
self_adversarial generator is STAGED behind a coverage miss — not in this file yet.
"""
import os, sys, json, re, copy, time, math, random, collections
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb

# ---- config (own SR_ namespace) ----
NAME       = os.environ.get("SR_MODEL", os.environ.get("WB_MODEL", "Qwen/Qwen2.5-0.5B"))
SOURCE     = os.environ.get("SR_SOURCE", "cf")
STREAMS    = int(os.environ.get("SR_STREAMS", 4))
ARTS       = int(os.environ.get("SR_ARTS", 5))
QA_PER     = int(os.environ.get("SR_QA", 5))
CPT_STEPS  = int(os.environ.get("SR_CPT_STEPS", 300))
CONS_STEPS = int(os.environ.get("SR_CONS_STEPS", 400))
SEEDS      = int(os.environ.get("SR_SEEDS", 1))
LR         = float(os.environ.get("SR_LR", 1e-5))
ARMS       = os.environ.get("SR_ARMS",
             "no_replay_compute_matched,stored_random_B,self_passive_random_B,self_passive_fragile_B").split(",")
BUDGET_FRAC  = float(os.environ.get("SR_BUDGET_FRAC", 0.5))
GEN_N        = int(os.environ.get("SR_GEN_N", 400))     # fixed candidate budget/stream (NOT O(facts))
GEN_TEMP     = float(os.environ.get("SR_GEN_TEMP", 0.9))
GEN_TOPP     = float(os.environ.get("SR_GEN_TOPP", 0.95))
GEN_VIEWS    = int(os.environ.get("SR_GEN_VIEWS", 2))   # extra paraphrase views for consensus (census: 2 => 3 total)
ADMIT_MIN_LOGP = float(os.environ.get("SR_ADMIT_MIN_LOGP", -4.0))  # min mean gold-answer logprob (nats/tok)
CONSENSUS    = os.environ.get("SR_CONSENSUS", "unanimous")         # unanimous | majority
SHADOW_STEPS = int(os.environ.get("SR_SHADOW_STEPS", 60))
JACC_THR     = float(os.environ.get("SR_JACC_THR", 0.34))
OUT        = os.environ.get("SR_OUT", "selfreplay_result.json")
DUMP_MATCH = int(os.environ.get("SR_DUMP_MATCH", 1))    # write blinded matcher-audit dump
device = wb.device

# propagate into wb BEFORE any model/tokenizer use downstream
wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CPT_STEPS, wb.CONS_STEPS, wb.LR = CPT_STEPS, CONS_STEPS, LR
wb.SOURCE = SOURCE
if NAME != wb.tok.name_or_path:                          # honor SR_MODEL if it differs from wb's import default
    from transformers import AutoTokenizer
    wb.tok = AutoTokenizer.from_pretrained(NAME)
    if wb.tok.pad_token is None:
        wb.tok.pad_token = wb.tok.eos_token
    wb.tok.padding_side = "left"
tok = wb.tok
QT = wb.QT
em, normalize, gen, score = wb.em, wb.normalize, wb.gen, wb.score

# ============================ STAGE 1: checkpoint-only candidate generation ============================
_FEWSHOT = ("Here are trivia questions and their short factual answers.\n"
            "Question: What is the capital of France?\nAnswer: Paris\n"
            "Question: Who wrote the play Romeo and Juliet?\nAnswer: William Shakespeare\n"
            "Question: What is the chemical symbol for gold?\nAnswer: Au\n"
            "Question:")

@torch.no_grad()
def generate_candidates(M, n, seed):
    """M generates n raw (question, sampled_answer) from the GENERIC seed only. No old-item input.
    Returns candidates + the actual generated-token count (ledger)."""
    cands, gen_toks = [], 0
    B = 32
    tok.padding_side = "left"
    for i in range(0, n, B):
        k = min(B, n - i)
        e = tok([_FEWSHOT] * k, return_tensors="pt", padding=True).to(device)
        torch.manual_seed(seed * 100003 + i)
        g = M.generate(**e, max_new_tokens=40, do_sample=True, temperature=GEN_TEMP, top_p=GEN_TOPP,
                       pad_token_id=tok.pad_token_id)
        gen_toks += int((g[:, e["input_ids"].shape[1]:] != tok.pad_token_id).sum().item())
        for j in range(g.shape[0]):
            txt = tok.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True)
            block = txt.split("Question:")[0]
            if "Answer:" not in block:
                continue
            q_part, a_part = block.split("Answer:", 1)
            q = q_part.strip().split("\n")[0].strip()
            a = a_part.strip().split("\n")[0].strip()
            if len(q) >= 8 and q.endswith("?") and 0 < len(a) <= 60:
                cands.append({"question": q, "sampled_answer": a})
    return cands, gen_toks

# ============================ STAGE 2: reliability admission (freeze canonical target) ============================
@torch.no_grad()
def _mean_logp(M, qas):
    """mean per-token gold-answer logprob (nats) for each qa (canonical target = qa['answers'][0])."""
    bits = wb.qa_answer_bits(M, qas, key="question")        # (total_bits, ntok)
    return [-(tb * math.log(2) / nt) if nt else -1e9 for (tb, nt) in bits]

def admit_candidates(M, cands, seed):
    """Admit iff M's greedy answer to q AND to GEN_VIEWS independently-worded views AGREE (consensus) and
    mean gold-answer logprob >= ADMIT_MIN_LOGP. Freeze M's canonical (pre-update) answer as CE target."""
    if not cands:
        return [], 0
    qs = [c["question"] for c in cands]
    a0 = gen(M, [QT.format(q=q) for q in qs])               # canonical answer a*
    view_ans = []
    for _ in range(GEN_VIEWS):
        paras = wb.gen_paraphrases(qs)                      # independent worded view (3B, data-prep, freed)
        view_ans.append(gen(M, [QT.format(q=p) for p in paras]))
    admitted = []
    for idx, (c, q, a) in enumerate(zip(cands, qs, a0)):
        if not a.strip():
            continue
        agree = sum(em(a, [view_ans[v][idx]]) == 1.0 for v in range(GEN_VIEWS))
        ok = (agree == GEN_VIEWS) if CONSENSUS == "unanimous" else (agree >= (GEN_VIEWS + 1) // 2)
        if not ok:
            continue
        rec = {"question": q, "answers": [a.strip()]}
        if _mean_logp(M, [rec])[0] < ADMIT_MIN_LOGP:
            continue
        admitted.append(rec)
    return admitted, len(cands)

# ============================ STAGE 3: shared shadow write -> fragility ============================
def make_shadow(M, base, passages, brng_state, use_qa_new_qa):
    """ONE frozen shadow = M copy + SHADOW_STEPS of the INTENDED counterfactual write (current LM + neutral
    anchor, same LR/optimizer, fixed batches replayed from brng_state), minus old replay. Same contract used
    later for adversarial search."""
    S = copy.deepcopy(M); S.train()
    opt = torch.optim.AdamW(S.parameters(), lr=LR)
    r = random.Random(brng_state)
    for _ in range(SHADOW_STEPS):
        loss = wb.lm_step(S, [r.choice(passages) for _ in range(4)])
        ne, nb = wb.base_anchor_logits(base, [r.choice(wb.NEUTRAL) for _ in range(8)])
        sa = S.lm_head(S.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
        loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step()
    S.eval()
    return S

@torch.no_grad()
def fragility_scores(M, S, admitted):
    """susceptibility = per-token gold-answer bits INCREASE M->shadow (higher => more threatened)."""
    if not admitted:
        return []
    bM = wb.qa_answer_bits(M, admitted, key="question")
    bS = wb.qa_answer_bits(S, admitted, key="question")
    return [(ts / ns if ns else 0.0) - (tm / nm if nm else 0.0)
            for (tm, nm), (ts, ns) in zip(bM, bS)]

# ============================ STAGE 4: OFFLINE coverage auditor (never feeds back) ============================
_STOP = set("what which who whom whose when where why how is are was were the a an of in on to for by with "
            "did does do has have had it its this that first name give short answer question".split())
def _sig(q):                                              # question signature = content tokens (drop stopwords)
    return set(w for w in normalize(q).split() if len(w) > 2 and w not in _STOP)

def audit_coverage(cands_raw, admitted, gold_old, final_t, B):
    """Match candidate -> old gold by QUESTION signature ONLY (answer NOT used), then score frozen target vs
    gold => matched-target error. Coverage gate: U_admit >= ceil(0.8*B); per-age coverage. Runs AFTER
    selection; NEVER feeds back."""
    def match(cands):
        pairs = []                                        # (cand, gold_hit)
        for c in cands:
            cq = _sig(c["question"])
            best, bj = None, JACC_THR
            for g in gold_old:
                gq = _sig(g["question"])
                j = len(cq & gq) / max(len(cq | gq), 1)
                if j >= bj:
                    best, bj = g, j
            if best is not None:
                pairs.append((c, best))
        return pairs
    O = len(gold_old)
    raw_pairs = match(cands_raw)
    adm_pairs = match(admitted)
    raw_qids = {g["qid"] for _, g in raw_pairs}
    adm_qids = {g["qid"] for _, g in adm_pairs}
    # matched-target error (poison): frozen candidate answer vs gold answer, only where a QUESTION match exists
    terr = [em(c["answers"][0], g["answers"]) == 0 for c, g in adm_pairs if "answers" in c]
    ages = collections.Counter(final_t - g["stream_t"] for g in gold_old)
    adm_age = collections.Counter(final_t - g["stream_t"] for g in gold_old if g["qid"] in adm_qids)
    Bc = max(1, math.ceil(0.8 * B)) if B > 0 else 0
    age_cov = {a: round(adm_age.get(a, 0) / ages[a], 3) for a in sorted(ages)}
    return dict(
        n_gold_old=O, n_raw=len(cands_raw), n_admit=len(admitted),
        U_raw=len(raw_qids), U_admit=len(adm_qids),
        C_raw=round(len(raw_qids) / max(O, 1), 3), C_admit=round(len(adm_qids) / max(O, 1), 3),
        coverage_gate=Bc, coverage_gate_pass=bool(len(adm_qids) >= Bc),
        matched_target_err=round(sum(terr) / max(len(terr), 1), 3), n_matched_for_err=len(terr),
        dup_matches=len(adm_pairs) - len(adm_qids),
        age_cov=age_cov, matched_qids=sorted(adm_qids),
        match_dump=[dict(cq=c["question"], ca=c["answers"][0], gq=g["question"],
                         ga=g["answers"][0], qid=g["qid"]) for c, g in adm_pairs[:40]] if DUMP_MATCH else [])

# ============================ lifecycle ============================
def run_selfreplay(base, streams, arm, seed):
    is_self = arm.startswith("self_")
    is_stored = arm == "stored_random_B"
    is_fragile = arm == "self_passive_fragile_B"
    M = wb.load_model()
    committed = []                      # committed-correct GOLD (stored-arm pool source + auditor). NOT read by generation.
    per_stream = []
    ledger = dict(persistent_old_items=0, gen_tokens=0, para3b_calls=0)
    n_exposed = 0                       # gold facts exposed BEFORE stream t (exogenous, common across arms)
    for t in range(len(streams)):
        arts = streams[t]
        new_qa = [q for a in arts for q in a["qas"]]
        passages = [a["context"] for a in arts]
        gold_old = [q for tt in range(t) for a in streams[tt] for q in a["qas"]]   # O_t (common universe)
        B = round(BUDGET_FRAC * n_exposed)                # EXOGENOUS common budget (not len(committed))
        brng_seed = 100003 * seed + 13 * t                # COMMON batch schedule across arms (no arm in seed)

        # ---- replay pool (per arm) ----
        pool, cov, n_admit, shortfall = [], None, 0, 0
        if t > 0 and B > 0:
            if is_stored:
                pool = wb.select_budget(committed, "random", min(B, len(committed)), seed)
                shortfall = max(0, B - len(committed))
                ledger["persistent_old_items"] = max(ledger["persistent_old_items"], len(committed))
            elif is_self:
                cands, gtok = generate_candidates(M, GEN_N, seed * 17 + t)
                ledger["gen_tokens"] += gtok
                admitted, n_raw = admit_candidates(M, cands, seed)
                ledger["para3b_calls"] += len(cands) * GEN_VIEWS
                n_admit = len(admitted)
                if is_fragile and admitted:
                    S = make_shadow(M, base, passages, brng_seed, False)
                    fr = fragility_scores(M, S, admitted)
                    del S; torch.cuda.empty_cache()
                    order = sorted(range(len(admitted)), key=lambda i: -fr[i])
                    pool = [admitted[i] for i in order[:B]]
                elif admitted:
                    pool = random.Random(f"{seed}:{t}:selfrand").sample(admitted, min(B, len(admitted)))
                shortfall = max(0, B - len(pool))
                cov = audit_coverage(cands, admitted, gold_old, t, B)    # OFFLINE audit (post-selection)

        # ---- transient scaffold S_t (diagnostic only; does NOT teach M) ----
        Sc = copy.deepcopy(M); Sc.train()
        opt = torch.optim.AdamW(Sc.parameters(), lr=LR)
        rsc = random.Random(brng_seed + 1)
        for _ in range(CPT_STEPS):
            loss = wb.lm_step(Sc, [rsc.choice(passages) for _ in range(4)])
            loss = loss + wb.qa_ce(Sc, [rsc.choice(new_qa) for _ in range(8)])   # QA-span supervision (acquisition)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(Sc.parameters(), 1.0); opt.step()
        Sc.eval(); s_pa, _ = score(Sc, new_qa, key="eval_question")

        # ---- consolidate into M_t: current LM + arm replay + neutral anchor (COMMON batch schedule) ----
        M.train(); opt = torch.optim.AdamW(M.parameters(), lr=LR)
        rc = random.Random(brng_seed + 2)
        prng = random.Random(f"{seed}:{arm}:{t}:pool")    # ONLY the pool draw depends on arm
        for _ in range(CONS_STEPS):
            loss = wb.lm_step(M, [rc.choice(passages) for _ in range(4)])
            loss = loss + wb.qa_ce(M, [rc.choice(new_qa) for _ in range(8)])       # COMMON acquisition (all arms)
            if pool:
                loss = loss + wb.qa_ce(M, [prng.choice(pool) for _ in range(8)])   # old retention (arm-specific)
            elif t > 0 and new_qa:
                loss = loss + wb.qa_ce(M, [prng.choice(new_qa) for _ in range(8)])  # compute-matched filler (arm-specific)
            ne, nb = wb.base_anchor_logits(base, [rc.choice(wb.NEUTRAL) for _ in range(8)])
            sa = M.lm_head(M.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
            loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        M.eval(); del Sc; torch.cuda.empty_cache()

        # commit committed-correct gold (auditor + stored pool; self arms never read this for generation)
        correct = [q for q, p in zip(new_qa, gen(M, [QT.format(q=q["question"]) for q in new_qa]))
                   if em(p, q["answers"]) == 1]
        committed += correct
        o_pa = round(score(M, gold_old, key="eval_question")[0], 3) if gold_old else None  # COMMON universe
        row = dict(t=t, scaffold_new_para=round(s_pa, 3), n_committed=len(correct), n_new=len(new_qa),
                   old_para_common=o_pa, B=B, n_exposed=n_exposed, n_admit=n_admit, n_pool=len(pool),
                   shortfall=shortfall, coverage=cov)
        per_stream.append(row)
        n_exposed += len(new_qa)
        cmsg = (f" C_raw/C_admit={cov['C_raw']}/{cov['C_admit']} U_admit={cov['U_admit']}>=gate{cov['coverage_gate']}?"
                f"{cov['coverage_gate_pass']} terr={cov['matched_target_err']}" if cov else "")
        print(f"      [{arm} s{seed} t{t}] scaf_para={s_pa:.2f} commit={len(correct)}/{len(new_qa)} "
              f"old_para={o_pa} B={B} pool={len(pool)}{cmsg}", flush=True)

    # ---- final closed-book eval on the COMMON old universe (streams < final_t), identical for every arm ----
    final_t = len(streams) - 1
    old_all = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"]]
    newest = [q for a in streams[final_t] for q in a["qas"]]
    matched = set()                                       # union of QUESTION-matched gold qids across streams
    for row in per_stream:
        if row.get("coverage"):
            matched.update(row["coverage"]["matched_qids"])
    def sc(qs):
        return round(score(M, qs, key="eval_question")[0], 3) if qs else None
    old_gen = [q for q in old_all if q["qid"] in matched]
    old_never = [q for q in old_all if q["qid"] not in matched]
    # compute ALL scores BEFORE deleting M
    old_para = sc(old_all)
    og_para, on_para = sc(old_gen), sc(old_never)
    newest_para = sc(newest)
    fp_em, fp_f1 = score(M, [q for s in streams for a in s for q in a["qas"]], key="eval_question")
    ages = collections.defaultdict(list)
    for q in old_all:
        ages[final_t - q["stream_t"]].append(q)
    age_para = {a: sc(qs) for a, qs in sorted(ages.items())}
    del M; torch.cuda.empty_cache()
    return dict(final_para_em=round(fp_em, 3), final_para_f1=round(fp_f1, 3),
                old_para_em=old_para, n_old=len(old_all),
                old_generated_para_em=og_para, n_old_generated=len(old_gen),
                old_never_para_em=on_para, n_old_never=len(old_never),
                newest_para_em=newest_para, age_para=age_para, ledger=ledger, per_stream=per_stream)

# ============================ verdict (ordered STATE, not a bare boolean) ============================
def verdict(summary, per_seed):
    g = lambda a, k: (summary.get(a) or {}).get(k)
    no, st = g("no_replay_compute_matched", "old_para_em"), g("stored_random_B", "old_para_em")
    sr, sf = g("self_passive_random_B", "old_para_em"), g("self_passive_fragile_B", "old_para_em")
    v = dict(state="unknown")
    if no is None or st is None:
        return v
    v["Delta_B"] = round(st - no, 3)
    # coverage gate: every decision stream+seed must pass; also no age quartile at 0
    cov_pass, cov_reason = True, None
    for arm in ("self_passive_random_B", "self_passive_fragile_B"):
        for row in per_seed.get(arm, []):
            for ps in (row.get("per_stream") or []):
                c = ps.get("coverage")
                if c and ps.get("B", 0) > 0:
                    if not c["coverage_gate_pass"]:
                        cov_pass, cov_reason = False, f"{arm} t{ps['t']} U_admit<gate"
                    if any(rate == 0.0 for rate in c["age_cov"].values()):
                        cov_pass, cov_reason = False, f"{arm} t{ps['t']} age-hole"
    if v["Delta_B"] < 0.10:
        v["state"] = "uninformative"; v["reason"] = f"Delta_B={v['Delta_B']}<0.10 (too little forgetting exposed)"
        return v
    if not cov_pass:
        v["state"] = "coverage_bound"; v["reason"] = cov_reason
        # still report recovery for context
    if sr is not None and v["Delta_B"] > 0:
        v["recovery"] = round((sr - no) / v["Delta_B"], 3)
    if sf is not None and sr is not None:
        v["fragility_gain"] = round(sf - sr, 3)
    if v["state"] == "coverage_bound":
        return v
    if v.get("recovery") is None:
        v["state"] = "unknown"; return v
    v["state"] = "self_pass" if v["recovery"] >= 0.50 else "self_fail"
    if v["state"] == "self_pass" and v.get("fragility_gain") is not None:
        v["fragility"] = "fragility_pass" if v["fragility_gain"] >= 0.05 else "fragility_null"
    return v

def main():
    print(f"SELFREPLAY ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cpt={CPT_STEPS} cons={CONS_STEPS} gen_N={GEN_N} views={GEN_VIEWS} consensus={CONSENSUS} "
          f"shadow={SHADOW_STEPS} bfrac={BUDGET_FRAC} seeds={SEEDS} arms={ARMS}", flush=True)
    base = wb.load_model()
    results = {a: [] for a in ARMS}
    for seed in range(SEEDS):
        builder = {"cf": wb.build_cf, "census": wb.build_census, "squad": wb.build_squad}.get(SOURCE)
        if builder is None:
            raise ValueError(f"unknown SR_SOURCE={SOURCE!r} (cf|census|squad)")
        streams = builder(seed, base)
        for t, s in enumerate(streams):
            for ai, a in enumerate(s):
                for j, q in enumerate(a["qas"]):
                    q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
        allqa = [q for s in streams for a in s for q in a["qas"]]
        print(f"  seed {seed}: streams={[len(s) for s in streams]} total_qa={len(allqa)}", flush=True)
        if not allqa or len(streams) < 2:
            print("  <2 streams or no QA — abort seed", flush=True); continue
        for arm in ARMS:
            t0 = time.time()
            res = run_selfreplay(base, streams, arm, seed)
            res["wall"] = round(time.time() - t0, 1)
            results[arm].append(res)
            print(f"    [{arm}] {json.dumps({k: v for k, v in res.items() if k != 'per_stream'})}", flush=True)
        # summarize
        summ, per_seed = {}, {}
        for arm in ARMS:
            rs = results[arm]
            if not rs:
                continue
            av = lambda k: (round(sum(r[k] for r in rs if r.get(k) is not None) /
                                  max(sum(1 for r in rs if r.get(k) is not None), 1), 3)
                            if any(r.get(k) is not None for r in rs) else None)
            summ[arm] = {k: av(k) for k in ("final_para_em", "old_para_em", "old_generated_para_em",
                                            "old_never_para_em", "newest_para_em")}
            per_seed[arm] = [{k: r.get(k) for k in ("old_para_em", "old_generated_para_em", "old_never_para_em",
                              "newest_para_em", "n_old", "n_old_generated", "ledger", "per_stream")} for r in rs]
        vd = verdict(summ, per_seed)
        json.dump({"config": dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, gen_N=GEN_N,
                                  views=GEN_VIEWS, consensus=CONSENSUS, budget_frac=BUDGET_FRAC,
                                  shadow_steps=SHADOW_STEPS, admit_min_logp=ADMIT_MIN_LOGP,
                                  seeds=seed + 1, arms=ARMS),
                   "summary": summ, "per_seed": per_seed, "verdict": vd}, open(OUT, "w"), indent=1)
        print(f"  VERDICT {json.dumps(vd)}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
