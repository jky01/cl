"""R49a — addressability (cued-recall) ladder. NO generated-target training / NO consolidation of dreams.

Question (codex-converged qa 2026-07-10 15:58): a checkpoint that ANSWERS an old fact when fully cued does
NOT surface it under free generation (R48 coverage_bound). HOW MUCH cue must be supplied before a written
fact enters the model's generated QA support? The answer decides whether a write-conditioned generator
(R49b) is worth building, per codex's frozen rules:
  * fixed-family win  : L1 gives >= +0.20 unique coverage over reachable THREATENED facts (both seeds) -> use
                        a fixed prompt mixture; no differentiable search needed.
  * search warranted  : L0/L1 ~ 0 but L2/L3 recovers >= half the L5 reachable ceiling -> basins exist but the
                        online learner lacks an ADDRESS -> build R49b real-shadow contrastive decoding.
  * narrow-basin fail : only L4/L5 works -> do NOT build the big system (would rediscover the whole item).
  * availability fail : even L5 weak pre-write -> go back to acquisition/retention.

Contract: the LEARNER-side probe (L0/L1) sees only M_{t-1} + fixed generic prompts + seed + budget. The
oracle rungs (L2 domain / L3 topic-words) DELIBERATELY use historical info and are AUDIT-ONLY diagnostics,
NOT candidate mechanisms. Gold answers/aliases are redacted from every cue. Matching (question-signature
Jaccard, validated in R48) runs OFFLINE. Model: Qwen2.5-0.5B, census real text.
"""
import os, sys, json, re, copy, time, math, random, collections
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr           # reuse _sig, _FEWSHOT, generate_candidates, make_shadow, _mean_logp

NAME       = os.environ.get("RL_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS    = int(os.environ.get("RL_STREAMS", 4))
ARTS       = int(os.environ.get("RL_ARTS", 5))
QA_PER     = int(os.environ.get("RL_QA", 5))
CPT_STEPS  = int(os.environ.get("RL_CPT_STEPS", 300))
CONS_STEPS = int(os.environ.get("RL_CONS_STEPS", 400))
SHADOW_STEPS = int(os.environ.get("RL_SHADOW_STEPS", 60))
SEEDS      = int(os.environ.get("RL_SEEDS", 1))
LR         = float(os.environ.get("RL_LR", 1e-5))
GEN_N      = int(os.environ.get("RL_GEN_N", 400))         # global budget for L0/L1/L2 pools
GEN_PER    = int(os.environ.get("RL_GEN_PER", 4))         # attempts per fact for L3 (O(facts), audit-only)
GEN_TEMP   = float(os.environ.get("RL_GEN_TEMP", 0.9))
JACC_THR   = float(os.environ.get("RL_JACC_THR", 0.34))
DAMAGE_MIN = float(os.environ.get("RL_DAMAGE_MIN", 0.5))  # bits/tok rise under real write => "threatened"
OUT        = os.environ.get("RL_OUT", "recall_ladder_result.json")
SOURCE     = os.environ.get("RL_SOURCE", "census")
device = wb.device

# propagate sizes into wb (data builders + helpers)
wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CPT_STEPS, wb.CONS_STEPS, wb.LR = CPT_STEPS, CONS_STEPS, LR
wb.SOURCE = SOURCE
sr.LR = LR; sr.SHADOW_STEPS = SHADOW_STEPS; sr.GEN_TEMP = GEN_TEMP; sr.GEN_N = GEN_N
tok = wb.tok
QT, em, normalize, gen, score = wb.QT, wb.em, wb.normalize, wb.gen, wb.score
_sig = sr._sig

# ---- fixed generic prompt family (L1): broad fact/question forms, FROZEN before seeing any stream ----
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

_ANSWER_STOP = None
def _redact(text, answers):
    """strip gold answer strings/aliases + standalone digits from a cue (no answer leakage)."""
    out = text
    for a in answers:
        out = re.sub(re.escape(a), " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d+\b", " ", out)
    return " ".join(out.split())

def _topic_words(q, answers):
    """content-word signature of the gold question with answer tokens removed (L3 oracle topic cue)."""
    red = _redact(q, answers)
    ws = [w for w in _sig(red)]
    return " ".join(sorted(ws)[:6])

@torch.no_grad()
def gen_pool(M, prompts, per_prompt, seed):
    """sample per_prompt continuations for each prompt; parse (question, answer). Returns list of dicts."""
    cands = []
    gstate = torch.get_rng_state()
    tok.padding_side = "left"
    try:
        flat = [p for p in prompts for _ in range(per_prompt)]
        for i in range(0, len(flat), 32):
            chunk = flat[i:i + 32]
            e = tok(chunk, return_tensors="pt", padding=True).to(device)
            torch.manual_seed(seed * 100003 + i)
            g = M.generate(**e, max_new_tokens=40, do_sample=True, temperature=GEN_TEMP, top_p=0.95,
                           pad_token_id=tok.pad_token_id)
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
    finally:
        torch.set_rng_state(gstate)
    return cands

def surfaced(M, cands, old):
    """OFFLINE: which old facts are surfaced_raw (a candidate Q matches by signature Jaccard>=THR) and
    surfaced_correct (M's frozen answer to that generated Q == gold). Returns (raw_qids, correct_qids)."""
    raw, correct = set(), set()
    hits = []                                    # (cand, gold)
    for c in cands:
        cq = _sig(c["question"])
        best, bj = None, JACC_THR
        for g in old:
            j = len(cq & _sig(g["question"])) / max(len(cq | _sig(g["question"])), 1)
            if j >= bj:
                best, bj = g, j
        if best is not None:
            raw.add(best["qid"]); hits.append((c, best))
    if hits:                                     # surfaced_correct: ask M the GENERATED question, compare to gold
        preds = gen(M, [QT.format(q=c["question"]) for c, _ in hits])
        for (c, g), p in zip(hits, preds):
            if em(p, g["answers"]) == 1:
                correct.add(g["qid"])
    return raw, correct

def bare_write(M, passages, new_qa, base, steps, seed):
    """the common no-old-protection write (current LM + qa_ce(new) + neutral anchor), `steps` long."""
    S = copy.deepcopy(M); S.train()
    opt = torch.optim.AdamW(S.parameters(), lr=LR)
    r = random.Random(seed)
    for _ in range(steps):
        loss = wb.lm_step(S, [r.choice(passages) for _ in range(4)])
        loss = loss + wb.qa_ce(S, [r.choice(new_qa) for _ in range(8)])
        ne, nb = wb.base_anchor_logits(base, [r.choice(wb.NEUTRAL) for _ in range(8)])
        sa = S.lm_head(S.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
        loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step()
    S.eval()
    return S

def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0] * n
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return round(num / den, 3) if den else None

def run_seed(base, seed):
    streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
    for t, s in enumerate(streams):
        for ai, a in enumerate(s):
            for j, q in enumerate(a["qas"]):
                q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
    if len(streams) < 2:
        return None
    final_t = len(streams) - 1
    old = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"]]   # old universe (streams < final)
    new_qa = [q for a in streams[final_t] for q in a["qas"]]                     # the impending write's data
    passages = [a["context"] for a in streams[final_t]]

    # ---- M_prev: bare acquisition through streams 0..final_t-1 (knows the old facts) ----
    M = wb.load_model()
    for t in range(final_t):
        p = [a["context"] for a in streams[t]]
        nq = [q for a in streams[t] for q in a["qas"]]
        M = bare_write(M, p, nq, base, CONS_STEPS, seed * 991 + t)
    M.eval()

    # ---- availability + threat (real write vs shadow) ----
    def bits(model, qas):
        return [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(model, qas, key="question")]
    avail_bits = bits(M, old)
    avail_em = [em(p, q["answers"]) for p, q in zip(gen(M, [QT.format(q=q["eval_question"]) for q in old]), old)]
    M_real = bare_write(M, passages, new_qa, base, CONS_STEPS, seed * 13 + 777)
    M_shadow = bare_write(M, passages, new_qa, base, SHADOW_STEPS, seed * 13 + 777)   # same seed => aligned prefix
    real_bits = bits(M_real, old); shadow_bits = bits(M_shadow, old)
    damage_real = [real_bits[i] - avail_bits[i] for i in range(len(old))]
    damage_shadow = [shadow_bits[i] - avail_bits[i] for i in range(len(old))]
    del M_real, M_shadow; torch.cuda.empty_cache()
    # shadow gate: does the 60-step shadow predict the 400-step real damage?
    rho = spearman(damage_shadow, damage_real)
    # diagnostic universes
    answerable = [i for i in range(len(old)) if avail_em[i] == 1]
    threatened = [i for i in answerable if damage_real[i] >= DAMAGE_MIN]
    old_ans = [old[i] for i in answerable]
    old_thr = [old[i] for i in threatened]

    # ---- cue ladder (generation from the SAME M_prev) ----
    levels = {}
    def record(name, cands):
        raw, cor = surfaced(M, cands, old_ans)          # match against the ANSWERABLE universe
        thr_ids = {old[i]["qid"] for i in threatened}
        cov_ans = round(len(raw) / max(len(old_ans), 1), 3)
        cov_thr = round(len(raw & thr_ids) / max(len(old_thr), 1), 3)
        cor_ans = round(len(cor) / max(len(old_ans), 1), 3)
        levels[name] = dict(n_cands=len(cands), U_raw=len(raw), U_correct=len(cor),
                            cov_answerable=cov_ans, cov_threatened=cov_thr, correct_answerable=cor_ans)
        print(f"    [{name}] n_cands={len(cands)} U_raw={len(raw)} cov_ans={cov_ans} cov_thr={cov_thr} "
              f"correct_ans={cor_ans}", flush=True)

    record("L0_free", gen_pool(M, [sr._FEWSHOT], GEN_N, seed))
    record("L1_fixed_family", gen_pool(M, _L1_BANK, max(1, GEN_N // len(_L1_BANK)), seed + 1))
    # L2 oracle domain
    doms = sorted({q.get("src", SOURCE) for q in old_ans})
    dom_prompts = [f"Here are trivia questions about {d.replace('_', ' ')} topics.\n"
                   f"Question: What is a well-known fact?\nAnswer: unknown\nQuestion:" for d in doms]
    record("L2_oracle_domain", gen_pool(M, dom_prompts, max(1, GEN_N // max(len(dom_prompts), 1)), seed + 2))
    # L3 oracle topic-words (answer-redacted content words of the gold question); O(facts), audit-only
    l3_prompts = [f"Write a factual question that involves: {_topic_words(q['question'], q['answers'])}.\n"
                  f"Question:" for q in old_ans]
    record("L3_oracle_topic", gen_pool(M, l3_prompts, GEN_PER, seed + 3))
    # L5 answerability ceiling: ask M the held-out paraphrase directly (surfaced==correct==available on para)
    l5_raw = {old_ans[k]["qid"] for k, p in
              enumerate(gen(M, [QT.format(q=q["eval_question"]) for q in old_ans]))
              if em(p, old_ans[k]["answers"]) == 1}
    levels["L5_full_question"] = dict(n_cands=len(old_ans), U_raw=len(l5_raw), U_correct=len(l5_raw),
                                      cov_answerable=round(len(l5_raw) / max(len(old_ans), 1), 3),
                                      cov_threatened=round(len({old[i]["qid"] for i in threatened} & l5_raw) /
                                                           max(len(old_thr), 1), 3),
                                      correct_answerable=round(len(l5_raw) / max(len(old_ans), 1), 3))
    print(f"    [L5_full_question] cov_ans={levels['L5_full_question']['cov_answerable']}", flush=True)
    del M; torch.cuda.empty_cache()
    return dict(n_old=len(old), n_answerable=len(old_ans), n_threatened=len(old_thr),
                shadow_real_spearman=rho, mean_damage_real=round(sum(damage_real) / max(len(old), 1), 3),
                levels=levels)

def decide(seed_results):
    """apply codex's frozen ladder decision rules on the pooled/both-seed levels."""
    if not seed_results:
        return dict(decision="no_data")
    def lv(name, key):
        vs = [r["levels"][name][key] for r in seed_results if name in r["levels"]]
        return round(sum(vs) / len(vs), 3) if vs else None
    L0, L1 = lv("L0_free", "cov_threatened"), lv("L1_fixed_family", "cov_threatened")
    L2, L3 = lv("L2_oracle_domain", "cov_threatened"), lv("L3_oracle_topic", "cov_threatened")
    L5 = lv("L5_full_question", "cov_threatened")
    d = dict(L0=L0, L1=L1, L2=L2, L3=L3, L5=L5)
    rho_ok = all((r.get("shadow_real_spearman") or 0) >= 0.30 for r in seed_results)
    d["shadow_gate_pass"] = rho_ok
    if L5 is not None and L5 < 0.2:
        d["decision"] = "availability_fail"          # even full question can't recover pre-write
    elif L1 is not None and L0 is not None and (L1 - max(L0, 0)) >= 0.20:
        d["decision"] = "fixed_family_win"           # cheap fixed prompt mixture beats free -> no search
    elif (L0 or 0) < 0.05 and (L1 or 0) < 0.05 and L5 and max(L2 or 0, L3 or 0) >= 0.5 * L5:
        d["decision"] = "search_warranted"           # basins exist, learner lacks address -> build R49b
    elif L5 and (L2 or 0) < 0.5 * L5 and (L3 or 0) < 0.5 * L5:
        d["decision"] = "narrow_basin_fail"          # only near-full-question works -> don't build big system
    else:
        d["decision"] = "inconclusive"
    return d

def main():
    print(f"RECALL_LADDER ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cons={CONS_STEPS} shadow={SHADOW_STEPS} gen_N={GEN_N} seeds={SEEDS}", flush=True)
    base = wb.load_model()
    seed_results = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(base, seed)
        if r is None:
            print("  <2 streams — abort seed", flush=True); continue
        print(f"  seed {seed}: n_old={r['n_old']} answerable={r['n_answerable']} threatened={r['n_threatened']} "
              f"shadow_real_rho={r['shadow_real_spearman']}", flush=True)
        seed_results.append(r)
        d = decide(seed_results)
        json.dump(dict(config=dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, cons=CONS_STEPS,
                                   shadow=SHADOW_STEPS, gen_N=GEN_N, seeds=seed + 1),
                       seeds=seed_results, decision=d), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(d)}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
