"""R50-A Stage A — shadow-query SEARCH (tomography), CORRECTED to ANSWER-damage ranking (codex qa 20:59).

Mechanism (history-free, discovery-only, NO proposer training):
  1. build a SHARED candidate question pool from generic sources: passive M_prev few-shot + FROZEN instruct
     (native chat). No proposer sees any old q/a/qid/entity/relation/title/passage/cue.
  2. for each candidate q, M_prev greedily self-answers a_M(q).
  3. teacher-force that SAME self-answer under M_prev and each shadow S; score answer-channel damage
     D_S(q) = mean_logp_Mprev(a_M|q) - mean_logp_S(a_M|q). Shadows: S_real (60-step real future write),
     S_diff (different held-out census stream = matched primary null), S_wrong (shuffled-answer adversary,
     secondary), S_identity=M_prev (D=0). RANK the shared pool by each score; historical gold used ONLY in the
     offline audit AFTER ranking.
  4. audit@K: of the top-K by a score, how many correctly address a THREATENED old fact (signature-match AND
     M_prev answers right AND proposition-equivalent (3B judge) AND acquired-lift over the ORIGINAL frozen base
     B0, plus B0 answers the generated q WRONG). Two outcomes: address_discovery (pool contains real addresses)
     vs tomography (D_real top-K enriched for realized-damaged facts vs the null shadows). Threatened is
     underpowered on census (~9) -> tomography gate reported as underpowered; answerable = address_discovery read.
Model: Qwen2.5-0.5B holder, Qwen2.5-1.5B-Instruct external proposer, Qwen2.5-3B-Instruct equivalence judge.
"""
import os, sys, json, math, random, collections
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr
from s3 import recall_ladder as rl
from s3 import margin_probe as mp

NAME    = os.environ.get("TG_MODEL", "Qwen/Qwen2.5-0.5B")
INSTRUCT = os.environ.get("TG_INSTRUCT", "Qwen/Qwen2.5-1.5B-Instruct")
STREAMS = int(os.environ.get("TG_STREAMS", 6))
ARTS    = int(os.environ.get("TG_ARTS", 5))
QA_PER  = int(os.environ.get("TG_QA", 5))
CONS_STEPS = int(os.environ.get("TG_CONS_STEPS", 400))
SHADOW_STEPS = int(os.environ.get("TG_SHADOW_STEPS", 60))
GEN_N   = int(os.environ.get("TG_GEN_N", 300))         # per generic source (passive, instruct)
TOPK    = int(os.environ.get("TG_TOPK", 60))           # ranking budget for coverage@K
SEEDS   = int(os.environ.get("TG_SEEDS", 2))
SOURCE  = os.environ.get("TG_SOURCE", "census")
DAMAGE_MIN = float(os.environ.get("TG_DAMAGE_MIN", 0.5))
OUT     = os.environ.get("TG_OUT", "tomography_result.json")
device = wb.device

wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CONS_STEPS, wb.LR = CONS_STEPS, rl.LR
wb.SOURCE = SOURCE
rl.CONS_STEPS = CONS_STEPS
tok = wb.tok
QT, em, gen, normalize = wb.QT, wb.em, wb.gen, wb.normalize
_sig = sr._sig


def _meanlogp(M, qas, key="question"):
    return [-(tb * math.log(2) / nt) if nt else -1e9 for (tb, nt) in wb.qa_answer_bits(M, qas, key=key)]


def instruct_propose(im, it, n, seed):
    """FROZEN instruct model as EXTERNAL history-free proposer via native chat template. No old keys."""
    sys_msg = ("Write a single specific factual trivia question with a short factual answer. Ask about a real "
               "person, place, organization, work, or event. Output only the question, ending with '?'.")
    outs = []
    it.padding_side = "left"
    B = 24
    for i in range(0, n, B):
        k = min(B, n - i)
        texts = [it.apply_chat_template([{"role": "system", "content": sys_msg},
                                         {"role": "user", "content": "Give me one trivia question."}],
                                        tokenize=False, add_generation_prompt=True)] * k
        e = it(texts, return_tensors="pt", padding=True).to(device)
        torch.manual_seed(seed * 7 + i)
        with torch.no_grad():
            g = im.generate(**e, max_new_tokens=40, do_sample=True, temperature=0.9, top_p=0.95,
                            pad_token_id=it.pad_token_id)
        for j in range(g.shape[0]):
            t = it.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
            if t.endswith("?") and len(t) >= 8:
                outs.append(t)
    return outs


def match_old(q, ans_by_qid):
    s = _sig(q); best, bj = None, rl.JACC_THR
    for g in ans_by_qid.values():
        j = len(s & _sig(g["question"])) / max(len(s | _sig(g["question"])), 1)
        if j >= bj:
            best, bj = g, j
    return best


def run_seed(seed, base, pm, pt, im, it):
    streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
    # grab spare articles (disjoint from the streams) for the different-stream matched null shadow
    spare = None
    for t, s in enumerate(streams):
        for ai, a in enumerate(s):
            for j, q in enumerate(a["qas"]):
                q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
    if len(streams) < 2:
        return None
    final_t = len(streams) - 1
    old = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"]]
    passages = [a["context"] for a in streams[final_t]]
    new_qa = [q for a in streams[final_t] for q in a["qas"]]
    M = wb.load_model()
    for t in range(final_t):
        M = rl.bare_write(M, [a["context"] for a in streams[t]],
                          [q for a in streams[t] for q in a["qas"]], base, CONS_STEPS, seed * 991 + t)
    M.eval()
    avail_em = [em(p, q["answers"]) for p, q in zip(gen(M, [QT.format(q=q["eval_question"]) for q in old]), old)]
    avail_b = [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(M, old, key="eval_question")]
    answerable = [q for q, e in zip(old, avail_em) if e == 1]
    ans_by_qid = {q["qid"]: q for q in answerable}
    if not answerable:
        del M; torch.cuda.empty_cache(); return None
    # threat via the REAL full write
    M_real = rl.bare_write(M, passages, new_qa, base, CONS_STEPS, seed * 13 + 777)
    real_b = [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(M_real, old, key="eval_question")]
    real_em = [em(p, q["answers"]) for p, q in zip(gen(M_real, [QT.format(q=q["eval_question"]) for q in old]), old)]
    del M_real; torch.cuda.empty_cache()
    thr_ids = {old[i]["qid"] for i in range(len(old)) if avail_em[i] == 1
               and (real_b[i] - avail_b[i] >= DAMAGE_MIN or real_em[i] == 0)}

    # ---- shadows (all 60-step, matched budget), scored on a SHARED candidate pool ----
    S_real = rl.bare_write(M, passages, new_qa, base, SHADOW_STEPS, seed * 13 + 777)
    wrong = [dict(q) for q in new_qa]; ans = [q["answers"] for q in wrong]; random.Random(seed).shuffle(ans)
    for q, a in zip(wrong, ans):
        q["answers"] = a
    S_wrong = rl.bare_write(M, passages, wrong, base, SHADOW_STEPS, seed * 13 + 999)
    # different-stream matched null: a disjoint census stream (build a fresh split, take articles not used)
    alt = wb.build_census(seed + 5000, base)
    alt_arts = [a for s in alt for a in s][:ARTS] if alt else []
    if alt_arts:
        S_diff = rl.bare_write(M, [a["context"] for a in alt_arts],
                               [q for a in alt_arts for q in a["qas"]], base, SHADOW_STEPS, seed * 13 + 555)
    else:
        S_diff = None

    # ---- shared candidate pool ----
    c_pass, _ = rl.gen_pool(M, [sr._FEWSHOT], 1, seed, total=GEN_N // 2)
    c_pass2, _ = rl.gen_pool(M, rl._L1_BANK, 0, seed + 1, total=GEN_N // 2)
    pool = list(dict.fromkeys([c["question"] for c in c_pass + c_pass2] + instruct_propose(im, it, GEN_N, seed)))
    src = {q: ("instruct" if k >= len([c for c in c_pass + c_pass2]) else "passive") for k, q in enumerate(pool)}
    # M_prev self-answers each candidate; teacher-force that self-answer under M and each shadow
    a_self = gen(M, [QT.format(q=q) for q in pool])
    cand = [{"question": q, "answers": [a if a.strip() else "<none>"]} for q, a in zip(pool, a_self)]
    lp_M = _meanlogp(M, cand)
    def dmg(S):
        if S is None:
            return [0.0] * len(cand)
        lp_S = _meanlogp(S, cand)
        return [lp_M[i] - lp_S[i] for i in range(len(cand))]
    D = dict(real=dmg(S_real), wrong=dmg(S_wrong), diff=dmg(S_diff), identity=[0.0] * len(cand))
    conf = [-x for x in lp_M]                                       # M_prev commitment (higher = more confident)
    del S_real, S_wrong
    if S_diff is not None:
        del S_diff
    torch.cuda.empty_cache()

    # ---- offline audit: for each candidate, is it a correct+equivalent+acquired address of a threatened fact? ----
    lp_B0 = _meanlogp(base, cand)
    b0_ans = gen(base, [QT.format(q=q) for q in pool])
    hitrec = {}                                                    # idx -> (qid, correct_bool, threatened_bool)
    hit_idx = [i for i in range(len(pool)) if match_old(pool[i], ans_by_qid) is not None]
    if hit_idx:
        eqv = mp.judge_equiv(pm, pt, [match_old(pool[i], ans_by_qid)["question"] for i in hit_idx],
                             [pool[i] for i in hit_idx])
        for n, i in enumerate(hit_idx):
            g = match_old(pool[i], ans_by_qid)
            m_ok = em(a_self[i], g["answers"]) == 1
            b0_wrong = em(b0_ans[i], g["answers"]) == 0
            acq = (lp_M[i] - lp_B0[i]) > 0.5
            correct = m_ok and eqv[n] and b0_wrong and acq
            hitrec[i] = dict(qid=g["qid"], correct=int(correct), thr=int(g["qid"] in thr_ids),
                             src=src[pool[i]], m_ok=int(m_ok), equiv=int(eqv[n]), b0_wrong=int(b0_wrong),
                             lift=round(lp_M[i] - lp_B0[i], 3), D_real=round(D["real"][i], 3))

    def coverage_at_k(scores):
        order = sorted(range(len(pool)), key=lambda i: -scores[i])[:TOPK]
        cor = {hitrec[i]["qid"] for i in order if i in hitrec and hitrec[i]["correct"]}
        return dict(cov_ans=round(len(cor) / max(len(answerable), 1), 3),
                    cov_thr=round(len(cor & thr_ids) / max(len(thr_ids), 1), 3), n_correct=len(cor))
    # address discovery = correct addresses anywhere in the pool; by source
    all_correct = {hitrec[i]["qid"] for i in hitrec if hitrec[i]["correct"]}
    by_src = collections.Counter(hitrec[i]["src"] for i in hitrec if hitrec[i]["correct"])
    ranks = {k: coverage_at_k(D[k]) for k in D}
    ranks["confidence"] = coverage_at_k(conf)
    ranks["random"] = coverage_at_k([random.Random(seed).random() for _ in pool])
    del M; torch.cuda.empty_cache()
    print(f"    pool={len(pool)} hits={len(hitrec)} address_discovery={len(all_correct)} bysrc={dict(by_src)} "
          f"| top{TOPK} cov_thr real/diff/wrong/conf/rand="
          f"{ranks['real']['cov_thr']}/{ranks['diff']['cov_thr']}/{ranks['wrong']['cov_thr']}/"
          f"{ranks['confidence']['cov_thr']}/{ranks['random']['cov_thr']}", flush=True)
    return dict(n_old=len(old), n_answerable=len(answerable), n_threatened=len(thr_ids), n_pool=len(pool),
                address_discovery=len(all_correct), address_by_source=dict(by_src), ranks=ranks,
                rows=[hitrec[i] for i in sorted(hitrec)])


def decide(seed_results):
    rows = [r for r in seed_results if r]
    if len(rows) < 2:
        return dict(phase="shakedown_only", n=len(rows))
    disc = [r["address_discovery"] for r in rows]
    inst = [r["address_by_source"].get("instruct", 0) for r in rows]
    pas = [r["address_by_source"].get("passive", 0) for r in rows]
    real_thr = [r["ranks"]["real"]["cov_thr"] for r in rows]
    diff_thr = [r["ranks"]["diff"]["cov_thr"] for r in rows]
    wrong_thr = [r["ranks"]["wrong"]["cov_thr"] for r in rows]
    nthr = [r["n_threatened"] for r in rows]
    v = dict(address_discovery=disc, instruct_correct=inst, passive_correct=pas, n_threatened=nthr,
             real_cov_thr=real_thr, diff_cov_thr=diff_thr, wrong_cov_thr=wrong_thr)
    # address discovery: does ANY history-free policy find real acquired addresses?
    v["address_discovery_signal"] = all(d >= 4 for d in disc)
    v["instruct_helps"] = all(inst[s] > pas[s] for s in range(2)) and any(inst)
    # tomography (threat localization) — underpowered on census; report but gate on power
    if any(n < 20 for n in nthr):
        v["tomography"] = "underpowered_threatened"
    elif all(real_thr[s] >= 0.25 and real_thr[s] >= 2 * max(diff_thr[s], wrong_thr[s], 0.01) for s in range(2)):
        v["tomography"] = "tomography_signal"
    else:
        v["tomography"] = "no_tomography_signal"
    v["verdict"] = ("address_discovery" if v["address_discovery_signal"] else
                    "instruct_policy_helps" if v["instruct_helps"] else
                    "all_history_free_null" if all(d == 0 for d in disc) else "weak")
    return v


def main():
    print(f"TOMOGRAPHY holder={NAME} instruct={INSTRUCT} ({device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cons={CONS_STEPS} shadow={SHADOW_STEPS} gen_N={GEN_N} topk={TOPK} seeds={SEEDS}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = wb.load_model()
    pm, pt = mp._load_3b()
    im = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16).to(device).eval()
    it = AutoTokenizer.from_pretrained(INSTRUCT)
    if it.pad_token is None:
        it.pad_token = it.eos_token
    out = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(seed, base, pm, pt, im, it)
        if r is None:
            print("  abort seed", flush=True); continue
        print(f"  seed {seed}: answerable={r['n_answerable']} threatened={r['n_threatened']} "
              f"address_discovery={r['address_discovery']} bysrc={r['address_by_source']}", flush=True)
        out.append(r)
        v = decide(out)
        json.dump(dict(config=dict(holder=NAME, instruct=INSTRUCT, source=SOURCE, streams=STREAMS, arts=ARTS,
                                   qa=QA_PER, cons=CONS_STEPS, shadow=SHADOW_STEPS, gen_N=GEN_N, topk=TOPK,
                                   seeds=seed + 1),
                       seeds=[{k: v2 for k, v2 in s.items() if k != "rows"} for s in out],
                       rows=[s["rows"] for s in out], decision=v), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
