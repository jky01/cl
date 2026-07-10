"""R50-A Stage A — shadow-query SEARCH (tomography): can a history-free proposal policy FIND a query that
elicits a THREATENED old proposition, without any old key? (codex qa 2026-07-10 20:36 + 20:49.)

R48/R49a/R50-scale: a checkpoint answers old facts when fully addressed but never self-proposes them; scale
(0.5B->1.5B) doesn't help. The missing operation is ADDRESS INVERSION: find q s.t. M_prev(q)=a is a real
acquired+threatened fact. Discovery-only, NO training of the proposer. Proposal policies (arms):
  * passive_base   : generic few-shot generation from M_prev (the R48/R49a coverage floor).
  * instruct_static: a FROZEN Qwen2.5-Instruct EXTERNAL proposer (native chat template), M_prev answers. Tests
    whether a better generic language policy enumerates useful addresses (the "is it a base-model artifact" Q).
  * shadow_contrastive     : contrastive-decode from M_prev steered by (M_prev - real 60-step future-write shadow
    S): boosts tokens M_prev commits to AND the impending write suppresses -> steer toward THREATENED facts.
  * wrong_shadow_contrastive: same but S trained on SHUFFLED current data (control; must lose >=half coverage).
Contract: proposer sees ONLY M_prev/S/base + generic seeds; NO old q/a/qid/entity/relation/cue. Hidden gold =
OFFLINE audit only. Every hit needs: signature-match to an old fact AND M_prev answers it right AND proposition-
equivalent (3B judge) AND positive acquired-lift over the frozen base B (not generic prior). Model: Qwen2.5-0.5B.
"""
import os, sys, json, math, random, collections, copy
import torch, torch.nn.functional as F

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
GEN_N   = int(os.environ.get("TG_GEN_N", 400))
LAM     = float(os.environ.get("TG_LAM", 1.0))         # contrastive strength
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


def _parse(txt):
    block = txt.split("Question:")[0]
    if "Answer:" not in block:
        return None
    q = block.split("Answer:", 1)[0].strip().split("\n")[0].strip()
    return q if (len(q) >= 8 and q.endswith("?")) else None


@torch.no_grad()
def contrastive_gen(M, S, n, seed, lam):
    """sample n questions from M steered by (logp_M - logp_S): boosts what M commits to and the future write S
    suppresses. Manual token loop (no cache; short seqs). Returns list of question strings."""
    outs = []
    B = 24
    tok.padding_side = "left"
    enc = tok([sr._FEWSHOT], return_tensors="pt").to(device)
    base_ids = enc["input_ids"]
    for i in range(0, n, B):
        k = min(B, n - i)
        ids = base_ids.repeat(k, 1)
        am = torch.ones_like(ids)
        torch.manual_seed(seed * 100003 + i)
        for _step in range(40):
            lm = M(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
            ls = S(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
            logp_m = F.log_softmax(lm, -1); logp_s = F.log_softmax(ls, -1)
            adj = logp_m + lam * (logp_m - logp_s)
            # restrict to M's top-p support so contrastive can't emit gibberish (codex: fluency-constrained)
            probs = F.softmax(logp_m / 0.9, -1)
            sp, si = probs.sort(descending=True)
            mask = (sp.cumsum(-1) - sp) > 0.95
            keep = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, si, ~mask)
            adj = adj.masked_fill(~keep, -1e9)
            nxt = torch.distributions.Categorical(logits=adj / 0.9).sample()
            ids = torch.cat([ids, nxt[:, None]], 1); am = torch.cat([am, torch.ones(k, 1, device=device, dtype=am.dtype)], 1)
        for j in range(k):
            q = _parse(tok.decode(ids[j, base_ids.shape[1]:], skip_special_tokens=True))
            if q:
                outs.append(q)
    return outs


def instruct_propose(im, it, n, seed):
    """FROZEN instruct model as EXTERNAL history-free proposer via its native chat template. No old keys."""
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


def audit(M, base, pm, pt, questions, ans_by_qid, thr_ids):
    """for a set of proposed questions: signature-match to an answerable old fact, M answers it right,
    proposition-equivalent (3B judge), positive acquired-lift over base. Returns coverage stats + rows."""
    # signature match (best) to any answerable fact
    hits = []
    for q in set(questions):
        s = _sig(q); best, bj = None, rl.JACC_THR
        for g in ans_by_qid.values():
            j = len(s & _sig(g["question"])) / max(len(s | _sig(g["question"])), 1)
            if j >= bj:
                best, bj = g, j
        if best is not None:
            hits.append((q, best))
    if not hits:
        return dict(n_q=len(set(questions)), n_hit=0, correct=[], n_correct=0), []
    # M answers the proposed q; equivalence judge; acquired lift
    preds = gen(M, [QT.format(q=q) for q, _ in hits])
    eqv = mp.judge_equiv(pm, pt, [g["question"] for _, g in hits], [q for q, _ in hits])
    qas = [{"question": q, "answers": g["answers"]} for q, g in hits]
    lp_M = _meanlogp(M, qas); lp_B = _meanlogp(base, qas)
    correct, rows = set(), []
    for idx, ((q, g), p) in enumerate(zip(hits, preds)):
        ok = em(p, g["answers"]) == 1 and eqv[idx] and (lp_M[idx] - lp_B[idx]) > 0
        if ok:
            correct.add(g["qid"])
        rows.append(dict(qid=g["qid"], gen_q=q, gold_q=g["question"], pred=p, gold=g["answers"][0],
                         em=int(em(p, g["answers"]) == 1), equiv=int(eqv[idx]),
                         lift=round(lp_M[idx] - lp_B[idx], 3), threatened=int(g["qid"] in thr_ids)))
    return dict(n_q=len(set(questions)), n_hit=len(hits), correct=sorted(correct), n_correct=len(correct)), rows


def run_seed(seed, base, pm, pt, im, it):
    streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
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
    # threat via real write; real + wrong shadows for contrastive search
    S_real = rl.bare_write(M, passages, new_qa, base, SHADOW_STEPS, seed * 13 + 777)
    M_real = rl.bare_write(M, passages, new_qa, base, CONS_STEPS, seed * 13 + 777)
    real_b = [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(M_real, old, key="eval_question")]
    real_em = [em(p, q["answers"]) for p, q in zip(gen(M_real, [QT.format(q=q["eval_question"]) for q in old]), old)]
    del M_real; torch.cuda.empty_cache()
    thr_ids = {old[i]["qid"] for i in range(len(old)) if avail_em[i] == 1
               and (real_b[i] - avail_b[i] >= DAMAGE_MIN or real_em[i] == 0)}
    # wrong shadow: shuffle which passage/QA pairs go together (break the true fact structure)
    wrong_qa = [dict(q) for q in new_qa]; ans = [q["answers"] for q in wrong_qa]
    random.Random(seed).shuffle(ans)
    for q, a in zip(wrong_qa, ans):
        q["answers"] = a
    S_wrong = rl.bare_write(M, passages, wrong_qa, base, SHADOW_STEPS, seed * 13 + 999)

    arms = {}
    # passive floor
    c_pass, _ = rl.gen_pool(M, [sr._FEWSHOT], 1, seed, total=GEN_N // 2)
    c_pass2, _ = rl.gen_pool(M, rl._L1_BANK, 0, seed + 1, total=GEN_N // 2)
    arms["passive_base"] = [c["question"] for c in c_pass + c_pass2]
    arms["instruct_static"] = instruct_propose(im, it, GEN_N, seed)
    arms["shadow_contrastive"] = contrastive_gen(M, S_real, GEN_N, seed, LAM)
    arms["wrong_shadow_contrastive"] = contrastive_gen(M, S_wrong, GEN_N, seed, LAM)
    del S_real, S_wrong; torch.cuda.empty_cache()

    out, allrows = {}, {}
    for name, qs in arms.items():
        stat, rows = audit(M, base, pm, pt, qs, ans_by_qid, thr_ids)
        cor = set(stat["correct"])
        stat["cov_correct_ans"] = round(len(cor) / max(len(answerable), 1), 3)
        stat["cov_correct_thr"] = round(len(cor & thr_ids) / max(len(thr_ids), 1), 3)
        out[name] = stat; allrows[name] = rows
        print(f"    [{name:24s}] n_q={stat['n_q']} hit={stat['n_hit']} correct={stat['n_correct']} "
              f"cov_ans={stat['cov_correct_ans']} cov_thr={stat['cov_correct_thr']}", flush=True)
    del M; torch.cuda.empty_cache()
    return dict(n_old=len(old), n_answerable=len(answerable), n_threatened=len(thr_ids), arms=out, rows=allrows)


def decide(seed_results):
    rows = [r for r in seed_results if r]
    if len(rows) < 2:
        return dict(phase="shakedown_only", n=len(rows))
    def arm(a, k):
        return [r["arms"][a][k] for r in rows]
    pas = arm("passive_base", "cov_correct_ans")
    ins = arm("instruct_static", "cov_correct_ans")
    sha = arm("shadow_contrastive", "cov_correct_ans")
    wro = arm("wrong_shadow_contrastive", "cov_correct_ans")
    v = dict(passive_cov_ans=pas, instruct_cov_ans=ins, shadow_cov_ans=sha, wrong_shadow_cov_ans=wro,
             n_threatened=[r["n_threatened"] for r in rows],
             shadow_cov_thr=arm("shadow_contrastive", "cov_correct_thr"),
             passive_cov_thr=arm("passive_base", "cov_correct_thr"))
    # instruct policy signal (codex gate): correct answerable cov >= 0.15 both seeds & clears passive
    v["instruct_policy_signal"] = all(ins[s] >= 0.15 and ins[s] > pas[s] for s in range(2))
    # shadow tomography signal on answerable (threatened underpowered on census): beats passive AND wrong-shadow
    v["shadow_signal"] = all(sha[s] >= 0.15 and sha[s] >= 2 * max(pas[s], 0.01)
                             and sha[s] >= 2 * max(wro[s], 0.01) for s in range(2))
    if v["shadow_signal"]:
        v["verdict"] = "tomography_promising"
    elif v["instruct_policy_signal"]:
        v["verdict"] = "instruct_policy_helps"
    elif all(max(ins[s], sha[s]) < 0.05 for s in range(2)):
        v["verdict"] = "all_history_free_null"     # base-model confound rejected; addressing wall holds
    else:
        v["verdict"] = "inconclusive"
    return v


def main():
    print(f"TOMOGRAPHY holder={NAME} instruct={INSTRUCT} ({device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cons={CONS_STEPS} shadow={SHADOW_STEPS} gen_N={GEN_N} lam={LAM} seeds={SEEDS}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = wb.load_model()
    pm, pt = mp._load_3b()                                     # 3B judge for proposition equivalence
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
        print(f"  seed {seed}: answerable={r['n_answerable']} threatened={r['n_threatened']}", flush=True)
        out.append(r)
        v = decide(out)
        dump = dict(config=dict(holder=NAME, instruct=INSTRUCT, source=SOURCE, streams=STREAMS, arts=ARTS,
                                qa=QA_PER, cons=CONS_STEPS, shadow=SHADOW_STEPS, gen_N=GEN_N, lam=LAM, seeds=seed + 1),
                    seeds=[{k: v2 for k, v2 in s.items() if k != "rows"} for s in out], decision=v)
        json.dump(dump, open(OUT, "w"), indent=1)
        json.dump({"rows": [s["rows"] for s in out]}, open(OUT.replace(".json", ".rows.json"), "w"))
        print(f"  DECISION {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
