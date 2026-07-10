"""R50-scale — is the self-ADDRESSING wall (R48/R49a) model-scale-sensitive? (codex qa 2026-07-10 18:55.)

R48/R49a showed a 0.5B checkpoint can't self-address (generate a proposition-preserving query about its own
fact) so passive self-replay is coverage-bound. If that dissolves at larger scale, much of our self-replay
pessimism is a 0.5B artifact. This scout crosses SIZE x history-free proposition SELECTION under a corrected
contract:
  * JOINT base-hard screen: audit only facts BOTH frozen bases fail closed-book on the full held-out question,
    so a larger model's prior knowledge cannot masquerade as better self-addressing.
  * acquire the SAME facts at each size (bare continued-PT + QA span CE + neutral anchor), then audit
    history-free L0_free / L1_fixed_family CORRECT proposition coverage over the ANSWERABLE and THREATENED sets.
  * report per-size FULL populations AND the INTERSECTION (facts BOTH sizes learned+answer) = the causal scale
    comparison. Normalize coverage by generation ATTEMPTS/tokens (a bigger model must not win on more search).
Not "no training": each size must acquire the facts (cost reported). Model: Qwen2.5-0.5B vs SS_LARGE (default 1.5B;
set 3B if the pod has memory). Qwen2.5 family shares one tokenizer, so wb helpers work for both sizes.
"""
import os, sys, json, math, random, collections
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr
from s3 import recall_ladder as rl

SMALL   = os.environ.get("SS_SMALL", "Qwen/Qwen2.5-0.5B")
LARGE   = os.environ.get("SS_LARGE", "Qwen/Qwen2.5-1.5B")
STREAMS = int(os.environ.get("SS_STREAMS", 6))
ARTS    = int(os.environ.get("SS_ARTS", 5))
QA_PER  = int(os.environ.get("SS_QA", 5))
CONS_STEPS = int(os.environ.get("SS_CONS_STEPS", 400))
GEN_N   = int(os.environ.get("SS_GEN_N", 400))
SEEDS   = int(os.environ.get("SS_SEEDS", 2))
SOURCE  = os.environ.get("SS_SOURCE", "census")
OUT     = os.environ.get("SS_OUT", "scale_scout_result.json")
DAMAGE_MIN = float(os.environ.get("SS_DAMAGE_MIN", 0.5))
device = wb.device

wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CONS_STEPS, wb.LR = CONS_STEPS, rl.LR
wb.SOURCE = SOURCE
rl.CONS_STEPS = CONS_STEPS
tok = wb.tok
QT, em, gen, normalize = wb.QT, wb.em, wb.gen, wb.normalize


def load(name):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()


def _emset(model, qas, key="eval_question"):
    return [em(p, q["answers"]) for p, q in zip(gen(model, [QT.format(q=q[key]) for q in qas]), qas)]


def _bits(model, qas, key="eval_question"):
    return [tb / nt if nt else 0.0 for (tb, nt) in wb.qa_answer_bits(model, qas, key=key)]


def run_size(model_name, base, streams, final_t, hard_qids, seed):
    """acquire the facts at this size, then audit history-free L0/L1 correct proposition coverage."""
    M = load(model_name)
    for t in range(final_t):
        M = rl.bare_write(M, [a["context"] for a in streams[t]],
                          [q for a in streams[t] for q in a["qas"]], base, CONS_STEPS, seed * 991 + t)
    M.eval()
    old = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"] if q["qid"] in hard_qids]
    if not old:
        del M; torch.cuda.empty_cache(); return None
    avail_em = _emset(M, old); avail_b = _bits(M, old)
    answerable = [q for q, e in zip(old, avail_em) if e == 1]
    ans_by_qid = {q["qid"]: q for q in answerable}
    # threat via the real (final-stream) write
    M_real = rl.bare_write(M, [a["context"] for a in streams[final_t]],
                           [q for a in streams[final_t] for q in a["qas"]], base, CONS_STEPS, seed * 13 + 777)
    real_b = _bits(M_real, old); real_em = _emset(M_real, old)
    del M_real; torch.cuda.empty_cache()
    dmg = {old[i]["qid"]: real_b[i] - avail_b[i] for i in range(len(old))}
    remq = {old[i]["qid"]: real_em[i] for i in range(len(old))}
    threatened = {q["qid"] for q in answerable if (dmg[q["qid"]] >= DAMAGE_MIN or remq[q["qid"]] == 0)}
    # history-free generation -> correct proposition coverage
    c0, l0 = rl.gen_pool(M, [sr._FEWSHOT], 1, seed, total=GEN_N)
    raw0, cor0 = rl.surfaced_global(M, c0, ans_by_qid)
    c1, l1 = rl.gen_pool(M, rl._L1_BANK, 0, seed + 1, total=GEN_N)
    raw1, cor1 = rl.surfaced_global(M, c1, ans_by_qid)
    del M; torch.cuda.empty_cache()
    cor = cor0 | cor1
    gen_tokens = l0["gen_tokens"] + l1["gen_tokens"]
    return dict(model=model_name, n_hard=len(old), n_answerable=len(answerable), n_threatened=len(threatened),
                answerable=sorted(ans_by_qid), threatened=sorted(threatened),
                l0_correct=sorted(cor0), l1_correct=sorted(cor1),
                cov_correct_ans=round(len(cor) / max(len(answerable), 1), 3),
                cov_correct_thr=round(len(cor & threatened) / max(len(threatened), 1), 3),
                unique_correct=len(cor), gen_tokens=gen_tokens,
                correct_per_1k_tokens=round(1000 * len(cor) / max(gen_tokens, 1), 3))


def run_seed(seed, base_s, base_l):
    streams = wb.build_census(seed, base_s) if SOURCE == "census" else wb.build_cf(seed, base_s)
    for t, s in enumerate(streams):
        for ai, a in enumerate(s):
            for j, q in enumerate(a["qas"]):
                q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
    if len(streams) < 2:
        return None
    final_t = len(streams) - 1
    old = [q for tt in range(final_t) for a in streams[tt] for q in a["qas"]]
    # JOINT base-hard screen: both frozen bases must fail the full held-out question closed-book
    hs = _emset(base_s, old); hl = _emset(base_l, old)
    hard_qids = {old[i]["qid"] for i in range(len(old)) if hs[i] == 0 and hl[i] == 0}
    print(f"    joint-hard {len(hard_qids)}/{len(old)} old facts", flush=True)
    rs = run_size(SMALL, base_s, streams, final_t, hard_qids, seed)
    rl_ = run_size(LARGE, base_l, streams, final_t, hard_qids, seed)
    if rs is None or rl_ is None:
        return None
    # intersection = facts BOTH sizes made answerable (+ both threatened) — the causal scale comparison
    both_ans = set(rs["answerable"]) & set(rl_["answerable"])
    both_thr = set(rs["threatened"]) & set(rl_["threatened"])
    def icov(r, S):
        return round(len((set(r["l0_correct"]) | set(r["l1_correct"])) & S) / max(len(S), 1), 3)
    inter = dict(n_both_answerable=len(both_ans), n_both_threatened=len(both_thr),
                 small_cov_thr=icov(rs, both_thr), large_cov_thr=icov(rl_, both_thr),
                 small_cov_ans=icov(rs, both_ans), large_cov_ans=icov(rl_, both_ans))
    return dict(n_hard=len(hard_qids), small=rs, large=rl_, intersection=inter)


def decide(seed_results):
    """scale-sensitive PASS: on the shared answerable/threatened intersection, LARGE improves correct
    history-free proposition coverage by >=+0.15 abs (or 4x), BOTH seeds, still after token normalization."""
    rows = [r for r in seed_results if r]
    if len(rows) < 2:
        return dict(phase="shakedown_only", n=len(rows))
    deltas_thr = [r["intersection"]["large_cov_thr"] - r["intersection"]["small_cov_thr"] for r in rows]
    deltas_ans = [r["intersection"]["large_cov_ans"] - r["intersection"]["small_cov_ans"] for r in rows]
    n_thr = [r["intersection"]["n_both_threatened"] for r in rows]
    v = dict(delta_cov_thr=deltas_thr, delta_cov_ans=deltas_ans, n_both_threatened=n_thr,
             small_cov_thr=[r["intersection"]["small_cov_thr"] for r in rows],
             large_cov_thr=[r["intersection"]["large_cov_thr"] for r in rows])
    if any(n < 15 for n in n_thr):
        v["verdict"] = "underpowered_threatened"          # fall back to answerable-intersection signal
    if all(d >= 0.15 for d in deltas_ans) and all(r["large"]["cov_correct_ans"] > 0 for r in rows):
        v["verdict"] = v.get("verdict_pre", "scale_sensitive")
        v["scale_sensitive_on_answerable"] = True
    elif all(abs(d) < 0.05 for d in deltas_ans):
        v.setdefault("verdict", "scale_insensitive")
    else:
        v.setdefault("verdict", "inconclusive")
    return v


def main():
    print(f"SCALE_SCOUT small={SMALL} large={LARGE} ({device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} "
          f"cons={CONS_STEPS} gen_N={GEN_N} seeds={SEEDS}", flush=True)
    base_s = load(SMALL); base_l = load(LARGE)
    out = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(seed, base_s, base_l)
        if r is None:
            print("  abort seed", flush=True); continue
        out.append(r)
        s, l, it = r["small"], r["large"], r["intersection"]
        print(f"    SMALL: hard={s['n_hard']} ans={s['n_answerable']} thr={s['n_threatened']} "
              f"cov_ans={s['cov_correct_ans']} cov_thr={s['cov_correct_thr']} corr/1k={s['correct_per_1k_tokens']}", flush=True)
        print(f"    LARGE: hard={l['n_hard']} ans={l['n_answerable']} thr={l['n_threatened']} "
              f"cov_ans={l['cov_correct_ans']} cov_thr={l['cov_correct_thr']} corr/1k={l['correct_per_1k_tokens']}", flush=True)
        print(f"    INTERSECTION both_ans={it['n_both_answerable']} both_thr={it['n_both_threatened']} "
              f"cov_thr small/large={it['small_cov_thr']}/{it['large_cov_thr']} "
              f"cov_ans small/large={it['small_cov_ans']}/{it['large_cov_ans']}", flush=True)
        v = decide(out)
        json.dump(dict(config=dict(small=SMALL, large=LARGE, source=SOURCE, streams=STREAMS, arts=ARTS,
                                   qa=QA_PER, cons=CONS_STEPS, gen_N=GEN_N, seeds=seed + 1),
                       seeds=out, decision=v), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
