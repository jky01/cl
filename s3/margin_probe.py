"""R49a-margin — cheap discriminator: is "recognition without recall" a DECODING competition or an ENCODING
(surface-sensitivity) deficit? (codex qa 2026-07-10 17:10, point 2.)

The R49a ladder is BEHAVIORAL: L3/L4 self-generated questions match a fact lexically but the model answers them
WRONG, while the full question (L5) is answered right. That is consistent with (a) a narrow surface basin
(ENCODING), or (b) the gold answer being present but losing the greedy decode to a stronger pretrained prior
(DECODING competition). This probe scores the GOLD-ANSWER margin under query variants to tell them apart:

  For each ANSWERABLE fact (full-question EM==1), over variants {full paraphrase, original, self-gen matched,
  token-deletion 75/50/25%}: record greedy EM, gold mean-logprob, and the model's OWN greedy-answer mean-logprob.
  Discriminator on the failure set (full EM==1 but variant EM==0):
    * DECODING  : gold_logp stays HIGH (close to the greedy competitor's logp) -> answer is accessible, decode lost.
    * ENCODING  : gold_logp COLLAPSES vs the full question -> that query form does not address the fact.

NO training beyond bare acquisition (reuses s3/recall_ladder.bare_write). Auditor-only. Model: Qwen2.5-0.5B census.
"""
import os, sys, json, math, random, collections
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr
from s3 import recall_ladder as rl

NAME    = os.environ.get("MP_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS = int(os.environ.get("MP_STREAMS", 6))
ARTS    = int(os.environ.get("MP_ARTS", 5))
QA_PER  = int(os.environ.get("MP_QA", 5))
CONS_STEPS = int(os.environ.get("MP_CONS_STEPS", 400))
GEN_PER = int(os.environ.get("MP_GEN_PER", 6))        # self-gen attempts per fact (entity cue)
SEEDS   = int(os.environ.get("MP_SEEDS", 1))
OUT     = os.environ.get("MP_OUT", "margin_probe_result.json")
SOURCE  = os.environ.get("MP_SOURCE", "census")
device = wb.device

wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CONS_STEPS, wb.LR = CONS_STEPS, rl.LR
wb.SOURCE = SOURCE
rl.CONS_STEPS = CONS_STEPS
tok = wb.tok
QT, em, gen, normalize = wb.QT, wb.em, wb.gen, wb.normalize
_sig = sr._sig


def _meanlogp(M, qas, key):
    """mean per-token logprob (nats) of qas[i]['answers'][0] under the closed-book template on qas[i][key]."""
    bits = wb.qa_answer_bits(M, qas, key=key)         # (total_bits, ntok)
    return [-(tb * math.log(2) / nt) if nt else -1e9 for (tb, nt) in bits]


def _variant_stats(M, facts, key):
    """for a list of {question:<variant>, answers:[gold]} return per-fact (greedy_em, gold_mlp, greedy_ans, greedy_mlp)."""
    gold_mlp = _meanlogp(M, facts, key)
    greedy = gen(M, [QT.format(q=f[key]) for f in facts])
    ems = [em(g, f["answers"]) for g, f in zip(greedy, facts)]
    gq = [dict(**{key: f[key]}, answers=[g if g.strip() else "<none>"]) for g, f in zip(greedy, facts)]
    greedy_mlp = _meanlogp(M, gq, key)
    return list(zip(ems, gold_mlp, greedy, greedy_mlp))


def _content(s):
    return [w for w in normalize(s).split() if len(w) > 2 and w not in rl._QWORDS]


def _delete_frac(q, keep_frac, rng):
    """keep the leading Q-word + keep_frac of the content tokens (deterministic order)."""
    toks = q.split()
    content_idx = [i for i, w in enumerate(toks) if len(normalize(w)) > 2 and normalize(w) not in rl._QWORDS]
    k = int(round(keep_frac * len(content_idx)))
    keep = set(content_idx[:k]) | {i for i, w in enumerate(toks) if normalize(w) in rl._QWORDS}
    out = [w for i, w in enumerate(toks) if i in keep]
    return " ".join(out) if out else q


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
    M = wb.load_model()
    for t in range(final_t):
        M = rl.bare_write(M, [a["context"] for a in streams[t]],
                          [q for a in streams[t] for q in a["qas"]], base, CONS_STEPS, seed * 991 + t)
    M.eval()

    # answerable = full held-out paraphrase EM == 1
    full = [dict(question=q["eval_question"], answers=q["answers"], qid=q["qid"],
                 orig=q["question"]) for q in old]
    fs = _variant_stats(M, full, "question")
    ans = [i for i, s in enumerate(fs) if s[0] == 1]
    print(f"    answerable {len(ans)}/{len(old)}", flush=True)
    if not ans:
        del M; torch.cuda.empty_cache(); return dict(n_old=len(old), n_answerable=0)

    aq = [full[i] for i in ans]
    # self-generated question per fact from the entity cue (best signature match to the gold question)
    gold_by_qid = {q["qid"]: q for q in old}
    ent = [rl._entity_cue(gold_by_qid[f["qid"]]["question"], f["answers"]) for f in aq]
    idx = [k for k, e in enumerate(ent) if e]
    prompts = [f"Write a factual question about {ent[k]}.\nQuestion:" for k in idx]
    cands, _ = rl.gen_pool(M, prompts, GEN_PER, seed + 3, qids=[aq[k]["qid"] for k in idx],
                           cues=[ent[k] for k in idx])
    best_gen = {}                                     # qid -> best signature-matched self-gen question
    for c in cands:
        g = gold_by_qid.get(c["target_qid"])
        if g is None:
            continue
        j = len(_sig(c["question"]) & _sig(g["question"])) / max(len(_sig(c["question"]) | _sig(g["question"])), 1)
        if j >= rl.JACC_THR and j > best_gen.get(c["target_qid"], (0, None))[0]:
            best_gen[c["target_qid"]] = (j, c["question"])

    # assemble variant tables (only over answerable facts)
    rng = random.Random(seed)
    variants = {}
    variants["full"] = [dict(question=f["question"], answers=f["answers"], qid=f["qid"]) for f in aq]
    variants["orig"] = [dict(question=f["orig"], answers=f["answers"], qid=f["qid"]) for f in aq]
    variants["del75"] = [dict(question=_delete_frac(f["question"], 0.75, rng), answers=f["answers"], qid=f["qid"]) for f in aq]
    variants["del50"] = [dict(question=_delete_frac(f["question"], 0.50, rng), answers=f["answers"], qid=f["qid"]) for f in aq]
    variants["del25"] = [dict(question=_delete_frac(f["question"], 0.25, rng), answers=f["answers"], qid=f["qid"]) for f in aq]
    gen_facts = [dict(question=best_gen[f["qid"]][1], answers=f["answers"], qid=f["qid"])
                 for f in aq if f["qid"] in best_gen]
    stats = {name: _variant_stats(M, v, "question") for name, v in variants.items()}
    gen_stats = _variant_stats(M, gen_facts, "question") if gen_facts else []
    del M; torch.cuda.empty_cache()

    # aggregate: for each variant, EM rate, mean gold_mlp, and (on the failure set: full-correct but variant-wrong)
    # gold_mlp vs greedy_mlp gap -> decoding(gap~0) vs encoding(gold_mlp collapses).
    full_mlp = {aq[k]["qid"]: stats["full"][k][1] for k in range(len(aq))}
    def summarize(name, st, keys):
        em_rate = round(sum(s[0] for s in st) / max(len(st), 1), 3)
        gold_mean = round(sum(s[1] for s in st) / max(len(st), 1), 3)
        fail = [(keys[i], st[i]) for i in range(len(st)) if st[i][0] == 0]     # variant wrong
        # decoding signature: gold_mlp close to greedy_mlp (model nearly emits gold); encoding: gold_mlp << full
        dec_gap = [st_i[3] - st_i[1] for _, st_i in fail]                       # greedy_mlp - gold_mlp (>=0)
        drop = [full_mlp[q] - st_i[1] for q, st_i in fail if q in full_mlp]     # how far gold_mlp fell from full
        return dict(n=len(st), em=em_rate, gold_mlp_mean=gold_mean, n_fail=len(fail),
                    fail_greedy_minus_gold=round(sum(dec_gap) / max(len(dec_gap), 1), 3) if dec_gap else None,
                    fail_gold_drop_from_full=round(sum(drop) / max(len(drop), 1), 3) if drop else None)
    summ = {name: summarize(name, st, [aq[k]["qid"] for k in range(len(aq))]) for name, st in stats.items()}
    if gen_stats:
        summ["gen_entity"] = summarize("gen_entity", gen_stats, [f["qid"] for f in gen_facts])
    return dict(n_old=len(old), n_answerable=len(aq), n_gen_matched=len(gen_facts), summary=summ)


def classify(seed_results):
    """decoding if, on the self-gen (or del50) failure set, gold_mlp stays close to greedy (small greedy-gold gap
    AND small drop from full); encoding if gold_mlp collapses from full."""
    rows = [r for r in seed_results if r.get("n_answerable")]
    if not rows:
        return dict(verdict="no_data")
    def avg(name, key):
        vs = [r["summary"][name][key] for r in rows if name in r["summary"] and r["summary"][name].get(key) is not None]
        return round(sum(vs) / len(vs), 3) if vs else None
    probe = "gen_entity" if any("gen_entity" in r["summary"] for r in rows) else "del50"
    gap = avg(probe, "fail_greedy_minus_gold")        # greedy_mlp - gold_mlp on failures (small => decoding)
    drop = avg(probe, "fail_gold_drop_from_full")     # full_mlp - variant gold_mlp (small => decoding)
    v = dict(probe=probe, fail_greedy_minus_gold=gap, fail_gold_drop_from_full=drop,
             em_curve={n: avg(n, "em") for n in ("full", "orig", "del75", "del50", "del25")})
    if gap is None or drop is None:
        v["verdict"] = "inconclusive"
    elif drop <= 0.5 and gap <= 0.5:
        v["verdict"] = "decoding_competition"         # gold accessible, loses the greedy decode
    elif drop >= 1.0:
        v["verdict"] = "encoding_surface_sensitive"   # gold logprob collapses under the variant query form
    else:
        v["verdict"] = "mixed"
    return v


def main():
    print(f"MARGIN_PROBE ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} cons={CONS_STEPS} "
          f"gen_per={GEN_PER} seeds={SEEDS}", flush=True)
    base = wb.load_model(); res = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(base, seed)
        if r is None:
            print("  <2 streams — abort", flush=True); continue
        res.append(r)
        for name, s in (r.get("summary") or {}).items():
            print(f"    [{name}] em={s['em']} gold_mlp={s['gold_mlp_mean']} n_fail={s['n_fail']} "
                  f"greedy-gold={s['fail_greedy_minus_gold']} drop_from_full={s['fail_gold_drop_from_full']}", flush=True)
        v = classify(res)
        json.dump(dict(config=dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, cons=CONS_STEPS,
                                   gen_per=GEN_PER, seeds=seed + 1), seeds=res, verdict=v), open(OUT, "w"), indent=1)
        print(f"  VERDICT {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
