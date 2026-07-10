"""R49a-margin (HARDENED) — clean discriminator: is "recognition without recall" a DECODING competition or an
ENCODING (surface-sensitivity) deficit? (codex qa 2026-07-10 17:10 + 17:30.)

The scout showed gold-answer logprob collapses under reformed queries, BUT the reformed queries were not
proposition-equivalence-audited and the deletion path removed information, not just surface. This hardened
version fixes codex's four blockers:
  1. persist per-(seed,qid,variant) RAW rows (medians/quartiles, not just means).
  2. proposition-equivalence AUDIT (3B judge) — the primary discriminator uses ONLY audited-equivalent variants;
     wrong-entity swap is a CALIBRATION control that SHOULD drop gold access.
  3. controlled variants: full paraphrase (ref) | 2 diverse audited paraphrases | self-gen entity (audited) |
     entity-preserving shortened (audited) | swap_entity (control, should drop).
  4. base-model LIFT: lift = logp_Mprev(gold|v) - logp_base(gold|v); drop = logp_Mprev(gold|full) - (gold|v).
     + the model's own greedy answer AND the base greedy answer (is the winner a base prior lure?).

Discriminator on the AUDITED-EQUIVALENT failure set (Mprev answers full but not the equivalent variant):
  * DECODING competition  : gold lift stays ~full-level (gold still supported) but greedy emits a base-prior lure.
  * ENCODING surface-sens : gold lift collapses toward the base floor under an EQUIVALENT reformulation.
swap_entity must drop (validity). No training beyond bare acquisition. Model: Qwen2.5-0.5B census.
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
GEN_PER = int(os.environ.get("MP_GEN_PER", 6))
N_PARA  = int(os.environ.get("MP_N_PARA", 2))         # diverse audited paraphrases per fact
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
WB_PARA = wb.WB_PARA


def _meanlogp(M, qas, key="question"):
    return [-(tb * math.log(2) / nt) if nt else -1e9 for (tb, nt) in wb.qa_answer_bits(M, qas, key=key)]


# ---- 3B instruct: generate diverse paraphrases + a shortened form + judge proposition-equivalence ----
def _load_3b():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pm = AutoModelForCausalLM.from_pretrained(WB_PARA, dtype=torch.bfloat16).to(device).eval()
    pt = AutoTokenizer.from_pretrained(WB_PARA)
    if pt.pad_token is None:
        pt.pad_token = pt.eos_token
    pt.padding_side = "left"
    return pm, pt


@torch.no_grad()
def _chat(pm, pt, sys_prompts, users, max_new=40, sample=False):
    out = []
    for i in range(0, len(users), 16):
        chunk_u = users[i:i + 16]; chunk_s = sys_prompts[i:i + 16]
        texts = [pt.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                        tokenize=False, add_generation_prompt=True)
                 for s, u in zip(chunk_s, chunk_u)]
        e = pt(texts, return_tensors="pt", padding=True).to(device)
        g = pm.generate(**e, max_new_tokens=max_new, do_sample=sample, temperature=0.8, top_p=0.95,
                        pad_token_id=pt.pad_token_id)
        for j in range(g.shape[0]):
            out.append(pt.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip())
    return out


_PARA_INSTR = [
    "Reword the question to have the SAME meaning and SAME answer but very DIFFERENT wording. Output only the question.",
    "Ask the same factual question in a completely different sentence structure, same answer. Output only the question.",
]
_SHORT_INSTR = "Rewrite this question as short as possible while keeping the SAME answer. Output only the question."
_JUDGE_SYS = ("You compare two questions. Answer 'yes' if they ask for the SAME fact and would have the SAME "
              "answer, otherwise 'no'. Output only yes or no.")


def judge_equiv(pm, pt, gold_qs, var_qs):
    users = [f"Question A: {a}\nQuestion B: {b}\nSame fact and same answer?" for a, b in zip(gold_qs, var_qs)]
    ans = _chat(pm, pt, [_JUDGE_SYS] * len(users), users, max_new=4)
    return [a.strip().lower().startswith("y") for a in ans]


def run_seed(base, seed, pm, pt):
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
    gold_by_qid = {q["qid"]: q for q in old}

    # answerable = full held-out paraphrase EM==1 under M
    full_q = [{"question": q["eval_question"], "answers": q["answers"], "qid": q["qid"]} for q in old]
    full_greedy = gen(M, [QT.format(q=f["question"]) for f in full_q])
    ans = [i for i in range(len(old)) if em(full_greedy[i], full_q[i]["answers"]) == 1]
    print(f"    answerable {len(ans)}/{len(old)}", flush=True)
    if not ans:
        del M; torch.cuda.empty_cache(); return dict(n_old=len(old), n_answerable=0, rows=[])
    aq = [old[i] for i in ans]
    gold_full = [full_q[i]["question"] for i in ans]

    # ---- build variant questions per answerable fact ----
    paras = [_chat(pm, pt, [_PARA_INSTR[k % len(_PARA_INSTR)]] * len(aq),
                   [q["question"] for q in aq], sample=(k > 0)) for k in range(N_PARA)]
    shorts = _chat(pm, pt, [_SHORT_INSTR] * len(aq), [q["question"] for q in aq])
    # self-gen entity question (best signature match), reuse recall_ladder
    ent = [rl._entity_cue(q["question"], q["answers"]) for q in aq]
    eidx = [k for k, e in enumerate(ent) if e]
    gp, _ = rl.gen_pool(M, [f"Write a factual question about {ent[k]}.\nQuestion:" for k in eidx],
                        GEN_PER, seed + 3, qids=[aq[k]["qid"] for k in eidx], cues=[ent[k] for k in eidx])
    best_gen = {}
    for c in gp:
        g = gold_by_qid.get(c["target_qid"])
        if not g:
            continue
        j = len(_sig(c["question"]) & _sig(g["question"])) / max(len(_sig(c["question"]) | _sig(g["question"])), 1)
        if j >= rl.JACC_THR and j > best_gen.get(c["target_qid"], (0, None))[0]:
            best_gen[c["target_qid"]] = (j, c["question"])
    # swap_entity control: replace this fact's entity with another fact's entity (should DROP gold access)
    rng = random.Random(seed)
    swap = []
    for k, q in enumerate(aq):
        other = [ent[m] for m in eidx if m != k and ent[m]]
        swap.append(q["question"].replace(ent[k], rng.choice(other)) if (ent[k] and other) else None)

    # ---- assemble variant list with proposition-equivalence audit ----
    variants = []                                     # (name, qid, question, audit)
    for k, q in enumerate(aq):
        qid = q["qid"]
        for pk in range(N_PARA):
            variants.append(["para", qid, paras[pk][k], None])
        variants.append(["short", qid, shorts[k], None])
        if qid in best_gen:
            variants.append(["gen_entity", qid, best_gen[qid][1], None])
        if swap[k]:
            variants.append(["swap_entity", qid, swap[k], "control"])
    # audit equivalence for para/short/gen_entity (swap is a known-negative control)
    to_judge = [(i, v) for i, v in enumerate(variants) if v[3] is None]
    if to_judge:
        eq = judge_equiv(pm, pt, [gold_by_qid[v[1]]["question"] for _, v in to_judge],
                         [v[2] for _, v in to_judge])
        for (i, _), e in zip(to_judge, eq):
            variants[i][3] = "equivalent" if e else "changed"

    # ---- score every (fact, variant): gold mlp under M and base, greedy under M and base ----
    gold_full_mlp = {aq[k]["qid"]: _meanlogp(M, [{"question": gold_full[k], "answers": aq[k]["answers"]}])[0]
                     for k in range(len(aq))}
    qas = [{"question": v[2], "answers": gold_by_qid[v[1]]["answers"]} for v in variants]
    mlp_M = _meanlogp(M, qas); mlp_B = _meanlogp(base, qas)
    greedy_M = gen(M, [QT.format(q=v[2]) for v in variants])
    greedy_B = gen(base, [QT.format(q=v[2]) for v in variants])
    rows = []
    for i, v in enumerate(variants):
        name, qid, q, audit = v
        gold = gold_by_qid[qid]["answers"]
        rows.append(dict(seed=seed, qid=qid, variant=name, audit=audit,
                         age=final_t - gold_by_qid[qid]["stream_t"], src=gold_by_qid[qid].get("src", SOURCE),
                         question=q, gold=gold[0], gold_mlp_M=round(mlp_M[i], 3), gold_mlp_base=round(mlp_B[i], 3),
                         lift=round(mlp_M[i] - mlp_B[i], 3),
                         drop_from_full=round(gold_full_mlp[qid] - mlp_M[i], 3),
                         greedy_M=greedy_M[i], em_M=em(greedy_M[i], gold),
                         greedy_base=greedy_B[i], winner_is_base_lure=int(normalize(greedy_M[i]) == normalize(greedy_B[i]) and em(greedy_M[i], gold) == 0)))
    del M; torch.cuda.empty_cache()
    return dict(n_old=len(old), n_answerable=len(aq), full_lift_mean=round(
        sum(gold_full_mlp[aq[k]["qid"]] - _meanlogp(base, [{"question": gold_full[k], "answers": aq[k]["answers"]}])[0]
            for k in range(len(aq))) / len(aq), 3), rows=rows)


def _q(vals, f):
    v = sorted(vals); n = len(v)
    return round(v[min(n - 1, int(f * n))], 3) if n else None


def summarize(rows):
    """group by (variant, audit); on the EQUIVALENT-failure subset report drop/lift medians + base-lure rate."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["variant"], r["audit"])].append(r)
    out = {}
    for (name, audit), rs in sorted(g.items()):
        em_rate = round(sum(x["em_M"] for x in rs) / len(rs), 3)
        fail = [x for x in rs if x["em_M"] == 0]
        out[f"{name}/{audit}"] = dict(
            n=len(rs), em=em_rate, n_fail=len(fail),
            gold_mlp_med=_q([x["gold_mlp_M"] for x in rs], 0.5),
            lift_med=_q([x["lift"] for x in rs], 0.5),
            fail_drop_med=_q([x["drop_from_full"] for x in fail], 0.5),
            fail_lift_med=_q([x["lift"] for x in fail], 0.5),
            fail_base_lure_rate=round(sum(x["winner_is_base_lure"] for x in fail) / max(len(fail), 1), 3))
    return out


def classify(all_rows, full_lift):
    """primary: audited-EQUIVALENT paraphrase failures. decoding if gold lift stays high (near full_lift) with
    base-lure winners; encoding if lift collapses toward 0. Requires >=20 equivalent failed variants."""
    eqv_fail = [r for r in all_rows if r["audit"] == "equivalent" and r["em_M"] == 0]
    swap_fail = [r for r in all_rows if r["variant"] == "swap_entity" and r["em_M"] == 0]
    v = dict(n_equiv_fail=len(eqv_fail), n_swap_fail=len(swap_fail), full_lift_mean=full_lift)
    if len(eqv_fail) < 20:
        v["verdict"] = "underpowered"; return v
    lift_med = _q([r["lift"] for r in eqv_fail], 0.5)
    drop_med = _q([r["drop_from_full"] for r in eqv_fail], 0.5)
    lure = round(sum(r["winner_is_base_lure"] for r in eqv_fail) / len(eqv_fail), 3)
    swap_drop = _q([r["drop_from_full"] for r in swap_fail], 0.5) if swap_fail else None
    v.update(equiv_fail_lift_med=lift_med, equiv_fail_drop_med=drop_med, equiv_fail_base_lure_rate=lure,
             swap_fail_drop_med=swap_drop)
    # decoding: gold still supported (lift >= half of full_lift) yet loses decode (often to a base lure)
    if lift_med is not None and full_lift and lift_med >= 0.5 * full_lift:
        v["verdict"] = "decoding_competition"
    elif lift_med is not None and (full_lift and lift_med <= 0.25 * full_lift):
        v["verdict"] = "encoding_surface_sensitive"
    else:
        v["verdict"] = "mixed"
    return v


def main():
    print(f"MARGIN_PROBE-H ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} cons={CONS_STEPS} "
          f"gen_per={GEN_PER} n_para={N_PARA} seeds={SEEDS}", flush=True)
    base = wb.load_model(); pm, pt = _load_3b()
    seeds_out, all_rows, full_lifts = [], [], []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(base, seed, pm, pt)
        if r is None:
            print("  <2 streams — abort", flush=True); continue
        seeds_out.append({k: v for k, v in r.items() if k != "rows"})
        all_rows += r["rows"]
        if r.get("full_lift_mean") is not None:
            full_lifts.append(r["full_lift_mean"])
        summ = summarize(r["rows"])
        for name, s in summ.items():
            print(f"    [{name:20s}] em={s['em']} n={s['n']} gold_mlp_med={s['gold_mlp_med']} "
                  f"lift_med={s['lift_med']} fail_drop_med={s['fail_drop_med']} lure={s['fail_base_lure_rate']}", flush=True)
        full_lift = round(sum(full_lifts) / len(full_lifts), 3) if full_lifts else None
        v = classify(all_rows, full_lift)
        json.dump(dict(config=dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, cons=CONS_STEPS,
                                   gen_per=GEN_PER, n_para=N_PARA, seeds=seed + 1),
                       seeds=seeds_out, summary=summarize(all_rows), verdict=v), open(OUT, "w"), indent=1)
        json.dump(dict(rows=all_rows), open(OUT.replace(".json", ".rows.json"), "w"))
        print(f"  VERDICT {json.dumps(v)}", flush=True)
    del pm; torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
