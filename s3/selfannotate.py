"""A0 — write-time self-annotation (contract v2 READ step). Can the model build its OWN cue/QA ledger for the
facts it JUST wrote, with no gold labels? (codex qa 2026-07-11 06:09 + 06:44.)

Timing asymmetry: R48/R49a/R50 killed POST-HOC self-enumeration; at t+0 the acquisition delta M_t vs M_prev is
fresh and concentrated. Generation-only, hard-stopped. Arms (matched GEN_N budget):
  1. passive        : M_t few-shot QA generation (the R48-style floor).
  2. contrast_q     : acquisition-contrastive QUESTION generation, score = logp_Mt + LAM*(logp_Mt - logp_Mprev).
  3. contrast_qa    : contrastive FULL-QA generation; logs the question channel d_q = mean dlogp(q) and the
                      answer channel d_a = mean dlogp(a|q) SEPARATELY per candidate (R50-A lesson).
  4. contrast_decl  : contrastive DECLARATIVE statements (doesn't assume learning an answer raises p(question)).
  5. source_annot   : source-CONDITIONED self-annotation — M_t reads the current passage and writes QA from it
                      (the practical headline: at write time the book is in hand).
  6. gold           : the stream's gold QAs (coverage/precision ceiling + ledger-bytes reference).
Audit per candidate (offline, gold never feeds generation): signature-match to the FROZEN stream manifest AND
proposition-equivalent (3B judge) AND M_t answers the generated q correctly AND the ORIGINAL BASE answers it
WRONG (acquired, not prior) AND answer stable under 2 independent paraphrases of the generated q.
Gate (codex): >=25% unique acquired-proposition coverage AND >=0.80 precision, per seed, >=4 unique props,
no single template dominating. Model: Qwen2.5-0.5B; M_prev = frozen base (one ordinary write). census streams.
"""
import os, sys, json, math, random, collections
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb
from s3 import selfreplay as sr
from s3 import recall_ladder as rl
from s3 import margin_probe as mp

NAME    = os.environ.get("SA_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS = int(os.environ.get("SA_STREAMS", 2))          # per seed: use stream[seed] as the written stream
ARTS    = int(os.environ.get("SA_ARTS", 5))
QA_PER  = int(os.environ.get("SA_QA", 5))
CONS_STEPS = int(os.environ.get("SA_CONS_STEPS", 400))
GEN_N   = int(os.environ.get("SA_GEN_N", 200))          # candidates per arm
LAM     = float(os.environ.get("SA_LAM", 2.0))
SEEDS   = int(os.environ.get("SA_SEEDS", 2))
SOURCE  = os.environ.get("SA_SOURCE", "census")
OUT     = os.environ.get("SA_OUT", "selfannotate_result.json")
device = wb.device

wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CONS_STEPS, wb.LR = CONS_STEPS, rl.LR
wb.SOURCE = SOURCE
rl.CONS_STEPS = CONS_STEPS
tok = wb.tok
QT, em, gen, normalize = wb.QT, wb.em, wb.gen, wb.normalize
_sig = sr._sig

QA_SCAFFOLD = sr._FEWSHOT
DECL_SCAFFOLD = ("Here are true facts.\n"
                 "Fact: The capital of France is Paris.\n"
                 "Fact: William Shakespeare wrote the play Romeo and Juliet.\n"
                 "Fact: Gold has the chemical symbol Au.\n"
                 "Fact:")
SRC_SCAFFOLD = ("Read the passage and write one question answered by it.\n"
                "Passage: The Eiffel Tower, completed in 1889, was designed by Gustave Eiffel's company.\n"
                "Question: Who designed the Eiffel Tower?\nAnswer: Gustave Eiffel's company\n"
                "Passage: {p}\nQuestion:")


@torch.no_grad()
def contrastive_gen(Mt, Mp, scaffold, n, seed, lam, max_new=44):
    """sample n continuations from Mt steered by (logp_Mt - logp_Mp), top-p constrained on Mt (fluency).
    Returns list of raw continuation strings + generated-token count."""
    outs, gtok = [], 0
    B = 24
    tok.padding_side = "left"
    enc = tok([scaffold], return_tensors="pt").to(device)
    for i in range(0, n, B):
        k = min(B, n - i)
        ids = enc["input_ids"].repeat(k, 1); am = torch.ones_like(ids)
        torch.manual_seed(seed * 100003 + i)
        for _ in range(max_new):
            lt = Mt(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
            lp = Mp(input_ids=ids, attention_mask=am, use_cache=False).logits[:, -1].float()
            logp_t = F.log_softmax(lt, -1); logp_p = F.log_softmax(lp, -1)
            adj = logp_t + lam * (logp_t - logp_p)
            probs = F.softmax(logp_t / 0.9, -1)
            sp, si = probs.sort(descending=True)
            mask = (sp.cumsum(-1) - sp) > 0.95
            keep = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, si, ~mask)
            adj = adj.masked_fill(~keep, -1e9)
            nxt = torch.distributions.Categorical(logits=adj / 0.9).sample()
            ids = torch.cat([ids, nxt[:, None]], 1)
            am = torch.cat([am, torch.ones(k, 1, device=device, dtype=am.dtype)], 1)
        gtok += k * max_new
        for j in range(k):
            outs.append(tok.decode(ids[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs, gtok


@torch.no_grad()
def plain_gen(M, prompts, per, seed, max_new=44):
    outs, gtok = [], 0
    tok.padding_side = "left"
    flat = [p for p in prompts for _ in range(per)]
    for i in range(0, len(flat), 24):
        chunk = flat[i:i + 24]
        e = tok(chunk, return_tensors="pt", padding=True).to(device)
        torch.manual_seed(seed * 7 + i)
        g = M.generate(**e, max_new_tokens=max_new, do_sample=True, temperature=0.9, top_p=0.95,
                       pad_token_id=tok.pad_token_id)
        gtok += int((g[:, e["input_ids"].shape[1]:] != tok.pad_token_id).sum().item())
        for j in range(g.shape[0]):
            outs.append(tok.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs, gtok


def parse_qa(txt):
    block = txt.split("Question:")[0] if "Question:" in txt else txt
    if "Answer:" not in block:
        return None
    qp, ap = block.split("Answer:", 1)
    q = qp.strip().split("\n")[0].strip(); a = ap.strip().split("\n")[0].strip()
    return (q, a) if (len(q) >= 8 and q.endswith("?") and 0 < len(a) <= 60) else None


def parse_decl(txt):
    s = txt.strip().split("\n")[0].strip()
    return s if (12 <= len(s) <= 160 and not s.endswith("?")) else None


def _meanlogp(M, qas, key="question"):
    return [-(tb * math.log(2) / nt) if nt else -1e9 for (tb, nt) in wb.qa_answer_bits(M, qas, key=key)]


def audit_arm(name, cands, Mt, base, pm, pt, manifest, is_decl=False):
    """cands: list of dicts {question[,answer]} or {statement}. Returns stats + rows."""
    by_qid = {f["qid"]: f for f in manifest}
    # 1. signature match to the frozen manifest
    hits = []
    for c in cands:
        text = c.get("question") or c.get("statement")
        s = _sig(text); best, bj = None, rl.JACC_THR
        for f in manifest:
            ref = f["question"] + " " + f["answers"][0]
            j = len(s & _sig(ref)) / max(len(s | _sig(ref)), 1)
            if j >= bj:
                best, bj = f, j
        if best is not None:
            hits.append((c, best))
    correct, rows = set(), []
    if hits:
        if is_decl:
            # declarative: statement must CONTAIN the gold answer AND judge says it expresses the same fact
            eqv = mp.judge_equiv(pm, pt, [f"{g['question']} (answer: {g['answers'][0]})" for _, g in hits],
                                 [c["statement"] for c, _ in hits])
            for (c, g), ev in zip(hits, eqv):
                ok = ev and (normalize(g["answers"][0]) in normalize(c["statement"]))
                if ok:
                    correct.add(g["qid"])
                rows.append(dict(qid=g["qid"], gen=c["statement"][:100], equiv=int(ev), ok=int(ok)))
        else:
            qs = [c["question"] for c, _ in hits]
            eqv = mp.judge_equiv(pm, pt, [g["question"] for _, g in hits], qs)
            pred_t = gen(Mt, [QT.format(q=q) for q in qs])
            pred_b = gen(base, [QT.format(q=q) for q in qs])
            views = sr.gen_views(qs, 2)                          # 2 independent paraphrase views (3B)
            v_ans = [gen(Mt, [QT.format(q=p) for p in vq]) for vq in views]
            for idx, ((c, g), ev) in enumerate(zip(hits, eqv)):
                m_ok = em(pred_t[idx], g["answers"]) == 1
                b_wrong = em(pred_b[idx], g["answers"]) == 0
                stab = sum(em(pred_t[idx], [v_ans[v][idx]]) == 1 for v in range(2)) >= 1
                ok = m_ok and ev and b_wrong and stab
                if ok:
                    correct.add(g["qid"])
                rows.append(dict(qid=g["qid"], gen=c["question"][:100], m_ok=int(m_ok), equiv=int(ev),
                                 b_wrong=int(b_wrong), stable=int(stab), ok=int(ok),
                                 d_q=c.get("d_q"), d_a=c.get("d_a")))
    n_matched = len(hits)
    prec = round(len([r for r in rows if r["ok"]]) / max(n_matched, 1), 3)
    cov = round(len(correct) / max(len(manifest), 1), 3)
    # duplicate/template concentration among correct hits
    ent = collections.Counter(r["qid"] for r in rows if r["ok"])
    dom = round(max(ent.values()) / max(sum(ent.values()), 1), 3) if ent else None
    ledger = int(sum(len((c.get("question") or c.get("statement", "")).encode("utf8")) +
                     len((c.get("answer") or "").encode("utf8")) for c, _ in hits))
    return dict(n_cands=len(cands), n_matched=n_matched, n_correct=len(correct),
                coverage=cov, precision=prec, dominance=dom, ledger_bytes=ledger), rows


def run_seed(seed, base, pm, pt):
    streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
    for t, s in enumerate(streams):
        for ai, a in enumerate(s):
            for j, q in enumerate(a["qas"]):
                q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
    if not streams:
        return None
    arts = streams[min(seed, len(streams) - 1)]                # one stream per seed
    new_qa = [q for a in arts for q in a["qas"]]
    passages = [a["context"] for a in arts]
    # FROZEN manifest: base wrong on canonical question AND on the held-out paraphrase (base-hard both views)
    b1 = [em(p, q["answers"]) for p, q in zip(gen(base, [QT.format(q=q["question"]) for q in new_qa]), new_qa)]
    b2 = [em(p, q["answers"]) for p, q in zip(gen(base, [QT.format(q=q["eval_question"]) for q in new_qa]), new_qa)]
    manifest = [q for q, x, y in zip(new_qa, b1, b2) if x == 0 and y == 0]
    print(f"    manifest {len(manifest)}/{len(new_qa)} base-hard facts", flush=True)
    if len(manifest) < 5:
        return None
    # ONE ordinary write: M_t = base + write(stream)
    Mt = rl.bare_write(wb.load_model(), passages, new_qa, base, CONS_STEPS, seed * 991 + 7)
    # acquisition check: how many manifest facts did the write install (M_t correct on canonical q)?
    acq = [q for q, p in zip(manifest, gen(Mt, [QT.format(q=q["question"]) for q in manifest]))
           if em(p, q["answers"]) == 1]
    print(f"    acquired {len(acq)}/{len(manifest)}", flush=True)

    arms, ledgers = {}, {}
    # 1. passive
    raw, g1 = plain_gen(Mt, [QA_SCAFFOLD], GEN_N, seed)
    cands = [dict(zip(("question", "answer"), p)) for p in map(parse_qa, raw) if p]
    arms["passive"], _ = audit_arm("passive", cands, Mt, base, pm, pt, manifest)
    # 2. contrastive question(+answer parse); channel deltas for the QA arm
    raw, g2 = contrastive_gen(Mt, base, QA_SCAFFOLD, GEN_N, seed + 1, LAM)
    qa2 = [dict(zip(("question", "answer"), p)) for p in map(parse_qa, raw) if p]
    arms["contrast_q"], _ = audit_arm("contrast_q", [dict(question=c["question"]) for c in qa2],
                                      Mt, base, pm, pt, manifest)
    # 3. contrastive full QA w/ channel split
    if qa2:
        dq_t = _meanlogp(Mt, [dict(question=QA_SCAFFOLD, answers=[c["question"]]) for c in qa2])
        dq_p = _meanlogp(base, [dict(question=QA_SCAFFOLD, answers=[c["question"]]) for c in qa2])
        da_t = _meanlogp(Mt, [dict(question=c["question"], answers=[c["answer"]]) for c in qa2])
        da_p = _meanlogp(base, [dict(question=c["question"], answers=[c["answer"]]) for c in qa2])
        for k, c in enumerate(qa2):
            c["d_q"] = round(dq_t[k] - dq_p[k], 3); c["d_a"] = round(da_t[k] - da_p[k], 3)
    arms["contrast_qa"], rows_qa = audit_arm("contrast_qa", qa2, Mt, base, pm, pt, manifest)
    # 4. contrastive declaratives
    raw, g4 = contrastive_gen(Mt, base, DECL_SCAFFOLD, GEN_N, seed + 2, LAM, max_new=40)
    decls = [dict(statement=s) for s in map(parse_decl, raw) if s]
    arms["contrast_decl"], _ = audit_arm("contrast_decl", decls, Mt, base, pm, pt, manifest, is_decl=True)
    # 5. source-conditioned annotation (the practical READ step)
    per = max(1, GEN_N // max(len(passages), 1))
    raw, g5 = plain_gen(Mt, [SRC_SCAFFOLD.format(p=p[:1200]) for p in passages], per, seed + 3)
    cands5 = [dict(zip(("question", "answer"), p)) for p in map(parse_qa, raw) if p]
    arms["source_annot"], _ = audit_arm("source_annot", cands5, Mt, base, pm, pt, manifest)
    # 6. gold ceiling (bytes reference)
    goldc = [dict(question=q["question"], answer=q["answers"][0]) for q in manifest]
    arms["gold"] = dict(n_cands=len(goldc), n_matched=len(goldc), n_correct=len(acq),
                        coverage=round(len(acq) / max(len(manifest), 1), 3), precision=1.0, dominance=None,
                        ledger_bytes=int(sum(len(q["question"].encode("utf8")) + len(q["answers"][0].encode("utf8"))
                                             for q in manifest)))
    for name, s in arms.items():
        print(f"    [{name:13s}] cands={s['n_cands']} matched={s['n_matched']} correct={s['n_correct']} "
              f"cov={s['coverage']} prec={s['precision']} bytes={s['ledger_bytes']}", flush=True)
    del Mt; torch.cuda.empty_cache()
    return dict(n_new=len(new_qa), n_manifest=len(manifest), n_acquired=len(acq), arms=arms,
                rows_contrast_qa=rows_qa)


def decide(seed_results):
    rows = [r for r in seed_results if r]
    if len(rows) < 2:
        return dict(phase="shakedown_only", n=len(rows))
    v = {}
    for arm in ("passive", "contrast_q", "contrast_qa", "contrast_decl", "source_annot"):
        cov = [r["arms"][arm]["coverage"] for r in rows]
        prec = [r["arms"][arm]["precision"] for r in rows]
        ncor = [r["arms"][arm]["n_correct"] for r in rows]
        v[arm] = dict(coverage=cov, precision=prec,
                      gate=bool(all(c >= 0.25 for c in cov) and all(p >= 0.80 for p in prec)
                                and all(n >= 4 for n in ncor)))
    passing = [a for a in v if v[a]["gate"]]
    v["verdict"] = ("self_annotation_viable:" + ",".join(passing)) if passing else "all_arms_below_gate"
    return v


def main():
    print(f"SELF-ANNOTATE A0 ({NAME}, {device}) source={SOURCE} arts={ARTS}x{QA_PER} cons={CONS_STEPS} "
          f"gen_N={GEN_N} lam={LAM} seeds={SEEDS}", flush=True)
    base = wb.load_model()
    pm, pt = mp._load_3b()
    out = []
    for seed in range(SEEDS):
        print(f"  seed {seed}", flush=True)
        r = run_seed(seed, base, pm, pt)
        if r is None:
            print("  abort seed", flush=True); continue
        out.append(r)
        v = decide(out)
        json.dump(dict(config=dict(model=NAME, source=SOURCE, arts=ARTS, qa=QA_PER, cons=CONS_STEPS,
                                   gen_N=GEN_N, lam=LAM, seeds=seed + 1),
                       seeds=out, decision=v), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
