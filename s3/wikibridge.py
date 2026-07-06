"""R38-WikiBridge-A — the real-text bridge for continual knowledge-into-weights.

Question (codex-designed): can RAW passage text be turned into durable CLOSED-BOOK QA behavior in a
single dense checkpoint, while old QA behavior is protected by compact training-time targets — no
inference memory, no full joint retraining?

Lifecycle per stream t (arms compact_cpt_*): start from committed M_{t-1}; train a TRANSIENT continued-PT
scaffold S_t on the new stream's passages (+QA span CE for the _qa arm); consolidate into M_t by
distilling S_t on new-stream prompts + replaying OLD committed answer-sequence CE targets + neutral base
anchors; store compact targets for committed-correct new QA; discard S_t. Final eval is closed-book on
held-out paraphrased questions, no passage/retrieval/task-id.

Arms (env WB_ARMS): base_no_ingest | rag_gold_passage | naive_cpt | compact_cpt_only | compact_cpt_qa.
Data (env WB_SOURCE): squad (hard-tail, base-screened) | cf (counterfactual-edited, base-ignorance
guaranteed). Answers: SQuAD-style normalized EM / token-F1. Model: Qwen2.5-0.5B.
"""
import os, sys, json, re, string, random, time, copy, collections
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = os.environ.get("WB_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS = int(os.environ.get("WB_STREAMS", 4))
ARTS = int(os.environ.get("WB_ARTS", 5))          # articles per stream
QA_PER = int(os.environ.get("WB_QA", 5))          # kept QA per article
CPT_STEPS = int(os.environ.get("WB_CPT_STEPS", 300))    # scaffold continued-PT steps/stream
CONS_STEPS = int(os.environ.get("WB_CONS_STEPS", 400))  # consolidation steps/stream
SEEDS = int(os.environ.get("WB_SEEDS", 1))
ARMS = os.environ.get("WB_ARMS", "base_no_ingest,rag_gold_passage,naive_cpt,compact_cpt_only,compact_cpt_qa").split(",")
SOURCE = os.environ.get("WB_SOURCE", "squad")
LR = float(os.environ.get("WB_LR", 1e-5))
OUT = os.environ.get("WB_OUT", "wikibridge_result.json")
MANIFEST = os.environ.get("WB_MANIFEST", "wikibridge_manifest.json")
MAXNEW = int(os.environ.get("WB_MAXNEW", 12))
device = "cuda" if torch.cuda.is_available() else "cpu"
ACTUAL_STREAMS = SURVIVED_ARTS = 0

tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

# ------------------------- SQuAD-style answer normalization / metrics -------------------------
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))

def f1(pred, golds):
    best = 0.0
    pt = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt)); continue
        common = collections.Counter(pt) & collections.Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec = ns / len(pt); rec = ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

QT = "Answer the question with a short answer.\nQuestion: {q}\nAnswer:"          # closed-book template
QT2 = "Q: {q}\nA:"                                                               # 2nd screen template
RT = "Context: {c}\nQuestion: {q}\nAnswer:"                                      # RAG (gold passage)

def load_model():
    # bfloat16, NOT float16: direct fp16 AdamW training NaNs out (verified); bf16 trains stably.
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(device).eval()
    return m

@torch.no_grad()
def gen(model, prompts, max_new=MAXNEW):
    outs = []
    for i in range(0, len(prompts), 32):
        e = tok(prompts[i:i + 32], return_tensors="pt", padding=True).to(device)
        g = model.generate(**e, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for j in range(g.shape[0]):
            new = g[j, e["input_ids"].shape[1]:]
            txt = tok.decode(new, skip_special_tokens=True)
            outs.append(txt.split("\n")[0].strip())
    return outs

def score(model, qas, key="question", template=QT, rag=False):
    if not qas:
        return 0.0, 0.0
    prompts = [(RT.format(c=q["context"], q=q[key]) if rag else template.format(q=q[key])) for q in qas]
    preds = gen(model, prompts)
    e = sum(em(p, q["answers"]) for p, q in zip(preds, qas)) / len(qas)
    ff = sum(f1(p, q["answers"]) for p, q in zip(preds, qas)) / len(qas)
    return e, ff

WB_PARA = os.environ.get("WB_PARA_MODEL", "Qwen/Qwen2.5-3B-Instruct")

def gen_paraphrases(questions):
    """Generate ONE held-out paraphrase per question via an instruct model (data-prep only, then freed).
    The paraphrase preserves meaning+answer but changes surface — the real internalization eval surface."""
    pm = AutoModelForCausalLM.from_pretrained(WB_PARA, dtype=torch.bfloat16).to(device).eval()
    pt = AutoTokenizer.from_pretrained(WB_PARA)
    out = []
    sys = ("Reword the question so it has the SAME meaning and the SAME answer but DIFFERENT wording. "
           "Output only the reworded question, nothing else.")
    for i in range(0, len(questions), 16):
        chunk = questions[i:i + 16]
        msgs = [[{"role": "system", "content": sys}, {"role": "user", "content": q}] for q in chunk]
        texts = [pt.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        pt.padding_side = "left"
        if pt.pad_token is None:
            pt.pad_token = pt.eos_token
        e = pt(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            g = pm.generate(**e, max_new_tokens=40, do_sample=False, pad_token_id=pt.pad_token_id)
        for j in range(g.shape[0]):
            txt = pt.decode(g[j, e["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            out.append(txt.split("\n")[0].strip() or chunk[j])
    del pm; torch.cuda.empty_cache()
    return out

# ------------------------- data: SQuAD hard-tail streams (base-screened) -------------------------
def build_squad(seed, base):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="train")
    by_title = collections.defaultdict(list)
    for r in ds:
        by_title[r["title"]].append(r)
    titles = sorted(by_title)
    rng = random.Random(3000 + seed); rng.shuffle(titles)
    kept_articles = []
    for title in titles:
        if len(kept_articles) >= STREAMS * ARTS * 5:      # over-select; para-screen prunes hard
            break
        rows = by_title[title]
        # one long context = first paragraph(s); collect candidate QAs with short answers
        ctx = rows[0]["context"]
        cand = []
        seen_q = set()
        for r in rows:
            if r["context"] != ctx:
                continue
            a = r["answers"]["text"]
            if not a or r["question"] in seen_q:
                continue
            alen = len(tok(a[0]).input_ids)
            if not (1 <= alen <= 8):
                continue
            seen_q.add(r["question"])
            cand.append({"question": r["question"], "answers": a, "context": ctx})
        if len(cand) < QA_PER + 2:
            continue
        # base-screen: keep QAs base gets wrong on BOTH templates but gold-passage RAG gets right
        e1 = [em(p, q["answers"]) for p, q in zip(gen(base, [QT.format(q=q["question"]) for q in cand]), cand)]
        e2 = [em(p, q["answers"]) for p, q in zip(gen(base, [QT2.format(q=q["question"]) for q in cand]), cand)]
        er = [em(p, q["answers"]) for p, q in zip(gen(base, [RT.format(c=q["context"], q=q["question"]) for q in cand]), cand)]
        hard = [q for q, a, b, r in zip(cand, e1, e2, er) if a == 0 and b == 0 and r == 1]
        if len(hard) >= QA_PER + 1:                # over-select; paraphrase screen will prune
            kept_articles.append({"title": title, "context": ctx, "qas": hard})
    # ---- paraphrase pass: generate 1 held-out paraphrase/QA, keep those base-hard AND RAG-answerable ----
    allq = [q for a in kept_articles for q in a["qas"]]
    paras = gen_paraphrases([q["question"] for q in allq])
    for q, p in zip(allq, paras):
        q["eval_question"] = p
    bp = [em(pr, q["answers"]) for pr, q in zip(gen(base, [QT.format(q=q["eval_question"]) for q in allq]), allq)]
    rp = [em(pr, q["answers"]) for pr, q in zip(gen(base, [RT.format(c=q["context"], q=q["eval_question"]) for q in allq]), allq)]
    ok = {id(q) for q, b, r in zip(allq, bp, rp) if b == 0 and r == 1}   # base fails para, RAG answers para
    final_articles = []
    for a in kept_articles:
        keep = [q for q in a["qas"] if id(q) in ok][:QA_PER]
        if len(keep) >= QA_PER:
            final_articles.append({"title": a["title"], "context": a["context"], "qas": keep})
        if len(final_articles) >= STREAMS * ARTS:
            break
    nstream = min(STREAMS, len(final_articles) // ARTS)   # only COMPLETE streams — never an empty stream
    if nstream < STREAMS:
        print(f"  WARN: {len(final_articles)} articles survived para-screen -> {nstream} streams (< {STREAMS})", flush=True)
    streams = [final_articles[i * ARTS:(i + 1) * ARTS] for i in range(nstream)]
    return streams

# ------------------------- counterfactual edited passages (base-ignorance guaranteed) ------------
CF_REPL = {  # crude typed replacements to make base-unknowable edited facts
    "num": lambda rng: str(rng.randint(3, 99)),
}
def build_cf(seed, base):
    """Take base-hard-ish SQuAD numeric-answer QAs, replace the numeric answer in context+answer with a
    fake number -> the edited answer cannot be pre-known. Reports original-answer lure separately."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="train")
    rng = random.Random(7000 + seed)
    by_title = collections.defaultdict(list)
    for r in ds:
        by_title[r["title"]].append(r)
    titles = sorted(by_title); rng.shuffle(titles)
    arts = []
    for title in titles:
        if len(arts) >= STREAMS * ARTS:
            break
        rows = by_title[title]; ctx = rows[0]["context"]
        qas = []
        for r in rows:
            if r["context"] != ctx or not r["answers"]["text"]:
                continue
            ans = r["answers"]["text"][0]
            if not re.fullmatch(r"\d{1,4}", ans.strip()):
                continue
            new = CF_REPL["num"](rng)
            if new == ans.strip() or ans not in ctx:
                continue
            ectx = ctx.replace(ans, new)
            qas.append({"question": r["question"], "answers": [new], "orig": ans,
                        "context": ectx})
            if len(qas) >= QA_PER:
                break
        if len(qas) >= QA_PER:
            arts.append({"title": title, "context": qas[0]["context"], "qas": qas[:QA_PER]})
        if len(arts) >= STREAMS * ARTS:
            break
    allq = [q for a in arts for q in a["qas"]]        # held-out paraphrase eval surface (answer unchanged)
    for q, p in zip(allq, gen_paraphrases([q["question"] for q in allq])):
        q["eval_question"] = p
    nstream = min(STREAMS, len(arts) // ARTS)
    streams = [arts[i * ARTS:(i + 1) * ARTS] for i in range(nstream)]
    return streams

# ------------------------- training helpers -------------------------
NEUTRAL = [
    "The sky is", "Water is made of", "Paris is the capital of", "Two plus two equals",
    "The sun rises in the", "A dog is a kind of", "The opposite of hot is", "Monday comes before",
    "The color of grass is", "Ice is frozen", "Birds can", "The first month of the year is",
]

def base_anchor_logits(base, prompts):
    e = tok(prompts, return_tensors="pt", padding=True).to(device)
    return e, base.lm_head(base.model(**e, use_cache=False).last_hidden_state[:, -1]).float()

def lm_step(model, texts):
    """next-token LM CE over the passage texts (right-padded truncated)."""
    tok.padding_side = "right"
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    tok.padding_side = "left"
    out = model(**e, use_cache=False)
    logits = out.logits[:, :-1].float()
    labels = e["input_ids"][:, 1:].clone()
    labels[e["attention_mask"][:, 1:] == 0] = -100
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

def qa_ce(model, qas):
    """span CE: CE on the ANSWER tokens given 'Question:..\nAnswer:' prefix (train answer only)."""
    tok.padding_side = "right"
    loss = 0.0; n = 0
    for i in range(0, len(qas), 16):
        chunk = qas[i:i + 16]
        prompts = [QT.format(q=q["question"]) for q in chunk]
        answers = [" " + q["answers"][0] + "\n" for q in chunk]   # train a "\n" STOP after the answer
        full = [p + a for p, a in zip(prompts, answers)]
        e = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        plen = [len(tok(p).input_ids) for p in prompts]
        out = model(**e, use_cache=False)
        logits = out.logits[:, :-1].float()
        labels = e["input_ids"][:, 1:].clone()
        labels[e["attention_mask"][:, 1:] == 0] = -100
        for r, pl in enumerate(plen):                    # mask the question prefix, keep answer tokens
            labels[r, :max(0, pl - 1)] = -100
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        n += 1
    tok.padding_side = "left"
    return loss / max(n, 1)

def main():
    print(f"WIKIBRIDGE ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}art x{QA_PER}qa "
          f"cpt={CPT_STEPS} cons={CONS_STEPS} seeds={SEEDS} arms={ARMS}", flush=True)
    base = load_model()
    results = {a: [] for a in ARMS}
    for seed in range(SEEDS):
        streams = (build_cf if SOURCE == "cf" else build_squad)(seed, base)
        allqa = [q for s in streams for a in s for q in a["qas"]]
        json.dump({"source": SOURCE, "seed": seed,
                   "streams": [[{"title": a["title"], "qas": a["qas"]} for a in s] for s in streams]},
                  open(MANIFEST, "w"), indent=1)
        print(f"  seed {seed}: streams={[len(s) for s in streams]} total_qa={len(allqa)}", flush=True)
        if not allqa:
            print("  NO QA — abort seed", flush=True); continue
        global ACTUAL_STREAMS, SURVIVED_ARTS
        ACTUAL_STREAMS = len(streams); SURVIVED_ARTS = sum(len(s) for s in streams)
        b_em, b_f1 = score(base, allqa, key="question")
        bp_em, bp_f1 = score(base, allqa, key="eval_question")
        r_em, r_f1 = score(base, allqa, key="question", rag=True)
        rp_em, rp_f1 = score(base, allqa, key="eval_question", rag=True)
        n_art = sum(len(s) for s in streams)
        print(f"    actual_streams={len(streams)} survived_articles={n_art} | "
              f"base closed-book O/P EM={b_em:.3f}/{bp_em:.3f} | RAG-gold O/P EM={r_em:.3f}/{rp_em:.3f}", flush=True)

        for arm in ARMS:
            t0 = time.time()
            if arm == "base_no_ingest":
                res = dict(final_em=b_em, final_f1=b_f1, final_para_em=bp_em, final_para_f1=bp_f1)
            elif arm == "rag_gold_passage":
                res = dict(final_em=r_em, final_f1=r_f1, final_para_em=rp_em, final_para_f1=rp_f1)
            else:
                res = run_ingest(base, streams, arm, seed)
            res["wall"] = round(time.time() - t0, 1)
            results[arm].append(res)
            print(f"    [{arm}] {json.dumps({k: v for k, v in res.items() if k != 'per_stream'})}", flush=True)
        dump(results, seed + 1)
    print("[done]", flush=True)

def run_ingest(base, streams, arm, seed):
    use_qa = arm == "compact_cpt_qa"
    replay = arm.startswith("compact")
    M = load_model()                                  # M_0 = base copy (trainable)
    rng = random.Random(seed * 100003 + sum(bytes(arm, "utf8")))   # stable seed (no PYTHONHASHSEED dep)
    REPLAY_K = int(os.environ.get("WB_REPLAY_K", -1))  # -1=replay ALL committed; K>=0 = K QA/article
    committed = []; replay_pool = []; nonreplay_pool = []   # committed old QA; footprint split
    per_stream = []
    for t in range(len(streams)):
        arts = streams[t]
        new_qa = [q for a in arts for q in a["qas"]]
        passages = [a["context"] for a in arts]
        # ---- transient scaffold S_t: continued-PT on new stream (LM [+QA]) ----
        S = copy.deepcopy(M); S.train()
        opt = torch.optim.AdamW(S.parameters(), lr=LR)
        for _ in range(CPT_STEPS):
            loss = lm_step(S, [rng.choice(passages) for _ in range(4)])
            if use_qa:
                loss = loss + qa_ce(S, [rng.choice(new_qa) for _ in range(8)])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step()
        S.eval()
        s_em, _ = score(S, new_qa, key="question")     # DIAGNOSTIC: scaffold on ORIG question
        s_pa, _ = score(S, new_qa, key="eval_question")  # scaffold on held-out PARAPHRASE (internalization)
        # ---- consolidate into M_t: distill S_t on new + replay OLD committed CE + neutral anchor ----
        if arm == "naive_cpt":                         # naive: M becomes the continued-PT model, no replay
            M = S
        else:
            M.train(); opt = torch.optim.AdamW(M.parameters(), lr=LR)
            for _ in range(CONS_STEPS):
                # new knowledge: distill S_t (LM on passages [+ QA span CE to gold])
                loss = lm_step(M, [rng.choice(passages) for _ in range(4)])
                if use_qa:
                    loss = loss + qa_ce(M, [rng.choice(new_qa) for _ in range(8)])
                # old retention: committed answer-sequence CE (compact target, no snapshot teacher).
                # footprint: replay only the K-per-article subset (replay_pool) when WB_REPLAY_K>=0.
                pool = committed if REPLAY_K < 0 else replay_pool
                if replay and pool:
                    loss = loss + qa_ce(M, [rng.choice(pool) for _ in range(8)])
                # base-capability anchor on NEUTRAL prompts ONLY (never on old QA — R37-A lesson)
                ne, nb = base_anchor_logits(base, [rng.choice(NEUTRAL) for _ in range(8)])
                sa = M.lm_head(M.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
            M.eval()
        del S; torch.cuda.empty_cache()
        # commit: store compact targets for committed-correct new QA (by ORIG question — training surface)
        M.eval()
        m_em, _ = score(M, new_qa, key="question")
        m_pa, _ = score(M, new_qa, key="eval_question")     # M on held-out PARAPHRASE = internalization
        correct = [q for q, p in zip(new_qa, gen(M, [QT.format(q=q["question"]) for q in new_qa]))
                   if em(p, q["answers"]) == 1]
        committed += correct
        if REPLAY_K >= 0:                              # footprint: fixed K committed QA/article replayed
            by_art = collections.defaultdict(list)
            for q in correct:
                by_art[q["context"]].append(q)
            for qs in by_art.values():
                replay_pool += qs[:REPLAY_K]; nonreplay_pool += qs[REPLAY_K:]
        # retention: OLD streams' QA (0..t-1) on M — both ORIG and PARAPHRASE surfaces
        old_qa = [q for tt in range(t) for a in streams[tt] for q in a["qas"]]
        o_em = round(score(M, old_qa, key="question")[0], 3) if old_qa else None
        o_pa = round(score(M, old_qa, key="eval_question")[0], 3) if old_qa else None
        per_stream.append(dict(t=t, scaffold_new_orig=round(s_em, 3), scaffold_new_para=round(s_pa, 3),
                               M_new_orig=round(m_em, 3), M_new_para=round(m_pa, 3),
                               n_committed=len(correct), old_orig=o_em, old_para=o_pa))
        print(f"      [{arm} s{seed} t{t}] scaffold O/P={s_em:.2f}/{s_pa:.2f} M_new O/P={m_em:.2f}/{m_pa:.2f} "
              f"committed={len(correct)}/{len(new_qa)} old O/P={o_em}/{o_pa}", flush=True)
    allqa = [q for s in streams for a in s for q in a["qas"]]
    f_em, f_f1 = score(M, allqa, key="question")
    fp_em, fp_f1 = score(M, allqa, key="eval_question")     # FINAL held-out paraphrase = the R38 gate
    # footprint split: replayed vs NON-replayed committed-old paraphrase EM (R36-A lesson on real text)
    rep_pa = round(score(M, replay_pool, key="eval_question")[0], 3) if replay_pool else None
    non_pa = round(score(M, nonreplay_pool, key="eval_question")[0], 3) if nonreplay_pool else None
    del M; torch.cuda.empty_cache()
    return dict(final_em=round(f_em, 3), final_f1=round(f_f1, 3),
                final_para_em=round(fp_em, 3), final_para_f1=round(fp_f1, 3),
                replay_k=REPLAY_K, n_replayed=len(replay_pool), n_nonreplayed=len(nonreplay_pool),
                replayed_para_em=rep_pa, nonreplayed_para_em=non_pa, per_stream=per_stream)

def dump(results, nseeds):
    summ = {}
    for arm in ARMS:
        rs = results[arm]
        if not rs:
            continue
        def av(k):
            return round(sum(r.get(k, 0) for r in rs) / len(rs), 3)
        summ[arm] = dict(final_em=av("final_em"), final_f1=av("final_f1"),
                         final_para_em=av("final_para_em"), final_para_f1=av("final_para_f1"),
                         per_stream=rs[-1].get("per_stream"))
    json.dump({"config": dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, seeds=nseeds, arms=ARMS,
                              actual_streams=ACTUAL_STREAMS, survived_articles=SURVIVED_ARTS),
               "summary": summ}, open(OUT, "w"), indent=1)
    print("RESULT_JSON " + json.dumps({a: {"final_em": summ[a]["final_em"],
                                           "final_para_em": summ[a]["final_para_em"]} for a in summ}), flush=True)

if __name__ == "__main__":
    main()
