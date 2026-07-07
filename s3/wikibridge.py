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
import os, sys, json, re, string, random, time, copy, collections, math
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
SYNTH_VARIANT = os.environ.get("WB_SYNTH_VARIANT", "shared_templates")  # shared_templates | unique_templates
LR = float(os.environ.get("WB_LR", 1e-5))
OUT = os.environ.get("WB_OUT", "wikibridge_result.json")
MANIFEST = os.environ.get("WB_MANIFEST", "wikibridge_manifest.json")
MAXNEW = int(os.environ.get("WB_MAXNEW", 12))
device = "cuda" if torch.cuda.is_available() else "cpu"
ACTUAL_STREAMS = SURVIVED_ARTS = 0
SURPRISE_BASE = {}                                  # R40->s3: seed -> {qid: base surprise/EM record}

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
        if len(kept_articles) >= STREAMS * ARTS * 8:      # over-select; para-screen prunes hard
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

# ------------------------- independent synthetic facts (same objective, no real-text redundancy) --------
# codex R38B-A control: swap ONLY the corpus, keep the identical run_ingest objective. Each fact has its own
# invented subject AND answer (no within-article entity reuse, no answer-string reuse anywhere) -> zero
# real-world redundancy / pretrained-prior overlap. Article = interface grouping for replay-by-context only.
_CONS = "bcdfghjklmnpqrstvwz"; _VOW = "aeiou"
def _coin(rng, nsyl):
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) + (rng.choice(_CONS) if rng.random() < 0.4 else "")
                   for _ in range(nsyl)).capitalize()
# relation templates: (statement, question, PARAPHRASE) with {s}=subject {a}=answer. shared_templates cycles
# a small set (high question-format sharing); unique_templates draws distinct phrasings per fact (low sharing).
_SHARED_TMPL = [
    ("The capital of {s} is {a}.",            "What is the capital of {s}?",      "Which city is the capital of {s}?"),
    ("The {s} was founded by {a}.",           "Who founded the {s}?",             "By whom was the {s} founded?"),
    ("The {s} is powered by {a}.",            "What powers the {s}?",             "What is the {s} powered by?"),
]
_UNIQUE_TMPL = _SHARED_TMPL + [
    ("{s} is located in the region of {a}.",  "In which region is {s} located?",  "Where is {s} located?"),
    ("The official language of {s} is {a}.",  "What is the official language of {s}?", "Which language is official in {s}?"),
    ("{s} was first described by {a}.",       "Who first described {s}?",         "By whom was {s} first described?"),
    ("The {s} produces mostly {a}.",          "What does the {s} mostly produce?","What is chiefly produced by the {s}?"),
    ("{s} is ruled by {a}.",                  "Who rules {s}?",                   "By whom is {s} ruled?"),
    ("The currency of {s} is the {a}.",       "What is the currency of {s}?",     "Which currency does {s} use?"),
    ("The {s} was destroyed by {a}.",         "Who destroyed the {s}?",           "By whom was the {s} destroyed?"),
]
def build_synth(seed, base):
    rng = random.Random(90000 + seed)
    tmpls = _SHARED_TMPL if SYNTH_VARIANT == "shared_templates" else _UNIQUE_TMPL
    used = set()
    def fresh():                                   # unique short-tokenizing invented name (<=4 tokens)
        for _ in range(200):
            w = _coin(rng, rng.choice([2, 3]))
            if w in used:
                continue
            if len(tok(w, add_special_tokens=False).input_ids) <= 4:
                used.add(w); return w
        raise RuntimeError("name pool exhausted")
    n_art = STREAMS * ARTS * 4                      # over-select; RAG-screen on lookalike names is lossy
    arts = []
    for _ in range(n_art):
        facts = []
        for j in range(QA_PER + 4):
            s, a = fresh(), fresh()
            st, q, p = (tmpls[(len(facts)) % len(tmpls)] if SYNTH_VARIANT == "shared_templates"
                        else rng.choice(tmpls))
            facts.append({"stmt": st.format(s=s, a=a), "question": q.format(s=s),
                          "eval_question": p.format(s=s), "answers": [a]})
        ctx = " ".join(f["stmt"] for f in facts)    # passage = concatenation of this article's fact sentences
        for f in facts:
            f["context"] = ctx
        arts.append({"title": f"synth_{len(arts)}", "context": ctx, "qas": facts})
    # screen (parity with squad): keep base-hard (fails Q AND paraphrase closed-book) AND RAG-answerable
    allq = [q for a in arts for q in a["qas"]]
    e1 = [em(pr, q["answers"]) for pr, q in zip(gen(base, [QT.format(q=q["question"]) for q in allq]), allq)]
    ep = [em(pr, q["answers"]) for pr, q in zip(gen(base, [QT.format(q=q["eval_question"]) for q in allq]), allq)]
    r1 = [em(pr, q["answers"]) for pr, q in zip(gen(base, [RT.format(c=q["context"], q=q["question"]) for q in allq]), allq)]
    rp = [em(pr, q["answers"]) for pr, q in zip(gen(base, [RT.format(c=q["context"], q=q["eval_question"]) for q in allq]), allq)]
    ok = {id(q) for q, a, b, c, d in zip(allq, e1, ep, r1, rp) if a == 0 and b == 0 and c == 1 and d == 1}
    final_articles = []
    for a in arts:
        keep = [q for q in a["qas"] if id(q) in ok][:QA_PER]
        if len(keep) >= QA_PER:
            final_articles.append({"title": a["title"], "context": a["context"], "qas": keep})
        if len(final_articles) >= STREAMS * ARTS:
            break
    alens = [len(tok(q["answers"][0], add_special_tokens=False).input_ids)
             for a in final_articles for q in a["qas"]]
    nstream = min(STREAMS, len(final_articles) // ARTS)
    ln = f"{min(alens)}/{sum(alens)/len(alens):.1f}/{max(alens)}" if alens else "n/a (0 survived)"
    print(f"  SYNTH[{SYNTH_VARIANT}]: {len(final_articles)} articles survived -> {nstream} streams; "
          f"ans_tok_len min/mean/max={ln}", flush=True)
    return [final_articles[i * ARTS:(i + 1) * ARTS] for i in range(nstream)]

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

@torch.no_grad()
def qa_answer_bits(model, qas, key="question"):
    """per-QA gold answer-sequence surprise in bits (total, #answer-tokens) under the closed-book template.
    Mirrors qa_ce's answer-token masking. On the frozen base this is R40's per-fact 'surprise' cost unit."""
    tok.padding_side = "right"
    res = []
    for i in range(0, len(qas), 16):
        chunk = qas[i:i + 16]
        prompts = [QT.format(q=q[key]) for q in chunk]
        answers = [" " + q["answers"][0] + "\n" for q in chunk]
        full = [p + a for p, a in zip(prompts, answers)]
        e = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        plen = [len(tok(p).input_ids) for p in prompts]
        logp = F.log_softmax(model(**e, use_cache=False).logits[:, :-1].float(), -1)
        labels = e["input_ids"][:, 1:].clone()
        labels[e["attention_mask"][:, 1:] == 0] = -100
        for r, pl in enumerate(plen):
            labels[r, :max(0, pl - 1)] = -100
        for r in range(len(chunk)):
            m = labels[r] != -100
            if bool(m.any()):
                nll = -logp[r][m].gather(1, labels[r][m][:, None]).squeeze(1)   # nats/token
                res.append((round((nll.sum() / math.log(2)).item(), 3), int(m.sum().item())))
            else:
                res.append((0.0, 0))
    tok.padding_side = "left"
    return res

def main():
    print(f"WIKIBRIDGE ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}art x{QA_PER}qa "
          f"cpt={CPT_STEPS} cons={CONS_STEPS} seeds={SEEDS} arms={ARMS}", flush=True)
    base = load_model()
    results = {a: [] for a in ARMS}
    for seed in range(SEEDS):
        builder = {"squad": build_squad, "cf": build_cf, "synth": build_synth}.get(SOURCE)
        if builder is None:
            raise ValueError(f"unknown WB_SOURCE={SOURCE!r} (expected squad|cf|synth)")
        streams = builder(seed, base)
        allqa = [q for s in streams for a in s for q in a["qas"]]
        json.dump({"source": SOURCE, "seed": seed,
                   "streams": [[{"title": a["title"], "qas": a["qas"]} for a in s] for s in streams]},
                  open(MANIFEST.replace(".json", f".s{seed}.json"), "w"), indent=1)   # per-seed (audit, codex)
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

        # R40->s3 surprise instrumentation: stable qid per QA + FROZEN-BASE per-QA surprise/EM/answer-length.
        for t, s in enumerate(streams):
            for ai, a in enumerate(s):
                for j, q in enumerate(a["qas"]):
                    q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t; q["art_title"] = a["title"]
        bits_o = qa_answer_bits(base, allqa, "question")
        bits_p = qa_answer_bits(base, allqa, "eval_question")
        gen_o = gen(base, [QT.format(q=q["question"]) for q in allqa])
        gen_p = gen(base, [QT.format(q=q["eval_question"]) for q in allqa])
        SURPRISE_BASE[seed] = {q["qid"]: dict(source=SOURCE, t=q["stream_t"], title=q["art_title"],
                                              base_em_orig=em(gen_o[k], q["answers"]),
                                              base_em_para=em(gen_p[k], q["answers"]),
                                              base_bits_orig=bits_o[k][0], base_bits_para=bits_p[k][0],
                                              ans_ntok=bits_p[k][1]) for k, q in enumerate(allqa)}
        _sb = [v["base_bits_para"] for v in SURPRISE_BASE[seed].values()]
        _al = sum(v["base_em_para"] for v in SURPRISE_BASE[seed].values())
        print(f"    [R40 surprise] mean base-para bits={sum(_sb)/max(len(_sb),1):.2f} "
              f"already-correct para={_al}/{len(_sb)}", flush=True)

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
    use_qa = arm.startswith("compact_cpt_qa")          # ..._k<K> footprint variants are still QA arms
    replay = arm.startswith("compact")
    M = load_model()                                  # M_0 = base copy (trainable)
    rng = random.Random(seed * 100003 + sum(bytes(arm, "utf8")))   # stable seed (no PYTHONHASHSEED dep)
    no_anchor = arm.endswith("_noanchor")              # attribution floor: gentle consolidate w/o neutral anchor
    a2 = arm[:-len("_noanchor")] if no_anchor else arm
    # -1=replay ALL committed; K>=0 = K committed QA/article (footprint). arm suffix _k<K> overrides env.
    REPLAY_K = int(a2.split("_k")[-1]) if "_k" in a2 else int(os.environ.get("WB_REPLAY_K", -1))
    committed = []; replay_pool = []                   # replay_pool = items actually replayed
    committed_log = []                                 # (qa, commit_t, is_replayed) — for OLD-ONLY final split
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
                # base-capability anchor on NEUTRAL prompts ONLY (never on old QA — R37-A lesson).
                # _noanchor floor arm drops this to separate "gentle updates" from "redundancy spillover".
                if not no_anchor:
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
        by_art = collections.defaultdict(list)         # nested random K-per-article (stable per article)
        for q in correct:
            by_art[q["context"]].append(q)
        for qs in by_art.values():
            random.Random(f"{seed}:{qs[0]['context']}").shuffle(qs)   # process-STABLE per-article order (str seed -> sha512)
            for i, q in enumerate(qs):
                rep = (REPLAY_K < 0) or (i < REPLAY_K)
                committed_log.append((q, t, rep))
                if rep:
                    replay_pool.append(q)
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
    # footprint split — OLD-ONLY (codex): only items committed BEFORE the final stream (exposed to later
    # forgetting). Exclude final-stream fresh items. Report replayed vs non-replayed OLD paraphrase.
    final_t = len(streams) - 1
    old_rep = [q for (q, ct, rep) in committed_log if ct < final_t and rep]
    old_non = [q for (q, ct, rep) in committed_log if ct < final_t and not rep]
    def sc(qs):
        return round(score(M, qs, key="eval_question")[0], 3) if qs else None
    # age buckets: how far back the committed item is from the final checkpoint
    ages = collections.defaultdict(list)
    for (q, ct, rep) in committed_log:
        if ct < final_t:
            ages[final_t - ct].append(q)
    age_para = {a: sc(qs) for a, qs in sorted(ages.items())}
    old_rep_pa, old_non_pa = sc(old_rep), sc(old_non)   # score BEFORE del M (closure captures M)
    # R40->s3: per-QA FINAL retention (all committed items) keyed by qid, for surprise-vs-retention join
    cl = committed_log
    perqa = {}
    if cl:
        ge = gen(M, [QT.format(q=q["eval_question"]) for (q, ct, rep) in cl])
        go = gen(M, [QT.format(q=q["question"]) for (q, ct, rep) in cl])
        perqa = {q["qid"]: dict(final_eval_em=em(ge[i], q["answers"]), final_orig_em=em(go[i], q["answers"]),
                                replayed=bool(rep), commit_t=ct) for i, (q, ct, rep) in enumerate(cl)}
    del M; torch.cuda.empty_cache()
    return dict(final_em=round(f_em, 3), final_f1=round(f_f1, 3),
                final_para_em=round(fp_em, 3), final_para_f1=round(fp_f1, 3),
                replay_k=REPLAY_K, no_anchor=no_anchor,
                n_old_replayed=len(old_rep), n_old_nonreplayed=len(old_non),
                old_replayed_para_em=old_rep_pa, old_nonreplayed_para_em=old_non_pa,
                age_para=age_para, per_stream=per_stream, perqa=perqa)

def surprise_summary(results):
    # WITHIN-source: for each arm, pool OLD-only NON-replayed committed items (ct < final stream), join base
    # para-surprise bits with final eval retention; report tercile(lo/mid/hi bits), retention, point-biserial
    # corr, retained/bit — plus a variant excluding base-already-correct items (write-difficulty, not prior).
    # codex: compute final_t PER SEED from that seed's committed items (not a global ACTUAL_STREAMS, which
    # only reflects the last processed seed and breaks under variable survived-stream counts).

    def stats(pairs):                                   # pairs: list of (bits, retained_0_1)
        n = len(pairs)
        if n < 3:
            return None
        b = [p[0] for p in pairs]; ret = [p[1] for p in pairs]
        order = sorted(range(n), key=lambda i: b[i]); t = max(1, n // 3)
        terc = [round(sum(ret[i] for i in order[a:a + t]) / max(len(order[a:a + t]), 1), 3)
                for a in (0, t, 2 * t)]
        mb = sum(b) / n; mr = sum(ret) / n
        sb = (sum((x - mb) ** 2 for x in b) / n) ** 0.5 or 1.0
        sr = (sum((x - mr) ** 2 for x in ret) / n) ** 0.5 or 1.0
        corr = round(sum((b[i] - mb) * (ret[i] - mr) for i in range(n)) / (n * sb * sr), 4)
        return dict(n=n, retained_overall=round(mr, 3), by_bits_tercile=terc,
                    corr_bits_vs_retained=corr, retained_per_bit=round(sum(ret) / (sum(b) or 1.0), 5),
                    mean_bits=round(mb, 3))
    summ = {}
    for arm in ARMS:
        allp, exclp = [], []
        for seed, base in SURPRISE_BASE.items():
            if seed >= len(results.get(arm, [])):
                continue
            pq = results[arm][seed].get("perqa") or {}
            if not pq:
                continue
            seed_final_t = max(r["commit_t"] for r in pq.values())   # per-seed final stream
            for qid, rec in pq.items():
                if qid in base and rec["commit_t"] < seed_final_t and not rec["replayed"]:
                    pair = (base[qid]["base_bits_para"], rec["final_eval_em"])
                    allp.append(pair)
                    if not base[qid]["base_em_para"]:
                        exclp.append(pair)
        s = stats(allp)
        if s is None:
            continue
        ex = stats(exclp) or {}
        summ[arm] = dict(**s, excl_corr=ex.get("corr_bits_vs_retained"),
                         excl_tercile=ex.get("by_bits_tercile"), excl_n=ex.get("n"))
    return summ

def dump(results, nseeds):
    summ = {}
    for arm in ARMS:
        rs = results[arm]
        if not rs:
            continue
        def av(k):
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None
        summ[arm] = dict(final_em=av("final_em"), final_f1=av("final_f1"),
                         final_para_em=av("final_para_em"), final_para_f1=av("final_para_f1"),
                         replay_k=rs[-1].get("replay_k"), no_anchor=rs[-1].get("no_anchor"),
                         n_old_replayed=rs[-1].get("n_old_replayed"), n_old_nonreplayed=rs[-1].get("n_old_nonreplayed"),
                         old_replayed_para_em=av("old_replayed_para_em"),
                         old_nonreplayed_para_em=av("old_nonreplayed_para_em"),
                         per_stream=rs[-1].get("per_stream"))
    # per-seed rows (codex R38B gate 3): the whole claim turns on seed-level sign agreement — persist them all
    per_seed = {arm: [{k: v for k, v in r.items() if k not in ("per_stream", "perqa")} for r in results[arm]]
                for arm in ARMS if results.get(arm)}
    ssum = surprise_summary(results)
    json.dump({"config": dict(source=SOURCE, synth_variant=SYNTH_VARIANT, streams=STREAMS, arts=ARTS,
                              qa=QA_PER, seeds=nseeds, arms=ARMS,
                              actual_streams=ACTUAL_STREAMS, survived_articles=SURVIVED_ARTS),
               "summary": summ, "per_seed": per_seed, "surprise": ssum}, open(OUT, "w"), indent=1)
    # per-QA sidecar: join FROZEN-BASE surprise/EM with each arm's per-QA final retention (source contrast
    # analysis is done OFFLINE across the squad and synth runs).
    side = []
    for seed, base in SURPRISE_BASE.items():
        arms_ret = {arm: (results[arm][seed].get("perqa") or {})
                    for arm in ARMS if seed < len(results[arm]) and results[arm][seed].get("perqa")}
        for qid, rec in base.items():
            row = dict(qid=qid, **rec)
            for arm, pq in arms_ret.items():
                if qid in pq:
                    row[f"{arm}__eval_em"] = pq[qid]["final_eval_em"]
                    row[f"{arm}__replayed"] = pq[qid]["replayed"]
                    row[f"{arm}__commit_t"] = pq[qid]["commit_t"]
            side.append(row)
    json.dump(dict(source=SOURCE, synth_variant=SYNTH_VARIANT, rows=side), open(OUT.replace(".json", ".perqa.json"), "w"))
    if ssum:
        print("  -- R40 surprise gate (WITHIN-source: base-para surprise vs OLD-only NONreplayed eval retention) --")
        for arm, sv in ssum.items():
            print(f"     {arm:26s} ret={sv['retained_overall']} tercile(lo/mid/hi bits)={sv['by_bits_tercile']} "
                  f"corr={sv['corr_bits_vs_retained']} n={sv['n']} | excl-base-correct: "
                  f"corr={sv.get('excl_corr')} tercile={sv.get('excl_tercile')} n={sv.get('excl_n')}", flush=True)
    print("RESULT_JSON " + json.dumps({a: {"final_em": summ[a]["final_em"],
                                           "final_para_em": summ[a]["final_para_em"]} for a in summ}), flush=True)

if __name__ == "__main__":
    main()
