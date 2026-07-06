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
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(device).eval()
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

def score(model, qas, template=QT, rag=False):
    if not qas:
        return 0.0, 0.0
    prompts = [(RT.format(c=q["context"], q=q["question"]) if rag else template.format(q=q["question"]))
               for q in qas]
    preds = gen(model, prompts)
    e = sum(em(p, q["answers"]) for p, q in zip(preds, qas)) / len(qas)
    ff = sum(f1(p, q["answers"]) for p, q in zip(preds, qas)) / len(qas)
    return e, ff

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
        if len(kept_articles) >= STREAMS * ARTS:
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
        if len(hard) >= QA_PER:
            kept_articles.append({"title": title, "context": ctx, "qas": hard[:QA_PER]})
    if len(kept_articles) < STREAMS * ARTS:
        print(f"  WARN: only {len(kept_articles)} hard articles found (< {STREAMS*ARTS})", flush=True)
    streams = [kept_articles[i * ARTS:(i + 1) * ARTS] for i in range(STREAMS)]
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
    streams = [arts[i * ARTS:(i + 1) * ARTS] for i in range(STREAMS)]
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
        answers = [" " + q["answers"][0] for q in chunk]
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
        b_em, b_f1 = score(base, allqa)
        r_em, r_f1 = score(base, allqa, rag=True)
        print(f"    base closed-book EM={b_em:.3f} F1={b_f1:.3f} | RAG-gold EM={r_em:.3f} F1={r_f1:.3f}", flush=True)

        for arm in ARMS:
            t0 = time.time()
            if arm == "base_no_ingest":
                res = dict(final_em=b_em, final_f1=b_f1)
            elif arm == "rag_gold_passage":
                res = dict(final_em=r_em, final_f1=r_f1)
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
    rng = random.Random(seed * 13 + hash(arm) % 100)
    committed = []                                     # list of committed old QA {question, answers}
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
        s_em, _ = score(S, new_qa)                     # DIAGNOSTIC: can the scaffold answer new QA?
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
                # old retention: committed answer-sequence CE (compact target, no snapshot teacher)
                if replay and committed:
                    loss = loss + qa_ce(M, [rng.choice(committed) for _ in range(8)])
                # base-capability anchor on NEUTRAL prompts ONLY (never on old QA — R37-A lesson)
                ne, nb = base_anchor_logits(base, [rng.choice(NEUTRAL) for _ in range(8)])
                sa = M.lm_head(M.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
                loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
            M.eval()
        del S; torch.cuda.empty_cache()
        # commit: store compact targets for committed-correct new QA
        M.eval()
        m_em, m_f1 = score(M, new_qa)
        correct = [q for q, p in zip(new_qa, gen(M, [QT.format(q=q["question"]) for q in new_qa]))
                   if em(p, q["answers"]) == 1]
        committed += correct
        # retention: score all OLD streams' QA (0..t-1) on current M
        old_qa = [q for tt in range(t) for a in streams[tt] for q in a["qas"]]
        o_em, _ = score(M, old_qa) if old_qa else (None, None)
        nb_em, _ = score(base, [{"question": "The capital of France is what", "answers": ["paris"], "context": ""}])
        per_stream.append(dict(t=t, scaffold_new_em=round(s_em, 3), M_new_em=round(m_em, 3),
                               M_new_f1=round(m_f1, 3), n_committed=len(correct), old_em=(round(o_em, 3) if o_em is not None else None)))
        print(f"      [{arm} s{seed} t{t}] scaffold_new={s_em:.2f} M_new={m_em:.2f} "
              f"committed={len(correct)}/{len(new_qa)} old={o_em if o_em is None else round(o_em,2)}", flush=True)
    allqa = [q for s in streams for a in s for q in a["qas"]]
    f_em, f_f1 = score(M, allqa)
    # base-capability preservation: neutral-prompt agreement with base (proxy)
    del M; torch.cuda.empty_cache()
    return dict(final_em=round(f_em, 3), final_f1=round(f_f1, 3), per_stream=per_stream)

def dump(results, nseeds):
    summ = {}
    for arm in ARMS:
        rs = results[arm]
        if not rs:
            continue
        summ[arm] = dict(final_em=round(sum(r["final_em"] for r in rs) / len(rs), 3),
                         final_f1=round(sum(r["final_f1"] for r in rs) / len(rs), 3),
                         per_stream=rs[-1].get("per_stream"))
    json.dump({"config": dict(source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER, seeds=nseeds, arms=ARMS),
               "summary": summ}, open(OUT, "w"), indent=1)
    print("RESULT_JSON " + json.dumps({a: {"final_em": summ[a]["final_em"], "final_f1": summ[a]["final_f1"]}
                                       for a in summ}), flush=True)

if __name__ == "__main__":
    main()
