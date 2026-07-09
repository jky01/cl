"""R44-schema_comp — MINER + AUDIT (inference-only, no training yet).

Question (codex-converged 2026-07-09): does consolidating N same-RELATION facts into weights make the
(N+1)th same-relation base-hard binding CHEAPER to write / more robust to retain? I.e. can lifetime cost
move from item-ledger O(#facts) toward reusable structure? This is the "existing-schema ACTIVATION" case
(the pretrained base already has latent relation structure) that R32/R34 do NOT kill — unlike automatic
NEW-schema acquisition, which they do.

R43-ladder killed surprise-as-ROUTER on real text (unimodal -> no safe-to-skip class); random@0.5 ≈ full.
So the open cost-curve question is schema COMPRESSION, not item routing. R44 headline (codex): WITHIN-schema
dose-response — B80(N) (budget to hit a retention threshold) FALLS with same-relation density N, and falls
MORE for schema_commit than for item_k_matched. Growth stays OUT.

THIS FILE = stage 1 only: build a relation-labeled, base-hard, RAG-answerable, self-contained, confound-
logged dataset and AUDIT whether it can form matched blocks. Training GO only if the audit passes
(codex: >=5 relations x >=20 usable, >=2 relations per answer-kind bucket for the deranged/item controls).
Source priority (codex): real T-REx/LAMA triples > real-KB triples + templated evidence > controlled fake.

Model: Qwen2.5-0.5B (base scoring), Qwen2.5-3B-Instruct (paraphrase). Reuses the WikiBridge eval contract.
"""
import os, json, re, random, collections, math
import torch
from s3.wikibridge import normalize, em, f1, QT, QT2, RT, gen, qa_answer_bits, load_model, tok, device
from s3.census import Instruct, RSYS

SEED = int(os.environ.get("SC_SEED", 0))
PER_REL = int(os.environ.get("SC_PER_REL", 200))     # candidate triples pulled per relation before filtering
WANT = int(os.environ.get("SC_WANT", 24))            # target usable base-hard items per relation
OUT = os.environ.get("SC_OUT", "schema_audit.json")
PARA_NAME = os.environ.get("WB_PARA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAXNEW = 12

# ------------------------- relation registry -------------------------
# 6 relations forming 3 answer-kind buckets x 2 relations each (codex: >=2 per bucket so schema_shuffle_
# same_kind has a real same-kind neighbor). q=closed-book question, p=held-out paraphrase seed, st=evidence
# statement (real-KB / templated evidence: the middle rung when natural T-REx sentences are thin), kind=answer
# bucket. {s}=subject {o}=object.
RELATIONS = {
    "P36":  dict(kind="place",    q="What is the capital of {s}?",           p="Which city is the capital of {s}?",       st="The capital of {s} is {o}."),
    "P17":  dict(kind="place",    q="In which country is {s} located?",       p="{s} is located in which country?",        st="{s} is located in the country of {o}."),
    "P37":  dict(kind="language", q="What is the official language of {s}?",   p="Which language is official in {s}?",       st="The official language of {s} is {o}."),
    "P103": dict(kind="language", q="What is the native language of {s}?",     p="Which language is the native language of {s}?", st="The native language of {s} is {o}."),
    "P50":  dict(kind="person",   q="Who is the author of {s}?",              p="Who wrote {s}?",                          st="The author of {s} is {o}."),
    "P86":  dict(kind="person",   q="Who is the composer of {s}?",            p="Who composed {s}?",                       st="The composer of {s} is {o}."),
}

# ------------------------- source loaders (priority order) -------------------------
def _norm_row(r):
    """extract (predicate_id, subject, object) across the field-name variants T-REx/LAMA mirrors use."""
    pid = r.get("predicate_id") or r.get("relation") or r.get("rel") or r.get("property")
    sub = r.get("sub_label") or r.get("subject") or r.get("sub") or r.get("head")
    obj = r.get("obj_label") or r.get("object") or r.get("obj") or r.get("tail")
    evi = None
    ev = r.get("evidences") or r.get("masked_sentences") or r.get("masked_sentence")
    if isinstance(ev, list) and ev:
        e0 = ev[0]
        evi = e0.get("masked_sentence") if isinstance(e0, dict) else e0
    elif isinstance(ev, str):
        evi = ev
    return pid, sub, obj, evi

def load_trex(rng):
    """try real T-REx/LAMA relation triples; keep only rows whose predicate is in RELATIONS."""
    from datasets import load_dataset
    cands = [("lama", "trex"), ("facebook/lama", "trex"), ("lama", None),
             ("relbert/t_rex", None), ("community-datasets/lama", "trex")]
    for name, cfg in cands:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True) if cfg else \
                 load_dataset(name, split="train", streaming=True)
            buf = collections.defaultdict(list); seen = set(); n = 0
            for r in ds:
                pid, sub, obj, evi = _norm_row(r)
                if pid not in RELATIONS or not sub or not obj:
                    continue
                key = (pid, sub)
                if key in seen:
                    continue
                if not (1 <= len(tok(str(obj), add_special_tokens=False).input_ids) <= 6):
                    continue
                seen.add(key)
                buf[pid].append(dict(sub=str(sub), obj=str(obj), evi=evi))
                n += 1
                if all(len(buf[p]) >= PER_REL for p in RELATIONS) or n > 400000:
                    break
            if sum(len(v) for v in buf.values()) >= 3 * len(RELATIONS):
                print(f"  SOURCE=trex[{name}/{cfg}] loaded { {p: len(buf[p]) for p in buf} }", flush=True)
                return buf, "trex"
        except Exception as e:
            print(f"  trex source {name}/{cfg} FAIL: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return None, None

# curated fallback (real-KB triples; evidence templated) — guarantees the miner runs if HF is unavailable.
# base-hard filtering picks the obscure ones; these are only the candidate pool.
_CURATED = {
    "P36": [("Tuvalu","Funafuti"),("Kiribati","Tarawa"),("Palau","Ngerulmud"),("Nauru","Yaren"),
            ("Bhutan","Thimphu"),("Brunei","Bandar Seri Begawan"),("Suriname","Paramaribo"),
            ("Eritrea","Asmara"),("Comoros","Moroni"),("Vanuatu","Port Vila"),("Belize","Belmopan"),
            ("Bolivia","Sucre"),("Myanmar","Naypyidaw"),("Kazakhstan","Astana"),("Malawi","Lilongwe"),
            ("Botswana","Gaborone"),("Lesotho","Maseru"),("Djibouti","Djibouti"),("Guyana","Georgetown"),
            ("Tajikistan","Dushanbe"),("Turkmenistan","Ashgabat"),("Zambia","Lusaka"),("Moldova","Chisinau"),
            ("Montenegro","Podgorica"),("Kyrgyzstan","Bishkek"),("Mauritania","Nouakchott")],
    "P17": [("Timbuktu","Mali"),("Samarkand","Uzbekistan"),("Maracaibo","Venezuela"),("Surabaya","Indonesia"),
            ("Chittagong","Bangladesh"),("Kaohsiung","Taiwan"),("Fez","Morocco"),("Cusco","Peru"),
            ("Aleppo","Syria"),("Mombasa","Kenya"),("Galle","Sri Lanka"),("Bruges","Belgium"),
            ("Oaxaca","Mexico"),("Kandy","Sri Lanka"),("Trondheim","Norway"),("Gdansk","Poland"),
            ("Esfahan","Iran"),("Mandalay","Myanmar"),("Arequipa","Peru"),("Nuremberg","Germany"),
            ("Salvador","Brazil"),("Kazan","Russia"),("Lviv","Ukraine"),("Medan","Indonesia"),
            ("Antwerp","Belgium"),("Kochi","India")],
    "P37": [("Suriname","Dutch"),("Angola","Portuguese"),("Andorra","Catalan"),("Palau","Palauan"),
            ("Bhutan","Dzongkha"),("Eritrea","Tigrinya"),("Paraguay","Guarani"),("Kiribati","Gilbertese"),
            ("Belarus","Belarusian"),("Moldova","Romanian"),("Malta","Maltese"),("Rwanda","Kinyarwanda"),
            ("Madagascar","Malagasy"),("Kazakhstan","Kazakh"),("Mongolia","Mongolian"),("Laos","Lao"),
            ("Cambodia","Khmer"),("Armenia","Armenian"),("Georgia","Georgian"),("Latvia","Latvian"),
            ("Estonia","Estonian"),("Iceland","Icelandic"),("Albania","Albanian"),("Slovenia","Slovene"),
            ("Tajikistan","Tajik"),("Turkmenistan","Turkmen")],
    "P103": [("Franz Kafka","German"),("Joseph Conrad","Polish"),("Vladimir Nabokov","Russian"),
             ("Rabindranath Tagore","Bengali"),("Naguib Mahfouz","Arabic"),("Pablo Neruda","Spanish"),
             ("Orhan Pamuk","Turkish"),("Ivo Andric","Serbian"),("Halldor Laxness","Icelandic"),
             ("Czeslaw Milosz","Polish"),("Kobo Abe","Japanese"),("Lu Xun","Chinese"),
             ("Nikos Kazantzakis","Greek"),("Sandor Marai","Hungarian"),("Bruno Schulz","Polish"),
             ("Andrei Platonov","Russian"),("Machado de Assis","Portuguese"),("Italo Svevo","Italian"),
             ("Knut Hamsun","Norwegian"),("Yasar Kemal","Turkish"),("Tove Jansson","Swedish"),
             ("Cesare Pavese","Italian"),("Bohumil Hrabal","Czech"),("Mikhail Bulgakov","Russian"),
             ("Elias Canetti","German"),("Stanislaw Lem","Polish")],
    "P50": [("The Leopard","Giuseppe Tomasi di Lampedusa"),("Petals of Blood","Ngugi wa Thiong'o"),
            ("Kokoro","Natsume Soseki"),("The Radetzky March","Joseph Roth"),("Independent People","Halldor Laxness"),
            ("The Master and Margarita","Mikhail Bulgakov"),("Snow Country","Yasunari Kawabata"),
            ("The Tin Drum","Gunter Grass"),("Season of Migration to the North","Tayeb Salih"),
            ("Pedro Paramo","Juan Rulfo"),("The Street of Crocodiles","Bruno Schulz"),
            ("Hunger","Knut Hamsun"),("The Bridge on the Drina","Ivo Andric"),("Blindness","Jose Saramago"),
            ("Zeno's Conscience","Italo Svevo"),("The Notebook","Agota Kristof"),
            ("Life and Fate","Vasily Grossman"),("Memed My Hawk","Yasar Kemal"),
            ("The Doll","Boleslaw Prus"),("Fatelessness","Imre Kertesz"),("Austerlitz","W. G. Sebald"),
            ("The Book of Disquiet","Fernando Pessoa"),("Embers","Sandor Marai"),
            ("Too Loud a Solitude","Bohumil Hrabal"),("The Vegetarian","Han Kang"),("Kristin Lavransdatter","Sigrid Undset")],
    "P86": [("Bolero","Maurice Ravel"),("The Planets","Gustav Holst"),("Finlandia","Jean Sibelius"),
            ("Peer Gynt","Edvard Grieg"),("Ma Vlast","Bedrich Smetana"),("Scheherazade","Nikolai Rimsky-Korsakov"),
            ("Carmina Burana","Carl Orff"),("The Firebird","Igor Stravinsky"),("Enigma Variations","Edward Elgar"),
            ("Pines of Rome","Ottorino Respighi"),("Gymnopedies","Erik Satie"),("Clair de Lune","Claude Debussy"),
            ("Pavane","Gabriel Faure"),("Nimrod","Edward Elgar"),("Vltava","Bedrich Smetana"),
            ("Danse Macabre","Camille Saint-Saens"),("Adagio for Strings","Samuel Barber"),
            ("Appalachian Spring","Aaron Copland"),("Concierto de Aranjuez","Joaquin Rodrigo"),
            ("The Lark Ascending","Ralph Vaughan Williams"),("Cavalleria Rusticana","Pietro Mascagni"),
            ("Also sprach Zarathustra","Richard Strauss"),("Pictures at an Exhibition","Modest Mussorgsky"),
            ("Karelia Suite","Jean Sibelius"),("Symphonie Fantastique","Hector Berlioz"),("Gayane","Aram Khachaturian")],
}
def load_curated(rng):
    buf = {p: [dict(sub=s, obj=o, evi=None) for (s, o) in _CURATED[p]] for p in RELATIONS}
    print(f"  SOURCE=curated { {p: len(buf[p]) for p in buf} }", flush=True)
    return buf, "curated"

# ------------------------- familiarity proxy: subject NLL in a NEUTRAL carrier -------------------------
NEUTRAL_CARRIER = "Here is some information about {s}."   # codex: measure subject familiarity OUTSIDE the relation Q
@torch.no_grad()
def subject_bits(base, subs):
    res = []
    for i in range(0, len(subs), 16):
        chunk = subs[i:i + 16]
        pre = "Here is some information about"
        full = [pre + " " + s + "." for s in chunk]
        tok.padding_side = "right"
        e = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        pl = len(tok(pre).input_ids)
        logp = torch.log_softmax(base(**e, use_cache=False).logits[:, :-1].float(), -1)
        labels = e["input_ids"][:, 1:].clone()
        labels[e["attention_mask"][:, 1:] == 0] = -100
        for r in range(len(chunk)):
            lab = labels[r].clone(); lab[:max(0, pl - 1)] = -100
            m = lab != -100
            nll = -logp[r][m].gather(1, lab[m][:, None]).squeeze(1) if bool(m.any()) else torch.tensor([0.])
            ntok = int(m.sum().item()) or 1
            res.append(round((nll.sum() / math.log(2)).item() / ntok, 3))
    tok.padding_side = "left"
    return res

def main():
    rng = random.Random(4400 + SEED)
    print(f"SCHEMA_COMP miner (base=Qwen2.5-0.5B, {device}) relations={list(RELATIONS)} want={WANT}/rel", flush=True)
    base = load_model()
    src = os.environ.get("SC_SOURCE", "auto")
    buf, used = (load_trex(rng) if src in ("auto", "trex") else (None, None))
    if buf is None:
        buf, used = load_curated(rng)
    for p in buf:
        rng.shuffle(buf[p]); buf[p] = buf[p][:PER_REL]

    # build candidate items with question/paraphrase/evidence
    items = []
    for pid, rows in buf.items():
        R = RELATIONS[pid]
        for r in rows:
            s, o = r["sub"], r["obj"]
            evi = r.get("evi") or R["st"].format(s=s, o=o)     # natural evidence if present else templated
            items.append(dict(pid=pid, kind=R["kind"], sub=s, answers=[o],
                              question=R["q"].format(s=s), eval_question=R["p"].format(s=s),
                              context=evi, src=used))
    print(f"  candidate items: {len(items)}", flush=True)

    # base-hard screen: closed-book WRONG on Q and paraphrase, RAG-answerable with evidence
    e1 = [em(x, q["answers"]) for x, q in zip(gen(base, [QT.format(q=q["question"]) for q in items]), items)]
    ep = [em(x, q["answers"]) for x, q in zip(gen(base, [QT.format(q=q["eval_question"]) for q in items]), items)]
    rg = [em(x, q["answers"]) for x, q in zip(gen(base, [RT.format(c=q["context"], q=q["question"]) for q in items]), items)]
    for q, a, b, c in zip(items, e1, ep, rg):
        q["base_em_orig"], q["base_em_para"], q["rag_em"] = a, b, c
    hard = [q for q in items if q["base_em_orig"] == 0 and q["base_em_para"] == 0 and q["rag_em"] == 1]
    print(f"  base-hard & RAG-answerable: {len(hard)}/{len(items)}", flush=True)

    # confound features: answer bpt (orig+para), subject familiarity bpt, token counts
    bo = qa_answer_bits(base, hard, "question"); bp = qa_answer_bits(base, hard, "eval_question")
    sb = subject_bits(base, [q["sub"] for q in hard])
    for q, o_, p_, s_ in zip(hard, bo, bp, sb):
        q["ans_ntok"] = o_[1]
        q["bpt_orig"] = round(o_[0] / max(o_[1], 1), 3)
        q["bpt_para"] = round(p_[0] / max(p_[1], 1), 3)
        q["subj_bits"] = s_
        q["subj_ntok"] = len(tok(q["sub"], add_special_tokens=False).input_ids)

    # held-out paraphrase via 3B (parity with census/wikibridge internalization surface)
    inst = Instruct()
    paras = inst.chat([(RSYS, q["question"]) for q in hard], max_new=40, bs=16)
    for q, pp in zip(hard, paras):
        q["eval_question_3b"] = (pp.split("\n")[0].strip() or q["eval_question"])
    inst.free()

    # leakage flags + dedup
    seen = set()
    for q in hard:
        na = normalize(q["answers"][0])
        q["ans_in_q"] = int(na in normalize(q["question"]))
        key = (q["pid"], normalize(q["sub"]))
        q["dup"] = int(key in seen); seen.add(key)

    usable = [q for q in hard if not q["dup"] and not q["ans_in_q"]]
    byrel = collections.defaultdict(list)
    for q in usable:
        byrel[q["pid"]].append(q)

    # ------- AUDIT report -------
    def dist(qs, k):
        v = sorted(q[k] for q in qs)
        return None if not v else dict(n=len(v), mean=round(sum(v) / len(v), 2),
                                       p10=v[len(v) // 10], p50=v[len(v) // 2], p90=v[min(len(v) - 1, 9 * len(v) // 10)])
    audit = {}
    for pid in RELATIONS:
        qs = byrel.get(pid, [])
        audit[pid] = dict(kind=RELATIONS[pid]["kind"], n_usable=len(qs),
                          ans_bpt=dist(qs, "bpt_para"), subj_bits=dist(qs, "subj_bits"),
                          ans_ntok=dist(qs, "ans_ntok"), subj_ntok=dist(qs, "subj_ntok"))
    kinds = collections.defaultdict(list)
    for pid in RELATIONS:
        if audit[pid]["n_usable"] >= WANT // 2:
            kinds[RELATIONS[pid]["kind"]].append(pid)
    ok_rels = [p for p in RELATIONS if audit[p]["n_usable"] >= WANT]
    ok_buckets = {k: v for k, v in kinds.items() if len(v) >= 2}
    GO = len(ok_rels) >= 5 and len(ok_buckets) >= 2
    audit_meta = dict(source=used, n_candidate=len(items), n_hard=len(hard), n_usable=len(usable),
                      relations_ge_want=ok_rels, want=WANT,
                      same_kind_buckets_ge2={k: v for k, v in kinds.items()},
                      GO_for_training=GO)
    json.dump(dict(meta=audit_meta, per_relation=audit,
                   items={q["pid"] + ":" + q["sub"]: {k: v for k, v in q.items() if k != "context"} for q in usable}),
              open(OUT, "w"), indent=1)
    with open(OUT.replace(".json", ".jsonl"), "w") as f:
        for q in usable:
            f.write(json.dumps({k: v for k, v in q.items() if k != "context"}) + "\n")
    print(f"  AUDIT source={used} candidate={len(items)} hard={len(hard)} usable={len(usable)}", flush=True)
    for pid in RELATIONS:
        a = audit[pid]
        print(f"    {pid}[{a['kind']:8s}] usable={a['n_usable']:3d} "
              f"ans_bpt={a['ans_bpt']} subj_bits={a['subj_bits']}", flush=True)
    print(f"  relations>=WANT({WANT}): {ok_rels}", flush=True)
    print(f"  same-kind buckets>=2: {ok_buckets}", flush=True)
    print(f"  GO_for_training={GO}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
