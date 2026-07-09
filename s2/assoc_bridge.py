"""R46-assoc_bridge — does a SHARED associative bridge-node enable latent 2-hop composition where
independent bindings (R32) failed? (codex-designed probe 2026-07-09.22.05.48)

Motivation (brainstorm rounds 1-3): transformers ARE associative memories (attention ≈ modern Hopfield);
we've been doing DATABASE continual learning on an ASSOCIATIVE substrate. Composition = graph traversal.
R32's two-hop null (store A->B, B->C, can't get A->C) may be because the bridge B was two INDEPENDENT
bindings, not a shared associative attractor. Probe: make B a shared node WITHOUT ever training A->C, and
see if attention's pattern-completion does the traversal. The decisive design separates "bridge geometry"
from "we just trained composition" and from "we just saw B more often".

Task (forks s2/composition_grok generator): many A_i per bridge B_j, one C_j per bridge; atomic
"A's friend is B", "B's pet is C"; held-out eval "A's friend's pet is" -> C, NEVER trained in main arms.
Single-token entities → argmax + gold-prob eval, no CoT, no memory, one dense checkpoint at eval.

Arms (codex): unique_bridge_atomic (R32 floor) | shared_bridge_atomic (hub multiplicity alone) |
shared_bridge_attractor (+ successor-feature bridge-unification loss) | deranged_attractor (align to WRONG
bridge; control) | freq_matched_unique (same data volume, no hub; rules out raw exposure) |
r34_direct_2hop (upper bound, trains sibling 2-hop; NOT a project-valid explanation).

Successor-feature loss (codex): L += lambda * || P h("A_i's friend is") - stopgrad(P target) ||^2, target =
h("B_j's pet is") [bpet] or h("B_j") [identity]; P is a train-time projection head, DELETED before eval.
This distills a one-step latent backup into hidden geometry — no external graph at inference.

Pass (codex): atomic recall >=0.95 all main arms; shared_bridge_atomic OR shared_bridge_attractor beats
unique_bridge_atomic AND deranged by >=+0.20 held-out A->C (or >=0.30 absolute); transfers to paraphrase;
NO A->C target/scratchpad/task-id/graph at inference.

  python -m s2.assoc_bridge
  env: AB_BRIDGE(60) AB_APER(6) AB_STEPS(8000) AB_EVAL(500) AB_SEEDS(2) AB_LR(1e-4) AB_BS(64)
       AB_LAMBDA(1.0) AB_ATTR(bpet|identity) AB_PROJ(256) AB_ARMS(...) AB_SMOKE(0)
"""
from __future__ import annotations
import os, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
BRIDGE = int(os.environ.get("AB_BRIDGE", 60))
APER = int(os.environ.get("AB_APER", 6))
STEPS = int(os.environ.get("AB_STEPS", 8000))
EVAL = int(os.environ.get("AB_EVAL", 500))
SEEDS = int(os.environ.get("AB_SEEDS", 2))
LR = float(os.environ.get("AB_LR", 1e-4))
BS = int(os.environ.get("AB_BS", 64))
LAMBDA = float(os.environ.get("AB_LAMBDA", 1.0))
ATTR = os.environ.get("AB_ATTR", "bpet")             # bpet | identity
PROJ = int(os.environ.get("AB_PROJ", 256))
OUT = os.environ.get("AB_OUT", "assoc_bridge_result.json")
ALL_ARMS = ["unique_bridge_atomic", "shared_bridge_atomic", "shared_bridge_attractor",
            "deranged_attractor", "freq_matched_unique", "r34_direct_2hop"]
ARMS = os.environ.get("AB_ARMS", ",".join(ALL_ARMS)).split(",")
SMOKE = int(os.environ.get("AB_SMOKE", 0))
if SMOKE:
    BRIDGE, APER, STEPS, EVAL, SEEDS = 12, 4, 600, 200, 1

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

def one_tok(s):
    t = tok(" " + s, add_special_tokens=False).input_ids
    return t[0] if len(t) == 1 else None

def harvest_pool():
    pool, seen = [], set()
    for tid in range(min(len(tok), 152000)):
        try:
            s = tok.convert_ids_to_tokens(tid)
        except Exception:
            continue
        if not isinstance(s, str):
            continue
        w = s.replace("Ġ", " ")
        if w.startswith(" ") and w[1:].isalpha() and len(w) >= 4 and w[1].isupper():
            name = w[1:]
            if name.lower() in seen:
                continue
            if one_tok(name) is not None:
                pool.append(name); seen.add(name.lower())
    return pool

def q_r1(a): return f"{a}'s friend is"                 # atomic A -> B
def q_r2(b): return f"{b}'s pet is"                     # atomic B -> C
def q_2hop(a): return f"{a}'s friend's pet is"         # composition A -> C
def q_2hop_para(a): return f"the pet of {a}'s friend is"   # held-out paraphrase surface
def q_ident(b): return f"{b}"                           # bridge identity (attractor target variant)

def build_task(seed, pool, n_bridge, aper, train2):
    rng = random.Random(5000 + seed + n_bridge * 7 + aper * 13)
    p = pool[:]; rng.shuffle(p)
    need = n_bridge * 2 + n_bridge * aper
    assert len(p) >= need, f"pool too small: need {need} have {len(p)}"
    Cs = p[:n_bridge]; Bs = p[n_bridge:2 * n_bridge]; As = p[2 * n_bridge:2 * n_bridge + n_bridge * aper]
    r2 = {Bs[j]: Cs[j] for j in range(n_bridge)}
    r1 = {}; a_of_b = {j: [] for j in range(n_bridge)}
    for i, a in enumerate(As):
        j = i // aper
        r1[a] = Bs[j]; a_of_b[j].append(a)
    # in main arms NO A->C is trained → every A is a held-out 2-hop. r34_direct trains a sibling split.
    train_A, held_A = [], []
    if train2:
        for j in range(n_bridge):
            grp = a_of_b[j][:]; rng.shuffle(grp)
            n_hold = min(len(grp) - 1, max(1, int(round(0.3 * len(grp)))))
            held_A += grp[:n_hold]; train_A += grp[n_hold:]
    else:
        held_A = list(As)
    atomic = [(q_r1(a), one_tok(r1[a])) for a in As] + [(q_r2(b), one_tok(r2[b])) for b in Bs]
    tr2 = [(q_2hop(a), one_tok(r2[r1[a]])) for a in train_A]
    held2 = [(q_2hop(a), one_tok(r2[r1[a]]), a) for a in held_A]
    held2_para = [(q_2hop_para(a), one_tok(r2[r1[a]]), a) for a in held_A]
    # derange: each held-out gets a WRONG single-token C (assert no leak / no accidental truth)
    allC = [one_tok(c) for c in Cs]; rdg = random.Random(9000 + seed)
    der = []
    for (pm, g, a) in held2:
        cand = [c for c in allC if c != g]
        der.append((pm, rdg.choice(cand) if cand else g))
    # leak assert: C never a completion of any A's atomic prompt
    acomp = {q_r1(a): one_tok(r1[a]) for a in As}
    for (pm, g, a) in held2:
        assert acomp.get(q_r1(a)) != g, "leak: A atomic completion == 2hop answer"
    return dict(atomic=atomic, tr2=tr2, held2=held2, held2_para=held2_para, derange=der,
                r1=r1, r2=r2, As=As, held_A=held_A, n_train2=len(tr2), n_held2=len(held2))

@torch.no_grad()
def hstate(m, prompts):
    outs = []
    for i in range(0, len(prompts), 256):
        e = tok(prompts[i:i + 256], return_tensors="pt", padding=True).to(device)
        h = m.model(**e, use_cache=False).last_hidden_state[:, -1].float()
        outs.append(h)
    return torch.cat(outs, 0)

@torch.no_grad()
def acc(m, items):
    if not items:
        return 0.0, 0.0
    ok = 0.0; gp = 0.0; n = 0
    for i in range(0, len(items), 256):
        ch = items[i:i + 256]
        e = tok([x[0] for x in ch], return_tensors="pt", padding=True).to(device)
        lg = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float()
        gold = torch.tensor([x[1] for x in ch], device=device)
        ok += (lg.argmax(-1) == gold).sum().item()
        gp += F.softmax(lg, -1).gather(1, gold[:, None]).sum().item()
        n += len(ch)
    return ok / n, gp / n

def run(seed, arm, pool):
    cfg = dict(unique_bridge_atomic=(BRIDGE, 1, False, None),
               shared_bridge_atomic=(BRIDGE, APER, False, None),
               shared_bridge_attractor=(BRIDGE, APER, False, "correct"),
               deranged_attractor=(BRIDGE, APER, False, "wrong"),
               freq_matched_unique=(BRIDGE * APER, 1, False, None),
               r34_direct_2hop=(BRIDGE, APER, True, None))[arm]
    n_bridge, aper, train2, attr = cfg
    task = build_task(seed, pool, n_bridge, aper, train2)
    atomic, tr2, held2, held2p, der = (task["atomic"], task["tr2"], task["held2"],
                                       task["held2_para"], task["derange"])
    # attractor helper lists: (A-prompt, target-prompt) pairs
    attr_pairs = []
    if attr:
        Bs = list(task["r2"].keys()); rdg = random.Random(7000 + seed)
        for a in task["As"]:
            b = task["r1"][a]
            if attr == "wrong":
                b = rdg.choice([x for x in Bs if x != b])
            tgt = q_r2(b) if ATTR == "bpet" else q_ident(b)
            attr_pairs.append((q_r1(a), tgt))

    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
    for p in m.parameters():
        p.requires_grad_(True)
    P = nn.Linear(m.config.hidden_size, PROJ).to(device) if attr else None   # train-time only, deleted
    params = list(m.parameters()) + (list(P.parameters()) if P else [])
    opt = torch.optim.AdamW(params, lr=LR)
    curve = []; t0 = time.time(); m.train()
    for step in range(1, STEPS + 1):
        # main objective: atomic (+ sibling 2-hop only for the r34 upper-bound arm)
        pooltr = atomic + (tr2 if train2 else [])
        batch = [random.choice(pooltr) for _ in range(BS)]
        e = tok([x[0] for x in batch], return_tensors="pt", padding=True).to(device)
        gold = torch.tensor([x[1] for x in batch], device=device)
        lg = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
        loss = F.cross_entropy(lg, gold)
        if attr:
            ap = [random.choice(attr_pairs) for _ in range(BS // 2)]
            ea = tok([x[0] for x in ap], return_tensors="pt", padding=True).to(device)
            za = m.model(**ea, use_cache=False).last_hidden_state[:, -1].float()
            with torch.no_grad():
                et = tok([x[1] for x in ap], return_tensors="pt", padding=True).to(device)
                zt = m.model(**et, use_cache=False).last_hidden_state[:, -1].float()
            la = (1 - F.cosine_similarity(P(za), P(zt).detach(), dim=-1)).mean()
            loss = loss + LAMBDA * la
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % EVAL == 0 or step == STEPS:
            m.eval()
            a1, _ = acc(m, [x for x in atomic if "'s friend is" in x[0]])
            a2, _ = acc(m, [x for x in atomic if "'s pet is" in x[0]])
            h_em, h_gp = acc(m, [(p, g) for (p, g, a) in held2])
            hp_em, _ = acc(m, [(p, g) for (p, g, a) in held2p])
            d_em, _ = acc(m, der)
            m.train()
            curve.append(dict(step=step, atomic_r1=round(a1, 3), atomic_r2=round(a2, 3),
                              held2_em=round(h_em, 3), held2_goldprob=round(h_gp, 4),
                              held2_para_em=round(hp_em, 3), derange_em=round(d_em, 3)))
            print(f"    [{arm} s{seed} step {step}] atomic {a1:.2f}/{a2:.2f} "
                  f"held2 {h_em:.3f} (gp {h_gp:.3f}) para {hp_em:.3f} derange {d_em:.3f}", flush=True)
            if step >= max(2000, STEPS // 3) and min(a1, a2) < 0.90:
                print(f"    KILL: atomic recall {a1:.2f}/{a2:.2f} < 0.90 — fix data/prompt/steps.", flush=True)
                break
    # hidden-similarity diagnostic: h(A friend) vs correct/wrong h(B pet)
    m.eval()
    hsim = None
    ha = held2[:min(len(held2), 200)]
    if ha:
        zA = hstate(m, [q_r1(a) for (p, g, a) in ha])
        zBc = hstate(m, [q_r2(task["r1"][a]) for (p, g, a) in ha])
        Bs = list(task["r2"].keys()); rdg = random.Random(1234 + seed)
        zBw = hstate(m, [q_r2(rdg.choice([x for x in Bs if x != task["r1"][a]])) for (p, g, a) in ha])
        cc = F.cosine_similarity(zA, zBc, dim=-1).mean().item()
        cw = F.cosine_similarity(zA, zBw, dim=-1).mean().item()
        hsim = dict(cos_A_correctBpet=round(cc, 4), cos_A_wrongBpet=round(cw, 4), margin=round(cc - cw, 4))
    fin = curve[-1] if curve else None
    del m, P; torch.cuda.empty_cache()
    return dict(arm=arm, seed=seed, n_bridge=n_bridge, aper=aper, n_held2=task["n_held2"],
                n_train2=task["n_train2"], attr_target=(ATTR if attr else None),
                hidden_sim=hsim, final=fin, curve=curve, wall=round(time.time() - t0, 1))

def main():
    pool = harvest_pool()
    print(f"ASSOC_BRIDGE ({NAME}, {device}) bridge={BRIDGE} A/bridge={APER} steps={STEPS} seeds={SEEDS} "
          f"lambda={LAMBDA} attr={ATTR} arms={ARMS} | single-tok pool={len(pool)}", flush=True)
    results = []
    for seed in range(SEEDS):
        for arm in ARMS:
            r = run(seed, arm, pool)
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
            f = r["final"]
            print(f"  => {arm} s{seed}: held2_em={f['held2_em'] if f else 'NA'} "
                  f"para={f['held2_para_em'] if f else 'NA'} derange={f['derange_em'] if f else 'NA'} "
                  f"hsim_margin={r['hidden_sim']['margin'] if r['hidden_sim'] else 'NA'} (wall {r['wall']}s)", flush=True)
    # summary: held-out A->C by arm (mean over seeds, final)
    print("\n== ASSOC_BRIDGE SUMMARY (final held-out A->C EM, mean over seeds) ==")
    by = {}
    for r in results:
        by.setdefault(r["arm"], []).append(r)
    for arm in ARMS:
        rs = by.get(arm, [])
        h = [r["final"]["held2_em"] for r in rs if r["final"]]
        hp = [r["final"]["held2_para_em"] for r in rs if r["final"]]
        dd = [r["final"]["derange_em"] for r in rs if r["final"]]
        a1 = [r["final"]["atomic_r1"] for r in rs if r["final"]]
        a2 = [r["final"]["atomic_r2"] for r in rs if r["final"]]
        mn = lambda v: round(sum(v) / len(v), 3) if v else None
        print(f"  {arm:24s} atomic={mn(a1)}/{mn(a2)} held2={mn(h)} para={mn(hp)} derange={mn(dd)}")
    print("  PASS: shared_bridge_atomic OR shared_bridge_attractor beats unique_bridge_atomic AND deranged")
    print("        by >=+0.20 held2 (or >=0.30 abs), transfers to paraphrase, atomic>=0.95, no A->C trained.")
    print(f"[done wrote {OUT}]", flush=True)

if __name__ == "__main__":
    main()
