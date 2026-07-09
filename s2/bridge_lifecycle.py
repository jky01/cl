"""R47-bridge_lifecycle — does the R46 bridge-unification primitive work in a CONTINUAL setting: enable
latent 2-hop composition AND preserve OLD knowledge under later writes, WITHOUT replaying old data?
(codex-designed 2026-07-09.22.35 — the project-valid continuation of R46's static positive.)

R46 (static, all atomics trained in one run) showed: aligning h("A's friend is")->h("B") (identity
bridge-unification) induces held-out A->C composition ~0.42 where independent bindings + all controls give
~0. But R46 is NOT a continual-learning result. R47 makes it lifecycle:

  Phase 1: train OLD edges B->C (all bridges). This is the prior knowledge to protect.
  Phase 2: train NEW edges A->B (no direct A->C labels) + arm-specific extra. Later writes (A->B) can
           interfere with / overwrite old B->C.
  Eval (closed-book, one dense checkpoint, no memory): held-out A->C composition, OLD B->C retention,
        atomic A->B, + R46 mechanism margins.

Arms: no_attractor (baseline: A->B only, old B->C forgets) | attractor (+identity bridge-unification) |
deranged_attractor (align WRONG bridge) | replay_old (+ replay old B->C CE — the R33/R43 compact-replay
baseline) | distill_old (+ KL to phase-1 logits on B->C) | direct_2hop (+ A->C CE, upper bound).

Pass (codex): attractor IMPROVES A->C while PRESERVING old B->C materially better than no_attractor, and
ideally comparable to replay/distill WITHOUT replaying old data; atomic A->B stays high; one dense
checkpoint, no inference memory, no full joint retraining. That is where R46 touches continual RETENTION,
not just static composition.

  python -m s2.bridge_lifecycle
  env: BL_BRIDGE(60) BL_APER(6) BL_P1(3000) BL_P2(7000) BL_EVAL(500) BL_SEEDS(2) BL_LR(1e-4) BL_BS(64)
       BL_LAMBDA(1.0) BL_ARMS(...) BL_SMOKE(0)
"""
from __future__ import annotations
import os, time, json, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from s2.assoc_bridge import (harvest_pool, q_r1, q_r2, q_2hop, q_ident, hstate, acc, held_probs,
                             NAME, device, tok)

BRIDGE = int(os.environ.get("BL_BRIDGE", 60))
APER = int(os.environ.get("BL_APER", 6))
P1 = int(os.environ.get("BL_P1", 3000))               # phase-1 (old B->C) steps
P2 = int(os.environ.get("BL_P2", 7000))               # phase-2 (new A->B + extra) steps
EVAL = int(os.environ.get("BL_EVAL", 500))
SEEDS = int(os.environ.get("BL_SEEDS", 2))
LR = float(os.environ.get("BL_LR", 1e-4))
BS = int(os.environ.get("BL_BS", 64))
LAMBDA = float(os.environ.get("BL_LAMBDA", 1.0))
OUT = os.environ.get("BL_OUT", "bridge_lifecycle_result.json")
ALL_ARMS = ["no_attractor", "attractor", "deranged_attractor", "replay_old", "distill_old", "direct_2hop"]
ARMS = os.environ.get("BL_ARMS", ",".join(ALL_ARMS)).split(",")
SMOKE = int(os.environ.get("BL_SMOKE", 0))
if SMOKE:
    BRIDGE, APER, P1, P2, EVAL, SEEDS = 12, 4, 400, 600, 200, 1

def one_tok(s):
    t = tok(" " + s, add_special_tokens=False).input_ids
    return t[0] if len(t) == 1 else None

def build(seed, pool):
    rng = random.Random(6000 + seed)
    p = pool[:]; rng.shuffle(p)
    need = BRIDGE * 2 + BRIDGE * APER
    assert len(p) >= need, f"pool too small: need {need}"
    Cs = p[:BRIDGE]; Bs = p[BRIDGE:2 * BRIDGE]; As = p[2 * BRIDGE:2 * BRIDGE + BRIDGE * APER]
    r2 = {Bs[j]: Cs[j] for j in range(BRIDGE)}; r1 = {}
    for i, a in enumerate(As):
        r1[a] = Bs[i // APER]
    bc = [(q_r2(b), one_tok(r2[b])) for b in Bs]           # OLD edges B->C  (phase 1)
    ab = [(q_r1(a), one_tok(r1[a])) for a in As]           # NEW edges A->B  (phase 2)
    held2 = [(q_2hop(a), one_tok(r2[r1[a]]), a) for a in As]   # held-out A->C (never trained, main arms)
    ac_direct = [(q_2hop(a), one_tok(r2[r1[a]])) for a in As]  # direct A->C target (upper-bound arm only)
    allC = [one_tok(c) for c in Cs]; rdg = random.Random(9000 + seed)
    der = [(pm, rdg.choice([c for c in allC if c != g])) for (pm, g, a) in held2]
    for (pm, g, a) in held2:                                 # leak assert
        assert one_tok(r1[a]) != g, "leak"
    return dict(bc=bc, ab=ab, held2=held2, ac_direct=ac_direct, der=der, r1=r1, r2=r2, As=As, Bs=Bs)

def step_ce(m, opt, items):
    e = tok([x[0] for x in items], return_tensors="pt", padding=True).to(device)
    gold = torch.tensor([x[1] for x in items], device=device)
    lg = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
    return F.cross_entropy(lg, gold)

@torch.no_grad()
def bc_logits(m, bc):
    out = []
    for i in range(0, len(bc), 256):
        e = tok([x[0] for x in bc[i:i + 256]], return_tensors="pt", padding=True).to(device)
        out.append(m.lm_head(m.model(**e).last_hidden_state[:, -1]).float().cpu())
    return torch.cat(out, 0)

def run(seed, arm, pool):
    t0 = time.time()
    task = build(seed, pool)
    bc, ab, held2, der = task["bc"], task["ab"], task["held2"], task["der"]
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    # ---- phase 1: OLD B->C ----
    m.train()
    for _ in range(P1):
        loss = step_ce(m, opt, [random.choice(bc) for _ in range(BS)])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); bc0, _ = acc(m, bc); ab0, _ = acc(m, ab)
    teach = bc_logits(m, bc) if arm == "distill_old" else None      # frozen phase-1 teacher for distill
    # codex: FREEZE the phase-1 h(B) identity bank as the attractor target — else it's a moving phase-2
    # target. Frozen bank = "old knowledge already in weights becomes a train-time consolidation target",
    # created at the phase boundary, deleted after training (NOT inference memory, NOT old B->C replay).
    Bs = task["Bs"]; bidx = {b: i for i, b in enumerate(Bs)}
    hB_frozen = hstate(m, [q_ident(b) for b in Bs]).detach() if arm in ("attractor", "deranged_attractor") else None
    print(f"    [{arm} s{seed}] after P1: B->C={bc0:.3f} A->B={ab0:.3f}", flush=True)
    # ---- phase 2: NEW A->B (+ arm extra) ----
    rdg = random.Random(7000 + seed)
    m.train(); curve = []
    for step in range(1, P2 + 1):
        loss = step_ce(m, opt, [random.choice(ab) for _ in range(BS)])   # new edges A->B
        if arm in ("attractor", "deranged_attractor"):
            aa = [random.choice(task["As"]) for _ in range(BS // 2)]
            zA = m.model(**tok([q_r1(a) for a in aa], return_tensors="pt", padding=True).to(device),
                         use_cache=False).last_hidden_state[:, -1].float()
            tb = [task["r1"][a] if arm == "attractor" else rdg.choice([x for x in Bs if x != task["r1"][a]])
                  for a in aa]
            zt = hB_frozen[torch.tensor([bidx[b] for b in tb], device=device)]   # FROZEN phase-1 target
            loss = loss + LAMBDA * (1 - F.cosine_similarity(zA, zt, -1)).mean()
        elif arm == "replay_old":
            loss = loss + step_ce(m, opt, [random.choice(bc) for _ in range(BS)])   # replay old B->C
        elif arm == "distill_old":
            idx = [random.randrange(len(bc)) for _ in range(BS)]
            e = tok([bc[i][0] for i in idx], return_tensors="pt", padding=True).to(device)
            lg = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
            tgt = teach[idx].to(device)
            loss = loss + F.kl_div(F.log_softmax(lg, -1), F.softmax(tgt, -1), reduction="batchmean")
        elif arm == "direct_2hop":
            loss = loss + step_ce(m, opt, [random.choice(task["ac_direct"]) for _ in range(BS // 2)])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % EVAL == 0 or step == P2:
            m.eval()
            ac_em, ac_gp, ac_wp, ac_mar = held_probs(m, [(p, g) for (p, g, a) in held2], der)
            bcr, _ = acc(m, bc); abr, _ = acc(m, ab)
            m.train()
            curve.append(dict(step=step, ac_em=round(ac_em, 3), ac_goldprob=round(ac_gp, 4),
                              ac_gold_minus_wrong=round(ac_mar, 4), oldBC_retention=round(bcr, 3),
                              AB=round(abr, 3)))
            print(f"    [{arm} s{seed} p2 {step}] A->C {ac_em:.3f} (g-w {ac_mar:+.4f}) "
                  f"oldB->C {bcr:.3f} A->B {abr:.3f}", flush=True)
    # mechanism margins (final)
    m.eval(); ha = held2[:200]
    Bcorr = [task["r1"][a] for (p, g, a) in ha]
    Bwrong = [rdg.choice([x for x in Bs if x != b]) for b in Bcorr]
    def margin(qfn, tfn):
        zQ = hstate(m, [qfn(a) for (p, g, a) in ha]); zc = hstate(m, [tfn(b) for b in Bcorr]); zw = hstate(m, [tfn(b) for b in Bwrong])
        return round((F.cosine_similarity(zQ, zc, -1).mean() - F.cosine_similarity(zQ, zw, -1).mean()).item(), 4)
    hsim = dict(aux=margin(lambda a: q_r1(a), q_ident), raw_bridge=margin(lambda a: f"{a}'s friend's", q_ident),
                readout=margin(lambda a: q_2hop(a), q_r2)) if ha else None
    # target-drift diagnostic (codex): how far did h(B) move from the frozen phase-1 bank during phase 2?
    tdrift = None
    if hB_frozen is not None:
        hB_now = hstate(m, [q_ident(b) for b in Bs])
        tdrift = round(F.cosine_similarity(hB_now, hB_frozen, -1).mean().item(), 4)
    fin = curve[-1] if curve else None
    del m; torch.cuda.empty_cache()
    return dict(arm=arm, seed=seed, bc_after_p1=bc0, ab_after_p1=ab0, hidden_sim=hsim,
                target_drift_cos=tdrift, final=fin, curve=curve, wall=round(time.time() - t0, 1))

def main():
    pool = harvest_pool()
    print(f"BRIDGE_LIFECYCLE ({NAME}, {device}) bridge={BRIDGE} A/bridge={APER} P1={P1} P2={P2} "
          f"seeds={SEEDS} lambda={LAMBDA} arms={ARMS} | pool={len(pool)}", flush=True)
    results = []
    for seed in range(SEEDS):
        for arm in ARMS:
            r = run(seed, arm, pool)
            results.append(r); json.dump(results, open(OUT, "w"), indent=1)
            f = r["final"]
            print(f"  => {arm} s{seed}: A->C={f['ac_em'] if f else 'NA'} oldB->C={f['oldBC_retention'] if f else 'NA'} "
                  f"A->B={f['AB'] if f else 'NA'} raw_bridge={r['hidden_sim']['raw_bridge'] if r['hidden_sim'] else 'NA'} "
                  f"(wall {r['wall']}s)", flush=True)
    print("\n== BRIDGE_LIFECYCLE SUMMARY (final, mean over seeds) ==")
    by = {}
    for r in results:
        by.setdefault(r["arm"], []).append(r)
    mn = lambda v: round(sum(v) / len(v), 3) if v else None
    for arm in ARMS:
        rs = by.get(arm, [])
        print(f"  {arm:20s} A->C={mn([r['final']['ac_em'] for r in rs if r['final']])} "
              f"oldB->C={mn([r['final']['oldBC_retention'] for r in rs if r['final']])} "
              f"A->B={mn([r['final']['AB'] for r in rs if r['final']])} "
              f"raw_bridge={mn([r['hidden_sim']['raw_bridge'] for r in rs if r['hidden_sim']])}")
    print("  PASS: attractor improves A->C AND preserves oldB->C > no_attractor, ~replay/distill WITHOUT")
    print("        replaying old data; A->B high; one dense checkpoint, no inference memory.")
    print(f"[done wrote {OUT}]", flush=True)

if __name__ == "__main__":
    main()
