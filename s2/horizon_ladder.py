"""R51 Phase A — triggered-growth horizon ladder, fixed-small frontier scan (codex qa 2026-07-10 22:14).

Question: does a FIXED 0.5B model hit a real CAPACITY/interference wall as the continual horizon grows, or only
a REHEARSAL-DILUTION wall (per-old-item replay exposure falls as history grows at fixed budget)? Only the former
would justify GROWTH. One continual real-text (census) lineage per arm; measure OLD-only held-out paraphrase EM
vs horizon. Arms share the per-seed stream manifest and differ ONLY in the retention mechanism:
  * fixed_1x     : budget replay (0.5*committed random), 1x consolidation compute — the baseline that may dilute.
  * fixed_4x     : same, 4x consolidation compute — does MORE compute restore retention?
  * full_cue     : replay from ALL committed (full coverage), 1x — does full coverage restore it?
  * const_expo   : each committed item gets a CONSTANT number of replay draws regardless of history (O(history)
    replay) — the decisive DILUTION control. If this restores retention, the wall is dilution, NOT capacity.
  * no_cue       : no old replay (compute-matched filler) — the forgetting floor.
Verdict: retention DROPS with horizon for fixed_1x AND const_expo/full/4x DON'T restore -> capacity/interference
wall (N* candidate -> Phase B growth). const_expo restores -> DILUTION wall (growth NOT justified). No drop -> no
saturation in range. cue-ledger bytes accounted (full Q+A here; compression ladder = follow-up). Qwen2.5-0.5B.
"""
import os, sys, json, math, random, collections
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3 import wikibridge as wb

NAME    = os.environ.get("HL_MODEL", "Qwen/Qwen2.5-0.5B")
STREAMS = int(os.environ.get("HL_STREAMS", 9))
ARTS    = int(os.environ.get("HL_ARTS", 5))
QA_PER  = int(os.environ.get("HL_QA", 5))
CONS_STEPS = int(os.environ.get("HL_CONS_STEPS", 400))    # 1x consolidation steps / stream
BUDGET_FRAC = float(os.environ.get("HL_BUDGET_FRAC", 0.5))
EXPOSURE = int(os.environ.get("HL_EXPOSURE", 6))          # const_expo: replay draws per committed item / stream
SEEDS   = int(os.environ.get("HL_SEEDS", 2))
ARMS    = os.environ.get("HL_ARMS", "fixed_1x,fixed_4x,full_cue,const_expo,no_cue").split(",")
LR      = float(os.environ.get("HL_LR", 1e-5))
SOURCE  = os.environ.get("HL_SOURCE", "census")
OUT     = os.environ.get("HL_OUT", "horizon_ladder_result.json")
device = wb.device

wb.NAME = NAME
wb.STREAMS, wb.ARTS, wb.QA_PER = STREAMS, ARTS, QA_PER
wb.CONS_STEPS, wb.LR = CONS_STEPS, LR
wb.SOURCE = SOURCE
tok = wb.tok
QT, em, gen, score = wb.QT, wb.em, wb.gen, wb.score

ARM_CFG = {
    "fixed_1x":  dict(mult=1, pool="budget",   const=False),
    "fixed_4x":  dict(mult=4, pool="budget",   const=False),
    "full_cue":  dict(mult=1, pool="full",     const=False),
    "const_expo": dict(mult=1, pool="full",    const=True),
    "no_cue":    dict(mult=1, pool="none",     const=False),
}


def _base_bits(base, qas):
    return [tb for (tb, nt) in wb.qa_answer_bits(base, qas, key="question")]


def cue_bytes(committed):
    return int(sum(len(q["question"]) + len(q["answers"][0]) for q, _ in committed))


def run_arm(base, streams, arm, seed):
    cfg = ARM_CFG[arm]
    M = wb.load_model()
    rng = random.Random(seed * 100003 + sum(bytes(arm, "utf8")))
    committed = []                                     # (qa, commit_t)
    surprise = 0.0
    curve = []
    for t in range(len(streams)):
        arts = streams[t]
        new_qa = [q for a in arts for q in a["qas"]]
        passages = [a["context"] for a in arts]
        pool = [q for q, _ in committed]
        steps = CONS_STEPS * cfg["mult"]
        # replay budget subset (budget arm) fixed once per stream
        if cfg["pool"] == "budget" and pool:
            bpool = rng.sample(pool, max(1, min(len(pool), round(BUDGET_FRAC * len(pool)))))
        else:
            bpool = pool
        M.train(); opt = torch.optim.AdamW(M.parameters(), lr=LR)
        prng = random.Random(f"{seed}:{arm}:{t}")
        for _ in range(steps):
            loss = wb.lm_step(M, [prng.choice(passages) for _ in range(4)])
            loss = loss + wb.qa_ce(M, [prng.choice(new_qa) for _ in range(8)])          # acquisition (common)
            if cfg["pool"] != "none" and bpool:
                loss = loss + wb.qa_ce(M, [prng.choice(bpool) for _ in range(8)])        # old replay
            elif cfg["pool"] == "none" and pool:
                loss = loss + wb.qa_ce(M, [prng.choice(new_qa) for _ in range(8)])       # compute-matched filler
            ne, nb = wb.base_anchor_logits(base, [prng.choice(wb.NEUTRAL) for _ in range(8)])
            sa = M.lm_head(M.model(**ne, use_cache=False).last_hidden_state[:, -1]).float()
            loss = loss + F.kl_div(F.log_softmax(sa, -1), F.softmax(nb, -1), reduction="batchmean")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        # constant exposure: extra replay-only passes so each committed item gets ~EXPOSURE draws this stream
        if cfg["const"] and pool:
            extra = math.ceil(len(pool) * EXPOSURE / 8)
            for _ in range(extra):
                loss = wb.qa_ce(M, [prng.choice(pool) for _ in range(8)])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step()
        M.eval(); del opt; torch.cuda.empty_cache()
        # commit newly-correct facts
        correct = [q for q, p in zip(new_qa, gen(M, [QT.format(q=q["question"]) for q in new_qa]))
                   if em(p, q["answers"]) == 1]
        bits = _base_bits(base, correct) if correct else []
        for q, b in zip(correct, bits):
            committed.append((q, t)); surprise += b
        # measure OLD-only (committed before this stream) held-out paraphrase retention, by age
        old = [(q, ct) for (q, ct) in committed if ct < t]
        if old:
            oq = [q for q, _ in old]
            o_em = round(score(M, oq, key="eval_question")[0], 3)
            ages = collections.defaultdict(list)
            for q, ct in old:
                ages[t - ct].append(q)
            age_em = {a: round(score(M, qs, key="eval_question")[0], 3) for a, qs in sorted(ages.items())}
        else:
            o_em, age_em = None, {}
        draws_per_item = round((8 * steps + (math.ceil(len(pool) * EXPOSURE / 8) * 8 if cfg["const"] else 0))
                               / max(len(pool), 1), 2) if pool else None
        curve.append(dict(t=t, n_committed=len(committed), n_old=len(old), surprise=round(surprise, 1),
                          old_para_em=o_em, age_em=age_em, cue_bytes=cue_bytes(committed),
                          replay_draws_per_item=draws_per_item, cons_steps=steps))
        print(f"      [{arm} s{seed} t{t}] committed={len(committed)} old={len(old)} old_para={o_em} "
              f"surprise={round(surprise,1)} draws/item={draws_per_item}", flush=True)
    del M; torch.cuda.empty_cache()
    return curve


def decide(seed_results):
    """capacity vs dilution: does fixed_1x old-retention DROP across horizon, and does const_expo (or full/4x)
    RESTORE it? Compare the LAST-horizon old_para_em across arms, averaged over seeds."""
    if len(seed_results) < 2:
        return dict(phase="shakedown_only", n=len(seed_results))
    def last(arm, seed):
        c = [x for x in seed_results[seed][arm] if x["old_para_em"] is not None]
        return c[-1]["old_para_em"] if c else None
    def first(arm, seed):
        c = [x for x in seed_results[seed][arm] if x["old_para_em"] is not None]
        return c[0]["old_para_em"] if c else None
    arms = list(ARM_CFG)
    lastv = {a: [last(a, s) for s in range(2)] for a in arms if a in seed_results[0]}
    firstv = {a: [first(a, s) for s in range(2)] for a in arms if a in seed_results[0]}
    v = dict(last_old_para=lastv, first_old_para=firstv)
    f1 = lastv.get("fixed_1x", [None, None])
    # fixed_1x declines across horizon in both seeds?
    decline = all(firstv["fixed_1x"][s] is not None and f1[s] is not None
                  and firstv["fixed_1x"][s] - f1[s] >= 0.10 for s in range(2)) if "fixed_1x" in lastv else False
    def restores(arm):
        return arm in lastv and all(lastv[arm][s] is not None and f1[s] is not None
                                    and lastv[arm][s] - f1[s] >= 0.10 for s in range(2))
    v["fixed_1x_declines"] = decline
    v["const_expo_restores"] = restores("const_expo")
    v["full_cue_restores"] = restores("full_cue")
    v["fixed_4x_restores"] = restores("fixed_4x")
    if not decline:
        v["verdict"] = "no_saturation_in_range"        # fixed small hasn't saturated -> growth unjustified
    elif v["const_expo_restores"] or v["full_cue_restores"] or v["fixed_4x_restores"]:
        v["verdict"] = "rehearsal_dilution_wall"        # more exposure/compute fixes it -> NOT capacity, no growth
    else:
        v["verdict"] = "capacity_wall_candidate"        # nothing restores -> N* candidate -> Phase B growth
    return v


def main():
    print(f"HORIZON_LADDER ({NAME}, {device}) source={SOURCE} streams={STREAMS}x{ARTS}x{QA_PER} cons={CONS_STEPS} "
          f"exposure={EXPOSURE} seeds={SEEDS} arms={ARMS}", flush=True)
    base = wb.load_model()
    results = []
    for seed in range(SEEDS):
        streams = wb.build_census(seed, base) if SOURCE == "census" else wb.build_cf(seed, base)
        for t, s in enumerate(streams):
            for ai, a in enumerate(s):
                for j, q in enumerate(a["qas"]):
                    q["qid"] = f"{seed}:{t}:{ai}:{j}"; q["stream_t"] = t
        print(f"  seed {seed}: streams={[len(s) for s in streams]}", flush=True)
        if len(streams) < 3:
            print("  <3 streams — abort seed", flush=True); continue
        seed_res = {}
        for arm in ARMS:
            seed_res[arm] = run_arm(base, streams, arm, seed)
        results.append(seed_res)
        v = decide(results)
        json.dump(dict(config=dict(model=NAME, source=SOURCE, streams=STREAMS, arts=ARTS, qa=QA_PER,
                                   cons=CONS_STEPS, exposure=EXPOSURE, budget_frac=BUDGET_FRAC, seeds=seed + 1,
                                   arms=ARMS), seeds=results, decision=v), open(OUT, "w"), indent=1)
        print(f"  DECISION {json.dumps(v)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
