"""ROUND 34 (candidate A) — IN-DISTRIBUTION composition GROKKING probe: the last reachable
growth-necessity test after R32 closed OOD latent composition and R33 nailed the CL comparison.

R32 lesson: latent held-out (OOD) 2-hop composition is 0.0 even for the upper bound — unusable
as a grow-vs-fixed stressor. The literature (Grokked Transformers are Implicit Reasoners, 2024)
says composition DOES emerge, but only IN-DISTRIBUTION and only after long past-overfitting
training (grokking). So we reframe (codex+claude 2026-07-04):

  - Large single-token entity pool harvested from the vocab (not the ~44-name pool).
  - TWO relations r1='friend', r2='pet'. atomic: "A's friend is B", "B's pet is C".
    2-hop query: "A's friend's pet is" -> C (C never appears as an A completion => no leak).
  - Bridge set B is SMALL (many A per bridge B), so held-out A's (B,C) are exercised by OTHER
    A's trained 2-hop examples => the held-out is IN-DISTRIBUTION query holdout, not new entities.
  - Train atomic (all) + 2-hop (train split); trace held-out 2-hop over training steps (grokking
    curve). Depth sweep +0 vs +4 layers: does growth shift grokking ONSET or CEILING?

Hard kill criteria (codex): if train-2hop fit < 0.95 by 2k steps -> stop (fix data/prompt); if
after fit>=0.99 another 5k steps leaves held-out <= 0.05 (and <= derange control) -> stop; if both
seeds show no held-out rise -> stop, don't sweep depth. Growth claim needs onset >=30% earlier OR
final held-out ceiling >=0.10 higher at matched compute, with atomic recall not dropping.

  python -m s2.composition_grok
  env: GK_BRIDGE(60) GK_APER(6) GK_HOLD(0.3) GK_STEPS(40000) GK_EVAL(1000) GK_SEEDS(2)
       GK_GROWS(0,4) GK_LR(1e-4) GK_BS(64) GK_SMOKE(0)
"""
from __future__ import annotations
import os
import time
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from s0.qwen_grow import grow_qwen

NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B")
BRIDGE = int(os.environ.get("GK_BRIDGE", 60))         # # of bridge B entities
APER = int(os.environ.get("GK_APER", 6))              # # of A entities per bridge B
HOLD = float(os.environ.get("GK_HOLD", 0.3))          # fraction of A's 2-hop held out (in-distribution)
STEPS = int(os.environ.get("GK_STEPS", 40000))
EVAL = int(os.environ.get("GK_EVAL", 1000))
SEEDS = int(os.environ.get("GK_SEEDS", 2))
GROWS = [int(x) for x in os.environ.get("GK_GROWS", "0,4").split(",")]
LR = float(os.environ.get("GK_LR", 1e-4))
BS = int(os.environ.get("GK_BS", 64))
SMOKE = int(os.environ.get("GK_SMOKE", 0))
if SMOKE:
    BRIDGE, APER, STEPS, EVAL, SEEDS, GROWS = 12, 4, 400, 100, 1, [0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def one_tok(s):
        t = tok(" " + s, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None

    # --- harvest a large single-token, capitalized-alpha entity pool from the vocab ---
    pool = []
    seen = set()
    for tid in range(min(len(tok), 152000)):
        try:
            s = tok.convert_ids_to_tokens(tid)
        except Exception:
            continue
        if not isinstance(s, str):
            continue
        w = s.replace("Ġ", " ")                  # BPE space marker
        if w.startswith(" ") and w[1:].isalpha() and len(w) >= 4 and w[1].isupper():
            name = w[1:]
            if name.lower() in seen:
                continue
            if one_tok(name) is not None:             # round-trips to a single id
                pool.append(name); seen.add(name.lower())
    print(f"COMPOSITION-GROK ({NAME}, {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}) "
          f"bridge={BRIDGE} A/bridge={APER} hold={HOLD} steps={STEPS} seeds={SEEDS} grows={GROWS} "
          f"| single-tok pool={len(pool)}", flush=True)

    def q_r1(a):
        return f"{a}'s friend is"                      # atomic1: A -> B

    def q_r2(b):
        return f"{b}'s pet is"                         # atomic2: B -> C

    def q_2hop(a):
        return f"{a}'s friend's pet is"                # composition: A -> C

    def build_task(seed):
        rng = random.Random(5000 + seed)
        p = pool[:]; rng.shuffle(p)
        need = BRIDGE + BRIDGE + BRIDGE * APER        # C's, B's, A's
        assert len(p) >= need, f"pool too small: need {need} have {len(p)}"
        Cs = p[:BRIDGE]
        Bs = p[BRIDGE:2 * BRIDGE]
        As = p[2 * BRIDGE:2 * BRIDGE + BRIDGE * APER]
        r2 = {Bs[j]: Cs[j] for j in range(BRIDGE)}    # each bridge B -> its C
        r1 = {}; a_of_b = {j: [] for j in range(BRIDGE)}
        for i, a in enumerate(As):
            j = i // APER                             # APER consecutive A's share bridge B_j
            r1[a] = Bs[j]; a_of_b[j].append(a)
        # 2-hop train/holdout split BY A, guaranteeing each bridge keeps >=1 trained A
        train_A, held_A = [], []
        for j in range(BRIDGE):
            grp = a_of_b[j][:]; rng.shuffle(grp)
            n_hold = min(len(grp) - 1, int(round(HOLD * len(grp))))
            held_A += grp[:n_hold]; train_A += grp[n_hold:]
        atomic = [(q_r1(a), one_tok(r1[a])) for a in As] + [(q_r2(b), one_tok(r2[b])) for b in Bs]
        train2 = [(q_2hop(a), one_tok(r2[r1[a]])) for a in train_A]
        held2 = [(q_2hop(a), one_tok(r2[r1[a]])) for a in held_A]
        # derange control: held-out prompts with a wrong (shifted) target
        htar = [g for _, g in held2]
        sh = htar[1:] + htar[:1]
        derange = [(held2[i][0], sh[i]) for i in range(len(held2))]
        # leak assert: no held-out A has C as any of its atomic completions
        acomp = {}
        for a in As:
            acomp[q_r1(a)] = one_tok(r1[a])
        for (pmt, g) in held2:
            a = pmt.split("'s")[0]
            assert acomp.get(q_r1(a)) != g, "leak: A's atomic completion == 2hop answer"
        return dict(atomic=atomic, train2=train2, held2=held2, derange=derange,
                    n_train2=len(train2), n_held2=len(held2))

    @torch.no_grad()
    def acc(m, items):
        if not items:
            return 0.0
        ok = 0
        for i in range(0, len(items), 256):
            ch = items[i:i + 256]
            e = tok([p for p, _ in ch], return_tensors="pt", padding=True).to(device)
            lg = m.lm_head(m.model(**e).last_hidden_state[:, -1]).float()
            gold = torch.tensor([g for _, g in ch], device=device)
            ok += (lg.argmax(-1) == gold).sum().item()
        return ok / len(items)

    def load(grow):
        m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(device)
        if grow > 0:
            grow_qwen(m, grow)
        for p in m.parameters():
            p.requires_grad_(True)
        return m

    def run(seed, grow, task):
        m = load(grow)
        # atomic:2hop = 1:1 per batch (codex); train jointly, long, watch held-out over steps
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        atomic, train2, held2, der = task["atomic"], task["train2"], task["held2"], task["derange"]
        curve = []; onset = None; t0 = time.time()
        m.train()
        for step in range(1, STEPS + 1):
            batch = ([random.choice(atomic) for _ in range(BS // 2)] +
                     [random.choice(train2) for _ in range(BS // 2)])
            e = tok([p for p, _ in batch], return_tensors="pt", padding=True).to(device)
            gold = torch.tensor([g for _, g in batch], device=device)
            lg = m.lm_head(m.model(**e, use_cache=False).last_hidden_state[:, -1]).float()
            loss = F.cross_entropy(lg, gold)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % EVAL == 0 or step == STEPS:
                m.eval()
                a1 = acc(m, [x for x in atomic if "'s friend is" in x[0]])
                a2 = acc(m, [x for x in atomic if "'s pet is" in x[0]])
                tf = acc(m, train2); hd = acc(m, held2); dc = acc(m, der)
                m.train()
                curve.append(dict(step=step, atomic_r1=a1, atomic_r2=a2, train2_fit=tf,
                                  held2=hd, derange=dc))
                if onset is None and hd >= 0.3 and hd > 2 * max(dc, 1e-6):
                    onset = step
                print(f"    [grow+{grow} seed {seed} step {step}] atomic {a1:.2f}/{a2:.2f} "
                      f"train2 {tf:.2f} held2 {hd:.3f} derange {dc:.3f}", flush=True)
                # hard kill criteria
                if step >= 2000 and tf < 0.95:
                    print(f"    KILL: train2 fit {tf:.2f} < 0.95 at step {step} — fix data/prompt.", flush=True)
                    break
                if tf >= 0.99 and step >= 7000 and hd <= 0.05 and hd <= dc + 0.02:
                    print(f"    KILL: fit>=0.99 but held2 {hd:.3f} flat after {step} steps — no grokking.", flush=True)
                    break
        del m; torch.cuda.empty_cache()
        return dict(grow=grow, seed=seed, onset=onset, curve=curve,
                    final=curve[-1] if curve else None, wall=time.time() - t0)

    results = []
    out = os.environ.get("GK_OUT", "composition_grok_result.json")
    for seed in range(SEEDS):
        task = build_task(seed)
        print(f"  seed {seed}: n_train2={task['n_train2']} n_held2={task['n_held2']}", flush=True)
        for grow in GROWS:
            r = run(seed, grow, task)
            results.append(r)
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            fin = r["final"]
            print(f"  => grow+{grow} seed {seed}: onset={r['onset']} "
                  f"final held2={fin['held2'] if fin else 'NA'} (wall {r['wall']:.0f}s)", flush=True)

    # verdict: compare depth on onset + final held-out ceiling
    print("\n== GROKKING SUMMARY ==")
    by_grow = {}
    for r in results:
        by_grow.setdefault(r["grow"], []).append(r)
    for g, rs in sorted(by_grow.items()):
        onsets = [r["onset"] for r in rs if r["onset"] is not None]
        ceils = [r["final"]["held2"] for r in rs if r["final"]]
        mo = (sum(onsets) / len(onsets)) if onsets else None
        mc = (sum(ceils) / len(ceils)) if ceils else 0.0
        print(f"  grow+{g}: mean onset={mo} mean final held2={mc:.3f} "
              f"({sum(1 for r in rs if r['onset'])}/{len(rs)} seeds grokked)")
    print("  growth-necessity: claim ONLY if grow cuts onset >=30% OR lifts final held2 >=0.10 at "
          "matched compute, with atomic recall not dropping. Else growth still unproven.")
    print(f"[wrote {out}]", flush=True)


if __name__ == "__main__":
    main()
