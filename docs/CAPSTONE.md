# Capstone — Replay-Consolidation: Continual Knowledge-into-Weights without Inference Memory

*Synthesis of the Grow-and-Consolidate research arc (R19–R36, 2026-07). Model: Qwen2.5-0.5B, all
results ≥2 seeds unless noted. Harness: `s2/lifecycle_bakeoff.py`. Logs: `docs/cloud_results/`.*

## Headline

> **Sequential no-gold scaffold distillation + compact committed-target replay** writes new facts into a
> **single dense checkpoint**, retains a lifelong stream **without catastrophic forgetting**, uses **no
> external memory at inference**, and avoids **joint full retraining**. It **beats standard no-replay
> continual-learning baselines** in the same harness, and (R36-A2) removes the expensive resident snapshot
> teacher and all replay-time teacher forwards — a concrete compute/VRAM win at zero retention cost.

This is a *cheap, memory-free, in-weights* continual-learning result. It is **not** a claim that continual
learning is solved, that rehearsal is unnecessary, or that model growth is required.

## The contract (invariant held throughout)

1. New knowledge must end up in the **dense weights** (not an external store).
2. **No external memory at inference** — one un-routed dense checkpoint answers everything; no task ID, no
   key bank, no history-conditioned routing.
3. **No joint full retraining** over all historical data — training is sequential/streamed.
4. Old knowledge must be **retained** (no catastrophic forgetting).

## 1. The positive — replay-consolidation

A transient per-stream **scaffold** (capsule memory) is trained as a teacher, its knowledge **distilled**
into a growing dense model, prior streams are preserved by **self-distillation replay** (no gold), and the
scaffold is **discarded** — inference is the dense checkpoint alone.

| milestone | result |
|---|---|
| **R19–R25** faithful no-gold loop | scaffold→distill→self-distill replay retains (oldest-S0 forget +0.01) vs naive forgets (+0.60); single dense, no inference memory |
| **R26–R30** reliability + 3-seed | answer-recall restart guard → all streams consolidate; replay all-seen **0.918**, oldest forget +0.017; not seed-specific |
| **R33** vs standard CL | replay-consolidation **beats** sequential-no-replay, continued-FT-with-gold, LoRA-merge, and matches external memory **without inference memory** (all-seen **0.914**) |
| **R35** brackets | **EWC ≪ replay ≤ gold-old oracle** — replay is near the gold-old upper bound with no gold |
| **R36-EV** external validity | holds on **KG-shaped real-entity counterfactuals** (~80 real subjects, 5 relations, 2 surface forms, frozen-base screen): all-seen **0.919**, oldest forget **+0.000**, oracle-gap seen +0.035 — **not a template artifact** |
| **R36-A2** cheaper | a **1 committed answer token / (fact,view)**, stored once at commit, replayed via CE, **== full snapshot self-distill** (0.875/0.765/+0.000) while removing the resident snapshot and **5000→0** replay teacher forwards (**peak VRAM 9407→6711 MB, −29%**) |

## 2. Bounded negatives — why the *cheaper* alternatives don't suffice

Each was a genuine attempt at the frontier, closed with the *right* experiment (not a strawman):

- **Capacity is not the bottleneck (R31).** A single fixed layer + replay holds 800 facts across 8 streams
  at 1.0. Growth adds no capacity at this scale.
- **Composition is out of reach for everyone (R32/R34).** Latent 2-hop `A→C` is 0.000 even for the
  full-FT direct-gold upper bound (OOD two-hop curse); in-distribution grokking already ~0.98 with no
  depth/growth advantage at matched compute.
- **Rehearsal-free first-order protection is bounded ≪ replay (R36-I/C).** The entire *protect-by-direction*
  family — input null-space (`nswrite`, ~0.40@24 streams), margin-bilinear (worse), drift-budget
  (short-horizon only), and **global answer-gradient OGD with exact realized-ΔΘ semantics (R36-C, ≈ naive)**
  — tops out far below replay. The harmful forgetting is **nonlinear/higher-order** parameter motion that no
  first-order gradient-orthogonality constraint intercepts at affordable rank; OGD is additionally
  rank-inefficient and memory-bound.
- **Sublinear-fact rehearsal is item rehearsal, not stream protection (R36-A).** Replaying a random K-subset
  retains the rehearsed items (~0.9) but leaves non-rehearsed stream-mates at ~naive (0.375@K8). For
  **independent** counterfactual facts this is information-theoretically forced: fact A carries zero
  information about unrelated fact B, so retention footprint is **irreducibly O(#facts)**.

## 3. Honest limits (what we did NOT prove)

- **Not rehearsal-free** — replay (in some compact form) is load-bearing.
- **Storage is O(#facts)** for independent facts — though R36-A2 makes per-fact cost a single token.
- **Growth is not justified as a retention driver** — every probed axis (R23 retention, R31 capacity,
  R32/R34 composition) is negative *with replay present*; growth's *unique* value (interference-free
  capacity by construction) was **not** validly tested (see §4).
- **Scale/scope**: 0.5B params, synthetic + KG-shaped single-token counterfactuals. Not open-domain, not
  compositional, not multi-token generation.

## 4. Future work — strict no-router growth isolation

The one untested justification for growth is **retention by isolation** (each stream writes fresh
parameters that cannot overwrite old ones) rather than by projection. It is deferred, not dismissed,
because hard isolation tends to require **routing at inference** (task ID / learned router / key bank),
which violates the no-inference-memory invariant; and merging isolated branches into one dense path
reduces to LoRA-merge/adapter-merge, which already fails without replay.

A valid future test must pass on **no-router arms only** (no task ID, no per-stream key lookup, no external
bank at inference; one un-routed checkpoint), with gates: `all_seen` materially above `nswrite` and far
above naive; `all_para` high (no prompt memorization); oldest forget ≤0.10; newest within 0.03 of the
replay reference; base-hop drop ≤0.03; report parameter growth and inference FLOPs against fixed-capacity
and large-from-start controls. Routed variants (`iso_oracle_route`, `iso_key_route`) are upper bounds
only, not project-valid claims.

Other appendix-level probes: curvature-aware protection (K-FAC Fisher — likely a bounded negative, a
stronger surrogate than the failed diagonal EWC but still a local approximation, not old-loss
optimization); structured/redundant-data regimes where coreset replay *can* generalize (a different
theorem boundary, not arbitrary-fact lifelong retention).

## 5. Reproducibility

- Harness `s2/lifecycle_bakeoff.py`; arms via `BK_ARMS` (consolidate: `ours`/`naive`/`continued`/
  `loramerge`/`oracle`; rehearsal-free: `nswrite`/`margin`/`ogd`; footprint: `ours_k<K>`,
  `ours_tgt_<mode>`). Standard config `LD_ROUNDS=6 LD_PER=40 LD_STEPS=1000 LD_SEEDS=2`, `PYTHONHASHSEED=0`,
  GROW=2. `BK_DATA=kg` for the external-validity benchmark.
- Artifacts: `docs/cloud_results/lifecycle_bakeoff_r33*/r35*`, `kg_bakeoff_r36ev.log`,
  `nswrite_r36i*`, `ogd_ce_pilot_r36c.log`, `replayk_sweep_r36a.log`, `r36a2_compact_targets.{log,json}`.
- Full round-by-round detail: `FINDINGS.md` (top matter).

## Bottom line

Replay-consolidation is a robust, externally-valid, now-cheap method for writing independent factual
knowledge into a single dense model with no inference memory and no joint retraining, and it beats standard
CL baselines. The cheaper dreams around it — rehearsal-free protection and sublinear-fact rehearsal — are
mapped and bounded. Growth remains unproven and is scoped as strict-contract future work. Continual
learning is **not solved**; this is an honest, reproducible advance on the *in-weights, memory-free* corner
of it, with the open frontiers (a genuinely-new rehearsal-free mechanism; no-router growth isolation)
named rather than overclaimed.
