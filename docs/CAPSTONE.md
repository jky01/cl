# Capstone — Replay-Consolidation: Continual Knowledge-into-Weights without Inference Memory

*Synthesis of the Grow-and-Consolidate research arc (R19–R38B-A, 2026-07). Model: Qwen2.5-0.5B, all
results ≥2 seeds unless noted. Harnesses: `s2/lifecycle_bakeoff.py` (synthetic/KG), `s3/wikibridge.py`
(real passage text). Logs: `docs/cloud_results/`.*

## Headline

> **Sequential no-gold scaffold distillation + compact committed-target replay** writes new facts into a
> **single dense checkpoint**, retains a **lifelong-style 6-stream factual stream at 0.5B scale**
> **without catastrophic forgetting**, uses **no external memory at inference**, and avoids **joint full
> retraining**. It **beats standard in-weights no-replay continual-learning baselines** in the same
> sequential harness (including a continued-FT baseline that has gold new-stream labels but no old replay),
> and (R36-A2) removes the expensive resident snapshot
> teacher and all replay-time teacher forwards — a concrete compute/VRAM win at zero retention cost.

This is a *cheap, memory-free, in-weights* continual-learning result. It is **not** a claim that continual
learning is solved, that rehearsal is unnecessary, or that model growth is required.

> **Real-text extension (R38-A → R38B-A).** The same lifecycle transfers from synthetic/KG tuples to **real
> Wikipedia passage text**: `s3/wikibridge.py` reads SQuAD passages into a **closed-book** dense checkpoint
> that answers **held-out paraphrased** questions at **0.893** para-EM (92% of gold-passage RAG) with no
> inference memory — provided ingestion builds **QA/answer-function targets**, not raw reading (raw
> continued-PT → ~0). Two hardened results bound the "read many books cheaply" hope: the strong **"small
> random replay buys non-replayed neighbor coverage" claim is RETRACTED** (R38B — the pilot signal was a
> fresh-stream accounting artifact; replay protects the replayed item, not its neighbors), while a **narrow
> positive survives** — **real/prior-anchored knowledge has a no-replay retention floor that independent
> invented facts lack** (R38B-A same-objective control: real-text zero-replay old-para 0.63–0.74 vs
> independent-synthetic 0.16–0.18, a decisive collapse). Reading real knowledge that overlaps the pretrained
> manifold is genuinely cheaper to *retain* than memorizing independent tuples — but not via replay coverage.

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
| **R33** vs standard CL (3-seed) | replay-consolidation all-seen **0.914** **beats** in-weights no-replay baselines: sequential-naive **0.408**, continued-FT-**with-gold**-new **0.460**, LoRA-merge **0.367**; matches external memory (0.875) **without inference memory**. Even gold-new continued-FT forgets; compact committed replay retains without old gold. |
| **R35** brackets | **EWC ≪ replay < gold-old oracle**: replay (0.890) far above regularization/no-replay (EWC 0.456) but still **below** the gold-old replay ceiling (oracle 0.994; gap **+0.104 seen / +0.156 para** — the honest no-gold self-distill headroom) |
| **R36-EV** external validity | holds on **KG-shaped real-entity counterfactuals** (~80 real subjects, 5 relations, 2 surface forms, frozen-base screen): all-seen **0.919**, oldest forget **+0.000**, oracle-gap seen +0.035 — **not a template artifact** |
| **R36-A2** cheaper | a **1 committed answer token / (fact,view)**, stored once at commit, replayed via CE, **== full snapshot self-distill** (0.875/0.765/+0.000) while removing the resident snapshot and **5000→0** replay teacher forwards (**peak VRAM 9407→6711 MB, −29%**) |

## 1b. Real-text arc (R37-A, R38) — from tuples to passages, and the footprint reckoning

| milestone | result |
|---|---|
| **R38-A** real-text bridge | `s3/wikibridge.py`: transient continued-PT scaffold **+ QA span-CE targets** on SQuAD passages → consolidate into one dense checkpoint with compact committed replay → **closed-book** held-out-**paraphrase** EM **0.893** (RAG upper bound 0.973), base/naive/read-only **0.000**. Reading is not enough; the load-bearing step is **building answer-function targets** from source text. |
| **R38B** footprint retraction | The "small random replay K buys **non-replayed** neighbor retention via real-text redundancy" pilot (0.733@K3) was a **fresh-stream accounting artifact**. Old-only + 2-seed hardening: K3−K0 = **+0.062** (≪ +0.15 bar), seed-sign unstable, **zero-replay no-anchor arm (0.738) beats every replay arm**. Replay protects the *replayed* item (k1 replayed 0.875 vs non 0.641) — R36-A item-rehearsal, **no neighbor spillover**. Strong sublinear-coverage claim **dead**. |
| **R38B-A** same-objective control | Swap ONLY the corpus (`WB_SOURCE=synth`, identical `run_ingest`) to **independent invented facts**: zero-replay old-para **collapses to 0.16–0.18** (all 3 seeds ≤0.27) vs real text 0.63–0.74. `shared_templates` (max format sharing) already collapsed ⇒ not format-sharing. **The real-text no-replay floor is genuine prior-anchoring / redundancy, not a gentle objective.** Narrow POSITIVE preserved. |
| **R37-A** localized-write growth | Penalize new grown-block forward footprint `‖Δh‖²/‖h‖²` on a non-new reference → block ≈ identity on old prompts. Clean 2-seed: decoy 0.608 (≈ nswrite), nogrow-decoy 0.325 (< naive 0.387) ⇒ **growth is load-bearing under isolation** — the first growth-necessary signal — but only in the strict no-router regime and still ≪ replay. Not a solved growth story. |

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
  **independent** counterfactual facts, random sublinear coverage does not protect non-replayed items:
  fact A carries zero information about unrelated fact B, so the retained information must scale with the
  number of independent facts preserved — an **O(#facts) information-footprint** result for arbitrary
  independent facts. (This is not a universal lower bound on all structured-data coresets or
  self-generated coverage schemes, nor an exact per-round replay-frequency bound.) **On real text this was
  re-tested and holds (R38B):** the apparent redundancy-spillover was a fresh-stream artifact; random
  sublinear replay still protects only replayed items. What real text *does* add (R38B-A) is a
  **prior-anchoring no-replay floor** — orthogonal to replay coverage, not a softening of it.

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
only, not project-valid claims. **A routed arm can diagnose whether isolation itself has value, but it is
not a project-valid solution unless the isolated knowledge can be merged into one un-routed dense path
without replay-like historical supervision** — otherwise "growth works" quietly means "routing works."

Other appendix-level probes: curvature-aware protection (K-FAC Fisher — likely a bounded negative, a
stronger surrogate than the failed diagonal EWC but still a local approximation, not old-loss
optimization); structured/redundant-data regimes where coreset replay *can* generalize (a different
theorem boundary, not arbitrary-fact lifelong retention).

**Next frontier (R39 — rehearsal-free answer-function protection).** R37-A (growth isolation, done above)
and R38B-A (real-text prior-anchoring floor) reframe the open problem: replay is still load-bearing for
*independent* new knowledge, and the only thing that retains without replay is knowledge already anchored in
the prior. The genuinely-unsolved target is therefore **writing NEW independent knowledge in an
interference-resistant way with no per-item rehearsal**. R38B-A shows the hardest honest regime to test it
in is **independent synthetic / KG-shaped facts** (real-text prior anchoring would mask failure). The design
must hold a strict line codex drew: *protecting old realized update directions ≠ protecting the old answer
function* (R36-C already failed that way), and a method that stores per-fact prompts/logits/activations/
Jacobians is **compressed rehearsal, not rehearsal-free** — the label requires that **no old item-specific
information is used during later writes**. Candidate mechanisms to pit against best `nswrite` + compact-replay
oracle: (i) generic-probe **functional anchoring** (KL to M_{t-1} on a fixed broad non-old probe set —
pure training-state constraint); (ii) **generative self-replay** (M_{t-1} reconstructs its own QA to
rehearse — no stored old data); (iii) realized-ΔΘ null-space writing (training-state constraint, but
direction≠function caveat applies). Strict gates: beat best `nswrite` by a material margin (not just naive),
newest within a small tolerance of the no-protection writer, base/neutral preserved, and honest train-time
storage/compute accounting with an explicit rehearsal-free-vs-compressed-rehearsal line.

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
in-weights no-replay CL baselines — now demonstrated from synthetic tuples through KG counterfactuals to
**real Wikipedia passages → closed-book paraphrase QA** (R38-A). The cheaper dreams around it — rehearsal-free
protection and sublinear-fact rehearsal — are mapped and bounded, and the real-text "read many books cheaply"
hope is split honestly: **no free redundancy-spillover from random replay (R38B), but a genuine no-replay
retention floor for prior-anchored knowledge that independent facts lack (R38B-A).** Continual learning is
**not solved**; this is an honest, reproducible advance on the *in-weights, memory-free* corner of it.
**Replay-consolidation is the closed positive; first-order rehearsal-free protection and random sublinear
replay are bounded negatives; localized-write growth is load-bearing only under strict no-router isolation
(R37-A) and is not yet a solved growth story; and the live open frontier is rehearsal-FREE answer-function
protection for genuinely new independent knowledge (R39) — where no old item-specific information may touch
later writes.**
