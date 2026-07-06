# s0 — Findings: continual learning without forgetting, via a scalable key-retrieval

> **PIVOT (2026-07-03/04, rounds R19-R25) — Grow-and-Consolidate (`s2/`).** Per the
> strategy doc `reference/2026.07.03.20.md`, the direction shifted from *scaling an
> external memory* to *consolidating a transient memory scaffold into a single dense
> checkpoint that answers everything with no memory at inference*. Arc (all real Qwen-0.5B,
> 2 seeds, logs in `docs/cloud_results/consolidate_*` / `lifecycle*`):
> - **R19** `consolidate_memory_to_weights.py`: grow (+4 identity layers) + consolidate — a
>   dense student answers new facts w/o memory (seen 0.99 / para 0.97 / reverse 0.88),
>   *generalizing beyond the teacher*. Bridge works.
> - **R20**: naive preservation fails (old hop-acc 0.205→0.009); **old-task REPLAY in the
>   preserve-KL fixes it** (hop 0.205→0.196) with no fact-recall cost. Anchor-agreement alone
>   is insufficient — must replay the actual old-task distribution.
> - **R21** `lifecycle.py`: 3-round grow+consolidate — **replay = zero forgetting** (oldest
>   stream 1.0→1.0) vs **naive forgets** (1.0→0.40), single dense checkpoint.
> - **R22**: faithful **distill-only (no gold)** — student reaches seen 1.0 supervised *only*
>   by the memory teacher (gap 0), proving the scaffold carries the knowledge; generalization
>   is bounded by teacher coverage (para 0.36).
> - **R23** (honest ablation): **REPLAY, not GROWTH, drives retention** — a *non-growing*
>   fixed model + replay retains identically (1.0). Growth's value is compute/capacity, not
>   retention. (Do not claim "growth prevents forgetting.")
> - **R24**: a **multi-view teacher** (memory answers seen+para) makes the no-gold student
>   generalize (para 0.36→0.70, tracking the teacher exactly).
> - **R25** `lifecycle_distill.py` (capstone): the **faithful full loop with NO GOLD anywhere**
>   — transient per-stream multi-view scaffolds → distill into a growing dense model →
>   self-distill prior streams (replay) → discard memory. **Replay retains** (oldest-S0 seen
>   0.85→0.84, forget +0.01; all-seen 0.89 / all-para 0.85; hop preserved) vs **naive forgets**
>   (+0.60), as a single dense checkpoint with no memory at inference. Softer than gold-replay
>   (self-distill < gold-CE), the honest cost of zero gold + zero inference memory.
>
> **Reliability chapter (R26-R29), longer lifetime (6 rounds):**
> - **R26**: retention does NOT drift over 6 rounds (oldest-S0 forget +0.025), but 2 of 6
>   streams failed to consolidate (0.03/0.05) — an isolated per-stream scaffold problem.
> - **R27** (honest correction): a retr@1-based restart-on-collapse did NOT help — retrieval
>   was healthy (0.82-0.98) while those streams failed. The R26 "retrieval collapse"
>   hypothesis was **wrong**; retr@1 is a misleading proxy.
> - **R28**: the true signal is the teacher's **answer-recall** (top-16 injection → value
>   token). Probing that + restart rescued the failures (S3 0.03→1.0, S4 0.05→0.57).
> - **R29**: tightening the guard (answer-recall THR 0.8 + keep the *best* scaffold across
>   restarts) gives the reliable loop — all 6 streams consolidate (seen mean 0.88,
>   min 0.75), S0 forget +0.025, hop preserved, no gold, no inference memory.
> - **R30** (3-seed confirmation): the reliable loop holds across seeds — **replay all-seen
>   0.918, oldest-S0 forget +0.017, hop preserved**; naive forgets (+0.742). Not seed-specific.
>
> **R31 (capacity-saturation)**: grow(+1/round) vs nogrow(fixed **1 layer**) + replay over 8
> streams × 100 facts — **both retain all 8 streams at 1.0**. A single fixed layer + replay
> holds 800 facts; capacity is not the bottleneck at this scale.
>
> **R32 (composition SOLVABILITY GATE — before any grow arms)** `s2/composition_gate.py`, 3-seed:
> a cross-stream chain `A→B` (stream t), `B→C` (stream t+k), held-out no-memory `A→C`. **Gate A**
> (upper bound: full-FT direct-gold all single-step edges) gives **perfect single-step recall
> (1.000 all views) but 2-hop A→C = 0.000** (goldC-prob 2.5e-8). **Gate B** (direct 2-hop
> supervision) **train-fits 1.000 but held-out 0.000**. So even the upper bound does **zero**
> latent held-out composition — the two-hop curse / OOD compositionality gap at 0.5B. **This
> closes the latent-composition avenue** (as R31 closed fact-count): grow-vs-fixed arms would be
> 0-vs-0 and unattributable, so the gate correctly **blocked** them. Open reframes: in-distribution
> composition *grokking* (does depth/growth change onset/ceiling?), CoT/scratchpad, or accept the
> bottom line below.
>
> **R33 (baseline BAKEOFF — the proven loop vs standard CL)** `s2/lifecycle_bakeoff.py`, **3-seed**,
> 6 rounds × 40, shared per-(seed,stream) scaffold/answer-recall/eval (teacher trained once, shared
> across arms → method-attributable). Arms: ours (replay-consolidation, no gold, no inference memory),
> naive (no replay), continued-FT (GOLD new-stream, no replay), LoRA-merge (per-stream adapter merged
> in, no replay), external-memory (persistent bank). all-seen / all-para / oldest-S0-forget:
> **ours 0.914 / 0.826 / +0.00**, naive 0.408 / 0.331 / +0.71, continued 0.460 / 0.472 / +0.85,
> LoRA-merge 0.367 / 0.312 / +0.73, extmem 0.875 / 0.690 / −0.01. Takeaways: (1) **replay-consolidation
> is the only arm that retains the lifelong stream with no inference memory and no old-gold** (matches
> R30's 0.918) — every in-weights no-replay baseline forgets catastrophically (age-graded), **including
> continued-FT with a gold advantage** (0.460 vs 0.914 → replay, not signal, drives retention);
> (2) **LoRA-merge both forgets and destroys base capability** (hop 0.205→0.062); (3) **ours (no memory)
> beats external memory on both seen (0.914 vs 0.875) and para (0.826 vs 0.690)** — knowledge-into-weights
> matches/beats a retrieval bank while dropping the inference dependency (extmem also transiently
> collapses, R17-style). The contribution is now comparative, not just internal.
>
> **R34 (composition GROKKING — the last growth probe)** `s2/composition_grok.py`. R32 closed
> *OOD* composition; R34 tests *in-distribution* composition (2 relations, small shared-bridge set so
> held-out A's (B,C) are exercised by other A's trained 2-hop; 13k-token vocab; per-example derange
> control). **Reachable and fast at +0**: held-out 2-hop groks to ~0.98 by step ~50-500 (unlike R32's
> 0.0). **Depth sweep +0 vs +4** (2-seed 10k-step + 4-seed fine-onset): **growth robustly FAILS** —
> onset tied (4-seed: +0 mean 87.5 vs +4 100, +0 marginally earlier), convergence ceiling tied
> (+0 0.983 ≥ +4 0.979), only a transient +0.031 at 1500 steps (< the +0.10 bar, reverses by
> convergence), and +4 is ~15% slower throughout. So composition is **OOD-impossible-for-all (R32)
> or in-distribution-easy-for-+0 (R34)** — in both regimes growth is unjustified. 4th independent
> growth negative (after R23 retention, R31 capacity, R32 OOD composition).
>
> **R35 (oracle/EWC brackets)** `s2/lifecycle_bakeoff.py`, 2-seed, around R33's replay-consolidation.
> Bracket (all-seen): naive/loramerge 0.37-0.39 < **EWC 0.456 ≈ continued-gold 0.460** < **ours 0.890**
> < **oracle (gold-old replay) 0.994**. Two clean reads: (1) a real regularizer (online-EWC) beats
> naive but is *crushed* by replay (0.456 vs 0.890) and still shows an age gradient — regularization
> can't replace rehearsal, so ours's dominance isn't a weak-baseline artifact; (2) **ORACLE_GAP =
> +0.10 seen / +0.16 para** — no-gold self-distill is close to but not *within* the gold-old ceiling;
> having old gold still buys ~0.10-0.16 (bottleneck = scaffold answer quality / snapshot fidelity), a
> lever to close, not a refutation. Closes the replay chapter; the remaining open problem is
> rehearsal-FREE CL (R36-I).
>
> **R36-I (frontier: rehearsal-free interference-aware writing) — POSITIVE** `nswrite`, 2-seed. Write
> new-stream knowledge with NO replay by projecting each fixed trainable Linear's gradient off the null
> space of old-stream input activations (per-module low-rank basis, training-state only; bias frozen;
> single dense checkpoint). Result (all-seen / mean-forget / newest): **nswrite 0.792 / +0.119 / 0.975**
> vs naive_fixed 0.485 / +0.425, ewc 0.494, and the matched-rank **random-basis control 0.490** (≈ naive).
> Four clean findings: (1) rehearsal-free nswrite beats the no-projection floor by **+0.30 all-seen**,
> closing ~40% of the gap to replay (0.910) with **no replay**; (2) **attributable to interference-
> awareness** — the random-basis control does nothing (0.490≈naive) while the activation basis captures
> **35× the gradient energy** (occupancy 0.92 vs 0.20) and gives the whole gain; (3) **zero plasticity
> cost** — newest tied at 0.975 across all arms, so nswrite *dominates* the stability–plasticity frontier;
> (4) occupancy rises 0.74→0.92, but **R36-I phase-2a (24 streams) shows this is NOT a capacity wall**:
> eff-grad stays ~0.22 (not →0) and newest stays high (0.68–1.0) while old retention degrades to ~0.33 —
> i.e. writable gradient and plasticity are intact; the failure is **protection-quality** (the
> input-activation null space preserves the hidden response but not the answer-token margin; per-round
> drift accumulates worst for the oldest, which — under old-first `update_U` — is the *most*-protected
> stream). So occupancy alone is NOT a growth-necessity signal (needs occupancy + low eff-grad + newest
> failure, which don't co-occur here). **Growth is off.** **Margin-bilinear FALSIFIED:** protecting the
> answer-margin subspace (`V_out V_outᵀ·G·U_in U_inᵀ`, removes 38% of grad) does WORSE (all-seen 0.369)
> than blunt input-only nswrite (removes 94%, 0.474) at matched newest — more protection retains better,
> so the residual is protection *incompleteness/drift*, NOT a wrong protected object. "Protect-by-direction"
> writers plateau ~0.47 (12 streams) rehearsal-free, below replay ~0.91; the next different primitive is
> closed-form FUNCTIONAL editing (constrain old key→value outputs) or attacking per-step drift.
> nswrite remains the first attributable replay-free plasticity-preserving forgetting reduction; partial.
> **DRIFT SWEEP — POSITIVE (the real lever):** nswrite's residual forgetting is per-step drift, not
> wrong-target. Writing LESS (LD_STEPS 1000→300, LR 1.5e-4→7.5e-5) lifts 12-stream all-seen **0.474→0.652
> (+0.18)** while newest *rises* 0.85→0.97 (so not under-fit — genuine drift reduction: 1000 steps
> massively over-writes a 40-fact stream, the excess just drifts old knowledge). Rehearsal-free retention
> now 0.652 vs replay ~0.91, purely by write-budgeting — no new mechanism, no replay, no growth. **But it
> ATTENUATES with horizon:** at 24 streams the same low-drift config gives only 0.385 (vs 0.356 high-drift,
> +0.03) — the +0.18 short-horizon gain shrinks to +0.03. Cumulative drift over many writes + imperfect
> input-null-space protection dominate long-horizon; write-budgeting alone does NOT scale. So drift-control
> is a real but partial component; the long-horizon rehearsal-free gap to replay stays large (0.385 vs 0.91
> at 24 streams). **Optimizer-leakage ruled OUT (diagnostic):** hypothesised AdamW preconditioner/weight-
> decay undoing the projection — but the realized-ΔW leak `‖(ΔW·U)Uᵀ‖²/‖ΔW‖²` is tiny even at baseline
> (0.033), and removing ALL leakage (reproject → ΔW exactly ⊥U, freeze norms, wd=0) gives no retention
> gain (24: 0.402 vs 0.385). So constraining the Linear weight change ⊥ old-INPUT directions is
> FUNDAMENTALLY INSUFFICIENT — the limiter is representational (preserving each layer's linear response
> to old inputs ≠ preserving the end-to-end answer through nonlinearity), not leakage/capacity. The whole
> input-direction-protection family (nswrite/drift/margin/reproject) tops out ~0.65@12 / ~0.40@24 streams
> rehearsal-free vs replay ~0.91. **Value-anchored functional editing is algebraically bounded by this:**
> enforcing `W_new·U = W_commit·U` is identically `ΔW·U=0` (= the reproject/full-fix constraint, already
> tested → 0.402@24); the output-projected `Vᵀ W U = C` form is strictly *weaker* (margin class); only a
> paired old-activation/value sketch is stronger, and that is compressed activation *rehearsal*, not
> rehearsal-free. So the input-direction / local-linear-summary family is a mapped, bounded frontier —
> **no free lunch in the compressed local class.**
>
> **REHEARSAL-FREE FRONTIER (R36-I) — mapped, bounded (honest result, not a failure):**
> `naive no-replay  <  nswrite / input-direction protection  <<  replay-consolidation  ≤  gold-old oracle`.
> nswrite is the first attributable rehearsal-free, no-inference-memory improvement over naive writing;
> write-budgeting adds a big short-horizon gain; but the whole family has a long-horizon representational
> ceiling (~0.40@24) far below replay; optimizer-leakage and capacity are ruled out; margin and functional
> editing don't cross it. **R36-C now also rules out the strongest first-order primitive** — global
> answer-gradient OGD (both margin and canonical CE-to-gold, exact realized-ΔΘ semantics) is ≈ naive
> (all-seen 0.408 vs naive 0.390 vs nswrite 0.673) because it captures only ~5% of the write-gradient at
> the standard config and is rank-inefficient + memory-bound. So the *entire* first-order protect-by-
> direction family is bounded well below replay. **Replay-consolidation (R33/R35/EV) remains the proven
> durable knowledge-into-weights method at this scale; next primitive = minimal-footprint rehearsal (A).**
>
> **R36-EV (EXTERNAL VALIDITY of the replay-consolidation positive) — PASS** `s2/lifecycle_bakeoff.py`
> `BK_DATA=kg`, 2-seed, 6 streams × 40 facts, log `docs/cloud_results/kg_bakeoff_r36ev.log`. Re-ran the
> R33/R35 bakeoff on **KG-shaped counterfactual triples over ~80 REAL entities** (Napoleon/Einstein/…,
> relations job/pet/drink/city/tongue, single-token objects, two natural-language surface forms per
> relation for seen-vs-para), with a frozen-base-recall screen (base seen 0.000 / para 0.033 ≤ 0.15, so
> the model is *not* already answering — every hit is learned). This tests whether the R33 positive was
> an artifact of the synthetic `"{name}'s {attr} is"` template. It was not. Result (mean/2 seeds):
> **ours (replay self-distill) all-seen 0.919 / all-para 0.802 / oldest-S0 forget +0.000**, vs
> **naive 0.287 / 0.248 / +0.838** and **ewc 0.348 / 0.304 / +0.863**, with **oracle (gold-old) 0.954 /
> 0.933 / +0.000**. Against codex's pre-registered pass bar (`qa/codex/2026-07-06.03.14.20.md`): ours
> beats naive/ewc all-seen by **+0.63 / +0.57** (bar ≥ +0.25) ✅; oldest forget +0.000 (≤ +0.10) ✅;
> all-seen 0.919 ≥ 0.75 ✅, all-para 0.802 ≥ 0.60 ✅; base-hop 0.205→0.201 drop 0.003 (≤ 0.03) ✅;
> ORACLE_GAP seen +0.035 / para +0.131 (≈ R35 scale, self-distill ≈ gold-old upper bound) ✅; single
> dense, no gold-old, no inference memory ✅. **The replay-consolidation positive is NOT template-specific
> — it holds on real-entity KG-shaped counterfactuals.** Honest caveats: (a) newest-stream fresh recall
> 0.75 (< the 0.91–1.0 of earlier streams) — one harder stream / seed variance, teacher-fresh mean ~0.93;
> (b) the oracle gap is **para-dominated** (+0.131 para vs +0.035 seen): self-distillation without gold
> trails the gold-old oracle specifically on *paraphrase generalization*, the known honest cost of
> zero-gold self-distill (R25/R35). Net: R33/R35 external validity confirmed; replay-consolidation is the
> robust positive across both synthetic and KG-shaped data.
>
> **R36-C (rehearsal-free: answer-level OGD — the STRONGEST first-order primitive) — CLEAN NEGATIVE**
> `s2/lifecycle_bakeoff.py` `run_ogd`, logs `docs/cloud_results/ogd_ce_pilot_r36c.log` (+ smoke/rank
> scans `ogd_{ce_smoke,rank64,rank96,rankscan}_r36c.log`), 2-seed 6×40. To close (not strawman) the
> rehearsal-free frontier, we tested the one primitive genuinely *different* from nswrite's per-module
> input-null-space: **joint flattened Orthogonal Gradient Descent** — store a low-rank basis `Q` of the
> old-answer gradient w.r.t. ALL trainable params jointly, project `g ← g − QQᵀg` (routes through the
> nonlinearity/norms/residual/lm_head; couples modules, which the per-module `V⊗U` factorization cannot).
> Two objects: `margin` (logit_gold−runnerup) and **`ce_gold`** (−logP(gold), the canonical OGD old-task
> loss). **codex review-gate caught a real confound** — projecting the gradient then AdamW (precond +
> decoupled decay) does NOT keep the *realized* ΔΘ ⊥ Q — fixed by reprojecting the actual ΔΘ each step
> (exact-OGD; `upd-leak ≈ 0.01`, so results are not a semantics artifact). **Result: OGD ≈ naive.** At the
> standard 6×40 config: **ogd_ce all-seen 0.408 / oldest-forget +0.762** vs **naive_fixed 0.390 / +0.762**
> (Δ +0.018, far below the +0.15-vs-naive lower-bound gate and the +0.05-vs-nswrite gate), while
> **nswrite 0.673 / +0.500** (occ 0.925) protects and **ours(replay) 0.877 / −0.013** is the reference.
> `margin`-OGD is the same (occ 0.052 ≈ naive). Mechanism: at 6×40 the joint answer-gradient basis
> captures only **~5% of the write-gradient energy** (occ curve ~0.03–0.05, flat as Q grows to rank 64) —
> the direction that reduces new-stream loss barely overlaps the old-answer-gradient subspace, so
> projecting it off changes ~nothing (eff-grad 0.975, plasticity untouched; newest tied 0.887). **Two
> structural limits:** (1) *rank-inefficiency* — a rank-64 GLOBAL basis is negligible in 29.8M-dim (occ
> IS ~36,000× a random subspace, so structure exists, but covering the write direction needs many hundreds
> of dims); a per=80 diagnostic reached occ 0.18 but retention there was untested and per=40 is the
> program's standard config. (2) *memory-bound* — the per-stream basis merge is O(P·rank); rank 96/160
> **OOM on 32 GB**, and even rank-64 OGD peaks **23 GB** (vs nswrite 5.5 GB). So flattened OGD cannot
> cheaply reach nswrite's factorized coverage (occ 0.925 at 5.5 GB). **Label hygiene:** OGD uses no old
> replay and no inference memory, but builds `Q` from committed old prompts + **answer identities at commit
> time** (`uses_commit_answers=true`) — less label-clean than nswrite, and it still fails. **Conclusion:**
> the entire *first-order protect-by-direction* rehearsal-free family — per-module input null-space
> (nswrite), margin-bilinear, and now global answer-gradient OGD — tops out well below replay; the harmful
> forgetting is nonlinear/higher-order motion no first-order gradient-orthogonality constraint intercepts
> at affordable rank. This **earns** (not assumes) the rehearsal-free bound and motivates the pivot to
> **Option A: minimal-footprint rehearsal** (how small can training-time rehearsal get — K stored/
> self-generated probes per old stream — while retaining), the practical successor to a bounded frontier.
>
> **Honest bottom line (R19-R36):** the demonstrated, robust, multi-seed contribution is
> **consolidation-via-replay into a single dense checkpoint** — a lifelong no-gold stream
> retained with no external memory at inference, and (R33) **it beats standard continual-learning
> baselines** (sequential no-replay, continued-FT-with-gold, LoRA-merge) while matching external
> memory without needing memory at inference. **Growth (the "小 → 大" framing) is NOT
> justified by any of these experiments**: R23 shows growth doesn't drive retention, R31 shows
> it doesn't add capacity even at 800 facts through 1 layer. Justifying growth requires what
> these synthetic fact/hop tasks don't stress. **Every probed growth axis at this scale is now
> negative**: retention (R23, replay does it), capacity (R31, 1 layer holds 800 facts), OOD
> composition (R32, impossible for all incl. the upper bound), and in-distribution composition
> grokking (R34, +0 already ~0.98 and depth gives no onset/ceiling advantage at matched compute).
> Those synthetic *capability/capacity* probes are exhausted. **But R36-I re-opens growth on a
> principled footing**: once you attack forgetting rehearsal-free (interference-aware `nswrite`, which
> reduces forgetting +0.30 over the floor with zero plasticity cost), the **interference-free write
> subspace measurably saturates** (occupancy 0.74→0.92) — and *that* saturation, not fact-count, is the
> first candidate regime where function-preserving growth could be genuinely necessary. Phase 2 tests it:
> grow-on-saturation, **depth vs width/parallel-adapter growth** at matched compute — the first principled
> test of *what kind* of growth CL needs. Until that result, still do not *claim* growth helps.
> The memory is a *training scaffold* (not an inference dependency); its one reliability knob is the
> scaffold's **answer** quality, guarded by an answer-recall restart. **Frontier status: rehearsal-free
> CL is partly cracked (nswrite ≫ floor, attributable, plasticity-free) but not solved (still < replay);
> the live thread is nswrite + growth-on-saturation.**

Consolidated report of the proxy (toy) + real-model (Qwen2.5-0.5B/1.5B) experiments.
Every number below is reproducible from a script in `s0/` with its log in
`docs/cloud_results/`. Cloud runs: RunPod RTX 5090 / 3090Ti (Blackwell/Ampere,
torch 2.8+cu128), 2026-07-01/02.

## Thesis (as repositioned by the evidence)

The goal was "continually grow a small model into a large one." The experiments
**refuted the naive reading** ("grow to become more capable per parameter") and
**support a precise one**:

> Grow / externalise to **keep learning WITHOUT forgetting**. A small model becomes
> a large *system* over its lifetime by attaching new knowledge to an **external
> memory** or **isolated growth branches**, addressed by a **trained key-retrieval**
> that scales to a real lifetime of facts and is robust across model scale — NOT by
> re-training shared weights (which forgets) and NOT because depth makes it smarter
> per parameter.

## The load-bearing result: no-forgetting on a real model

Sequential fact-learning: K sessions × facts, learned one session at a time; recall
every prior session at the end. Three carriers, **in-place and growth train the
identical per-session params (top-2 decoder layers)** — only shared-vs-isolated
differs.

| carrier | 0.5B K=4 (`qwen_capstone`) | 0.5B K=6 N=3 (`qwen_capstone2`) | 1.5B K=6 (`qwen_capstone2`) |
|---|---|---|---|
| in-place (shared weights) | 0.41 (only newest intact) | 0.39 (oldest S0 0.23) | **0.41** — still forgets |
| memory (external, router-free) | 0.99 | 0.99 (S0 0.98) | **1.00** |
| growth + routing | 1.00 | 1.00 | **1.00** |

**In-place fine-tuning forgets catastrophically at both 0.5B and 1.5B** (the worry
that a bigger model would interfere less did not happen). Memory and growth retain.

## Four robustness axes — all closed (each via "discover bottleneck → clean fix")

1. **No-forgetting mechanism** (`qwen_capstone`, `qwen_capstone2`): in-place forgets;
   memory & growth retain. Multi-seed, 0.5B & 1.5B.

2. **Bank scale → ~10k facts** (`qwen_memscale`, `qwen_memscale2`, `qwen_memscale_big`):
   router-free retrieval@1 stays ~0.97 and top-k(ANN) hit-rate = 1.00 from 128 up to
   10 000 facts. Two bottlenecks found + fixed: (a) **train/eval mismatch** — training
   over in-batch keys but evaluating over the whole bank collapsed recall at large N;
   fixed by **full-bank training** (N=1500 recall 0.995 vs collapse before). (b) at 10k
   with multi-token entity names, **mean-pooled value readout dilutes the answer**
   (recall 0.06); fixed by **answer-position (last-token) value readout** (10k recall
   0.80, 4k 0.975). Residual rare injection collapses handled by **restart-on-collapse**.

3. **Model scale → 0.5B & 1.5B** (`qwen_capstone2` @ QWEN_MODEL): the no-forgetting
   contrast is unchanged at 1.5B (table above). Porting the exp-A fixes (cache /
   full-bank / restart) into the memory arm made 1.5B both faster and stable.

4. **Router-free for BOTH carriers** (`qwen_growroute`): memory retrieves the VALUE by
   key; growth retrieves the BRANCH by key — same trained key-retriever, no oracle
   session-id. in-place 0.49 / growth-oracle 1.00 / **growth-routed 0.99 at routing
   accuracy 0.99**. (Raw-feature NN routed at chance 0.27 → a trained retrieval router
   at 6000 steps reached 0.99; retention is gated by routing accuracy.)

**Unifying component:** a single **trained key-retrieval** (proj_k/proj_q + InfoNCE
on frozen Qwen features) underlies all of it — it retrieves values (memory), selects
branches (growth), and routes sessions, and it scales (retrieval@1 ~0.97 to 10k). ANN
gives speed, not accuracy (top-k hit-rate is already 1.0).

## Honest negatives (what did NOT work — these shaped the thesis)

- **Growth-for-capability is real but CADENCE-critical (a corrected story).** Naive
  growth often fails: matched-param in-place beats growth on real Qwen (`qwen_growcap`
  0.98/0.98/0.94 vs 0.96/0.92/0.87), the toy high-K breakthrough is a **lottery**
  (`diag_grow_hops_ms`), and **grow-EVERY-stage LOSES** to fixed-small on a curriculum
  (`diag_growlarge` 0.47 vs 0.54 — re-warming burns budget). BUT the failure is
  cadence, not growth per se: **ONE controller-timed grow ADDS capability** —
  `diag_growlarge3` (same curriculum/budget): grow-every-stage 0.47 < fixed-small 0.54
  < **once-mid (one grow L2→L6) 0.72** < fixed-L6-from-scratch 0.82 (K7: once-mid 0.44
  vs fixed-small 0.17). So grow *rarely and well-timed* (exactly what the §28 Ω / robust
  controllers pick) genuinely makes a small model smarter; frequent naive growth is the
  failure mode. Boundaries that stand: from-scratch-at-final-size still wins if you know
  the size and can retrain (continual learning can't); and DISTILL-into-core COMPOSITION
  (cross-session 2-hop) does **not** emerge at toy scale (`diag_compose`, even at
  grokking length) — that "smarter" path awaits real scale. (Multi-seed Qwen check: the cadence capability gain is WITHIN NOISE on a pretrained Qwen-0.5B — grown 0.76 == fixed-small 0.76 over 3 seeds — so grow-for-capability is robust only for a genuinely capacity-bound base like toy-L2, not a pretrained deep model.)
- **The naive growth controller failed.** A one-chunk training-loss-delta trigger grows
  on temporary plateaus and **loses to from-scratch** (`diag_controller2`). Fixed by a
  **held-out-slope + patience** trigger (`diag_controller3`): grows once to the sweet
  spot, beats fixed L2/L4/L8 (mean 0.68–0.76; the N=3 win was optimistic vs N=8).
- **Mean-pool value readout** and **raw-feature routing** both failed at scale and were
  replaced (answer-position readout; trained retrieval router).

## Does growth make it SMARTER? — the multi-round resolution

The naive "grow → more capable per parameter" is false, but a precise version is TRUE.
Established over rounds 1–6 (all in `s0/diag_growlarge*.py`, `diag_autocap*.py`,
`diag_stream.py`, `diag_depthcross.py`, `qwen_growcap_curric.py`):

- **Cadence is everything.** Growing every stage LOSES to fixed-small (re-warming burns
  budget); ONE well-timed grow ADDS capability (once-mid 0.72 vs fixed-small 0.54).
- **Autonomous + robust.** A controller picks the timing itself (autocap 0.77); with a
  keep-best checkpoint it reaches 0.84 and never collapses (autocap2), beating every
  fixed baseline including from-scratch-at-final-size.
- **Growth wins at DEPTH.** Depth-crossover (`diag_depthcross`): from-scratch wins only
  for shallow targets; with keep-best, incremental warm-start growth beats from-scratch
  at **every** depth (L4 +0.08, L6 +0.32, L8 +0.29) — the toy analogue of real-LLM
  stacking efficiency. Width-robust (reproduces at d=256).
- **Why grow at all, if from-scratch-large wins?** Because a continual learner in a
  STREAM cannot do from-scratch-at-final-size (unknown final size, no full replay).
  Among FEASIBLE stream strategies grown wins decisively (`diag_stream`: 0.84 vs
  fixed-small 0.54 vs retrain-each 0.34), and even beats the infeasible oracle when the
  target is deep.
- **Honest bounds.** (1) On a PRETRAINED Qwen-0.5B with a shallow trainable stack the
  capability gain is WITHIN NOISE (multi-seed: grown 0.76 == fixed-small 0.76) — growth
  helps only for a genuinely capacity-bound base, and a pretrained deep model with a few
  top layers isn't that. (2) COMPOSITION-via-distill (cross-session 2-hop) does not
  emerge at toy scale even at d=512 — that "smarter" path awaits real scale.

**Net:** grow-and-get-smarter is real, specifically as *sparse, controller-timed,
keep-best, incremental deepening* in the *streaming / continual* regime at *genuine
depth* — not as naive frequent growth, not on an already-capable pretrained shallow
probe, and not (yet) as latent composition.

## Neural vs heuristic — honest audit of the "all-neural" constraint

The design goal was **fully neural modulation**. Status: the **substrate is neural /
learned**, but the **meta-control layer is still hand-coded heuristics** — so "all-neural"
is **not yet achieved**.

**Neural / learned:** key–query retriever (proj_k/proj_q + InfoNCE), value encoder/
decoder, injection gate σ(net([H,R])), commit/admission gate, versioning ctx_enc, the
grown branch layers, and the frozen Qwen features. Routing uses *learned* similarities
(not rules).

**Growth controller — now NEURALIZED (§28 first cut, `diag_omega.py`):** a learned
policy Ω (small MLP over per-chunk observables, meta-trained by REINFORCE across a
K-hop task distribution) replaces the hand rule for *when/how-much to grow*. Held-out
(unseen kmax): Ω grows appropriately and **matches the hand heuristic** (kmax5 0.81 vs
0.84; kmax8 0.73 vs 0.78) and beats fixed-L2/L4 — so the growth *decision* is now a
learned, generalizing neural policy. Caveat: Ω does **not yet beat** the heuristic or
fixed-deep-from-start; "learned > hand-crafted" is still open (needs reward shaping /
more episodes / better credit assignment). Earlier v1 degenerated to never-grow until
bigger batch restored the grow-pays regime.

**Write-vs-grow decision — now NEURALIZED (`diag_writegrow.py`):** a gate over the
frozen-core item feature learns to route each incoming item to memory (write) or
growth (consolidate) — held-out routing 1.00, reward = oracle, >> always-one. So the
integrated loop's modulation is now neural at every decision point: what-to-store
(admission gate), write-vs-grow (this), when/how-much-to-grow (§28 Ω), route/recall
(trained retriever). Hardened to the genuine COST-TRADEOFF too (`diag_writegrow2.py`):
with every item a fact and the right choice depending on system state, the gate learns
a state-dependent policy (consolidate-threshold drops as memory load rises), matching
the oracle (97% of its reward) and beating fixed heuristics.

**Still heuristic / hand-set (NOT neural):**
- restart-on-collapse (quick-check < 0.5 → reinit): hand threshold;
- readout / selection: top-k=32, answer-position last-token, and the discrete
  argmax→branch swap are hand-chosen (the router is learned, the *selection* is hard);
- hyperparameters: LTOP=2, "grow 2 layers", temperatures, lr, curriculum;
- the capstone's growth-**oracle** column was a hand-given session-id (later *replaced*
  by the trained router in exp C).

**To close it:** the §28 meta-learned Ω controller — learn "when/how-much to grow,
where, how to route/read out" across simulated lifelong streams — is the open piece
that would make the modulation end-to-end neural.

## What is validated vs open

**Validated (this repo):** no-forgetting on real Qwen (0.5B & 1.5B), router-free for
memory and growth, memory scaling to ~10k facts, an autonomous grow-to-sweet-spot
controller with a robust signal, function-preserving growth transfers to Qwen
(`qwen_grow`, Δlogits=0), and — the pieces COMPOSE into ONE autonomous lifelong loop
(`diag_system.py`): a single system accumulates facts (recall 0.88 / 192) while
growing its core (depth 2→6, capability 0.33→0.47), with memory re-synced cheaply as
the core drifts. The growth *decision* is neuralized (§28 Ω, matches the heuristic).
And the AUTONOMOUS controller realises grow-AND-get-smarter by itself
(`diag_autocap.py`): on the escalating curriculum it picks the growth timing and
reaches mean 0.77 — beating fixed-small 0.54, grow-every 0.47 and even hand-tuned
once-mid 0.72 (variance: 2/3 seeds strong, 1 collapsed).

**Open / caveats:** 0.5B–1.5B only (not 3B+); synthetic single-token-**answer** facts;
2–3 seeds on the real-model runs; routing validated over ≤~100-fact session banks
(large-session-count routing rides on the exp-D scaled retrieval); the
growth controller signal is hand-built (not yet the meta-learned Ω of §28).

**Round-17 correction (load-bearing — do not overclaim the capstone):** the clean 2-seed
capstone re-run (`qwen_capstone_lifelong`, now with a MONO-REPLAY arm) shows the decomposed
external MEMORY is **collapse-prone at 4800 facts**, not merely "residual rare" instability —
recall was seed0 0.628 / seed1 0.032 (a full retrieval collapse from phase 0), mean **0.330**.
The round-16 "0.909 / decomposed does both" was a **favorable-seed artifact** of the same
sharp-temp cold-start InfoNCE collapse (rounds 10-11). The robust no-forgetting baseline here
is **MONO-REPLAY (recall 0.961, stable across both seeds)**, which naive-monolith forgetting
(0.492) and the current decomposed memory (0.330) both lose to. The decomposition's potential
edge over replay is COST (frozen features computed once, never backprops the 0.5B backbone;
storage O(N) for both), but that is moot until the memory trains reliably. Being fixed in
round 18 via a robust-recipe stabilizer (temp+LR warmup + restart-on-collapse on the cold
phase-0 stage; `qwen_mem_stability.py`).

## Next steps (in rough priority)

1. **Stabilize the decomposed memory** (round 18) — eliminate the 4800-fact seed-fragile
   collapse (temp+LR warmup + restart-on-collapse) so decomposed recall is reliable, THEN
   redo the cost accounting vs replay (`qwen_cost_accounting.py`, drafted).
2. **Consolidate multi-seed** the load-bearing real-model numbers (capstone, C, D) at
   N≥5 to tighten the estimates.
3. **§28 meta-learned Ω** — replace the hand-built controller signal + routing/growth
   decisions with a learned controller over simulated lifelong streams.
4. **3B+ scale** and **larger session counts** for routing.
5. A better value encoder than answer-position readout (learned pooler) to lift the 10k
   recall from ~0.80 toward the 4k level (~0.97).

## Reproduce

Cloud: recreate a torch-cu128 env, `git clone`, `pip install transformers accelerate
safetensors`, then `python -m s0.<script>` (env `QWEN_MODEL` to pick size). Scripts and
their logs: `qwen_capstone(2)` (no-forgetting), `qwen_memscale(2)`/`qwen_memscale_big`
(bank scale), `qwen_growroute` (router-free growth), `qwen_growcap` (growth-vs-in-place),
`diag_controller3` (robust controller), `diag_grow_hops*`/`diag_grow_hops_scale*`
(growth-capability audits). Logs in `docs/cloud_results/`; condensed state in
`docs/memory/s0-step0-state.md`.
