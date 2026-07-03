# s0 — Findings: continual learning without forgetting, via a scalable key-retrieval

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
(large-session-count routing rides on the exp-D scaled retrieval); residual rare
injection-training instability (mitigated by restart-on-collapse, not eliminated); the
growth controller signal is hand-built (not yet the meta-learned Ω of §28).

## Next steps (in rough priority)

1. **Consolidate multi-seed** the load-bearing real-model numbers (capstone, C, D) at
   N≥5 to tighten the estimates.
2. **§28 meta-learned Ω** — replace the hand-built controller signal + routing/growth
   decisions with a learned controller over simulated lifelong streams.
3. **3B+ scale** and **larger session counts** for routing.
4. A better value encoder than answer-position readout (learned pooler) to lift the 10k
   recall from ~0.80 toward the 4k level (~0.97).

## Reproduce

Cloud: recreate a torch-cu128 env, `git clone`, `pip install transformers accelerate
safetensors`, then `python -m s0.<script>` (env `QWEN_MODEL` to pick size). Scripts and
their logs: `qwen_capstone(2)` (no-forgetting), `qwen_memscale(2)`/`qwen_memscale_big`
(bank scale), `qwen_growroute` (router-free growth), `qwen_growcap` (growth-vs-in-place),
`diag_controller3` (robust controller), `diag_grow_hops*`/`diag_grow_hops_scale*`
(growth-capability audits). Logs in `docs/cloud_results/`; condensed state in
`docs/memory/s0-step0-state.md`.
