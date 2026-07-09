# Research Map — rehearsal-free, weights-only continual learning (living document)

> **Purpose.** A single living map of the known/unknown, the mechanism ledger, the build order, and the
> pass/null gates. Update the STATUS column and the "last updated" line every time an experiment returns.
> This is the plan-of-record; `FINDINGS.md` is the evidence log, `docs/CAPSTONE.md` is the narrative.
>
> **Last updated:** 2026-07-10 (after R46: FIRST POSITIVE on the composition wall. A training-time,
> weights-only, NO-A→C-label, rehearsal-free consolidation primitive — hidden-state BRIDGE UNIFICATION
> (align h("A's friend is")→h("B"), direct cosine) — induces latent 2-hop composition 0.42 (0.62/0.23
> seeds) where independent bindings + shared-hub + freq-matched + deranged controls ALL give ~0; atomic
> 1.0; grokking onset then plateau; unification cos 0.9995. Caveats: paraphrase null (BUT so is the direct
> upper bound → surface too hard, uninformative); seed-variable; synthetic single-token; NOT yet a
> lifecycle/retention result. This cracks the R32 failure mode (atomic knowledge in weights, bridge not a
> reusable node) — but "composition solved" it is NOT. NEXT = R47 lifecycle bridge consolidation (phase1
> old B→C, phase2 new A→B + attractor, eval A→C + old-B→C RETENTION vs replay/distill controls) + mechanism
> probe cos(h("A's friend's"),h(B)) (raw 2-hop bridge position). Growth still OUT until bridge unification
> saturates. This is the association/composition shot from the 3-round brainstorm — the ONLY wall with a
> positive.)
>
> **Prior last-updated:** 2026-07-09 (after R43-ladder: surprise is a RISK SIGNAL, NOT a reliable ROUTER on real
> text — the R41 routing win was a bimodal-construct artifact; real text is unimodal so there is no
> safe-to-skip class. **STRATEGIC PIVOT:** the item-replay+surprise-routing+generic-growth family is at its
> ceiling for the FULL goal. Bank R33/R38 as the engineering result; the one wall-moving shot = R44
> schema-compressibility (does reusable structure make marginal cost/book FALL?). Adopt the honest reframe
> below. Next = R44 schema_comp with matched-budget + deranged control + slope pass/fail.).
>
> **HONEST REFRAME OF THE GOAL (2026-07-09, codex+claude converged).** The full frontier goal (rehearsal-free
> + compositional + growth-justified + sublinear, unbounded arbitrary-fact memorization) is unsolved by us AND
> by the field, and our current mechanism family is near its ceiling. Replace it with a *scientific cost model*
> that is honest AND still ambitious:
> `lifetime cost = reusable structure + exceptions + write interference + recompression`.
> Claim progress iff the **reusable-structure term grows SUBLINEARLY with books** and the **exception term is
> measured honestly**. Story: small models first COMPRESS and REUSE structure into weights; growth is justified
> only by OBSERVED saturation, not aspiration. The composition wall is TWO walls: existing-schema ACTIVATION
> (base already knows biography/geography/causality schemas — probably reachable on real text, NOT killed by
> R32/R34) vs new-schema ACQUISITION (R32/R34 kill AUTOMATIC acquisition). R44 attacks the first.
> **Governing constraints (never relax):** knowledge must end in ONE dense checkpoint; no inference-time
> memory / router / task-id / retrieval; no joint full retraining; catastrophic forgetting measured
> explicitly (old-only, non-replayed, paraphrase EM over streams 0..R-2 is the headline).

---

## 0. The goal, stated honestly

Small model **continually reads and internalizes new knowledge into its weights**, without catastrophic
forgetting, without joint retraining, without external memory at inference — and eventually **grows** into a
larger model when (and only when) capacity/interference genuinely demands it. Concrete stretch target:
"read hundreds of books and absorb them into weights."

**Where we actually stand (strongest reproducible result):** read real Wikipedia text → self-quiz QA
targets → replay-consolidate into one dense checkpoint reaches **92% of RAG's closed-book answer quality on
held-out paraphrases (0.893 para-EM)**, retained across streams, no inference memory (R38-A). But that
retention still relies on **replay = O(#facts)**, and R38B-A shows it only survives zero-replay when the
knowledge is **already prior-anchored** (real text 0.63–0.74; independent invented facts collapse 0.16–0.18).

---

## 1. The organizing frame — decompose the lifetime cost

Every mechanism attacks a **different term**. Getting the term right is why keytie failed (it hit a term that
doesn't dominate). Current best accounting (codex-refined, 6 terms):

```
lifetime cost =
    target-construction cost                     # R38: raw reading fails; QA/answer-fn targets are load-bearing
  + reusable-structure bits (#patterns × bits)    # schema / abstraction  → schema_comp
  + exception bits (#exceptions × bits)           # independent facts that do not compress (irreducible)
  + write-path interference + nonlinear drift      # nswrite helps but SATURATES; grow_sep; merge-conflict
  + consolidation / recompression amortization     # replay repays O(N) unless tiered  → LSM
  + readout / composition activation cost          # R32/R34: stored pieces unused if the model won't execute
```

**Reframe that changes the x-axis (highest-leverage insight, R4):** cost is better measured as
**accumulated SURPRISE**, not raw #facts. Low-surprise facts (already implied by the model) cost ~0;
only surprising residuals cost bits. This is R38B-A restated. **Consequence:** every footprint claim should
be reported **per accumulated surprise bit**, not per fact. "Read 300 books" is sublinear iff most content
is low-surprise; genuinely novel knowledge stays expensive (an honest wall, not a dissolved one).

**Capacity reality-check (back-of-envelope, to be validated by R40):** at ~2 bits/param (Physics-of-LLMs
law), 0.5B ≈ 1e9 bits ≈ 125 MB factual budget. 300 books × 1e4 novel facts × 50 bits ≈ 1.5e8 bits ≈ 15% of
budget → **300 books is NOT a raw capacity wall at 0.5B**; the lever is **write efficiency**, not growth.
Growth stays unjustified until write-efficiency saturates in the fixed model. (Our writes are far from
information-optimal today; R31 = 800 facts/1 layer is a lower bound, not the ceiling.)

---

## 1c. Architecture frame — addressing, the CLS convergence, the feasibility ladder (2026-07-07, codex-vetted)

**Thesis (narrowed after codex):** the immediate limiter is **addressable, low-interference write/read
allocation under finite optimization, plus consolidation of compressible structure** — NOT raw bit capacity
(0.5B ≈ 1e9 bits ≈ ~1e7 facts, yet we wall out at ~1e2–1e3 facts). Every retired method (keytie/merge/EWC/
OGD/generic growth) tried to solve CL WITHOUT addressing. **Caveat (codex):** do NOT read nswrite saturation
as a proven hard capacity/Gardner wall — R36-I 24-stream showed effective gradient stayed alive while old
retention degraded, so it's protection-quality + representational drift, not null-space exhaustion. Capacity
becomes relevant only after write/read/consolidation addressing is already routing correctly.

**CLS convergence (forced by constraints):** a sustainable CL system needs (1) fast non-interfering
addressable write buffer, (2) slow shared compressed store [=dense weights, HAVE], (3) consolidation fast→slow
[=replay-distill + schema_comp, HAVE], (4) routing signal [=surprise, R40-s3, HAVE]. The open pieces are #1
AND whether #3 can turn O(#facts)→O(#structures) on real regularity (not just #1 as I first claimed).

**Memory-layer ruling (codex):** a fixed-capacity, jointly-trained, content-addressed memory layer (PKM /
Hopfield / SDM / TTT-state) is ALLOWED iff it is checkpointed model parameters — fixed declared counted
capacity, single forward pass, no task-id, no growing item DB, closed-book eval, training cache deletable.
FORBIDDEN if it's one-raw-record-per-fact retrieved at inference / a growing DB / a hidden inter-episode cache
/ RAG-called-a-layer. **Scientific caveat:** a memory layer with NO consolidation is only *formally* legal —
it relocates the exception ledger into the model without solving the deep problem. The real win = fast writes
+ consolidation freeing/reusing slots.

**Feasibility ladder (climb, don't leap):**
- **Rung 0:** ✅ DONE — matched-stream guardrail PASSES: k0 squad 0.68 vs synth 0.24 (+0.44), k0_noanchor
  0.68 vs 0.22 (+0.46), no seed inversion (clears codex bars). Age confound was ~0.2 of the unmatched +0.68;
  categorical on/off-manifold effect survives clean. R40-s3 is now a clean matched Phase-0 pass.
- **Rung 1 (R41): MECHANISM VALIDATED — STRONG PASS.** surprise-gated replay (top-50% by frozen-base
  bits/token) beats random by +0.117 all-old / **+0.266 high-bits (exceptions)**, ties FULL replay on all-old
  (0.617) and beats it on exceptions (0.533>0.500) at HALF budget; beats even the source-oracle (bits/token is
  a sufficient router, no label needed); lowbits ties random (direction matters); on-manifold skipped stays
  safe (0.700). → deployable **O(#exceptions)** allocation policy, routed by the model's own surprise, no new
  arch. TODO: SEEDS=2 + budget ladder (0.25/0.5/0.75) for the exact exception-tail density number.
- **Rung 1½ (R42 Step-A census): ✅ DONE — the REAL-text density number is in.** `s3/census.py`,
  `docs/cloud_results/r42_census.*`. Tail at the R40 off-manifold anchor (τ=10 bpt): **squad_human 7.3%,
  wiki_gen 7.6%, news_gen 14.0%** (τ=8: 21/24/29%); constructed 50/50 overstated real tails ~4–7×.
  Distribution is smooth/UNIMODAL (no natural binary class → τ is a budget/risk knob); tail is
  TYPE-structured (novel proper names & open-class phrases, ~never numbers/dates); short answers have
  fatter per-token tails (length confound — stratify in Step B); part of the news tail is context-deictic
  probe noise (self-containedness screen needed). Quality: judge calibration 0.865 (≥0.85 gate), dup ≤2%,
  paraphrase stability corr 0.986. **Rung 2 density trigger NOT met** (CI-upper 0.20–0.37 at τ=8 vs ≈0.5
  trigger). → Step B: REAL-stratified budget ladder, tail-aware grid {f/2, f, 2f, 0.5}, f≈0.2–0.3.
- **Rung 1½ census-XL (R43): ✅ self-containedness screen → CLEANER, SMALLER tail.** After removing
  context-deictic probes ("what is the dog's name?"), SC exception tail is τ=8 ≈16–25%, τ=10 ≈6–8% (wiki
  τ=10 0.058, squad 0.076, news 0.075) — deictic probes were inflating R42's tail. 460 SC-gated probes,
  67 passages ≥3 SC probes → 7-stream census ladder. Rung-2 trigger still NOT met (SC τ=8 CI-upper <0.34).
  Reusable lesson: an LLM-judge user-turn must ASK the question or the model just answers the content.
- **Rung 2 (gated by density): fixed-capacity content-addressed associative memory layer for exceptions.**
- **Rung 3 (gated by Rung 2 wall): TTT/Titans (surprise-gated state=weights), local/predictive-coding update
  rules, VSA superposition (graceful √ interference vs hard null-space saturation).** Titans independently uses
  surprise (grad norm) to gate memory writes — external convergence with R40-s3.

## 2. Mechanism / reframe ledger

STATUS legend: ✅ positive · ❌ tested-negative · 🔬 queued (gated) · 💡 idea (unbuilt) · 📏 evaluation-contract only.

| # | mechanism / reframe | cost term attacked | STATUS | one-line verdict / gate |
|---|---|---|---|---|
| — | replay / self-distill → committed answerid targets (`ours_tgt_answerid`) | patterns+exceptions (item-wise) | ✅ | proven positive; but O(#facts), NOT rehearsal-free. The ceiling to beat. |
| — | real-text bridge: read→self-quiz→consolidate (R38-A) | target-construction | ✅ | 0.893 para-EM = 92% of RAG, weights-only. Reading alone fails; QA-target teacher is load-bearing. |
| — | `nswrite` interference-aware null-space write | write-path interference | ✅ | **best rehearsal-free writer**: old-para 0.742 ≈ 89% of replay, zero replay. But subspace occupancy **saturates** 0.74→0.91. |
| — | localized-write growth (`grow_local_decoy`, R37-A) | write-path (via capacity) | ✅(narrow) | first "growth load-bearing" result: NOGROW < naive; decoy 0.61 (< replay 0.875). |
| — | key-tie to frozen-base key (`keytie`, R39-A) | readout/prior-addressing | ❌ | +0.06 over naive, −0.38 vs nswrite. Anchoring the KEY ≠ protecting the WRITE. **Retired.** |
| — | EWC / online-EWC (R35) | write-path (regularizer) | ❌ | 0.456, crushed by replay 0.890. Regularization ≠ rehearsal. |
| — | answer-level OGD (`run_ogd`, R36-C) | write-path (direction) | ❌ | near naive. Direction protection ≠ function protection. |
| — | generic growth (width/depth/capacity) R23/R31/R32/R34 | — | ❌ | 4 independent negatives; growth not justified by generic capacity/composition. |
| — | sequential `loramerge` (fold into evolving dense) R33/R35 | write-path | ❌ | 0.367 all-seen, base-hop collapse 0.06. Sequential drift. (≠ parallel-merge below.) |
| 1 | **surprise-cost reframe / surprise-gated residual write** | changes x-axis of ALL terms | ✅(categorical) R40-s3 | PASS as CATEGORICAL on/off-manifold: squad k0 old-nonrep para 0.76 vs synth 0.08 (+0.68); bits/token 4.45 vs 11.37 (survives answer-length control). NULL as within-source continuous bits law (corr −0.05..−0.26, noise). → cost = **O(#off-manifold exceptions)**, not a smooth bits law. Founds Phase-1. |
| 2 | **interference-as-signal** → collision-discovered schema (unsupervised) | patterns (discovery) | 🔬 Phase-2 | current-current collision clusters recover R34 bridge structure above derange. Legal (no old items). |
| A | parallel-train-from-frozen-base + merge (`merge_ties`/`merge_sum`) | write-path (resolve at merge) | ❌ R40 | **RETIRED.** merge 0.22–0.25 < loramerge 0.50 << nswrite 0.975; collapses newest+base-hop; conflict 0.244. Independent-fact task vectors collide destructively at merge — worse than sequential. (Task arithmetic needs *related* tasks.) |
| B | schema-extraction consolidation (`schema_comp`) | patterns | 🔬 Phase-2 | O(#facts)→O(#patterns). Headline = footprint SLOPE dtargets/dA, not point. `schema_commit` vs equal-budget `item_k_matched` is the load-bearing control. |
| C | targeted pattern-separation growth (`grow_sep_decoy`) | write-path @ collision | 🔬 later | separate against collision-MODES (training-state), NOT nearest old fact (=rehearsal). Extends R37-A. Trigger ≠ occupancy alone. |
| D | LSM-tree multi-timescale weight tiers | recompression amortization | 💡 | make replay-touches/fact O(log N) not O(N). Not rehearsal-free; a compute win. Compare vs `ours_tgt`. |
| E | recombinant generative self-replay + coverage sketch | patterns (cheap maintenance) | 💡 | probe the manifold, don't index items (attacks self-indexing). Only valid inside schema regime; sketch must not become item ledger. |
| F | ICL context-distillation (`compact_cpt_qa_icl`) | target-construction | 💡 Phase-3 | use model's OWN in-context reading as the teacher; distill closed-book, drop context. Reduces teacher cost, not retention. |
| G | axioms + explicit-CoT execution engine | readout/composition | 💡 Phase-4 | R32 kills LATENT derivation (0.000), NOT explicit-CoT. If CoT executor groks held-out chains → readout is a trainable skill, store axioms not endpoints. |
| H | meta-learned write operator (learned optimizer/mask) | write-path (learned prior) | 💡 Phase-5 | "learn to continually learn." Highest build cost; easiest to fool via meta-distribution leakage. Build last, audit with #1/#2 diagnostics. |
| I | VSA / superposition + coding-theory (syndrome-drift) storage | bits/fact + interference substrate | 💡 | distributed √-degradation vs localized saturation; redundancy + drift detection. Risk: "a bigger LoRA in disguise"; needs base-capability controls. |
| J | sparsity / lottery-ticket / supermask allocation | write-path (structural) | 💡 | allocate sparse subnets from overparam net ("growth without growth"). Valid ONLY as write-time allocation merged to dense — NO inference-time mask routing. |
| K | loss-landscape geometry / permutation symmetry (Git Re-Basin) | write-path (why) | 📏 | diagnostic for merge (delta cosine, sign-conflict by layer), not a standalone build. |
| L | CLS patterns/exceptions split | evaluation contract | 📏 | report pattern-targets vs exception-targets vs slope separately. Not a standalone arm. |
| M | RG / Kolmogorov structure-function (MDL proxy) | evaluation contract | 📏 | description_length = schema_targets + exception_targets + residual_errors; slope must fall. Decision rule for schema_comp. |
| N | capability-delta metrics (downstream QA, contradiction, calibration) | metric suite | 📏 | ADD to (not replace) old-only retention. Require Pareto: capability gain w/o unacceptable forgetting. |

---

## 3. Build order (phased, each with a decision gate)

**R40 — Phase 0 + merge-smoke + capacity accounting (DONE 2026-07-07: merge RETIRED; surprise NULL on KG →
must re-run on s3 real-vs-synth; nswrite re-confirmed. See FINDINGS R40).** Original design was one tiny KG bakeoff:
`naive_fixed, nswrite, ours_tgt_answerid, loramerge, merge_sum, merge_ties`; instrument per-fact surprise
(base/current CE bits, gold margin, entropy, already-correct, delta norm) + merge-conflict (sign-conflict,
drop fraction, base-hop). Tiny config (`LD_ROUNDS=3 LD_PER=20 LD_SEEDS=1`, reduced steps) as smoke.
- **Surprise gate:** surprise buckets predict retention/forgetting better than fact-count/age → adopt
  per-surprise-bit accounting for all future claims. Else #1 is rhetoric; return to schema/growth.
- **Merge gate:** `merge_ties > loramerge` (baseline) → keep as baseline; `≥ nswrite + 0.05` old-para w/
  newest preserved, base-hop drop ≤0.03 (frontier) → merging becomes a main branch; `≈ ours_tgt` → changes
  program direction.
- **Capacity-efficiency gate:** retained-facts/surprise-bit beats naive/sequential-LoRA and conflict metrics
  explain the residual loss → write-efficiency is the lever (not growth).

**Phase 1 — surprise-gated residual writing.** low-surprise→low budget/no replay; schema-covered→shared rule
+ small residual; else→compact exception target. Headline = storage slope vs summed surprise.

**Phase 2 — schema_comp (supervised bound + unsupervised collision-discovery).** R34 bridge surface (held-out
2-hop groks 0.98, NOT the R32 0-vs-0 trap). Arms: `no_schema_fixed, schema_commit, schema_random,
item_k_matched (equal true-target budget, coverage-accounted), ours_tgt_answerid`. Headline =
`late_2hop_new_residual_final` + `held_2hop_old_final` with `schema_targets_per_A` falling as A_per_bridge
grows. Kill if item_k_matched ties schema_commit (gain is just target count) or only support-A's improve.

**Phase 3 — ICL context-distillation in WikiBridge.** does the model's own in-context reading replace
external QA-target construction (support-filtered to avoid hallucination poisoning)?

**Phase 4 — explicit-CoT execution gate (fork `composition_gate.py`).** store single-step axioms; train a CoT
executor on separate chains; test held-out A→C with no memory. Pass → readout is trainable, store axioms not
endpoints (reclassifies cost-term 6). Fail → no program-storage at this scale.

**Phase 5 — meta-learned write operator.** only after #1/#2/execution diagnostics exist to audit what it
learned. Report meta-training as pretraining-like cost; test on held-out stream FAMILIES.

**Growth (`grow_sep_decoy`) — inserted only when justified**, i.e. high-surprise residuals remain after schema
compression AND current-current collision clusters are dense/persistent AND fixed-capacity control cannot
store the residual without old drift AND grown arm beats nswrite/locality at matched newest+base-hop.

---

## 4. Known unknowns (open questions — update as answered)

1. **Does surprise predict retention?** ANSWERED (R40-s3): YES categorically (on/off-manifold: squad 0.76 vs
   synth 0.08 zero-replay, bits/token 4.45 vs 11.37), NO as a within-source continuous bits law (weak/noise).
   Cost = O(#off-manifold exceptions). Open refinement: SEEDS=2 + matched stream counts to firm the floor gap.
2. **Is parallel-merge a better route or just a missing baseline?** (R40 = ANSWERED NO: merge collapses
   below sequential loramerge for independent facts; task vectors collide. Retired.)
3. **What is the LEGAL growth trigger?** occupancy alone is a false alarm (R36-I). Candidate: persistent
   mutual collision among current-stream writes + surviving high-surprise residual. Unproven.
4. **Surprise circularity:** measured under base (clean, underestimates) or current (true cost, path-dependent)
   model? Plan: log both; headline on base-surprise as the reproducible unit.
5. **Does unsupervised collision clustering recover semantic schema, or just template/optimizer artifacts?**
   (derange control mandatory in Phase 2.)
6. **Is the readout wall (R32) a storage wall or an execution wall?** (Phase 4 CoT executor decides.)
7. **Can independent (high-surprise) facts EVER be retained sublinearly?** Current evidence says no (R38B-A
   lower bound). Target may be O(#patterns)+O(#exceptions), not O(1).
8. **Does write-efficiency approach the 2-bit/param law**, or is our overhead so large that 300 books is
   effectively out of reach at 0.5B without growth? (R40 capacity accounting = first estimate.)

---

## 5. Standing methodological rules (from the arc so far)

- Old-only, non-replayed, **paraphrase** EM over streams 0..R-2 is THE headline (guards prompt memorization,
  guards fresh-stream accounting artifacts — the R38B retraction lesson).
- Every positive re-tested under a stricter control before it's believed (random/equal-budget/derange).
- "Rehearsal-free" is strict: NO old prompts/answers/logits/activations/Jacobians/per-item targets in later
  writes. Aggregate training-state summaries (e.g. nswrite occupied basis) are allowed but must be named
  separately from a "no old-derived state of any kind" claim.
- Reclaim the pod after every run; push code + `docs/cloud_results/` + `FINDINGS.md` + `qa/`.
- codex is a peer, not an authority: verify the math, push back, run the control it (or you) might be wrong.
