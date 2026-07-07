# Research Map — rehearsal-free, weights-only continual learning (living document)

> **Purpose.** A single living map of the known/unknown, the mechanism ledger, the build order, and the
> pass/null gates. Update the STATUS column and the "last updated" line every time an experiment returns.
> This is the plan-of-record; `FINDINGS.md` is the evidence log, `docs/CAPSTONE.md` is the narrative.
>
> **Last updated:** 2026-07-07 (after R39-A + four qa/ brainstorm rounds; next = R40 Phase-0).
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
| 1 | **surprise-cost reframe / surprise-gated residual write** | changes x-axis of ALL terms | 🔬 R40 Phase-0 | gate: surprise buckets predict retention better than fact-count/age. **Build FIRST.** |
| 2 | **interference-as-signal** → collision-discovered schema (unsupervised) | patterns (discovery) | 🔬 Phase-2 | current-current collision clusters recover R34 bridge structure above derange. Legal (no old items). |
| A | **parallel-train-from-frozen-base + merge** (`merge_ties`/`merge_sum`/DARE) | write-path (resolve at merge) | 🔬 R40 smoke | removes sequential drift; interference → merge-conflict. Gate: > loramerge; frontier if ≥ nswrite+0.05. |
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

**R40 — Phase 0 + merge-smoke + capacity accounting (NEXT, cheap).** One tiny KG bakeoff:
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

1. **Does surprise predict retention?** (R40 decides.) If yes, the whole accounting unit changes.
2. **Is parallel-merge a better route or just a missing baseline?** (R40 smoke → full R-run if it clears nswrite.)
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
