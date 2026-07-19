# s0 — Findings: continual learning without forgetting, via a scalable key-retrieval

> **LM-PORT R39 — STRUCTURAL continual (arith→quad, single-token F_p): replay-free CONSOLIDATION into
> shared weights beats modular GROWTH; capacity is sufficient, growth is NOT necessary.** `s4/recur.py` +
> `s4/recur_continual.py` + `s4/recur_struct.py`, `docs/cloud_results/lm_recur*.txt` (RTX 2070, p=23,
> single-token finite-difference recurrences). Substrate pivot (after the multi-digit numseq wall): numbers
> as ONE token over F_p, rule inferable by SUBTRACTION (arith=const 1st diff; quad=const 2nd diff), so the
> step is a genuine local recurrence (like csum) and free-runs TRAINED rules to 4x the training horizon.
> Two separable facts up front: (a) **horizon extrapolation of SEEN rules is SOLVED** — arith AND quad
> free-run to 4xH at 1.00 from unseen seeds; (b) **rule generalization to HELD-OUT rules FAILS** at p=23
> and p=211 (the model learns a per-rule catalogue, not the shared law `2s_{n-1}-s_{n-2}`; unseen delta =
> unseen region of the modular-add domain = grokking/coverage wall). Structural continual arm matrix
> (arith phase A → quad phase B, prefix-only, analytic order gate = "is 2nd diff 0?"):
> ```
>                   arith retain(H/2H/4H)  quad acquire   held
> joint A|B ORACLE  1.00/1.00/1.00         1.00/1.00/1.00  0   <- fixed capacity HOLDS BOTH
> fixed A->B naive  0.00/0.00/0.00         1.00/1.00/1.00  0   <- sequential erases arith
> consolidation     1.00/1.00/1.00         1.00/1.00/1.00  0   <- replay-free shared-weight WIN
> grown+gate        1.00/1.00/1.00         0.00/0.00/0.00  0   <- exact retain (Δ=0) but adapter can't learn quad
> ```
> **Result (codex outcome-table, strongest cell): joint-oracle succeeds AND consolidation succeeds ⇒ growth
> is NOT necessary; shared-weight continual integration is the stronger mechanism.** (1) Capacity is
> sufficient (oracle holds both) so the sequential failure is interference/optimization, NOT insufficiency.
> (2) fixed-capacity naive AND fair-EWC (non-degenerate Fisher norm 0.122, `lm_recur_continual.txt`) both
> ERASE the old rule-set. (3) **replay-free consolidation** (distill frozen arith teacher on generated
> arith prefixes — pseudo-rehearsal, memory-free at DEPLOY, teacher=training-time memory) retains the arith
> EXTRAPOLATING behavior at 4xH AND acquires quad in shared weights, no raw-A replay, no task ID. (4) growth
> (frozen trunk + r=16 gated adapter) gives EXACT arith retention (analytic order gate, max|Δ|=0) but the
> adapter could NOT acquire the order-2 quad recurrence — modular quarantine bought perfect retention at the
> cost of new-task learnability. HONEST BOUNDS: consolidation is pseudo-rehearsal (not literally replay-free
> in training; teacher queried on old-function support so retention is by-construction); grown quad-failure
> is confounded by adapter width/placement (r-sweep pending) but a successful grown arm would only MATCH
> consolidation while capacity-sufficiency already refutes necessity; single seed p=23 (replication +
> cost-accounting + param-matched-wide control + grokking held-probe pending). Consistent with the project's
> standing negative on growth at synthetic scale; now a POSITIVE for shared-weight consolidation.

> **LM-PORT continual R37 — PROTECTED CONDITIONAL CAPACITY (routing, not added params) turns catastrophic
> forgetting into exact retention + full acquisition; but this is ROUTING necessity, NOT growth necessity
> (codex-corrected).** `s4/continual2.py` + `s4/width_sweep.py`, `docs/cloud_results/lm_continual2.txt` +
> `lm_width_sweep.txt` (RTX 2070 local, 1.79M-param local-window transducer; W=5). Substrate = the
> scratchpad model that length-EXTRAPOLATES `csum_reset` to L40. Phase2 acquires csum (retain target);
> phase3 learns interfering `rmax_reset` from NEW DATA ONLY (no csum replay, no joint retrain). Retention
> metric = the ALGORITHM (csum L8–40 extrapolation + reset-counterfactual), not example recall. 4 arms,
> identical phase-2 init & phase-3 data (ALL arms share the same total param count — the adapter is
> preallocated & dormant in phase2, so this is RESERVED capacity, not dynamic growth):
> ```
> phase2          : csum L8-40 = 1.00, cf 0.93 ; rmax 0        (extrapolating algorithm learned)
> naive           : csum 0.00 everywhere, cf 0 ; rmax 1.00     (catastrophic forgetting)
> ewc (lam 2000)  : csum 0.00 everywhere, cf 0 ; rmax 1.00     (== naive: Fisher DEGENERATE)
> adapter_ungated : csum 0.00 everywhere, cf 0 ; rmax 1.00     (same params, always-on -> output override)
> adapter_routed  : csum L8-40 = 1.00, cf 0.93 ; rmax 1.00     (FULL retention + FULL acquisition)
> ```
> **Headline (codex): freezing protects the old PARAMETERS; routing protects the old FUNCTION.** Decisive
> contrast = ungated vs routed: SAME adapter params, SAME frozen trunk, SAME rmax data. Always-on destroys
> csum (output-override trap: an always-on branch adds rmax-shaped residuals over the intact frozen csum
> path and overrides the head — freeze is necessary, not sufficient); command-masked routing (route=0 on
> csum) retains csum BYTE-IDENTICALLY (measured invariant: csum logits max|Δ| vs phase2 = **0.00e+00**) AND
> acquires rmax. => interference resolved by SELECTIVE ACTIVATION, not by added capacity. Retention is an
> ARCHITECTURAL invariant, not hoped-for regularization. **Width frontier** (`lm_width_sweep.txt`, phase2
> trained once & cached; per-r frozen-trunk routed adapter, invariant=0 at every r): even **r=1 (2.3K =
> 0.13% of trunk)** retains csum fully AND acquires rmax to min-L8-40 0.987; **r=2 (3.8K = 0.21%)** = full
> 1.00. r* = 1 under the preregistered min-L8-40≥0.97 criterion (2 under ≥0.99). So the isolated trainable
> capacity a new operator needs is TINY. **EWC ≡ naive** because at convergence the near-deterministic
> softmax makes the fair model-sampled diagonal Fisher numerically ~0 (`frac~0=1.00, norm=5.5e-5`) — λ inert
> (this saturated local estimator is dead; do NOT over-claim ALL curvature penalties fail — temp-softened
> Fisher / SI / function-space anchoring are untested and are different metrics, per codex).
> **CRITICAL BOUND (codex): this establishes ROUTING necessity relative to the two adapter arms, NOT GROWTH
> necessity.** The winning adapter was PREALLOCATED (params fixed from t=0), so the result is *protected
> conditional (reserved) capacity* — it never grew the param count. A fixed-capacity model that reserves a
> routed subspace in advance is the same thing. Growth NECESSITY needs a LONGER OPERATOR STREAM that
> EXHAUSTS the reserved interference-free subspace: the point where a fixed reserved budget can no longer
> add operators without interference but bounded expansion still can. Other honest bounds: command token is
> an explicit task cue read directly (protected modular expansion with an INPUT-PROVIDED route, not task-
> free route discovery); hard-frozen trunk => modular INTEGRATION, not rewriting shared weights; compute
> advantage NOT yet honest until inactive adapter matmuls are conditionally SKIPPED (report stored size vs
> active FLOPs separately). NEXT (codex-converged): (1) conditional-skip FLOP honesty + r=1 convergence
> diagnostic (4× steps, seeds); (2) multi-operator STREAM with fixed-reserved vs per-operator-growth to
> test genuine growth necessity; (3) learned INPUT-INFERRED routing on identifiability-controlled inputs
> (staged: oracle→learned-route→continual-router-update→composition).

> **R50-scale scout — the self-addressing wall survives a 3× native scale step (0.5B→1.5B).** `s3/scale_scout.py`,
> `docs/cloud_results/r50_scale_scout.*`, 2 seeds, JOINT base-hard screen (both frozen bases fail the full
> held-out question; 88/79 facts). Each size bare-acquires the SAME facts, then audits history-free L0_free+L1_
> fixed_family CORRECT proposition coverage (800 combined attempts/model/seed). **Dissociation**: 1.5B ACQUIRES
> more (answerable 38/29 vs 0.5B 31/25 — knowing scales) but history-free correct proposition coverage is
> ABSORBING-ZERO at BOTH sizes (cov_ans 0.0/0.0, unique_correct 0 of 800 each; intersection delta 0.0/0.0 on
> 22/14 shared answerable facts). Scope (codex-bounded): "through Qwen2.5-1.5B, under this 6-stream acquisition +
> fixed-budget L0/L1 generation regime, native scale improves ADDRESSED ACQUISITION descriptively but produces no
> measurable AUTONOMOUS correct proposition selection over shared answerable facts." Do NOT bank "wall is
> fundamental" — the shared THREATENED set is underpowered (n 2,1); a 3B/7B threshold or a larger-model-only
> search policy remain possible. Enough to STOP the scale escape-hatch and proceed. **NEXT (codex-converged):
> R50-A Stage A shadow-query SEARCH** (no-training discovery-only: rank generic candidates by real-write-shadow
> damage + M_prev commitment + base-lift, strict no-old-key contract, controls = passive/wrong-shadow/no-damage/
> stored-oracle, gates 25% & 4×-passive & ≤20% target-error & ≥half-lost-under-wrong-shadow); cue-ledger cost
> curve is the precommitted FALLBACK if Stage A fails; 3B dormant until a nonzero addressing policy reopens it.

> **R49a (addressability cued-recall ladder + margin probe) — the R48 coverage wall is NOT an access/encoding
> deficit: cued proposition access is paraphrase-ROBUST; the wall is AUTONOMOUS PROPOSITION SELECTION (the
> checkpoint can't pose a proposition-preserving query about its own fact from a partial cue — it drifts to
> another/higher-prior proposition about the same entity).** `s3/recall_ladder.py`, `s3/margin_probe.py`,
> `docs/cloud_results/r49a_{shakedown,confirm,margin_h}.*`, Qwen2.5-0.5B census real text.
> - **Ladder** (correct recall of a THREATENED fact vs cue strength; STREAMS=6→9, shakedown+2 seeds):
>   correct-threatened coverage is ABSORBING-ZERO for L0_free / L1_fixed_family / L2_oracle_domain /
>   L3_oracle_entity, ~0 at L4 (answer-redacted near-complete Q), and 1.0 only at L5 (full held-out question).
>   Formal branch = availability_unstable + underpowered (census is small: more streams → availability drops
>   below the 0.20 floor, threatened n < 20), but the ladder zero is not power-limited. shadow proxy is
>   EXCELLENT (per-seed Spearman 0.956/0.992, pooled 0.973, top-quartile enrichment ~3×) — a 60-step shadow
>   strongly predicts the 400-step real damage — yet `build_r49b=false` because the wall isn't search.
> - **Margin probe (hardened, 3B proposition-equivalence audit + base-model lift + swap-entity control)
>   OVERTURNED the scout**: audited proposition-EQUIVALENT paraphrases retain access (para/equivalent EM 0.833,
>   short/equivalent 0.86, gold mean-logprob ≈ 0, tiny drop) on OBSCURE facts ("What architect created the
>   Peirce–Nichols House?"→Samuel McIntire). The scout's 4.5-nat "collapse" was an ARTIFACT: self-generated
>   entity questions are almost never proposition-equivalent (gen_entity 10 changed : 1 equivalent) — they ask a
>   DIFFERENT proposition about the entity (which the model often answers correctly). swap_entity control drops
>   as designed (metric valid). So it is NOT literal-string encoding and NOT (per codex) a proven base-prior
>   decoding competition (base_lure rate 0.04) — it is an **address/proposition-SELECTION** failure. Scope is a
>   trace, not powered (20 unique equivalent-failures; self-gen evidence seed-0 only; same-3B judge not fully
>   independent — one false-positive: BBC News→London vs gold "Broadcasting House").
> - **Consequence**: passive self-replay (R48) fails because the checkpoint can't self-address (generate the
>   right query), NOT because knowledge is inaccessible. Since access is paraphrase-robust, a minimal cue-ledger
>   (tiny stubs) is a validated engineering FLOOR (O(facts), compressed rehearsal — not the strict frontier).
>   **NEXT R50-A (codex-converged): checkpoint-only SHADOW QUERY SEARCH (tomography)** — history-free prefix/
>   question search maximizing M_prev-vs-shadow disagreement to FIND a threatened address (the fact is
>   retrievable if addressed), with strict no-old-key contract + wrong-shadow control + ≥25%/≥4×-passive/≤20%-
>   target-error gates; structured (entity,relation)→query multiview is a query-POLICY alt; growth stays off.
> - **Process note**: codex peer review caught ~15 bugs across R48+R49a and OVERTURNED a scout conclusion of
>   mine (encoding→address-selection) via the demanded proposition-equivalence audit — the audit discipline paid.

> **R48 (checkpoint-only self-replay on REAL TEXT) — clean, matcher-VALIDATED NEGATIVE: self-replay is
> COVERAGE-BOUND. A dense checkpoint that HAS internalized obscure facts cannot ADDRESS them via free
> generation — "knowing ≠ being able to spontaneously recall."** `s3/selfreplay.py`, Qwen2.5-0.5B, census
> real-text streams (`docs/cloud_results/r48_census_decision.{json,log}`, 2 seeds, 5 arms, GEN_N=400,
> GEN_VIEWS=3). Contract: candidate generation sees ONLY M_{t-1} + a fixed GENERIC prompt (no old
> qids/titles/subjects); gold match is OFFLINE audit only. Two-axis verdict:
>
> | arm | old-only para EM (2-seed) | per-seed |
> |---|---|---|
> | stored_random_B (gold replay) | **0.435** | [0.441, 0.429] |
> | no_replay_bare (true floor) | 0.376 | [0.424, 0.327] |
> | self_passive_fragile_B | 0.355 | [0.322, 0.388] |
> | self_passive_random_B | 0.293 | [0.322, 0.265] |
> | no_replay_compute_matched | 0.292 | [0.339, 0.245] |
>
> `generator_mechanism=coverage_bound` (U_admit=0 of 24–59 old facts, EVERY stream, BOTH seeds),
> `retention_outcome=informative` (Delta_B=0.143 stored−no_replay), `recovery=0.007` (self-replay recovers
> **~0%** of the stored-replay gap; ≈ no_replay both seeds, no sign inversion), `self_replay=untested`
> (correctly gated: coverage failed ⇒ retention not validly tested). **Matcher-validated (enriched
> raw_nomatch dump):** the admitted dreams are ALL high-prior pretrained facts ("capital of Australia/Japan/
> Germany", "largest continent", "current US President") with top-1 question-signature overlap ~0.0–0.11 to
> the obscure census gold ("what year was the Peirce–Nichols House designated…") — genuine absence, not a
> matcher miss. **The checkpoint answers these facts when CUED (commit=full; stored gold replay protects
> them +0.06–0.14) yet NEVER produces them under free generation** — the generative distribution collapses
> onto pretrained priors. Also: `no_replay_bare` (0.376) ≫ `no_replay_compute_matched` (0.292) — an extra
> off-target qa_ce block (self-replay's or the filler's) is NET INTERFERENCE, so self-replay is actually
> *below* the do-nothing floor. **Binding constraint = generative ADDRESSING (uncued recall), NOT selection.**
> Prior arc: acquisition first failed (qa_ce(new) was only in one arm ⇒ commit≈0); once made common to all
> arms, acquisition + stored retention both work, isolating coverage as the wall. Six real bugs across two
> codex peer-review gates were caught before the decision pod. **Next (codex escalation): matched-compute
> passive-vs-ADVERSARIAL dream generation** — a disposable shadow current-write, search prompts of maximal
> M_{t-1}↔shadow disagreement, label with M_{t-1}, replay high-consensus disagreements — to address the
> *threatened* facts free generation won't surface. Growth stays OFF the path until addressing also saturates.
>
> **R47 (lifecycle bridge consolidation — the continual version of R46) — lifecycle-POSITIVE but
> SUB-THRESHOLD: the bridge primitive survives a continual setting on top of replay-based endpoint
> retention, but the effect is small and its durability is the weak point.** `s2/bridge_lifecycle.py`,
> `docs/cloud_results/r47_closure.{json,log}`, Qwen2.5-0.5B, P1=3000 (old B→C) then P2=7000 (new A→B, NO
> A→C labels), 2 seeds. Paired (both arms keep old B→C=1.0 and A→B=1.0 via replay):
>
> | arm | seed0 A→C | seed1 A→C | raw_bridge s0/s1 |
> |---|---|---|---|
> | replay_old | 0.031 | 0.033 | 0.085 / 0.090 |
> | **attractor_replay** (bridge + replay) | **0.117** | **0.117** | 0.113 / 0.206 |
>
> Paired held-out A→C gain **+0.086 / +0.084** (both seeds, no sign inversion, ≈3.7× replay); `raw_bridge`
> mechanism margin moves in the intended direction both seeds. **Verdict (codex-adjudicated):** seed-robust
> lifecycle-positive with a **sub-threshold final effect** — it MISSES the pre-registered +0.10 paired
> promotion bar by ~0.015 and the attractor A→C curve DECAYS over phase-2 before landing at 0.117, so a weak
> bridge survives but is **not strongly durable**. Not a null; do not relax the bar. Pure `attractor` (no
> replay) forgets old B→C→0 ⇒ A→C=0 (route to a destroyed endpoint) — composition NEEDS the endpoint kept
> alive, which here is supplied by replay (so this is NOT rehearsal-free retention). Remaining weakness =
> bridge durability, not endpoint retention.

> **R46 (shared-associative-bridge composition probe) — FIRST POSITIVE on the composition wall: a
> weights-only, NO-A→C-label, rehearsal-free consolidation primitive (hidden-state BRIDGE UNIFICATION)
> induces latent 2-hop composition (0.23–0.62) where independent bindings + all four controls give ~0.**
> `s2/assoc_bridge.py` (forks the R34 `composition_grok` generator), `docs/cloud_results/r46_full.{json,log}`,
> Qwen2.5-0.5B fp32, single-token synthetic entities, atomic-only training (A→B, B→C), held-out A→C NEVER
> trained, 60 bridges × 6 A/bridge, 8000 steps, 2 seeds, ATTR=identity. Successor-feature loss
> `L += λ(1 − cos(h("A's friend is"), stopgrad(h("B"))))`, DIRECT cosine (no proj head, codex), deleted-
> nothing at eval (one dense checkpoint). **Held-out A→C EM (never trained):**
>
> | arm | seed0 | seed1 | mean | hidden-sim margin |
> |---|---|---|---|---|
> | unique_bridge_atomic (R32 floor) | 0.0 | 0.0 | **0.0** | 0.016 |
> | shared_bridge_atomic (hub multiplicity only) | 0.0 | 0.0 | **0.0** | 0.06 |
> | **shared_bridge_attractor (identity)** | **0.617** | **0.228** | **~0.42** | 0.50 / 0.75 |
> | deranged_attractor (align to WRONG bridge) | 0.003 | 0.006 | **0.005** | 0.03 |
> | freq_matched_unique (same volume, NO hub) | 0.0 | 0.0 | **0.0** | 0.05 |
> | r34_direct_2hop (upper bound, trains sibling 2-hop) | 0.992 | ~0.99 | ~0.99 | 0.25 |
>
> **Clears codex's pre-registered gate on the trained surface:** shared_bridge_attractor beats
> unique/shared/freq_matched/deranged by ≫+0.20 (and ≥0.30 abs on seed0), atomic recall 1.0 everywhere,
> NO A→C target/scratchpad/task-id/graph, one dense checkpoint. **Mechanism confirmed:** the attractor loss
> drives hidden-sim margin cos(h(A friend), h(correct B)) − cos(·, wrong B) to 0.50–0.75 (vs ~0.03–0.06
> controls), i.e. it really unifies "A's friend" with B's subject-state; and the DERANGED control (aligning
> to the wrong bridge) gives ~0 with near-zero margin — the composition comes from the CORRECT bridge
> geometry, not from the auxiliary per se. **Grokking-like onset + plateau:** both seeds show a delayed jump
> (seed0 →0.67 at step 6500, seed1 →0.29 at step 5500) then plateau by 8000 (not still rising → no 15k
> extension). Bridge unification is essentially COMPLETE: cos(h(A friend), h(correct B)) = 0.9995 (vs wrong-B
> 0.25–0.50); non-monotonic across seeds (seed1 higher margin 0.75 but lower composition 0.23 vs seed0
> 0.50/0.62) — unification is necessary but the composition ceiling it buys is seed-variable.
> R32-floor cleanly reproduced (independent bindings = 0);
> **hub multiplicity ALONE is insufficient (shared_bridge_atomic = 0)** — the explicit bridge-unification
> objective is load-bearing. **Honest caveats (do NOT overclaim):** (1) **paraphrase transfer = 0.0 for the
> attractor arm — BUT the r34_direct upper bound is ALSO 0.0 on paraphrase**, so the paraphrase surface
> ("the pet of A's friend is") is uninformative/too-hard for a 0.5B even under direct composition training;
> the composition is bound to the trained 2-hop surface, not a fully abstract traversal. (2) **Magnitude is
> seed-variable** (0.62 vs 0.23). (3) single-token SYNTHETIC entities, not real text; not yet a
> continual/lifecycle test (old-hop retention untested); not shown to beat the R43 `random@0.5` retention
> baseline (this is a COMPOSITION-axis result, a different axis than fact-retention). **Verdict:** the FIRST
> mechanism in the whole arc (R32/R34/R40/R42/R43/R44) to move our only zero-signal wall (latent
> composition) off exactly 0 — a real rehearsal-free, weights-only composition primitive — but "composition
> solved" it is NOT. Next: mechanism probe (does it route the actual 2-hop prefix through B?), real-text /
> multi-token bridges, and a lifecycle version (does bridge unification survive later writes + preserve old
> hops). Connects the 3-round brainstorm: transformers-as-associative-memory + composition=graph-traversal;
> the bridge is made a shared attractor by an explicit training-time objective, distilled into weights.
> **MECHANISM CONFIRMED (re-run `docs/cloud_results/r46_mech.json`, codex's key probe):** the raw 2-hop
> bridge-POSITION margin cos(h("A's friend's"), h(B)) − cos(·, wrong-B) is HIGH for the attractor arm
> (0.208 s0 / 0.103 s1) and ~0 for all controls (deranged 0.04/−0.04, shared-atomic 0.01/0.04); and
> aux ≈ raw_bridge (0.209≈0.208) — the alignment trained at "A's friend is" TRANSFERS to the actual 2-hop
> bridge position "A's friend's". So the held-out A→C gain is the model genuinely ROUTING the composed
> prompt through B's identity state, NOT a shallow readout artifact of the aux position. (Re-run held2:
> attractor 0.20/0.169 vs controls ~0 — reproduces R46 with seed-variable magnitude.) → the "bridge-node
> traversal primitive" wording is earned, not downgraded.

> **R44 (schema-compressibility, prime-then-bind on real T-REx relations) — NULL for RELATION-specific
> schema compression (the deranged same-kind control matches schema_commit), but a REAL coarser effect:
> reusable structure exists at the ANSWER-KIND granularity, and consolidating incompatible-kind facts
> together INTERFERES.** `s3/schema_comp.py` (`SC_STAGE=train`), `docs/cloud_results/r44_ladder.{json,log}`,
> Qwen2.5-0.5B, real T-REx triples (`relbert/t_rex_relation_similarity`), 6 base-hard relations (person:
> P50/P57/P86/P58; org: P178/P176), 2 seeds, T=8 held-out targets/relation, MANUAL held-out paraphrase EM.
> Protocol: PRIME N same/other/same-kind facts into weights, BIND T held-out same-relation targets at
> cumulative budget B, measure target held-out-paraphrase EM. Mean target EM at bind-budget B=6 (mid-curve):
>
> | bucket | floor N=0 | schema_commit N=16 | item_k_matched N=16 | shuffle(deranged) N=16 |
> |---|---|---|---|---|
> | person | 0.438 | 0.891 | 0.828 | **0.906** |
> | org | 0.719 | 0.969 | **0.562** | 0.875 |
>
> **Fails codex's pre-registered gate** (schema_commit steeper than item_k_matched in BOTH buckets AND
> deranged shuffle NOT passing): (1) **person: schema_commit ≈ item_k_matched ≈ shuffle** — no
> relation-specific advantage (schema_commit − item_k_matched flips sign across N, mean ≈0; shuffle even
> highest). (2) **org: schema_commit > item_k_matched (Δ≈+0.31–0.41 at N=16) BUT schema_commit ≈ shuffle**
> — the deranged same-kind control matches schema_commit, so by codex's criterion the effect is
> answer-KIND/template support, NOT relation schema. **Findings:** (a) **Priming helps binding** (person
> floor 0.44 → primed ~0.88 at B=6): consolidating facts first roughly doubles low-budget binding EM. (b)
> **The reusable structure is answer-KIND, not relation-specific:** same-kind-different-relation priming
> (shuffle) works as well as same-relation (schema_commit). (c) **Cross-kind consolidation INTERFERES:**
> org `item_k_matched` (0.562) falls BELOW floor (0.719) — mixing person-name + org-name + other-kind facts
> in one consolidation hurts org-target binding. This is a clean mechanistic datum: the write path shares
> an answer-TYPE schema, and mixing incompatible types causes destructive interference (connects to the
> R43 interference story). (d) Prime retention post-bind 0.77–0.97 (no catastrophic prime forgetting).
> **Caveats:** org n=4/cell (2 rel × 2 seeds) — noisy; person n=8 is the cleaner (and clearly null) bucket.
> **Verdict:** R44's specific claim (relation-schema compression → sublinear marginal cost) is NOT
> supported; the compression that exists is coarse output-type priming. Per the reframe, this moves us to
> "report the honest negative map" — with the new positive nuance that answer-KIND schema is reusable and
> cross-kind mixing interferes. Consistent with R32/R34 (fine composition/schema does not emerge at 0.5B).

> **R43-ladder (Step B — real-density budget ladder on self-contained real text) — HONEST NEGATIVE:
> surprise-gated replay does NOT beat random subsampling on real text. The R41 Rung-1 win was an
> artifact of the constructed bimodal 50/50 corpus; on real text (unimodal surprise, R42/R43) there is
> no categorical "safe-to-skip" population, so item-level routing gives no advantage — although the
> surprise SIGNAL is genuine (high-bpt old items forget more, corr≈−0.3).** `s3/wikibridge.py`
> `WB_SOURCE=census` (length-stratified `select_budget`), `docs/cloud_results/r43_ladder.{json,log}`,
> Qwen2.5-0.5B, 6 streams × [4 art × 3 QA], **2 seeds**, old committed pool = 60 (commit ≈100%),
> budgets = fraction of old committed replayed at final consolidation, length-stratified selection.
> **OLD-only held-out paraphrase EM (all old items, replayed+skipped), per seed:**
>
> | arm | seed0 | seed1 | mean | vs full |
> |---|---|---|---|---|
> | compact_cpt_qa_k0 (no replay, floor) | 0.217 | 0.267 | 0.242 | — |
> | compact_cpt_qa (full replay, ceiling) | 0.483 | 0.483 | **0.483** | — |
> | bgt_surprise@0.125 | 0.284 | 0.433 | 0.359 | 74% |
> | bgt_random@0.125 | 0.300 | 0.367 | 0.334 | 69% |
> | **bgt_surprise@0.25** | 0.333 | 0.533 | 0.433 | 90% |
> | **bgt_random@0.25** | 0.467 | 0.450 | **0.459** | 95% |
> | bgt_lowbits@0.25 | 0.350 | 0.350 | 0.350 | 72% |
> | bgt_surprise@0.5 | 0.400 | 0.550 | 0.475 | 98% |
> | bgt_random@0.5 | 0.450 | 0.567 | **0.509** | 105% |
>
> **Fails codex's pre-registered gate** (surprise > random on BOTH seeds): (1) **sign INVERTS across seeds**
> at B=0.125 and B=0.25 (surprise loses seed0, wins seed1); (2) **on the mean, random ≥ surprise at B≥0.25**
> (0.459 vs 0.433 at 0.25; 0.509 vs 0.475 at 0.5). NOT a surprise win; a null leaning random, high seed
> variance. (2) **But the surprise signal IS real:** `surprise_summary` corr(base-bits, retention) = −0.29
> (k0) … −0.43 (@0.5); k0 retention by bits tercile = **0.45 / 0.175 / 0.10** (low→high bpt) — high-bpt old
> items forget ~4× more. (3) **Mechanism of the non-transfer:** on real text the surprise distribution is
> UNIMODAL (R42/R43), so there is no "easy, safe-to-skip" class: surprise's SKIPPED low-bpt items still
> decay to 0.31–0.49 (vs R41's constructed corpus where skipped items held at 0.70), and its REPLAYED
> high-bpt items are intrinsically hard to rescue. Reallocating budget toward high-bpt (surprise) vs random
> therefore nets ~zero. (4) **Replay itself works & subsampling gives a router-FREE compute win:** k0 0.242
> → full 0.483; **random@0.5 (0.509) ≈ full (0.483)** — half the replay budget, randomly chosen, matches
> full. The O(#facts)→O(#exceptions) *routing* claim does not hold on real text; a plain O(½·#facts) random
> subsample does. **Takeaway: the census program did its job — R41's optimistic Rung-1 result did NOT
> transfer to real text, and the reason is exactly the unimodality R42/R43 measured.** Caveats: 2 seeds,
> n_old=60, large variance — the NULL is "no reliable routing advantage", not "random provably better".
> Next: more seeds to tighten, OR accept null and test whether real exceptions are schema-COMPRESSIBLE
> (O(#patterns) not O(#exceptions)) + the dC/dbook 2-book cross-book probe.

> **R43 (census-XL — 4× scale + self-containedness screen) — the CLEAN Step-B planning number: after
> removing context-deictic probes, the real-text exception tail is EVEN SMALLER (τ=8 ≈ 16–25%, τ=10 ≈
> 6–8%), confirming R42 and confirming deictic probes were inflating the apparent tail.** `s3/census.py`
> (self-contained judge, generator names-the-subject; two prompt-bug fixes: 3B was answering the trivia
> instead of judging, and generator was making passage-scoped Qs — both caught by pod-side judge unit
> test 9/10), `docs/cloud_results/r43_censusxl.{json,jsonl,log}`, 120 passages/domain × 8 probes, 3090.
>
> | domain | n_gated | n_sc | sc_rate | faith | **SC tail τ=8 [CI95]** | SC tail τ=10 [CI95] |
> |---|---|---|---|---|---|---|
> | squad_human | 554 | 131 | 0.233 | 0.906 | 0.198 [0.12,0.28] | 0.076 [0.04,0.12] |
> | wiki_gen | 487 | 223 | 0.464 | 0.755 | 0.157 [0.11,0.21] | 0.058 [0.03,0.09] |
> | news_gen | 531 | 106 | 0.199 | 0.813 | 0.245 [0.16,0.34] | 0.075 [0.03,0.12] |
>
> **Findings:** (1) **Self-containedness screen LOWERS the tail** (gated→SC: squad τ=8 0.256→0.198, news
> 0.284→0.245) — the deictic probes R42 flagged really were fake high-surprise exceptions; the honest
> globally-addressable exception tail is τ=8 ≈16–25%, τ=10 ≈6–8%. (2) **Density reproduced R42 at 4×
> scale** before screening (wiki τ=10 0.078 vs R42 0.076; news 0.113 vs 0.140). (3) **Self-contained yield
> is domain-structured:** wiki_gen 46% (richest, generator can name wiki entities), news 20% / squad-human
> 23% (news is deictic-heavy; human SQuAD Qs are often passage-scoped). 460 SC-gated probes total; 67
> passages with ≥3 SC probes (29 with ≥4) → enough for a 7-stream ladder. (4) paraphrase stability corr
> 0.981 (bpt measures the item). **Rung-2 density trigger still NOT met** (SC τ=8 CI-upper 0.21–0.34 < 0.5).
> → Step-B ladder on `WB_SOURCE=census` (natural density, self-contained only). NOTE the two prompt bugs are
> a reusable lesson: an LLM judge's user-turn must ASK the judgment question, or it just answers the content.

> **R42 (Step A — real-text exception-density CENSUS, inference-only) — the real tail is a MINORITY
> (≈7–14% at the synth-anchor threshold, ≈21–29% at τ=8), the distribution is a smooth UNIMODAL continuum
> (no natural binary exception class), and bits/token is answer-TYPE- and LENGTH-structured.**
> `s3/census.py` (codex-reviewed GO `qa/codex/2026-07-08.21.48.44.md`), artifacts
> `docs/cloud_results/r42_census.{json,jsonl,log}`, Qwen2.5-0.5B base bits/token (same unit as R41 `_bpt`),
> 3B-generated atomic probes + human-QA anchor, ~15min on one 3090 (two relaunches: HF hub namespace id;
> 3B copies a literal `<TAB>` from format prompts — fixed with pipe format + one-shot example).
>
> | domain (gated probes) | n | faith | mean bpt | tail τ=6 | tail τ=8 [CI95] | tail τ=10 [CI95] |
> |---|---|---|---|---|---|---|
> | squad_human (wiki, HUMAN QAs) | 177 | 0.865* | 5.82 | 0.429 | 0.209 [0.14,0.28] | 0.073 [0.03,0.12] |
> | wiki_gen (wiki, 3B probes) | 119 | 0.685 | 5.77 | 0.420 | 0.244 [0.16,0.33] | 0.076 [0.02,0.14] |
> | news_gen (CNN, 3B probes) | 150 | 0.844 | 6.61 | 0.547 | 0.293 [0.21,0.37] | 0.140 [0.08,0.20] |
>
> *squad_human faith = 3B judge CALIBRATION on known-good human probes → 0.865 clears codex's ≥0.85
> interpret gate (short of the 0.90 "clean" bar). Quality audit: dup rate 0.6–2.2% (tail not dup-driven);
> paraphrase stability corr **0.986** / mean|Δ|=0.31 bpt (bits/token measures the ITEM, not the wording);
> base already-correct ≈0% → need_write ≈ everything; R40 anchors: on-manifold ≈4.45, off-manifold ≈11.37.
> **Findings:** (1) **Real-text exception density at the R40 off-manifold anchor (τ=10) is only 7–14%**
> — the constructed 50/50 of R41 overstates real tails by ~4–7×; surprise-routed compact replay at
> B≈10–25% is a genuine O(#exceptions) compute advantage on real text. **Rung 2 density trigger NOT met**
> (CI-upper 0.20–0.37 at τ=8, trigger was ≈0.5). (2) **Deciles are smooth (2.5→10, no gap)** — real text
> has NO bimodal on/off-manifold split; τ must be chosen by budget/risk tradeoff, and Step B's grid should
> be tail-aware ({f/2, f, 2f, 0.5} with f≈0.2–0.3). (3) **The tail is TYPE-structured:** number/date
> answers are ~never in the τ=8 tail (0–2%) while proper names hit 12–43% and open-class phrases 38–50% —
> novel NAMES are the real-text analogue of R40's invented-name synth facts. (4) **Confound (flagged):**
> 1–2-token answers have much fatter bpt tails (0.43–0.68) than 5+ (0.05–0.19) — later answer tokens
> condition on earlier ones, amortizing per-token surprise; Step B selection should stratify or
> length-adjust. (5) **Caveat (spot-check):** part of the news tail is context-DEICTIC probes ("What is
> the name of the dog?" → Sam) — not globally addressable facts; a self-containedness screen is needed
> before treating the news tail as pure knowledge novelty. Two-hop canary: 33 faithful, distribution ≈
> one-hop (mean 5.71) — composition probes are not surprise-visible at the span level (consistent with
> R32/R34: composition is a different axis than storage cost).

> **R41 (Rung 1 — surprise-gated replay BUDGET on a mixed squad+synth stream) — STRONG PASS: the model's own
> frozen-base bits/token is an ACTIONABLE routing signal, not just a category marker.** `s3/wikibridge.py`
> `build_mixed` + `select_budget`, logs `docs/cloud_results/r41_rung1.{json,perqa.json,log}`, Qwen2.5-0.5B,
> mixed stream (3 streams × [3 squad + 3 synth] articles), 1 seed, budget = 50% of old committed items replayed,
> selected by rule. Old-only (60 items) final paraphrase EM, split by frozen-base bits/token (median 9.12;
> high = off-manifold exceptions, low = on-manifold):
>
> | arm (replays 50% of old) | all-old | **HIGH-bits old (exceptions)** | low-bits old (on-manifold) |
> |---|---|---|---|
> | **bgt_surprise** (replay top-50% by bits/token) | **0.617** | **0.533** | 0.700 |
> | bgt_random (replay random 50%) | 0.500 | 0.267 | 0.733 |
> | bgt_lowbits (replay bottom-50%) | 0.517 | 0.267 | 0.767 |
> | bgt_sourceoracle (replay synth-first) | 0.583 | 0.500 | 0.667 |
> | compact_cpt_qa (full replay, ceiling) | 0.617 | 0.500 | 0.733 |
> | compact_cpt_qa_k0 (zero replay, floor) | 0.417 | 0.233 | 0.600 |
>
> **Clears every one of codex's pre-registered gates:** (1) **surprise − random = +0.117 all-old AND +0.266
> HIGH-bits** (bars were +0.10 / +0.15). (2) **lowbits (0.267 high-bits) ≪ surprise** — allocation DIRECTION
> matters; lowbits ties random by abandoning the exceptions. (3) **sourceoracle competitive (0.500 vs 0.533
> high-bits) — surprise even beats the source oracle**, proving frozen-base bits/token is a SUFFICIENT routing
> signal (no squad/synth label needed; the model's own surprise finds the exceptions). (4) low-bits skipped
> items stay safe (surprise 0.700, above k0 0.600). (5) newest not underlearned (surprise final-para 0.589 ≥
> full 0.578). **Headline: surprise-gated replay of the top-50%-by-bits/token MATCHES full replay on all-old
> (0.617 = 0.617) and EXCEEDS it on the exceptions (0.533 > 0.500) at HALF the budget** — it concentrates the
> scarce budget on off-manifold exceptions and correctly skips on-manifold items that survive unreplayed
> (R40-s3 mechanism, now a control knob). **This turns the surprise result from a category marker into a
> deployable allocation policy: spend O(#exceptions) committed-target budget, routed by the model's own
> bits/token, instead of O(#facts).** Exception-tail density signal: at B=0.5 surprise already ties full → the
> tail here is ≤50% (corpus is 50% synth by construction). **Caveats:** 1 seed; exact density needs a budget
> ladder (0.25/0.5/0.75, codex's next step); "replayed" = final-consolidation pool (ever_replayed audit field
> logged). **Ladder: Rung 1 mechanism VALIDATED.** Whether Rung 2 (fixed-capacity memory layer) is needed now
> depends only on the measured exception-tail density for real corpora: sparse tail → surprise-gated compact
> replay suffices with NO new architecture; dense tail → Rung 2 justified. Next: SEEDS=2 sign-stability, then
> budget ladder → the density number.

> **R40-s3 (surprise gate on the RIGHT venue — real `squad` vs independent `synth` through one ingest path)
> — the surprise/manifold cost model PASSES at the CATEGORICAL level, is NULL as a within-source continuous
> bits law.** `s3/wikibridge.py` surprise instrumentation (per-QA frozen-base answer-seq bits + per-arm final
> retention keyed by qid), logs `docs/cloud_results/r40s3_{squad,synth}.{json,perqa.json,log}`, Qwen2.5-0.5B,
> 1 seed (squad 2 streams / synth 4 streams — smoke). **Old-only NON-replayed paraphrase EM (the zero-replay
> floor), matched arms:**
>
> | arm | squad (real, on-manifold) | synth (invented, off-manifold) | gap |
> |---|---|---|---|
> | compact_cpt_qa_k0 (no replay, +anchor) | **0.76** | **0.08** | **+0.68** |
> | compact_cpt_qa_k0_noanchor (no replay, no anchor) | **0.60** | **0.187** | **+0.41** |
> | compact_cpt_qa_k1 (1 residual/article) | 0.65 | 0.10 | +0.55 |
> | compact_cpt_qa (full replay, ceiling) | 0.88 (rep) | 0.52 (rep) | +0.36 |
>
> **(1) Categorical source = on/off-manifold DECISIVELY predicts zero-replay retention** (+0.68 at k0, far
> above codex's +0.25 scale bar) — R40's KG null was a **venue artifact** (KG counterfactuals were uniformly
> high-surprise; no low-surprise population). On the correct real-vs-synth venue the R38B-A gap reproduces
> cleanly with the surprise instrumentation. **And the surprise contrast is real and survives answer-length
> control:** mean **bits/token = 4.45 (squad) vs 11.37 (synth)** — the off-manifold source is ~2.5× higher
> per-token surprise AND the one that collapses. Total bits are answer-length-confounded (squad 4.9 tok /
> 19.3 bits vs synth 3.9 tok / 44.5 bits); bits/token is the clean measure and cleanly separates the sources.
>
> **(2) Continuous bits are NOT a within-source retention law (NULL/weak).** Within squad k0, base-para
> surprise vs retention corr = −0.05 (total) / −0.26 (bits/token); within synth k0 −0.10 / −0.13; k0_noanchor
> near zero. Weakly in the expected direction (higher surprise → lower retention) but within noise at n=25–75.
> **So the effect is CATEGORICAL (manifold membership), not a smooth per-item bits gradient** — codex's
> pre-registered "partial pass": the usable planning variable is an **on/off-manifold residual BUCKET**, not
> "cost = k × bits".
>
> **Net (surprise reframe, verdict):** the surprise-cost model is REAL and measurable, but it reclassifies
> the cost as **O(#off-manifold exceptions)**, NOT a continuous surprise law and NOT a dissolution of the
> exception wall. On-manifold (low bits/token) knowledge is retained ~0.6–0.76 with ZERO replay; genuinely
> novel off-manifold knowledge still collapses to ~0.08–0.19 unreplayed. **Consequence for "read hundreds of
> books": cheap/no-replay for content that builds on existing knowledge; the irreducible cost is exactly the
> off-manifold novelty**, which still needs residual/replay budget. This directly founds Phase-1 (surprise-
> gated residual write: spend replay/residual budget only on high-bits/token off-manifold items). **Caveats:**
> 1 seed; squad 2 streams (old=stream 0 only) vs synth 4 streams (different counts — codex guardrail); within-
> source underpowered. Worth scaling to SEEDS=2, but the categorical floor gap is large and robust to the
> anchor/no-anchor split. Instrumentation (`qa_answer_bits`, `.perqa.json`) reusable.
>
> **MATCHED-STREAM CONFIRM (R40-s3m, both sources STREAMS=2, 2 seeds — removes the age confound): PASSES.**
> `docs/cloud_results/r40s3m_squad.*` + `r40s3m2_synth.*` (synth re-run with `WB_SYNTH_OVERSEL=12` after the
> first synth under-yielded 1 stream). Matched old-only non-replayed para EM: **k0 squad 0.68 vs synth 0.24 =
> +0.44**; **k0_noanchor squad 0.68 vs synth 0.22 = +0.46**; no seed inversion (squad 0.64/0.72 both > synth
> 0.20/0.28). Clears codex's pre-registered bars (k0 ≥+0.30, k0_noanchor ≥+0.20). **The age confound was real
> but partial:** matched synth k0 rose 0.08→0.24 (the original 4-stream synth aged more), so ~0.2 of the
> unmatched +0.68 was age; the categorical on/off-manifold effect survives cleanly at +0.44/+0.46. Full replay
> narrows but does not close (synth old-replayed 0.70 vs squad 0.82), confirming the deficit is
> retention/write budget for off-manifold exceptions, not total unlearnability. **The surprise/manifold
> categorical result is now a clean matched Phase-0 pass.**

> **R40 (Phase-0 smoke — surprise instrumentation + parallel-train-from-frozen-base MERGE arms) —
> MERGE is a CLEAN NEGATIVE; surprise-as-continuous-predictor is NULL in this venue (wrong venue).**
> `s2/lifecycle_bakeoff.py` `run_mergeparallel` + `surprise_probe`/`surprise_summary`, logs
> `docs/cloud_results/r40_smoke.{json,perfact.json,log}`, Qwen2.5-0.5B KG counterfactuals, 3 streams × 20,
> 1 seed, LD_STEPS=600 (cheap decision smoke). old_para_final (streams 0..R-2):
>
> | arm | old_para | newest | base-hop after | retained/surprise-bit | merge-conflict |
> |---|---|---|---|---|---|
> | naive_fixed | 0.625 | 0.85 | 0.191 | 0.072 | — |
> | **nswrite** | **0.975** | 0.85 | 0.215 | 0.096 | — |
> | ours_tgt_answerid | 1.000 | 0.85 | 0.205 | 0.098 | — |
> | loramerge (sequential fold) | 0.500 | 0.85 | 0.135 | 0.063 | — |
> | **merge_sum** (parallel + task-vector sum) | **0.250** | 0.25 | 0.101 | 0.022 | 0.245 |
> | **merge_ties** (parallel + TIES) | **0.225** | 0.20 | 0.146 | 0.021 | 0.244 |
>
> **(1) Parallel-train-from-frozen-base + MERGE — DECISIVE NEGATIVE, retired.** Both merge arms land
> **below even the sequential `loramerge` baseline** (0.225/0.25 < 0.50), far below `nswrite` (0.975); they
> **collapse newest-stream learning** (0.2–0.25 vs 0.85 everywhere else) and base-hop (0.10–0.15). The
> hypothesis that parallel training "removes sequential drift" is FALSE for independent facts: merging
> independent per-stream LoRA task vectors on the shared q/v subspace **interferes destructively** (sign-
> conflict 0.244 by round 3) — worse than sequential fold. This is the R38B-A / O(#facts) wall in weight-
> space: independent facts occupy *conflicting* write directions, so their task vectors collide at merge.
> (Task arithmetic works for *related/structured* tasks whose vectors align; not for independent tuples.)
> codex predicted exactly this. Merge fails both its baseline gate (> loramerge) and frontier gate
> (≥ nswrite+0.05). **Retire the merge paradigm for independent-fact CL.**
>
> **(2) Surprise gate — NULL here, but this is the WRONG venue.** Base-para surprise does NOT predict
> retention in the expected direction: for the most-informative arm (naive_fixed, the one with forgetting
> variance) the correlation is weakly POSITIVE (+0.15, excl-already-correct +0.21; tercile lo/mid/hi
> retention 0.65/0.60/0.85) — within noise at n=58. **Reason: KG counterfactuals are uniformly high-
> surprise** (mean 9.72 bits, only 2/60 already-base-correct), so there is NO low-surprise (prior-anchored)
> population to expose the R38B-A floor. The surprise hypothesis is inherently a *real-text-vs-synthetic*
> (on-manifold vs off-manifold) contrast, which this venue cannot create. **Conclusion: the surprise gate
> must be re-run on `s3/wikibridge.py` real-vs-synth, not KG counterfactuals.** The R38B-A phenomenon
> (real 0.63–0.74 vs synth 0.16–0.18) stands; R40 only shows base-surprise is not a smooth *within-
> counterfactual* retention predictor.
>
> **(3) nswrite re-confirmed** as the best rehearsal-free writer (0.975 ≈ replay 1.0 at this small scale),
> and most surprise-bit-efficient (0.096). Consistent with R39-A. **Net R40:** cheaply killed the merge
> paradigm for independent facts; relocated the surprise test to s3; nswrite remains the frontier writer.
> Instrumentation (`surprise_probe`, `perfact` sidecar, merge-conflict diag) now reusable for the s3 surprise run.

> **R39-A (key-tied schema anchoring — first rehearsal-free "manufacture prior-anchoring for NEW facts"
> probe) — CLEAN NEGATIVE for the anchoring mechanism; RE-CONFIRMS `nswrite` as the strong rehearsal-free
> writer.** `s2/lifecycle_bakeoff.py` `run_keytie`, logs `docs/cloud_results/r39a_keytie.{json,log}`,
> Qwen2.5-0.5B KG, 6 streams × 40 facts, 2 seeds, fixed-size (no growth). Hypothesis (from R38B-A): pin each
> NEW fact's retrieval key-stem pooled rep to its **frozen-base** rep (cosine, λ=1), so the model writes only
> minimal association bits on a stable pretrained key → inherit the real-text zero-replay retention floor.
> Rehearsal-free by the strict line (anchor = the fact's OWN kstem + frozen base; no old prompts/answers/
> logits/targets in later writes; single dense checkpoint; no inference key bank). Headline = **old-only
> non-replayed paraphrase EM** (streams 0..R-2). 2-seed:
>
> | arm | old_para | old_seen | newest | base-hop after | rehearsal-free? |
> |---|---|---|---|---|---|
> | naive_fixed (no protection) | 0.302 | 0.332 | 0.95 | 0.210 | yes |
> | **keytie_base** (tie key → frozen base) | **0.360** | 0.365 | 0.95 | 0.198 | yes |
> | keytie_random (shuffled anchor, matched compute) | 0.297 | 0.320 | 0.95 | 0.201 | yes |
> | **nswrite** (interference-aware null-space write) | **0.742** | 0.843 | 0.95 | 0.214 | yes (training-state, no old-item info) |
> | ours_tgt_answerid (compact committed replay) | 0.838 | 0.925 | 0.95 | 0.200 | no (O(#facts) ledger) |
>
> **Verdict against codex's pre-registered gates: FAIL.** (1) `keytie_base − keytie_random = +0.062`,
> `keytie_base − naive = +0.057`: there IS a **real, sign-consistent target-specific anchoring signal**
> (both seeds: seed0 +0.095, seed1 +0.03) — tying a fact to its OWN frozen-base key beats tying to a random
> key — but it is **trivially small**. (2) **PRIMARY gate `keytie_base > nswrite`: −0.382. Massive fail.**
> Anchoring the retrieval KEY barely moves retention; it does not approach the best rehearsal-free baseline.
> keytie does NOT sacrifice fresh learning (newest 0.95 = naive) and does not drift base (hop 0.198), so the
> failure is "no benefit," not "benefit bought by damage." **Interpretation (compression frame):
> interference is codebook-*contention* — which write DIRECTIONS/subspace you overwrite — not where the
> retrieval key points. keytie stabilized the key's direction; `nswrite` protects the whole write from
> colliding with occupied subspace. Only the latter matters.** **The louder positive: `nswrite` (0.742 old-
> para, 0.843 old-seen) recovers ~89% of committed-replay's old-para (0.742/0.838) with ZERO replay and zero
> committed targets** — the standing-best rehearsal-free result (R36-I), here directly bracketed against
> replay in one run. Its write-subspace occupancy saturates 0.74→0.91 over 6 rounds — the R36-I saturation
> that is the one principled regime where function-preserving *growth* may finally be necessary (phase-2:
> depth vs width on saturation). **R39-A conclusion:** manufacturing prior-anchoring by KEY-representation
> tying is not the lever; the rehearsal-free frontier stays with subspace-protected writing + its saturation.
> Next (codex phase-2 brainstorm, `qa/`): schema-extraction consolidation (`BK_DATA=schema_comp`, reuse R34
> bridge surface where held-out 2-hop groks to 0.98) testing whether O(#facts)→O(#patterns) — replay K
> support instances per pattern ONCE, compile the rule, stop paying per instance (footprint SLOPE dtargets/dA
> as the headline, not a point estimate).

> **R38-WikiBridge-A (REAL-TEXT bridge — read Wikipedia passages → durable closed-book QA in weights) —
> POSITIVE** `s3/wikibridge.py`, logs `docs/cloud_results/r38_squad_para.{log,json,manifest}`, Qwen2.5-0.5B,
> 3 streams × 5 SQuAD articles × 5 QA (75 QA, base-hard + RAG-answerable screened, **held-out paraphrase
> eval**). First step from synthetic tuples to real passage text. Per stream: transient continued-PT
> scaffold on new passages (+QA span-CE for the qa arm) → consolidate into one dense checkpoint by replaying
> OLD **committed answer-sequence CE** targets + neutral base anchors (base-KL never on old QA — R37 lesson);
> discard scaffold; **closed-book** eval, no passage/retrieval/task-id. The decisive metric is EM on
> **held-out reworded questions** (paraphrase = internalization, not prompt memorization).
>
> | arm | orig-EM | **para-EM (internalization)** |
> |---|---|---|
> | base_no_ingest | 0.000 | 0.000 |
> | rag_gold_passage (upper bound, uses inference memory) | 0.973 | 0.973 |
> | naive_cpt (sequential continued-PT, no replay) | 0.000 | 0.000 |
> | compact_cpt_only (LM-only scaffold + compact replay) | 0.013 | 0.000 |
> | **compact_cpt_qa (QA-target scaffold + compact replay)** | **1.000** | **0.893** |
>
> **compact_cpt_qa recovers 92% of RAG's answer quality (0.893/0.973) FROM WEIGHTS ALONE on held-out
> paraphrased questions** — genuine internalization (memorization gap orig 1.0 → para 0.893 is only ~0.11),
> retained across streams (old-para 0.88–0.90), single dense checkpoint, no inference memory, no joint
> retrain. **Two decisive findings:** (1) **reading is NOT enough** — raw next-token continued-PT
> (`naive_cpt`, `compact_cpt_only`) yields ~0 closed-book QA; the load-bearing step is **S1 teacher
> construction**: converting passage text into *source-grounded QA/answer-function targets* (self-quiz).
> (2) **compact committed answer-sequence targets (R36-A2 generalized to multi-token) retain old QA** while
> new streams are written. Bugs found + fixed en route (both would silently sink it): **fp16 AdamW training
> NaNs → use bfloat16**; answer needs a `\n` stop token for clean greedy decode. **Honest caveats:** 0.5B,
> 15 articles / 3 streams / 1 seed (para-screen is strict); extractive short answers only; not
> cross-document reasoning; RAG (0.973) still exceeds weights-only (0.893). But the **real-text bridge
> works**: source text becomes internalized, retained, memory-free in-weight knowledge — the first credible
> step toward "read and internalize," with the honest requirement that ingestion must build answer-function
> targets, not just read. Next: scale articles/streams/seeds; footprint sweep (R36-A-style) on old replay;
> counterfactual audit (`WB_SOURCE=cf`); multi-view teacher if paraphrase transfer needs strengthening.
>
> **R38 FOOTPRINT SWEEP — the "real-text redundancy softens O(#facts)" claim is RETRACTED after
> attribution hardening (R38B).** The original 1-seed pilot (`r38_footprint.json`) reported non-replayed
> old-paraphrase rising to 0.733 at K=3 (vs 0.507 at K=0) and read this as replay-coverage spillover. That
> signal was a **fresh-stream accounting artifact**: the replayed/non-replayed split scored *all* committed
> items, including the final stream's items that had never been exposed to a later forgetting event. R38B
> (`docs/cloud_results/r38b.{json,log}`, `WB_..._k<K>` + `_noanchor`, 2 seeds, **old-only split** = items
> committed strictly before the final stream) corrects this. Old-only non-replayed old-paraphrase EM:
>
> | arm (K committed-QA/article replayed) | seed0 | seed1 | **2-seed avg** | replayed-item para |
> |---|---|---|---|---|
> | naive_cpt (full CPT, no replay) | 0.000 | 0.000 | **0.000** | — |
> | k0 (compact, zero old replay, +anchor) | 0.575 | 0.675 | **0.625** | — |
> | k1 | 0.500 | 0.781 | **0.641** | 0.875 |
> | k3 | 0.562 | 0.812 | **0.687** | 0.729 |
> | **k0_noanchor** (zero old replay, no anchor) | 0.675 | 0.800 | **0.738** | — |
> | compact_cpt_qa (replay ALL committed) | — | — | final-para 0.80 | 0.787 |
>
> **Verdict against codex's pre-registered pass bar (K3 non-replayed ≥ K0 + 0.15; holds ≥2 seeds; not
> order-dependent): FAIL.** (1) 2-seed avg K3−K0 = +0.062, far under the +0.15 threshold. (2) The two seeds
> **disagree on sign**: seed0 replay is flat/slightly harmful (k1 0.500 < k0 0.575), seed1 mildly helpful
> (k3 0.812 > k0 0.675) — so any K-benefit is inside seed/ordering noise. (3) The **zero-replay, no-anchor
> arm (0.738) beats every replay arm in both seeds** — the best non-replayed retention comes from *not*
> replaying old items at all. **So replay does NOT buy non-replayed neighbor retention; it protects the
> replayed items themselves** (k1 replayed 0.875 vs non-replayed 0.641) — exactly the item-rehearsal picture
> of synthetic R36-A. The independent-fact O(#facts) coverage bound **stands** on real text: to reliably
> retain a specific fact you must replay it. "Read hundreds of books cheaply via redundancy" is **not**
> supported at this scale.
>
> **Two honest positives survive the hardening:** (a) **the compact-QA consolidation is genuinely gentle** —
> with *zero* old replay it still holds ~0.63–0.74 non-replayed old paraphrase (vs naive_cpt 0.0), i.e.
> sequential QA-target consolidation from M_{t-1} is far less destructive than raw continued-PT (this is a
> consolidation-objective effect, **not** a replay-coverage effect, and its origin — real-text shared
> structure vs objective gentleness — is *not* resolved here). (b) **the neutral base-capability KL anchor
> mildly HURTS old-fact retention** — k0_noanchor > k0 on non-replayed old para in both seeds (+0.10, +0.125)
> and on final para (0.717/0.767 vs 0.617/0.617); the anchor pulls M back toward the base prior and erodes
> learned facts. A cleaner anchor (on truly neutral tokens only, or dropped) is the better consolidation
> recipe. **Caveats:** 0.5B, 12 articles / 3 streams (para-screen caps survivors), 2 seeds. The retraction is
> the science: the hardened accounting overturned a single-seed positive.
>
> **R38B-A SAME-OBJECTIVE SYNTHETIC CONTROL — the zero-replay retention floor is a REAL-TEXT effect, NOT
> objective gentleness (narrow POSITIVE).** `s3/wikibridge.py` `WB_SOURCE=synth` (`build_synth`),
> `docs/cloud_results/r38ba_synth.{json,log}`, **3 seeds**. Resolves the one open question R38B left: is the
> ~0.65 non-replayed floor from *real-text redundancy/pretrained-prior overlap*, or just a *gentler
> objective*? Method (codex-designed): run the **identical `run_ingest` path** — same scaffold+QA-CE,
> consolidate-from-M_{t-1}, compact old replay, anchor/noanchor, old-only split, closed-book paraphrase eval
> — swapping ONLY the corpus to **independent invented facts** (unique proper-noun subject AND answer per
> fact, no reuse anywhere → zero real-world redundancy/prior). Matched 3 streams / 12 articles; base
> closed-book 0.000, RAG-gold 1.000. Old-only non-replayed old-paraphrase EM:
>
> | arm | **synth (3-seed avg)** | synth per-seed | **R38B real text** |
> |---|---|---|---|
> | naive_cpt | 0.000 | — | 0.000 |
> | **k0** (zero old replay, +anchor) | **0.181** | 0.175/0.267/0.100 | **0.625** |
> | **k0_noanchor** (zero old replay) | **0.164** | 0.100/0.217/0.175 | **0.738** |
> | k1 | 0.243 | — | 0.641 |
> | k3 | 0.312 | — | 0.687 |
> | compact_cpt_qa (full replay) | final-para 0.614, replayed 0.664 | — | 0.80 / 0.787 |
>
> **Decisive against codex's pre-registered threshold (synth k0/k0_noanchor ≤ 0.25 → real-text effect; ≥ 0.55
> → objective gentleness):** synth k0/k0_noanchor = **0.181/0.164, all three seeds ≤ 0.27**, vs real text
> 0.625/0.738. On independent facts the zero-replay floor **collapses toward naive** — gentle consolidation
> does NOT preserve unreplayed old facts once real-world structure is removed. Sanity controls all hold:
> naive 0.0; full replay retains (replayed 0.664 » non-replayed); replay protects *replayed* items (k1
> replayed 0.764 vs non-replayed 0.243) = R36-A item-rehearsal, no neighbor spillover. Because
> `shared_templates` (max question-format sharing) ALREADY collapsed, the floor is **not** format/template
> sharing either — so the `unique_templates` follow-up is unnecessary. **Conclusion:** the R38B real-text
> ~0.65 no-replay floor is a genuine property of **prior-anchored / redundant real knowledge** — real facts
> overlap the pretrained manifold and resist interference even without rehearsal; independent invented facts
> have no such anchor and are overwritten by any later update unless explicitly replayed. **This closes the
> footprint thread with an honest split:** the strong "K-replay gives sub-O(#facts) coverage via redundancy
> spillover" claim is dead (R38B); the narrow claim **"real/prior-anchored corpora leave a no-replay
> retention floor that independent facts lack" is SUPPORTED** (R38B-A) — "reading" real knowledge that
> overlaps the prior is genuinely cheaper to retain than memorizing independent tuples, but not via replay
> coverage. **Caveats:** 0.5B, 3 streams / 12 articles, 3 seeds; synth is globally harder (full-replay 0.61 <
> real 0.80), which is itself consistent with weaker prior anchoring for invented facts.
>

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
> **R36-A (minimal-footprint rehearsal — is a TINY committed replay set enough?) — NEGATIVE (item
> rehearsal, not stream protection)** `s2/lifecycle_bakeoff.py` `BK_REPLAY_K`, log
> `docs/cloud_results/replayk_sweep_r36a.log`, 2-seed 6×40. After bounding rehearsal-*free* CL (R36-I/C),
> the practical question is how *small* training-time rehearsal can be: replay only a **fixed nested
> committed K-subset per prior stream** (self-distill to the pre-round snapshot, no gold-old, single dense,
> no inference memory), sweeping `K ∈ {0,1,2,4,8,16,40}`. The decisive readout is **replayed vs
> NON-replayed** old-fact recall — do a few probes protect the *whole* stream, or only the rehearsed
> items? Result (2-seed):
>
> | K | all-seen | oldest-forget | REPLAYED-seen | NON-REPLAYED-seen |
> |---|----------|---------------|---------------|-------------------|
> | naive | 0.371 | +0.762 | — | — |
> | 0 | 0.371 | +0.750 | — | 0.268 |
> | 1 | 0.367 | +0.712 | 0.900 | 0.246 |
> | 2 | 0.406 | +0.688 | 0.900 | 0.279 |
> | 4 | 0.446 | +0.675 | 0.950 | 0.289 |
> | 8 | 0.546 | +0.600 | 0.887 | 0.375 |
> | 16 | 0.646 | +0.425 | 0.906 | 0.396 |
> | 40 (full ours) | 0.875 | +0.000 | — | — |
> | oracle | 0.977 | −0.088 | — | — |
>
> **Sanity gates pass** (K=0 all-seen 0.371 = `naive` 0.371; K=40 = full-ours 0.875 ≈ historical 0.877 —
> reproducibility confirmed). **The result is a clean negative:** REPLAYED items are always retained
> (~0.89–0.95) but **NON-REPLAYED items are forgotten as badly as naive** — 0.375 at K=8, only 0.396 even
> at K=16 (40% of the stream rehearsed), vs naive 0.371 and full-ours ~0.88. `all-seen` rises with K
> **purely as a mixture** (the rehearsed fraction is saved, the rest is not); there is **no within-stream
> generalization** from rehearsing some items to protecting their stream-mates. Fails codex's decisive
> gate (NON-REPLAYED within 0.08 of full-ours): K=8 gap is ~0.50. **Interpretation:** replay-consolidation's
> anti-forgetting is **item-specific, not stream-distributional** at this scale — to protect the stream you
> must essentially rehearse the stream. So *random* minimal-footprint rehearsal does NOT compress the R33/R35
> positive; footprint ≈ O(PER·T), not O(K·T). This is a useful refutation of the "few probes protect the
> stream" hypothesis and points the next lever at **non-random coverage**: coreset/diversity subset
> selection or **key/relation-conditioned self-generated probes** that span the stream's distribution
> rather than memorizing K fixed items (R36-A2). (No growth claim: growth fixed throughout; R23/R31 stand.)
>
> **R36-A2 (compact/precomputed replay TARGET — cheaper replay-consolidation) — POSITIVE** `run_consolidate`
> `BK_REPLAY_TGT` / `ours_tgt_*`, log `docs/cloud_results/r36a2_compact_targets.log` (+ JSON), 2-seed 6×40.
> R36-A ruled out *fewer* replayed facts (independent facts ⇒ footprint O(#facts)). The surviving lever is
> *less per fact / less compute*: replace live self-distillation (a resident pre-round **snapshot** model
> + a teacher forward **every** replay step) with a **compact per-fact target captured once at commit** and
> replayed cheaply. Result (all cover every fact, single dense, no gold-old, no inference memory):
>
> | arm | all-seen | all-para | oldest-forget | replay-teacher-fwd | peak VRAM | snapshot |
> |-----|----------|----------|---------------|--------------------|-----------|----------|
> | naive | 0.387 | 0.348 | +0.738 | 0 | — | — |
> | ours (snapshot KL) | 0.875 | 0.765 | +0.000 | **5000** | 9407 MB | yes |
> | **ours_tgt_answerid** | **0.875** | **0.765** | **+0.000** | **0** | **6711 MB** | **no** |
> | ours_tgt_topk8 | 0.873 | 0.765 | +0.000 | 0 | 6711 MB | no |
> | ours_tgt_current (control) | 0.317 | 0.304 | +0.750 | 0 | — | no |
>
> **`answerid` matches `ours` exactly** (0.875/0.765/+0.000, identical age-curve) — a **single committed
> answer token per (fact,view)**, stored once at each stream's commit and replayed via CE, fully replaces
> snapshot self-distillation while eliminating the resident snapshot model (**−29% peak VRAM: 9407→6711 MB**)
> and **all replay-time teacher forwards (5000→0)**. `topk8` (soft top-8 sketch) is equivalent (0.873) at
> more bytes. The circular control `current` (self-target the model's live argmax) collapses to naive
> (0.317, forget +0.750) — proving the **stored target information** does the work, not mere old-prompt
> exposure. All codex gates pass (all-seen ≥0.84, all-para ≥0.70, oldest-forget ≤0.06, newest within 0.03,
> base-hop drop ≤0.03, teacher-fwd 0, no snapshot). **Label hygiene:** the target is the *committed dense
> model's* answer at commit time (`target_source=committed_dense_argmax`), i.e. a committed-teacher signal
> — for already-correct facts it equals the counterfactual value, but no old **dataset gold** is read
> during replay. **Contribution:** replay-consolidation's per-fact information requirement (R36-A lower
> bound) can be met by a **1-token committed target**, making the proven in-weights consolidation strictly
> cheaper (no teacher model resident, no replay-time teacher compute) with zero retention/plasticity cost.
> Storage stays O(#facts) — as it must for independent facts — but per-fact cost is minimal and the
> training-time teacher is removed. (Growth still fixed; no growth claim.)
>
> **R37-A (localized-write growth isolation — rehearsal-free by minimal forward footprint) — PARTIAL
> POSITIVE + first GROWTH-NECESSARY result** `s2/lifecycle_bakeoff.py` `run_grow_local`, logs
> `docs/cloud_results/r37a_grow_local.log` (confounded) + `r37a_clean.{log,json}` + `r37a_clean_nogrow.json`,
> 2-seed 6×40. `naive` is already grow+freeze+isolate+no-replay (every old prompt READS the newest always-on
> block ⇒ it forgets +0.74). New lever: penalize each grown block's **relative last-token forward footprint**
> `‖Δh‖²/‖h‖²` on a NON-new reference so the block fires only on new keys, ~0 (identity) on old prompts ⇒
> can't perturb them. No router, no replay, single dense. **codex caught a confound** (base-anchor KL was
> also hitting old counterfactual prompts = anti-memory) — after splitting base-KL (anchors only) from the
> locality reference, clean 2-seed:
>
> | arm | all-seen | oldest-forget |
> |---|---|---|
> | naive (grow+freeze, no replay) | 0.387 | +0.738 |
> | grow_local_decoy (same-template unlabeled coverage) | **0.608** | +0.425 |
> | grow_local_decoy **NOGROW** (fixed capacity) | 0.325 | +0.787 |
> | grow_local_oracle (old-prompt coverage — INVALID upper bound) | **0.835** | +0.062 |
> | nswrite (best prior rehearsal-free) | 0.673 | +0.500 |
> | replay (ceiling) | 0.875 | +0.000 |
>
> **Two real results.** (1) Forward-footprint locality is a **genuinely new, competitive rehearsal-free
> mechanism**: `decoy` 0.608 (same-template *unlabeled* coverage, no replay/router/old-labels) beats naive
> by +0.22 and approaches `nswrite` 0.673; and the `oracle` upper bound (footprint-identity on the actual
> old prompts) reaches **0.835, nearly matching replay (0.875)** — so the mechanism can nearly close the
> replay gap *if old-prompt coverage improves*; the decoy→oracle gap is a **coverage** gap, not a mechanism
> ceiling. The earlier "oracle paradox" (oracle destroyed old recall) was purely the base-KL confound —
> **footprint-identity on old prompts preserves recall**, overturning my initial "footprint ≠ answer
> preservation" read. (2) **First GROWTH-NECESSARY result in the whole arc**: fixed-capacity + same locality
> loss (`NOGROW` 0.325) is *worse than naive*, while per-round growth + locality (0.608) works — the
> localized write **requires fresh grown capacity** to write into (R23/R31/R32/R34 all found growth useless;
> here it is load-bearing *for this mechanism*). **Honest bound:** decoy (0.608) is still **below replay
> (0.875)**, so this is not a growth-justification for retention *in general* — it is a competitive
> rehearsal-free mechanism whose specific form needs growth. Same-template coverage is unlabeled generated
> data (within contract, not old replay). Open: close the coverage gap (better generated / relation-
> conditioned coverage) to push decoy → oracle → replay.
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
