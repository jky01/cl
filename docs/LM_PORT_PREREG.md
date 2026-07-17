# LM Port — Preregistration (algorithmic micro-language)

Preregistered BEFORE the first full run, per the qa/codex dialogue (2026-07-17). This fixes the
generator, shortcut family, intervention matrix, phase boundaries, and acceptance thresholds so results
cannot be reverse-fit. It operationalizes the toy-line conclusion: **rule acquisition is gated by
identifiability (equivalence-breaking evidence), not optimization; retention of a genuinely-shared,
softly-parameterized rule across later plastic learning is the second open interface.**

Generator: `s4/microlang.py`. Model/training/eval harness: `s4/` (later files). Local GPU (RTX 2070)
for the small transformer; a pod only if it doesn't fit.

## 1. Language (typed sequence transduction)

Vocabulary `V` value tokens `{0..V-1}` (default V=16) plus reserved specials: `RESET`, one `CMD_*`
per operator, `BOS`, `EOS`, `SEP`. An example is `CMD_op  x_1 .. x_L  SEP  y_1 .. y_L` where
`y = op(x)`; the model is trained to produce `y` autoregressively given `CMD_op` + `x`. Inference = the
command token + a fresh input; **no demonstrations, no retrieval, no environment label.**

### Operators
Old primitives (phase 1):
- `copy`  : y[i] = x[i]
- `inc`   : y[i] = (x[i]+1) mod V
- `shift` : y[i] = x[i-1]  (y[0]=0)   — positional

New stateful operator to DISCOVER (phase 2) — a short program, NOT a builtin/rename:
- `csum_reset` : state s=0; for i: if x[i]==RESET → s=0, y[i]=0; else s=(s+x[i]) mod V, y[i]=s.
  (cumulative sum with conditional reset — depends on the whole history since the last reset.)

Interference-capable operator (phase 3) — shares the stateful+reset structure so it plausibly damages
phase-2 shared params:
- `rmax_reset` : state s=0; for i: if x[i]==RESET → s=0, y[i]=0; else s=max(s,x[i]), y[i]=s.

Compositions (test set): `B(A(x))` for A,B in {old, new}; specifically old→new (`inc` then
`csum_reset`) and new→old (`csum_reset` then `shift`) that are NEVER shown jointly in training.

### OOD axis
Train on lengths `L ∈ [3, 12]` and composition depth ≤ 1 (single op) plus a few depth-2 in-training
combos that EXCLUDE the held-out composition pairs. Test on lengths `L ∈ [16, 40]` (longer) and the
held-out compositions (deeper/unseen). Extrapolation = did the model learn the RECURRENCE vs memorize
fixed-length/positional behavior.

## 2. Representable shortcut family (must be broken by evidence, not by fiat)

For `csum_reset`, enumerate the shortcuts a model could fit on short training data:
1. **fixed-length / bounded unrolling**: memorize the map per length ≤ L_train; fails for longer.
2. **position-only**: y[i] = g(i) ignoring content; survives if reset positions are fixed.
3. **local-window**: y[i] = f(x[i], x[i-1], .., x[i-k]) for small k; fails because state spans the
   whole segment since the last reset.
4. **reset-near-boundary**: if RESET only ever appears at position 0–2, the model can ignore reset
   logic and treat it as plain csum.
5. **token-identity / restricted-alphabet**: fit only on the training token subset.
6. **template / delimiter / SEP-position** cues.
7. **task-order / environment-specific** cues (if env is inferable from surface form).

## 3. Environments = targeted interventions (each breaks ≥1 named shortcut, preserves the program)

All environments live INSIDE the training regime (short sequences); OOD (long) is held out for test.

| Env | Intervention | Shortcut broken |
|---|---|---|
| E0 base    | fixed length 8, single reset at pos≤2, full alphabet | (reference; many shortcuts survive) |
| E_len      | length varies uniformly in [3,12]                    | fixed-length / position-only |
| E_reset    | reset count/position varies (0–2 resets, any pos)    | reset-near-boundary / position-only |
| E_alpha    | two disjoint token subsets across sub-envs           | token-identity / restricted-alphabet |
| E_tmpl     | delimiter/format varies                              | template / delimiter cues |

Marginals balanced so a shortcut cannot survive through a proxy (e.g. length uncorrelated with reset
position). **Environment labels may shape the training objective (e.g. an invariance/worst-env loss)
but are ABSENT at validation and inference.**

Held-out sets: (a) **primary** = familiar interventions in unseen COMBINATIONS (e.g. E_len×E_reset
jointly, never trained jointly); (b) **nuisance** = one preregistered transformation never used in
training (e.g. a value-preserving token relabeling / a new delimiter).

## 4. Minimal-environment identifiability study (central)

For every subset S of {E0,E_len,E_reset,E_alpha,E_tmpl}: report the EQUIVALENCE CLASS of candidate
programs still consistent with S's observations (enumerated over the declared hypothesis/shortcut
family), not just accuracy. Conditions: single-env, insufficient, minimally-sufficient, redundant.
Key readout = the smallest environment set that makes `csum_reset` the UNIQUE consistent program.

## 5. Sequential protocol (no replay of earlier raw examples)

1. Learn base language + old primitives {copy, inc, shift}.
2. DISCOVER `csum_reset` from multi-environment evidence; consolidate into weights (self-distill old
   ops over model-generated inputs + new-op data; no old raw replay).
3. Learn `rmax_reset` (interference-capable) from NEW data only — must update SHARED params.
4. Evaluate: old ops, csum_reset, longer/deeper OOD, unseen old/new compositions (both orders).

Diagnostic decomposition: if discovered vs supplied `csum_reset` diverge BEFORE phase 3 → acquisition/
consolidation bottleneck; if they match before but diverge AFTER → later interference selectively
damages the learned rule.

## 6. Arms / controls

- discovered rule, **sufficient** environments (main).
- discovered rule, **single/insufficient** environments (should fail to identify → predicts OOD fail).
- **oracle/supplied** rule, same consolidation path.
- **flexible sequential** learner WITHOUT the invariance intervention (expected: shortcut, no OOD).
- **protected-module** upper control (frozen isolated new op) — storage engineering, not the headline.
- **joint full training** — upper reference only, never the method.

## 7. Preregistered acceptance table (ALL required; ID accuracy alone is insufficient)

| Axis | Pass condition |
|---|---|
| Knowledge into weights | after phase 2, remove demos/records/retriever/env-label → CMD+input suffices |
| Rule acquisition | strong on longer L, deeper comps, unseen intervention combos, held-out nuisance; minimally-sufficient env materially beats insufficient |
| Continual retention | after phase 3, old/new/OOD within preregistered tolerance (Δ ≤ 0.03) of pre-phase-3 |
| Cross-integration | unseen old↔new compositions work in BOTH orders (isolated command acc doesn't count) |
| Growth necessity | growth solves a capacity/interference frontier a fixed model at MATCHED active-inference-compute cannot, OR matching needs materially more cumulative compute; compare larger-from-start too |
| Compute advantage | report cumulative train FLOPs, peak train memory, active inference FLOPs, param count (not param alone) |
| No-memory inference | no replay/demos/retrieval/task-selector/env-ID at inference; persisted params counted |
| No joint full retraining | each phase trains only on its current data; joint is eval context only |

## 8. Interpretation discipline

Strongest claim earned only if a softly-parameterized, shared, continually-updated model reproduces the
chain (equivalence-breaking evidence → invariant-rule discovery → consolidation into weights →
retention through phase 3 → memory-free inference, no joint retrain) AND passes cross-integration. If it
fails, report WHERE it breaks — identifiability / optimization / consolidation / interference / capacity
— each is a directed result, not an invitation to a hyperparameter sweep.
