# s0 — Progress & Handoff

A from-scratch, all-neural **continual-learning small-model** prototype, built to
validate the architecture in `reference/` (esp. §27 Sleep Consolidator / §28
meta-training). Everything here runs on a controlled **synthetic** world with a
small **frozen proxy transformer** standing in for a real LM — the goal is to
prove the *mechanisms*, not to ship a product. Runs on CPU or CUDA
(`.venv`, torch 2.11.0+cu128, validated on an RTX 2070).

```
python -m s0.run --smoke      # fast end-to-end demo (d=64), ~few min on GPU
python -m s0.run              # default config (d=128, 60-vocab)
```

## Status snapshot (all validated in the synthetic sandbox)

| Capability | Result |
|---|---|
| Step 0 — single-fact write/read (§11.4 gate) | acc 1.0, locality 1.0 |
| Step 1 — multi-fact via difficulty curriculum | nf 1–16 ≈1.0, locality ~1.0 |
| Step 2 — conflict versioning (§27.10) | now/before 1.0, routing-fail 0 |
| Step 3A — occupancy allocation / capacity | survival 1.0 within capacity; FIFO over |
| Step 4 — commit gate / admission (§27.18) | reliable-recall-after-attack 1.0; admit 1.0 / reject 0.0 |
| Step 5 — lifelong zero-forgetting | capsule flat 1.0 across 8 sessions; LoRA rival ~0.02 |
| Growth — function-preserving deepen | hidden Δ = 0.00 at growth (zero-forgetting) |
| Growth — adds capability | K-hop: L=2 stuck (K2 0.52), grow→L=4 0.98; control (2× steps) stuck |
| Growth — autonomous controller | plateau-trigger grows L=2→4→6 by itself, K2 0.46→0.96 |
| Growth — neural gate (learned) | grow blocks ablation-essential, K-hop fully solved (K5 0.97) |

## Commit history

```
473b941  Step 0  single-fact capsule write/read passes the §11.4 gate
639698e  Step 1  multi-fact retrieval via difficulty curriculum
97d4510  Step 2  conflict versioning — non-destructive, context-routed updates
c662a43  Step 3A occupancy-aware allocation — zero within-capacity forgetting
01d6b8b  Step 4  commit gate — reject untrustworthy conflicting writes
5bd7008  Step 5  lifelong zero-forgetting demo + fix version-routing recency bug
d125ef1  grow_deeper — function-preserving growth operator
4226826  demo that growth IMPROVES capability (K-hop), not just preserves
327746b  autonomous growth controller (plateau-triggered) — self-driven small→large
257d8b9  neural growth trigger via learned per-layer gates (GrowableCore)
9d6b0a2  ungameable norm-based capacity cost for the growth gate (honest mixed result)
```

## Component map

- `world.py` — synthetic (subject,relation,object) world; templates; versioned
  queries (`now`/`before`); conflict episodes. CLAIM BOUNDARY: tests
  *combinatorial binding over a known vocab*, not novel-symbol / real language.
- `core.py` — `ProxyCore` (frozen decoder LM proxy) + `pretrain_core`;
  `grow_deeper` (function-preserving identity-block deepening);
  `GrowableCore` + `GatedBlock` (learned per-layer growth gates).
- `capsule.py` — `CapsuleMemory`: write/read, `SRKeyEncoder` (retrieval key from
  subject/relation **token hiddens**), occupancy allocation, version routing
  (time stamp + `ctx_enc`), relevance gate, commit gate (admission).
- `train.py` — `train_omega0` (curriculum, warmup, losses: answer CE, InfoNCE
  retrieval, locality, conflict, safety) + evals (`eval_capsule`,
  `eval_conflict`, `eval_safety`, `eval_lifelong`).
- `baselines.py` — no-mem, in-context (≈RAG), external-KV, oracle-slot, LoRA.
- `diag_*.py` — growth experiments (grow_hops, autogrow, neurogrow).

## Key findings (hard-won; don't re-derive)

1. **Retrieval contrastive collapse → fixed by token-level keys.** Attention/mean
   pooling of hidden states collapses under contrastive loss (matched and random
   query·key become identical). Building keys from the **subject/relation token
   hiddens** (`SRKeyEncoder`, by token-id range) + cross-batch InfoNCE fixed it
   (retrieval@1 ≈0.91). This is also the brittle scaffolding that must be
   replaced for real text (see roadmap).
2. **Multi-fact needs a difficulty curriculum.** Training directly on large
   N_facts collapses to chance; ramping episode size 1→max over 70% of training
   fixes it.
3. **Recall decay under load was OVERWRITES, not retrieval** (rec|survived flat
   ~0.87). The content-addressed product-key allocator collided; **occupancy
   (free-slot) placement** gives survival 1.0 within capacity and also fixed
   multi-fact (nf=16 0.76→1.0).
4. **Growth without forgetting works**: identity-initialised added layers give
   bit-for-bit identical hidden states (Δ=0.00), and the added capacity is
   usable; it raises a depth-limited capability ceiling that more in-place
   training cannot (controlled by a 2×-steps baseline).
5. Bugs found & fixed: divergent residual injection (added inject-LN + gated +
   clip); version routing applied to plain queries → false forgetting (gated by
   `has_ctx`).

## Open problems / honest caveats

- **Not practically usable yet.** Synthetic world, frozen tiny proxy core (not a
  real LM), templated facts. The `gather_sr` retrieval key exploits the rigid
  template (token-id ranges) — it will NOT work on free text.
- **Clean neural growth DIAL is unsolved.** The growth *mechanism* + ablation
  verification hold, but neither L1-on-gate (gameable) nor the norm cost gives a
  clean monotonic "how-much-to-grow" control (single-seed, non-monotonic; a
  tiny-norm contribution can be essential). Needs multi-seed / scheduling / a
  different controller — a real research point.
- **vs RAG.** The in-context baseline (≈RAG) scores 1.0; the memory must justify
  itself on what RAG can't do (no context-length cost, versioning, admission,
  consolidation) — must be benchmarked directly against RAG.
- Much of §27/§28 is unbuilt: dream replay, merge/abstraction, distill-into-core,
  expert/MoE growth, and the whole §28 meta-training of Ω.

## Qwen real-model path (started — `qwen_iface.py`, `qwen_retrieval.py`)

- **Interface verified**: Qwen2.5-0.5B (transformers 5.12.1, fp16, ~1GB GPU),
  `max|lm_head(final_hidden) - logits| = 0.00` → inject `h_last + g*R → renorm →
  lm_head` exactly as the proxy. Arch: hidden=896, 24 layers, GQA, vocab=151936,
  tied emb, Qwen2RMSNorm.
- **Blocker #1 CRACKED**: free-text retrieval key works. Qwen (frozen) mean-pooled
  last hidden + a small trained projection + InfoNCE → **retrieval@1 = 1.000** on
  free-text paraphrased facts (toy proxy mean-pool only hit 0.41). Only the tiny
  projection trains — no backprop through Qwen.
- **Full write/read on Qwen works** (`qwen_memory.py`). Free-text fact written
  (key + value from frozen Qwen features), cloze query retrieves it and injects
  `H_ans + g*R → lm_head` (single-token answers; direct additive injection — the
  tied lm_head makes R≈emb[answer] a clean logit boost; NO inject_ln). Result:
  capsule recall with **NO context 0.961** vs no-mem 0.047. Needed lr 5e-4 +
  grad-clip to stabilise (was 0.97/0.05 across runs without).
- **HONEST vs RAG**: fair few-shot RAG = **0.945** (zero-shot cloze RAG only 0.195
  because Qwen-0.5B-base copies unreliably in cloze — a model limitation, verified
  in `qwen_ragcheck.py`, NOT our bug). So accuracy is COMPARABLE — the memory does
  NOT beat RAG on accuracy; its value is no-context-cost + the §27 features
  (versioning/admission/consolidation) that raw RAG lacks. Data must use SENSIBLE
  typed values (random attr/value pairing makes nonsensical facts even RAG fights).

- **WHERE MEMORY BEATS RAG — temporal versioning on Qwen** (`qwen_conflict.py`).
  Each fact has two versions over time (v1@t=0, v2@t=1, sharing a key built from
  (name,attr) only); a Currently/Originally query routes by explicit time stamp.
  **Memory: current 0.984, original 0.992. Raw-text RAG (both statements shuffled,
  undated, in context): ~0.22 both** — it can't tell which undated/unordered fact
  is current vs original. Needed explicit retrieval InfoNCE + DIRECT c_target
  supervision (current→1/original→0) to avoid ctx_enc collapse (same fix as the
  proxy relevance gate). Honest caveat: this beats UNDATED text-RAG; a RAG that
  embeds timestamps in the text could also do it — but at context cost + reasoning,
  which is exactly the structured-memory advantage.

- **Blocker #2 (multi-token answers) CRACKED** (`qwen_multitoken.py`). Per-step
  injection (inject the retrieved R at EVERY answer position) lets the memory
  drive a multi-token free-form answer ("San Francisco", "software engineer")
  with NO context. Free-generation exact-match (both tokens) = **0.979** vs
  no-mem 0.000 (few-shot RAG 1.000 — parity, memory's edge is context-cost +
  structure). NB: needed **left padding** (right-align real tokens) so the last
  hidden is `[:,-1]` and appended generations land at the sequence end — a
  right-pad bug was silently capping accuracy. Both real-model blockers (#1
  free-text key, #2 multi-token value) are now down.

## Roadmap (next, prioritized)

1. **Broaden the Qwen memory case**: (a) port the remaining §27 features
   (admission/trust, capacity, lifelong) onto Qwen + benchmark vs RAG; (b) scale
   (many facts, efficient ANN retrieval, persistent bank); (c) longer answers
   (3+ tokens / KV-prefix); (d) multi-seed stability.
2. **Clean neural growth controller** — multi-seed/scheduled; tie the growth
   trigger to a learned SleepGate/CapacityNet (§27.16) rather than a heuristic.
3. **Expert/MoE growth (§27.11) + distill-into-core (§27.12)** — the other
   capacity-growth axis and the "slow knowledge → core weights" step.
4. **§28 end-to-end Ω meta-training** — the largest piece; meta-learn the whole
   sleep/grow controller over simulated lifelong sequences.

See the memory files under the project's `memory/` for the same state in
condensed, recall-oriented form (`s0-step0-state`, `s0-step0-design-constraints`).
