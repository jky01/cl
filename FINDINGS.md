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

- **Growth ≠ capability-per-param.** On real Qwen at matched trainable params,
  **in-place fine-tune BEATS growth** (`qwen_growcap`: hop1/2/3 INPLACE 0.98/0.98/0.94
  vs GROW 0.96/0.92/0.87). At toy scale, growth's high-K "breakthrough" is a **bimodal
  lottery** (2/5 seeds; `diag_grow_hops_ms`) and naive width-scaling does not fix it —
  that was undertraining, and even with compute+lr scaled, growth's edge over a
  wide-shallow model **vanishes** (`diag_grow_hops_scale/2`). Growth's value is
  retention, not raw capability.
- **The naive growth controller failed.** A one-chunk training-loss-delta trigger grows
  on temporary plateaus and **loses to from-scratch** (`diag_controller2`). Fixed by a
  **held-out-slope + patience** trigger (`diag_controller3`): grows once to the sweet
  spot, beats fixed L2/L4/L8 (mean 0.68–0.76; the N=3 win was optimistic vs N=8).
- **Mean-pool value readout** and **raw-feature routing** both failed at scale and were
  replaced (answer-position readout; trained retrieval router).

## What is validated vs open

**Validated (this repo):** no-forgetting on real Qwen (0.5B & 1.5B), router-free for
memory and growth, memory scaling to ~10k facts, an autonomous grow-to-sweet-spot
controller with a robust signal, function-preserving growth transfers to Qwen
(`qwen_grow`, Δlogits=0).

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
