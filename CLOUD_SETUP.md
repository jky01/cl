# Cloud GPU setup + Tier-1 experiment plan

This repo (`s0/`) is a from-scratch prototype of a neuromodulated continual-learning
small model that **learns without forgetting AND grows its core**. The full story,
results, and honest caveats are in **`s0/PROGRESS.md`**; condensed state (the
hard-won findings — read these first when picking up) is in **`docs/memory/`**
(`s0-step0-state.md`, `s0-step0-design-constraints.md`, `MEMORY.md`).

Validated so far on a local RTX 2070 (8GB) + Qwen2.5-**0.5B** (frozen, feature
extraction only). This doc is the handoff to a **Tier-1** box: a single **24–40GB
Ampere+ GPU** (RTX 4090 / A100-40G — bf16 required) to push to **1–3B** real models.

## 0. Setup on the cloud box

```bash
git clone <this-repo> && cd <repo>
python3 -m venv .venv && . .venv/bin/activate
# torch matched to the box's CUDA (NOT the local cu128 build). e.g. A100/cu124:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```
Recreate the venv on the box — do NOT copy the local `.venv` (it's gitignored and
CUDA-specific). Then open Claude Code in the repo; it will re-orient from
`PROGRESS.md` + `docs/memory/`.

## 1. Reproduce (sanity that the port works)

```bash
python -m s0.run --smoke          # proxy: single+multi-fact, versioning, commit gate, lifelong
python -m s0.qwen_iface           # Qwen injection interface (0.00 diff)
python -m s0.qwen_grow            # function-preserving growth on Qwen (0.00 diff)
python -m s0.qwen_integrated      # growth + memory compose on Qwen (fp32)
python -m s0.qwen_lifelong        # 192-fact lifelong + growth on Qwen
```

## 2. Tier-1 experiment plan (what the bigger GPU unlocks)

Priorities, roughly in order. All the `qwen_*.py` scripts hardcode
`NAME = "Qwen/Qwen2.5-0.5B"` — bump to `-1.5B` / `-3B` there. Switch fp32→**bf16**
(`dtype=torch.bfloat16`) once on Ampere+ (fixes the fp16 instability without the
fp32 memory cost; see PROGRESS.md "gotcha").

1. **Real-model scale-up.** Re-run `qwen_memory`, `qwen_conflict`, `qwen_admission`,
   `qwen_multitoken` at **1.5B and 3B**. Confirm the wins (recall, versioning
   0.98 vs RAG, admission, multi-token) hold / sharpen.
   NOTE: the **frozen-feature** scripts (memory/conflict/admission/multitoken/
   retrieval) already fit at **1.5B on an 8GB card** (1.5B fp16 = ~3.1GB, 28
   layers, hidden 1536) — validate those at 1.5B LOCALLY before renting; only the
   TRAINING-heavy ones (`qwen_integrated`/`qwen_lifelong` train grown layers in
   fp32 → too big for 8GB at 1.5B) need the cloud. `n_base` is now derived
   dynamically (was hardcoded 24 for 0.5B; 1.5B has 28).
2. **Growth that ADDS capability on a real model.** Extend `qwen_integrated`:
   grow Qwen by several layers, train the grown layers (bf16, more steps, richer
   data than the toy fact-LM), and measure capability GAIN on a task Qwen-0.5B/1.5B
   is weak at (multi-hop, long-context reasoning). We only showed growth *preserves*
   + *composes* on Qwen; this shows growth *helps* on a real model (shown on the
   toy in `diag_grow_hops`).
3. **Growth penalty at bigger scale.** `diag_growpenalty2` with larger `d_model`,
   more layers (2→16+), more seeds — does the "growth ≥ from-scratch" toy result
   hold as models grow? (The from-scratch baseline is the compute cost here.)
4. **Full re-sync at scale.** `qwen_lifelong` at 500–2000 facts with more re-sync
   steps (the 192-fact run only partially recovered — needs more retrieval training).
   Add capacity/eviction (proxy Step 3A) on Qwen for a persistent growing bank.
5. **Benchmarks that matter.** vs **RAG** (retriever + context) and vs **full
   fine-tuning / sequential-LoRA** on real facts — the memory should win on
   context-cost, versioning, admission, and no-forgetting, not raw single-fact acc.

Rough budget: items 1,4,5 are cheap (frozen base + feature extraction / small
training) — hours on one 24GB card. Item 2 (train grown layers) and item 3
(from-scratch baselines) are the compute sinks; a single A100-40G handles 1–3B.
7B+ / true "super-large" and §28 meta-training are Tier-2/3 (see PROGRESS.md).

## 3. Notes / gotchas (from the local session)

- **bf16 on Ampere+** — the local RTX 2070 lacked it; direct fp16 training NaN'd
  freshly-grown layers (had to use fp32). On A100/4090 use bf16.
- **Left padding** for the Qwen scripts (`tok.padding_side="left"`) so `[:,-1]` is
  the real last token and generations append at the end.
- **Retrieval at scale** needs a big enough training batch (B≥96 for a ~200-fact
  bank; B=48 collapsed) and a DISTINCT-key bank.
- Growth = **deepening** (identity-init layers), not widening (widening changes
  `d_model` and breaks every module that reads hidden states).
- The memory side never needs backprop through the LM (frozen feature extraction);
  only training the grown core layers does, and only the top layers get gradients.
