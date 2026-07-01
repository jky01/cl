# Cloud GPU setup + Tier-1 experiment plan

This repo (`s0/`) is a from-scratch prototype of a neuromodulated continual-learning
small model that **learns without forgetting AND grows its core**. The full story,
results, and honest caveats are in **`s0/PROGRESS.md`**; condensed state (the
hard-won findings — read these first when picking up) is in **`docs/memory/`**
(`s0-step0-state.md`, `s0-step0-design-constraints.md`, `MEMORY.md`).

Validated so far on a local RTX 2070 (8GB) + Qwen2.5-**0.5B** (frozen, feature
extraction only). This doc is the handoff to a **Tier-1** box: a single **24–40GB
Ampere+ GPU** (RTX 4090 / A100-40G — bf16 required) to push to **1–3B** real models.

**The local mechanism-validation phase is COMPLETE, including two honest boundaries
that DEFINE the first cloud experiments (2026-07):**
- **Growth-adds-capability is real but LOTTERY at toy scale.** Multi-seed audit
  (`diag_grow_hops_ms.py`, N=5): "growth adds DEPTH-capability beyond compute" is
  sign-robust (grown-L4 > 2×-compute-L2 in 5/5 seeds at K4), and **warm-start
  growth is the ONLY arm that ever cracks high-K** (deep-from-scratch NEVER breaks
  through, max K4 0.41). BUT the breakthrough is bimodal: only **2/5 seeds** hit
  ~1.0, the rest stay ~0.3, and a post-growth K-curriculum did NOT lift it
  (`diag_grow_hops_curric.py`: 2/5 vs 2/5) → reliability is set by the init/basin,
  i.e. a **scale/architecture question, not a local-tuning one**.
- **The hand-crafted growth CONTROLLER failed (`diag_controller2.py`).** A
  single-window loss-delta is a bad saturation signal — temporary plateaus before
  phase transitions trigger premature, repeated growth that burns budget re-warming
  layers; the controller LOST to plain from-scratch. Growth TIMING needs a robust
  signal (held-out validation slope / patience / the meta-learned Ω of §28).

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

Priorities, in order. Items **A–C are the findings-driven core** (they resolve the
two local boundaries above — do these first); items 1–5 are the supporting
scale-up. All `qwen_*.py` scripts read `NAME = os.environ.get("QWEN_MODEL",
"Qwen/Qwen2.5-0.5B")` — set `QWEN_MODEL=Qwen/Qwen2.5-1.5B` (or `-3B`) instead of
editing. Switch fp32→**bf16** (`dtype=torch.bfloat16`) once on Ampere+ (fixes the
fp16 instability without the fp32 memory cost; see PROGRESS.md "gotcha").

### A. Is the growth-capability breakthrough RELIABLE at real scale? (the #1 question)
The toy is bimodal (2/5 seeds). Test whether real depth/width converges it toward
~5/5. Extend `diag_grow_hops_ms.py` (keep the 4 arms A/B/C/D and the per-seed
breakthrough count) but scale `d_model` (128→512/1024) and depth (2→4 → grow to
8/12), N≥8 seeds. **Success metric: breakthrough rate (K4&K5≥0.8) rises with
width/scale**; also confirm the sign-robust "grown > 2×-compute-L2" and "grown is
the ONLY arm that cracks high-K (from-scratch never does)" hold. If breakthrough
stays ~2/5 even at scale, growth needs a better warm-start/basin fix, not just size.

### B. Growth-adds-capability on a REAL model, with a PARAM-MATCHED control
Extend `qwen_integrated`: pick a task Qwen-1.5B is weak at (3–4-hop / long-context
composition), then compare three arms at **equal trainable-param & compute budget**:
grow +N layers & train them **vs** LoRA/in-place adapters on existing top layers
**vs** the frozen base. Growth wins the thesis only if **grown > param-matched
in-place** at high difficulty (depth, not just params). Caveat learned locally:
Qwen is already 24–28 layers deep, so +N is a small *relative* depth increase —
the toy's large 2→8 lift may not reproduce; a null here is itself informative
(the capability benefit may be specific to genuinely-shallow→deep).

### C. Replace the failed hand-crafted controller with a learned signal (§28 Ω)
`diag_controller2.py` proved a single-window loss-delta is a bad grow trigger.
Build the growth decision on a **held-out validation slope + patience** (grow only
after K chunks of no held-out gain, not one noisy plateau), then graduate to the
**meta-learned Ω** (§27-28): train the controller to predict "grow now / how much"
from (budget, saturation, headroom) across many tasks. Validate it beats both
fixed depth and the naive trigger on breakthrough-rate-per-FLOP.

### Supporting scale-up (cheaper, do alongside)
1. **Real-model win scale-up.** Re-run `qwen_memory`, `qwen_conflict`,
   `qwen_admission`, `qwen_multitoken` at **1.5B and 3B**; confirm recall,
   versioning (0.98 vs RAG), admission, multi-token hold / sharpen. The
   **frozen-feature** ones already fit at 1.5B on 8GB — validate LOCALLY first;
   only the training-heavy `qwen_integrated`/`qwen_lifelong` need the cloud.
2. **Growth penalty at bigger scale.** `diag_growpenalty2` with larger `d_model`,
   more layers (2→16+), more seeds — does "growth ≥ from-scratch" hold as models
   grow? (from-scratch baseline = the compute cost).
3. **Full re-sync at scale.** `qwen_lifelong` at 500–2000 facts (192-fact needed
   3500 re-sync steps for full recovery — scale the steps). Add capacity/eviction
   (proxy Step 3A) on Qwen for a persistent growing bank.
4. **Benchmarks that matter.** vs **RAG** (retriever + context) and vs **full
   fine-tuning / sequential-LoRA** — memory should win on context-cost, versioning,
   admission, no-forgetting, not raw single-fact acc.

Rough budget: the supporting items + B/C's frozen parts are cheap (hours on one
24GB card); A (wide from-scratch baselines) and B (train grown layers) are the
compute sinks — a single A100-40G handles 1–3B. 7B+ / true "super-large" and full
§28 meta-training are Tier-2/3 (see PROGRESS.md).

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
- **Toy tasks are high-variance / bimodal** — always run **multi-seed** (N≥5) and
  report breakthrough RATE + [min,max], never a single seed (a single run can read
  0.3 or 1.0 for the *same* config). Load-bearing claims must be sign-stability
  checks across seeds, not one number.
- **Don't grow on a one-chunk loss plateau** — loss briefly flattens right before a
  phase transition, so a naive plateau trigger grows prematurely/repeatedly and
  wastes budget re-warming identity layers. Gate growth on a held-out slope with
  patience (or the meta-learned Ω), not a single-window training-loss delta.
