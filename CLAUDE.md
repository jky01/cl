# Project instructions

## Standing rule — qa/codex ↔ qa/claude exchange (every conversation)

At the **start of every conversation**, before other work:

1. **Read the latest file in `qa/codex/`** (the most recent by filename timestamp).
2. **Analyze it in detail** — engage with its substance: confirm where the evidence agrees, refine or push back where warranted, and add genuinely new value (concrete experimental design, failure modes, success criteria). Ground the analysis in the repo's evidence (`FINDINGS.md`, `docs/memory/s0-step0-state.md`, `docs/cloud_results/`).
3. **Write the response to `qa/claude/`** as a new Markdown file named `yyyy-mm-dd.hh.mm.ss.md` (current timestamp; get it with `date +"%Y-%m-%d.%H.%M.%S"`). Do not overwrite existing files — each reply is a new timestamped file.

This is an ongoing back-and-forth design dialogue between codex and claude about the research direction. Keep replies additive and honest, not just agreement.

## Research context (2026-07)

Current thrust is the **Grow-and-Consolidate** pivot (`s2/`): consolidate transient scaffold-memory knowledge into a single dense checkpoint (no external memory at inference). Honest bottom line so far (R19–R31): the robust, multi-seed result is **knowledge-into-weights via replay/self-distillation**; **growth is NOT yet justified** (R23: not retention; R31: not capacity) — the open task is a **cross-stream composition** saturation benchmark to test whether growth is ever necessary. See `FINDINGS.md` (top) for the full arc and `docs/memory/s0-step0-state.md` for round-by-round detail.
