# Project instructions

## Standing rule — per-round plan → codex dialogue → confidence-gated execution

The codex side is **automated locally**: `scripts/qa_claude_watch.sh` (crontab, every minute)
watches `qa/claude/`; whenever the newest `qa/claude` file is newer than the newest `qa/codex`
file, codex (gpt-5.5, xhigh) reads it and replies into `qa/codex/`. So the exchange is real-time.

**Every round of work follows this loop:**

1. **Plan first.** Before starting a task, write a *detailed* plan of what you're about to do to
   `qa/claude/` as `yyyy-mm-dd.hh.mm.ss.md` (`date +"%Y-%m-%d.%H.%M.%S"`; never overwrite).
2. **Detect codex reply by timestamp.** Poll `qa/codex/`; a file whose name-timestamp is *newer
   than the plan you just wrote* is codex's fresh response. (Codex normally answers within ~1–2 min.)
3. **Read + analyze it in detail** — additive and honest, grounded in the repo's evidence
   (`FINDINGS.md`, `docs/memory/s0-step0-state.md`, `docs/cloud_results/`).
4. **Act by confidence:**
   - **High confidence** (design converged, no blocking objection) → start the task directly.
   - **Need clearer direction** → write another `qa/claude/` file asking the specific question,
     wait one more round for codex, then proceed. (Iterate to obtain instructions.)
5. **Execute.** Use a remote pod if the task needs GPU; **reclaim (terminate) the pod when done**;
   **`git push`** all changes (code, `docs/cloud_results/`, `FINDINGS.md`, `qa/`).
6. **Review results → plan the next round → continue.** Keep going for the full autonomous window.

Keep replies substantive (refine/push back, don't just agree). **`qa/claude` ↔ `qa/codex` dialogue
files may be written entirely in English** (codex-facing technical exchange); only messages to the
user are in 繁體中文. RunPod key lives at `~/.runpod_key` (chmod 600) — **NEVER echo it**.
Non-Blackwell GPUs only (RTX 4090/A-series; 5090 sm_120 fails cu124).

## Research context (2026-07)

Current thrust is the **Grow-and-Consolidate** pivot (`s2/`): consolidate transient scaffold-memory knowledge into a single dense checkpoint (no external memory at inference). Honest bottom line so far (R19–R31): the robust, multi-seed result is **knowledge-into-weights via replay/self-distillation**; **growth is NOT yet justified** (R23: not retention; R31: not capacity) — the open task is a **cross-stream composition** saturation benchmark to test whether growth is ever necessary. See `FINDINGS.md` (top) for the full arc and `docs/memory/s0-step0-state.md` for round-by-round detail.
