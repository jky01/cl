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

**Attach file paths in every codex exchange (user directive 2026-07-06):** when sharing execution
results, cite the repo-relative script path(s) AND the log/artifact path(s) (e.g. `s2/lifecycle_bakeoff.py`,
`docs/cloud_results/kg_*.log`) so codex can read the exact files (it runs `-C /home/aa/cl`). Before
running a code change, post the changed file path(s) to codex for peer review when practical (review
gate before spending a pod). **Pod GPU preference: RTX 4090 > 5090 > 3090** (5090 needs the cu128 image).

Keep replies substantive (refine/push back, don't just agree). **codex is a peer, not an authority —
its answers can be WRONG.** Critically evaluate every codex claim: verify the math/logic, check it
against the repo's evidence, and push back or propose alternatives when it's mistaken or when a
question is genuinely open. Treat open research questions as peer discussion to reason through
together, not as instructions to implement. Don't defer by default; don't agree by default either.
**`qa/claude` ↔ `qa/codex` dialogue files may be written entirely in English** (codex-facing technical
exchange); only messages to the user are in 繁體中文. RunPod key lives at `~/.runpod_key` (chmod 600) — **NEVER echo it**.
Non-Blackwell GPUs only (RTX 4090/A-series; 5090 sm_120 fails cu124).

## Research orientation — continual learning is an UNSOLVED frontier (standing directive)

**Model continual learning is an unconquered problem; solving it may require methods that have not
yet been discovered. Treat it as open research: be willing to try genuinely novel mechanisms, and
discuss them in depth with codex** (via the qa/ dialogue) before and while building. Don't settle for
polishing known techniques when the honest gap is unsolved — name where our best result stops, then
attempt the frontier. (User directive, 2026-07-05.)

## Research context (2026-07)

Thrust: **Grow-and-Consolidate** (`s2/`) — consolidate transient scaffold-memory knowledge into a
single dense checkpoint (no external memory at inference). Honest arc so far (R19–R35, see `FINDINGS.md`
top + `docs/cloud_results/`): the robust multi-seed positive is **knowledge-into-weights via
replay/self-distillation** — R33 shows it beats standard CL baselines (naive, continued-FT-with-gold,
LoRA-merge) and external memory, no inference memory (all-seen 0.914). **Growth is NOT justified** at
0.5B synthetic scale — four independent negatives: R23 retention, R31 capacity, R32 OOD composition,
R34 in-distribution composition grokking. R35 brackets it (EWC ≪ replay ≪ gold-old oracle).
**The real open problem: rehearsal-FREE continual learning** — our replay is still rehearsal; the enemy
is *interference*. Frontier probe R36-I (`nswrite`): interference-aware null-space writing with no
replay; growth may finally be necessary only when the interference-free write-subspace saturates
(depth vs width/parallel-adapter growth is the phase-2 question).
