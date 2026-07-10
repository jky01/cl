#!/usr/bin/env bash
set -euo pipefail

ROOT="${QA_WATCH_ROOT:-/home/aa/cl}"
CODEX_BIN="${CODEX_BIN:-/home/aa/.local/bin/codex}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
LOG_DIR="$ROOT/.codex"
LOG_FILE="$LOG_DIR/qa-claude-watch.log"
LOCK_FILE="$LOG_DIR/qa-claude-watch.lock"

mkdir -p "$LOG_DIR" "$ROOT/qa/claude" "$ROOT/qa/codex"

exec >>"$LOG_FILE" 2>&1

{
  flock -n 9 || {
    echo "$(date -Is) skip: previous qa watcher run is still active"
    exit 0
  }

  cd "$ROOT"

  latest_claude="$(
    find qa/claude -maxdepth 1 -type f -name '????-??-??.??.??.??.md' -printf '%f\n' |
      sort |
      tail -1
  )"
  latest_codex="$(
    find qa/codex -maxdepth 1 -type f -name '????-??-??.??.??.??.md' -printf '%f\n' |
      sort |
      tail -1
  )"

  if [[ -z "${latest_claude:-}" ]]; then
    echo "$(date -Is) idle: no qa/claude files"
    exit 0
  fi

  if [[ -n "${latest_codex:-}" ]] &&
    { [[ "$latest_codex" == "$latest_claude" ]] || [[ "$latest_codex" > "$latest_claude" ]]; }; then
    echo "$(date -Is) idle: latest claude=$latest_claude latest codex=$latest_codex"
    exit 0
  fi

  echo "$(date -Is) processing: qa/claude/$latest_claude latest codex=${latest_codex:-none} model=$CODEX_MODEL reasoning=$CODEX_REASONING_EFFORT"

  prompt="You are running from the qa/claude watcher scheduler in /home/aa/cl.

Project objective (contract v2, 2026-07-11):
The dialogue must advance this research goal: a small model continually READS many real books/documents, incorporates the new knowledge INTO ITS WEIGHTS, avoids catastrophic forgetting, and may grow into a larger model when growth is justified by measurement. Hard constraints (unchanged): no external memory at INFERENCE (the final artifact is one dense checkpoint, closed-book, no retrieval/task-id/memory-router), and no joint full retraining over all historical data as the main method.

Explicitly LEGAL under contract v2: training-time rehearsal, including compact cue / self-generated-QA replay AND scheduled selective RE-READING of previously read source material (sequential refresh, like human re-reading). Sources persist externally; a reading-list pointer is not inference memory. Strict rehearsal-free retention stays a banked negative result (R36/R43/R48/R49a/R50), not a requirement.

Primary optimization target: MINIMIZE LIFETIME REFRESH COST — refresh FLOPs / revisited source tokens / persistent ledger bytes per RETAINED surprise bit as book count grows. Key open questions: (1) weights-level SPACING EFFECT — do required refresh intervals lengthen after each successful refresh (amortized sublinear lifetime cost)? (2) how much does natural book redundancy cut the refresh bill (redundancy-supported core vs singleton-exception tail)? (3) write-time self-annotation — the model generates its own cue/QA ledger from the CURRENT source at acquisition time (no human/gold labels).

Evaluation lens:
- Prefer ideas that strengthen knowledge-into-weights with no external memory at inference.
- Preserve old knowledge and base capabilities without catastrophic forgetting.
- Growth is justified ONLY by measurement: it must bend the refresh-demand curve (fewer lifetime refresh FLOPs per retained bit) or pass a capability/compute frontier fixed-small cannot reach; never a slogan.
- Avoid proposals whose only answer is full joint retraining over all historical data.
- Account costs honestly: refresh compute scales with revisited source tokens; do not hide O(history) work behind small pointers.
- Keep analysis concrete: experiments, controls, pass/fail criteria, likely failure modes, and how each step moves the project objective forward.

Task:
1. Read qa/claude/$latest_claude in full.
2. Analyze it in detail in English, grounded in the repository context when relevant.
3. Write a new Markdown response to qa/codex/yyyy-mm-dd.hh.mm.ss.md using the current timestamp.
4. The Markdown must record the source file name: qa/claude/$latest_claude.
5. Do not overwrite existing files.
6. After writing, ensure the latest qa/codex filename is later than $latest_claude.
7. When the exchange is a design/strategy discussion (not a narrow code review, launch confirmation, or yes/no ruling), append a section with open-ended, boundary-pushing, high-risk/high-reward brainstorming aligned with the project objective. If the message is a focused review/confirmation/ruling, stay concise and skip that section unless a genuinely promising angle exists.

Do not perform unrelated code changes."

  "$CODEX_BIN" \
    exec \
    --ephemeral \
    -m "$CODEX_MODEL" \
    -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
    -C "$ROOT" \
    -s workspace-write \
    "$prompt"

  echo "$(date -Is) done: qa/claude/$latest_claude"
} 9>"$LOCK_FILE"
