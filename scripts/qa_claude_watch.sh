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

Project objective:
The dialogue must advance this research goal: build a path where a small model can learn continually and grow into a larger model; new knowledge must be incorporated into the model weights; catastrophic forgetting must be avoided; and the solution must not rely on joint full retraining / full-data joint learning as the main method.

Evaluation lens:
- Prefer ideas that strengthen knowledge-into-weights with no external memory at inference.
- Preserve old knowledge and base capabilities without catastrophic forgetting.
- Treat growth as something that must be justified by capability saturation or compute advantage, not as a slogan.
- Avoid proposals whose only answer is full joint retraining over all historical data.
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
