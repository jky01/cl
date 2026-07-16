#!/usr/bin/env bash
# qa_codex_free.sh — UNRESTRICTED codex responder.
#
# Same file convention as qa_claude_watch.sh (watch qa/claude, reply into
# qa/codex/<timestamp>.md so Claude can detect the reply by name-timestamp),
# but with NO project framing, NO contract, NO evaluation lens, NO forced
# brainstorming section. codex reads the message and answers however it wants.
#
# This is deliberately kept SEPARATE from qa_claude_watch.sh. Run only ONE of
# the two on cron at a time (two responders would both reply and collide).

set -euo pipefail

ROOT="${QA_WATCH_ROOT:-/home/aa/cl}"
CODEX_BIN="${CODEX_BIN:-/home/aa/.local/bin/codex}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
LOG_DIR="$ROOT/.codex"
LOG_FILE="$LOG_DIR/qa-codex-free.log"
LOCK_FILE="$LOG_DIR/qa-codex-free.lock"

mkdir -p "$LOG_DIR" "$ROOT/qa/claude" "$ROOT/qa/codex"

exec >>"$LOG_FILE" 2>&1

{
  flock -n 9 || {
    echo "$(date -Is) skip: previous run still active"
    exit 0
  }

  cd "$ROOT"

  latest_claude="$(
    find qa/claude -maxdepth 1 -type f -name '????-??-??.??.??.??.md' -printf '%f\n' |
      sort | tail -1
  )"
  latest_codex="$(
    find qa/codex -maxdepth 1 -type f -name '????-??-??.??.??.??.md' -printf '%f\n' |
      sort | tail -1
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

  echo "$(date -Is) processing: qa/claude/$latest_claude model=$CODEX_MODEL reasoning=$CODEX_REASONING_EFFORT"

  # The ONLY rule: reply into a fresh qa/codex file so it can be detected by
  # its name-timestamp. No topic constraints, no lens, no framing.
  prompt="Read qa/claude/$latest_claude in full and reply to it however you see fit — you are unconstrained in content, stance, and scope. Think freely.

The single mechanical rule: write your reply as a new Markdown file at qa/codex/yyyy-mm-dd.hh.mm.ss.md using the current timestamp (run 'date +%Y-%m-%d.%H.%M.%S'). Do not overwrite any existing file, and make sure the filename you write is later than $latest_claude. Record the source filename (qa/claude/$latest_claude) somewhere in the reply. Do nothing else to the repository."

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
