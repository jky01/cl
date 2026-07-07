#!/usr/bin/env bash
# LOCAL pod reaper (cron, every minute). Terminates RunPod pods that have finished their job + a grace
# window, or that hit an absolute max-lifetime backstop. The RunPod API key NEVER leaves this machine.
#
# This runs on the user's always-on box (same host as qa_claude_watch cron), so it is independent of
# Claude's conversation liveness / token budget -> a pod that finishes while Claude is unresponsive is
# still reclaimed, and no idle-spin billing accrues.
#
# Registry (append one line per launched pod):
#   printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' PODID HOST PORT LOGFILE DONEGREP LAUNCH_EPOCH GRACE MAXLIFE >> ~/.cl_pods.tsv
# Reaper removes a line once the pod is terminated. Deregister on a normal manual reclaim to avoid clutter.
set -uo pipefail

REG="$HOME/.cl_pods.tsv"
STATE="$HOME/.cl_pods"
LOG="$STATE/reaper.log"
mkdir -p "$STATE"
[ -f "$REG" ] || exit 0
KEY="$(cat "$HOME/.runpod_key" 2>/dev/null)" || exit 0
[ -n "$KEY" ] || exit 0

now="$(date +%s)"
term() {   # terminate a pod via the RunPod API using the LOCAL key (never leaves this host)
  curl -s https://api.runpod.io/graphql \
    -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$1\\\"})}\"}" >/dev/null 2>&1
}
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes"

tmp="$(mktemp)"
while IFS=$'\t' read -r podid host port logfile donegrep launch grace maxlife; do
  [ -z "${podid:-}" ] && continue
  # hard backstop: hung job that never prints its done marker
  if [ $(( now - launch )) -ge "${maxlife:-18000}" ]; then
    term "$podid"; rm -f "$STATE/$podid.done"
    echo "$(date -Is) reaped(maxlife) $podid" >> "$LOG"; continue
  fi
  # remote check: job finished? (done marker in log, OR no training python running)
  done="$(ssh $SSHO -p "$port" root@"$host" \
    "grep -qF -- '$donegrep' '$logfile' 2>/dev/null && echo D || (ps aux | grep -q '[p]ython -m s' && echo R || echo D)" 2>/dev/null)"
  rc=$?
  if [ $rc -ne 0 ]; then
    # ssh unreachable -> could be transient; keep the entry, maxlife is the safety net
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$podid" "$host" "$port" "$logfile" "$donegrep" "$launch" "$grace" "$maxlife" >> "$tmp"
    continue
  fi
  if [ "$done" = "D" ]; then
    df="$STATE/$podid.done"
    [ -f "$df" ] || echo "$now" > "$df"
    dsince="$(cat "$df" 2>/dev/null || echo "$now")"
    if [ $(( now - dsince )) -ge "${grace:-1800}" ]; then
      term "$podid"; rm -f "$df"
      echo "$(date -Is) reaped(done+grace) $podid" >> "$LOG"; continue
    fi
  else
    rm -f "$STATE/$podid.done"   # still running -> reset the done timer
  fi
  # keep in registry
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$podid" "$host" "$port" "$logfile" "$donegrep" "$launch" "$grace" "$maxlife" >> "$tmp"
done < "$REG"
mv "$tmp" "$REG"
