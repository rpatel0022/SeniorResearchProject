#!/usr/bin/env bash
# Generic watchdog. Replaces run3-specific watchdog_run3.sh.
#
# Inputs (env vars):
#   WD_PID       - PID to monitor (required)
#   WD_LOG       - log file path to inspect on death and stall (required)
#   WD_LABEL     - label for the alert file (default: job)
#   WD_STALL_SEC - warn (don't kill) if log mtime exceeds this many seconds (default: 600)
#   WD_POLL_SEC  - poll interval (default: 60)
#   WD_ALERT_DIR - dir to write alerts to (default: dirname of WD_LOG)
#
# Behavior:
#   - Polls `kill -0 $WD_PID` every WD_POLL_SEC seconds.
#   - On PID death: writes alert with last 100 lines of WD_LOG; exits.
#   - If log mtime exceeds WD_STALL_SEC and process is still alive, prints
#     a STALL warning to stdout. Does NOT kill (could be a long validation pass).
#   - Does NOT auto-restart (would just re-trigger whatever killed it).

set -euo pipefail

: "${WD_PID:?WD_PID is required}"
: "${WD_LOG:?WD_LOG is required}"

WD_LABEL="${WD_LABEL:-job}"
WD_STALL_SEC="${WD_STALL_SEC:-600}"
WD_POLL_SEC="${WD_POLL_SEC:-60}"
WD_ALERT_DIR="${WD_ALERT_DIR:-$(dirname "$WD_LOG")}"

mkdir -p "$WD_ALERT_DIR"

echo "[watchdog] $(date -Iseconds) starting: PID=$WD_PID log=$WD_LOG label=$WD_LABEL stall=${WD_STALL_SEC}s poll=${WD_POLL_SEC}s"

last_stall_warn_t=0

while true; do
  if ! kill -0 "$WD_PID" 2>/dev/null; then
    ALERT="$WD_ALERT_DIR/watchdog_alert_${WD_LABEL}_$(date +%s).txt"
    {
      echo "=== watchdog alert ==="
      echo "PID:    $WD_PID"
      echo "LABEL:  $WD_LABEL"
      echo "LOG:    $WD_LOG"
      echo "TIME:   $(date -Iseconds)"
      echo ""
      echo "=== last 100 log lines ==="
      if [[ -f "$WD_LOG" ]]; then
        tail -n 100 "$WD_LOG"
      else
        echo "(log file missing)"
      fi
    } > "$ALERT"
    echo "[watchdog] $(date -Iseconds) PID $WD_PID exited; alert written to $ALERT"
    exit 0
  fi

  # Stall check
  if [[ -f "$WD_LOG" ]]; then
    now_t=$(date +%s)
    log_mtime=$(stat -c %Y "$WD_LOG")
    age=$(( now_t - log_mtime ))
    if (( age > WD_STALL_SEC )); then
      # warn at most once every WD_STALL_SEC to avoid log spam
      if (( now_t - last_stall_warn_t > WD_STALL_SEC )); then
        echo "[watchdog] $(date -Iseconds) STALL: log idle for ${age}s (threshold ${WD_STALL_SEC}s); PID $WD_PID still alive"
        last_stall_warn_t=$now_t
      fi
    fi
  fi

  sleep "$WD_POLL_SEC"
done
