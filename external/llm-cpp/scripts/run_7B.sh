#!/usr/bin/env bash
# run_7B.sh — Launch 7B training + monitoring in tmux
#
# Both training and monitoring run inside tmux. You can disconnect
# safely and everything keeps running.
#
# Usage:
#   ./scripts/run_7B.sh
#   ./scripts/run_7B.sh --conf conf/olmo_7B_h100.conf --timeout 600
#
# tmux controls:
#   tmux attach -t train7B        # Attach to session
#   Ctrl+B d                      # Detach (leave running)
#   Ctrl+B n / Ctrl+B p           # Switch windows (training ↔ monitor)
#   tmux kill-session -t train7B  # Kill everything

set -euo pipefail

# ── Defaults ──
CONF="conf/olmo_7B_h100.conf"
LOG="7B.log"
HEARTBEAT="heartbeat_7B.txt"
STALE_TIMEOUT=600
POLL_INTERVAL=30
UPDATE_INTERVAL=600
SESSION="train7B"
ENV_FILE=".env.alerts"
_MONITOR=0

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conf)       CONF="$2";            shift 2 ;;
    --log)        LOG="$2";             shift 2 ;;
    --heartbeat)  HEARTBEAT="$2";       shift 2 ;;
    --timeout)    STALE_TIMEOUT="$2";   shift 2 ;;
    --update)     UPDATE_INTERVAL="$2"; shift 2 ;;
    --session)    SESSION="$2";         shift 2 ;;
    --env)        ENV_FILE="$2";        shift 2 ;;
    --_monitor)   _MONITOR=1;           shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ── Load secrets from env file ──
ALERT_EMAILS=""
DISCORD_ALERTS=""
DISCORD_UPDATES=""

if [[ -f "$ENV_FILE" ]]; then
  set +u
  source "$ENV_FILE"
  set -u
else
  [[ "$_MONITOR" == 0 ]] && echo "WARNING: $ENV_FILE not found. cp .env.alerts.example .env.alerts"
fi

HOSTNAME="$(hostname)"

# ── Messaging helpers ──
send_email() {
  local subject="$1" body="$2"
  [[ -z "$ALERT_EMAILS" ]] && return
  IFS=',' read -ra ADDR <<< "$ALERT_EMAILS"
  for addr in "${ADDR[@]}"; do
    addr="$(echo "$addr" | xargs)"
    [[ -z "$addr" ]] && continue
    echo "$body" | mail -s "$subject" "$addr" 2>/dev/null || \
      echo "[$(date)] WARNING: email to $addr failed"
  done
}

_discord_post() {
  local webhook="$1" subject="$2" body="$3" color="$4"
  [[ -z "$webhook" ]] && return
  local desc="${body:0:4000}"
  desc=$(printf '%s' "$desc" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
  local title
  title=$(printf '%s' "$subject" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')
  local payload="{\"embeds\":[{\"title\":$title,\"description\":$desc,\"color\":$color}]}"
  curl -s -H "Content-Type: application/json" -d "$payload" "$webhook" >/dev/null 2>&1 || \
    echo "[$(date)] WARNING: Discord post failed"
}

send_alert() {
  local subject="$1" body="$2"
  local color=16776960  # yellow
  if [[ "$subject" == *"FAILED"* || "$subject" == *"STALLED"* ]]; then color=16711680; fi  # red
  if [[ "$subject" == *"COMPLETED"* || "$subject" == *"RECOVERED"* ]]; then color=65280; fi  # green
  if [[ "$subject" == *"Started"* ]]; then color=3447003; fi  # blue
  send_email "$subject" "$body"
  _discord_post "$DISCORD_ALERTS" "$subject" "$body" "$color"
}

send_progress() {
  local subject="$1" body="$2"
  _discord_post "$DISCORD_UPDATES" "$subject" "$body" 3447003  # blue
}

get_stats() {
  local last_line
  last_line=$(grep -E 'Step [0-9]+/' "$LOG" 2>/dev/null | tail -1 || echo "")
  if [[ -z "$last_line" ]]; then
    echo "epoch=? step=? loss=? tok/s=?"
    return
  fi
  local epoch step loss toks
  epoch=$(echo "$last_line" | grep -oP 'Epoch \K[0-9]+' || echo "?")
  step=$(echo "$last_line" | grep -oP 'Step \K[0-9]+/[0-9]+' || echo "?")
  loss=$(echo "$last_line" | grep -oP 'loss: \K[0-9.]+' || echo "?")
  toks=$(echo "$last_line" | grep -oP 'tok/s: \K[0-9]+' || echo "?")
  echo "epoch=$epoch step=$step loss=$loss tok/s=$toks"
}

# ══════════════════════════════════════════════════════════════
# MONITOR MODE — runs inside tmux window "monitor"
# ══════════════════════════════════════════════════════════════
if [[ "$_MONITOR" == 1 ]]; then
  echo "[$(date)] Monitor started. Watching $LOG ..."
  echo "  Polling every ${POLL_INTERVAL}s, progress updates every ${UPDATE_INTERVAL}s"
  echo "  Heartbeat timeout: ${STALE_TIMEOUT}s"
  echo ""
  alerted=0
  last_update=0

  while true; do
    sleep "$POLL_INTERVAL"
    now=$(date +%s)

    # ── Check if training ended ──
    if grep -q 'EXIT_CODE=' "$LOG" 2>/dev/null; then
      exit_line=$(grep 'EXIT_CODE=' "$LOG" | tail -1)
      exit_code="${exit_line#*EXIT_CODE=}"
      stats=$(get_stats)
      tail_log=$(tail -30 "$LOG" 2>/dev/null || echo "(no log)")

      if [[ "$exit_code" == "0" ]]; then
        send_alert "[7B] COMPLETED on $HOSTNAME" \
          "Training completed at $(date).
$stats

$tail_log"
        send_progress "[7B] COMPLETED" "$stats"
        echo "[$(date)] Training COMPLETED."
      else
        send_alert "[7B] FAILED (exit $exit_code) on $HOSTNAME" \
          "FAILED at $(date) with exit code: $exit_code
$stats

$tail_log"
        send_progress "[7B] FAILED (exit $exit_code)" "$stats"
        echo "[$(date)] Training FAILED: exit $exit_code"
      fi
      echo ""
      echo "Monitor done. Press any key to close."
      read -n1
      break
    fi

    # ── Periodic progress ──
    if (( now - last_update >= UPDATE_INTERVAL )); then
      stats=$(get_stats)
      if [[ "$stats" != *"epoch=?"* ]]; then
        send_progress "[7B] Progress on $HOSTNAME" "$stats"
        echo "[$(date)] Progress: $stats"
      fi
      last_update=$now
    fi

    # ── Heartbeat staleness ──
    if [[ -f "$HEARTBEAT" ]]; then
      last_mod=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || stat -f %m "$HEARTBEAT" 2>/dev/null)
      age=$(( now - last_mod ))

      if (( age > STALE_TIMEOUT )); then
        if (( alerted == 0 )); then
          stats=$(get_stats)
          hb=$(cat "$HEARTBEAT" 2>/dev/null || echo "(empty)")
          tail_log=$(tail -20 "$LOG" 2>/dev/null || echo "(no log)")
          send_alert "[7B] STALLED on $HOSTNAME (${age}s)" \
            "No heartbeat for ${age}s (threshold: ${STALE_TIMEOUT}s)
$stats

Heartbeat: $hb

$tail_log"
          echo "[$(date)] ALERT: heartbeat stale ${age}s"
          alerted=1
        fi
      else
        if (( alerted == 1 )); then
          send_alert "[7B] RECOVERED on $HOSTNAME" "Heartbeat recovered. Age: ${age}s"
          echo "[$(date)] Recovered."
        fi
        alerted=0
      fi
    fi
  done

  exit 0
fi

# ══════════════════════════════════════════════════════════════
# LAUNCH MODE — creates tmux session with both windows, then exits
# ══════════════════════════════════════════════════════════════

# ── Verify prerequisites ──
if [[ ! -x build/olmo_train ]]; then
  echo "ERROR: build/olmo_train not found. Run ./scripts/build.sh --cuda first."
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "ERROR: Config $CONF not found."
  exit 1
fi

rm -f "$HEARTBEAT"

echo "============================================"
echo "  7B Training Launch"
echo "============================================"
echo "  Config:      $CONF"
echo "  Log:         $LOG"
echo "  Heartbeat:   $HEARTBEAT"
echo "  Stale:       ${STALE_TIMEOUT}s"
echo "  Updates:     every ${UPDATE_INTERVAL}s"
echo "  Emails:      ${ALERT_EMAILS:-none}"
echo "  Discord:     alerts=${DISCORD_ALERTS:+yes} updates=${DISCORD_UPDATES:+yes}"
echo "  tmux:        $SESSION"
echo "============================================"

# Kill any existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Window 0: "training" — live output + log file
# expandable_segments reduces fragmentation from Muon's Newton-Schulz temporaries
tmux new-session -d -s "$SESSION" -n "training" \
  "bash -c 'cd $PROJECT_DIR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./build/olmo_train $CONF 2>&1 | tee $LOG; echo EXIT_CODE=\${PIPESTATUS[0]} >> $LOG; echo; echo Training ended. Press any key.; read -n1'"

# Window 1: "monitor" — alerts, heartbeat checks, progress updates
tmux new-window -t "$SESSION" -n "monitor" \
  "bash $PROJECT_DIR/scripts/run_7B.sh --_monitor --conf '$CONF' --log '$LOG' --heartbeat '$HEARTBEAT' --timeout $STALE_TIMEOUT --update $UPDATE_INTERVAL --session '$SESSION' --env '$ENV_FILE'"

# Send start alert
send_alert "[7B] Started on $HOSTNAME" \
  "Config: $CONF
Host: $HOSTNAME
Time: $(date)
tmux: $SESSION"

echo ""
echo "[$(date)] Launched in tmux session '$SESSION'"
echo ""
echo "  Attach:   tmux attach -t $SESSION"
echo "  Detach:   Ctrl+B d"
echo "  Switch:   Ctrl+B n (next) / Ctrl+B p (prev)"
echo "  Kill all: tmux kill-session -t $SESSION"
echo ""
echo "  Windows:"
echo "    0:training  — live training output"
echo "    1:monitor   — alerts & progress"
