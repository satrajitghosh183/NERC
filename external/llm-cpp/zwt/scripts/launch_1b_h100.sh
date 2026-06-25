#!/usr/bin/env bash
# launch_1b_h100.sh — train the 1B zwt model on an H100, using pre-tokenized
# data that already lives on the attached volume so root disk stays clean.
#
# Also wires Discord + email alerts in the same style as scripts/run_7B.sh:
# a second tmux window runs a monitor loop that tails the log, posts
# periodic progress to a Discord "updates" webhook, and fires alerts
# (Discord + email) on start / stall / fail / recover / complete.
#
# Assumptions:
#   * You're in the repo root (has zwt/conf/owt_1B_h100.conf).
#   * The tokenized OpenWebText .npy is already somewhere under
#     /media/volume/Prep_and_Voice_Training.
#   * An H100 is visible to nvidia-smi.
#   * .env.alerts exists (or alerts will be silently skipped). Copy from
#     .env.alerts.example and fill in DISCORD_ALERTS / DISCORD_UPDATES /
#     ALERT_EMAILS.
#
# Usage:
#   bash zwt/scripts/launch_1b_h100.sh            # locate + train + monitor
#   bash zwt/scripts/launch_1b_h100.sh --dry      # stop after dry-run
#   bash zwt/scripts/launch_1b_h100.sh --attach   # reattach to running run
#   bash zwt/scripts/launch_1b_h100.sh --no-alerts  # skip monitor window
#
# Stop a run:  tmux kill-session -t zwt_1b

set -euo pipefail

# ── Defaults ──
# Token data is located via zwt/scripts/find_tokens.sh:
#   ZWT_TOKENS=/abs/path/tokens.npy  (highest priority — direct file)
#   ZWT_VOLUME=/some/dir              (search this dir first)
#   default candidate dirs: /media/volume/Prep_and_Voice_Training,
#     $HOME/data, /data, ./downloads, ./data
CONF="zwt/conf/owt_1B_h100.conf"
BIN="./build/zwt_pretrain"
TMUX_SESSION="zwt_1b"
ENV_FILE=".env.alerts"
STALE_TIMEOUT=600       # if log hasn't moved in this many seconds, stall
POLL_INTERVAL=30
UPDATE_INTERVAL=600     # progress post cadence
LABEL="1B"

DO_DRY_ONLY=0
DO_ATTACH=0
DO_NO_ALERTS=0
_MONITOR=0
MONITOR_LOG=""          # set when --_monitor
for a in "$@"; do
  case "$a" in
    --dry)          DO_DRY_ONLY=1 ;;
    --attach)       DO_ATTACH=1 ;;
    --no-alerts)    DO_NO_ALERTS=1 ;;
    --_monitor=*)   _MONITOR=1; MONITOR_LOG="${a#--_monitor=}" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [[ "$DO_ATTACH" -eq 1 ]]; then
  exec tmux attach -t "$TMUX_SESSION"
fi

# ── Load secrets ──
ALERT_EMAILS=""
DISCORD_ALERTS=""
DISCORD_UPDATES=""
if [[ -f "$ENV_FILE" ]]; then
  set +u; source "$ENV_FILE"; set -u
elif [[ "$_MONITOR" -eq 0 && "$DO_NO_ALERTS" -eq 0 && "$DO_DRY_ONLY" -eq 0 ]]; then
  echo "note: $ENV_FILE not found — alerts disabled."
  echo "      cp .env.alerts.example .env.alerts to enable."
fi
HOSTNAME="$(hostname)"

# ── Messaging helpers (shared by launch + monitor) ──
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
  if [[ "$subject" == *"FAILED"* || "$subject" == *"STALLED"* ]]; then color=16711680; fi
  if [[ "$subject" == *"COMPLETED"* || "$subject" == *"RECOVERED"* ]]; then color=65280; fi
  if [[ "$subject" == *"Started"* ]]; then color=3447003; fi
  send_email "$subject" "$body"
  _discord_post "$DISCORD_ALERTS" "$subject" "$body" "$color"
}

send_progress() {
  local subject="$1" body="$2"
  _discord_post "$DISCORD_UPDATES" "$subject" "$body" 3447003
}

# Parse zwt_pretrain's log line:
#   "step   1234  loss 3.1416  lr 3.00e-04  |g| 0.987  12345 tok/s"
get_stats() {
  local log="$1"
  local last_line
  last_line=$(grep -E '^step +[0-9]+ +loss ' "$log" 2>/dev/null | tail -1 || echo "")
  if [[ -z "$last_line" ]]; then
    echo "step=? loss=? lr=? |g|=? tok/s=?"
    return
  fi
  local step loss lr gnorm toks
  step=$(echo  "$last_line" | grep -oP '^step +\K[0-9]+'                || echo "?")
  loss=$(echo  "$last_line" | grep -oP 'loss \K[0-9.]+'                 || echo "?")
  lr=$(echo    "$last_line" | grep -oP 'lr \K[0-9.eE+-]+'               || echo "?")
  gnorm=$(echo "$last_line" | grep -oP '\|g\| \K[0-9.]+'                || echo "?")
  toks=$(echo  "$last_line" | grep -oP '\K[0-9]+(?= tok/s)'             || echo "?")
  echo "step=$step loss=$loss lr=$lr |g|=$gnorm tok/s=$toks"
}

# ══════════════════════════════════════════════════════════════
# MONITOR MODE — runs inside tmux window "monitor"
# ══════════════════════════════════════════════════════════════
if [[ "$_MONITOR" -eq 1 ]]; then
  LOG="$MONITOR_LOG"
  echo "[$(date)] Monitor started. Watching $LOG ..."
  echo "  Polling every ${POLL_INTERVAL}s, progress updates every ${UPDATE_INTERVAL}s"
  echo "  Stale threshold: ${STALE_TIMEOUT}s (uses log mtime as heartbeat)"
  echo ""
  alerted=0
  last_update=0

  # Wait up to 60s for the log file to appear.
  for _ in $(seq 60); do [[ -f "$LOG" ]] && break; sleep 1; done

  while true; do
    sleep "$POLL_INTERVAL"
    now=$(date +%s)

    # ── End-of-run detection ──
    if [[ -f "$LOG" ]] && grep -q 'EXIT_CODE=' "$LOG" 2>/dev/null; then
      exit_line=$(grep 'EXIT_CODE=' "$LOG" | tail -1)
      exit_code="${exit_line#*EXIT_CODE=}"
      stats=$(get_stats "$LOG")
      tail_log=$(tail -40 "$LOG" 2>/dev/null || echo "(no log)")

      if [[ "$exit_code" == "0" ]]; then
        send_alert "[$LABEL] COMPLETED on $HOSTNAME" \
"Training completed at $(date).
$stats

$tail_log"
        send_progress "[$LABEL] COMPLETED" "$stats"
        echo "[$(date)] Training COMPLETED."
      else
        send_alert "[$LABEL] FAILED (exit $exit_code) on $HOSTNAME" \
"FAILED at $(date) with exit code: $exit_code
$stats

$tail_log"
        send_progress "[$LABEL] FAILED (exit $exit_code)" "$stats"
        echo "[$(date)] Training FAILED: exit $exit_code"
      fi
      echo ""
      echo "Monitor done. Press any key to close."
      read -n1
      break
    fi

    # ── Periodic progress post ──
    if (( now - last_update >= UPDATE_INTERVAL )); then
      stats=$(get_stats "$LOG")
      if [[ "$stats" != *"step=?"* ]]; then
        send_progress "[$LABEL] Progress on $HOSTNAME" "$stats"
        echo "[$(date)] Progress: $stats"
      fi
      last_update=$now
    fi

    # ── Staleness check (log mtime as heartbeat) ──
    if [[ -f "$LOG" ]]; then
      last_mod=$(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG" 2>/dev/null)
      age=$(( now - last_mod ))
      if (( age > STALE_TIMEOUT )); then
        if (( alerted == 0 )); then
          stats=$(get_stats "$LOG")
          tail_log=$(tail -30 "$LOG" 2>/dev/null || echo "(no log)")
          send_alert "[$LABEL] STALLED on $HOSTNAME (${age}s)" \
"Log has not advanced for ${age}s (threshold: ${STALE_TIMEOUT}s)
$stats

$tail_log"
          echo "[$(date)] ALERT: log stale ${age}s"
          alerted=1
        fi
      else
        if (( alerted == 1 )); then
          send_alert "[$LABEL] RECOVERED on $HOSTNAME" "Log advanced again. Age: ${age}s"
          echo "[$(date)] Recovered."
        fi
        alerted=0
      fi
    fi
  done
  exit 0
fi

# ══════════════════════════════════════════════════════════════
# LAUNCH MODE
# ══════════════════════════════════════════════════════════════

# ---- 0. sanity: binary, config, GPU ----------------------------------------
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — build first: bash ./scripts/build.sh" >&2
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "missing $CONF — are you in the repo root?" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found — this is a GPU script, bailing" >&2
  exit 1
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
echo

# ---- 1. locate the tokenized .npy ------------------------------------------
# find_tokens.sh handles the resolution: ZWT_TOKENS direct path > ZWT_VOLUME
# search root > default candidate dirs (Jetstream volume, $HOME/data, /data,
# ./downloads, ./data). On a server with a roomy root disk you don't need
# any mounted volume — drop the tokens under $HOME/data or ./downloads and
# this script finds them.
if ! TOKENS=$(bash "$(dirname "$0")/find_tokens.sh"); then
  exit 1
fi
if command -v du >/dev/null && du -BG "$TOKENS" >/dev/null 2>&1; then
  SIZE_HUMAN=$(du -BG "$TOKENS" | cut -f1)
else
  SIZE_HUMAN=$(du -h "$TOKENS" | cut -f1)
fi
echo "using tokens: $TOKENS ($SIZE_HUMAN)"
echo

# ---- 2. wire the data path to what the config expects ----------------------
mkdir -p data/owt
LINK="data/owt/owt_tokens.npy"
if [[ -L "$LINK" || -f "$LINK" ]]; then
  EXISTING=$(readlink -f "$LINK" 2>/dev/null || echo "")
  if [[ "$EXISTING" != "$(readlink -f "$TOKENS")" ]]; then
    rm -f "$LINK"
    ln -s "$TOKENS" "$LINK"
  fi
else
  ln -s "$TOKENS" "$LINK"
fi
echo "symlink: $LINK -> $(readlink -f "$LINK")"

# ---- 3. checkpoints --------------------------------------------------------
# By default ckpts/ is a plain dir on the root disk. Override with
#   ZWT_CKPT_DIR=/path/to/ckpts   (e.g. on a mounted volume)
# and the script will symlink ckpts/ -> that dir. On a roomy root disk
# (e.g. 400 GB) leave ZWT_CKPT_DIR unset.
if [[ -n "${ZWT_CKPT_DIR:-}" ]]; then
  mkdir -p "$ZWT_CKPT_DIR"
  if [[ -e ckpts && ! -L ckpts ]]; then
    echo "ckpts/ already exists as a real dir — leaving it alone." >&2
  else
    rm -f ckpts
    ln -s "$ZWT_CKPT_DIR" ckpts
  fi
else
  mkdir -p ckpts
fi
echo "ckpts dir: $(readlink -f ckpts)"
echo

# ---- 4. dry run ------------------------------------------------------------
echo "=== Dry-run (build model, exit before step 1) ==="
"$BIN" "$CONF" --dry-run

if [[ "$DO_DRY_ONLY" -eq 1 ]]; then
  echo "dry-run requested; stopping."
  exit 0
fi

# ---- 5. launch under tmux --------------------------------------------------
mkdir -p logs
LOG="logs/1b_$(date +%Y%m%d_%H%M%S).log"
PROJECT_DIR="$(pwd)"

echo "============================================"
echo "  1B Training Launch"
echo "============================================"
echo "  Config:   $CONF"
echo "  Log:      $LOG"
echo "  Stale:    ${STALE_TIMEOUT}s (log mtime)"
echo "  Updates:  every ${UPDATE_INTERVAL}s"
echo "  Emails:   ${ALERT_EMAILS:-none}"
echo "  Discord:  alerts=${DISCORD_ALERTS:+yes} updates=${DISCORD_UPDATES:+yes}"
echo "  tmux:     $TMUX_SESSION"
echo "============================================"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session '$TMUX_SESSION' already exists — attach with:" >&2
  echo "  tmux attach -t $TMUX_SESSION" >&2
  echo "or kill it first:  tmux kill-session -t $TMUX_SESSION" >&2
  exit 1
fi

# Window 0: training — live output + log. Capture exit code in the log so
# the monitor can detect end-of-run without polling a separate file.
tmux new-session -d -s "$TMUX_SESSION" -n "training" \
  "bash -c 'cd $PROJECT_DIR && stdbuf -oL -eL $BIN $CONF 2>&1 | tee $LOG; echo EXIT_CODE=\${PIPESTATUS[0]} >> $LOG; echo; echo Training ended. Press any key.; read -n1'"

# Window 1: monitor — progress posts + stall/fail/complete alerts.
if [[ "$DO_NO_ALERTS" -eq 0 ]]; then
  tmux new-window -t "$TMUX_SESSION" -n "monitor" \
    "bash $PROJECT_DIR/zwt/scripts/launch_1b_h100.sh --_monitor=$LOG"
  send_alert "[$LABEL] Started on $HOSTNAME" \
"Config: $CONF
Host:   $HOSTNAME
Time:   $(date)
Tokens: $TOKENS
Log:    $LOG
tmux:   $TMUX_SESSION"
fi

echo
echo "training started in tmux."
echo "attach:   tmux attach -t $TMUX_SESSION      (detach: Ctrl-b then d)"
echo "switch:   Ctrl-b n  (next window)  /  Ctrl-b p  (prev)"
echo "tail:     tail -f $LOG"
echo "kill:     tmux kill-session -t $TMUX_SESSION"
echo "windows:  0:training   1:monitor"
