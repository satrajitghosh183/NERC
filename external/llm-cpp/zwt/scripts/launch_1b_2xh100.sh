#!/usr/bin/env bash
# launch_1b_2xh100.sh — train the 1B zwt model on 2 H100s via NCCL DDP.
# Sister of launch_1b_h100.sh; reuses the same Discord/email monitor code.
# The only structural difference is that the training tmux window invokes
# launch_ddp.sh instead of the binary directly.
#
# Assumptions:
#   * You're in the repo root (has zwt/conf/owt_1B_2xh100.conf).
#   * Tokenized OWT .npy is somewhere under /media/volume/Prep_and_Voice_Training.
#   * 2 H100s visible to nvidia-smi.
#   * .env.alerts exists (or alerts will be silently skipped). Same format
#     as launch_1b_h100.sh.
#
# Usage:
#   bash zwt/scripts/launch_1b_2xh100.sh
#   bash zwt/scripts/launch_1b_2xh100.sh --dry        # build model, exit before step 1 on rank 0
#   bash zwt/scripts/launch_1b_2xh100.sh --attach     # reattach to running tmux
#   bash zwt/scripts/launch_1b_2xh100.sh --no-alerts  # skip monitor window
#
# Stop a run:  tmux kill-session -t zwt_1b_2x

set -euo pipefail

# Token data resolved by zwt/scripts/find_tokens.sh; see launcher header.
CONF="zwt/conf/owt_1B_2xh100.conf"
LAUNCHER="zwt/scripts/launch_ddp.sh"
TMUX_SESSION="zwt_1b_2x"
ENV_FILE=".env.alerts"
STALE_TIMEOUT=600
POLL_INTERVAL=30
UPDATE_INTERVAL=600
LABEL="1B-2x"
WORLD=2

DO_DRY_ONLY=0
DO_ATTACH=0
DO_NO_ALERTS=0
_MONITOR=0
MONITOR_LOG=""
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

# ── Load alert secrets (same convention as launch_1b_h100.sh) ──
ALERT_EMAILS=""
DISCORD_ALERTS=""
DISCORD_UPDATES=""
if [[ -f "$ENV_FILE" ]]; then
  set +u; source "$ENV_FILE"; set -u
elif [[ "$_MONITOR" -eq 0 && "$DO_NO_ALERTS" -eq 0 && "$DO_DRY_ONLY" -eq 0 ]]; then
  echo "note: $ENV_FILE not found — alerts disabled."
fi
HOSTNAME="$(hostname)"

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
  local color=16776960
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

# ════════════ MONITOR MODE ════════════
if [[ "$_MONITOR" -eq 1 ]]; then
  LOG="$MONITOR_LOG"
  echo "[$(date)] Monitor started. Watching $LOG ..."
  alerted=0
  last_update=0
  for _ in $(seq 60); do [[ -f "$LOG" ]] && break; sleep 1; done
  while true; do
    sleep "$POLL_INTERVAL"
    now=$(date +%s)
    if [[ -f "$LOG" ]] && grep -q 'EXIT_CODE=' "$LOG" 2>/dev/null; then
      exit_line=$(grep 'EXIT_CODE=' "$LOG" | tail -1)
      exit_code="${exit_line#*EXIT_CODE=}"
      stats=$(get_stats "$LOG")
      tail_log=$(tail -40 "$LOG" 2>/dev/null || echo "(no log)")
      if [[ "$exit_code" == "0" ]]; then
        send_alert "[$LABEL] COMPLETED on $HOSTNAME" \
          "Training completed at $(date).\n$stats\n\n$tail_log"
        send_progress "[$LABEL] COMPLETED" "$stats"
      else
        send_alert "[$LABEL] FAILED (exit $exit_code) on $HOSTNAME" \
          "FAILED at $(date) with exit code: $exit_code\n$stats\n\n$tail_log"
        send_progress "[$LABEL] FAILED (exit $exit_code)" "$stats"
      fi
      echo
      echo "Monitor done. Press any key to close."
      read -n1
      break
    fi
    if (( now - last_update >= UPDATE_INTERVAL )); then
      stats=$(get_stats "$LOG")
      if [[ "$stats" != *"step=?"* ]]; then
        send_progress "[$LABEL] Progress on $HOSTNAME" "$stats"
        echo "[$(date)] Progress: $stats"
      fi
      last_update=$now
    fi
    if [[ -f "$LOG" ]]; then
      last_mod=$(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG" 2>/dev/null)
      age=$(( now - last_mod ))
      if (( age > STALE_TIMEOUT )); then
        if (( alerted == 0 )); then
          stats=$(get_stats "$LOG")
          tail_log=$(tail -30 "$LOG" 2>/dev/null || echo "(no log)")
          send_alert "[$LABEL] STALLED on $HOSTNAME (${age}s)" \
            "Log has not advanced for ${age}s\n$stats\n\n$tail_log"
          alerted=1
        fi
      else
        if (( alerted == 1 )); then
          send_alert "[$LABEL] RECOVERED on $HOSTNAME" "Log advanced again. Age: ${age}s"
        fi
        alerted=0
      fi
    fi
  done
  exit 0
fi

# ════════════ LAUNCH MODE ════════════
if [[ ! -x "./build/zwt_pretrain" ]]; then
  echo "missing ./build/zwt_pretrain — build first: ./scripts/build.sh --cuda" >&2
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

N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$N_GPU" -lt "$WORLD" ]]; then
  echo "need >= $WORLD GPUs, only $N_GPU visible" >&2
  exit 1
fi

echo "=== GPUs ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
echo

# ── locate tokens (delegated to find_tokens.sh) ──
# Honors ZWT_TOKENS (direct path) > ZWT_VOLUME (search root) > defaults
# (/media/volume/Prep_and_Voice_Training, $HOME/data, /data, ./downloads,
# ./data). Works whether the data lives on a mounted volume or on a
# roomy root disk.
if ! TOKENS=$(bash "$(dirname "$0")/find_tokens.sh"); then
  exit 1
fi
echo "using tokens: $TOKENS"

mkdir -p data/owt
LINK="data/owt/owt_tokens.npy"
if [[ -L "$LINK" || -f "$LINK" ]]; then
  EXISTING=$(readlink -f "$LINK" 2>/dev/null || echo "")
  if [[ "$EXISTING" != "$(readlink -f "$TOKENS")" ]]; then
    rm -f "$LINK"; ln -s "$TOKENS" "$LINK"
  fi
else
  ln -s "$TOKENS" "$LINK"
fi
echo "symlink: $LINK -> $(readlink -f "$LINK")"

# ckpts/ is local by default; override with ZWT_CKPT_DIR if you want a
# specific location (mounted volume, etc.).
if [[ -n "${ZWT_CKPT_DIR:-}" ]]; then
  mkdir -p "$ZWT_CKPT_DIR"
  if [[ -e ckpts && ! -L ckpts ]]; then
    echo "ckpts/ already exists as a real dir — leaving it alone." >&2
  else
    rm -f ckpts; ln -s "$ZWT_CKPT_DIR" ckpts
  fi
else
  mkdir -p ckpts
fi
echo "ckpts dir: $(readlink -f ckpts)"
echo

# ── dry run on rank 0 only ──
echo "=== Dry-run (rank 0 only, exits before step 1) ==="
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  ./build/zwt_pretrain "$CONF" --dry-run

if [[ "$DO_DRY_ONLY" -eq 1 ]]; then
  echo "dry-run requested; stopping."
  exit 0
fi

# ── launch under tmux ──
mkdir -p logs
LOG="logs/1b_2x_$(date +%Y%m%d_%H%M%S).log"
PROJECT_DIR="$(pwd)"

echo "============================================"
echo "  1B 2-GPU Training Launch"
echo "============================================"
echo "  Config:   $CONF"
echo "  Launcher: $LAUNCHER"
echo "  World:    $WORLD"
echo "  Log:      $LOG"
echo "  tmux:     $TMUX_SESSION"
echo "============================================"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session '$TMUX_SESSION' exists — attach with:" >&2
  echo "  tmux attach -t $TMUX_SESSION" >&2
  exit 1
fi

# rank 0's stderr goes through launch_ddp.sh (which tee's to logs/zwt_ddp_r0.log
# AND to its own stdout), so this tmux window's tee captures the rank 0
# stream into $LOG. Other ranks land in logs/zwt_ddp_r1.log etc.
tmux new-session -d -s "$TMUX_SESSION" -n "training" \
  "bash -c 'cd $PROJECT_DIR && stdbuf -oL -eL bash $LAUNCHER $CONF $WORLD 2>&1 | tee $LOG; echo EXIT_CODE=\${PIPESTATUS[0]} >> $LOG; echo; echo Training ended. Press any key.; read -n1'"

if [[ "$DO_NO_ALERTS" -eq 0 ]]; then
  tmux new-window -t "$TMUX_SESSION" -n "monitor" \
    "bash $PROJECT_DIR/zwt/scripts/launch_1b_2xh100.sh --_monitor=$LOG"
  send_alert "[$LABEL] Started on $HOSTNAME" \
"Config: $CONF
World:  $WORLD
Host:   $HOSTNAME
Time:   $(date)
Tokens: $TOKENS
Log:    $LOG
tmux:   $TMUX_SESSION"
fi

echo
echo "training started in tmux."
echo "attach:   tmux attach -t $TMUX_SESSION      (detach: Ctrl-b then d)"
echo "tail r0:  tail -f logs/zwt_ddp_r0.log"
echo "tail r1:  tail -f logs/zwt_ddp_r1.log"
echo "kill:     tmux kill-session -t $TMUX_SESSION"
