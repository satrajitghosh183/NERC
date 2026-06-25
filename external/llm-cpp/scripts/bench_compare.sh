#!/usr/bin/env bash
# scripts/bench_compare.sh
#
# Fast-inference roadmap item [3]: head-to-head bench of the current
# (fast-inference) build vs a frozen baseline build. Both run the same
# checkpoint, same prompt length, same decode length; numbers are emitted
# as JSON and diffed into a markdown table.
#
# Workflow:
#   1. Builds the baseline at $BASELINE_REF into a git worktree at
#      $BASELINE_WORKTREE (default: ./build_baseline).
#   2. Builds the current branch into ./build (standard).
#   3. Runs `bench_chat` from each on the same args.
#   4. Diffs the two JSONs, prints a table to stdout.
#
# Required env / flags:
#   --checkpoint PATH    .pt checkpoint
#   --config PATH        model config .json
#   --vocab PATH         GPT-2 vocab.json
#   --merges PATH        GPT-2 merges.txt
#   --device {cuda|mps|cpu}
#   --baseline-ref REF   git ref to freeze as baseline (default: parent of fast-inference)
#   --prompt-len N       (default: 128)
#   --decode-len N       (default: 256)
#   --batch N            (default: 1)
#   --warmup N           (default: 3)
#   --iters N            (default: 5)
#   --keep-worktree      do not remove the baseline worktree after run
#
# Output:
#   results/bench_baseline.json
#   results/bench_current.json
#   results/bench_compare.md   (markdown table)
#
# Notes:
#   - Builds are skipped if the binary already exists; pass --rebuild to force.
#   - Greedy decoding only (no RNG variance between runs).

set -euo pipefail

CKPT=""
CFG=""
VOCAB=""
MERGES=""
DEVICE="cuda"
BASELINE_REF=""
PROMPT_LEN=128
DECODE_LEN=256
BATCH=1
WARMUP=3
ITERS=5
KEEP_WORKTREE=0
REBUILD=0
BASELINE_WORKTREE="$(pwd)/build_baseline_wt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --config) CFG="$2"; shift 2;;
    --vocab) VOCAB="$2"; shift 2;;
    --merges) MERGES="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --baseline-ref) BASELINE_REF="$2"; shift 2;;
    --prompt-len) PROMPT_LEN="$2"; shift 2;;
    --decode-len) DECODE_LEN="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --keep-worktree) KEEP_WORKTREE=1; shift;;
    --rebuild) REBUILD=1; shift;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -z "$CKPT"   ]] && { echo "Missing --checkpoint" >&2; exit 1; }
[[ -z "$CFG"    ]] && { echo "Missing --config"     >&2; exit 1; }
[[ -z "$VOCAB"  ]] && { echo "Missing --vocab"      >&2; exit 1; }
[[ -z "$MERGES" ]] && { echo "Missing --merges"     >&2; exit 1; }

# Default baseline: parent commit of fast-inference's first commit on this branch.
# i.e. the merge base with main, or simply the commit before any fast-inference work.
if [[ -z "$BASELINE_REF" ]]; then
  BASELINE_REF=$(git merge-base HEAD main 2>/dev/null \
                 || git rev-parse HEAD~1)
fi

mkdir -p results

CURRENT_BIN="$(pwd)/build/bench_chat"
BASELINE_BIN="$BASELINE_WORKTREE/build/bench_chat"

build_in() {
  local dir="$1"
  pushd "$dir" >/dev/null
  if [[ ! -f build/bench_chat || $REBUILD -eq 1 ]]; then
    if [[ -x scripts/build.sh ]]; then
      ./scripts/build.sh
    else
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
      cmake --build build -j --target bench_chat
    fi
  fi
  popd >/dev/null
}

# 1) Baseline build (in a worktree, so we don't disturb our working copy).
if [[ -d "$BASELINE_WORKTREE" ]]; then
  echo "Baseline worktree exists: $BASELINE_WORKTREE"
else
  echo "Creating baseline worktree at $BASELINE_WORKTREE @ $BASELINE_REF"
  git worktree add "$BASELINE_WORKTREE" "$BASELINE_REF"
fi
# bench_chat may not exist on the baseline branch yet — copy it in if missing.
if [[ ! -f "$BASELINE_WORKTREE/tools/bench_chat.cpp" ]]; then
  echo "Patching bench_chat.cpp into baseline worktree (it didn't ship there)"
  cp tools/bench_chat.cpp "$BASELINE_WORKTREE/tools/bench_chat.cpp"
  # Also patch CMakeLists if needed: append our add_executable block.
  if ! grep -q "add_executable(bench_chat" "$BASELINE_WORKTREE/CMakeLists.txt"; then
    cat <<'CMAKE_PATCH' >> "$BASELINE_WORKTREE/CMakeLists.txt"

# Patched in by bench_compare.sh
add_executable(bench_chat tools/bench_chat.cpp)
target_link_libraries(bench_chat PRIVATE olmo_cpp)
target_include_directories(bench_chat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${JSON_INCLUDE_DIR})
target_compile_options(bench_chat PRIVATE -O3 -march=native)
if(JSON_INCLUDE_DIR)
  target_compile_definitions(bench_chat PRIVATE HAS_NLOHMANN_JSON=1)
endif()
CMAKE_PATCH
  fi
fi
build_in "$BASELINE_WORKTREE"

# 2) Current build.
build_in "$(pwd)"

run_bench() {
  local bin="$1" out="$2" label="$3"
  echo "=== Running $label ==="
  "$bin" \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --vocab-file "$VOCAB" \
    --merges-file "$MERGES" \
    --device "$DEVICE" \
    --prompt-len "$PROMPT_LEN" \
    --decode-len "$DECODE_LEN" \
    --batch "$BATCH" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --output "$out"
}

run_bench "$BASELINE_BIN" results/bench_baseline.json baseline
run_bench "$CURRENT_BIN"  results/bench_current.json  current

# 3) Diff into a markdown table.
python3 - <<'PYEOF'
import json, pathlib
b = json.load(open("results/bench_baseline.json"))
c = json.load(open("results/bench_current.json"))

def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

def speedup(b_v, c_v, lower_is_better=True):
    if c_v == 0: return "—"
    r = b_v / c_v if lower_is_better else c_v / b_v
    return f"{r:.2f}x"

rows = [
    ("TTFT (ms, mean)",        b["ttft_ms_mean"],         c["ttft_ms_mean"],         True),
    ("TPOT (ms, mean)",        b["tpot_ms_mean"],         c["tpot_ms_mean"],         True),
    ("TPOT (ms, p50)",         b["tpot_ms_p50"],          c["tpot_ms_p50"],          True),
    ("TPOT (ms, p99)",         b["tpot_ms_p99"],          c["tpot_ms_p99"],          True),
    ("Throughput (tok/s)",     b["throughput_tok_per_s"], c["throughput_tok_per_s"], False),
]

lines = ["| metric | baseline | current | speedup |", "|---|---|---|---|"]
for name, bv, cv, lower in rows:
    lines.append(f"| {name} | {fmt(bv)} | {fmt(cv)} | {speedup(bv, cv, lower)} |")

header = (f"# bench_compare\n\n"
          f"- device: `{c['device']}`  batch: `{c['batch']}`  "
          f"prompt_len: `{c['prompt_len']}`  decode_len: `{c['decode_len']}`\n"
          f"- iters: `{c['iters']}`  warmup: `{c['warmup']}`\n\n")
md = header + "\n".join(lines) + "\n"
pathlib.Path("results/bench_compare.md").write_text(md)
print(md)
PYEOF

if [[ $KEEP_WORKTREE -eq 0 ]]; then
  echo "Removing baseline worktree (use --keep-worktree to retain)"
  git worktree remove --force "$BASELINE_WORKTREE" || true
fi
