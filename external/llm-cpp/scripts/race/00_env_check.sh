#!/usr/bin/env bash
# scripts/race/00_env_check.sh
#
# Pre-flight: verify the box is ready before the race. Enforces the
# Blackwell (sm_120 / 5060 Ti) toolchain floor: CUDA Toolkit ≥ 12.8 and
# a PyTorch wheel that actually ships sm_120 kernels. Bails on any gap
# with a copy-pasteable fix.

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

say()  { printf "\033[1;36m[env]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m  ✓\033[0m  %s\n" "$*"; }
warn() { printf "\033[1;33m  ⚠\033[0m  %s\n" "$*"; }
fail() { printf "\033[1;31m  ✗\033[0m  %s\n" "$*"; exit 1; }

# ver_ge A B  → success (0) iff A >= B under version sort.
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

say "── GPU ──"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not on PATH (NVIDIA driver not installed?)"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$gpu_count" -ge 1 ]] || fail "no CUDA device detected"
ok "$gpu_count CUDA device(s) visible"

# Compute capability as an integer, e.g. 12.0 → 120, 8.9 → 89.
cc_dotted=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
cc=$(printf '%s' "$cc_dotted" | tr -d '.')
[[ "$cc" -ge 80 ]] || fail "compute capability sm_$cc < sm_80 — kernels will not run"
ok "compute capability sm_${cc} (GPU reports $cc_dotted)"

# Map GPU arch → minimum CUDA toolkit that knows how to target it.
#   sm_120 (Blackwell consumer, 5060 Ti) → CUDA 12.8
#   sm_100 (Blackwell datacenter)        → CUDA 12.8
#   sm_90  (Hopper)                      → CUDA 12.0
#   sm_89/86/80 (Ada/Ampere)            → CUDA 11.8
if   [[ "$cc" -ge 100 ]]; then CUDA_FLOOR="12.8"
elif [[ "$cc" -ge 90  ]]; then CUDA_FLOOR="12.0"
else                            CUDA_FLOOR="11.8"
fi
say "→ this GPU (sm_${cc}) requires CUDA Toolkit ≥ ${CUDA_FLOOR}"

say "── CUDA toolkit (nvcc) ──"
command -v nvcc >/dev/null || fail "nvcc not on PATH.
       Install CUDA Toolkit ≥ ${CUDA_FLOOR} and:
         export PATH=/usr/local/cuda-${CUDA_FLOOR}/bin:\$PATH
         export CUDAToolkit_ROOT=/usr/local/cuda-${CUDA_FLOOR}"
nvcc_ver=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}')
[[ -n "$nvcc_ver" ]] || fail "could not parse nvcc version"
if ver_ge "$nvcc_ver" "$CUDA_FLOOR"; then
  ok "nvcc $nvcc_ver  (≥ ${CUDA_FLOOR} ✓)"
else
  fail "nvcc $nvcc_ver is too old for sm_${cc}. Need ≥ ${CUDA_FLOOR}.
       sm_120 (Blackwell) support landed in CUDA 12.8 — older nvcc
       rejects 'sm_120' at the first .cu file.
       Install CUDA ${CUDA_FLOOR}+ and point PATH/CUDAToolkit_ROOT at it."
fi

say "── Build deps ──"
for tool in cmake make gcc g++ git python3 pip3; do
  command -v "$tool" >/dev/null || fail "missing: $tool"
done
cmake_ver=$(cmake --version | head -1 | awk '{print $3}')
ver_ge "$cmake_ver" "3.18" || fail "cmake $cmake_ver < 3.18"
ok "cmake $cmake_ver, gcc $(gcc -dumpversion), git $(git --version | awk '{print $3}')"

say "── PyTorch (must ship sm_${cc} kernels) ──"
python3 -c "import torch" 2>/dev/null || fail "torch not installed. For Blackwell:
       pip3 install torch --index-url https://download.pytorch.org/whl/cu128"

# One python call returns: version | cuda | available | arch_list | dev_cc
read -r TORCH_VER TORCH_CUDA TORCH_AVAIL TORCH_ARCHES DEV_CC < <(python3 - <<'PY'
import torch
arches = ",".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else ""
try:
    cap = torch.cuda.get_device_capability()
    dev_cc = f"{cap[0]}{cap[1]}"
except Exception:
    dev_cc = "NA"
print(torch.__version__, torch.version.cuda or "none",
      torch.cuda.is_available(), arches or "none", dev_cc)
PY
)
ok "torch $TORCH_VER (built against CUDA $TORCH_CUDA)"
[[ "$TORCH_AVAIL" == "True" ]] || fail "torch.cuda.is_available() == False — driver/runtime mismatch.
       Reinstall a matching wheel:  pip3 install torch --index-url https://download.pytorch.org/whl/cu128"
ok "torch.cuda.is_available() == True"

# The definitive Blackwell check: torch's compiled arch list must include
# this GPU's sm_NN. If it doesn't, ATen ops fall back to PTX-JIT (slow) or
# fail outright. cu128 wheels include sm_120; cu121 wheels do not.
say "  torch arch list: $TORCH_ARCHES"
if printf '%s' "$TORCH_ARCHES" | grep -q "sm_${cc}"; then
  ok "torch ships sm_${cc} kernels"
elif printf '%s' "$TORCH_ARCHES" | grep -q "compute_${cc}"; then
  warn "torch has compute_${cc} (PTX-JIT) but not native sm_${cc} — will work, slower first-run"
else
  fail "torch wheel has no sm_${cc} support (arch list: $TORCH_ARCHES).
       For the 5060 Ti you need a Blackwell wheel:
         pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
       (PyTorch ≥ 2.7 cu128 ships sm_120.)"
fi

say "── Disk space ──"
free_gb=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9' || echo 999)
[[ "$free_gb" -ge 10 ]] || fail "need ≥ 10 GB free, have ${free_gb} GB"
ok "${free_gb} GB free"

say "── Repo state ──"
# CMakeLists.txt is the repo-root sentinel (CLAUDE.md is gitignored, so it
# doesn't exist in a fresh clone). The conf check proves the race kit is here.
[[ -f CMakeLists.txt && -f scripts/race/configs/race_250m_cpp.conf ]] \
  || fail "run from repo root; race configs missing"
ok "repo layout OK ($(git rev-parse --short HEAD 2>/dev/null || echo 'no-git'))"

say "── OLMo-core (Python reference) ──"
if python3 -c "import olmo_core" 2>/dev/null; then
  ok "olmo_core importable ($(python3 -c 'import olmo_core; print(olmo_core.__version__)' 2>/dev/null || echo '?'))"
elif [[ -d olmo-python && -f olmo-python/pyproject.toml ]]; then
  ok "olmo-python/ present (05_train_python.sh will pip install -e it)"
else
  warn "olmo-python/ missing AND olmo_core not importable — Python side will fail"
fi

# Persist the resolved arch so 01_build_cpp.sh targets exactly this GPU.
echo "$cc" > scripts/race/results/.gpu_arch 2>/dev/null || true
mkdir -p scripts/race/results && echo "$cc" > scripts/race/results/.gpu_arch

printf "\n\033[1;32mALL ENV CHECKS PASSED — toolchain is Blackwell-ready (sm_%s)\033[0m\n" "$cc"
