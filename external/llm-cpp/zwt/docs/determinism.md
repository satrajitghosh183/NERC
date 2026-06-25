# Determinism mode

Set `deterministic = 1` in the `[runtime]` section of a train config to enable.
When on, zwt aims for bit-identical training steps across runs with the same
seed on the same hardware.

## What's deterministic today

- **cuBLAS GEMMs.** `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before the first
  handle is created, and `cublasGemmEx` / `cublasGemmStridedBatchedEx` are
  issued with `CUBLAS_GEMM_DEFAULT` instead of the heuristic-picked
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`. This keeps reduction order stable across
  launches at a small throughput cost.
- **RNG-seeded paths.** Every RNG site takes a seed from config — parameter
  init salts, data shuffle seed. Re-running with identical `init_seed` and
  `data_seed` reproduces the weights and the minibatch order.

## Known non-deterministic paths (followups)

These currently use fp32 `atomicAdd`, which is order-non-deterministic and
therefore produces a different bit pattern even for identical inputs:

- `k_rmsnorm_bwd_bf16` → `grad_weight` accumulation (zwt/src/ops/norm.cu).
  Fix: split into two kernels — a row-parallel `grad_x`-only pass, plus a
  column-parallel reduction pass that reads x, grad_y, rstd and emits
  `grad_weight` via a block-deterministic reduction with no atomics.
- `k_softmax_xent_fused_bf16` → loss and token-count accumulation
  (zwt/src/ops/xent.cu). Fix: two-stage — each row writes its partial loss
  and `(target != ignore)` count into a scratch buffer; a second kernel
  reduces the buffer deterministically in a single block.

Both variants are straightforward but require GPU validation before shipping,
which is out of scope for the current host-only CPU build.

## What determinism costs

- cuBLAS: typically 1–5% slower on bf16 GEMMs (algo choice frozen).
- Deterministic reductions: 5–15% slower than atomic-based kernels, once the
  kernels land.
- Not a free toggle — only enable for reproducibility experiments, paper
  rebuttals, or debugging divergent runs.

## What it does **not** claim

- Not cross-hardware deterministic. Different GPU generations or CUDA
  versions may produce different numerics.
- Not deterministic across different world sizes (NCCL all-reduce order
  depends on ring topology). When TP/DDP land, expect a rank-count tag in
  any reproducibility claim.
