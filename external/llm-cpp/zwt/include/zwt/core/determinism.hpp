#pragma once

// Global determinism mode. When enabled, kernels and cuBLAS calls take a
// reproducible path: no atomics for reductions, fixed cuBLAS workspace,
// no TF32 for F32 matmuls. The trade-off is throughput — the determinism
// flag is for reproducibility experiments and sanity checks, not for
// production training.

namespace zwt {

bool is_deterministic();
void set_deterministic(bool enabled);

// Apply process-wide settings that must be in place before the first
// cuBLAS handle is created (CUBLAS_WORKSPACE_CONFIG env var). Safe to call
// multiple times; a no-op after the env var has been fixed. Call from main
// before any op dispatch if cfg.deterministic is true.
void init_determinism_env();

}  // namespace zwt
