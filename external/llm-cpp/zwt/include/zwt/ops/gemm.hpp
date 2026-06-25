#pragma once

#include "zwt/core/tensor.hpp"

namespace zwt::ops {

// Row-major GEMM: C = alpha * A[.T?] @ B[.T?] + beta * C.
// Shapes are [M, K] x [K, N] -> [M, N] before transposition flags.
//
// On CUDA, this calls cuBLAS(Lt) with bf16/f16 tensor-core paths. On CPU,
// a naïve openmp loop (test-only — do not train on CPU path in anger).
// No autograd — callers compute the backward by calling gemm again with
// the appropriate transpositions.
void gemm(const Tensor& a, bool transa,
          const Tensor& b, bool transb,
          Tensor& c,
          float alpha = 1.0f,
          float beta  = 0.0f);

// Batched variant: A is [B, M, K], B is [B, K, N], C is [B, M, N].
// Uses cuBLAS strided-batched on GPU. Stride is implied by contiguous row-major.
void gemm_batched(const Tensor& a, bool transa,
                  const Tensor& b, bool transb,
                  Tensor& c,
                  float alpha = 1.0f,
                  float beta  = 0.0f);

// One-time global init. Safe to call from multiple threads; first caller wins.
void gemm_init();

// Re-read ZWT_DISABLE_WGMMA from the environment. The dispatch caches this
// flag at first use to avoid a per-call getenv (which takes a global mutex
// in glibc). The bench tool A/Bs cuBLAS vs WGMMA in-process by toggling the
// env var and calling this between phases. Production code never needs it.
void reset_wgmma_disable_cache();

}  // namespace zwt::ops
