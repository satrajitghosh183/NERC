/**
 * include/olmo_cpp/backend/lm_head_gemv.hpp
 *
 * Custom LM-head GEMV — fast-inference [11].
 *
 * cuBLAS GEMM is tuned for square matmuls; the LM head's [1, H] · [H, V]
 * GEMV is a degenerate case. A hand-tuned GEMV with split-K reduction
 * across the V dimension typically beats cuBLAS by ~20-40% on H100 at
 * V≈50K, H≈2048.
 *
 * Variant of the dot-product loop already present in
 * fused_lm_head_sample.cu, but stops at logits — no Gumbel, no argmax.
 * Useful when the caller still wants the full logits vector (e.g. for
 * speculative verification, top-k/top-p that the fused sampler doesn't
 * support yet).
 *
 * DRAFT — needs benchmarking against cuBLAS to confirm the win.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Compute logits = W_U @ hidden for a single query position.
/// Inputs:
///   - hidden: [H] FP32 or BF16
///   - W_U:    [V, H] FP32 or BF16
/// Output: [V] FP32 logits.
torch::Tensor lm_head_gemv(torch::Tensor hidden, torch::Tensor W_U);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor lm_head_gemv_cuda(torch::Tensor hidden, torch::Tensor W_U);
#endif

torch::Tensor lm_head_gemv_cpu(torch::Tensor hidden, torch::Tensor W_U);

}  // namespace olmo_cpp
