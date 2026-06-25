/**
 * include/olmo_cpp/backend/fused_lm_head_sample.hpp
 *
 * Fused LM-head + Gumbel-max sampler — fast-inference roadmap [6].
 *
 * Replaces the standard "compute logits via cuBLAS GEMV → softmax → sample"
 * pipeline with a single kernel that:
 *
 *   for each row i in W_U:
 *     l_i  = dot(W_U[i], hidden)            // the GEMV row
 *     g_i  = -log(-log(uniform(seed, pos, i)))   // Gumbel(0,1) via Philox
 *     s_i  = l_i / T + g_i
 *   token = argmax_i s_i
 *
 * The Gumbel-max identity:  argmax(logits/T + Gumbel) ≡ sample(softmax(logits/T))
 *
 * Eliminates the [V] logits write to HBM, the multi-kernel
 * softmax/topk/sort chain, AND the D->H copy of [V] floats. Output is
 * a single int64 token id.
 *
 * Phase 1: greedy + temperature only. No top-k, no top-p, no rep
 * penalty. The CPU fallback (sample_logits in chat.cpp) handles those
 * paths — switch in via a feature flag.
 *
 * Determinism: samples are deterministic given (seed, position, vocab_idx).
 * NOT compatible with std::mt19937 — same seed produces different output.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace olmo_cpp {

/// Fused LM-head GEMV + Gumbel-max sampling.
///
/// Inputs:
///   - hidden:      [H] FP32 or BF16. The last hidden state.
///   - W_U:         [V, H] FP32 or BF16. The unembedding matrix (LM head weight).
///   - temperature: float, > 0. Scales logits before adding Gumbel noise.
///   - seed:        uint64. RNG seed for the run.
///   - position:    uint32. Token position in the sequence — drives RNG sequence.
///
/// Returns: int64 sampled token id.
///
/// Currently CUDA-only. CPU fallback returns -1 (caller should use the
/// non-fused path on CPU/MPS).
///   - rep_tokens:  previously-emitted ids. Each is penalized once, inside the
///                  kernel, BEFORE temperature + Gumbel — so no [V] logits
///                  tensor is materialized even with rep-penalty on.
///   - rep_penalty: 1.0 → no penalty (the rep_tokens are ignored).
int64_t fused_lm_head_sample(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens = {},
    double rep_penalty = 1.0);

#ifdef OLMO_HAS_CUDA_KERNELS
/// CUDA implementation. Caller must guarantee both tensors are on CUDA.
/// Defined in kernels/fused_lm_head_sample.cu.
int64_t fused_lm_head_sample_cuda(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens = {},
    double rep_penalty = 1.0);
#endif

/// CPU reference. Slow but correct. Used by unit tests.
int64_t fused_lm_head_sample_cpu(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens = {},
    double rep_penalty = 1.0);

}  // namespace olmo_cpp
