/**
 * include/olmo_cpp/backend/flash_decode.hpp
 *
 * FlashAttention-2 decode kernel — fast-inference [12].
 *
 * Single-query decode attention. Differs from training attention because
 * Q is a single position vs many K/V positions. Uses online softmax to
 * keep memory pressure to O(d_head) instead of O(seq_len).
 *
 * DRAFT. Plain (non-paged) variant. The paged variant lives in
 * paged_attention.cu — both share the same online-softmax pattern but
 * differ in how they index K/V (contiguous vs page table).
 *
 * Phase-1 scope:
 *   - One query, one batch element (extend later)
 *   - GQA via head-mapping
 *   - Online softmax accumulator in shared mem
 *   - FP32 compute, BF16/FP16 weights via cast on read
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Plain (non-paged) decode attention.
/// Inputs:
///   - q: [n_q_heads, head_dim]
///   - k: [n_tokens, n_kv_heads, head_dim]
///   - v: [n_tokens, n_kv_heads, head_dim]
///   - sm_scale: usually 1/sqrt(head_dim)
/// Output: [n_q_heads, head_dim].
torch::Tensor flash_decode(torch::Tensor q,
                           torch::Tensor k,
                           torch::Tensor v,
                           float sm_scale);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor flash_decode_cuda(torch::Tensor q,
                                torch::Tensor k,
                                torch::Tensor v,
                                float sm_scale);
#endif

torch::Tensor flash_decode_cpu(torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v,
                               float sm_scale);

}  // namespace olmo_cpp
