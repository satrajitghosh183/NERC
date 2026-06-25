#pragma once

/**
 * include/olmo_cpp/backend/fused_qkv_rope.hpp
 *
 * Fused Q/K/V projection + reshape + RoPE — item G.
 *
 * Replaces the current 5-6 launch sequence (3 separate Linears, 3 view
 * reshapes that get materialized through contiguous() on some paths,
 * 1 RoPE apply) with one CUDA kernel that takes the residual stream x
 * and the concatenated QKV weight, and writes Q, K, V directly in the
 * [B, n_heads, S, head_dim] layout with RoPE already applied to Q and K.
 *
 * Why Python doesn't have this: PyTorch eager has F.linear and an
 * rope() helper but not a fused QKV+RoPE op — every transformer impl
 * does it as discrete ATen calls.
 *
 * Forward:
 *   x        : [B, S, d]                       (bf16 / fp32)
 *   w_qkv    : [(n_q + 2*n_kv)*head_dim, d]    (bf16)  -- concatenated [Wq;Wk;Wv]
 *   cos, sin : [S, head_dim/2]                 (fp32)  -- RoPE buffers
 * Outputs:
 *   q : [B, n_q_heads, S, head_dim]
 *   k : [B, n_kv_heads, S, head_dim]
 *   v : [B, n_kv_heads, S, head_dim]
 *
 * Backward kernel: takes grad_q, grad_k, grad_v (each in the model's
 * head-major layout) and produces grad_x [B, S, d] and grad_w_qkv
 * [(n_q+2*n_kv)*head_dim, d]. Unrotated by inverse RoPE on grad_q and
 * grad_k before the matmul.
 *
 * Current scope: forward kernel + CPU reference. The backward kernel
 * lands in a follow-on; until then, autograd recompute is acceptable
 * for the speculative + inference paths (training path keeps the
 * existing separate-Linear+RoPE chain when grad tracking is on).
 */

#include <torch/torch.h>
#include <tuple>

namespace olmo_cpp {

/// Returns (q, k, v) tensors in head-major shape with RoPE applied
/// to q and k. cos/sin must be precomputed for the current sequence
/// positions (typically the rope buffers at position start_pos+i).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope(torch::Tensor x,
               torch::Tensor w_qkv,
               torch::Tensor cos,
               torch::Tensor sin,
               int64_t n_q_heads,
               int64_t n_kv_heads,
               int64_t head_dim);

#ifdef OLMO_HAS_CUDA_KERNELS
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_cuda(torch::Tensor x,
                    torch::Tensor w_qkv,
                    torch::Tensor cos,
                    torch::Tensor sin,
                    int64_t n_q_heads,
                    int64_t n_kv_heads,
                    int64_t head_dim);

/// Tensor-core (WMMA) variant — preferred on sm_80+ for bf16 inputs when
/// N=B*S, F=(n_q+2*n_kv)*head_dim, and d are all multiples of 16, and
/// the shmem footprint (16 * F * 2B) fits in the SM. Falls back to
/// fused_qkv_rope_cuda otherwise.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_wmma_cuda(torch::Tensor x,
                           torch::Tensor w_qkv,
                           torch::Tensor cos,
                           torch::Tensor sin,
                           int64_t n_q_heads,
                           int64_t n_kv_heads,
                           int64_t head_dim);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_cpu(torch::Tensor x,
                   torch::Tensor w_qkv,
                   torch::Tensor cos,
                   torch::Tensor sin,
                   int64_t n_q_heads,
                   int64_t n_kv_heads,
                   int64_t head_dim);

/// Autograd-aware forward. Calls into fused_qkv_rope (CUDA or CPU) and
/// records the graph so backward through (x, w_qkv) works. Use from
/// training call sites; inference (no_grad) can call fused_qkv_rope
/// directly without paying the autograd graph cost.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_autograd(torch::Tensor x,
                          torch::Tensor w_qkv,
                          torch::Tensor cos,
                          torch::Tensor sin,
                          int64_t n_q_heads,
                          int64_t n_kv_heads,
                          int64_t head_dim);

}  // namespace olmo_cpp
