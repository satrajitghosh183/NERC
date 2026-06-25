/**
 * include/olmo_cpp/backend/sparse_attn.hpp
 *
 * NSA-style content-selected sparse attention. Phase-1 reference for the
 * SubQ head-to-head (see scripts/bench_subq.sh, tools/bench_attn.cpp).
 *
 * Algorithm (decode):
 *   1. Split the KV cache of length T into n_blocks = T / block_size
 *      contiguous chunks along the sequence dimension.
 *   2. For each (batch, kv_head), score every block by the dot-product of
 *      the query against the block-mean K vector. Cheap, O(n_blocks * D).
 *   3. Pick the top-k blocks per (batch, kv_head). Selection is shared
 *      across all query heads in the GQA group — the same K/V positions
 *      are read by every group member, which is what lets the kernel be
 *      cache-efficient.
 *   4. Run exact dense attention over the gathered top-k * block_size KV
 *      positions only.
 *
 * What this is NOT:
 *   - It is not the optimized fused selector+SDPA CUDA kernel — that's
 *     Phase-1 step 3 in the SubQ plan and lives elsewhere.
 *   - Selection is not differentiable here; the training-side soft-selector
 *     surrogate lives in zwt/ and is added in Phase 2.
 *   - Prefill (full causal sparse attention over all queries) is not
 *     implemented yet.
 *
 * Why decode-first: SubQ's headline 52x is dominated by long-context
 * decode/prefill where the KV is huge. A correct decode reference unblocks
 * the bench harness; the prefill kernel is a follow-up.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Single-position content-selected attention.
///
/// Inputs:
///   q:          [B, H,   1, D]   single query per batch element.
///   k:          [B, Hkv, T, D]   KV cache. T must be divisible by block_size.
///   v:          [B, Hkv, T, D]   value cache, same shape as k.
///   sm_scale:   softmax scale, typically 1/sqrt(D).
///   block_size: number of contiguous KV positions per scoring block.
///   top_k:      number of blocks to keep per (B, kv_head). Clamped to n_blocks.
///
/// Output: [B, H, 1, D] — same shape as a dense SDPA call.
///
/// Dispatch: when q is on CUDA and OLMO_HAS_CUDA_KERNELS is set, this calls
/// the fused kernel below; otherwise it runs the ATen reference (which also
/// works on CUDA but issues 4 separate kernels).
torch::Tensor sparse_attn_decode(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& v,
                                 float sm_scale,
                                 int64_t block_size,
                                 int64_t top_k);

#ifdef OLMO_HAS_CUDA_KERNELS
/// Fused CUDA kernel — one launch does block-mean scoring + top-k +
/// online-softmax SDPA. Block per (B, query_head). The selection is
/// implicitly shared across the GQA group by loading all sibling-head Q
/// vectors and averaging before scoring (matches the CPU reference).
torch::Tensor sparse_attn_decode_cuda(const torch::Tensor& q,
                                      const torch::Tensor& k,
                                      const torch::Tensor& v,
                                      float sm_scale,
                                      int64_t block_size,
                                      int64_t top_k);
#endif

/// Full-sequence (training/prefill) sparse attention. For each query
/// position, top-k blocks of *causally visible* past KV are selected
/// using the same block-mean scoring as decode. Selection sharing across
/// the GQA group matches sparse_attn_decode.
///
/// Differentiable forward: every op is ATen (matmul/topk/gather/softmax),
/// so autograd handles backward via straight-through (topk indices have
/// no gradient; the gradient flows through gather → matmul → q/k/v).
///
/// Inputs:
///   q:          [B, H,   S, D]
///   k:          [B, Hkv, S, D]   self-attn shape — k is the same length
///                                 as q, padded up to a multiple of block_size.
///   v:          [B, Hkv, S, D]
///   sm_scale:   1/sqrt(D)
///   block_size, top_k: same meaning as decode.
///
/// Output: [B, H, S, D]
torch::Tensor sparse_attn_prefill(const torch::Tensor& q,
                                  const torch::Tensor& k,
                                  const torch::Tensor& v,
                                  float sm_scale,
                                  int64_t block_size,
                                  int64_t top_k);

}  // namespace olmo_cpp
