/**
 * include/olmo_cpp/backend/paged_attention.hpp
 *
 * Paged attention kernel — fast-inference [9].
 *
 * Replaces the contiguous-K/V attention with a kernel that gathers K/V
 * from a paged store via a per-request page table. This is what unlocks
 * (a) CUDA graphs (stable shapes), (b) batched decoding with mixed
 * lengths, (c) prefix sharing across requests.
 *
 * DRAFT — see kernels/paged_attention.cu. Not yet plugged into the
 * Transformer's attention forward path; that's a separate refactor.
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Single-query paged attention for one decode step.
///
/// Inputs (for layer L, single batch):
///   - q:           [n_q_heads, head_dim]    FP16/BF16/FP32, the new query
///   - k_pool:      [max_pages, page_size, n_kv_heads, head_dim]
///   - v_pool:      same shape as k_pool
///   - page_table:  [n_blocks] int32 — logical block i → physical page
///   - n_tokens:    total cached tokens (logical seq_len, may be < n_blocks*page_size)
///   - sm_scale:    softmax scale (typically 1/sqrt(head_dim))
///
/// Returns:
///   - out:         [n_q_heads, head_dim] same dtype as q
///
/// Currently CUDA-only. CPU fallback returns a reference attention
/// computed via gather + matmul + softmax + matmul. Useful for
/// validation; not optimized.
torch::Tensor paged_attention_decode(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale);

#ifdef OLMO_HAS_CUDA_KERNELS
/// CUDA kernel launcher.
torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale);
#endif

torch::Tensor paged_attention_decode_cpu(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale);

// ── Graph-capture-friendly variant ────────────────────────────────────────
//
// Identical contract to paged_attention_decode, but `n_tokens` is a 0-D
// int32 tensor on the same device. The CUDA kernel reads `*n_tokens_ptr` at
// launch time, so the captured graph stays correct as the cache grows: the
// caller updates the scalar tensor between replays, the same launched
// kernel reads the new value, no recapture needed.
//
// Required precondition for whole-step CUDA graph capture of the decode
// step. CPU reference exists for validation parity.
torch::Tensor paged_attention_decode_dyn(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,   // 0-D int32 on the same device
    float sm_scale);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor paged_attention_decode_dyn_cuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale);
#endif

torch::Tensor paged_attention_decode_dyn_cpu(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale);

// ── Graph-safe paged K/V write ────────────────────────────────────────────
//
// Writes `k_src`, `v_src` (S consecutive new positions) into the per-layer
// page pool. The destination (page, slot) for each source row is computed
// at runtime from `(n_tokens - S + i) / page_size`, where n_tokens is read
// from a 0-D int32 tensor on the same device. This is the write-side analog
// of paged_attention_decode_dyn: keep n_tokens in a stable-address tensor,
// update it between graph replays, and the captured launch keeps writing to
// the right slot as the cache grows.
//
//   k_src       : [S, n_kv_heads, head_dim]  (or its float32 view)
//   v_src       : same shape
//   k_pool      : [max_pages, page_size, n_kv_heads, head_dim] — written in place
//   v_pool      : same — written in place
//   page_table  : [>= ceil(n_tokens / page_size)] int32 on the same device
//   n_tokens    : 0-D int32 — the seq_len AFTER this write
//
// Mutates k_pool / v_pool. Returns nothing meaningful (just for chainability
// the existing torch wrappers return void).
void paged_kv_write_dyn(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens);

#ifdef OLMO_HAS_CUDA_KERNELS
void paged_kv_write_dyn_cuda(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens);
#endif

void paged_kv_write_dyn_cpu(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens);

// ── INT4-KV variant of paged_attention_decode (item U follow-on) ──────────
//
// Same contract as paged_attention_decode_dyn but k_pool / v_pool are
// int4-packed:
//   k_pool : [max_pages, page_size, n_kv_heads, head_dim/2]  uint8
//   v_pool : [max_pages, page_size, n_kv_heads, head_dim/2]  uint8
//   k_scales / v_scales : [max_pages, page_size, n_kv_heads] fp16
// The kernel dequantizes each K/V on the fly inside the dot-product
// accumulation. 4× memory savings on the cache, ~1% perplexity hit
// under sane block sizes.
torch::Tensor paged_attention_decode_int4(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor paged_attention_decode_int4_cuda(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale);
#endif

torch::Tensor paged_attention_decode_int4_cpu(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale);

}  // namespace olmo_cpp
