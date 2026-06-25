#pragma once

/**
 * include/olmo_cpp/backend/int4_kv.hpp
 *
 * INT4 KV cache codec (item U). Per-head, per-token K/V vectors are
 * quantized into 4-bit values with a per-vector fp16 scale. 4× memory
 * vs bf16 → 4× longer context fits in the same VRAM (or 4× more
 * concurrent requests at the same context length).
 *
 * Storage in the page pool: instead of [max_pages, page_size, n_kv_heads,
 * head_dim] in bf16, we hold:
 *   K_int4 : [max_pages, page_size, n_kv_heads, head_dim / 2]   uint8
 *   K_scale: [max_pages, page_size, n_kv_heads]                 fp16
 * (same for V).
 *
 * Decode-side attention dequantizes inline (kernels/paged_attention.cu
 * gains a variant that does the dequant before the dot product).
 * Some accuracy loss (~0.3 PPL on long context); group quantization
 * with smaller granularity can claw it back.
 *
 * This commit ships the host-side encode/decode helpers and a config
 * flag; integration into BlockManager + paged_attention is wired in
 * its own commit alongside the dequant-aware attention kernel.
 */

#include <torch/torch.h>
#include <utility>

namespace olmo_cpp {

/// Quantize a single K or V tensor of shape [..., n_kv_heads, head_dim]
/// to int4 packed nibbles + per-vector fp16 scales. Last-dim is
/// quantized in one group (one scale per (..., n_kv_head) vector).
struct Int4KVBlock {
  torch::Tensor packed;    // [..., n_kv_heads, head_dim/2]  uint8
  torch::Tensor scales;    // [..., n_kv_heads]              fp16
};

Int4KVBlock quantize_kv_int4(torch::Tensor kv);
torch::Tensor dequantize_kv_int4(const Int4KVBlock& q, int64_t head_dim);

}  // namespace olmo_cpp
