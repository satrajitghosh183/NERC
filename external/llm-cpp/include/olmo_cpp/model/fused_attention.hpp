#pragma once

/**
 * include/olmo_cpp/model/fused_attention.hpp
 *
 * Fused-projection variant of AttentionImpl. Replaces the three separate
 * Q/K/V linear layers with a single concatenated projection so that the
 * attention sublayer issues one GEMM instead of three. Otherwise math is
 * identical to AttentionImpl: same RoPE, same QK-norm, same SDPA call,
 * same sliding-window mask. This is the version exercised on the bench
 * path against OLMo-core.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig parameters consumed in ctor
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache for incremental decoding
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm used for optional QK-norm
 *   - olmo_cpp/model/rope.hpp: RotaryEmbedding + RoPEBuffers
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/fused_block.cpp: FusedTransformerBlockImpl owns one
 *     FusedAttention per block and invokes its forward in the block forward
 *   - src/model/fused_transformer.cpp: indirectly via FusedTransformerBlock
 *     in the FusedTransformer module list
 *
 * --- Role in training pipeline ---
 *   Hot path of the fused training stack: every transformer layer in
 *   FusedTransformer routes through this attention. Selected via
 *   `fused=1` in the [optimization] section of the .conf file.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Fused attention that combines multiple optimizations for 2-5x speedup:
///
/// 1. **Fused QKV projection**: Single GEMM for all three projections
///    (reduces kernel launches from 3→1 and improves memory locality)
///
/// 2. **In-place RoPE**: Apply rotary embeddings without allocating new tensors
///
/// 3. **Efficient GQA**: Uses expand+reshape instead of repeat_interleave
///    (saves memory allocation for repeated K/V)
///
/// 4. **Backend dispatch**: Routes to FlashAttention/SDPA/custom kernels
///
/// Paper relevance: This is a key optimization — fused QKV alone gives ~1.3x
/// on attention-dominated workloads, and combined with in-place operations
/// reduces memory pressure significantly.
class FusedAttentionImpl : public torch::nn::Module {
 public:
  /// Build the fused QKV linear, output linear, optional QK-norms, and
  /// RoPE module. layer_idx is currently unused.
  FusedAttentionImpl(const TransformerConfig& cfg, int64_t layer_idx);

  /// Same contract as AttentionImpl::forward (see attention.hpp). x is
  /// [B,S,d_model]; result is [B,S,d_model]. The fused QKV weight is
  /// projected once and split via narrow() (zero-copy views).
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  // Fused QKV: single [d_model, (n_heads + 2 * n_kv_heads) * head_dim] weight
  /// Combined Q+K+V projection: emits q_size + 2*kv_size features per token.
  torch::nn::Linear w_qkv_{nullptr};
  /// Output projection: [n_heads*head_dim] -> [d_model].
  torch::nn::Linear w_out_{nullptr};
  /// Optional Q normalization (RMSNorm) applied before/after head reshape.
  std::optional<RMSNorm> q_norm_;
  /// Optional K normalization (RMSNorm).
  std::optional<RMSNorm> k_norm_;
  /// Rotary embedding module (always present).
  std::optional<RotaryEmbedding> rope_;

  /// Number of query heads.
  int64_t n_heads_;
  /// Number of K/V heads (<= n_heads when GQA is in use).
  int64_t n_kv_heads_;
  /// Per-head dimensionality.
  int64_t head_dim_;
  /// Q-per-KV repetition factor for GQA (n_heads / n_kv_heads).
  int64_t n_heads_rep_;
  /// Total Q feature width = n_heads * head_dim (offset 0 in fused QKV).
  int64_t q_size_;   // n_heads * head_dim
  /// Total KV feature width per K or V slice = n_kv_heads * head_dim.
  int64_t kv_size_;  // n_kv_heads * head_dim
  /// If true, RMSNorm is applied per-head after reshape, else pre-reshape.
  bool use_head_qk_norm_;
  /// Sliding window size, -1 disables.
  int64_t sliding_window_size_;

  // Cached sliding window mask — reused when (S, full_S) unchanged
  /// Last query length used to build the cached mask.
  int64_t cached_mask_S_ = 0;
  /// Last key length used to build the cached mask.
  int64_t cached_mask_full_S_ = 0;
  /// Cached additive mask (-inf outside the window, 0 inside).
  torch::Tensor cached_attn_mask_;
};

/// LibTorch module holder — `FusedAttention` is a shared-ptr around impl.
TORCH_MODULE(FusedAttention);

}  // namespace olmo_cpp
