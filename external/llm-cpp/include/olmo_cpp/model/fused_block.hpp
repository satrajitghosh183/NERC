#pragma once

/**
 * include/olmo_cpp/model/fused_block.hpp
 *
 * Fused per-layer transformer block. Differs from
 * ReorderedNormTransformerBlock in two ways: (1) it owns a FusedAttention
 * (single QKV GEMM) and (2) its FeedForward is constructed with the
 * fused gate+up projection enabled. Forward also brackets the body in
 * `backend.begin_scope()`/`end_scope()` so scratch tensors flow through
 * the arena allocator and norm+residual fuses into a single op.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig
 *   - olmo_cpp/model/fused_attention.hpp: FusedAttention sublayer
 *   - olmo_cpp/model/feed_forward.hpp: FeedForward (fused gate+up enabled)
 *   - olmo_cpp/model/kv_cache.hpp: LayerKVCache passthrough
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm with forward_add (norm+resid)
 *   - olmo_cpp/model/rope.hpp: RoPEBuffers passthrough
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/fused_transformer.cpp: FusedTransformerImpl pushes one
 *     FusedTransformerBlock per layer into its ModuleList
 *
 * --- Role in training pipeline ---
 *   Per-layer compute unit of the optimized model. Each forward through
 *   FusedTransformer dispatches n_layers calls to this block's forward,
 *   sandwiched between embedding and LM head.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/fused_attention.hpp"
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/rope.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Fused transformer block with optimized attention.
///
/// Optimizations over ReorderedNormTransformerBlock:
/// 1. Uses FusedAttention (fused QKV projection, efficient GQA)
/// 2. Fused gate_up in feed-forward (already in FeedForward with flag)
/// 3. Residual connections are written in-place where safe
/// 4. Arena allocator scopes for zero-copy scratch memory
///
/// Architecture: h = x + norm1(fused_attn(x)), out = h + norm2(ffn(h))
class FusedTransformerBlockImpl : public torch::nn::Module {
 public:
  /// Build the fused attention, fused-gate-up FeedForward, and the two
  /// RMSNorms. block_idx is forwarded to attention for future per-layer
  /// overrides.
  FusedTransformerBlockImpl(const TransformerConfig& cfg, int64_t block_idx);

  /// Forward: same algebra as the reordered-norm block, but uses
  /// RMSNorm::forward_add (fused norm+residual) and the arena allocator.
  torch::Tensor forward(
      torch::Tensor x,
      const RoPEBuffers* rope_bufs = nullptr,
      std::optional<int64_t> start_pos = std::nullopt,
      LayerKVCache* layer_cache = nullptr);

 private:
  /// Fused-QKV attention sublayer.
  FusedAttention attention_;
  /// RMSNorm fused with the post-attention residual add.
  RMSNorm attention_norm_;
  /// RMSNorm fused with the post-FFN residual add.
  RMSNorm feed_forward_norm_;
  /// SwiGLU FFN with fused gate+up projection.
  FeedForward feed_forward_;
};

/// Module holder for FusedTransformerBlockImpl.
TORCH_MODULE(FusedTransformerBlock);

}  // namespace olmo_cpp
