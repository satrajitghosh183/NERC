#pragma once

/**
 * include/olmo_cpp/model/fused_transformer.hpp
 *
 * Optimized variant of TransformerImpl that swaps in FusedTransformerBlock
 * (fused QKV + fused gate+up FFN) and uses RMSNorm::forward_add to fuse
 * norm with the residual add. Same architecture, same weights layout, same
 * outputs as TransformerImpl — only the kernel-issuing pattern changes.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: TransformerConfig
 *   - olmo_cpp/model/fused_block.hpp: per-layer FusedTransformerBlock
 *   - olmo_cpp/model/kv_cache.hpp: KVCache for inference
 *   - olmo_cpp/model/layer_norm.hpp: RMSNorm for embedding norm
 *   - olmo_cpp/model/lm_head.hpp: LMHead
 *   - olmo_cpp/model/mtp_head.hpp: MTPHead for multi-token prediction
 *   - olmo_cpp/nn/multi_res_embedding.hpp: optional DC-MRE embed
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: builds FusedTransformer when [optimization].fused == 1
 *   - src/train.cpp: training loop (operates on a generic shared_ptr to
 *     either Transformer or FusedTransformer)
 *
 * --- Role in training pipeline ---
 *   The fused training stack head module — the bench path against
 *   OLMo-core uses this. forward() returns either loss (when labels are
 *   provided) or logits.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/fused_block.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include "olmo_cpp/model/layer_norm.hpp"
#include "olmo_cpp/model/lm_head.hpp"
#include "olmo_cpp/model/mtp_head.hpp"
#include "olmo_cpp/nn/multi_res_embedding.hpp"
#include <torch/torch.h>
#include <optional>
#include <vector>

namespace olmo_cpp {

/// Optimized transformer using fused operations.
///
/// Differences from TransformerImpl:
/// 1. Uses FusedTransformerBlock (fused QKV + fused gate_up)
/// 2. Supports torch::jit::optimize_for_inference() on the forward path
/// 3. Compatible with CUDA Graphs when input shapes are static
///
/// The model is architecturally identical (same weights, same outputs)
/// but executes ~1.5-3x faster due to reduced kernel launches and
/// improved memory access patterns.
class FusedTransformerImpl : public torch::nn::Module {
 public:
  /// Construct embedding, blocks, LM head and MTP heads from config.
  /// init_weights() must be called before training.
  explicit FusedTransformerImpl(const TransformerConfig& cfg);

  /// Forward: returns CE loss when labels are provided (incl. MTP aux),
  /// otherwise returns logits of shape [B, S, vocab_size].
  torch::Tensor forward(
      torch::Tensor input_ids,
      c10::optional<torch::Tensor> labels = c10::nullopt,
      int64_t ignore_index = -100,
      KVCache* kv_cache = nullptr);

  /// Run the embedding + block stack only; returns hidden states.
  torch::Tensor forward_backbone(
      torch::Tensor input_ids,
      KVCache* kv_cache = nullptr);

  /// Returns MTP head logits for a single position; used by speculative
  /// decoding to draft k tokens after the verified one.
  std::vector<torch::Tensor> forward_mtp_draft(torch::Tensor hidden_state);

  /// Truncated-normal init for all parameters.
  void init_weights(torch::optional<torch::Generator> gen = c10::nullopt);

  // Returns a reference to the cached per-layer RoPE buffers, recomputing them
  // only when seq_len grows past the cache or dtype changes. Returning by const
  // ref avoids a per-forward std::vector<RoPEBuffers> copy (and 2*n_layers
  // tensor refcount ops). The returned reference is stable until the next call
  // that triggers a recompute.
  const std::vector<RoPEBuffers>& get_rope_buffers(
      int64_t seq_len, torch::Device device,
      torch::Dtype dtype = torch::kFloat32);

  /// Number of transformer layers.
  int64_t n_layers() const { return config_.n_layers; }
  /// Number of MTP heads (0 disables MTP).
  int64_t num_mtp_heads() const { return config_.num_mtp_heads; }
  /// Convenience predicate for MTP enabled.
  bool has_mtp() const { return config_.num_mtp_heads > 0; }

  /// Apply LM head to hidden states → logits.
  torch::Tensor apply_lm_head(torch::Tensor hidden_states) { return lm_head_(hidden_states); }

 private:
  // Embedding: either plain or multi-resolution (DC-MRE)
  /// Plain embedding (active when use_multi_res_ is false).
  torch::nn::Embedding embeddings_{nullptr};
  /// Multi-resolution (DC-MRE) embedding (active when use_multi_res_ true).
  MultiResEmbedding multi_res_embed_{nullptr};
  /// Selects between the two embedding paths above.
  bool use_multi_res_ = false;

  /// Pre-block RMSNorm applied to embedding output.
  std::optional<RMSNorm> embedding_norm_;
  /// n_layers FusedTransformerBlock modules.
  torch::nn::ModuleList blocks_;
  /// Final norm + tied linear LM projection.
  LMHead lm_head_;
  /// Optional embedding scale (multiplier applied to embed weights).
  std::optional<double> embed_scale_;
  /// Cached config (used by forward to read MTP weight etc.).
  TransformerConfig config_;

  // Multi-Token Prediction heads
  /// num_mtp_heads MTPHead modules; empty when MTP is disabled.
  torch::nn::ModuleList mtp_heads_;

  // Cached RoPE — recomputed only when seq_len grows or dtype changes
  /// Last allocated RoPE buffer length (in tokens).
  int64_t cached_rope_len_ = 0;
  /// Activation dtype the cached buffers were built for.
  torch::Dtype cached_rope_dtype_ = torch::kFloat32;
  /// Per-layer RoPE buffers (currently shared via tensor refcount).
  std::vector<RoPEBuffers> cached_rope_bufs_;
};

/// Module-holder macro: FusedTransformer is shared-ptr around the impl.
TORCH_MODULE(FusedTransformer);

}  // namespace olmo_cpp
